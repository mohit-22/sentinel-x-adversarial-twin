"""FastAPI routes (CLAUDE.md §7 -- exact six endpoints, no more, no less).

Route handlers call existing, already-verified service-layer functions from
simulator/, red_team/, and blue_team/ -- no business logic is reimplemented
here. Request/response wrapper models below (SimulateRequest/Response,
DetectRequest/Response, ArenaRunRequest, MetricsResponse) are NOT in
CLAUDE.md §6/§8's canonical schema list -- schemas.py isn't in this phase's
ALLOWED_TO_TOUCH, so these are API-layer DTOs local to this file, wrapping
the existing domain schemas (TransactionBase, DetectionResult,
ArenaRunSummary) rather than replacing them.

Model/data caching: generated and trained ONCE at app startup (see
initialize_app_state, called from main.py's lifespan handler), held as
module-level state -- not regenerated per request. Confirmed approach,
Day 6-final planning turn.
"""

import time
import uuid
from datetime import datetime, timezone
from typing import Dict, List, Optional

import pandas as pd
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.blue_team.detector import FEATURE_COLUMNS, evaluate_detector, run_blue_team_pipeline
from app.blue_team.explainability import compute_counterfactual, compute_reason_codes, find_cached_feature_row
from app.blue_team.features import combine_clean_and_injected, engineer_features
from app.blue_team.graph_engine import apply_graph_features
from app.core.config import N_CUSTOMERS, N_MERCHANTS, N_TRANSACTIONS, SEED, SIMULATION_DAYS
from app.core.schemas import ArenaRunSummary, CustomerProfile, DetectionResult, InjectedTransaction, TransactionBase
from app.red_team.arena import (
    embed_and_engineer,
    generate_matched_population_attacks,
    run_arena_mvp_gate,
    run_multi_family_hardening,
)
from app.red_team.attack_genomes import (
    BEHAVIORAL_CAMOUFLAGE_GENOME,
    MICRO_STRUCTURING_GENOME,
    SOCIAL_ENGINEERING_COERCION_GENOME,
    SYNTHETIC_IDENTITY_DRIFT_GENOME,
    SYNTHETIC_VOICE_AUTHORIZATION_GENOME,
)
from app.red_team.attack_injector import generate_micro_structuring_attacks, validate_injected_transactions
from app.red_team.sandbox_compiler import (
    SandboxCompilerError,
    compile_genome,
    generate_sandbox_instance,
    merge_genome,
)
from app.simulator.clean_generator import (
    generate_customer_profiles,
    generate_merchants,
    generate_transaction_base,
    simulate_payment_twin,
    validate_customers,
    validate_transactions,
)
from app.simulator.fidelity import compute_fidelity_report
from app.blue_team.zero_day import (
    train_novelty_detector, 
    compute_novelty_score, 
    find_novelty_threshold, 
    cluster_unknowns, 
    generate_cluster_report
)

router = APIRouter()

# Decision thresholds, exact values from CLAUDE.md §6. Belong in config.py
# as the single source of truth per §6's own instruction, but config.py
# isn't in this phase's ALLOWED_TO_TOUCH -- defined locally here instead,
# same precedent as Day 4/5's LGBM_PARAMS/mutation constants living in
# detector.py/arena.py when config.py wasn't in scope.
_DECISION_THRESHOLDS = ((0.35, "ALLOW"), (0.65, "STEP_UP"), (0.85, "REVIEW"), (1.01, "BLOCK"))

# Instances used to train/cache M0 at startup -- a scale sufficient for a
# real, non-trivial trained detector without a slow startup. Not the same
# as run_arena_mvp_gate's own official n_instances=2000 standard for
# /arena/run itself.
_STARTUP_ATTACK_INSTANCES = 500

_GENOME_REGISTRY: Dict[str, Dict] = {
    g["genome_id"]: g
    for g in (
        MICRO_STRUCTURING_GENOME,
        SYNTHETIC_IDENTITY_DRIFT_GENOME,
        BEHAVIORAL_CAMOUFLAGE_GENOME,
        SOCIAL_ENGINEERING_COERCION_GENOME,
        SYNTHETIC_VOICE_AUTHORIZATION_GENOME,
    )
}

from app.red_team.immune_memory import ImmuneMemoryStore, MemoryRecord
from app.red_team.adaptive_attack import run_evolutionary_search

_APP_STATE: Dict = {}
_IMMUNE_MEMORY = ImmuneMemoryStore()

# Most recent /arena/run result, this server lifetime only. Written ONLY as
# a side effect of an actual /arena/run call -- /metrics never triggers a
# run itself, and startup never populates this (stays None until the first
# real arena run happens).
_LATEST_ARENA_RUN: Optional[ArenaRunSummary] = None

# Most recent /arena/adaptive result's real per-generation lineage, this
# server lifetime only. Same pattern as _LATEST_ARENA_RUN -- written ONLY
# as a side effect of an actual /arena/adaptive call, never fabricated.
# Lets /defense/evolution serve real data once at least one adaptive run
# has happened, instead of a permanent 501.
_LATEST_ADAPTIVE_RUN: Optional[Dict] = None


def initialize_app_state(seed: int = SEED) -> None:
    """Generate the payment twin, inject micro_structuring, run Day 4's
    full feature/train/test pipeline -- ONCE. Populates module-level
    _APP_STATE for all routes to reuse. Called from main.py's startup hook.
    """
    global _LATEST_ARENA_RUN
    merchants = generate_merchants(N_MERCHANTS, seed=seed)
    customers = generate_customer_profiles(N_CUSTOMERS, merchants, seed=seed)
    clean_history = generate_transaction_base(customers, merchants, N_TRANSACTIONS, SIMULATION_DAYS, seed=seed)
    attacks = generate_micro_structuring_attacks(
        MICRO_STRUCTURING_GENOME, customers, merchants, n_instances=_STARTUP_ATTACK_INSTANCES, seed=seed
    )
    combined = combine_clean_and_injected(clean_history, attacks)
    featured = engineer_features(combined, customers)
    result = run_blue_team_pipeline(featured, seed=seed)

    radar_state = train_novelty_detector(result["train_df"], seed=seed)
    radar_threshold = find_novelty_threshold(radar_state, result["train_df"])

    _APP_STATE.clear()
    _APP_STATE.update(
        {
            "customers": customers,
            "merchants": merchants,
            "clean_history": clean_history,
            "model": result["model"],
            "train_df": result["train_df"],
            "test_df": result["test_df"],
            "graph_features": result["graph_features"],
            "radar_state": radar_state,
            "radar_threshold": radar_threshold,
        }
    )
    _LATEST_ARENA_RUN = None  # a fresh startup means no arena run has happened yet this session


def _get_state() -> Dict:
    if not _APP_STATE:
        raise HTTPException(status_code=503, detail="app state not initialized -- startup event did not run")
    return _APP_STATE


def _decision_from_score(score: float) -> str:
    for threshold, decision in _DECISION_THRESHOLDS:
        if score < threshold:
            return decision
    return "BLOCK"


# --- 1. POST /simulate ------------------------------------------------------


class SimulateRequest(BaseModel):
    # Bounds are resource limits, not scientific parameters -- generous
    # enough to remain useful for custom-scale demos (5x the official
    # N_CUSTOMERS/N_TRANSACTIONS default), but bounded so this stateless,
    # unauthenticated endpoint can't be used to trigger unbounded dataset
    # generation. Same Field(le=...) pattern already used by
    # JudgeScenario/CertificationRequest.
    n_customers: int = Field(default=N_CUSTOMERS, le=50_000)
    n_merchants: int = Field(default=N_MERCHANTS, le=2_500)
    n_transactions: int = Field(default=N_TRANSACTIONS, le=250_000)
    days: int = Field(default=SIMULATION_DAYS, le=90)
    seed: int = SEED


class SimulateResponse(BaseModel):
    customer_count: int
    merchant_count: int
    transaction_count: int
    fidelity_report: Dict[str, float]


@router.post("/simulate", response_model=SimulateResponse)
def simulate(request: SimulateRequest = SimulateRequest()) -> SimulateResponse:
    """Generate the payment twin dataset (Day 1-2's simulate_payment_twin).

    Stateless: does not mutate the cached serving state used by /detect and
    /metrics -- avoids cache-invalidation complexity between a differently-
    sized simulation run and the already-trained cached detector.
    """
    customers, transactions = simulate_payment_twin(
        n_customers=request.n_customers,
        n_merchants=request.n_merchants,
        n_transactions=request.n_transactions,
        days=request.days,
        seed=request.seed,
    )
    fidelity_report = compute_fidelity_report(customers, transactions, seed=request.seed)
    return SimulateResponse(
        customer_count=len(customers),
        merchant_count=request.n_merchants,
        transaction_count=len(transactions),
        fidelity_report=fidelity_report,
    )


# --- 2. POST /detect ---------------------------------------------------------


class DetectRequest(BaseModel):
    transactions: List[TransactionBase]


class DetectResponse(BaseModel):
    results: List[DetectionResult]


@router.post("/detect", response_model=DetectResponse)
def detect(request: DetectRequest) -> DetectResponse:
    """Score a batch of transactions with the cached, already-trained M0 +
    Day 4 feature pipeline. reason_codes is [] -- SHAP is Day 8 work;
    /detect itself is not deferred, only the SHAP content within it.
    """
    state = _get_state()
    customers, clean_history, merchants = state["customers"], state["clean_history"], state["merchants"]

    tx_df = pd.DataFrame([t.model_dump() for t in request.transactions])
    if tx_df.empty:
        return DetectResponse(results=[])

    unknown = sorted(set(tx_df["customer_id"]) - set(customers["customer_id"]))
    if unknown:
        raise HTTPException(status_code=400, detail=f"unknown customer_id(s): {unknown}")

    tx_for_embedding = tx_df.copy()
    tx_for_embedding["is_fraud"] = 0  # unknown -- that's what this endpoint determines; placeholder only
    tx_for_embedding["attack_family"] = None
    tx_for_embedding["genome_id"] = None

    featured = embed_and_engineer(tx_for_embedding, customers, clean_history, merchants)
    featured = apply_graph_features(featured, state["graph_features"])
    scored = featured.set_index("transaction_id").loc[tx_df["transaction_id"]].reset_index()

    t0 = time.perf_counter()
    risk_scores = state["model"].predict_proba(scored[FEATURE_COLUMNS])[:, 1]
    latency_ms = (time.perf_counter() - t0) * 1000

    results = [
        DetectionResult(
            transaction_id=tx_id,
            risk_score=float(score),
            decision=_decision_from_score(float(score)),
            reason_codes=[],
            latency_ms=latency_ms,
        )
        for tx_id, score in zip(scored["transaction_id"], risk_scores)
    ]
    return DetectResponse(results=results)


# --- 3. POST /arena/run -------------------------------------------------------


class ArenaRunRequest(BaseModel):
    genome_id: str
    # None -> run_arena_mvp_gate's own default (2000, the official Day
    # 5/6.5 standard) is used. Explicit override lets a caller knowingly
    # trade fidelity for speed; the documented default never changes.
    n_instances: Optional[int] = None


@router.post("/arena/run", response_model=ArenaRunSummary)
def arena_run(request: ArenaRunRequest) -> ArenaRunSummary:
    """Execute the full adversarial loop for one attack family
    (run_arena_mvp_gate). Response is strictly ArenaRunSummary's canonical
    7 fields -- mutation_breakdown/_diagnostics are still computed
    internally but not part of today's HTTP contract.

    Side effect: writes the result to _LATEST_ARENA_RUN so /metrics can
    surface it. This is the ONLY place that cache is written -- /metrics
    never triggers a run itself.
    """
    global _LATEST_ARENA_RUN

    genome = _GENOME_REGISTRY.get(request.genome_id)
    if genome is None:
        raise HTTPException(
            status_code=404,
            detail=f"unknown genome_id: {request.genome_id!r}. Known: {sorted(_GENOME_REGISTRY)}",
        )
    state = _get_state()

    kwargs = {}
    if request.n_instances is not None:
        kwargs["n_instances"] = request.n_instances

    summary = run_arena_mvp_gate(
        genome,
        state["model"],
        state["train_df"],
        state["test_df"],
        state["customers"],
        state["clean_history"],
        state["merchants"],
        state["graph_features"],
        feature_columns=FEATURE_COLUMNS,
        seed=SEED,
        **kwargs,
    )
    result = ArenaRunSummary(
        run_id=summary["run_id"],
        attack_family=summary["attack_family"],
        initial_evasion_rate=summary["initial_evasion_rate"],
        final_evasion_rate=summary["final_evasion_rate"],
        robustness_gain=summary["robustness_gain"],
        hard_examples_count=summary["hard_examples_count"],
        retrained_f1_score=summary["retrained_f1_score"],
    )
    _LATEST_ARENA_RUN = result
    return result


# --- 4. GET /explain/{transaction_id} -- Day 8 -------------------------------


class ExplainResponse(BaseModel):
    transaction_id: str
    reason_codes: List[Dict[str, str]]


@router.get("/explain/{transaction_id}", response_model=ExplainResponse)
def explain(transaction_id: str) -> ExplainResponse:
    """SHAP TreeExplainer top-3 reason codes (Day 8a, PRD §7.3 exact shape).

    Honest scope: only transactions already present in the startup
    pipeline's cached train_df/test_df can be explained -- SHAP needs the
    exact engineered feature row, which cannot be recomputed from a bare
    id (feature engineering is context-dependent). A transaction_id not
    found there (e.g. a freshly-generated /payment-twin counterfactual
    instance) gets an honest 404, not a fake explanation -- a disclosed
    scope boundary (PRD_SENTINEL_X §13.1), not an error.
    """
    state = _get_state()
    row = find_cached_feature_row(transaction_id, state["train_df"], state["test_df"])
    if row is None:
        raise HTTPException(
            status_code=404,
            detail=(
                f"no cached feature vector for transaction_id {transaction_id!r} -- "
                "SHAP explainability today covers M0's original train/test dataset only, "
                "not transactions generated fresh for other views (e.g. /payment-twin)"
            ),
        )
    reason_codes = compute_reason_codes(row, state["model"])
    return ExplainResponse(transaction_id=transaction_id, reason_codes=reason_codes)


@router.get("/explain/counterfactual/{transaction_id}")
def explain_counterfactual(transaction_id: str) -> Dict:
    """What is the smallest realistic change to this transaction that would
    have flipped the decision to ALLOW? Same cached-dataset scope as
    /explain/{transaction_id} -- a 404 for a transaction not in M0's
    cached train/test set, never a fabricated result.
    """
    state = _get_state()
    row = find_cached_feature_row(transaction_id, state["train_df"], state["test_df"])
    if row is None:
        raise HTTPException(
            status_code=404,
            detail=(
                f"no cached feature vector for transaction_id {transaction_id!r} -- "
                "counterfactual explanation today covers M0's original train/test dataset only, "
                "not transactions generated fresh for other views (e.g. /payment-twin)"
            ),
        )
    return compute_counterfactual(row, state["model"], FEATURE_COLUMNS, state["train_df"])


# --- 5. GET /metrics ----------------------------------------------------------


class MetricsResponse(BaseModel):
    precision: float
    recall: float
    f1: float
    pr_auc: float
    fpr: float
    test_set_size: int
    # None means no /arena/run has happened yet this server session -- an
    # honest "not computed yet" state, never a fabricated ARG value.
    latest_arena_run: Optional[ArenaRunSummary] = None


@router.get("/metrics", response_model=MetricsResponse)
def metrics() -> MetricsResponse:
    """Current global model (M0) metrics on Day 4's held-out test set,
    plus the most recent /arena/run result if one has happened this
    session (never triggers a run itself).
    """
    state = _get_state()
    result = evaluate_detector(state["model"], state["test_df"], FEATURE_COLUMNS)
    return MetricsResponse(
        precision=result["precision"],
        recall=result["recall"],
        f1=result["f1"],
        pr_auc=result["pr_auc"],
        fpr=result["fpr"],
        test_set_size=len(state["test_df"]),
        latest_arena_run=_LATEST_ARENA_RUN,
    )


# --- 6. POST /sandbox/compile -- Day 8b --------------------------------------


class SandboxCompileRequest(BaseModel):
    text: str


class SandboxCompileResponse(BaseModel):
    family: str
    genome_id: str
    parameters_used: Dict
    rationale: str
    results: List[DetectionResult]


@router.post("/sandbox/compile", response_model=SandboxCompileResponse)
def sandbox_compile(request: SandboxCompileRequest) -> SandboxCompileResponse:
    """Free text -> LLM selects the closest of the 5 canonical families and
    proposes bounded parameter overrides -> Pydantic + bounds validated ->
    merged onto a deep copy of that family's canonical genome
    (attack_genomes.py itself untouched) -> simulated via that family's own
    existing generator (Day 6.5 ATTACK_GENERATORS registry) -> scored via
    the exact same /detect logic (called directly, not reimplemented).

    CLAUDE.md §5's design rule: the LLM never scores a transaction or makes
    a decision -- only genome JSON. All scoring below is the same
    deterministic ML /detect already uses.
    """
    state = _get_state()
    customers, merchants = state["customers"], state["merchants"]

    try:
        proposal = compile_genome(request.text)
    except SandboxCompilerError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    genome = merge_genome(proposal)
    rows = generate_sandbox_instance(genome, customers, merchants)
    transactions = validate_injected_transactions(rows)

    detect_response = detect(DetectRequest(transactions=transactions))

    return SandboxCompileResponse(
        family=genome["family"],
        genome_id=genome["genome_id"],
        parameters_used=genome["parameters"],
        rationale=proposal.rationale,
        results=detect_response.results,
    )


# --- 7. GET /payment-twin/{customer_id} -- APPROVED EXCEPTION to §7's -------
# "exactly six endpoints, no more" rule (Day 7 Screen 3 planning turn).
#
# Investigated first, not assumed: neither /simulate's response (aggregate
# counts + fidelity only) nor /arena/run's ArenaRunSummary (aggregate
# evasion/ARG stats only) exposes any per-customer or per-transaction
# detail. Screen 3's entire purpose is a real normal-vs-attacked customer
# comparison, so there was no honest way to build it from the existing six
# endpoints. Explicitly approved as a 7th endpoint rather than faked with
# static "illustrative" data. CLAUDE.md §7 updated to document it.
#
# Reuses existing, already-verified functions exclusively -- no new
# business logic here, matching the same rule that governs the other six:
#   - generate_matched_population_attacks (arena.py, Day 6.5) for the one
#     counterfactual instance, called with a single-customer list so it's
#     genuinely n=1, not a full arena run -- no LightGBM training involved.
#   - validate_customers / validate_transactions (clean_generator.py,
#     Day 1-2) and validate_injected_transactions (attack_injector.py,
#     Day 3) for the Pydantic-boundary conversion.


class PaymentTwinResponse(BaseModel):
    customer: CustomerProfile
    normal_transactions: List[TransactionBase]
    counterfactual_transactions: List[InjectedTransaction]


@router.get("/payment-twin/{customer_id}", response_model=PaymentTwinResponse)
def payment_twin(customer_id: str, attack_family: str = "micro_structuring") -> PaymentTwinResponse:
    """One customer's real clean transaction history, plus ONE freshly-
    generated counterfactual attack instance for that same customer (per
    the requested attack_family) -- for Screen 3's side-by-side comparison.
    """
    state = _get_state()
    customers, clean_history, merchants = state["customers"], state["clean_history"], state["merchants"]

    customer_rows = customers[customers["customer_id"] == customer_id]
    if customer_rows.empty:
        raise HTTPException(status_code=404, detail=f"unknown customer_id: {customer_id!r}")

    genome = next((g for g in _GENOME_REGISTRY.values() if g["family"] == attack_family), None)
    if genome is None:
        known_families = sorted({g["family"] for g in _GENOME_REGISTRY.values()})
        raise HTTPException(
            status_code=404,
            detail=f"unknown attack_family: {attack_family!r}. Known: {known_families}",
        )

    normal_rows = clean_history[clean_history["customer_id"] == customer_id].sort_values("timestamp")
    counterfactual_raw = generate_matched_population_attacks(
        genome, customers, merchants, customer_ids=[customer_id], seed=SEED
    )

    return PaymentTwinResponse(
        customer=validate_customers(customer_rows)[0],
        normal_transactions=validate_transactions(normal_rows),
        counterfactual_transactions=validate_injected_transactions(counterfactual_raw),
    )


# --- 8. POST /arena/multi-family-run -- APPROVED EXCEPTION to §7's -----------
# "exactly six endpoints, no more" rule (Cross-Family Generalization Matrix
# planning turn, post-Day 8b). Same precedent as /payment-twin (Day 7
# Screen 3): none of the existing seven endpoints can harvest hard
# negatives from all 5 families into one combined retrain -- /arena/run
# only ever handles a single genome_id. CLAUDE.md §7 updated to document
# this exception.
#
# Reuses run_multi_family_hardening (arena.py) exclusively -- no new
# business logic here, matching the same rule that governs every other
# route.


class MultiFamilyRunRequest(BaseModel):
    # None -> run_multi_family_hardening's own default (500, a reduced-
    # scale run -- ~2 min) applies. Explicit override lets a caller
    # request the full official n=2000 (~8-10 min), knowingly trading
    # speed for the documented standard.
    n_instances: Optional[int] = None


class FamilyResult(BaseModel):
    genome_id: str
    initial_evasion_rate: float
    final_evasion_rate: float
    robustness_gain: float
    hard_examples_count: int


class MultiFamilyRunResponse(BaseModel):
    per_family: Dict[str, FamilyResult]
    total_hard_examples_count: int
    # M-multi's precision/recall/F1/FPR on Day 4's ORIGINAL held-out test
    # set -- reported alongside the evasion-rate improvement so a cost to
    # general performance (if any) isn't hidden, same discipline as
    # retrain's own documented single-family precision/recall trade-off.
    retrained_precision: float
    retrained_recall: float
    retrained_f1: float
    retrained_fpr: float


_ALL_GENOMES = [
    MICRO_STRUCTURING_GENOME,
    SYNTHETIC_IDENTITY_DRIFT_GENOME,
    BEHAVIORAL_CAMOUFLAGE_GENOME,
    SOCIAL_ENGINEERING_COERCION_GENOME,
    SYNTHETIC_VOICE_AUTHORIZATION_GENOME,
]


@router.post("/arena/multi-family-run", response_model=MultiFamilyRunResponse)
def multi_family_run(request: MultiFamilyRunRequest = MultiFamilyRunRequest()) -> MultiFamilyRunResponse:
    """Harvest hard negatives from ALL 5 attack families into one combined
    retrain (run_multi_family_hardening), then report each family's
    evasion rate against the resulting single model (M-multi) -- the
    Cross-Family Generalization Matrix's backing data.
    """
    state = _get_state()

    kwargs = {}
    if request.n_instances is not None:
        kwargs["n_instances"] = request.n_instances

    result = run_multi_family_hardening(
        _ALL_GENOMES,
        state["model"],
        state["train_df"],
        state["test_df"],
        state["customers"],
        state["clean_history"],
        state["merchants"],
        state["graph_features"],
        feature_columns=FEATURE_COLUMNS,
        seed=SEED,
        **kwargs,
    )

    metrics = result["retrained_metrics"]
    return MultiFamilyRunResponse(
        per_family={family: FamilyResult(**data) for family, data in result["per_family"].items()},
        total_hard_examples_count=result["total_hard_examples_count"],
        retrained_precision=metrics["precision"],
        retrained_recall=metrics["recall"],
        retrained_f1=metrics["f1"],
        retrained_fpr=metrics["fpr"],
    )


# --- 7. POST /zero-day/scan ----------------------------------------------------


class ZeroDayScanRequest(BaseModel):
    transactions: List[TransactionBase]


class NoveltyResult(BaseModel):
    transaction_id: str
    novelty_score: float
    is_unknown: bool


class UnknownCluster(BaseModel):
    cluster_id: int
    transaction_count: int
    novelty_score_mean: float
    novelty_score_max: float
    first_seen_timestamp: str
    last_seen_timestamp: str
    feature_means: Dict[str, float]
    representative_transaction_ids: List[str]


class ZeroDayScanResponse(BaseModel):
    results: List[NoveltyResult]
    clusters: List[UnknownCluster]
    aggregate_metrics: Dict[str, float]


@router.post("/zero-day/scan", response_model=ZeroDayScanResponse)
def scan_zero_day(request: ZeroDayScanRequest) -> ZeroDayScanResponse:
    """Score a batch of transactions with the Zero-Day Radar to identify 
    novel, unknown behavior that doesn't fit the known distributions.
    """
    state = _get_state()
    customers, clean_history, merchants = state["customers"], state["clean_history"], state["merchants"]

    tx_df = pd.DataFrame([t.model_dump() for t in request.transactions])
    if tx_df.empty:
        return ZeroDayScanResponse(results=[], clusters=[], aggregate_metrics={})

    tx_for_embedding = tx_df.copy()
    tx_for_embedding["is_fraud"] = 0
    tx_for_embedding["attack_family"] = None
    tx_for_embedding["genome_id"] = None

    featured = embed_and_engineer(tx_for_embedding, customers, clean_history, merchants)
    featured = apply_graph_features(featured, state["graph_features"])

    scored = featured.set_index("transaction_id").loc[tx_df["transaction_id"]].reset_index()

    radar_state = state["radar_state"]
    threshold = state["radar_threshold"]

    novelty_scores = compute_novelty_score(radar_state, scored)
    scored["novelty_score"] = novelty_scores
    scored["is_unknown"] = novelty_scores > threshold

    results = [
        NoveltyResult(
            transaction_id=tx_id,
            novelty_score=float(score),
            is_unknown=bool(is_unk)
        )
        for tx_id, score, is_unk in zip(scored["transaction_id"], novelty_scores, scored["is_unknown"])
    ]

    unknown_df = scored[scored["is_unknown"]].copy()
    clustered = cluster_unknowns(unknown_df)
    cluster_reports = generate_cluster_report(clustered)

    clusters = [UnknownCluster(**rep) for rep in cluster_reports]

    aggregate_metrics = {
        "total_scanned": len(scored),
        "total_unknown": len(unknown_df),
        "cluster_count": len(clusters)
    }

    return ZeroDayScanResponse(results=results, clusters=clusters, aggregate_metrics=aggregate_metrics)

class AdaptiveArenaRequest(BaseModel):
    genome_id: str
    # Bounds are resource limits (this endpoint runs population_size x
    # generations full generate+engineer+predict cycles), same pattern
    # already used by JudgeScenario/CertificationRequest. Generous relative
    # to every real value used this session (population_size<=5,
    # generations<=3, n_instances<=100).
    population_size: int = Field(default=5, ge=1, le=20)
    generations: int = Field(default=3, ge=1, le=10)
    elite_count: int = Field(default=1, ge=1, le=20)
    mutation_probability: float = Field(default=0.5, ge=0.0, le=1.0)
    n_instances: int = Field(default=50, ge=1, le=500)
    seed: int = SEED

@router.post("/arena/adaptive")
def run_adaptive_arena(req: AdaptiveArenaRequest):
    global _LATEST_ADAPTIVE_RUN

    if not _APP_STATE:
        raise HTTPException(503, "System initializing")

    base_genome = _GENOME_REGISTRY.get(req.genome_id)
    if not base_genome:
        raise HTTPException(404, "Genome not found")

    result = run_evolutionary_search(
        base_genome=base_genome,
        model=_APP_STATE["model"],
        radar_state=_APP_STATE["radar_state"],
        customers=_APP_STATE["customers"],
        clean_history=_APP_STATE["clean_history"],
        merchants=_APP_STATE["merchants"],
        graph_features=_APP_STATE["graph_features"],
        population_size=req.population_size,
        generations=req.generations,
        elite_count=req.elite_count,
        mutation_probability=req.mutation_probability,
        n_instances=req.n_instances,
        seed=req.seed,
    )

    # Real, full lineage for /defense/evolution -- every genome evaluated in
    # every generation, not just per-generation peaks, so this is genuinely
    # "its lineage" rather than a summary. evasion/fitness kept alongside
    # evasion_rate as aliases -- Command Center's existing trajectory
    # rendering reads t.evasion/t.fitness; dropping them would silently
    # break an already-working display.
    lineage = result["lineage"]
    trajectory = [
        {
            "generation": entry["generation"],
            "genome_id": entry["genome"]["genome_id"],
            "evasion_rate": entry["evasion_rate"],
            "evasion": entry["evasion_rate"],
            "fitness": entry["total_fitness"],
        }
        for entry in lineage
    ]
    
    # Run identity must be a complete, deterministic representation of the
    # actual execution configuration -- not just genome_id/generations/
    # population_size (the old format silently collided across requests
    # that differed only in elite_count/mutation_probability/n_instances,
    # and always used the global SEED since no per-request seed existed).
    run_id = (
        f"arena-adaptive-{req.genome_id}-{req.seed}-{req.generations}-{req.population_size}"
        f"-{req.elite_count}-{req.mutation_probability}-{req.n_instances}"
    )
    created_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    
    _LATEST_ADAPTIVE_RUN = {
        "status": "ok", 
        "run_id": run_id,
        "base_genome_id": req.genome_id,
        "created_at": created_at,
        "trajectory": trajectory,
        "lineage": lineage
    }

    # Store the best one in Immune Memory
    best = result["best_attack"]
    
    if best["validity_status"] == "VALID" and best["evasion_rate"] > 0.05:
        # Convert to MemoryRecord
        mem_rec = MemoryRecord(
            memory_id=f"MEM-{best['genome']['genome_id']}",
            attack_family=best['genome']['family'],
            genome_id=best['genome']['genome_id'],
            genome=best['genome'],
            parent_attack_id=best['parent_attack_id'],
            generation=best['generation'],
            initial_evasion=best['evasion_rate'],
            best_evasion=best['evasion_rate'],
            defense_version="M0",
            current_status="DISCOVERED",
            residual_evasion=best['evasion_rate'],
            novelty_score=best['novelty_score'],
            realism_score=best['realism_score'],
            provenance="training"
        )
        _IMMUNE_MEMORY.add_record(mem_rec)
        
    return {
        "status": "success",
        "best_attack": best,
        "lineage": result["lineage"],
        "memory_additions": 1 if (best["validity_status"] == "VALID" and best["evasion_rate"] > 0.05) else 0
    }

@router.get("/immune-memory")
def get_immune_memory():
    return {"records": [r.model_dump() for r in _IMMUNE_MEMORY.get_all()]}

from app.blue_team.defense_compiler import AttackFailureAnalysis, DefensePolicy, analyze_attack, compile_policy
from app.blue_team.policy_simulator import simulate_policy_utility

_ACTIVE_POLICIES = []
_CANDIDATE_POLICIES: Dict[str, DefensePolicy] = {}

class AnalyzeAttackRequest(BaseModel):
    base_genome_id: str
    evolved_genome_id: str

@router.post("/defense/analyze-attack")
def api_analyze_attack(req: AnalyzeAttackRequest):
    """analyze_attack() needs real base/evolved attack transaction
    DataFrames and both genome dicts. base_genome_id resolves via
    _GENOME_REGISTRY, but evolved_genome_id (e.g. a MUT-* id from
    /arena/adaptive) has no stored lookup back to its genome or the
    transactions it generated. Honest 501, not fabricated analysis.
    """
    if not _APP_STATE:
        raise HTTPException(503, "System initializing")

    base = _GENOME_REGISTRY.get(req.base_genome_id)
    if not base:
        raise HTTPException(404, "Base genome not found")

    raise HTTPException(
        status_code=501,
        detail=(
            "defense/analyze-attack not implemented yet -- requires a persistence layer "
            "for adaptive run results (evolved genome + its generated transactions, "
            "keyed by id), not yet built"
        ),
    )

@router.post("/defense/compile")
def api_compile_defense(analysis: AttackFailureAnalysis):
    policies = compile_policy(analysis)
    for p in policies:
        _CANDIDATE_POLICIES[p.policy_id] = p
    return {"policies": [p.model_dump() for p in policies]}

class SimulatePolicyRequest(BaseModel):
    policy: DefensePolicy

@router.post("/defense/simulate")
def api_simulate_defense(req: SimulatePolicyRequest):
    """Real simulate_policy_utility() call: clean side is _APP_STATE's real
    clean_history (engineered + graph-featured, matching the same two-step
    pattern used everywhere else in this file); attack side is a small,
    freshly-generated micro_structuring batch (n=50), reusing the exact
    generate_micro_structuring_attacks + embed_and_engineer +
    apply_graph_features pipeline /detect and arena.py already use --
    no new business logic, no fabricated utility number.
    """
    state = _get_state()
    customers, clean_history, merchants = state["customers"], state["clean_history"], state["merchants"]
    graph_features, model = state["graph_features"], state["model"]

    clean_featured = engineer_features(clean_history.copy(), customers)
    clean_featured = apply_graph_features(clean_featured, graph_features)
    clean_featured = clean_featured.dropna(subset=FEATURE_COLUMNS)
    m0_predictions_clean = model.predict(clean_featured[FEATURE_COLUMNS])

    attacks_raw = generate_micro_structuring_attacks(
        MICRO_STRUCTURING_GENOME, customers, merchants, n_instances=50, seed=SEED
    )
    attack_featured = embed_and_engineer(attacks_raw, customers, clean_history, merchants)
    attack_featured = apply_graph_features(attack_featured, graph_features)
    attack_featured = attack_featured[attack_featured["is_fraud"] == 1].copy()
    m0_predictions_attack = model.predict(attack_featured[FEATURE_COLUMNS])

    return simulate_policy_utility(
        clean_history_featured=clean_featured,
        attack_featured=attack_featured,
        policy=req.policy,
        m0_predictions_clean=m0_predictions_clean,
        m0_predictions_attack=m0_predictions_attack,
    )

@router.get("/defense/policies")
def api_get_policies():
    return {"policies": [p.model_dump() for p in _ACTIVE_POLICIES]}

class ApprovePolicyRequest(BaseModel):
    policy_id: str
    action: str

@router.post("/defense/approve")
def api_approve_policy(req: ApprovePolicyRequest):
    global _ACTIVE_POLICIES, _CANDIDATE_POLICIES
    
    if req.policy_id not in _CANDIDATE_POLICIES:
        raise HTTPException(status_code=404, detail="Candidate policy not found")
        
    policy = _CANDIDATE_POLICIES[req.policy_id]
    
    if policy.status != "CANDIDATE":
        raise HTTPException(status_code=422, detail=f"Policy is in invalid state for approval: {policy.status}")
        
    if req.action == "APPROVE":
        policy.status = "ACTIVE"
        # Prevent duplicates
        if not any(p.policy_id == policy.policy_id for p in _ACTIVE_POLICIES):
            _ACTIVE_POLICIES.append(policy)
    elif req.action == "REJECT":
        policy.status = "REJECTED"
    else:
        raise HTTPException(status_code=422, detail="Invalid action")
        
    return {
        "status": "success",
        "policy_id": policy.policy_id,
        "action": req.action,
        "new_status": policy.status
    }
@router.get("/defense/radar")
def api_get_radar():
    """Real novelty summary over Day 4's held-out test_df (already engineered,
    already available in _APP_STATE -- no fresh generation or "which attack"
    choice needed). Reuses the exact same compute_novelty_score/threshold/
    cluster_unknowns/generate_cluster_report pipeline /zero-day/scan uses.
    """
    state = _get_state()
    test_df = state["test_df"]
    radar_state = state["radar_state"]
    threshold = state["radar_threshold"]

    scored = test_df.copy()
    novelty_scores = compute_novelty_score(radar_state, scored)
    scored["novelty_score"] = novelty_scores
    scored["is_unknown"] = novelty_scores > threshold

    unknown_df = scored[scored["is_unknown"]].copy()
    clustered = cluster_unknowns(unknown_df)
    cluster_reports = generate_cluster_report(clustered)

    return {
        "unknown_events": int(scored["is_unknown"].sum()),
        "unknown_clusters": len(cluster_reports),
        "novelty_score": float(novelty_scores.mean()) if len(novelty_scores) > 0 else 0.0,
        "first_seen": str(unknown_df["timestamp"].min()) if not unknown_df.empty else "N/A",
        "last_seen": str(unknown_df["timestamp"].max()) if not unknown_df.empty else "N/A",
        "status": "MONITORING"
    }

@router.get("/defense/evolution")
def api_get_evolution():
    """Real lineage from the most recent /arena/adaptive call this server
    session (_LATEST_ADAPTIVE_RUN), same caching pattern as /metrics
    serving _LATEST_ARENA_RUN. Honest empty state (200, not 501) when no
    adaptive run has happened yet -- this is a real, well-defined "nothing
    to report yet" response, not a missing feature.
    """
    if _LATEST_ADAPTIVE_RUN is not None:
        return _LATEST_ADAPTIVE_RUN

    return {"status": "no_adaptive_run_this_session", "trajectory": []}

# --- STEP 6b: THREAT OBSERVATORY (reuses existing caches, no new state) ---

@router.get("/observatory/lineage")
def api_observatory_lineage():
    """Fraud DNA lineage for the dashboard's evolution-tree view. Pure
    passthrough of _LATEST_ADAPTIVE_RUN -- same cache /defense/evolution
    reads, no new computation or state.
    """
    if _LATEST_ADAPTIVE_RUN is not None:
        return _LATEST_ADAPTIVE_RUN

    return {"status": "no_run", "trajectory": []}

@router.get("/observatory/impact")
def api_observatory_impact():
    """Economic impact of the most recent /arena/run this session.
    Honest zero-state when no arena run has happened yet.
    """
    if not _APP_STATE:
        raise HTTPException(503, "System initializing")

    from app.red_team.arena import _LATEST_ARENA_IMPACT

    if _LATEST_ARENA_RUN is None or _LATEST_ARENA_RUN.run_id not in _LATEST_ARENA_IMPACT:
        return {
            "status": "run_arena_first",
            "run_id": "",
            "attack_family": "",
            "total_attack_transactions": 0,
            "total_attack_value_inr": 0.0,
            "value_caught_by_m0_inr": 0.0,
            "value_caught_after_hardening_inr": 0.0,
            "incremental_value_prevented_inr": 0.0,
            "m0_evasion_rate": 0.0,
            "post_hardening_evasion_rate": 0.0,
            "additional_transactions_caught": 0,
            "methodology": "This is a synthetic benchmark measurement computed from the actual generated attack transaction amounts and measured M0 vs post-hardening detector outcomes in the Sentinel-X Payment Twin. It is not a production financial-loss estimate."
        }

    impact = _LATEST_ARENA_IMPACT[_LATEST_ARENA_RUN.run_id]
    
    return {
        "status": "ok",
        "run_id": impact["run_id"],
        "attack_family": impact["attack_family"],
        "total_attack_transactions": impact["total_attack_transactions"],
        "total_attack_value_inr": impact["total_attack_value_inr"],
        "value_caught_by_m0_inr": impact["value_caught_by_m0_inr"],
        "value_caught_after_hardening_inr": impact["value_caught_after_hardening_inr"],
        "incremental_value_prevented_inr": impact["incremental_value_prevented_inr"],
        "m0_evasion_rate": impact["m0_evasion_rate"],
        "post_hardening_evasion_rate": impact["post_hardening_evasion_rate"],
        "additional_transactions_caught": impact["additional_transactions_caught"],
        "methodology": "This is a synthetic benchmark measurement computed from the actual generated attack transaction amounts and measured M0 vs post-hardening detector outcomes in the Sentinel-X Payment Twin. It is not a production financial-loss estimate."
    }

import hashlib

class ObservatoryExportRequest(BaseModel):
    run_id: str
    genome_id: str

@router.post("/observatory/export")
def api_observatory_export(req: ObservatoryExportRequest):
    """STIX 2.1-shaped threat-intel export for one known genome."""
    if _LATEST_ADAPTIVE_RUN is None:
        raise HTTPException(status_code=404, detail="No adaptive run has occurred.")

    if req.run_id != _LATEST_ADAPTIVE_RUN.get("run_id"):
        raise HTTPException(status_code=404, detail=f"Invalid run_id: {req.run_id}")

    lineage = _LATEST_ADAPTIVE_RUN.get("lineage", [])
    entry = next((e for e in lineage if e["genome"]["genome_id"] == req.genome_id), None)
    
    if not entry:
        raise HTTPException(status_code=404, detail=f"Genome {req.genome_id} does not belong to run {req.run_id}")

    genome = entry["genome"]
    created_at = _LATEST_ADAPTIVE_RUN["created_at"]

    def stable_id(obj_type: str) -> str:
        s = f"{req.run_id}_{req.genome_id}_{obj_type}"
        h = hashlib.sha256(s.encode()).hexdigest()
        return f"{obj_type}--{h}"

    return {
        "type": "bundle",
        "id": stable_id("bundle"),
        "objects": [{
            "type": "attack-pattern",
            "spec_version": "2.1",
            "id": stable_id("attack-pattern"),
            "created": created_at,
            "modified": created_at,
            "name": genome["family"],
            "description": genome.get("objective", ""),
            "x_sentinel_run_id": req.run_id,
            "x_sentinel_genome_id": genome["genome_id"],
            "x_sentinel_base_genome_id": _LATEST_ADAPTIVE_RUN.get("base_genome_id", ""),
            "x_sentinel_attack_family": genome["family"],
            "x_sentinel_generation": entry["generation"],
            "x_sentinel_parent_attack_id": entry["parent_attack_id"],
            "x_sentinel_evasion_rate": entry["evasion_rate"],
            "x_sentinel_fitness": entry["total_fitness"],
            "x_sentinel_novelty_score": entry["novelty_score"],
            "x_sentinel_impact_score": entry["impact_score"],
            "x_sentinel_realism_score": entry["realism_score"],
            "x_sentinel_validity_status": entry["validity_status"],
            "x_sentinel_parameters": genome.get("parameters", {}),
            "x_sentinel_mutations": genome.get("mutations", []),
            "kill_chain_phases": [{
                "kill_chain_name": "sentinel-x-fraud-lifecycle",
                "phase_name": genome["family"],
            }],
        }],
    }

from app.blue_team.soc_agent import run_soc_agent, AgentVerdict

@router.post("/soc/investigate/{transaction_id}", response_model=AgentVerdict)
def soc_investigate(transaction_id: str) -> AgentVerdict:
    """Autonomous SOC Agent: investigates a flagged transaction using SHAP
    analysis + immune memory + LLM reasoning. Returns structured verdict
    with evidence and audit log."""
    state = _get_state()

    try:
        verdict = run_soc_agent(
            transaction_id=transaction_id,
            model=state["model"],
            train_df=state["train_df"],
            test_df=state["test_df"],
            immune_memory=_IMMUNE_MEMORY,
        )
        return verdict
    except ValueError as e:
        raise HTTPException(
            status_code=404,
            detail=str(e) + " -- SOC Agent only covers M0's cached dataset.",
        )

# --- STEP 7: JUDGE MODE API ---
from app.judge.schemas import JudgeScenario
from app.judge.scenario_runner import ScenarioOrchestrator

@router.post("/judge/scenario")
def api_create_judge_scenario(req: JudgeScenario):
    state = ScenarioOrchestrator.create_scenario(req)
    return state.dict()

import threading

@router.post("/judge/scenario/{scenario_id}/run")
def api_run_judge_scenario(scenario_id: str):
    # Run async so the frontend can poll
    t = threading.Thread(target=ScenarioOrchestrator.run_scenario, args=(scenario_id,))
    t.start()
    return {"status": "started", "scenario_id": scenario_id}

@router.get("/judge/scenario/{scenario_id}")
def api_get_judge_scenario(scenario_id: str):
    state = ScenarioOrchestrator.get_state(scenario_id)
    if not state:
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail="Scenario not found")
    return state.dict()

@router.post("/judge/scenario/{scenario_id}/reset")
def api_reset_judge_scenario(scenario_id: str):
    ScenarioOrchestrator.reset(scenario_id)
    return {"status": "reset"}

@router.post("/judge/scenario/{scenario_id}/approve")
def api_approve_judge_scenario(scenario_id: str):
    ScenarioOrchestrator.approve_and_continue(scenario_id)
    return {"status": "approved"}

# --- NEW: Recursive Defense Certification Engine ---

from app.defense.schemas import CertificationRequest, CertificationResult
from app.defense.recursive_engine import run_certification

@router.post("/defense/certify", response_model=CertificationResult)
def api_defense_certify(request: CertificationRequest) -> CertificationResult:
    """
    Executes the recursive defense certification loop:
    D0 -> Attack M0 -> Discover W1 -> Defense D1 -> validate D1 -> ATTACK D1 -> etc.
    """
    if not _APP_STATE:
        raise HTTPException(status_code=503, detail="app state not initialized")

    return run_certification(request)


@router.get("/threat-map")
def api_threat_map() -> Dict:
    """Aggregated fraud-detection data by city, for the Live Threat Map.

    test_df already carries a real `location` field per transaction --
    engineer_features() only ever adds columns, it never drops the
    original TransactionBase fields, so no join with clean_history is
    needed (and joining would be wrong: the injected fraud rows were never
    in clean_history, so a join would silently lose location for exactly
    the rows this endpoint cares about most). No risk_score is cached on
    test_df, so every row is scored live here.

    "Fraud"/"blocked" means the detector's own predicted flag
    (model.predict()==1) -- the same "caught" convention already used
    everywhere else in this codebase (arena.py/detector.py), not the
    ground-truth is_fraud label, which wouldn't be known operationally in
    a real-time monitoring view.
    """
    state = _get_state()
    test_df = state["test_df"]
    model = state["model"]

    y_pred = model.predict(test_df[FEATURE_COLUMNS])
    scored = test_df[["location", "amount"]].copy()
    scored["_flagged"] = y_pred

    cities = []
    for city, group in scored.groupby("location"):
        total = len(group)
        flagged = int(group["_flagged"].sum())
        fraud_rate = float(flagged / total) if total > 0 else 0.0
        amount_blocked = float(group.loc[group["_flagged"] == 1, "amount"].sum())

        if fraud_rate > 0.05:
            risk_level = "HIGH"
        elif fraud_rate > 0.02:
            risk_level = "MEDIUM"
        else:
            risk_level = "LOW"

        cities.append(
            {
                "city": city,
                "total_transactions": int(total),
                "fraud_transactions": flagged,
                "fraud_rate": round(fraud_rate, 4),
                "total_amount_blocked_inr": round(amount_blocked, 2),
                "risk_level": risk_level,
            }
        )

    cities.sort(key=lambda c: c["fraud_rate"], reverse=True)
    total_blocked = sum(c["total_amount_blocked_inr"] for c in cities)

    return {
        "cities": cities,
        "summary": {
            "total_fraud_blocked_inr": round(total_blocked, 2),
            "highest_risk_city": cities[0]["city"] if cities else None,
            "cities_monitored": len(cities),
        },
    }
