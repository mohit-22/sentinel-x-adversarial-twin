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
from typing import Dict, List, Optional

import pandas as pd
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.blue_team.detector import FEATURE_COLUMNS, evaluate_detector, run_blue_team_pipeline
from app.blue_team.explainability import compute_reason_codes, find_cached_feature_row
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

_APP_STATE: Dict = {}

# Most recent /arena/run result, this server lifetime only. Written ONLY as
# a side effect of an actual /arena/run call -- /metrics never triggers a
# run itself, and startup never populates this (stays None until the first
# real arena run happens).
_LATEST_ARENA_RUN: Optional[ArenaRunSummary] = None


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
    n_customers: int = N_CUSTOMERS
    n_merchants: int = N_MERCHANTS
    n_transactions: int = N_TRANSACTIONS
    days: int = SIMULATION_DAYS
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
