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
from app.blue_team.features import combine_clean_and_injected, engineer_features
from app.blue_team.graph_engine import apply_graph_features
from app.core.config import N_CUSTOMERS, N_MERCHANTS, N_TRANSACTIONS, SEED, SIMULATION_DAYS
from app.core.schemas import ArenaRunSummary, DetectionResult, TransactionBase
from app.red_team.arena import embed_and_engineer, run_arena_mvp_gate
from app.red_team.attack_genomes import (
    BEHAVIORAL_CAMOUFLAGE_GENOME,
    MICRO_STRUCTURING_GENOME,
    SOCIAL_ENGINEERING_COERCION_GENOME,
    SYNTHETIC_IDENTITY_DRIFT_GENOME,
    SYNTHETIC_VOICE_AUTHORIZATION_GENOME,
)
from app.red_team.attack_injector import generate_micro_structuring_attacks
from app.simulator.clean_generator import (
    generate_customer_profiles,
    generate_merchants,
    generate_transaction_base,
    simulate_payment_twin,
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


def initialize_app_state(seed: int = SEED) -> None:
    """Generate the payment twin, inject micro_structuring, run Day 4's
    full feature/train/test pipeline -- ONCE. Populates module-level
    _APP_STATE for all routes to reuse. Called from main.py's startup hook.
    """
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
    """
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
    return ArenaRunSummary(
        run_id=summary["run_id"],
        attack_family=summary["attack_family"],
        initial_evasion_rate=summary["initial_evasion_rate"],
        final_evasion_rate=summary["final_evasion_rate"],
        robustness_gain=summary["robustness_gain"],
        hard_examples_count=summary["hard_examples_count"],
        retrained_f1_score=summary["retrained_f1_score"],
    )


# --- 4. GET /explain/{transaction_id} -- Day 8 -------------------------------


@router.get("/explain/{transaction_id}")
def explain(transaction_id: str):
    """SHAP TreeExplainer reason codes -- implemented in Day 8.

    Not built today: no SHAP wiring exists anywhere in blue_team/ yet.
    Returns a clean 501, not fake reason codes.
    """
    raise HTTPException(
        status_code=501,
        detail="explain endpoint not implemented yet -- SHAP explainability is Day 8 work",
    )


# --- 5. GET /metrics ----------------------------------------------------------


class MetricsResponse(BaseModel):
    precision: float
    recall: float
    f1: float
    pr_auc: float
    fpr: float


@router.get("/metrics", response_model=MetricsResponse)
def metrics() -> MetricsResponse:
    """Current global model (M0) metrics on Day 4's held-out test set."""
    state = _get_state()
    result = evaluate_detector(state["model"], state["test_df"], FEATURE_COLUMNS)
    return MetricsResponse(
        precision=result["precision"],
        recall=result["recall"],
        f1=result["f1"],
        pr_auc=result["pr_auc"],
        fpr=result["fpr"],
    )


# --- 6. POST /sandbox/compile -- Day 8 ---------------------------------------


@router.post("/sandbox/compile")
def sandbox_compile():
    """Free text -> LLM -> Pydantic-validated genome -> live simulation --
    implemented in Day 8.

    Not built today: no LLM genome-compiler exists yet. Returns a clean
    501, not fake genome output.
    """
    raise HTTPException(
        status_code=501,
        detail="sandbox/compile endpoint not implemented yet -- LLM genome compiler is Day 8 work",
    )
