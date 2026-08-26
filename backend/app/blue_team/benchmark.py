"""Leakage-Free Robustness Benchmark (Step 2 Evaluation Gate).

Provides a comprehensive, reproducible benchmark to evaluate Sentinel-X's
performance on clean holdout data, fresh known attacks, mutated attacks, 
and cross-family zero-day attacks, all while enforcing strict customer
disjointness.
"""

import time
from typing import Dict, List, TypedDict, Any

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, confusion_matrix, f1_score, precision_score, recall_score

from app.blue_team.detector import FEATURE_COLUMNS, evaluate_detector
from app.red_team.arena import compute_arg, run_attack, harvest_hard_negatives, retrain, re_test
from app.red_team.attack_genomes import (
    MICRO_STRUCTURING_GENOME,
    SYNTHETIC_IDENTITY_DRIFT_GENOME,
    BEHAVIORAL_CAMOUFLAGE_GENOME,
    SOCIAL_ENGINEERING_COERCION_GENOME,
    SYNTHETIC_VOICE_AUTHORIZATION_GENOME,
)

ALL_GENOMES = [
    MICRO_STRUCTURING_GENOME,
    SYNTHETIC_IDENTITY_DRIFT_GENOME,
    BEHAVIORAL_CAMOUFLAGE_GENOME,
    SOCIAL_ENGINEERING_COERCION_GENOME,
    SYNTHETIC_VOICE_AUTHORIZATION_GENOME,
]

class DecisionBand(TypedDict):
    decision: str
    count: int
    fraud_caught: int
    legitimate_impacted: int

class CalibrationBucket(TypedDict):
    range: str
    transaction_count: int
    fraud_count: int
    observed_fraud_rate: float

class PerFamilyResult(TypedDict):
    family: str
    attack_transaction_count: int
    evasion_rate: float
    recall: float
    f1: float
    fpr: float

class MutationResult(TypedDict):
    family: str
    mutation_name: str
    original_evasion_rate: float
    mutated_evasion_rate: float
    degradation: float

class CrossFamilyResult(TypedDict):
    evaluation_family: str
    trained_families: List[str]
    evasion_rate: float
    recall: float
    f1: float
    fpr: float

class RobustnessScorecard(TypedDict):
    benchmark_version: str
    dataset_seed: int
    model_version: str
    train_customer_count: int
    clean_holdout_customer_count: int
    adversarial_holdout_customer_count: int

    clean_precision: float
    clean_recall: float
    clean_f1: float
    clean_pr_auc: float
    clean_fpr: float

    known_attack_evasion: Dict[str, float]
    mutated_attack_evasion: Dict[str, float]
    cross_family_evasion: Dict[str, float]

    per_family_results: List[PerFamilyResult]
    mutation_results: List[MutationResult]
    cross_family_results: List[CrossFamilyResult]

    calibration_summary: List[CalibrationBucket]
    decision_bands: List[DecisionBand]
    latency_summary: Dict[str, Any]


def _evaluate_threshold_decisions(y_true: np.ndarray, y_proba: np.ndarray) -> List[DecisionBand]:
    """Analyze predictions against the current API thresholds."""
    # ((0.35, "ALLOW"), (0.65, "STEP_UP"), (0.85, "REVIEW"), (1.01, "BLOCK"))
    bands = []
    thresholds = [(0.0, 0.35, "ALLOW"), (0.35, 0.65, "STEP_UP"), (0.65, 0.85, "REVIEW"), (0.85, 1.01, "BLOCK")]
    
    for lower, upper, decision in thresholds:
        mask = (y_proba >= lower) & (y_proba < upper)
        count = int(mask.sum())
        fraud_caught = int(y_true[mask].sum())
        legitimate_impacted = int(count - fraud_caught)
        bands.append(DecisionBand(
            decision=decision,
            count=count,
            fraud_caught=fraud_caught,
            legitimate_impacted=legitimate_impacted
        ))
    return bands


def _evaluate_calibration(y_true: np.ndarray, y_proba: np.ndarray, bins: int = 10) -> List[CalibrationBucket]:
    """Score distribution and observed fraud rate per bucket."""
    buckets = []
    edges = np.linspace(0.0, 1.0, bins + 1)
    
    for i in range(bins):
        lower, upper = edges[i], edges[i+1]
        # Include upper bound for the last bin
        mask = (y_proba >= lower) & ((y_proba <= upper) if i == bins - 1 else (y_proba < upper))
        
        count = int(mask.sum())
        fraud_count = int(y_true[mask].sum())
        rate = (fraud_count / count) if count > 0 else 0.0
        
        buckets.append(CalibrationBucket(
            range=f"{lower:.2f}-{upper:.2f}",
            transaction_count=count,
            fraud_count=fraud_count,
            observed_fraud_rate=rate
        ))
    return buckets


def _compute_attack_metrics(
    model, 
    attacks_raw: pd.DataFrame, 
    test_df: pd.DataFrame, 
    customers: pd.DataFrame, 
    clean_history: pd.DataFrame, 
    merchants: pd.DataFrame, 
    graph_features: Dict, 
    feature_columns: List[str]
) -> Dict:
    """Combine fresh attacks with clean holdout transactions to compute full F1/FPR metrics."""
    from app.red_team.arena import embed_and_engineer
    from app.blue_team.graph_engine import apply_graph_features
    
    featured_attacks = embed_and_engineer(attacks_raw, customers, clean_history, merchants)
    featured_attacks = apply_graph_features(featured_attacks, graph_features)
    
    clean_test_df = test_df[test_df["is_fraud"] == 0].copy()
    combined = pd.concat([clean_test_df, featured_attacks], ignore_index=True)
    return evaluate_detector(model, combined, feature_columns)


def run_robustness_benchmark(
    model,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    customers: pd.DataFrame,
    clean_history: pd.DataFrame,
    merchants: pd.DataFrame,
    graph_features: Dict,
    feature_columns: List[str] = FEATURE_COLUMNS,
    n_instances: int = 2000,
    seed: int = 42,
) -> RobustnessScorecard:
    """Execute the full leakage-free evaluation benchmark.
    
    Partitions:
    - TRAIN / HARDENING: Uses `train_df` customers.
    - CLEAN / ADVERSARIAL HOLDOUT: Uses `test_df` customers.
    """
    train_customer_ids = train_df["customer_id"].unique()
    test_customer_ids = test_df["customer_id"].unique()
    train_customers = customers[customers["customer_id"].isin(train_customer_ids)]
    test_customers = customers[customers["customer_id"].isin(test_customer_ids)]

    # --- Clean Holdout Evaluation ---
    clean_metrics = evaluate_detector(model, test_df, feature_columns)
    
    y_test_clean = test_df["is_fraud"].to_numpy()
    y_proba_clean = model.predict_proba(test_df[feature_columns])[:, 1]
    
    decision_bands = _evaluate_threshold_decisions(y_test_clean, y_proba_clean)
    calibration = _evaluate_calibration(y_test_clean, y_proba_clean)
    
    # --- Latency Validation ---
    # We strictly measure model batch inference latency here since the API wrapper
    # currently only measures the predict_proba step (not feature engineering).
    t0 = time.perf_counter()
    _ = model.predict_proba(test_df[feature_columns].iloc[:1000])
    latency_ms = (time.perf_counter() - t0) * 1000
    latency_summary = {
        "measurement": "model_batch_inference_latency_ms",
        "batch_size": min(1000, len(test_df)),
        "value": latency_ms,
        "description": "Measures LightGBM predict_proba batch execution time only. Excludes network, DB, and feature engineering overhead."
    }

    per_family_results: List[PerFamilyResult] = []
    mutation_results: List[MutationResult] = []
    
    # --- Known Attack Generalization & Mutation Generalization ---
    known_attack_evasion = {}
    mutated_attack_evasion = {}

    for genome in ALL_GENOMES:
        family = genome["family"]
        
        # 1. Evaluate M0 on fresh attacks (Adversarial Holdout)
        attack = run_attack(
            genome, model, test_customers, clean_history, merchants, graph_features,
            feature_columns, min(n_instances, len(test_customers)), seed
        )
        
        attack_metrics = _compute_attack_metrics(
            model, attack["attacks_raw"], test_df, customers, clean_history, merchants, graph_features, feature_columns
        )
        
        known_attack_evasion[family] = attack["evasion_rate"]
        per_family_results.append(PerFamilyResult(
            family=family,
            attack_transaction_count=attack["total_fraud"],
            evasion_rate=attack["evasion_rate"],
            recall=attack_metrics["recall"],
            f1=attack_metrics["f1"],
            fpr=attack_metrics["fpr"]
        ))
        
        # 2. Evaluate M0 on mutated attacks (Adversarial Holdout)
        for mutation_index, mutation_name in enumerate(genome["mutations"]):
            m_attack = re_test(
                genome=genome,
                model=model,
                customers=customers,
                clean_history=clean_history,
                merchants=merchants,
                graph_features=graph_features,
                customer_ids=attack["customer_ids_used"],
                exclude_transaction_ids=set(),
                mutation_name=mutation_name,
                feature_columns=feature_columns,
                seed=seed + 3000 + mutation_index
            )
            mutated_attack_evasion[f"{family}::{mutation_name}"] = m_attack["evasion_rate"]
            mutation_results.append(MutationResult(
                family=family,
                mutation_name=mutation_name,
                original_evasion_rate=attack["evasion_rate"],
                mutated_evasion_rate=m_attack["evasion_rate"],
                degradation=compute_arg(attack["evasion_rate"], m_attack["evasion_rate"])
            ))

    # --- Cross-Family Generalization ---
    cross_family_results: List[CrossFamilyResult] = []
    cross_family_evasion = {}
    
    for eval_genome in ALL_GENOMES:
        eval_family = eval_genome["family"]
        train_genomes = [g for g in ALL_GENOMES if g["family"] != eval_family]
        trained_families = [g["family"] for g in train_genomes]
        
        all_hard_negatives = []
        for tg in train_genomes:
            # Harvest from TRAIN customers only
            t_attack = run_attack(
                tg, model, train_customers, clean_history, merchants, graph_features,
                feature_columns, min(n_instances, len(train_customers)), seed + 10
            )
            harvest = harvest_hard_negatives(
                t_attack["evaded_rows"], t_attack["attacks_raw"], customers, clean_history,
                merchants, graph_features, tg, seed
            )
            all_hard_negatives.append(harvest["hard_negatives"])
            
        combined_hn = pd.concat(all_hard_negatives, ignore_index=True)
        m_cross = retrain(train_df, combined_hn, feature_columns)
        
        # Evaluate on the left-out zero-day family using ADVERSARIAL HOLDOUT customers
        zero_day_eval = run_attack(
            eval_genome, m_cross, test_customers, clean_history, merchants, graph_features,
            feature_columns, min(n_instances, len(test_customers)), seed + 1000
        )
        
        zero_day_metrics = _compute_attack_metrics(
            m_cross, zero_day_eval["attacks_raw"], test_df, customers, clean_history, merchants, graph_features, feature_columns
        )
        
        cross_family_evasion[eval_family] = zero_day_eval["evasion_rate"]
        cross_family_results.append(CrossFamilyResult(
            evaluation_family=eval_family,
            trained_families=trained_families,
            evasion_rate=zero_day_eval["evasion_rate"],
            recall=zero_day_metrics["recall"],
            f1=zero_day_metrics["f1"],
            fpr=zero_day_metrics["fpr"]
        ))

    return RobustnessScorecard(
        benchmark_version="1.0",
        dataset_seed=seed,
        model_version="M0 (Baseline)",
        train_customer_count=len(train_customer_ids),
        clean_holdout_customer_count=len(test_customer_ids),
        adversarial_holdout_customer_count=len(test_customer_ids),
        
        clean_precision=clean_metrics["precision"],
        clean_recall=clean_metrics["recall"],
        clean_f1=clean_metrics["f1"],
        clean_pr_auc=clean_metrics["pr_auc"],
        clean_fpr=clean_metrics["fpr"],
        
        known_attack_evasion=known_attack_evasion,
        mutated_attack_evasion=mutated_attack_evasion,
        cross_family_evasion=cross_family_evasion,
        
        per_family_results=per_family_results,
        mutation_results=mutation_results,
        cross_family_results=cross_family_results,
        
        calibration_summary=calibration,
        decision_bands=decision_bands,
        latency_summary=latency_summary
    )
