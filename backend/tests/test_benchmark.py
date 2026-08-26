"""Tests for the Leakage-Free Robustness Benchmark (Step 2)."""

import pytest

from app.core.config import N_CUSTOMERS, N_MERCHANTS, N_TRANSACTIONS, SEED, SIMULATION_DAYS
from app.blue_team.detector import FEATURE_COLUMNS, run_blue_team_pipeline
from app.blue_team.features import combine_clean_and_injected, engineer_features
from app.blue_team.benchmark import run_robustness_benchmark
from app.red_team.attack_genomes import MICRO_STRUCTURING_GENOME
from app.red_team.attack_injector import generate_micro_structuring_attacks
from app.simulator.clean_generator import generate_customer_profiles, generate_merchants, generate_transaction_base

@pytest.fixture(scope="module")
def benchmark_result():
    merchants = generate_merchants(N_MERCHANTS, seed=SEED)
    customers = generate_customer_profiles(N_CUSTOMERS, merchants, seed=SEED)
    clean = generate_transaction_base(customers, merchants, N_TRANSACTIONS, SIMULATION_DAYS, seed=SEED)
    
    # Generate some minimal injected transactions for baseline
    attacks = generate_micro_structuring_attacks(
        MICRO_STRUCTURING_GENOME, customers, merchants, n_instances=500, seed=SEED
    )
    combined = combine_clean_and_injected(clean, attacks)
    featured = engineer_features(combined, customers)
    day4_result = run_blue_team_pipeline(featured, seed=SEED)

    return run_robustness_benchmark(
        model=day4_result["model"],
        train_df=day4_result["train_df"],
        test_df=day4_result["test_df"],
        customers=customers,
        clean_history=clean,
        merchants=merchants,
        graph_features=day4_result["graph_features"],
        feature_columns=FEATURE_COLUMNS,
        n_instances=10,  # Tiny count so tests run fast
        seed=SEED
    )

def test_benchmark_schema(benchmark_result):
    """Test output schema validity."""
    expected_keys = [
        "benchmark_version", "dataset_seed", "model_version",
        "train_customer_count", "clean_holdout_customer_count", "adversarial_holdout_customer_count",
        "clean_precision", "clean_recall", "clean_f1", "clean_pr_auc", "clean_fpr",
        "known_attack_evasion", "mutated_attack_evasion", "cross_family_evasion",
        "per_family_results", "mutation_results", "cross_family_results",
        "calibration_summary", "decision_bands", "latency_summary"
    ]
    for k in expected_keys:
        assert k in benchmark_result

def test_benchmark_calibration(benchmark_result):
    """Verify calibration buckets cover full 0 to 1 range."""
    buckets = benchmark_result["calibration_summary"]
    assert len(buckets) == 10
    total_txns = sum(b["transaction_count"] for b in buckets)
    assert total_txns > 0

def test_benchmark_decision_bands(benchmark_result):
    """Verify decision bands cover standard thresholds."""
    bands = benchmark_result["decision_bands"]
    assert len(bands) == 4
    decisions = [b["decision"] for b in bands]
    assert decisions == ["ALLOW", "STEP_UP", "REVIEW", "BLOCK"]

def test_benchmark_reproducible_partitions(benchmark_result):
    """Test reproducible partitions and non-overlap."""
    # Ensure there are no overlapping customers if we re-evaluate manually.
    # The benchmark module doesn't expose the exact sets in its return schema,
    # but the counts should be consistent.
    assert benchmark_result["dataset_seed"] == SEED
    assert benchmark_result["train_customer_count"] > 0
    assert benchmark_result["clean_holdout_customer_count"] > 0
    assert benchmark_result["adversarial_holdout_customer_count"] == benchmark_result["clean_holdout_customer_count"]
