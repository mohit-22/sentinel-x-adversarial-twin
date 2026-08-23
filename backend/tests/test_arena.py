"""Tests for the Adversarial Arena (CLAUDE.md §0.10, §4.3, Day 5 MVP gate)."""

import numpy as np
import pandas as pd
import pytest

from app.core.config import N_CUSTOMERS, N_MERCHANTS, N_TRANSACTIONS, SEED, SIMULATION_DAYS
from app.blue_team.detector import FEATURE_COLUMNS, run_blue_team_pipeline
from app.blue_team.features import combine_clean_and_injected, engineer_features
from app.red_team.arena import compute_arg, run_arena_mvp_gate, validate_mutation
from app.red_team.attack_genomes import MICRO_STRUCTURING_GENOME
from app.red_team.attack_injector import generate_micro_structuring_attacks
from app.simulator.clean_generator import generate_customer_profiles, generate_merchants, generate_transaction_base


def _base_customer():
    return pd.Series(
        {
            "customer_id": "CUST-000000",
            "mean_spend": 1000.0,
            "spend_variance": 100.0,
            "primary_devices": ["DEV-000000-0", "DEV-000000-1"],
            "usual_beneficiaries": ["CUST-000005", "CUST-000010"],
        }
    )


def _base_mutated_row(**overrides):
    row = {
        "amount": 3000.0,
        "timestamp": pd.Timestamp("2026-01-10 12:00:00"),
        "device_id": "DEV-000000-0",
        "beneficiary_id": "MULE-0001-0",
        "is_new_device": 0,
        "is_new_beneficiary": 1,
    }
    row.update(overrides)
    return pd.Series(row)


def test_validate_mutation_amount_bound_failure():
    customer = _base_customer()  # bound = mean_spend * 20 = 20,000
    mutated_row = _base_mutated_row(amount=25000.0)  # exceeds bound
    result = validate_mutation(mutated_row, mutated_row, customer, None, None)
    assert result["amount_ok"] is False
    assert result["valid"] is False
    # Other two constraints should still pass independently -- this case
    # only fails on amount.
    assert result["chronological_ok"] is True
    assert result["novelty_ok"] is True


def test_validate_mutation_amount_bound_success():
    customer = _base_customer()
    mutated_row = _base_mutated_row(amount=15000.0)  # within bound
    result = validate_mutation(mutated_row, mutated_row, customer, None, None)
    assert result["amount_ok"] is True


def test_validate_mutation_chronological_order_failure():
    customer = _base_customer()
    prior_timestamp = pd.Timestamp("2026-01-10 10:00:00")
    next_timestamp = pd.Timestamp("2026-01-10 14:00:00")
    # Mutated timestamp pushed BEFORE the prior real transaction -- violates order.
    mutated_row = _base_mutated_row(timestamp=pd.Timestamp("2026-01-10 09:00:00"))
    result = validate_mutation(mutated_row, mutated_row, customer, prior_timestamp, next_timestamp)
    assert result["chronological_ok"] is False
    assert result["valid"] is False


def test_validate_mutation_chronological_order_success():
    customer = _base_customer()
    prior_timestamp = pd.Timestamp("2026-01-10 10:00:00")
    next_timestamp = pd.Timestamp("2026-01-10 14:00:00")
    mutated_row = _base_mutated_row(timestamp=pd.Timestamp("2026-01-10 12:00:00"))
    result = validate_mutation(mutated_row, mutated_row, customer, prior_timestamp, next_timestamp)
    assert result["chronological_ok"] is True


def test_validate_mutation_novelty_flag_inconsistency_failure():
    customer = _base_customer()
    # device_id is NOT in customer's primary_devices (so is_new_device should
    # be 1), but the row's flag is stale/left at 0 -- exactly the "mutate
    # device identity but leave the flag unset" case CLAUDE.md §4.3 names.
    mutated_row = _base_mutated_row(device_id="DEV-999999-0", is_new_device=0)
    result = validate_mutation(mutated_row, mutated_row, customer, None, None)
    assert result["novelty_ok"] is False
    assert result["valid"] is False


def test_validate_mutation_novelty_flag_consistency_success():
    customer = _base_customer()
    mutated_row = _base_mutated_row(device_id="DEV-999999-0", is_new_device=1)
    result = validate_mutation(mutated_row, mutated_row, customer, None, None)
    assert result["novelty_ok"] is True


def test_compute_arg_matches_exact_formula():
    # ARG (%) = ((Initial - Final) / Initial) * 100
    assert compute_arg(0.10, 0.05) == pytest.approx(50.0)
    assert compute_arg(0.10, 0.10) == pytest.approx(0.0)
    assert compute_arg(0.10, 0.20) == pytest.approx(-100.0)


def test_compute_arg_zero_initial_raises():
    with pytest.raises(ValueError):
        compute_arg(0.0, 0.05)


@pytest.fixture(scope="module")
def small_arena_run():
    merchants = generate_merchants(N_MERCHANTS, seed=SEED)
    customers = generate_customer_profiles(N_CUSTOMERS, merchants, seed=SEED)
    clean = generate_transaction_base(customers, merchants, N_TRANSACTIONS, SIMULATION_DAYS, seed=SEED)
    attacks = generate_micro_structuring_attacks(
        MICRO_STRUCTURING_GENOME, customers, merchants, n_instances=500, seed=SEED
    )
    combined = combine_clean_and_injected(clean, attacks)
    featured = engineer_features(combined, customers)
    day4_result = run_blue_team_pipeline(featured, seed=SEED)

    summary = run_arena_mvp_gate(
        MICRO_STRUCTURING_GENOME,
        day4_result["model"],
        day4_result["train_df"],
        day4_result["test_df"],
        customers,
        clean,
        merchants,
        day4_result["graph_features"],
        feature_columns=FEATURE_COLUMNS,
        n_instances=30,  # small scale for test speed -- full 500 run verified manually
        seed=SEED,
    )
    return summary


def test_run_arena_mvp_gate_smoke(small_arena_run):
    summary = small_arena_run

    for key in (
        "run_id", "attack_family", "initial_evasion_rate", "final_evasion_rate",
        "robustness_gain", "hard_examples_count", "retrained_f1_score", "mutation_breakdown",
    ):
        assert key in summary

    assert summary["attack_family"] == "micro_structuring"
    assert 0.0 <= summary["initial_evasion_rate"] <= 1.0
    assert 0.0 <= summary["final_evasion_rate"] <= 1.0
    assert np.isfinite(summary["robustness_gain"])
    assert summary["hard_examples_count"] >= 0
    assert 0.0 <= summary["retrained_f1_score"] <= 1.0

    assert set(summary["mutation_breakdown"].keys()) == set(MICRO_STRUCTURING_GENOME["mutations"])
    for mutation_metrics in summary["mutation_breakdown"].values():
        assert 0.0 <= mutation_metrics["final_evasion_rate"] <= 1.0
        assert np.isfinite(mutation_metrics["robustness_gain"])


def test_run_arena_mvp_gate_matched_population_disjoint_transactions(small_arena_run):
    """Population is now MATCHED (same customers), not disjoint by customer --
    the hold-out property that matters is disjoint TRANSACTIONS relative to
    whatever fed hard-negative retraining.
    """
    diagnostics = small_arena_run["_diagnostics"]
    assert diagnostics["populations_matched"] is True
    assert set(diagnostics["final_customer_ids"]) == set(diagnostics["initial_customer_ids"])
    assert diagnostics["retest_disjoint_from_retraining_transactions"] is True


def test_run_arena_mvp_gate_harvest_reports_accept_reject_counts(small_arena_run):
    diagnostics = small_arena_run["_diagnostics"]
    assert diagnostics["harvest_accepted_count"] >= 0
    assert diagnostics["harvest_rejected_count"] >= 0
    assert diagnostics["harvest_accepted_count"] == small_arena_run["hard_examples_count"]
