"""Tests for the Adversarial Arena (CLAUDE.md §0.10, §4.3, Day 5 MVP gate)."""

import numpy as np
import pandas as pd
import pytest

from app.core.config import N_CUSTOMERS, N_MERCHANTS, N_TRANSACTIONS, SEED, SIMULATION_DAYS
from app.blue_team.detector import FEATURE_COLUMNS, run_blue_team_pipeline
from app.blue_team.features import combine_clean_and_injected, engineer_features
from app.red_team.arena import MUTATION_REGISTRY, apply_mutation, compute_arg, run_arena_mvp_gate, validate_mutation
from app.red_team.attack_genomes import (
    BEHAVIORAL_CAMOUFLAGE_GENOME,
    MICRO_STRUCTURING_GENOME,
    SOCIAL_ENGINEERING_COERCION_GENOME,
    SYNTHETIC_IDENTITY_DRIFT_GENOME,
    SYNTHETIC_VOICE_AUTHORIZATION_GENOME,
)
from app.red_team.attack_injector import ATTACK_GENERATORS, generate_micro_structuring_attacks
from app.simulator.clean_generator import generate_customer_profiles, generate_merchants, generate_transaction_base

ALL_GENOMES = [
    MICRO_STRUCTURING_GENOME,
    SYNTHETIC_IDENTITY_DRIFT_GENOME,
    BEHAVIORAL_CAMOUFLAGE_GENOME,
    SOCIAL_ENGINEERING_COERCION_GENOME,
    SYNTHETIC_VOICE_AUTHORIZATION_GENOME,
]


def _base_customer():
    return pd.Series(
        {
            "customer_id": "CUST-000000",
            "mean_spend": 1000.0,
            "spend_variance": 100.0,
            "base_location": "Erode",
            "primary_devices": ["DEV-000000-0", "DEV-000000-1"],
            "usual_beneficiaries": ["CUST-000005", "CUST-000010"],
            "usual_merchants": ["MERCH-0001", "MERCH-0002"],
        }
    )


def _base_merchants():
    return pd.DataFrame(
        {"merchant_id": ["MERCH-0001", "MERCH-0002"], "merchant_category": ["grocery", "dining"]}
    )


def _base_rows(**overrides):
    row = {
        "transaction_id": ["TX-0000-F00"],
        "timestamp": [pd.Timestamp("2026-01-10 12:00:00")],
        "customer_id": ["CUST-000000"],
        "amount": [3000.0],
        "device_id": ["DRIFT-DEV-0000-0"],
        "beneficiary_id": ["MULE-0000-0"],
        "semantic_risk_score": [0.85],
    }
    row.update(overrides)
    return pd.DataFrame(row)


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


# ============================================================================
# Day 6.5: generator registry + generic mutation dispatch
# ============================================================================


def test_attack_generators_registry_covers_all_five_families():
    for genome in ALL_GENOMES:
        family = genome["family"]
        assert family in ATTACK_GENERATORS, f"{family} missing from ATTACK_GENERATORS"
        assert callable(ATTACK_GENERATORS[family]["instance_fn"])
        assert callable(ATTACK_GENERATORS[family]["attacks_fn"])


def test_mutation_registry_covers_every_genome_mutation_no_missing_entries():
    """Completeness check: every mutation string in every genome's
    "mutations" list has a corresponding registry entry.
    """
    missing = []
    total_pairs = 0
    for genome in ALL_GENOMES:
        for mutation_name in genome["mutations"]:
            total_pairs += 1
            if (genome["family"], mutation_name) not in MUTATION_REGISTRY:
                missing.append((genome["family"], mutation_name))
    assert missing == []
    assert total_pairs == 15  # 5 families x 3 mutations each
    assert len(MUTATION_REGISTRY) == 15


def test_apply_mutation_unknown_pair_raises():
    customer = _base_customer()
    merchants = _base_merchants()
    rows = _base_rows()
    with pytest.raises(ValueError):
        apply_mutation(rows, rows, customer, merchants, "not_a_real_family", "not_a_real_mutation", np.random.default_rng(0))


def test_stretch_timing_generic_via_increase_time_spacing():
    customer = _base_customer()
    merchants = _base_merchants()
    sibling_rows = _base_rows(timestamp=[pd.Timestamp("2026-01-10 10:00:00")])  # instance start
    rows = _base_rows(timestamp=[pd.Timestamp("2026-01-10 12:00:00")])  # 2h offset from start

    mutated = apply_mutation(rows, sibling_rows, customer, merchants, "micro_structuring", "increase_time_spacing", np.random.default_rng(0))
    expected_offset_hours = 2.0 * 2.5  # TIME_SPACING_MULTIPLIER
    actual_offset_hours = (mutated["timestamp"].iloc[0] - sibling_rows["timestamp"].iloc[0]).total_seconds() / 3600
    assert actual_offset_hours == pytest.approx(expected_offset_hours)


def test_swap_entity_field_recycled_pool_via_rotate_mule_accounts():
    customer = _base_customer()
    merchants = _base_merchants()
    sibling_rows = _base_rows(beneficiary_id=["MULE-0000-0"])
    rows = _base_rows(beneficiary_id=["MULE-0000-0"])

    mutated = apply_mutation(rows, sibling_rows, customer, merchants, "micro_structuring", "rotate_mule_accounts", np.random.default_rng(0))
    assert mutated["beneficiary_id"].iloc[0].startswith("MUTATED-RECYCLED-BENEFICIARY_ID-")
    assert mutated["beneficiary_id"].iloc[0] != "MULE-0000-0"


def test_swap_entity_field_customers_own_via_reuse_known_device():
    customer = _base_customer()
    merchants = _base_merchants()
    rows = _base_rows(device_id=["DRIFT-DEV-0000-0"])

    mutated = apply_mutation(rows, rows, customer, merchants, "synthetic_identity_drift", "reuse_known_device", np.random.default_rng(0))
    assert mutated["device_id"].iloc[0] in customer["primary_devices"]


def test_swap_entity_field_brand_new_via_combine_with_new_device():
    customer = _base_customer()
    merchants = _base_merchants()
    rows = _base_rows(device_id=["DEV-000000-0"])  # base genome uses customer's own device

    mutated = apply_mutation(rows, rows, customer, merchants, "social_engineering_coercion", "combine_with_new_device", np.random.default_rng(0))
    assert mutated["device_id"].iloc[0].startswith("MUTATED-DEVICE_ID-")
    assert mutated["device_id"].iloc[0] not in customer["primary_devices"]


def test_add_extra_rows_generic_via_add_legitimate_micro_purchases():
    customer = _base_customer()
    merchants = _base_merchants()
    rows = _base_rows()

    mutated = apply_mutation(rows, rows, customer, merchants, "micro_structuring", "add_legitimate_micro_purchases", np.random.default_rng(0))
    assert len(mutated) == 1 + 3  # original row untouched + N_LEGIT_PURCHASES_TO_ADD
    # the original fraud row must be present unchanged
    assert rows.iloc[0]["transaction_id"] in mutated["transaction_id"].values
    new_rows = mutated[mutated["transaction_id"] != rows.iloc[0]["transaction_id"]]
    assert (new_rows["is_fraud"] == 0).all()


def test_scale_risk_field_generic_via_lower_semantic_risk_score():
    customer = _base_customer()
    merchants = _base_merchants()
    rows = _base_rows(semantic_risk_score=[0.9])

    mutated = apply_mutation(rows, rows, customer, merchants, "social_engineering_coercion", "lower_semantic_risk_score", np.random.default_rng(0))
    assert mutated["semantic_risk_score"].iloc[0] == pytest.approx(0.9 * 0.5)


def test_narrow_amount_generic_via_match_customer_amount_profile():
    customer = _base_customer()
    merchants = _base_merchants()
    rows = _base_rows(amount=[5000.0])

    mutated = apply_mutation(rows, rows, customer, merchants, "behavioral_camouflage", "match_customer_amount_profile", np.random.default_rng(0))
    assert len(mutated) == 1
    assert mutated["amount"].iloc[0] > 0
    assert mutated["amount"].iloc[0] != 5000.0  # re-drawn, not left unchanged


def test_no_op_generic_via_vary_coercion_pretext():
    customer = _base_customer()
    merchants = _base_merchants()
    rows = _base_rows(semantic_risk_score=[0.9])

    mutated = apply_mutation(rows, rows, customer, merchants, "social_engineering_coercion", "vary_coercion_pretext", np.random.default_rng(0))
    pd.testing.assert_frame_equal(mutated.reset_index(drop=True), rows.reset_index(drop=True))


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


# ============================================================================
# Cross-Family Generalization Matrix -- run_multi_family_hardening
# (post-Day 8b differentiator). Reuses run_attack/harvest_hard_negatives/
# retrain/re_test unchanged -- these tests cover only the new orchestration.
# ============================================================================


@pytest.fixture(scope="module")
def multi_family_result():
    from app.red_team.arena import run_multi_family_hardening

    merchants = generate_merchants(N_MERCHANTS, seed=SEED)
    customers = generate_customer_profiles(N_CUSTOMERS, merchants, seed=SEED)
    clean = generate_transaction_base(customers, merchants, N_TRANSACTIONS, SIMULATION_DAYS, seed=SEED)
    attacks = generate_micro_structuring_attacks(
        MICRO_STRUCTURING_GENOME, customers, merchants, n_instances=500, seed=SEED
    )
    combined = combine_clean_and_injected(clean, attacks)
    featured = engineer_features(combined, customers)
    day4_result = run_blue_team_pipeline(featured, seed=SEED)

    return run_multi_family_hardening(
        ALL_GENOMES,
        day4_result["model"],
        day4_result["train_df"],
        day4_result["test_df"],
        customers,
        clean,
        merchants,
        day4_result["graph_features"],
        feature_columns=FEATURE_COLUMNS,
        n_instances=30,  # small scale for test speed -- full n=500 verified manually
        seed=SEED,
    )


def test_run_multi_family_hardening_covers_all_five_families(multi_family_result):
    assert set(multi_family_result["per_family"].keys()) == {g["family"] for g in ALL_GENOMES}


def test_run_multi_family_hardening_per_family_result_shape(multi_family_result):
    for family, data in multi_family_result["per_family"].items():
        for key in ("genome_id", "initial_evasion_rate", "final_evasion_rate", "robustness_gain", "hard_examples_count"):
            assert key in data, f"{family} missing {key}"
        assert 0.0 <= data["initial_evasion_rate"] <= 1.0
        assert 0.0 <= data["final_evasion_rate"] <= 1.0
        assert np.isfinite(data["robustness_gain"])
        assert data["hard_examples_count"] >= 0


def test_run_multi_family_hardening_returns_one_shared_model(multi_family_result):
    """All 5 families are re-tested against the SAME model object -- unlike
    run_arena_for_all_families, which trains an independent M1 per family.
    """
    assert "model" in multi_family_result
    assert multi_family_result["model"] is not None


def test_run_multi_family_hardening_total_hard_examples_is_sum_across_families(multi_family_result):
    per_family_sum = sum(d["hard_examples_count"] for d in multi_family_result["per_family"].values())
    assert multi_family_result["total_hard_examples_count"] == per_family_sum


def test_run_multi_family_hardening_reports_retrained_metrics_on_original_test_set(multi_family_result):
    metrics = multi_family_result["retrained_metrics"]
    for key in ("precision", "recall", "f1", "pr_auc", "fpr"):
        assert key in metrics
        assert 0.0 <= metrics[key] <= 1.0
