"""Tests for SHAP explainability (Day 8a, CLAUDE.md §7, PRD §7.3)."""

import pytest

from app.blue_team.detector import FEATURE_COLUMNS, run_blue_team_pipeline
from app.blue_team.explainability import FEATURE_DESCRIPTIONS, compute_reason_codes, find_cached_feature_row
from app.blue_team.features import combine_clean_and_injected, engineer_features
from app.core.config import N_CUSTOMERS, N_MERCHANTS, N_TRANSACTIONS, SEED, SIMULATION_DAYS
from app.red_team.attack_genomes import MICRO_STRUCTURING_GENOME
from app.red_team.attack_injector import generate_micro_structuring_attacks
from app.simulator.clean_generator import generate_customer_profiles, generate_merchants, generate_transaction_base

# Deferred features (Day 4 + Day 6 project-owner decisions) -- still excluded
# from FEATURE_COLUMNS, so SHAP can never attribute to them. Mirrors
# test_blue_team.py's LEAKAGE_PRONE_COLUMNS deferred set, plus Day 6's
# semantic/voice fields.
DEFERRED_FEATURES = {
    "is_new_beneficiary", "beneficiary_in_degree", "beneficiary_out_degree",
    "semantic_risk_score", "voice_confidence_score",
}


@pytest.fixture(scope="module")
def full_pipeline_result():
    merchants = generate_merchants(N_MERCHANTS, seed=SEED)
    customers = generate_customer_profiles(N_CUSTOMERS, merchants, seed=SEED)
    clean = generate_transaction_base(customers, merchants, N_TRANSACTIONS, SIMULATION_DAYS, seed=SEED)
    attacks = generate_micro_structuring_attacks(
        MICRO_STRUCTURING_GENOME, customers, merchants, n_instances=500, seed=SEED
    )
    combined = combine_clean_and_injected(clean, attacks)
    featured = engineer_features(combined, customers)
    return run_blue_team_pipeline(featured, seed=SEED)


def test_feature_descriptions_covers_every_feature_column():
    assert set(FEATURE_DESCRIPTIONS.keys()) == set(FEATURE_COLUMNS)


def test_feature_descriptions_never_mentions_deferred_features():
    assert set(FEATURE_DESCRIPTIONS.keys()).isdisjoint(DEFERRED_FEATURES)


def test_find_cached_feature_row_found_in_test_df(full_pipeline_result):
    test_df = full_pipeline_result["test_df"]
    tx_id = test_df.iloc[0]["transaction_id"]
    row = find_cached_feature_row(tx_id, full_pipeline_result["train_df"], test_df)
    assert row is not None
    assert row["transaction_id"] == tx_id


def test_find_cached_feature_row_found_in_train_df(full_pipeline_result):
    train_df = full_pipeline_result["train_df"]
    tx_id = train_df.iloc[0]["transaction_id"]
    row = find_cached_feature_row(tx_id, train_df, full_pipeline_result["test_df"])
    assert row is not None
    assert row["transaction_id"] == tx_id


def test_find_cached_feature_row_not_found_returns_none(full_pipeline_result):
    row = find_cached_feature_row(
        "NOT-A-REAL-TXN-ID", full_pipeline_result["train_df"], full_pipeline_result["test_df"]
    )
    assert row is None


def test_compute_reason_codes_returns_top_3_sorted_by_absolute_magnitude(full_pipeline_result):
    test_df, model = full_pipeline_result["test_df"], full_pipeline_result["model"]
    row = test_df.iloc[0]
    codes = compute_reason_codes(row, model)

    assert len(codes) == 3
    for code in codes:
        assert set(code.keys()) == {"feature", "contribution", "description"}
        assert code["feature"] in FEATURE_COLUMNS

    magnitudes = [abs(float(code["contribution"])) for code in codes]
    assert magnitudes == sorted(magnitudes, reverse=True)


def test_compute_reason_codes_contribution_is_signed_two_decimal_string(full_pipeline_result):
    test_df, model = full_pipeline_result["test_df"], full_pipeline_result["model"]
    codes = compute_reason_codes(test_df.iloc[0], model)
    for code in codes:
        assert code["contribution"][0] in "+-"
        decimals = code["contribution"].split(".")[1]
        assert len(decimals) == 2


def test_compute_reason_codes_differs_per_transaction_not_static(full_pipeline_result):
    """A real per-row explanation, not a templated/static response."""
    test_df, model = full_pipeline_result["test_df"], full_pipeline_result["model"]
    fraud_row = test_df[test_df["is_fraud"] == 1].iloc[0]
    clean_row = test_df[test_df["is_fraud"] == 0].iloc[0]

    fraud_codes = compute_reason_codes(fraud_row, model)
    clean_codes = compute_reason_codes(clean_row, model)
    assert fraud_codes != clean_codes


def test_compute_reason_codes_never_mentions_deferred_features(full_pipeline_result):
    test_df, model = full_pipeline_result["test_df"], full_pipeline_result["model"]
    for _, row in test_df.head(20).iterrows():
        codes = compute_reason_codes(row, model)
        features_used = {code["feature"] for code in codes}
        assert features_used.isdisjoint(DEFERRED_FEATURES)
