"""Tests for the Blue Team feature/graph/detector pipeline (CLAUDE.md §0.10, Day 4)."""

import numpy as np
import pandas as pd
import pytest

from app.core.config import N_CUSTOMERS, N_MERCHANTS, N_TRANSACTIONS, SEED, SIMULATION_DAYS
from app.blue_team.detector import FEATURE_COLUMNS, run_blue_team_pipeline
from app.blue_team.features import add_rolling_velocity_features, combine_clean_and_injected, engineer_features
from app.red_team.attack_genomes import MICRO_STRUCTURING_GENOME
from app.red_team.attack_injector import generate_micro_structuring_attacks
from app.simulator.clean_generator import generate_customer_profiles, generate_merchants, generate_transaction_base

LEAKAGE_PRONE_COLUMNS = {
    "is_fraud", "genome_id", "attack_family",
    "merchant_id", "merchant_category", "channel",
    "transaction_id", "customer_id", "beneficiary_id", "device_id", "timestamp",
    # INTENTIONALLY DEFERRED (Day 4, project-owner decision): is_new_beneficiary,
    # beneficiary_in_degree, and beneficiary_out_degree are a near-deterministic
    # tell against the single un-mutated micro_structuring genome (every fraud
    # leg routes to a brand-new mule beneficiary), which would push F1 to
    # ~0.997 and leave the Day 5 Adversarial Arena nothing real to evade.
    # TODO: re-add after Day 6 (multiple attack families) once beneficiary
    # novelty is no longer a near-total shortcut for a single genome.
    "is_new_beneficiary", "beneficiary_in_degree", "beneficiary_out_degree",
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
    result = run_blue_team_pipeline(featured, seed=SEED)
    result["customers"] = customers
    result["featured"] = featured
    return result


def test_rolling_features_hand_built_example():
    """Hand-built mini example: independently recompute expected prior-only
    rolling stats via explicit boolean filtering, not by re-deriving from the
    same rolling code -- a real check, not a tautology.
    """
    base = pd.Timestamp("2024-01-01 00:00:00")
    df = pd.DataFrame(
        {
            "customer_id": ["C1"] * 5,
            "timestamp": [
                base,
                base + pd.Timedelta(minutes=2),
                base + pd.Timedelta(minutes=90),
                base + pd.Timedelta(hours=24),
                base + pd.Timedelta(days=8),
            ],
            "amount": [100.0, 200.0, 300.0, 400.0, 500.0],
        }
    )

    result = add_rolling_velocity_features(df)

    windows = {"5m": pd.Timedelta(minutes=5), "1h": pd.Timedelta(hours=1),
               "24h": pd.Timedelta(hours=24), "7d": pd.Timedelta(days=7)}
    for i, row in result.iterrows():
        prior = result[result["timestamp"] < row["timestamp"]]
        for label, span in windows.items():
            window_start = row["timestamp"] - span
            in_window = prior[prior["timestamp"] > window_start]
            assert row[f"count_{label}"] == len(in_window), f"row {i} count_{label}"
            assert np.isclose(row[f"sum_{label}"], in_window["amount"].sum()), f"row {i} sum_{label}"

    # Every row's own timestamp must never appear in its own prior set.
    for i, row in result.iterrows():
        prior = result[result["timestamp"] < row["timestamp"]]
        assert row["timestamp"] not in prior["timestamp"].values


def test_worked_example_no_self_leakage(full_pipeline_result):
    """Pick one real transaction and confirm the exact rows that fed into its
    5m/1h/24h/7d windows, independently recomputed -- not re-deriving from
    the same function.
    """
    featured = full_pipeline_result["featured"]
    candidates = featured[(featured["count_24h"] >= 2) & (featured["count_7d"] >= 4)]
    assert len(candidates) > 0
    target = candidates.iloc[0]

    cust_rows = featured[featured["customer_id"] == target["customer_id"]]
    prior = cust_rows[cust_rows["timestamp"] < target["timestamp"]]

    assert target["transaction_id"] not in prior["transaction_id"].values

    for label, hours in [("5m", 5 / 60), ("1h", 1), ("24h", 24), ("7d", 24 * 7)]:
        window_start = target["timestamp"] - pd.Timedelta(hours=hours)
        in_window = prior[prior["timestamp"] > window_start]
        assert target[f"count_{label}"] == len(in_window), f"count_{label} mismatch"
        assert np.isclose(target[f"sum_{label}"], in_window["amount"].sum()), f"sum_{label} mismatch"


def test_is_new_beneficiary_flags_mule_ids(full_pipeline_result):
    featured = full_pipeline_result["featured"]
    mule_rows = featured[featured["beneficiary_id"].str.startswith("MULE-")]
    assert len(mule_rows) > 0
    assert (mule_rows["is_new_beneficiary"] == 1).all()

    # Every fraud row is a mule payment and must be flagged new.
    fraud_rows = featured[featured["is_fraud"] == 1]
    assert (fraud_rows["is_new_beneficiary"] == 1).all()

    # Clean rows should very rarely be flagged new (persistence > 0.95).
    clean_rows = featured[featured["is_fraud"] == 0]
    assert clean_rows["is_new_beneficiary"].mean() < 0.05


def test_is_new_location_logic(full_pipeline_result):
    featured = full_pipeline_result["featured"]
    customers = full_pipeline_result["customers"]
    base_location_map = customers.set_index("customer_id")["base_location"]

    expected = featured["customer_id"].map(base_location_map) != featured["location"]
    assert (featured["is_new_location"] == expected.astype(int)).all()
    assert featured["is_new_location"].mean() > 0  # some deviation actually occurs
    assert featured["is_new_location"].mean() < 0.5  # but it's the minority case


def test_graph_built_on_train_split_only(full_pipeline_result):
    graph_features = full_pipeline_result["graph_features"]
    train_df = full_pipeline_result["train_df"]
    featured = full_pipeline_result["featured"]

    assert graph_features["train_row_count"] == len(train_df)
    assert graph_features["train_row_count"] < len(featured)


def test_feature_columns_exclude_leakage_prone_fields():
    assert set(FEATURE_COLUMNS).isdisjoint(LEAKAGE_PRONE_COLUMNS)


def test_model_metrics_believable_range(full_pipeline_result):
    metrics = full_pipeline_result["metrics"]

    assert 0.6 <= metrics["f1"] <= 0.97, (
        f"F1={metrics['f1']:.4f} outside the believable [0.6, 0.97] band -- "
        "flagging per CLAUDE.md §0.9 rather than silently accepting it."
    )
    assert 0 < metrics["precision"] < 1
    assert 0 < metrics["recall"] < 1
    assert 0 < metrics["pr_auc"] < 1
    assert metrics["fpr"] < 0.1


def test_class_imbalance_ratio_reported(full_pipeline_result):
    featured = full_pipeline_result["featured"]
    fraud_ratio = featured["is_fraud"].mean()
    assert 0.05 < fraud_ratio < 0.20  # sanity: neither trivially rare nor balanced
