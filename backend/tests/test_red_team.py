"""Tests for the micro_structuring attack injector (CLAUDE.md §0.10, Day 3)."""

import numpy as np
import pandas as pd
import pytest

from app.core.config import N_CUSTOMERS, N_MERCHANTS, SEED
from app.red_team.attack_genomes import MICRO_STRUCTURING_GENOME
from app.red_team.attack_injector import (
    generate_micro_structuring_attacks,
    validate_injected_transactions,
)
from app.simulator.clean_generator import generate_customer_profiles, generate_merchants

N_INSTANCES = 500
TARGET_AMOUNT = MICRO_STRUCTURING_GENOME["target_amount"]
LOW, HIGH = MICRO_STRUCTURING_GENOME["parameters"]["amount_per_tx_range"]


@pytest.fixture(scope="module")
def generated_attacks():
    merchants = generate_merchants(N_MERCHANTS, seed=SEED)
    customers = generate_customer_profiles(N_CUSTOMERS, merchants, seed=SEED)
    attacks = generate_micro_structuring_attacks(
        MICRO_STRUCTURING_GENOME, customers, merchants, n_instances=N_INSTANCES, seed=SEED
    )
    return customers, merchants, attacks


def test_split_amounts_sum_near_target(generated_attacks):
    customers, merchants, attacks = generated_attacks
    fraud = attacks[attacks["is_fraud"] == 1]
    per_instance_sum = fraud.groupby("instance_id")["amount"].sum()
    split_count = fraud.groupby("instance_id").size()
    dev_pct = 100 * (per_instance_sum - TARGET_AMOUNT) / TARGET_AMOUNT

    # split_count 11-15 have enough headroom (k * HIGH >= TARGET_AMOUNT * 0.98)
    # to hit the approved +-2% tolerance.
    feasible = split_count[split_count > 10].index
    assert (dev_pct.loc[feasible].abs() <= 2.0).all()

    # split_count == 10 is structurally capped at 10*HIGH = 48000, a hard
    # ceiling below TARGET_AMOUNT * 0.98 = 49000 -- it can never meet the
    # tolerance band no matter the reconciliation. Assert it lands
    # consistently at that ceiling rather than silently ignoring it.
    infeasible = split_count[split_count == 10].index
    assert len(infeasible) > 0  # sanity: this edge case actually occurs
    assert np.isclose(per_instance_sum.loc[infeasible], 10 * HIGH).all()

    # No instance ever exceeds the physical max or falls under the physical min.
    assert (per_instance_sum <= split_count * HIGH + 1e-6).all()
    assert (per_instance_sum >= split_count * LOW - 1e-6).all()


def test_split_count_distribution(generated_attacks):
    _, _, attacks = generated_attacks
    fraud = attacks[attacks["is_fraud"] == 1]
    split_count = fraud.groupby("instance_id").size()

    assert split_count.between(10, 15).all()
    value_counts = split_count.value_counts()
    assert value_counts.shape[0] >= 5  # spans most of the [10,15] range
    assert (value_counts / len(split_count) < 0.4).all()  # not clustered on one value


def test_inter_arrival_looks_exponential(generated_attacks):
    _, _, attacks = generated_attacks
    fraud = attacks[attacks["is_fraud"] == 1]

    def gaps_for_group(group):
        t = group.sort_values("timestamp")["timestamp"].values.astype("datetime64[s]").astype(float)
        return np.diff(t) / 3600.0

    gaps = np.concatenate(
        [gaps_for_group(g) for _, g in fraud.groupby("instance_id")]
    )
    assert len(gaps) > 0

    mean, std = gaps.mean(), gaps.std()
    coefficient_of_variation = std / mean
    # Exponential: CV ~= 1. Uniform: CV ~= 0.577. This distinguishes the two.
    assert 0.7 < coefficient_of_variation < 1.4

    skewness = pd.Series(gaps).skew()
    assert skewness > 0.5  # exponential is right-skewed; uniform is ~0


def test_noise_counts_and_fraud_flags(generated_attacks):
    _, _, attacks = generated_attacks
    fraud = attacks[attacks["is_fraud"] == 1]
    noise = attacks[attacks["is_fraud"] == 0]

    assert (fraud["attack_family"] == "micro_structuring").all()
    assert (fraud["genome_id"] == "ATK-MS-001").all()
    assert noise["attack_family"].isna().all()
    assert noise["genome_id"].isna().all()

    noise_count = noise.groupby("instance_id").size()
    fraud_count = fraud.groupby("instance_id").size()
    share = noise_count / (noise_count + fraud_count)
    assert share.between(0.25, 0.45).mean() > 0.9  # centered near the 0.35 target


def test_mule_beneficiary_uniqueness_and_no_collision(generated_attacks):
    customers, _, attacks = generated_attacks
    fraud = attacks[attacks["is_fraud"] == 1]

    mule_ids = fraud["beneficiary_id"]
    assert mule_ids.str.startswith("MULE-").all()

    all_usual_beneficiaries = {b for lst in customers["usual_beneficiaries"] for b in lst}
    assert len(set(mule_ids.unique()) & all_usual_beneficiaries) == 0
    assert len(set(mule_ids.unique()) & set(customers["customer_id"])) == 0

    # Each mule id must belong to exactly one instance (no cross-instance reuse).
    per_mule_instances = fraud.groupby("beneficiary_id")["instance_id"].nunique()
    assert (per_mule_instances == 1).all()


def test_injected_transaction_schema_validation(generated_attacks):
    _, _, attacks = generated_attacks
    validated = validate_injected_transactions(attacks)
    assert len(validated) == len(attacks)
    fraud_validated = [v for v in validated if v.is_fraud == 1]
    assert len(fraud_validated) == (attacks["is_fraud"] == 1).sum()


# ============================================================================
# synthetic_identity_drift (ATK-ID-001) -- Day 6. Appended below; the
# micro_structuring tests above are untouched.
# ============================================================================

from app.red_team.attack_genomes import SYNTHETIC_IDENTITY_DRIFT_GENOME  # noqa: E402
from app.red_team.attack_injector import generate_identity_drift_attacks  # noqa: E402

DRIFT_N_INSTANCES = 500
DRIFT_WINDOW_RANGE = SYNTHETIC_IDENTITY_DRIFT_GENOME["parameters"]["drift_window_days_range"]
EXTRACTION_COUNT_RANGE = SYNTHETIC_IDENTITY_DRIFT_GENOME["parameters"]["extraction_transaction_count_range"]
EXTRACTION_WINDOW_HOURS = SYNTHETIC_IDENTITY_DRIFT_GENOME["parameters"]["extraction_window_hours"]
MULTIPLIER_RANGE = SYNTHETIC_IDENTITY_DRIFT_GENOME["parameters"]["extraction_amount_multiplier_range"]


@pytest.fixture(scope="module")
def generated_drift_attacks():
    merchants = generate_merchants(N_MERCHANTS, seed=SEED)
    customers = generate_customer_profiles(N_CUSTOMERS, merchants, seed=SEED)
    attacks = generate_identity_drift_attacks(
        SYNTHETIC_IDENTITY_DRIFT_GENOME, customers, merchants, n_instances=DRIFT_N_INSTANCES, seed=SEED
    )
    return customers, merchants, attacks


def test_drift_window_within_approved_range(generated_drift_attacks):
    _, _, attacks = generated_drift_attacks
    drift_per_instance = attacks.groupby("instance_id")["drift_window_days"].first()
    assert len(drift_per_instance) == DRIFT_N_INSTANCES
    assert drift_per_instance.between(*DRIFT_WINDOW_RANGE).all()


def test_extraction_burst_count_and_timing(generated_drift_attacks):
    _, _, attacks = generated_drift_attacks
    counts = attacks.groupby("instance_id").size()
    assert counts.between(*EXTRACTION_COUNT_RANGE).all()

    def offsets_within_window(group):
        start = group["timestamp"].min()
        drift_days = group["drift_window_days"].iloc[0]
        # extraction burst start = SIMULATION_START_DATE + drift_days; every
        # row in the burst must land within extraction_window_hours of that.
        from app.core.config import SIMULATION_START_DATE
        burst_start = pd.Timestamp(SIMULATION_START_DATE) + pd.Timedelta(days=drift_days)
        offsets_hours = (group["timestamp"] - burst_start).dt.total_seconds() / 3600
        return offsets_hours.between(-1e-6, EXTRACTION_WINDOW_HOURS + 1e-6).all()

    all_within = attacks.groupby("instance_id").apply(offsets_within_window, include_groups=False)
    assert all_within.all()


def test_extraction_amount_scales_with_customer_mean_spend(generated_drift_attacks):
    customers, _, attacks = generated_drift_attacks
    mean_spend_map = customers.set_index("customer_id")["mean_spend"]

    totals = attacks.groupby("instance_id")["amount"].sum()
    customer_per_instance = attacks.groupby("instance_id")["customer_id"].first()
    achieved_multiplier = totals / customer_per_instance.map(mean_spend_map)

    assert achieved_multiplier.between(*MULTIPLIER_RANGE).all()

    # Real cross-tier check: at least two distinct income tiers represented,
    # and achieved totals differ meaningfully (not a flat amount).
    tier_map = customers.set_index("customer_id")["_income_tier"]
    tiers_used = customer_per_instance.map(tier_map).unique()
    assert len(tiers_used) >= 2
    assert totals.std() > 0


def test_drift_extraction_uses_distinct_prefixes_no_collision(generated_drift_attacks):
    customers, _, attacks = generated_drift_attacks

    assert attacks["device_id"].str.startswith("DRIFT-DEV-").all()
    assert attacks["beneficiary_id"].str.startswith("DRIFT-PAYEE-").all()

    primary_devices_map = customers.set_index("customer_id")["primary_devices"]
    usual_beneficiaries_map = customers.set_index("customer_id")["usual_beneficiaries"]
    device_collision = attacks.apply(lambda r: r["device_id"] in primary_devices_map[r["customer_id"]], axis=1)
    beneficiary_collision = attacks.apply(
        lambda r: r["beneficiary_id"] in usual_beneficiaries_map[r["customer_id"]], axis=1
    )
    assert not device_collision.any()
    assert not beneficiary_collision.any()


def test_drift_injected_transaction_schema_validation(generated_drift_attacks):
    _, _, attacks = generated_drift_attacks
    validated = validate_injected_transactions(attacks)
    assert len(validated) == len(attacks)


def test_drift_fraud_fields_set_correctly(generated_drift_attacks):
    _, _, attacks = generated_drift_attacks
    # This injector only ever emits the extraction burst -- the customer's
    # real pre-existing history is a separate DataFrame entirely (approved
    # design decision F), so every row here must be fraud-labeled.
    assert (attacks["is_fraud"] == 1).all()
    assert (attacks["attack_family"] == "synthetic_identity_drift").all()
    assert (attacks["genome_id"] == "ATK-ID-001").all()


# ============================================================================
# behavioral_camouflage (ATK-BC-001) -- Day 6. Appended below; the
# micro_structuring and synthetic_identity_drift tests above are untouched.
# ============================================================================

from app.red_team.attack_genomes import BEHAVIORAL_CAMOUFLAGE_GENOME  # noqa: E402
from app.red_team.attack_injector import generate_behavioral_camouflage_attacks  # noqa: E402

CAMO_N_INSTANCES = 500
CAMO_BURST_COUNT_RANGE = BEHAVIORAL_CAMOUFLAGE_GENOME["parameters"]["burst_transaction_count_range"]
CAMO_WINDOW_HOURS = BEHAVIORAL_CAMOUFLAGE_GENOME["parameters"]["burst_window_hours"]
CAMO_FRAUD_RATIO = BEHAVIORAL_CAMOUFLAGE_GENOME["parameters"]["fraud_leg_ratio"]


@pytest.fixture(scope="module")
def generated_camouflage_attacks():
    merchants = generate_merchants(N_MERCHANTS, seed=SEED)
    customers = generate_customer_profiles(N_CUSTOMERS, merchants, seed=SEED)
    attacks = generate_behavioral_camouflage_attacks(
        BEHAVIORAL_CAMOUFLAGE_GENOME, customers, merchants, n_instances=CAMO_N_INSTANCES, seed=SEED
    )
    return customers, merchants, attacks


def test_camouflage_burst_count_and_fraud_ratio(generated_camouflage_attacks):
    _, _, attacks = generated_camouflage_attacks
    counts = attacks.groupby("instance_id").size()
    assert len(counts) == CAMO_N_INSTANCES
    assert counts.between(*CAMO_BURST_COUNT_RANGE).all()

    fraud_counts = attacks[attacks["is_fraud"] == 1].groupby("instance_id").size().reindex(counts.index, fill_value=0)
    achieved_ratio = fraud_counts / counts
    # Small-integer rounding means per-instance ratio varies; the achieved
    # mean across 500 instances should sit close to the 0.3 target.
    assert achieved_ratio.mean() == pytest.approx(CAMO_FRAUD_RATIO, abs=0.05)
    assert (fraud_counts > 0).all()  # every instance has at least one fraud leg


def test_camouflage_burst_within_window(generated_camouflage_attacks):
    _, _, attacks = generated_camouflage_attacks

    def span_hours(group):
        return (group["timestamp"].max() - group["timestamp"].min()).total_seconds() / 3600

    spans = attacks.groupby("instance_id").apply(span_hours, include_groups=False)
    assert (spans <= CAMO_WINDOW_HOURS + 1e-6).all()


def test_camouflage_fraud_legs_use_customers_own_device(generated_camouflage_attacks):
    """Core mechanism: 0% novel devices on fraud rows (reuse_customer_device)."""
    customers, _, attacks = generated_camouflage_attacks
    fraud = attacks[attacks["is_fraud"] == 1]
    primary_devices_map = customers.set_index("customer_id")["primary_devices"]
    novel_device = fraud.apply(lambda r: r["device_id"] not in primary_devices_map[r["customer_id"]], axis=1)
    assert novel_device.sum() == 0


def test_camouflage_fraud_legs_avoid_p2p_transfer_placeholder(generated_camouflage_attacks):
    """Core mechanism: fraud legs use real merchants/channels, never the
    P2P-TRANSFER placeholder families #1/#2 use (use_real_merchant_and_channel).
    """
    _, _, attacks = generated_camouflage_attacks
    fraud = attacks[attacks["is_fraud"] == 1]
    assert (fraud["merchant_id"] != "P2P-TRANSFER").all()
    assert (fraud["merchant_category"] != "p2p_transfer").all()
    assert fraud["channel"].isin(["POS", "WEB", "P2P"]).all()
    assert fraud["channel"].nunique() > 1  # genuinely uses the normal channel mix, not one fixed value


def test_camouflage_fraud_amounts_close_to_customer_baseline(generated_camouflage_attacks):
    """Core mechanism: fraud amounts statistically resemble the customer's
    own normal spend, unlike family #1's distinct [2500,4800] band.
    """
    customers, _, attacks = generated_camouflage_attacks
    fraud = attacks[attacks["is_fraud"] == 1]
    mean_spend_map = customers.set_index("customer_id")["mean_spend"]

    fraud_mean_per_customer = fraud.groupby("customer_id")["amount"].mean()
    baseline_mean_per_customer = fraud_mean_per_customer.index.to_series().map(mean_spend_map)

    # Ratio of fraud-leg mean amount to the customer's own mean_spend should
    # cluster near 1.0 (same distribution), not near a fixed unrelated value.
    ratio = fraud_mean_per_customer / baseline_mean_per_customer
    assert ratio.median() == pytest.approx(1.0, abs=0.5)


def test_camouflage_payee_prefix_and_no_collision(generated_camouflage_attacks):
    customers, _, attacks = generated_camouflage_attacks
    fraud = attacks[attacks["is_fraud"] == 1]
    assert fraud["beneficiary_id"].str.startswith("CAMO-PAYEE-").all()

    usual_beneficiaries_map = customers.set_index("customer_id")["usual_beneficiaries"]
    collision = fraud.apply(lambda r: r["beneficiary_id"] in usual_beneficiaries_map[r["customer_id"]], axis=1)
    assert collision.sum() == 0


def test_camouflage_injected_transaction_schema_validation(generated_camouflage_attacks):
    _, _, attacks = generated_camouflage_attacks
    validated = validate_injected_transactions(attacks)
    assert len(validated) == len(attacks)
    fraud_validated = [v for v in validated if v.is_fraud == 1]
    assert len(fraud_validated) == (attacks["is_fraud"] == 1).sum()
