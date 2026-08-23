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
