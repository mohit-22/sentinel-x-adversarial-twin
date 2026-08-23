"""Tests for the synthetic payment twin generator (CLAUDE.md §0.10, §12)."""

import numpy as np
import pandas as pd

from app.core.config import (
    ENTITY_PERSISTENCE_PROB,
    FIDELITY_SIMILARITY_TARGET,
    N_CUSTOMERS,
    N_MERCHANTS,
    N_TRANSACTIONS,
    SEED,
    SIMULATION_DAYS,
)
from app.core.schemas import CustomerProfile, TransactionBase
from app.simulator.clean_generator import (
    generate_customer_profiles,
    generate_merchants,
    generate_transaction_base,
    simulate_payment_twin,
    validate_customers,
    validate_transactions,
)
from app.simulator.fidelity import compute_fidelity_report


def test_generate_merchants_count_and_categories():
    merchants = generate_merchants(N_MERCHANTS, seed=SEED)
    assert len(merchants) == N_MERCHANTS
    assert merchants["merchant_id"].nunique() == N_MERCHANTS
    assert merchants["merchant_category"].nunique() > 1


def test_generate_customer_profiles_count_and_schema():
    merchants = generate_merchants(N_MERCHANTS, seed=SEED)
    customers = generate_customer_profiles(N_CUSTOMERS, merchants, seed=SEED)
    assert len(customers) == N_CUSTOMERS
    assert customers["customer_id"].nunique() == N_CUSTOMERS

    device_counts = customers["primary_devices"].apply(len)
    assert device_counts.between(1, 2).all()

    validated = validate_customers(customers)
    assert len(validated) == N_CUSTOMERS
    assert isinstance(validated[0], CustomerProfile)


def test_income_tier_spend_ordering():
    merchants = generate_merchants(N_MERCHANTS, seed=SEED)
    customers = generate_customer_profiles(N_CUSTOMERS, merchants, seed=SEED)
    medians = customers.groupby("_income_tier")["mean_spend"].median()
    assert medians["mass"] < medians["affluent"] < medians["premium"]
    assert customers["mean_spend"].skew() > 0  # right-skewed, log-normal signature


def test_generate_transaction_base_count_and_schema():
    merchants = generate_merchants(N_MERCHANTS, seed=SEED)
    customers = generate_customer_profiles(N_CUSTOMERS, merchants, seed=SEED)
    transactions = generate_transaction_base(
        customers, merchants, N_TRANSACTIONS, SIMULATION_DAYS, seed=SEED
    )
    assert len(transactions) >= N_TRANSACTIONS
    assert (transactions["currency"] == "INR").all()

    start = pd.Timestamp("2026-01-01T00:00:00")
    end = start + pd.Timedelta(days=SIMULATION_DAYS)
    assert (transactions["timestamp"] >= start).all()
    assert (transactions["timestamp"] < end).all()

    validated = validate_transactions(transactions)
    assert len(validated) >= N_TRANSACTIONS
    assert isinstance(validated[0], TransactionBase)

    # No customer should ever pay themself.
    assert (transactions["customer_id"] != transactions["beneficiary_id"]).all()


def test_diurnal_timing_distribution():
    merchants = generate_merchants(N_MERCHANTS, seed=SEED)
    customers = generate_customer_profiles(N_CUSTOMERS, merchants, seed=SEED)
    transactions = generate_transaction_base(
        customers, merchants, N_TRANSACTIONS, SIMULATION_DAYS, seed=SEED
    )
    hour = transactions["timestamp"].dt.hour
    daytime_fraction = hour.between(9, 20).mean()
    assert daytime_fraction > 0.6  # heavy 09:00-21:00 vs light overnight


def test_entity_persistence_ratios():
    merchants = generate_merchants(N_MERCHANTS, seed=SEED)
    customers = generate_customer_profiles(N_CUSTOMERS, merchants, seed=SEED)
    transactions = generate_transaction_base(
        customers, merchants, N_TRANSACTIONS, SIMULATION_DAYS, seed=SEED
    )

    devices_by_customer = customers.set_index("customer_id")["primary_devices"]
    merchants_by_customer = customers.set_index("customer_id")["usual_merchants"]
    beneficiaries_by_customer = customers.set_index("customer_id")["usual_beneficiaries"]

    is_usual_device = [
        d in devices_by_customer[c]
        for c, d in zip(transactions["customer_id"], transactions["device_id"])
    ]
    is_usual_merchant = [
        m in merchants_by_customer[c]
        for c, m in zip(transactions["customer_id"], transactions["merchant_id"])
    ]
    is_usual_beneficiary = [
        b in beneficiaries_by_customer[c]
        for c, b in zip(transactions["customer_id"], transactions["beneficiary_id"])
    ]

    assert np.mean(is_usual_device) > 0.95
    assert np.mean(is_usual_merchant) > 0.95
    assert np.mean(is_usual_beneficiary) > 0.95
    assert ENTITY_PERSISTENCE_PROB > 0.95


def test_reproducibility_fixed_seed():
    customers_a, transactions_a = simulate_payment_twin(
        n_customers=500, n_merchants=50, n_transactions=2000, days=SIMULATION_DAYS, seed=SEED
    )
    customers_b, transactions_b = simulate_payment_twin(
        n_customers=500, n_merchants=50, n_transactions=2000, days=SIMULATION_DAYS, seed=SEED
    )
    pd.testing.assert_frame_equal(customers_a, customers_b)
    pd.testing.assert_frame_equal(transactions_a, transactions_b)


def test_fidelity_scores_above_threshold():
    merchants = generate_merchants(N_MERCHANTS, seed=SEED)
    customers = generate_customer_profiles(N_CUSTOMERS, merchants, seed=SEED)
    transactions = generate_transaction_base(
        customers, merchants, N_TRANSACTIONS, SIMULATION_DAYS, seed=SEED
    )
    report = compute_fidelity_report(customers, transactions, seed=SEED)
    assert report["amount_similarity_overall"] > FIDELITY_SIMILARITY_TARGET
    assert report["temporal_similarity_overall"] > FIDELITY_SIMILARITY_TARGET
