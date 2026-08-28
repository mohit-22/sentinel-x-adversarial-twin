"""Vectorized generation of the Sentinel-X synthetic payment twin.

Produces CustomerProfile and TransactionBase rows (PRD §7.1): log-normal
spend per income tier, diurnal timing, and persistent device/merchant/
beneficiary behavior (persistence probability > 0.95). No row-wise Python
loops over DataFrames — all sampling uses vectorized NumPy/Pandas ops.
"""

from typing import List, Tuple

import numpy as np
import pandas as pd
from faker import Faker
from pydantic import TypeAdapter

from app.core.config import (
    BENEFICIARIES_PER_CUSTOMER_RANGE,
    CHANNELS,
    CHANNEL_PROBS,
    DIURNAL_DAY_HOURS,
    DIURNAL_DAY_WEIGHT,
    DIURNAL_NIGHT_WEIGHT,
    ENTITY_PERSISTENCE_PROB,
    INCOME_TIERS,
    MERCHANT_CATEGORIES,
    MERCHANTS_PER_CUSTOMER_RANGE,
    NOVEL_LOCATION_PROB,
    N_CUSTOMERS,
    N_MERCHANTS,
    N_TRANSACTIONS,
    SEED,
    SIMULATION_DAYS,
    SIMULATION_START_DATE,
    TRANSACTION_CV,
)
from app.core.schemas import CustomerProfile, TransactionBase


def _ragged_row_positions(counts: np.ndarray) -> np.ndarray:
    """Position-within-row index for a flattened ragged array, e.g. counts=[2,3] -> [0,1,0,1,2]."""
    offsets = np.repeat(np.cumsum(counts) - counts, counts)
    return np.arange(counts.sum()) - offsets


def _sample_unique_from_pool(
    pool: np.ndarray, counts: np.ndarray, seed: int, exclude_self: bool = False,
    chunk_size: int = 500,
) -> Tuple[np.ndarray, np.ndarray]:
    """Sample `counts[i]` unique items from `pool` for each of len(counts) rows.

    Returns (flat_values, counts) where flat_values is row-major concatenated
    (row i occupies flat_values[offset_i : offset_i + counts[i]]). If
    exclude_self is True, pool must equal the row index domain (0..n-1) and
    row i never samples itself. Vectorized via per-row random ranking,
    processed in row-chunks (each chunk is still one vectorized NumPy op,
    not a per-row Python loop) to bound peak memory: a single n_rows x
    pool_size random matrix is O(n^2) when pool_size ~ n_rows -- exactly the
    usual_beneficiaries case (exclude_self=True, pool=all customer_ids), a
    ~1.7GB matrix+argsort+mask at N_CUSTOMERS=10,000 that OOM-killed the
    process on a 512MB host. Chunking bounds this to
    chunk_size x pool_size regardless of n_rows.

    NOTE: this consumes the seeded `np.random` stream in a different order
    than the single-matrix version, so exact per-row selections for a given
    seed differ from the pre-chunking implementation. The sampling
    distribution itself (uniform, unique, excludes self) is unchanged.
    """
    np.random.seed(seed)
    n_rows = counts.shape[0]
    pool_size = pool.shape[0]
    flat_chunks = []
    for start in range(0, n_rows, chunk_size):
        end = min(start + chunk_size, n_rows)
        rows_in_chunk = end - start
        rand_matrix = np.random.rand(rows_in_chunk, pool_size)
        if exclude_self:
            rand_matrix[np.arange(rows_in_chunk), np.arange(start, end)] = np.inf
        ranked_idx = np.argsort(rand_matrix, axis=1)
        col_rank = np.arange(pool_size)[None, :]
        mask = col_rank < counts[start:end, None]
        selected_idx = ranked_idx[mask]
        flat_chunks.append(pool[selected_idx])
    flat_values = np.concatenate(flat_chunks)
    return flat_values, counts


def generate_merchants(n_merchants: int = N_MERCHANTS, seed: int = SEED) -> pd.DataFrame:
    """Generate the shared merchant pool as a plain DataFrame (no dedicated
    Pydantic schema exists for merchants per §6/§8 — approved design decision).
    """
    np.random.seed(seed)
    faker = Faker(locale="en_IN")
    Faker.seed(seed)
    merchant_id = np.array([f"MERCH-{i:04d}" for i in range(n_merchants)])
    merchant_category = np.random.choice(MERCHANT_CATEGORIES, size=n_merchants)
    region = np.array([faker.city() for _ in range(n_merchants)])
    return pd.DataFrame(
        {"merchant_id": merchant_id, "merchant_category": merchant_category, "region": region}
    )


def assign_income_tiers(n_customers: int = N_CUSTOMERS, seed: int = SEED) -> np.ndarray:
    """Assign each customer an income tier per the approved 60/30/10 split."""
    np.random.seed(seed)
    tiers = list(INCOME_TIERS.keys())
    probs = [INCOME_TIERS[t]["share"] for t in tiers]
    return np.random.choice(tiers, size=n_customers, p=probs)


def sample_mean_spend_and_variance(
    income_tiers: np.ndarray, seed: int = SEED
) -> Tuple[np.ndarray, np.ndarray]:
    """Vectorized per-customer mean_spend (log-normal around the tier median)
    and spend_variance (derived via a fixed coefficient of variation).
    """
    np.random.seed(seed)
    n = income_tiers.shape[0]
    median_spend = np.empty(n, dtype=float)
    for tier, cfg in INCOME_TIERS.items():
        median_spend[income_tiers == tier] = cfg["median_spend"]
    customer_dispersion_sigma = 0.4
    mu = np.log(median_spend)
    mean_spend = np.random.lognormal(mean=mu, sigma=customer_dispersion_sigma, size=n)
    spend_variance = (mean_spend * TRANSACTION_CV) ** 2
    return mean_spend, spend_variance


def assign_primary_devices(
    n_customers: int = N_CUSTOMERS, seed: int = SEED
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate 1-2 synthetic primary device_ids per customer.

    Returns (flat_device_ids, counts) in row-major ragged layout.
    """
    np.random.seed(seed)
    counts = np.random.randint(1, 3, size=n_customers)
    customer_idx_repeated = np.repeat(np.arange(n_customers), counts)
    position_within = _ragged_row_positions(counts)
    flat_device_ids = np.array(
        [f"DEV-{c:06d}-{p}" for c, p in zip(customer_idx_repeated, position_within)]
    )
    return flat_device_ids, counts


def _ragged_to_list_column(flat_values: np.ndarray, counts: np.ndarray) -> List[List[str]]:
    """Split a row-major flattened ragged array into a list-of-lists column."""
    split_points = np.cumsum(counts)[:-1]
    return [arr.tolist() for arr in np.split(flat_values, split_points)]


def generate_customer_profiles(
    n_customers: int = N_CUSTOMERS,
    merchants: pd.DataFrame = None,
    seed: int = SEED,
) -> pd.DataFrame:
    """Generate the CustomerProfile-shaped DataFrame for all customers."""
    np.random.seed(seed)
    faker = Faker(locale="en_IN")
    Faker.seed(seed)

    if merchants is None:
        merchants = generate_merchants(seed=seed)

    customer_id = np.array([f"CUST-{i:06d}" for i in range(n_customers)])
    base_location = np.array([faker.city() for _ in range(n_customers)])

    income_tiers = assign_income_tiers(n_customers, seed=seed)
    mean_spend, spend_variance = sample_mean_spend_and_variance(income_tiers, seed=seed + 1)

    device_flat, device_counts = assign_primary_devices(n_customers, seed=seed + 2)
    primary_devices = _ragged_to_list_column(device_flat, device_counts)

    np.random.seed(seed + 3)
    merchant_counts = np.random.randint(
        MERCHANTS_PER_CUSTOMER_RANGE[0], MERCHANTS_PER_CUSTOMER_RANGE[1] + 1, size=n_customers
    )
    merchant_flat, _ = _sample_unique_from_pool(
        merchants["merchant_id"].to_numpy(), merchant_counts, seed=seed + 4
    )
    usual_merchants = _ragged_to_list_column(merchant_flat, merchant_counts)

    np.random.seed(seed + 5)
    beneficiary_counts = np.random.randint(
        BENEFICIARIES_PER_CUSTOMER_RANGE[0], BENEFICIARIES_PER_CUSTOMER_RANGE[1] + 1,
        size=n_customers,
    )
    beneficiary_flat, _ = _sample_unique_from_pool(
        customer_id, beneficiary_counts, seed=seed + 6, exclude_self=True
    )
    usual_beneficiaries = _ragged_to_list_column(beneficiary_flat, beneficiary_counts)

    return pd.DataFrame(
        {
            "customer_id": customer_id,
            "base_location": base_location,
            "primary_devices": primary_devices,
            "mean_spend": mean_spend,
            "spend_variance": spend_variance,
            "usual_merchants": usual_merchants,
            "usual_beneficiaries": usual_beneficiaries,
            "_income_tier": income_tiers,  # internal only, dropped before schema validation
        }
    )


def sample_diurnal_timestamps(n: int, days: int = SIMULATION_DAYS, seed: int = SEED) -> pd.Series:
    """Vectorized timestamp sampling: uniform day-in-window, diurnal hour-of-day."""
    np.random.seed(seed)
    hour_weights = np.array(
        [DIURNAL_DAY_WEIGHT if h in DIURNAL_DAY_HOURS else DIURNAL_NIGHT_WEIGHT for h in range(24)]
    )
    hour_probs = hour_weights / hour_weights.sum()

    day_offset = np.random.randint(0, days, size=n)
    hour = np.random.choice(24, size=n, p=hour_probs)
    minute = np.random.randint(0, 60, size=n)
    second = np.random.randint(0, 60, size=n)

    start = pd.Timestamp(SIMULATION_START_DATE)
    return (
        start
        + pd.to_timedelta(day_offset, unit="D")
        + pd.to_timedelta(hour, unit="h")
        + pd.to_timedelta(minute, unit="m")
        + pd.to_timedelta(second, unit="s")
    )


def sample_transaction_amounts(
    mean_spend: np.ndarray, spend_variance: np.ndarray, seed: int = SEED
) -> np.ndarray:
    """Vectorized log-normal amount draw per transaction from its owning
    customer's (mean_spend, spend_variance) via standard moment matching:
    sigma^2 = ln(1 + V/M^2), mu = ln(M) - sigma^2/2.
    """
    np.random.seed(seed)
    m2 = mean_spend**2
    sigma2 = np.log1p(spend_variance / m2)
    sigma = np.sqrt(sigma2)
    mu = np.log(mean_spend) - sigma2 / 2
    return np.random.lognormal(mean=mu, sigma=sigma)


def assign_persistent_entity(
    customer_idx_for_tx: np.ndarray,
    flat_pool: np.ndarray,
    pool_counts: np.ndarray,
    fallback_pool: np.ndarray,
    persistence_prob: float = ENTITY_PERSISTENCE_PROB,
    seed: int = SEED,
) -> np.ndarray:
    """Per-transaction entity assignment (device/merchant/beneficiary):
    with probability > persistence_prob draw from the owning customer's
    ragged `flat_pool` segment, else draw a novel value from `fallback_pool`.
    """
    np.random.seed(seed)
    n = customer_idx_for_tx.shape[0]
    offsets = np.cumsum(pool_counts) - pool_counts
    counts_for_tx = pool_counts[customer_idx_for_tx]
    offsets_for_tx = offsets[customer_idx_for_tx]

    within_segment = np.random.randint(0, counts_for_tx)
    persistent_value = flat_pool[offsets_for_tx + within_segment]
    novel_value = np.random.choice(fallback_pool, size=n)

    use_persistent = np.random.random(n) < persistence_prob
    return np.where(use_persistent, persistent_value, novel_value)


def generate_transaction_base(
    customers: pd.DataFrame,
    merchants: pd.DataFrame,
    n_transactions: int = N_TRANSACTIONS,
    days: int = SIMULATION_DAYS,
    seed: int = SEED,
) -> pd.DataFrame:
    """Generate the TransactionBase-shaped DataFrame for the clean payment twin."""
    np.random.seed(seed)
    n_customers = len(customers)

    np.random.seed(seed)
    customer_idx_for_tx = np.random.randint(0, n_customers, size=n_transactions)
    customer_id_for_tx = customers["customer_id"].to_numpy()[customer_idx_for_tx]

    timestamps = sample_diurnal_timestamps(n_transactions, days=days, seed=seed + 1)

    mean_spend_for_tx = customers["mean_spend"].to_numpy()[customer_idx_for_tx]
    spend_variance_for_tx = customers["spend_variance"].to_numpy()[customer_idx_for_tx]
    amounts = sample_transaction_amounts(mean_spend_for_tx, spend_variance_for_tx, seed=seed + 2)

    device_counts = customers["primary_devices"].apply(len).to_numpy()
    device_flat = np.array([d for row in customers["primary_devices"] for d in row])
    device_id = assign_persistent_entity(
        customer_idx_for_tx, device_flat, device_counts, device_flat, seed=seed + 3
    )

    merchant_counts = customers["usual_merchants"].apply(len).to_numpy()
    merchant_flat = np.array([m for row in customers["usual_merchants"] for m in row])
    merchant_id = assign_persistent_entity(
        customer_idx_for_tx,
        merchant_flat,
        merchant_counts,
        merchants["merchant_id"].to_numpy(),
        seed=seed + 4,
    )
    merchant_category_map = merchants.set_index("merchant_id")["merchant_category"]
    merchant_category = merchant_category_map.reindex(merchant_id).to_numpy()

    beneficiary_counts = customers["usual_beneficiaries"].apply(len).to_numpy()
    beneficiary_flat = np.array([b for row in customers["usual_beneficiaries"] for b in row])
    beneficiary_id = assign_persistent_entity(
        customer_idx_for_tx,
        beneficiary_flat,
        beneficiary_counts,
        customer_id_for_tx,
        seed=seed + 5,
    )
    # A beneficiary must not be the payer themself; resample the rare collisions.
    self_pay_mask = beneficiary_id == customer_id_for_tx
    if self_pay_mask.any():
        np.random.seed(seed + 6)
        replacement = np.random.choice(customer_id_for_tx, size=self_pay_mask.sum())
        beneficiary_id = beneficiary_id.copy()
        beneficiary_id[self_pay_mask] = replacement

    np.random.seed(seed + 7)
    channel = np.random.choice(CHANNELS, size=n_transactions, p=CHANNEL_PROBS)

    base_location_for_tx = customers["base_location"].to_numpy()[customer_idx_for_tx]
    novel_location_mask = np.random.random(n_transactions) < NOVEL_LOCATION_PROB
    faker = Faker(locale="en_IN")
    Faker.seed(seed + 7)
    novel_locations = np.array([faker.city() for _ in range(novel_location_mask.sum())])
    location = base_location_for_tx.copy()
    location[novel_location_mask] = novel_locations
    ip_region = location.copy()

    transaction_id = np.array([f"TXN-{i:08d}" for i in range(n_transactions)])

    return pd.DataFrame(
        {
            "transaction_id": transaction_id,
            "timestamp": timestamps,
            "customer_id": customer_id_for_tx,
            "merchant_id": merchant_id,
            "beneficiary_id": beneficiary_id,
            "amount": amounts,
            "currency": "INR",
            "channel": channel,
            "device_id": device_id,
            "ip_region": ip_region,
            "location": location,
            "merchant_category": merchant_category,
            "semantic_risk_score": 0.0,
            "voice_confidence_score": 1.0,
        }
    )


def validate_customers(df: pd.DataFrame) -> List[CustomerProfile]:
    """Bulk Pydantic-boundary validation of generated customer rows."""
    records = df.drop(columns=["_income_tier"], errors="ignore").to_dict(orient="records")
    return TypeAdapter(List[CustomerProfile]).validate_python(records)


def validate_transactions(df: pd.DataFrame) -> List[TransactionBase]:
    """Bulk Pydantic-boundary validation of generated transaction rows."""
    records = df.to_dict(orient="records")
    return TypeAdapter(List[TransactionBase]).validate_python(records)


def simulate_payment_twin(
    n_customers: int = N_CUSTOMERS,
    n_merchants: int = N_MERCHANTS,
    n_transactions: int = N_TRANSACTIONS,
    days: int = SIMULATION_DAYS,
    seed: int = SEED,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Top-level entry point: generate the full clean payment twin."""
    merchants = generate_merchants(n_merchants, seed=seed)
    customers = generate_customer_profiles(n_customers, merchants, seed=seed)
    transactions = generate_transaction_base(customers, merchants, n_transactions, days, seed=seed)
    return customers, transactions
