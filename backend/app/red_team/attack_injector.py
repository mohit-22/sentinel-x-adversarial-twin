"""Genome -> injected transactions, for micro_structuring (ATK-MS-001) and
synthetic_identity_drift (ATK-ID-001).

A deterministic Python simulator converts each genome into actual
transaction rows (CLAUDE.md §3 design rule: no LLM ever scores or decides
here). Reuses clean_generator.py's customer/merchant generation and its
per-customer amount-sampling function directly rather than reimplementing
spending-pattern logic.
"""

from typing import Dict, List

import numpy as np
import pandas as pd
from pydantic import TypeAdapter

from app.core.config import CHANNELS, CHANNEL_PROBS, SEED, SIMULATION_DAYS, SIMULATION_START_DATE
from app.core.schemas import InjectedTransaction
from app.simulator.clean_generator import sample_transaction_amounts

FRAUD_MERCHANT_ID = "P2P-TRANSFER"
FRAUD_MERCHANT_CATEGORY = "p2p_transfer"


def reconcile_split_amounts(
    raw_amounts: np.ndarray,
    target_amount: float,
    amount_range: tuple,
    tolerance: float = 0.02,
    max_iters: int = 10,
) -> np.ndarray:
    """Rescale raw per-tx draws to sum to target_amount, iteratively clipping
    to amount_range and redistributing the residual across unclipped entries.

    Not always achievable within `tolerance` when split_count * range_high <
    target_amount * (1 - tolerance) — a structural constraint of the genome's
    own numbers, not a bug. Callers should check the achieved sum.
    """
    low, high = amount_range
    amounts = raw_amounts.astype(float).copy()

    for _ in range(max_iters):
        current_sum = amounts.sum()
        if current_sum <= 0:
            break
        amounts = amounts * (target_amount / current_sum)
        amounts = np.clip(amounts, low, high)

        deficit = target_amount - amounts.sum()
        if abs(deficit) <= tolerance * target_amount:
            break

        free_mask = ~np.isclose(amounts, high) if deficit > 0 else ~np.isclose(amounts, low)
        if not free_mask.any():
            break
        amounts[free_mask] += deficit / free_mask.sum()
        amounts = np.clip(amounts, low, high)

    return amounts


def generate_mule_beneficiaries(instance_id: int, recipient_count: int) -> np.ndarray:
    """Brand-new synthetic mule beneficiary IDs, never drawn from any customer pool."""
    return np.array([f"MULE-{instance_id:04d}-{k}" for k in range(recipient_count)])


def sample_exponential_arrivals(
    split_count: int, window_hours: float, seed: int
) -> np.ndarray:
    """Exponential inter-arrival offsets (hours from window start), rescaled
    to fit inside window_hours if the raw cumulative sum overflows it.
    """
    np.random.seed(seed)
    n_gaps = max(split_count - 1, 1)
    mean_gap = window_hours / n_gaps
    gaps = np.random.exponential(scale=mean_gap, size=split_count - 1)
    offsets = np.concatenate([[0.0], np.cumsum(gaps)])
    if offsets[-1] > window_hours:
        offsets = offsets * (window_hours * 0.99 / offsets[-1])
    return offsets


def generate_noise_transactions(
    customer: pd.Series,
    merchants: pd.DataFrame,
    n_noise: int,
    window_start: pd.Timestamp,
    window_hours: float,
    instance_id: int,
    seed: int,
) -> pd.DataFrame:
    """Legitimate-looking noise transactions reusing the customer's normal
    spending pattern (persistence over their own usual merchants/devices/
    beneficiaries, amounts via clean_generator's own lognormal sampler).
    """
    if n_noise == 0:
        return pd.DataFrame(columns=list(InjectedTransaction.model_fields.keys()))

    np.random.seed(seed)
    offsets_hours = np.random.uniform(0, window_hours, size=n_noise)
    timestamps = window_start + pd.to_timedelta(offsets_hours, unit="h")

    mean_spend = np.full(n_noise, customer["mean_spend"])
    spend_variance = np.full(n_noise, customer["spend_variance"])
    amounts = sample_transaction_amounts(mean_spend, spend_variance, seed=seed)

    usual_merchants = np.array(customer["usual_merchants"])
    merchant_id = np.random.choice(usual_merchants, size=n_noise)
    merchant_category_map = merchants.set_index("merchant_id")["merchant_category"]
    merchant_category = merchant_category_map.reindex(merchant_id).to_numpy()

    primary_devices = np.array(customer["primary_devices"])
    device_id = np.random.choice(primary_devices, size=n_noise)

    usual_beneficiaries = np.array(customer["usual_beneficiaries"])
    beneficiary_id = np.random.choice(usual_beneficiaries, size=n_noise)

    channel = np.random.choice(CHANNELS, size=n_noise, p=CHANNEL_PROBS)
    location = np.full(n_noise, customer["base_location"])

    return pd.DataFrame(
        {
            "transaction_id": [f"ATKTXN-{instance_id:04d}-N{k:03d}" for k in range(n_noise)],
            "timestamp": timestamps,
            "customer_id": customer["customer_id"],
            "merchant_id": merchant_id,
            "beneficiary_id": beneficiary_id,
            "amount": amounts,
            "currency": "INR",
            "channel": channel,
            "device_id": device_id,
            "ip_region": location,
            "location": location,
            "merchant_category": merchant_category,
            "semantic_risk_score": 0.0,
            "voice_confidence_score": 1.0,
            "is_fraud": 0,
            "attack_family": None,
            "genome_id": None,
        }
    )


def generate_micro_structuring_instance(
    genome: Dict,
    customer: pd.Series,
    merchants: pd.DataFrame,
    instance_id: int,
    start_time: pd.Timestamp,
    seed: int,
) -> pd.DataFrame:
    """Generate fraud + interleaved noise rows for one attack instance
    targeting one customer, per the ATK-MS-001 genome.
    """
    np.random.seed(seed)
    params = genome["parameters"]
    camo = genome["behavioral_camouflage"]
    low, high = params["amount_per_tx_range"]

    split_count = int(
        np.random.randint(params["split_count_range"][0], params["split_count_range"][1] + 1)
    )
    raw_amounts = np.random.uniform(low, high, size=split_count)
    amounts = reconcile_split_amounts(raw_amounts, genome["target_amount"], (low, high))

    mule_ids = generate_mule_beneficiaries(instance_id, params["recipient_count"])
    beneficiary_id = np.random.choice(mule_ids, size=split_count)

    offsets_hours = sample_exponential_arrivals(split_count, params["time_window_hours"], seed=seed)
    timestamps = start_time + pd.to_timedelta(offsets_hours, unit="h")

    primary_devices = np.array(customer["primary_devices"])
    device_id = np.random.choice(primary_devices, size=split_count)
    location = np.full(split_count, customer["base_location"])

    fraud_df = pd.DataFrame(
        {
            "transaction_id": [f"ATKTXN-{instance_id:04d}-F{k:02d}" for k in range(split_count)],
            "timestamp": timestamps,
            "customer_id": customer["customer_id"],
            "merchant_id": FRAUD_MERCHANT_ID,
            "beneficiary_id": beneficiary_id,
            "amount": amounts,
            "currency": "INR",
            "channel": "P2P",
            "device_id": device_id,
            "ip_region": location,
            "location": location,
            "merchant_category": FRAUD_MERCHANT_CATEGORY,
            "semantic_risk_score": 0.0,
            "voice_confidence_score": 1.0,
            "is_fraud": 1,
            "attack_family": genome["family"],
            "genome_id": genome["genome_id"],
        }
    )

    if camo.get("interleave_legitimate_noise"):
        noise_ratio = camo["noise_ratio"]
        # Interpretation (i): noise is a share of the combined interleaved stream.
        noise_count = int(round(noise_ratio / (1 - noise_ratio) * split_count))
        noise_df = generate_noise_transactions(
            customer, merchants, noise_count, start_time, params["time_window_hours"],
            instance_id, seed=seed + 1,
        )
    else:
        noise_df = pd.DataFrame(columns=fraud_df.columns)

    combined = pd.concat([fraud_df, noise_df], ignore_index=True)
    combined["instance_id"] = instance_id
    return combined.sort_values("timestamp").reset_index(drop=True)


def generate_micro_structuring_attacks(
    genome: Dict,
    customers: pd.DataFrame,
    merchants: pd.DataFrame,
    n_instances: int,
    seed: int = SEED,
) -> pd.DataFrame:
    """Top-level entry point: inject n_instances micro_structuring attacks,
    one per distinct randomly sampled customer.
    """
    np.random.seed(seed)
    window_days = genome["parameters"]["time_window_hours"] / 24.0
    max_start_offset_days = max(SIMULATION_DAYS - window_days, 0.0)

    customer_positions = np.random.choice(len(customers), size=n_instances, replace=False)
    start_offsets = np.random.uniform(0, max_start_offset_days, size=n_instances)
    base = pd.Timestamp(SIMULATION_START_DATE)

    instances: List[pd.DataFrame] = []
    for i, (pos, offset) in enumerate(zip(customer_positions, start_offsets)):
        customer = customers.iloc[pos]
        start_time = base + pd.to_timedelta(offset, unit="D")
        instance_df = generate_micro_structuring_instance(
            genome, customer, merchants, instance_id=i, start_time=start_time, seed=seed + i + 1
        )
        instances.append(instance_df)

    return pd.concat(instances, ignore_index=True)


def generate_identity_drift_instance(
    genome: Dict,
    customer: pd.Series,
    merchants: pd.DataFrame,
    instance_id: int,
    start_time: pd.Timestamp,
    seed: int,
) -> pd.DataFrame:
    """Generate the extraction-burst rows for one synthetic_identity_drift
    instance targeting one customer, per the ATK-ID-001 genome.

    `start_time` is the extraction burst's own start (i.e. already offset by
    the drift window) -- no rows are generated for the drift/trust-building
    period itself. That period is real: the customer's actual Day 1-2 clean
    transaction history structurally plays that role (approved design
    decision F), so this function only ever emits is_fraud=1 rows.
    """
    np.random.seed(seed)
    params = genome["parameters"]

    extraction_count = int(
        np.random.randint(
            params["extraction_transaction_count_range"][0],
            params["extraction_transaction_count_range"][1] + 1,
        )
    )
    window_hours = params["extraction_window_hours"]
    offsets_hours = np.sort(np.random.uniform(0, window_hours, size=extraction_count))
    timestamps = start_time + pd.to_timedelta(offsets_hours, unit="h")

    multiplier = np.random.uniform(*params["extraction_amount_multiplier_range"])
    total_extraction_amount = customer["mean_spend"] * multiplier
    # Moderate concentration (3.0): random but not wildly skewed per-tx splits.
    amounts = np.random.dirichlet(np.full(extraction_count, 3.0)) * total_extraction_amount

    device_id = f"DRIFT-DEV-{instance_id:04d}-0"
    beneficiary_id = f"DRIFT-PAYEE-{instance_id:04d}-0"
    location = np.full(extraction_count, customer["base_location"])

    fraud_df = pd.DataFrame(
        {
            "transaction_id": [f"ATKTXN-ID-{instance_id:04d}-E{k:02d}" for k in range(extraction_count)],
            "timestamp": timestamps,
            "customer_id": customer["customer_id"],
            "merchant_id": FRAUD_MERCHANT_ID,
            "beneficiary_id": beneficiary_id,
            "amount": amounts,
            "currency": "INR",
            "channel": "P2P",
            "device_id": device_id,
            "ip_region": location,
            "location": location,
            "merchant_category": FRAUD_MERCHANT_CATEGORY,
            "semantic_risk_score": 0.0,
            "voice_confidence_score": 1.0,
            "is_fraud": 1,
            "attack_family": genome["family"],
            "genome_id": genome["genome_id"],
        }
    )
    return fraud_df.sort_values("timestamp").reset_index(drop=True)


def generate_identity_drift_attacks(
    genome: Dict,
    customers: pd.DataFrame,
    merchants: pd.DataFrame,
    n_instances: int,
    seed: int = SEED,
) -> pd.DataFrame:
    """Top-level entry point: inject n_instances synthetic_identity_drift
    attacks, one per distinct randomly sampled customer.

    Unlike generate_micro_structuring_attacks, the extraction burst is NOT
    placed at a random offset across the full 30-day simulation -- it's
    anchored to drift_window_days after SIMULATION_START_DATE (the genome's
    own definition: ~20 days of real trust-building history, then a sudden
    extraction). drift_window_days_range's upper bound (22) plus
    extraction_window_hours (a few hours) always comfortably fits inside
    SIMULATION_DAYS (30), so no capping is needed here the way family #1
    needed max_start_offset_days.
    """
    np.random.seed(seed)
    params = genome["parameters"]

    customer_positions = np.random.choice(len(customers), size=n_instances, replace=False)
    drift_days = np.random.uniform(
        params["drift_window_days_range"][0], params["drift_window_days_range"][1], size=n_instances
    )
    base = pd.Timestamp(SIMULATION_START_DATE)

    instances: List[pd.DataFrame] = []
    for i, (pos, drift) in enumerate(zip(customer_positions, drift_days)):
        customer = customers.iloc[pos]
        start_time = base + pd.to_timedelta(drift, unit="D")
        instance_df = generate_identity_drift_instance(
            genome, customer, merchants, instance_id=i, start_time=start_time, seed=seed + i + 1
        )
        instance_df["instance_id"] = i
        instance_df["drift_window_days"] = drift  # diagnostic only, dropped before schema validation
        instances.append(instance_df)

    return pd.concat(instances, ignore_index=True)


# --- behavioral_camouflage (ATK-BC-001) -------------------------------------
#
# PREDICTION (written here for comparison once this runs through arena.py --
# not done today; recorded before running to keep the comparison honest):
# this is expected to be the hardest-to-detect family built so far. Unlike
# micro_structuring/identity_drift, fraud legs here deliberately carry NONE
# of the tells the model keys on: is_new_device=0 (reuse_customer_device),
# real merchant/channel instead of the P2P-TRANSFER placeholder
# (use_real_merchant_and_channel -- reusing that placeholder here would
# just recreate the Day 4 artifact-tell problem under a different feature),
# and amounts drawn from the customer's own normal distribution rather than
# a distinct fixed band. What's left for M0 to catch this on is narrow:
# amount_deviation_ratio (if the burst even slightly exceeds the customer's
# baseline) and velocity features (the burst's aggregate frequency, even if
# every individual leg looks clean). Because there is no excluded-feature
# "free pass" left to exploit here (unlike rotate_mule_accounts/
# add_legitimate_micro_purchases for family #1, which attacked features
# that were never model inputs), all three of this family's mutations
# (reduce_burst_density, increase_camouflage_ratio,
# match_customer_amount_profile) are expected to have a REAL effect on
# evasion, not the mostly-no-op pattern seen for family #1.


def generate_camouflage_fraud_transactions(
    customer: pd.Series,
    merchants: pd.DataFrame,
    n_fraud: int,
    window_start: pd.Timestamp,
    window_hours: float,
    instance_id: int,
    genome: Dict,
    seed: int,
) -> pd.DataFrame:
    """Fraud legs for one behavioral_camouflage instance -- deliberately a
    near-identical sibling of generate_noise_transactions (same device,
    same merchant pool, same channel mix, same amount-sampling mechanism as
    genuine spending) so the ONLY things that distinguish these rows from
    real noise are the is_fraud/attack_family/genome_id labels and the
    destination beneficiary. That's the entire point of "camouflage" (see
    prediction above): reusing clean_generator's own amount sampler here
    rather than a distinct range is intentional, not an oversight.
    """
    if n_fraud == 0:
        return pd.DataFrame(columns=list(InjectedTransaction.model_fields.keys()))

    np.random.seed(seed)
    offsets_hours = np.random.uniform(0, window_hours, size=n_fraud)
    timestamps = window_start + pd.to_timedelta(offsets_hours, unit="h")

    mean_spend = np.full(n_fraud, customer["mean_spend"])
    spend_variance = np.full(n_fraud, customer["spend_variance"])
    amounts = sample_transaction_amounts(mean_spend, spend_variance, seed=seed)

    usual_merchants = np.array(customer["usual_merchants"])
    merchant_id = np.random.choice(usual_merchants, size=n_fraud)
    merchant_category_map = merchants.set_index("merchant_id")["merchant_category"]
    merchant_category = merchant_category_map.reindex(merchant_id).to_numpy()

    primary_devices = np.array(customer["primary_devices"])
    device_id = np.random.choice(primary_devices, size=n_fraud)

    beneficiary_id = f"CAMO-PAYEE-{instance_id:04d}-0"
    channel = np.random.choice(CHANNELS, size=n_fraud, p=CHANNEL_PROBS)
    location = np.full(n_fraud, customer["base_location"])

    return pd.DataFrame(
        {
            "transaction_id": [f"ATKTXN-BC-{instance_id:04d}-C{k:02d}" for k in range(n_fraud)],
            "timestamp": timestamps,
            "customer_id": customer["customer_id"],
            "merchant_id": merchant_id,
            "beneficiary_id": beneficiary_id,
            "amount": amounts,
            "currency": "INR",
            "channel": channel,
            "device_id": device_id,
            "ip_region": location,
            "location": location,
            "merchant_category": merchant_category,
            "semantic_risk_score": 0.0,
            "voice_confidence_score": 1.0,
            "is_fraud": 1,
            "attack_family": genome["family"],
            "genome_id": genome["genome_id"],
        }
    )


def generate_behavioral_camouflage_instance(
    genome: Dict,
    customer: pd.Series,
    merchants: pd.DataFrame,
    instance_id: int,
    start_time: pd.Timestamp,
    seed: int,
) -> pd.DataFrame:
    """Generate one behavioral_camouflage instance: a burst of mostly
    genuine-looking transactions (generate_noise_transactions, reused
    unchanged from Day 3) with a minority of camouflaged fraud legs
    interleaved (generate_camouflage_fraud_transactions above), per the
    ATK-BC-001 genome.
    """
    np.random.seed(seed)
    params = genome["parameters"]

    burst_count = int(
        np.random.randint(
            params["burst_transaction_count_range"][0], params["burst_transaction_count_range"][1] + 1
        )
    )
    fraud_count = int(round(params["fraud_leg_ratio"] * burst_count))
    legit_count = burst_count - fraud_count
    window_hours = params["burst_window_hours"]

    legit_df = generate_noise_transactions(
        customer, merchants, legit_count, start_time, window_hours, instance_id, seed=seed
    )
    fraud_df = generate_camouflage_fraud_transactions(
        customer, merchants, fraud_count, start_time, window_hours, instance_id, genome, seed=seed + 1
    )

    combined = pd.concat([legit_df, fraud_df], ignore_index=True)
    combined["instance_id"] = instance_id
    return combined.sort_values("timestamp").reset_index(drop=True)


def generate_behavioral_camouflage_attacks(
    genome: Dict,
    customers: pd.DataFrame,
    merchants: pd.DataFrame,
    n_instances: int,
    seed: int = SEED,
) -> pd.DataFrame:
    """Top-level entry point: inject n_instances behavioral_camouflage
    attacks, one per distinct randomly sampled customer. Burst is placed at
    a random offset across the full 30-day simulation (like family #1, not
    anchored like family #2's drift window).
    """
    np.random.seed(seed)
    window_days = genome["parameters"]["burst_window_hours"] / 24.0
    max_start_offset_days = max(SIMULATION_DAYS - window_days, 0.0)

    customer_positions = np.random.choice(len(customers), size=n_instances, replace=False)
    start_offsets = np.random.uniform(0, max_start_offset_days, size=n_instances)
    base = pd.Timestamp(SIMULATION_START_DATE)

    instances: List[pd.DataFrame] = []
    for i, (pos, offset) in enumerate(zip(customer_positions, start_offsets)):
        customer = customers.iloc[pos]
        start_time = base + pd.to_timedelta(offset, unit="D")
        instance_df = generate_behavioral_camouflage_instance(
            genome, customer, merchants, instance_id=i, start_time=start_time, seed=seed + i + 1
        )
        instances.append(instance_df)

    return pd.concat(instances, ignore_index=True)


def validate_injected_transactions(df: pd.DataFrame) -> List[InjectedTransaction]:
    """Bulk Pydantic-boundary validation of injected transaction rows."""
    records = df.drop(columns=["instance_id", "drift_window_days"], errors="ignore").to_dict(orient="records")
    return TypeAdapter(List[InjectedTransaction]).validate_python(records)
