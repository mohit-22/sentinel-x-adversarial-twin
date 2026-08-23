"""Fidelity scoring for the synthetic payment twin (PRD §7.1).

Fidelity is measured against the *intended* statistical distributions the
generator targets (KS statistic on amounts, Jensen-Shannon divergence on
temporal hour-of-day) — there is no real cardholder data to compare against
per CLAUDE.md's hard data rule, so "reference" means the theoretical target
distribution, not an external dataset.

similarity = 1 - KS_statistic (amounts)
similarity = 1 - JS_divergence (temporal), approved formulas.
"""

from typing import Dict

import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon
from scipy.stats import ks_2samp

from app.core.config import (
    DIURNAL_DAY_HOURS,
    DIURNAL_DAY_WEIGHT,
    DIURNAL_NIGHT_WEIGHT,
    SEED,
)
from app.simulator.clean_generator import sample_transaction_amounts


def ks_amount_fidelity(observed: np.ndarray, reference: np.ndarray) -> float:
    """Similarity (1 - KS statistic) between observed and reference amount samples."""
    statistic, _ = ks_2samp(observed, reference)
    return 1.0 - statistic


def js_temporal_fidelity(
    observed_hours: np.ndarray, reference_probs: np.ndarray, n_bins: int = 24
) -> float:
    """Similarity (1 - JS divergence) between observed hour-of-day histogram
    and the reference (intended) diurnal probability distribution.
    """
    counts = np.bincount(observed_hours, minlength=n_bins).astype(float)
    observed_probs = counts / counts.sum()
    distance = jensenshannon(observed_probs, reference_probs, base=2)
    divergence = distance**2
    return 1.0 - divergence


def _intended_diurnal_probs(n_bins: int = 24) -> np.ndarray:
    weights = np.array(
        [DIURNAL_DAY_WEIGHT if h in DIURNAL_DAY_HOURS else DIURNAL_NIGHT_WEIGHT for h in range(n_bins)]
    )
    return weights / weights.sum()


def compute_fidelity_report(
    customers: pd.DataFrame, transactions: pd.DataFrame, seed: int = SEED
) -> Dict[str, float]:
    """Compute amount (KS) and temporal (JS) fidelity similarity scores,
    overall and per income tier.
    """
    report: Dict[str, float] = {}

    customer_tier = customers.set_index("customer_id")["_income_tier"]
    tier_for_tx = transactions["customer_id"].map(customer_tier)

    mean_spend_map = customers.set_index("customer_id")["mean_spend"]
    variance_map = customers.set_index("customer_id")["spend_variance"]
    mean_spend_for_tx = transactions["customer_id"].map(mean_spend_map).to_numpy()
    variance_for_tx = transactions["customer_id"].map(variance_map).to_numpy()
    reference_amounts = sample_transaction_amounts(mean_spend_for_tx, variance_for_tx, seed=seed + 100)

    report["amount_similarity_overall"] = ks_amount_fidelity(
        transactions["amount"].to_numpy(), reference_amounts
    )
    for tier in customer_tier.unique():
        mask = (tier_for_tx == tier).to_numpy()
        report[f"amount_similarity_{tier}"] = ks_amount_fidelity(
            transactions.loc[mask, "amount"].to_numpy(), reference_amounts[mask]
        )

    reference_probs = _intended_diurnal_probs()
    observed_hours = transactions["timestamp"].dt.hour.to_numpy()
    report["temporal_similarity_overall"] = js_temporal_fidelity(observed_hours, reference_probs)

    return report
