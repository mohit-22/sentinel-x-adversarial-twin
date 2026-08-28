"""SHAP TreeExplainer reason codes for a single scored transaction
(CLAUDE.md §7, PRD_SENTINEL_X §7.3 -- exact reason_codes shape).

Honest scope, established at the Day 8a planning turn: SHAP needs the
exact engineered feature row that was fed to the model, and feature
engineering is context-dependent (velocity/graph features depend on the
customer's transaction history at scoring time) -- it cannot be
recomputed from a bare transaction_id. This module only explains
transactions already present in the startup pipeline's cached, already-
engineered train_df/test_df (verified disjoint, transaction_id unique in
each). Transactions generated fresh per-request elsewhere (e.g.
/payment-twin's counterfactual instances) are not covered -- that's a
disclosed limitation (PRD_SENTINEL_X §13.1), not a bug.
"""

from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import shap

from app.blue_team.detector import FEATURE_COLUMNS

TOP_N_REASON_CODES = 3

# CLAUDE.md §6 decision thresholds, exact values (same locked constants
# endpoints.py's own _DECISION_THRESHOLDS uses) -- not re-derived from
# config.py since it isn't in this module's scope, same precedent as
# endpoints.py's own local copy.
_ALLOW_THRESHOLD = 0.35
_STEP_UP_THRESHOLD = 0.65
_REVIEW_THRESHOLD = 0.85

_COUNTERFACTUAL_N_SAMPLES = 1000
_COUNTERFACTUAL_SEED = 42
# "Small" gaussian noise per feature is scaled to that feature's own real
# spread in train_df (10% of its std) -- so "small" means something sensible
# per feature (amount's noise step isn't the same magnitude as a 0/1 flag's).
_NOISE_STD_FRACTION = 0.10
# Only report a feature as "changed" if the shift is at least half a noise
# step -- otherwise nearly every one of the 14 features would show up with
# negligible residual jitter, which isn't an honest "what actually needs to
# change" answer.
_MEANINGFUL_CHANGE_FRACTION = 0.05

_CURRENCY_FEATURES = {"amount", "sum_5m", "sum_1h", "sum_24h", "sum_7d"}
_BINARY_FEATURES = {"is_new_device", "is_new_location"}
# Real-world non-negative quantities -- a noisy sample with a negative
# transaction count/amount is not a realizable counterfactual.
_NON_NEGATIVE_FEATURES = {f for f in FEATURE_COLUMNS if f != "amount_deviation_ratio"}


def _decision_from_score(score: float) -> str:
    if score < _ALLOW_THRESHOLD:
        return "ALLOW"
    if score < _STEP_UP_THRESHOLD:
        return "STEP_UP"
    if score < _REVIEW_THRESHOLD:
        return "REVIEW"
    return "BLOCK"


def _plain_english(feature: str, original: float, counterfactual: float) -> str:
    description = FEATURE_DESCRIPTIONS[feature]
    if feature in _BINARY_FEATURES:
        becomes_true = counterfactual > original
        return f"{description} would need to {'become true' if becomes_true else 'no longer be true'}"
    direction = "drop" if counterfactual < original else "increase"
    if feature in _CURRENCY_FEATURES:
        label = feature.replace("_", " ").capitalize()
        return f"{label} would need to {direction} from ₹{original:,.0f} to ₹{counterfactual:,.0f}"
    return f"{description} would need to {direction} from {original:.3f} to {counterfactual:.3f}"

# Grounded in each feature's actual meaning as implemented in
# features.py/graph_engine.py -- not invented. Keyed by the real
# FEATURE_COLUMNS names (not PRD §7.3's illustrative example names, which
# don't match the real column names 1:1).
FEATURE_DESCRIPTIONS: Dict[str, str] = {
    "amount": "Transaction amount",
    "count_5m": "Number of this customer's other transactions in the preceding 5 minutes",
    "count_1h": "Number of this customer's other transactions in the preceding 1 hour",
    "count_24h": "Number of this customer's other transactions in the preceding 24 hours",
    "count_7d": "Number of this customer's other transactions in the preceding 7 days",
    "sum_5m": "Total amount of this customer's other transactions in the preceding 5 minutes",
    "sum_1h": "Total amount of this customer's other transactions in the preceding 1 hour",
    "sum_24h": "Total amount of this customer's other transactions in the preceding 24 hours",
    "sum_7d": "Total amount of this customer's other transactions in the preceding 7 days",
    "amount_deviation_ratio": "Ratio of this transaction's amount to the customer's average amount over the preceding 7 days",
    "is_new_device": "Transaction initiated from a device not among the customer's known devices",
    "is_new_location": "Transaction location differs from the customer's base location",
    "shared_device_count": "Number of other distinct customers who have used this same device (train-graph)",
    "two_hop_fraud_risk": "Fraction of this customer's 2-hop network neighbors (shared beneficiary or device) with at least one fraud transaction in train",
}


def find_cached_feature_row(transaction_id: str, train_df: pd.DataFrame, test_df: pd.DataFrame) -> Optional[pd.Series]:
    """Look up a transaction's already-engineered feature row in the
    startup pipeline's cached train/test split. Returns None if not found
    -- the honest signal that this transaction isn't in today's SHAP scope.
    """
    for df in (test_df, train_df):
        match = df[df["transaction_id"] == transaction_id]
        if not match.empty:
            return match.iloc[0]
    return None


def compute_reason_codes(row: pd.Series, model) -> List[Dict[str, str]]:
    """Top-3 local SHAP feature attributions for one already-engineered
    feature row, per PRD_SENTINEL_X §7.3's exact shape. Reuses the
    cached, already-trained model -- no retraining.
    """
    x_row = row[FEATURE_COLUMNS].to_frame().T.astype(float)
    explainer = shap.TreeExplainer(model)
    # shap==0.52.0, verified empirically: for a binary LGBMClassifier this
    # returns a single (1, 14) ndarray of the positive-class contributions,
    # not a per-class list -- shap's own UserWarning notes this return
    # shape has changed across versions for binary LightGBM models.
    shap_values = explainer.shap_values(x_row)[0]

    contributions = list(zip(FEATURE_COLUMNS, shap_values))
    top = sorted(contributions, key=lambda pair: abs(pair[1]), reverse=True)[:TOP_N_REASON_CODES]

    return [
        {
            "feature": feature,
            "contribution": f"{value:+.2f}",
            "description": FEATURE_DESCRIPTIONS[feature],
        }
        for feature, value in top
    ]


def compute_counterfactual(
    row: pd.Series,
    model,
    feature_columns: List[str],
    train_df: pd.DataFrame,
    n_samples: int = _COUNTERFACTUAL_N_SAMPLES,
) -> Dict:
    """What is the smallest realistic change to this transaction's features
    that would have flipped the decision to ALLOW?

    Algorithm (as specified): take the flagged row's real feature vector,
    generate n_samples nearby points by adding small per-feature gaussian
    noise (scaled to that feature's real spread in train_df), score every
    sample with the real cached model, and report the closest sample (by
    std-normalized distance, so "amount" in the thousands doesn't dominate
    a 0/1 flag) that scores below the real ALLOW threshold. Honest "not
    found" response if none of the n_samples samples cross it -- never a
    fabricated counterfactual.
    """
    original_vector = row[feature_columns].astype(float).to_numpy()
    original_frame = row[feature_columns].to_frame().T.astype(float)
    original_proba = float(model.predict_proba(original_frame)[:, 1][0])
    original_decision = _decision_from_score(original_proba)

    # Robust (median absolute deviation, normal-consistent via the 1.4826
    # factor) rather than raw .std(): amount_deviation_ratio has a genuine
    # heavy tail (near-zero avg_amount_7d denominators produce occasional
    # enormous ratios -- ~32% of train_df exceeds 100, max ~7e10), so its
    # raw std is ~2.9 billion vs a MAD-based scale of ~1.3. Using raw std
    # there produced nonsensical multi-hundred-million-unit "counterfactual"
    # noise for that one feature; MAD keeps "small noise" small for every
    # feature, well-behaved or heavy-tailed alike.
    train_features = train_df[feature_columns]
    medians = train_features.median()
    mad = (train_features - medians).abs().median().to_numpy()
    feature_stds = np.where(mad == 0, 1e-6, mad * 1.4826)  # guard constant columns
    noise_scale = _NOISE_STD_FRACTION * feature_stds

    rng = np.random.default_rng(_COUNTERFACTUAL_SEED)
    noise = rng.normal(loc=0.0, scale=noise_scale, size=(n_samples, len(feature_columns)))
    samples = original_vector[None, :] + noise

    non_negative_cols = [i for i, f in enumerate(feature_columns) if f in _NON_NEGATIVE_FEATURES]
    samples[:, non_negative_cols] = np.clip(samples[:, non_negative_cols], 0, None)

    samples_df = pd.DataFrame(samples, columns=feature_columns)
    probas = model.predict_proba(samples_df)[:, 1]

    allow_mask = probas < _ALLOW_THRESHOLD
    if not allow_mask.any():
        return {
            "transaction_id": row["transaction_id"],
            "original_decision": original_decision,
            "counterfactual_decision": None,
            "changes_needed": [],
            "minimum_changes_required": None,
            "status": "no_nearby_allow_found",
        }

    allow_samples = samples[allow_mask]
    distances = np.linalg.norm((allow_samples - original_vector[None, :]) / feature_stds[None, :], axis=1)
    best_sample = allow_samples[np.argmin(distances)]

    changes = []
    for i, feature in enumerate(feature_columns):
        orig_val = float(original_vector[i])
        new_val = float(best_sample[i])
        if abs(new_val - orig_val) < _MEANINGFUL_CHANGE_FRACTION * feature_stds[i]:
            continue
        orig_rounded = round(orig_val, 4)
        new_rounded = round(new_val, 4)
        # A change judged "meaningful" relative to a (possibly tiny) robust
        # feature scale can still round to an identical displayed value for
        # a large-magnitude feature -- never report a "change" that looks
        # like no change at the precision actually shown to the user.
        if orig_rounded == new_rounded:
            continue
        pct = ((new_val - orig_val) / orig_val * 100) if abs(orig_val) > 1e-9 else None
        changes.append(
            {
                "feature": feature,
                "original_value": orig_rounded,
                "counterfactual_value": new_rounded,
                "change": f"{pct:+.1f}%" if pct is not None else f"{new_val - orig_val:+.4f}",
                "plain_english": _plain_english(feature, orig_val, new_val),
            }
        )
    changes.sort(key=lambda c: abs(c["counterfactual_value"] - c["original_value"]), reverse=True)

    return {
        "transaction_id": row["transaction_id"],
        "original_decision": original_decision,
        "counterfactual_decision": "ALLOW",
        "changes_needed": changes,
        "minimum_changes_required": len(changes),
    }
