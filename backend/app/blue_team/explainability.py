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

import pandas as pd
import shap

from app.blue_team.detector import FEATURE_COLUMNS

TOP_N_REASON_CODES = 3

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
