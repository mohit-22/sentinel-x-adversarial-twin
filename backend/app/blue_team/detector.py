"""LightGBM fraud detector: customer-grouped split, train, evaluate (PRD §7.3).

Split ratio/method, LightGBM hyperparameters, and the feature list are Day-4
decisions approved by the project owner (not in CLAUDE.md/PRD, flagged before
implementation). config.py is intentionally NOT touched here -- Day 4's
ALLOWED_TO_TOUCH in CLAUDE.md §2 excludes it, so these live as local
constants instead.
"""

from typing import Dict, List, Tuple

import lightgbm as lgb
import pandas as pd
from sklearn.metrics import average_precision_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

from app.blue_team.graph_engine import apply_graph_features, compute_graph_features

TEST_SIZE: float = 0.25

LGBM_PARAMS: Dict = {
    "n_estimators": 200,
    "num_leaves": 31,
    "learning_rate": 0.05,
    "max_depth": -1,
    "random_state": 42,
    "is_unbalance": True,
}

# Explicitly excludes: is_fraud (target), genome_id/attack_family (direct
# proxies for is_fraud in this single-family dataset), merchant_id/
# merchant_category/channel (the fraud legs use a fixed "P2P-TRANSFER"/"P2P"
# placeholder -- including these as raw categoricals would let the model
# just memorize the generation artifact instead of learning behavioral
# signal), and all raw identifier/timestamp columns.
#
# Also excludes is_new_beneficiary, beneficiary_in_degree, and
# beneficiary_out_degree, by explicit project-owner decision (Day 4):
# with only one un-mutated attack family, routing to a brand-new mule
# beneficiary is a deterministic tell (P(is_new_beneficiary=1|fraud)=1.0 vs
# 0.027 for clean). Including it pushed F1 to ~0.9965 -- not because of a
# leak, but because it gives the detector a near-total shortcut against this
# one genome, leaving no real evasion signal for the Day 5 Adversarial Arena
# to demonstrate (Initial Evasion Rate would already be ~0). Still computed
# by features.py/graph_engine.py (available to explainability.py) -- just
# held out of THIS model's inputs.
# TODO: re-add after Day 6 (multiple attack families) once beneficiary
# novelty is no longer a near-total shortcut for a single genome.
FEATURE_COLUMNS: List[str] = [
    "amount",
    "count_5m", "count_1h", "count_24h", "count_7d",
    "sum_5m", "sum_1h", "sum_24h", "sum_7d",
    "amount_deviation_ratio",
    "is_new_device", "is_new_location",
    "shared_device_count",
    "two_hop_fraud_risk",
]


def customer_grouped_stratified_split(
    df: pd.DataFrame, test_size: float = TEST_SIZE, seed: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Every row for a given customer_id goes entirely to one side. Stratified
    on whether the customer has >=1 fraud row, so the held-out set gets a
    representative share of attacked customers despite them being a minority.
    """
    customer_labels = df.groupby("customer_id")["is_fraud"].max().rename("has_fraud").reset_index()
    train_customers, test_customers = train_test_split(
        customer_labels["customer_id"],
        test_size=test_size,
        random_state=seed,
        stratify=customer_labels["has_fraud"],
    )
    train_df = df[df["customer_id"].isin(train_customers)].copy()
    test_df = df[df["customer_id"].isin(test_customers)].copy()
    return train_df, test_df


def train_lightgbm_detector(train_df: pd.DataFrame, feature_columns: List[str] = FEATURE_COLUMNS):
    """Fit the LightGBM binary classifier on engineered features only."""
    model = lgb.LGBMClassifier(**LGBM_PARAMS)
    model.fit(train_df[feature_columns], train_df["is_fraud"])
    return model


def evaluate_detector(model, test_df: pd.DataFrame, feature_columns: List[str] = FEATURE_COLUMNS) -> Dict:
    """Real precision/recall/F1/PR-AUC/FPR on the held-out test set."""
    x_test = test_df[feature_columns]
    y_test = test_df["is_fraud"]
    import inspect
    if hasattr(model, 'predict'):
        sig = inspect.signature(model.predict)
        if 'context' in sig.parameters:
            y_pred = model.predict(x_test, context={'eval_df': test_df, 'featured_df': test_df})
        else:
            y_pred = model.predict(x_test)
    else:
        y_pred = model.predict(x_test)
        
    if hasattr(model, 'predict_proba'):
        sig = inspect.signature(model.predict_proba)
        if 'context' in sig.parameters:
            y_proba = model.predict_proba(x_test, context={'eval_df': test_df, 'featured_df': test_df})[:, 1]
        else:
            y_proba = model.predict_proba(x_test)[:, 1]
    else:
        y_proba = model.predict_proba(x_test)[:, 1]

    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    return {
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "pr_auc": average_precision_score(y_test, y_proba),
        "fpr": fp / (fp + tn),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def run_blue_team_pipeline(
    featured_df: pd.DataFrame, seed: int = 42, feature_columns: List[str] = FEATURE_COLUMNS
) -> Dict:
    """Split -> train-only graph -> apply graph features to both sides ->
    train -> evaluate. featured_df must already have novelty/rolling/
    deviation-ratio features from features.engineer_features.
    """
    train_df, test_df = customer_grouped_stratified_split(featured_df, seed=seed)

    graph_features = compute_graph_features(train_df)
    train_df = apply_graph_features(train_df, graph_features)
    test_df = apply_graph_features(test_df, graph_features)

    model = train_lightgbm_detector(train_df, feature_columns)
    metrics = evaluate_detector(model, test_df, feature_columns)

    return {
        "model": model,
        "metrics": metrics,
        "train_df": train_df,
        "test_df": test_df,
        "graph_features": graph_features,
        "feature_columns": feature_columns,
    }
