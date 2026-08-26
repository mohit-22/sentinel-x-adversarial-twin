"""Zero-Day Attack Radar (Phase 3).

Unsupervised novelty detection (Isolation Forest) and attack clustering (DBSCAN)
to discover unknown behavioral clusters that evade the supervised detector.
"""

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

from app.blue_team.detector import FEATURE_COLUMNS

# We exclude features that might encode the supervised label directly.
# While 'two_hop_fraud_risk' is a supervised graph feature, it evaluates historical 
# training labels, not the current row's label. However, purely for unsupervised 
# reference building, it's safer to only use raw behavioral features. But the plan 
# approved it as safe since it's just a historic node property.
SAFE_NOVELTY_FEATURES = [f for f in FEATURE_COLUMNS if f not in ("is_fraud", "attack_family", "genome_id")]


def train_novelty_detector(
    train_df: pd.DataFrame, 
    feature_columns: List[str] = SAFE_NOVELTY_FEATURES,
    contamination: float = 0.01,
    seed: int = 42
) -> Dict:
    """Fit Isolation Forest on the reference space (clean and fraud training data).
    
    Returns the model and normalization bounds. We do not drop fraud from the train set 
    because we want to establish the *entire* known behavior space (both known clean 
    and known fraud). Novelty means "different from BOTH known clean and known fraud".
    """
    model = IsolationForest(
        n_estimators=100, 
        max_samples="auto", 
        contamination=contamination, 
        random_state=seed, 
        n_jobs=-1
    )
    
    # IsolationForest does not require standardization.
    X_train = train_df[feature_columns]
    model.fit(X_train)
    
    # Compute scores on the train set to establish normalization bounds
    train_scores = model.score_samples(X_train)
    # score_samples returns negative anomaly scores. Lower = more anomalous.
    # We want novelty in [0, 1] where 1 is highly novel.
    # Raw scores typically range from -1.0 to 0.0.
    # Novelty = (max_score - score) / (max_score - min_score)
    # We use robust bounds (e.g. 1st and 99th percentiles) to prevent extreme outliers 
    # from compressing the scale.
    min_bound = np.percentile(train_scores, 0.1)
    max_bound = np.percentile(train_scores, 99.9)
    
    return {
        "model": model,
        "min_bound": min_bound,
        "max_bound": max_bound,
        "feature_columns": feature_columns
    }


def compute_novelty_score(
    radar_state: Dict, 
    df: pd.DataFrame
) -> np.ndarray:
    """Compute normalized [0.0, 1.0] novelty scores. 1.0 = highly novel."""
    if df.empty:
        return np.array([])
        
    model = radar_state["model"]
    cols = radar_state["feature_columns"]
    min_b = radar_state["min_bound"]
    max_b = radar_state["max_bound"]
    
    raw_scores = model.score_samples(df[cols])
    
    # Invert and scale:
    # Lower raw score (more negative) -> higher novelty
    # If raw_score == max_b (most normal), novelty = 0
    # If raw_score == min_b (most anomalous), novelty = 1
    novelty = (max_b - raw_scores) / (max_b - min_b + 1e-9)
    
    # Clip to exactly [0, 1] in case of values beyond the percentiles
    novelty = np.clip(novelty, 0.0, 1.0)
    
    return novelty


def find_novelty_threshold(
    radar_state: Dict, 
    validation_df: pd.DataFrame, 
    target_false_unknown_rate: float = 0.01
) -> float:
    """Select threshold on validation set (e.g. 20% of train data held out) to bound False Unknown Rate."""
    # We only care about bounding False Unknowns on CLEAN data.
    clean_val = validation_df[validation_df["is_fraud"] == 0]
    if clean_val.empty:
        return 0.8 # Fallback
        
    scores = compute_novelty_score(radar_state, clean_val)
    # The threshold is the score at the (1 - target_FUR) quantile.
    # e.g. for target 0.01 (1%), we want 99% of clean scores to fall below threshold.
    threshold = float(np.percentile(scores, 100 * (1.0 - target_false_unknown_rate)))
    return threshold


def cluster_unknowns(
    unknown_df: pd.DataFrame, 
    feature_columns: List[str] = SAFE_NOVELTY_FEATURES,
    eps: float = 1.5,
    min_samples: int = 5
) -> pd.DataFrame:
    """Cluster transactions tagged as UNKNOWN using DBSCAN.
    
    Requires standard scaling first since DBSCAN is distance-based.
    Returns the dataframe with an added 'cluster_id' column (-1 = noise).
    """
    df = unknown_df.copy()
    if len(df) < min_samples:
        df["cluster_id"] = -1
        return df
        
    X = df[feature_columns]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    clustering = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=-1)
    df["cluster_id"] = clustering.fit_predict(X_scaled)

    return df


def generate_cluster_report(clustered_df: pd.DataFrame, feature_columns: List[str] = SAFE_NOVELTY_FEATURES) -> List[Dict]:
    """Create a structured summary of each identified cluster."""
    reports = []
    
    if "cluster_id" not in clustered_df or clustered_df.empty:
        return reports
        
    for cluster_id, group in clustered_df.groupby("cluster_id"):
        if cluster_id == -1:
            continue # Skip noise
            
        # Find dominant feature deviations compared to typical mean of 0 (since it's raw data, 
        # we just compute raw means, but it's more informative to look at percentiles)
        # We'll just return the mean of the safe features for the LLM.
        feature_means = group[feature_columns].mean().to_dict()
        
        rep = {
            "cluster_id": int(cluster_id),
            "transaction_count": len(group),
            "novelty_score_mean": float(group["novelty_score"].mean()) if "novelty_score" in group else 0.0,
            "novelty_score_max": float(group["novelty_score"].max()) if "novelty_score" in group else 0.0,
            "first_seen_timestamp": str(group["timestamp"].min()),
            "last_seen_timestamp": str(group["timestamp"].max()),
            "feature_means": feature_means,
            "representative_transaction_ids": group["transaction_id"].head(5).tolist()
        }
        reports.append(rep)
        
    # Sort by novelty mean descending
    reports.sort(key=lambda x: x["novelty_score_mean"], reverse=True)
    return reports
