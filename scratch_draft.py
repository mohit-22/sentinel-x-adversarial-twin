import pandas as pd
import numpy as np
from typing import Dict, Any, List

def analyze_attack(
    base_attack_df: pd.DataFrame, 
    evolved_attack_df: pd.DataFrame, 
    clean_history: pd.DataFrame,
    base_genome: Dict,
    evolved_genome: Dict,
    detector: Any = None,
    features: List[str] = None
):
    if detector is None or features is None:
        raise ValueError("Missing detector or features required to compute evasion rates")
        
    if not base_attack_df.empty:
        preds_base = detector.predict(base_attack_df[features])
        base_evasion = 1.0 - preds_base.mean()
    else:
        base_evasion = 1.0
        
    if not evolved_attack_df.empty:
        preds_ev = detector.predict(evolved_attack_df[features])
        evolved_evasion = 1.0 - preds_ev.mean()
    else:
        evolved_evasion = 1.0

    def calc_duration(df):
        if df.empty: return 0.0
        durations = []
        for _, group in df.groupby("instance_id"):
            d = (group["timestamp"].max() - group["timestamp"].min()).total_seconds() / 3600.0
            durations.append(d)
        return np.mean(durations) if durations else 0.0
        
    base_duration = calc_duration(base_attack_df)
    ev_duration = calc_duration(evolved_attack_df)
    
    # Feature comparison
    check_features = [
        "amount", "count_5m", "count_1h", "count_24h", "count_7d", 
        "sum_5m", "sum_1h", "sum_24h", "sum_7d", 
        "amount_deviation_ratio", "is_new_device", "is_new_location", 
        "shared_device_count", "two_hop_fraud_risk"
    ]
    
    before = {}
    after = {}
    dev = {}
    dominant_features = []
    
    for f in check_features:
        if f in base_attack_df.columns and f in evolved_attack_df.columns:
            b_val = float(base_attack_df[f].mean()) if not base_attack_df.empty else 0.0
            e_val = float(evolved_attack_df[f].mean()) if not evolved_attack_df.empty else 0.0
            before[f] = b_val
            after[f] = e_val
            dev[f] = e_val - b_val
            
    suspected_blind_spot = "UNKNOWN"
    evidence = []
    
    # Classification Logic
    if ev_duration > base_duration * 1.5 and ev_duration > 24.0:
        suspected_blind_spot = "TEMPORAL_VELOCITY_DILUTION"
        evidence.append(f"Attack duration extended from {base_duration:.1f}h to {ev_duration:.1f}h.")
        # Check velocity features
        vel_drops = [f for f in ["count_24h", "sum_24h"] if dev.get(f, 0) < 0]
        if vel_drops:
            evidence.append(f"Velocity features {vel_drops} decreased correspondingly.")
            dominant_features.extend(vel_drops)
        else:
            suspected_blind_spot = "UNKNOWN"
            evidence.append("But velocity features did not decrease.")
            
    elif abs(dev.get("amount", 0)) > 0.2 * before.get("amount", 1):
        suspected_blind_spot = "AMOUNT_CAMOUFLAGE"
        evidence.append(f"Amount distribution changed materially. Mean changed by {dev['amount']:.1f}.")
        dominant_features.append("amount")
        
    elif dev.get("is_new_device", 0) < -0.1 or dev.get("shared_device_count", 0) < -0.1:
        suspected_blind_spot = "DEVICE_CAMOUFLAGE"
        evidence.append("Device novelty/relationship features decreased.")
        dominant_features.extend([f for f in ["is_new_device", "shared_device_count"] if dev.get(f, 0) < 0])
        
    if suspected_blind_spot == "UNKNOWN" and not evidence:
        evidence.append("Insufficient evidence to establish a clear root cause.")
        
    return {
        "baseline_evasion": float(base_evasion),
        "evolved_evasion": float(evolved_evasion),
        "dominant_failure_features": dominant_features,
        "feature_value_before": before,
        "feature_value_after": after,
        "feature_deviation": dev,
        "suspected_blind_spot": suspected_blind_spot,
        "evidence": "; ".join(evidence)
    }

print("Draft ready")
