import time
import uuid
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
from pydantic import BaseModel, Field

class AttackFailureAnalysis(BaseModel):
    attack_id: str
    attack_family: str
    baseline_evasion: float
    evolved_evasion: float
    dominant_failure_features: List[str]
    feature_value_before: Dict[str, float]
    feature_value_after: Dict[str, float]
    feature_deviation: Dict[str, float]
    temporal_pattern: Dict[str, Any]
    graph_pattern: Dict[str, Any]
    novelty_pattern: Dict[str, Any]
    suspected_blind_spot: str
    evidence: str

class DefensePolicy(BaseModel):
    policy_id: str
    version: int
    source_attack_id: str
    source_attack_family: str
    root_cause: str
    policy_type: str
    conditions: Dict[str, Any]
    action: str
    severity: str
    confidence: float
    created_at: float = Field(default_factory=time.time)
    status: str
    provenance: str

def analyze_attack(
    base_attack_df: pd.DataFrame, 
    evolved_attack_df: pd.DataFrame, 
    clean_history: pd.DataFrame,
    base_genome: Dict,
    evolved_genome: Dict,
    detector: Any = None,
    features: List[str] = None
) -> AttackFailureAnalysis:
    """
    Analyzes the features of the evolved attack vs the base attack to identify the root cause.
    """
    if detector is None or features is None:
        raise ValueError("Missing detector or features required to compute evasion rates")
        
    if not base_attack_df.empty:
        preds_base = detector.predict(base_attack_df[features])
        base_evasion = float(1.0 - preds_base.mean())
    else:
        base_evasion = 1.0
        
    if not evolved_attack_df.empty:
        preds_ev = detector.predict(evolved_attack_df[features])
        evolved_evasion = float(1.0 - preds_ev.mean())
    else:
        evolved_evasion = 1.0

    # Compare the parameters
    base_params = base_genome.get("parameters", {})
    ev_params = evolved_genome.get("parameters", {})
    
    # Calculate duration
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
    found_causes = []
    
    # 1. Temporal dilution check
    if ev_duration > base_duration * 1.5 and ev_duration > 24.0:
        vel_drops = [f for f in ["count_24h", "sum_24h"] if dev.get(f, 0) < 0]
        if vel_drops:
            found_causes.append("TEMPORAL_VELOCITY_DILUTION")
            evidence.append(f"Attack duration extended from {base_duration:.1f}h to {ev_duration:.1f}h, surpassing the 24h rolling velocity feature window. Velocity features {vel_drops} decreased correspondingly.")
            dominant_features.extend(vel_drops)
            
    # 2. Amount camouflage check
    if abs(dev.get("amount", 0)) > 0.2 * before.get("amount", 1):
        found_causes.append("AMOUNT_CAMOUFLAGE")
        evidence.append(f"Amount distribution changed materially. Mean changed by {dev['amount']:.1f}.")
        dominant_features.append("amount")
        
    # 3. Device camouflage check
    if dev.get("is_new_device", 0) < -0.1 or dev.get("shared_device_count", 0) < -0.1:
        found_causes.append("DEVICE_CAMOUFLAGE")
        evidence.append("Device novelty/relationship features decreased.")
        dominant_features.extend([f for f in ["is_new_device", "shared_device_count"] if dev.get(f, 0) < 0])

    if len(found_causes) == 1:
        suspected_blind_spot = found_causes[0]
    elif len(found_causes) > 1:
        suspected_blind_spot = "MULTI_FACTOR"
        evidence.insert(0, "Multiple evasion vectors detected concurrently.")
        
    if suspected_blind_spot == "UNKNOWN" and not evidence:
        evidence.append("Insufficient evidence to establish a clear root cause.")
        
    analysis = AttackFailureAnalysis(
        attack_id=evolved_genome.get("genome_id", "unknown"),
        attack_family=evolved_genome.get("family", "unknown"),
        baseline_evasion=base_evasion,
        evolved_evasion=evolved_evasion,
        dominant_failure_features=dominant_features,
        feature_value_before=before,
        feature_value_after=after,
        feature_deviation=dev,
        temporal_pattern={"base_duration_hours": base_duration, "evolved_duration_hours": ev_duration},
        graph_pattern={},
        novelty_pattern={},
        suspected_blind_spot=suspected_blind_spot,
        evidence="; ".join(evidence)
    )
    return analysis

def compile_policy(analysis: AttackFailureAnalysis) -> List[DefensePolicy]:
    """
    Compiles candidate defense policies deterministically from the failure analysis.
    """
    policies = []
    
    if analysis.suspected_blind_spot == "TEMPORAL_VELOCITY_DILUTION":
        # Propose a policy that adds a 72h or 96h rolling window check
        duration = analysis.temporal_pattern.get("evolved_duration_hours", 72)
        window = "72h" if duration <= 72 else "96h"
        if duration > 96:
            window = "168h" # 7d
            
        policy = DefensePolicy(
            policy_id=f"POL-TEMP-{str(uuid.uuid4())[:8]}",
            version=1,
            source_attack_id=analysis.attack_id,
            source_attack_family=analysis.attack_family,
            root_cause=analysis.suspected_blind_spot,
            policy_type="TEMPORAL_POLICY",
            conditions={
                "rolling_window": window,
                "count_threshold": 5, # We can tune this in simulator, start conservative
                "amount_threshold": 2000.0,
                "is_new_beneficiary_required": True
            },
            action="BLOCK",
            severity="HIGH",
            confidence=0.8,
            status="CANDIDATE",
            provenance="compiler_deterministic_rule"
        )
        policies.append(policy)
        
    return policies
