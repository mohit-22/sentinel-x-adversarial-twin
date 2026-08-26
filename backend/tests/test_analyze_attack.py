import pytest
import pandas as pd
import numpy as np
from app.blue_team.defense_compiler import analyze_attack, AttackFailureAnalysis

class MockDetector:
    def __init__(self, return_value=0):
        self.return_value = return_value
        self.last_X = None
        
    def predict(self, X):
        self.last_X = X
        # return_value = 0 means 100% evasion
        return np.full(len(X), self.return_value)

def create_mock_attack_df(duration_hours=2.0, amount_mean=100.0, is_new_device=0.0):
    timestamps = pd.date_range(start="2023-01-01", periods=10, freq=f"{duration_hours/10}h")
    return pd.DataFrame({
        "instance_id": [1] * 10,
        "timestamp": timestamps,
        "amount": np.random.normal(amount_mean, 10, 10),
        "is_new_device": [is_new_device] * 10,
        "count_24h": [5] * 10,
        "sum_24h": [500] * 10,
        "shared_device_count": [0] * 10
    })

def test_evasion_computation():
    base_df = create_mock_attack_df()
    ev_df = create_mock_attack_df()
    
    det1 = MockDetector(return_value=0) # 100% evasion
    res1 = analyze_attack(base_df, ev_df, pd.DataFrame(), {}, {}, detector=det1, features=["amount"])
    assert res1.baseline_evasion == 1.0
    assert res1.evolved_evasion == 1.0
    
    det2 = MockDetector(return_value=1) # 0% evasion
    res2 = analyze_attack(base_df, ev_df, pd.DataFrame(), {}, {}, detector=det2, features=["amount"])
    assert res2.baseline_evasion == 0.0
    assert res2.evolved_evasion == 0.0
    
    # Prove no static 0.0460 exists
    assert res1.baseline_evasion != 0.0460

def test_feature_deltas_change_with_input():
    base_df = create_mock_attack_df(amount_mean=100.0)
    ev_df = create_mock_attack_df(amount_mean=200.0)
    
    det = MockDetector()
    res = analyze_attack(base_df, ev_df, pd.DataFrame(), {}, {}, detector=det, features=["amount"])
    
    assert res.feature_deviation["amount"] > 80.0 # ~100 diff
    
    # When input changes, deviation changes
    ev_df2 = create_mock_attack_df(amount_mean=300.0)
    res2 = analyze_attack(base_df, ev_df2, pd.DataFrame(), {}, {}, detector=det, features=["amount"])
    assert res2.feature_deviation["amount"] > 180.0 # ~200 diff

def test_root_cause_changes():
    det = MockDetector()
    
    # 1. Temporal dilution
    base_df = create_mock_attack_df(duration_hours=2.0)
    base_df["count_24h"] = 10
    ev_df = create_mock_attack_df(duration_hours=48.0)
    ev_df["count_24h"] = 2
    res_temporal = analyze_attack(base_df, ev_df, pd.DataFrame(), {}, {}, detector=det, features=["amount"])
    assert res_temporal.suspected_blind_spot == "TEMPORAL_VELOCITY_DILUTION"
    assert "surpassing the 24h rolling velocity feature window" in res_temporal.evidence
    assert "Mocking dominant features" not in res_temporal.evidence
    
    # 2. Amount camouflage
    base_df = create_mock_attack_df(amount_mean=100.0)
    ev_df = create_mock_attack_df(amount_mean=200.0)
    res_amount = analyze_attack(base_df, ev_df, pd.DataFrame(), {}, {}, detector=det, features=["amount"])
    assert res_amount.suspected_blind_spot == "AMOUNT_CAMOUFLAGE"
    
    # 3. Device camouflage
    base_df = create_mock_attack_df(is_new_device=1.0)
    ev_df = create_mock_attack_df(is_new_device=0.0)
    res_device = analyze_attack(base_df, ev_df, pd.DataFrame(), {}, {}, detector=det, features=["amount"])
    assert res_device.suspected_blind_spot == "DEVICE_CAMOUFLAGE"
    
    # 4. Unknown
    base_df = create_mock_attack_df()
    ev_df = create_mock_attack_df()
    res_unknown = analyze_attack(base_df, ev_df, pd.DataFrame(), {}, {}, detector=det, features=["amount"])
    assert res_unknown.suspected_blind_spot == "UNKNOWN"

def test_identical_input_produces_identical_analysis():
    np.random.seed(42)
    base_df = create_mock_attack_df()
    ev_df = create_mock_attack_df(amount_mean=200.0)
    det = MockDetector()
    
    res1 = analyze_attack(base_df, ev_df, pd.DataFrame(), {}, {}, detector=det, features=["amount"])
    res2 = analyze_attack(base_df, ev_df, pd.DataFrame(), {}, {}, detector=det, features=["amount"])
    
    assert res1.model_dump() == res2.model_dump()

def test_insufficient_evidence_returns_unknown():
    det = MockDetector()
    base_df = create_mock_attack_df()
    ev_df = create_mock_attack_df()
    # Almost no change
    res = analyze_attack(base_df, ev_df, pd.DataFrame(), {}, {}, detector=det, features=["amount"])
    assert res.suspected_blind_spot == "UNKNOWN"
    assert "Insufficient evidence" in res.evidence
