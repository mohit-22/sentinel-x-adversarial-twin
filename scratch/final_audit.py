import sys
import os
import hashlib
import time
import pandas as pd
import numpy as np

sys.path.append(os.path.abspath("backend"))

from app.defense.recursive_engine import CompositeDefenseAdapter, run_certification
from app.defense.schemas import CertificationRequest
from app.api.endpoints import initialize_app_state, _APP_STATE

def run_adapter_proof():
    print("\n--- 3. ADAPTER PROOF ---")
    class DummyM0:
        def predict(self, X):
            return np.array([0, 0, 1, 0])
            
    class DummyPolicy1:
        def __init__(self):
            self.policy_id = "P1"
        def __call__(self, df):
            return pd.Series([False, True, False, False])
            
    class DummyPolicy2:
        def __init__(self):
            self.policy_id = "P2"
        def __call__(self, df):
            return pd.Series([True, False, False, False])

    # Mock the apply_policy function globally for this test
    import app.defense.recursive_engine as engine
    original_apply = engine.apply_policy
    engine.apply_policy = lambda df, pol: pol(df)
    
    adapter = CompositeDefenseAdapter(DummyM0(), [DummyPolicy1(), DummyPolicy2()])
    
    # Needs a dataframe of len 4
    df = pd.DataFrame({"dummy": [1, 2, 3, 4]})
    df.index = [0, 1, 2, 3]
    
    preds = adapter.predict(df)
    print("M0: [0, 0, 1, 0]")
    print("P1: [0, 1, 0, 0]")
    print("P2: [1, 0, 0, 0]")
    print(f"Final adapter output: {list(preds)}")
    
    engine.apply_policy = original_apply


def run_targeting_and_leakage_proof():
    print("\n--- 4. TARGETING & 6. LEAKAGE & 5. FRESH ATTACKS ---")
    initialize_app_state(seed=42)
    
    req = CertificationRequest(
        attack_family="micro_structuring",
        seed=42,
        rounds=2,
        generations_per_round=2,
        population_size=3,
        attack_scale=20
    )
    
    res = run_certification(req)
    
    for rnd in res.rounds:
        print(f"\nRound: {rnd.round_number}")
        print(f"Defense ID: {rnd.defense_id}")
        print(f"Attack Run ID: {rnd.attack_run_id}")
        print(f"Evasion: {rnd.evasion_rate}")
        print(f"F1: {rnd.f1}")
        print(f"FPR: {rnd.fpr}")
        
    print("\nLeakage Check:")
    train_cust = set(_APP_STATE["train_df"]["customer_id"].unique())
    test_cust = set(_APP_STATE["test_df"]["customer_id"].unique())
    inter = train_cust.intersection(test_cust)
    print(f"Train/Test Customer Intersection Size: {len(inter)}")
    
    
if __name__ == "__main__":
    run_adapter_proof()
    run_targeting_and_leakage_proof()
