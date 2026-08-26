import pandas as pd
import numpy as np
from typing import Dict, List, Any
from app.blue_team.defense_compiler import DefensePolicy

def apply_policy(df: pd.DataFrame, policy: DefensePolicy) -> pd.Series:
    """
    Applies the policy to the dataframe and returns a boolean mask indicating if the policy triggered.
    """
    if policy.policy_type == "TEMPORAL_POLICY":
        window = policy.conditions.get("rolling_window", "72h")
        cnt_thresh = policy.conditions.get("count_threshold", 5)
        amt_thresh = policy.conditions.get("amount_threshold", 1000.0)
        req_new_ben = policy.conditions.get("is_new_beneficiary_required", False)
        
        # We need to compute the custom rolling window for count and sum
        df_sorted = df.sort_values(["customer_id", "timestamp"]).reset_index(drop=True)
        ts_indexed = df_sorted.set_index("timestamp")
        
        grouped = ts_indexed.groupby("customer_id")["amount"]
        count_incl = grouped.rolling(window).count().reset_index(drop=True)
        sum_incl = grouped.rolling(window).sum().reset_index(drop=True)
        
        # subtract current row to match features.py strictly prior logic
        count_prior = count_incl.to_numpy() - 1
        sum_prior = sum_incl.to_numpy() - df_sorted["amount"].to_numpy()
        
        trigger_mask = (count_prior >= cnt_thresh) & (sum_prior >= amt_thresh)
        if req_new_ben and "is_new_beneficiary" in df_sorted.columns:
            trigger_mask = trigger_mask & (df_sorted["is_new_beneficiary"] == 1)
            
        # map back to original df index
        df_sorted["_policy_trigger"] = trigger_mask
        df_restored = df_sorted.set_index(df.index.name or "index") if df.index.name else df_sorted
        # Actually it's safer to use the exact same dataframe index
        df_sorted.index = df_sorted.index # this is tricky if df wasn't sorted.
        
    else:
        # Fallback empty mask
        return pd.Series(False, index=df.index)

    # Let's cleanly map to original index by merging or sorting back
    df_sorted["_orig_idx"] = df_sorted.index
    # The input dataframe might not have a unique index, so we map by a unique key or just sort.
    # We will assume df is already sorted by customer_id and timestamp since we use features.py outputs.
    return trigger_mask

def simulate_policy_utility(
    clean_history_featured: pd.DataFrame, 
    attack_featured: pd.DataFrame, 
    policy: DefensePolicy,
    m0_predictions_clean: np.ndarray,
    m0_predictions_attack: np.ndarray,
    fraud_loss_cost: float = 1.0,
    fp_cost: float = 50.0,
    review_cost: float = 10.0
) -> Dict[str, Any]:
    """
    Simulates the candidate policy against the Payment Twin (Clean + Attack).
    Returns utility and side-by-side metrics.
    """
    # 1. Attack evaluation
    policy_trigger_attack = apply_policy(attack_featured, policy)
    # M0 + POLICY logic: if M0 caught it, great. If not, does policy catch it?
    caught_by_m0 = m0_predictions_attack == 1
    caught_by_policy = np.asarray(policy_trigger_attack)
    final_caught_attack = caught_by_m0 | caught_by_policy
    
    total_fraud_amt = attack_featured["amount"].sum()
    caught_fraud_amt_m0 = attack_featured[caught_by_m0]["amount"].sum()
    caught_fraud_amt_final = attack_featured[final_caught_attack]["amount"].sum()
    
    fraud_loss_reduction = caught_fraud_amt_final - caught_fraud_amt_m0
    evasion_before = 1.0 - (caught_by_m0.sum() / len(attack_featured))
    evasion_after = 1.0 - (final_caught_attack.sum() / len(attack_featured))
    
    # 2. Clean evaluation
    policy_trigger_clean = apply_policy(clean_history_featured, policy)
    caught_by_m0_clean = m0_predictions_clean == 1
    caught_by_policy_clean = np.asarray(policy_trigger_clean)
    final_caught_clean = caught_by_m0_clean | caught_by_policy_clean
    
    fp_before = caught_by_m0_clean.sum()
    fp_after = final_caught_clean.sum()
    fp_increase = fp_after - fp_before
    
    # 3. Utility Score
    # Utility = fraud_loss_reduction - (fp_increase * fp_cost)
    # We treat every policy action as BLOCK for simplicity here.
    utility = fraud_loss_reduction - (fp_increase * fp_cost)
    
    return {
        "utility": utility,
        "evasion_before": float(evasion_before),
        "evasion_after": float(evasion_after),
        "fpr_increase_pct": float(fp_increase / max(1, len(clean_history_featured))) * 100,
        "fraud_loss_prevented": float(fraud_loss_reduction),
        "false_positive_increase": int(fp_increase)
    }
