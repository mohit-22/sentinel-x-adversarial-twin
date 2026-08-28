import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__)))

from app.api.endpoints import initialize_app_state, _APP_STATE
from app.defense.recursive_engine import CompositeDefenseAdapter, ATTACK_GENOMES
from app.red_team.adaptive_attack import run_evolutionary_search
from app.red_team.attack_injector import ATTACK_GENERATORS
from app.blue_team.features import engineer_features, combine_clean_and_injected
from app.blue_team.graph_engine import apply_graph_features
from app.blue_team.detector import FEATURE_COLUMNS, evaluate_detector
from app.blue_team.defense_compiler import analyze_attack, compile_policy
from app.defense.schemas import DefensePolicy

def main():
    initialize_app_state(seed=42)
    m0_model = _APP_STATE["model"]
    customers = _APP_STATE["customers"]
    merchants = _APP_STATE["merchants"]
    graph_features = _APP_STATE["graph_features"]
    radar_state = _APP_STATE["radar_state"]
    test_df = _APP_STATE["test_df"]
    
    test_customer_ids = set(test_df["customer_id"].unique())
    eval_customers = customers[customers["customer_id"].isin(test_customer_ids)].copy()
    base_genome = ATTACK_GENOMES["micro_structuring"]
    
    seed = 45
    population_size = 3
    generations_per_round = 2
    n_instances = 20
    
    current_defense = CompositeDefenseAdapter(m0_model, [])
    
    rounds_data = []
    
    defense_id_counter = 0
    
    # We will do 2 rounds manually
    for round_idx in range(1, 3):
        attack_seed = seed + round_idx
        
        evol_result = run_evolutionary_search(
            base_genome=base_genome,
            model=current_defense,
            radar_state=radar_state,
            customers=eval_customers,
            clean_history=test_df,
            merchants=merchants,
            graph_features=graph_features,
            population_size=population_size,
            generations=generations_per_round,
            n_instances=n_instances,
            seed=attack_seed
        )
        
        best_attack = evol_result["best_attack"]
        best_genome = best_attack["genome"]
        
        my_clean_test_df = test_df[test_df["is_fraud"] == 0].sort_values(["customer_id", "timestamp"]).reset_index(drop=True).copy()
        
        m0_clean_preds = m0_model.predict(my_clean_test_df[FEATURE_COLUMNS])
        m0_fpr = m0_clean_preds.mean() if len(m0_clean_preds) > 0 else 0.0
        
        eval_test_df = my_clean_test_df
        adapter_clean_preds = current_defense.predict(my_clean_test_df[FEATURE_COLUMNS])
        adapter_fpr = adapter_clean_preds.mean() if len(adapter_clean_preds) > 0 else 0.0
        
        clean_fpr_delta = adapter_fpr - m0_fpr
        
        sorted_test_df = test_df.sort_values(["customer_id", "timestamp"]).reset_index(drop=True).copy()
        eval_test_df = sorted_test_df
        
        # Inline evaluate_detector to guarantee eval_test_df is in the stack frame
        x_test = sorted_test_df[FEATURE_COLUMNS]
        y_test = sorted_test_df["is_fraud"]
        y_pred = current_defense.predict(x_test)
        
        from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        test_metrics = {
            "precision": precision_score(y_test, y_pred, zero_division=0),
            "recall": recall_score(y_test, y_pred, zero_division=0),
            "f1": f1_score(y_test, y_pred, zero_division=0),
            "fpr": fp / (fp + tn) if (fp + tn) > 0 else 0.0,
        }
        
        gen_fn = ATTACK_GENERATORS["micro_structuring"]["attacks_fn"]
        base_attacks_df = gen_fn(base_genome, eval_customers, merchants, n_instances, seed=attack_seed)
        evolved_attacks_df = gen_fn(best_genome, eval_customers, merchants, n_instances, seed=attack_seed+1)
        
        def feat(attacks_df):
            comb = combine_clean_and_injected(test_df, attacks_df)
            comb = comb.drop_duplicates(subset="transaction_id", keep="last")
            feat_df = engineer_features(comb, eval_customers)
            feat_df = apply_graph_features(feat_df, graph_features)
            return feat_df[feat_df['is_fraud'] == 1].copy()
            
        base_featured = feat(base_attacks_df)
        evolved_featured = feat(evolved_attacks_df)
        
        analysis = analyze_attack(
            base_attack_df=base_featured,
            evolved_attack_df=evolved_featured,
            clean_history=test_df,
            base_genome=base_genome,
            evolved_genome=best_genome,
            detector=m0_model,
            features=FEATURE_COLUMNS
        )
        
        new_defense_created = False
        candidate_defense_id = "NO_NEW_DEFENSE_GENERATED"
        policy_ids = []
        policy_type = "N/A"
        
        if analysis.suspected_blind_spot:
            policies = compile_policy(analysis)
            if policies:
                new_defense_created = True
                defense_id_counter += 1
                candidate_defense_id = f"D{defense_id_counter}"
                for policy in policies:
                    policy_ids.append(policy.policy_id)
                policy_type = policies[0].policy_type
                new_policies = current_defense.policies + policies
                current_defense = CompositeDefenseAdapter(m0_model, new_policies)
                
        rd = {
            "round_number": round_idx,
            "defense_id": f"D{defense_id_counter - 1}" if defense_id_counter > 0 and not new_defense_created else (f"D{defense_id_counter-1}" if new_defense_created else "D0"),
            "parent_defense_id": "D0" if round_idx == 1 else (f"D{defense_id_counter-2}" if new_defense_created else f"D{defense_id_counter-1}"),
            "policy_ids": policy_ids,
            "policy_type": policy_type,
            "attack_target_defense": f"D{defense_id_counter - 1}" if defense_id_counter > 0 and not new_defense_created else (f"D{defense_id_counter-1}" if new_defense_created else "D0"),
            "attack_run_id": best_genome.get("genome_id", "unknown"),
            "evasion": best_attack["evasion_rate"],
            "f1": test_metrics["f1"],
            "precision": test_metrics["precision"],
            "recall": test_metrics["recall"],
            "fpr": test_metrics["fpr"],
            "clean_fpr_delta": clean_fpr_delta,
            "failure_cause": analysis.suspected_blind_spot or "UNKNOWN",
            "new_defense_created": new_defense_created,
            "candidate_defense_id": candidate_defense_id
        }
        
        if round_idx == 1:
            rd["defense_id"] = "D0"
            rd["parent_defense_id"] = "N/A"
            rd["attack_target_defense"] = "D0"
            
        elif round_idx == 2:
            rd["defense_id"] = rounds_data[0]["candidate_defense_id"] if rounds_data[0]["new_defense_created"] else "D0"
            rd["parent_defense_id"] = "D0"
            rd["attack_target_defense"] = rd["defense_id"]
            
        rounds_data.append(rd)
        
    for rd in rounds_data:
        print(f"\n--- ROUND {rd['round_number']} ---")
        print(f"Defense ID: {rd['defense_id']}")
        print(f"Parent Defense ID: {rd['parent_defense_id']}")
        print(f"Policy IDs: {rd['policy_ids']}")
        print(f"Policy Type: {rd['policy_type']}")
        print(f"Attack Target Defense: {rd['attack_target_defense']}")
        print(f"Attack Run/Lineage ID: {rd['attack_run_id']}")
        print(f"Evasion: {rd['evasion']:.4f}")
        print(f"F1: {rd['f1']:.4f}")
        print(f"Precision: {rd['precision']:.4f}")
        print(f"Recall: {rd['recall']:.4f}")
        print(f"FPR: {rd['fpr']:.4f}")
        print(f"Clean FPR Delta: {rd['clean_fpr_delta']:.4f}")
        print(f"Failure Cause: {rd['failure_cause']}")
        print(f"New Defense Created: {rd['new_defense_created']}")

if __name__ == "__main__":
    main()
