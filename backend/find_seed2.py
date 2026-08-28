import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__)))

from app.api.endpoints import initialize_app_state, _APP_STATE
from app.defense.recursive_engine import CompositeDefenseAdapter, ATTACK_GENOMES
from app.red_team.adaptive_attack import run_evolutionary_search
from app.red_team.attack_injector import ATTACK_GENERATORS
from app.blue_team.features import engineer_features, combine_clean_and_injected
from app.blue_team.graph_engine import apply_graph_features
from app.blue_team.detector import FEATURE_COLUMNS
from app.blue_team.defense_compiler import analyze_attack

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
    
    adapter = CompositeDefenseAdapter(m0_model, [])
    
    for seed in range(40, 43):
        print(f"Testing seed {seed}...")
        evol_result = run_evolutionary_search(
            base_genome=base_genome,
            model=adapter,
            radar_state=radar_state,
            customers=eval_customers,
            clean_history=test_df,
            merchants=merchants,
            graph_features=graph_features,
            population_size=3,
            generations=2,
            n_instances=20,
            seed=seed
        )
        
        best_attack = evol_result["best_attack"]
        best_genome = best_attack["genome"]
        
        gen_fn = ATTACK_GENERATORS["micro_structuring"]["attacks_fn"]
        base_attacks_df = gen_fn(base_genome, eval_customers, merchants, 20, seed=seed)
        evolved_attacks_df = gen_fn(best_genome, eval_customers, merchants, 20, seed=seed+1)
        
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
            detector=adapter,
            features=FEATURE_COLUMNS
        )
        
        print(f"Seed {seed}: evasion={best_attack['evasion_rate']:.4f}, cause={analysis.suspected_blind_spot}")
        print(f"Base param: {base_genome['parameters']}")
        print(f"Evol param: {best_genome['parameters']}")
        print(f"Evidence: {analysis.evidence}")
        print(f"Dev: {analysis.feature_deviation}")
        print("-" * 50)

if __name__ == "__main__":
    main()
