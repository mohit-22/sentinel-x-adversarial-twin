import os
import sys
import json

sys.path.append(os.path.join(os.path.dirname(__file__)))

from app.api.endpoints import initialize_app_state, _APP_STATE
from app.defense.schemas import CertificationRequest
from app.defense.recursive_engine import run_certification

def main():
    initialize_app_state(seed=42)
    
    req = CertificationRequest(
        attack_family="micro_structuring",
        seed=45,
        rounds=2,
        generations_per_round=2,
        population_size=3,
        attack_scale=20
    )
    
    res = run_certification(req)
        
    print("\n================ CERTIFICATION RESULT ================")
    print(f"Certification Status: {res.certification_status}")
    print(f"Customer Leakage: {res.customer_leakage}")
    print(f"Row Leakage: {res.row_leakage}")
    
    for i, rnd in enumerate(res.rounds):
        print(f"\n--- ROUND {rnd.round_number} ---")
        print(f"Defense ID: {rnd.defense_id}")
        parent_defense_id = "N/A"
        if i == 0:
            parent_defense_id = res.starting_defense_id
        elif res.rounds[i-1].candidate_defense_id and res.rounds[i-1].candidate_defense_id != "NO_NEW_DEFENSE_GENERATED":
            parent_defense_id = res.rounds[i-1].candidate_defense_id
            
        print(f"Parent Defense ID: {parent_defense_id}")
        
        print(f"Attack Target Defense: {rnd.defense_id}")
        print(f"Attack Run/Lineage ID: {rnd.attack_run_id}")
        print(f"Evasion: {rnd.evasion_rate:.4f}")
        print(f"F1: {rnd.f1:.4f}")
        print(f"Precision: {rnd.precision:.4f}")
        print(f"Recall: {rnd.recall:.4f}")
        print(f"FPR: {rnd.fpr:.4f}")
        print(f"Clean FPR Delta: {rnd.clean_fpr_delta:.4f}")
        print(f"Failure Cause: {rnd.failure_cause}")
        print(f"New Defense Created: {rnd.new_defense_created}")
        print(f"Candidate Defense ID: {rnd.candidate_defense_id}")
        
    # Also verify edge case behavior by running a seed we know fails to create a policy (e.g. 40)
    print("\n================ EDGE CASE BEHAVIOR ================")
    req_edge = CertificationRequest(
        attack_family="micro_structuring",
        seed=40,
        rounds=1,
        generations_per_round=2,
        population_size=3,
        attack_scale=20
    )
    res_edge = run_certification(req_edge)
    for rnd in res_edge.rounds:
        print(f"Round {rnd.round_number} New Defense Created: {rnd.new_defense_created}")
        print(f"Candidate Defense ID: {rnd.candidate_defense_id}")
        print(f"Evasion: {rnd.evasion_rate:.4f}")

if __name__ == "__main__":
    main()
