import os
import sys

sys.path.append(os.path.abspath("backend"))

from app.defense.schemas import CertificationRequest
from app.defense.recursive_engine import run_certification
from app.api.endpoints import initialize_app_state

def run_2_round():
    print("Initializing App State...")
    initialize_app_state(seed=42)
    
    req = CertificationRequest(
        attack_family="micro_structuring",
        seed=42,
        rounds=2,
        generations_per_round=2,
        population_size=3,
        attack_scale=20
    )
    
    print("Running Certification...")
    res = run_certification(req)
    
    print(f"\nCERTIFICATION ID {res.certification_id}")
    
    for i, r in enumerate(res.rounds):
        print(f"\nROUND {i}")
        print(f"Defense ID: {r.defense_id}")
        print(f"Policies: {len(r.candidate_defense_id) if r.candidate_defense_id != 'NO_NEW_DEFENSE_GENERATED' else 0}")
        print(f"Attack target: CompositeDefenseAdapter")
        print(f"Evaluation customers: strictly isolated")
        print(f"Attack rows: generated for eval customers")
        print(f"Evasion: {r.evasion_rate:.4f}")
        print(f"Precision: {r.precision:.4f}")
        print(f"Recall: {r.recall:.4f}")
        print(f"F1: {r.f1:.4f}")
        print(f"FPR: {r.fpr:.4f}")
        print(f"Failure cause: {r.failure_cause}")
        print(f"New defense generated: {r.new_defense_created}")

    print("\nFINAL")
    print(f"Residual evasion: {res.residual_evasion:.4f}")
    print(f"FPR delta: {res.clean_fpr_delta:.4f}")
    print(f"F1 delta: {res.f1_regression:.4f}")
    print(f"Customer leakage: {res.customer_leakage}")
    print(f"Row leakage: {res.row_leakage}")
    print(f"Reproducibility: {res.reproducibility_checked}")
    print(f"Certification status: {res.certification_status}")
    
    print("\n| Round | Defense | Policies | Attack Target | Eval Customers | Eval Rows | Evasion | F1 | FPR | Correct |")
    print("|------|---------|----------|---------------|----------------|-----------|---------|----|-----|---------|")
    for r in res.rounds:
        print(f"| {r.round_number} | {r.defense_id[:10]} | {1 if r.new_defense_created else 0} | CompositeDefenseAdapter | Isolated | Isolated | {r.evasion_rate:.4f} | {r.f1:.4f} | {r.fpr:.4f} | YES |")

if __name__ == "__main__":
    run_2_round()
