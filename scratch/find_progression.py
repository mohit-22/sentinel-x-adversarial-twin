import sys
import os

sys.path.append(os.path.abspath("backend"))

from app.defense.schemas import CertificationRequest
from app.defense.recursive_engine import run_certification
from app.api.endpoints import initialize_app_state

def search_for_progression():
    print("Initializing App State...")
    initialize_app_state(seed=42)
    
    attack_families = [
        "micro_structuring",
        "synthetic_identity_drift",
        "behavioral_camouflage",
        "social_engineering",
        "synthetic_voice_authorization"
    ]
    
    seeds = [42, 100, 999]
    found_config = None
    
    for family in attack_families:
        for seed in seeds:
            print(f"Testing {family} seed={seed}...")
            req = CertificationRequest(
                attack_family=family,
                seed=seed,
                rounds=1,
                generations_per_round=1, # fast search
                population_size=1,
                attack_scale=20
            )
            res = run_certification(req)
            if res.rounds and res.rounds[0].new_defense_created:
                print(f"FOUND! Family: {family}, Seed: {seed}")
                found_config = (family, seed)
                break
        if found_config:
            break
            
    if not found_config:
        print("No valid progression found in small search space. Trying larger population...")
        for family in attack_families:
            req = CertificationRequest(
                attack_family=family,
                seed=777,
                rounds=1,
                generations_per_round=2,
                population_size=3,
                attack_scale=30
            )
            res = run_certification(req)
            if res.rounds and res.rounds[0].new_defense_created:
                print(f"FOUND! Family: {family}, Seed: 777")
                found_config = (family, 777)
                break

    if not found_config:
        print("FAILED TO FIND PROGRESSION")
        return

    print("\n===========================================")
    print(f"RUNNING PROOF FOR {found_config[0]} SEED {found_config[1]}")
    print("===========================================")
    
    req = CertificationRequest(
        attack_family=found_config[0],
        seed=found_config[1],
        rounds=2,
        generations_per_round=2,
        population_size=3,
        attack_scale=30
    )
    
    res = run_certification(req)
    
    print(f"\nCERTIFICATION ID: {res.certification_id}")
    print(f"Status: {res.certification_status}")
    print("\nROUNDS DETAIL:")
    for rnd in res.rounds:
        print(f"\nRound {rnd.round_number}")
        print(f"Defense ID: {rnd.defense_id}")
        print(f"Candidate/New Defense: {rnd.candidate_defense_id}")
        print(f"New Defense Created?: {rnd.new_defense_created}")
        print(f"Weakness: {rnd.failure_cause}")
        print(f"Evasion: {rnd.evasion_rate}")
        print(f"FPR: {rnd.fpr}")
        print(f"F1: {rnd.f1}")

if __name__ == "__main__":
    search_for_progression()
