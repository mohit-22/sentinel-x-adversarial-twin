import os
import sys
import json

sys.path.append(os.path.join(os.path.dirname(__file__)))

from app.api.endpoints import initialize_app_state, _APP_STATE
from app.defense.schemas import CertificationRequest
from app.defense.recursive_engine import run_certification

def main():
    print("Initializing app state...")
    initialize_app_state(seed=42)
    print("App state initialized. Searching for config...")

    for seed in range(1, 20):
        for pop in [5]:
            for gens in [2]:
                for scale in [20, 50]:
                    req = CertificationRequest(
                        attack_family="micro_structuring",
                        attack_scale=scale,
                        population_size=pop,
                        generations_per_round=gens,
                        rounds=2,
                        seed=seed
                    )
                    try:
                        res = run_certification(req)
                        print(f"Seed {seed}, pop {pop}, gens {gens}, scale {scale}: Rounds={len(res.rounds)}")
                        for i, r in enumerate(res.rounds):
                            print(f"  Round {i+1}: evasion={r.evasion_rate:.4f}, cause={r.failure_cause}, new_defense={r.new_defense_created}")
                            if r.new_defense_created:
                                print("SUCCESS!")
                                print(json.dumps(res.dict(), default=str, indent=2))
                                return
                    except Exception as e:
                        print(f"Error for seed {seed}: {e}")

if __name__ == "__main__":
    main()
