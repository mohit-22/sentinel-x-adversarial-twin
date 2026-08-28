import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__)))

from app.api.endpoints import initialize_app_state
from app.defense.schemas import CertificationRequest
from app.defense.recursive_engine import run_certification

def main():
    initialize_app_state(seed=42)
    req = CertificationRequest(
        attack_family="micro_structuring",
        seed=42,
        rounds=1,
        generations_per_round=1,
        population_size=2,
        attack_scale=20
    )
    res = run_certification(req)
    for i, r in enumerate(res.rounds):
        print(f"Round {i}: evasion={r.evasion_rate:.4f}, cause={r.failure_cause}")

if __name__ == "__main__":
    main()
