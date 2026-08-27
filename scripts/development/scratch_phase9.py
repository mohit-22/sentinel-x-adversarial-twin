from fastapi.testclient import TestClient
from backend.app.main import app

client = TestClient(app)

print("Starting Judge scenario for Phase 9...")
payload = {
    "scenario_id": "integration-real-001",
    "seed": 42,
    "attack_family": "micro_structuring",
    "attack_scale": 50,
    "difficulty": "HARD",
    "adaptive_red_team_enabled": True,
    "zero_day_radar_enabled": True,
    "defense_compiler_enabled": True,
    "human_approval_required": True,
    "evolution_generations": 2
}

client.post("/api/v1/judge/scenario", json=payload)
client.post("/api/v1/judge/scenario/integration-real-001/run")

import time
for _ in range(60):
    r = client.get("/api/v1/judge/scenario/integration-real-001")
    state = r.json()
    if state["current_phase"] in ["APPROVE", "SCORE", "FAILED"]:
        break
    time.sleep(1)

state = client.get("/api/v1/judge/scenario/integration-real-001").json()
# The failure analysis is saved in the candidate policy (if one was generated) or we can look at the scorecard
# Wait! In SCORE phase, candidate_policy_id is either there or not.
# Since we fixed analyze_attack, it should have generated a policy and stopped in APPROVE.
phase = state["current_phase"]
print(f"Final Phase: {phase}")

# We can query the compiled policy
policy_id = state.get("candidate_policy_id")
if policy_id:
    # get policy
    import json
    from backend.app.api.endpoints import _CANDIDATE_POLICIES
    policy = _CANDIDATE_POLICIES.get(policy_id)
    print("Found candidate policy:", policy.policy_id)
    # The evidence is in policy.provenance or we can just print the analysis fields from the scenario state?
    # Scenario state only has 'failure_cause' = analysis.suspected_blind_spot
    print("Failure cause:", state["failure_cause"])
    
print("Baseline evasion:", state["baseline_evasion"])
print("Evolved evasion:", state["evolved_evasion"])
diff = state["evolved_evasion"] - state["baseline_evasion"]
print("Evasion difference:", diff)

