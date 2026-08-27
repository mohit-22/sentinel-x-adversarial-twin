from fastapi.testclient import TestClient
from backend.app.main import app

client = TestClient(app)

# Create Scenario
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

print("=== Creating Scenario ===")
r = client.post("/api/v1/judge/scenario", json=payload)
print(r.json())

print("=== Running Scenario ===")
r = client.post("/api/v1/judge/scenario/integration-real-001/run")
print(r.json())

import time
for i in range(60):
    r = client.get("/api/v1/judge/scenario/integration-real-001")
    state = r.json()
    phase = state["current_phase"]
    print(f"[{i}s] Phase: {phase}")
    if phase in ["APPROVE", "SCORE", "FAILED"]:
        break
    time.sleep(1)

state = client.get("/api/v1/judge/scenario/integration-real-001").json()
print("=== Scenario State ===")
import json
print(json.dumps(state, indent=2))

if state["current_phase"] == "APPROVE":
    print("=== Approving Policy ===")
    policy_id = state.get("candidate_policy_id")
    print(f"Policy ID: {policy_id}")
    r = client.post("/api/v1/defense/approve", json={"policy_id": policy_id, "action": "APPROVE"})
    print("Approve API response:", r.json())
    
    r = client.post("/api/v1/judge/scenario/integration-real-001/approve")
    print("Judge approve response:", r.json())
    
    time.sleep(1)
    state = client.get("/api/v1/judge/scenario/integration-real-001").json()
    print("=== Final State ===")
    print(json.dumps(state, indent=2))

