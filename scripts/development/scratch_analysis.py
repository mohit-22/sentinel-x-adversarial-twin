from fastapi.testclient import TestClient
from backend.app.main import app

client = TestClient(app)

print("Starting Judge scenario...")
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

from backend.app.judge.scenario_runner import _SCENARIOS
scenario_state = _SCENARIOS["integration-real-001"]

print("Base evasion:", scenario_state.baseline_evasion)
print("Evolved evasion:", scenario_state.evolved_evasion)
print("Evasion difference:", scenario_state.evolved_evasion - scenario_state.baseline_evasion)

# We want the actual AttackFailureAnalysis. 
# It is discarded in scenario_runner after DEFEND if policies is empty.
# So we need to call analyze_attack ourselves to print it!

from backend.app.blue_team.defense_compiler import analyze_attack
# Let's get it directly from scenario_runner scope or re-run analyze_attack
# But we can't easily extract df_base_attack and df_evolved_attack from _SCENARIOS.
