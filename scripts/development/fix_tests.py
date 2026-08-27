from fastapi.testclient import TestClient
from backend.app.main import app

client = TestClient(app)
payload = {
    "attack_id": "test_attack",
    "attack_family": "test_family",
    "baseline_evasion": 0.5,
    "evolved_evasion": 0.5,
    "dominant_failure_features": [],
    "feature_value_before": {},
    "feature_value_after": {},
    "suspected_blind_spot": "TEMPORAL_VELOCITY_DILUTION",
    "feature_deviation": {},
    "temporal_pattern": {"evolved_duration_hours": 72},
    "graph_pattern": {},
    "novelty_pattern": {},
    "evidence": "test"
}
res = client.post("/api/v1/defense/compile", json=payload)
print(res.status_code)
print(res.json())
