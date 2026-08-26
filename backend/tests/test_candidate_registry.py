import pytest
from fastapi.testclient import TestClient
from app.main import app
from app.api.endpoints import _CANDIDATE_POLICIES, _ACTIVE_POLICIES

@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c

@pytest.fixture(autouse=True)
def reset_registries():
    # Restart/session limitation: In-memory registries reset per session/test
    _CANDIDATE_POLICIES.clear()
    _ACTIVE_POLICIES.clear()

def test_compile_stores_candidate(client):
    # 1. compile stores candidate
    # 2. candidate status is CANDIDATE
    # 11. candidate full policy body is preserved
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
    assert res.status_code == 200
    data = res.json()
    assert len(data["policies"]) > 0
    policy = data["policies"][0]
    
    assert policy["status"] == "CANDIDATE"
    
    policy_id = policy["policy_id"]
    assert policy_id in _CANDIDATE_POLICIES
    stored = _CANDIDATE_POLICIES[policy_id]
    assert stored.status == "CANDIDATE"
    assert stored.policy_type == "TEMPORAL_POLICY"

def test_approve_existing_candidate(client):
    # 3. approve existing candidate
    # 4. approval changes status to ACTIVE
    # 5. ACTIVE policy appears in GET /defense/policies
    # 12. no fake success
    
    # First compile
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
    res_compile = client.post("/api/v1/defense/compile", json=payload)
    policy_id = res_compile.json()["policies"][0]["policy_id"]
    
    # Approve
    res_approve = client.post("/api/v1/defense/approve", json={"policy_id": policy_id, "action": "APPROVE"})
    assert res_approve.status_code == 200
    data = res_approve.json()
    assert data["status"] == "success"
    assert data["new_status"] == "ACTIVE"
    
    # Check GET /defense/policies
    res_get = client.get("/api/v1/defense/policies")
    active = res_get.json()["policies"]
    assert len(active) == 1
    assert active[0]["policy_id"] == policy_id
    assert active[0]["status"] == "ACTIVE"

def test_reject_existing_candidate(client):
    # 6. reject existing candidate
    # 7. rejected policy does not appear as ACTIVE
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
    res_compile = client.post("/api/v1/defense/compile", json=payload)
    policy_id = res_compile.json()["policies"][0]["policy_id"]
    
    # Reject
    res_reject = client.post("/api/v1/defense/approve", json={"policy_id": policy_id, "action": "REJECT"})
    assert res_reject.status_code == 200
    assert res_reject.json()["new_status"] == "REJECTED"
    
    # Ensure not active
    res_get = client.get("/api/v1/defense/policies")
    active = res_get.json()["policies"]
    assert len(active) == 0

def test_approve_unknown_policy_returns_404(client):
    # 8. approve unknown policy -> 404
    res = client.post("/api/v1/defense/approve", json={"policy_id": "UNKNOWN", "action": "APPROVE"})
    assert res.status_code == 404

def test_invalid_action_returns_validation_error(client):
    # 9. invalid action -> validation error
    payload = {
        "attack_id": "test",
        "attack_family": "test",
        "baseline_evasion": 0.5,
            "evolved_evasion": 0.5,
        "dominant_failure_features": [],
        "feature_value_before": {},
        "feature_value_after": {},
        "suspected_blind_spot": "TEMPORAL_VELOCITY_DILUTION",
        "feature_deviation": {},
        "temporal_pattern": {},
        "graph_pattern": {},
        "novelty_pattern": {},
        "evidence": "test"
    }
    res_compile = client.post("/api/v1/defense/compile", json=payload)
    policy_id = res_compile.json()["policies"][0]["policy_id"]
    
    res = client.post("/api/v1/defense/approve", json={"policy_id": policy_id, "action": "INVALID"})
    assert res.status_code == 422

def test_duplicate_activation_handled_safely(client):
    # 10. duplicate activation is handled safely
    payload = {
        "attack_id": "test",
        "attack_family": "test",
        "baseline_evasion": 0.5,
            "evolved_evasion": 0.5,
        "dominant_failure_features": [],
        "feature_value_before": {},
        "feature_value_after": {},
        "suspected_blind_spot": "TEMPORAL_VELOCITY_DILUTION",
        "feature_deviation": {},
        "temporal_pattern": {},
        "graph_pattern": {},
        "novelty_pattern": {},
        "evidence": "test"
    }
    res_compile = client.post("/api/v1/defense/compile", json=payload)
    policy_id = res_compile.json()["policies"][0]["policy_id"]
    
    # First approve
    res1 = client.post("/api/v1/defense/approve", json={"policy_id": policy_id, "action": "APPROVE"})
    assert res1.status_code == 200
    
    # Second approve should fail as state is no longer CANDIDATE
    res2 = client.post("/api/v1/defense/approve", json={"policy_id": policy_id, "action": "APPROVE"})
    assert res2.status_code == 422
    
    # Ensure only 1 active policy is stored
    res_get = client.get("/api/v1/defense/policies")
    active = res_get.json()["policies"]
    assert len(active) == 1

