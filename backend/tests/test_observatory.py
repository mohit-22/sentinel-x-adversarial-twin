import pytest
from fastapi.testclient import TestClient
from app.main import app
from app.api.endpoints import _APP_STATE, _LATEST_ARENA_RUN
import app.api.endpoints as endpoints
import app.red_team.arena as arena
from app.red_team.attack_genomes import MICRO_STRUCTURING_GENOME

client = TestClient(app)

@pytest.fixture(autouse=True)
def reset_state():
    # Clear out any previous state before each test
    endpoints._APP_STATE.clear()
    endpoints._LATEST_ARENA_RUN = None
    endpoints._LATEST_ADAPTIVE_RUN = None
    arena._LATEST_ARENA_IMPACT.clear()

def test_impact_no_run():
    endpoints.initialize_app_state(seed=42)
    response = client.get("/api/v1/observatory/impact")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "run_arena_first"
    assert data["total_attack_transactions"] == 0
    assert data["total_attack_value_inr"] == 0.0
    assert data["value_caught_by_m0_inr"] == 0.0
    assert data["value_caught_after_hardening_inr"] == 0.0
    assert data["incremental_value_prevented_inr"] == 0.0

def test_impact_actual_amounts_used():
    endpoints.initialize_app_state(seed=42)
    
    # Run the arena
    run_response = client.post("/api/v1/arena/run", json={"genome_id": MICRO_STRUCTURING_GENOME["genome_id"], "n_instances": 10})
    assert run_response.status_code == 200
    
    # Get impact
    response = client.get("/api/v1/observatory/impact")
    assert response.status_code == 200
    data = response.json()
    
    assert data["status"] == "ok"
    assert data["total_attack_transactions"] > 0
    assert data["total_attack_value_inr"] > 0.0
    
    clean_history_avg = endpoints._APP_STATE["clean_history"]["amount"].mean()
    
    # Prove clean history average is not used
    # The total attack value should not be a multiple of clean history average
    assert data["total_attack_value_inr"] != data["total_attack_transactions"] * clean_history_avg

def test_impact_negative_incremental_value():
    endpoints.initialize_app_state(seed=42)
    
    # Fake an impact with negative incremental value
    from app.core.schemas import ArenaRunSummary
    
    endpoints._LATEST_ARENA_RUN = ArenaRunSummary(
        run_id="fake-run",
        attack_family="fake-family",
        initial_evasion_rate=0.5,
        final_evasion_rate=0.6,
        robustness_gain=-0.2,
        hard_examples_count=10,
        retrained_f1_score=0.9
    )
    
    arena._LATEST_ARENA_IMPACT["fake-run"] = {
        "run_id": "fake-run",
        "attack_family": "fake-family",
        "total_attack_transactions": 100,
        "total_attack_value_inr": 1000.0,
        "value_caught_by_m0_inr": 800.0,
        "value_caught_after_hardening_inr": 700.0,
        "additional_transactions_caught": -5,
        "incremental_value_prevented_inr": -100.0,
        "m0_evasion_rate": 0.2,
        "post_hardening_evasion_rate": 0.3
    }
    
    response = client.get("/api/v1/observatory/impact")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["incremental_value_prevented_inr"] == -100.0
    assert data["additional_transactions_caught"] == -5
    assert data["value_caught_by_m0_inr"] == 800.0
    assert data["value_caught_after_hardening_inr"] == 700.0

def test_same_run_produces_deterministic_metrics():
    endpoints.initialize_app_state(seed=42)
    
    run1 = client.post("/api/v1/arena/run", json={"genome_id": MICRO_STRUCTURING_GENOME["genome_id"], "n_instances": 10}).json()
    impact1 = client.get("/api/v1/observatory/impact").json()
    
    # Clear state and run again
    endpoints._APP_STATE.clear()
    endpoints._LATEST_ARENA_RUN = None
    arena._LATEST_ARENA_IMPACT.clear()
    
    endpoints.initialize_app_state(seed=42)
    run2 = client.post("/api/v1/arena/run", json={"genome_id": MICRO_STRUCTURING_GENOME["genome_id"], "n_instances": 10}).json()
    impact2 = client.get("/api/v1/observatory/impact").json()
    
    assert impact1 == impact2
    assert run1 == run2
