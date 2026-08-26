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

def test_lineage_no_run():
    endpoints.initialize_app_state(seed=42)
    response = client.get("/api/v1/observatory/lineage")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "no_run"
    assert data["trajectory"] == []

def test_lineage_after_run_structure():
    endpoints.initialize_app_state(seed=42)
    run_response = client.post("/api/v1/arena/adaptive", json={"genome_id": MICRO_STRUCTURING_GENOME["genome_id"], "population_size": 2, "generations": 2, "n_instances": 5})
    assert run_response.status_code == 200
    
    response = client.get("/api/v1/observatory/lineage")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert len(data["trajectory"]) > 0
    assert data["run_id"] is not None
    
    # Ensure lineage exists in internal state
    assert endpoints._LATEST_ADAPTIVE_RUN["lineage"] is not None
    lineage = endpoints._LATEST_ADAPTIVE_RUN["lineage"]
    
    best_count = 0
    elite_count = 0
    parent_tracked_found = False
    
    # Root self-reference should not count as missing
    genome_ids = {n["genome"]["genome_id"] for n in lineage}
    
    for node in lineage:
        parent_id = node["parent_attack_id"]
        genome_id = node["genome"]["genome_id"]
        
        # Test C and D: Every non-root has valid parent, no parent_id_tracked
        if parent_id and "parent_id_tracked" in str(parent_id):
            parent_tracked_found = True
            
        if parent_id is not None and parent_id != genome_id:
            assert parent_id in genome_ids
            
        if node.get("is_best"):
            best_count += 1
            
        if node.get("is_elite"):
            elite_count += 1

    assert not parent_tracked_found, "parent_id_tracked found in lineage"
    assert best_count >= 1, "There should be at least one is_best node"
    assert elite_count > 0, "There should be some elite nodes"
    
def test_lineage_deterministic_seed():
    endpoints.initialize_app_state(seed=42)
    client.post("/api/v1/arena/adaptive", json={"genome_id": MICRO_STRUCTURING_GENOME["genome_id"], "population_size": 2, "generations": 2, "n_instances": 5})
    lineage1 = client.get("/api/v1/observatory/lineage").json()["trajectory"]
    
    # Clear and rerun with same seed
    endpoints._APP_STATE.clear()
    endpoints._LATEST_ADAPTIVE_RUN = None
    endpoints.initialize_app_state(seed=42)
    client.post("/api/v1/arena/adaptive", json={"genome_id": MICRO_STRUCTURING_GENOME["genome_id"], "population_size": 2, "generations": 2, "n_instances": 5})
    lineage2 = client.get("/api/v1/observatory/lineage").json()["trajectory"]
    
    assert lineage1 == lineage2

def test_economic_impact_arithmetic_consistency():
    endpoints.initialize_app_state(seed=42)
    # Run arena to generate impact
    client.post("/api/v1/arena/run", json={"genome_id": MICRO_STRUCTURING_GENOME["genome_id"], "n_instances": 10})
    
    response = client.get("/api/v1/observatory/impact")
    assert response.status_code == 200
    data = response.json()
    
    val_m0 = data["value_caught_by_m0_inr"]
    val_m1 = data["value_caught_after_hardening_inr"]
    inc_val = data["incremental_value_prevented_inr"]
    
    # Enforce canonical rule:
    # incremental_value_prevented_inr = value_caught_after_hardening_inr - value_caught_by_m0_inr
    assert abs(inc_val - (val_m1 - val_m0)) < 1e-5, f"Mismatch: {inc_val} != {val_m1} - {val_m0}"
