import pytest
from fastapi.testclient import TestClient
from app.main import app
from app.api.endpoints import _APP_STATE, _LATEST_ADAPTIVE_RUN
import app.api.endpoints as endpoints
from app.red_team.attack_genomes import MICRO_STRUCTURING_GENOME

client = TestClient(app)

@pytest.fixture(autouse=True)
def reset_state():
    # Clear out any previous state before each test
    endpoints._APP_STATE.clear()
    endpoints._LATEST_ARENA_RUN = None
    endpoints._LATEST_ADAPTIVE_RUN = None
    endpoints.initialize_app_state(seed=42)

def run_adaptive():
    # Run the adaptive arena
    return client.post("/api/v1/arena/adaptive", json={
        "genome_id": MICRO_STRUCTURING_GENOME["genome_id"],
        "population_size": 2,
        "generations": 1,
        "n_instances": 10
    })

def test_invalid_run_id():
    run_adaptive()
    response = client.post("/api/v1/observatory/export", json={
        "run_id": "invalid-run-id",
        "genome_id": MICRO_STRUCTURING_GENOME["genome_id"]
    })
    assert response.status_code == 404
    assert "Invalid run_id" in response.json()["detail"]

def test_genome_not_belonging_to_run():
    res = run_adaptive()
    assert res.status_code == 200
    # The returned lineage has the genomes
    run_id = endpoints._LATEST_ADAPTIVE_RUN["run_id"]
    response = client.post("/api/v1/observatory/export", json={
        "run_id": run_id,
        "genome_id": "invalid-genome-id"
    })
    assert response.status_code == 404
    assert "does not belong to run" in response.json()["detail"]

def test_valid_export_and_stix_structure():
    res = run_adaptive()
    assert res.status_code == 200
    run_id = endpoints._LATEST_ADAPTIVE_RUN["run_id"]
    
    # Get the best genome id from the run
    best_attack = res.json()["best_attack"]
    best_genome_id = best_attack["genome"]["genome_id"]
    
    response = client.post("/api/v1/observatory/export", json={
        "run_id": run_id,
        "genome_id": best_genome_id
    })
    assert response.status_code == 200
    data = response.json()
    
    # 4. Export type == "bundle"
    assert data["type"] == "bundle"
    
    # Bundle should have an ID
    assert "id" in data and data["id"].startswith("bundle--")
    
    objects = data["objects"]
    assert len(objects) == 1
    
    attack_pattern = objects[0]
    assert attack_pattern["type"] == "attack-pattern"
    
    # 5. Attack-pattern spec_version == "2.1"
    assert attack_pattern["spec_version"] == "2.1"
    
    # 8. Exported genome_id equals requested genome_id
    assert attack_pattern["x_sentinel_genome_id"] == best_genome_id
    
    # 9. Exported parameters equal the actual evolved genome parameters
    assert attack_pattern["x_sentinel_parameters"] == best_attack["genome"]["parameters"]
    
    # 10. Exported evasion_rate equals the exact lineage value
    assert attack_pattern["x_sentinel_evasion_rate"] == best_attack["evasion_rate"]
    
    # 11. Export contains no fabricated actor/CVE/malware data
    assert "threat_actor" not in attack_pattern
    assert "malware" not in attack_pattern
    assert "cve" not in attack_pattern.get("description", "").lower()
    
    # 6. STIX object IDs are deterministic
    # 7. created/modified are deterministic
    # 12. Two exports of the same run/genome produce identical parsed JSON
    response2 = client.post("/api/v1/observatory/export", json={
        "run_id": run_id,
        "genome_id": best_genome_id
    })
    
    data2 = response2.json()
    assert data == data2
    
    # Just to be extra sure, verify timestamps don't change
    assert data["objects"][0]["created"] == data2["objects"][0]["created"]
    assert data["objects"][0]["modified"] == data2["objects"][0]["modified"]
