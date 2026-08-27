import json
from app.api.endpoints import initialize_app_state, run_adaptive_arena, AdaptiveArenaRequest, api_observatory_lineage, api_observatory_impact
from app.red_team.attack_genomes import MICRO_STRUCTURING_GENOME

initialize_app_state(seed=42)
req = AdaptiveArenaRequest(
    genome_id=MICRO_STRUCTURING_GENOME["genome_id"],
    population_size=2,
    generations=2,
    n_instances=5
)
run_adaptive_arena(req)

print("\n--- LINEAGE ---")
lineage = api_observatory_lineage()
# Truncating trajectory for display
trajectory = lineage.pop("trajectory", [])
lineage["trajectory"] = trajectory[:2]
print(json.dumps(lineage, indent=2))

print("\n--- IMPACT ---")
impact = api_observatory_impact()
print(json.dumps(impact, indent=2))

print("\n--- STIX ---")
from fastapi.testclient import TestClient
from app.main import app
client = TestClient(app)
export_res = client.post("/api/v1/observatory/export", json={"run_id": lineage["run_id"], "genome_id": trajectory[0]["genome_id"]})
print(f"Status: {export_res.status_code}")
bundle = export_res.json()
print("Bundle ID:", bundle.get("id"))
print("Objects count:", len(bundle.get("objects", [])))
