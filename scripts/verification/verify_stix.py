import json
from app.api.endpoints import initialize_app_state, run_adaptive_arena, AdaptiveArenaRequest, api_observatory_export, ObservatoryExportRequest
import app.api.endpoints as endpoints
from app.red_team.attack_genomes import MICRO_STRUCTURING_GENOME

print("Initializing state...")
initialize_app_state(seed=42)

print("Running adaptive arena...")
req = AdaptiveArenaRequest(
    genome_id=MICRO_STRUCTURING_GENOME["genome_id"],
    population_size=2,
    generations=1,
    n_instances=10
)
res = run_adaptive_arena(req)

best = res["best_attack"]
best_genome_id = best["genome"]["genome_id"]
run_id = endpoints._LATEST_ADAPTIVE_RUN["run_id"]

print(f"\nCaptured:")
print(f"RUN_ID: {run_id}")
print(f"BEST/SELECTED GENOME_ID: {best_genome_id}")
print(f"GENERATION: {best['generation']}")
print(f"PARENT_ID: {best['parent_attack_id']}")
print(f"EVASION: {best['evasion_rate']}")
print(f"FITNESS: {best['total_fitness']}")

print("\nExporting...")
req_export = ObservatoryExportRequest(run_id=run_id, genome_id=best_genome_id)
export_1 = api_observatory_export(req_export)
print("\nFirst Export JSON:")
print(json.dumps(export_1, indent=2))

export_2 = api_observatory_export(req_export)

print("\n--- Verification ---")
print(f"Identical exports? {export_1 == export_2}")
print(f"run_id in request == run_id in export? {run_id == export_1['objects'][0]['x_sentinel_run_id']}")
print(f"genome_id in request == genome_id in export? {best_genome_id == export_1['objects'][0]['x_sentinel_genome_id']}")
print(f"exported parameters == actual evolved parameters? {best['genome']['parameters'] == export_1['objects'][0]['x_sentinel_parameters']}")
print(f"spec_version == '2.1'? {export_1['objects'][0]['spec_version'] == '2.1'}")
