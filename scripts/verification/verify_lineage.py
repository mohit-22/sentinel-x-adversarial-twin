import json
from app.api.endpoints import initialize_app_state, run_adaptive_arena, AdaptiveArenaRequest, api_observatory_lineage
from app.red_team.attack_genomes import MICRO_STRUCTURING_GENOME

initialize_app_state(seed=42)
req = AdaptiveArenaRequest(
    genome_id=MICRO_STRUCTURING_GENOME["genome_id"],
    population_size=3,
    generations=2,
    n_instances=5
)
run_adaptive_arena(req)

res = api_observatory_lineage()
lineage = res["lineage"]

nodes = len(lineage)
edges = 0
missing_parents = 0
cycles = 0
root_genome_id = MICRO_STRUCTURING_GENOME["genome_id"]

ids = set([n["genome"]["genome_id"] for n in lineage])

for node in lineage:
    parent_id = node["parent_attack_id"]
    node_id = node["genome"]["genome_id"]
    if parent_id is not None and parent_id != node_id:
        edges += 1
        if parent_id not in ids:
            missing_parents += 1
    elif parent_id == node_id:
        cycles += 1
        
print("--- LINEAGE RECONSTRUCTION ---")
print(f"root: {root_genome_id}")
print(f"number of nodes: {nodes}")
print(f"number of parent-child edges: {edges}")
print(f"number of missing parents: {missing_parents}")
print(f"number of cycles: {cycles}")
