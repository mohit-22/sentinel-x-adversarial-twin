import json
from app.api.endpoints import initialize_app_state, run_adaptive_arena, AdaptiveArenaRequest, api_observatory_lineage
from app.red_team.attack_genomes import MICRO_STRUCTURING_GENOME

initialize_app_state(seed=42)
req = AdaptiveArenaRequest(
    genome_id=MICRO_STRUCTURING_GENOME["genome_id"],
    population_size=3,
    generations=3,
    n_instances=5
)
run_adaptive_arena(req)

res = api_observatory_lineage()
lineage = res["lineage"]

nodes = len(lineage)
edges = 0
best_count = 0
elite_count = 0
root_genome = MICRO_STRUCTURING_GENOME["genome_id"]
best_genome = None

for node in lineage:
    parent_id = node["parent_attack_id"]
    genome_id = node["genome"]["genome_id"]
    if parent_id is not None and parent_id != genome_id:
        edges += 1
    if node.get("is_best"):
        best_count += 1
        best_genome = genome_id
    if node.get("is_elite"):
        elite_count += 1

print(f"Nodes: {nodes}")
print(f"Edges: {edges}")
print(f"Root: {root_genome}")
print(f"Best: {best_genome} (count {best_count})")
print(f"Elites: {elite_count}")
