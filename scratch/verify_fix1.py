import os
import sys
from pprint import pprint

# setup sys.path to find 'app'
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "backend")))

from app.api.endpoints import initialize_app_state, _GENOME_REGISTRY, _APP_STATE
from app.red_team.adaptive_attack import run_evolutionary_search

def test_lineage():
    # Initialize the detector and state
    print("Initializing state...")
    initialize_app_state()
    app_state = _APP_STATE
    model = app_state["model"]
    radar = app_state["radar_state"]
    customers = app_state["customers"]
    merchants = app_state["merchants"]
    clean = app_state["clean_history"]
    graph = app_state["graph_features"]
    
    base_genome = _GENOME_REGISTRY["ATK-MS-001"].copy()
    
    print("Running adaptive search 1...")
    res1 = run_evolutionary_search(
        base_genome=base_genome,
        model=model,
        radar_state=radar,
        customers=customers,
        clean_history=clean,
        merchants=merchants,
        graph_features=graph,
        n_instances=20,
        generations=3,
        seed=42
    )
    
    print("Running adaptive search 2...")
    res2 = run_evolutionary_search(
        base_genome=base_genome,
        model=model,
        radar_state=radar,
        customers=customers,
        clean_history=clean,
        merchants=merchants,
        graph_features=graph,
        n_instances=20,
        generations=3,
        seed=42
    )

    print("\n--- LINEAGE 1 ---")
    lineage1 = res1["lineage"]
    known_ids = set()
    known_ids.add(base_genome["genome_id"])

    for entry in lineage1:
        genome_id = entry["genome"]["genome_id"]
        known_ids.add(genome_id)

    missing_parents = False
    for entry in lineage1:
        gen = entry["generation"]
        genome_id = entry["genome"]["genome_id"]
        parent_id = entry["parent_attack_id"]
        evasion = entry["evasion_rate"]
        fitness = entry["total_fitness"]
        is_elite = entry.get("is_elite")
        is_best = entry.get("is_best")
        validity = entry.get("validity_status")
        print(f"Gen {gen} | ID: {genome_id} | Parent: {parent_id} | Evasion: {evasion:.2f} | Fitness: {fitness:.2f} | Elite: {is_elite} | Best: {is_best} | {validity}")
        
        if parent_id not in known_ids:
            print(f"  -> WARNING: Parent ID {parent_id} not found in lineage or base!")
            missing_parents = True
            
    if not missing_parents:
        print("-> SUCCESS: All non-root genomes point to a valid parent genome_id.")

    # Check identical
    lineage1_ids = [(e["generation"], e["genome"]["genome_id"], e["parent_attack_id"]) for e in lineage1]
    lineage2_ids = [(e["generation"], e["genome"]["genome_id"], e["parent_attack_id"]) for e in res2["lineage"]]

    if lineage1_ids == lineage2_ids:
        print("-> SUCCESS: Lineage structure is identical across runs with the same seed.")
    else:
        print("-> FAILURE: Lineage structures differ.")

if __name__ == "__main__":
    test_lineage()
