import os
from fastapi.testclient import TestClient
from app.main import app
def check():
    from app.api import endpoints
    endpoints.initialize_app_state(seed=42)
    client = TestClient(app)
    
    client.post("/api/v1/arena/adaptive", json={"genome_id": "ATK-MS-001", "generations": 3, "population_size": 5})
    
    lineage_res = client.get("/api/v1/observatory/lineage").json()
    if lineage_res["status"] != "ok":
        print("Run arena first!")
        return

    lineage = lineage_res["lineage"]
    
    total_records = len(lineage)
    genomes = [x["genome"]["genome_id"] for x in lineage]
    unique_genomes = len(set(genomes))
    
    # simulate lineage_node_id
    lineage_node_ids = [f'{x["genome"]["genome_id"]}-gen{x["generation"]}' for x in lineage]
    unique_lineage_node_ids = len(set(lineage_node_ids))
    
    dup_genomes = total_records - unique_genomes
    dup_lineage = total_records - unique_lineage_node_ids
    
    print("total lineage records:", total_records)
    print("unique genome_ids:", unique_genomes)
    print("unique lineage_node_ids:", unique_lineage_node_ids)
    print("duplicate genome_ids:", dup_genomes)
    print("duplicate lineage_node_ids:", dup_lineage)
    
    # print an example of duplicates
    seen = {}
    for x in lineage:
        gid = x["genome"]["genome_id"]
        if gid not in seen:
            seen[gid] = []
        seen[gid].append(x)
        
    for gid, entries in seen.items():
        if len(entries) > 1:
            print(f"\nExample Duplicate Genome: {gid}")
            for e in entries:
                print(f"  gen={e['generation']} parent={e['parent_attack_id']} elite={e['is_elite']} best={e['is_best']}")
            break
            
    # Verify edges
    gens = {}
    for e in lineage:
        g = e["generation"]
        if g not in gens: gens[g] = []
        gens[g].append(e)
        
    def get_parent(target_gen, p_id):
        for g in range(target_gen-1, -1, -1):
            if g in gens:
                for cand in gens[g]:
                    if cand["genome"]["genome_id"] == p_id:
                        return f"{p_id}-gen{g}"
        return None
        
    missing = 0
    self_loops = 0
    edges = 0
    for e in lineage:
        gid = e["genome"]["genome_id"]
        pid = e["parent_attack_id"]
        if not pid or pid == gid:
            if e["generation"] != 0:
                self_loops += 1
            continue
            
        p_node = get_parent(e["generation"], pid)
        if not p_node:
            missing += 1
        else:
            edges += 1
            
    print(f"\nMissing parents: {missing}")
    print(f"Accidental self loops (non-root): {self_loops}")
    print(f"Reconstructed edges: {edges}")
    print(f"Node count match: {total_records == len(lineage)}")

    impact = client.get("/api/v1/observatory/impact").json()
    print("\nECONOMIC IMPACT REGRESSION")
    print("total_attack_value_inr:", impact["total_attack_value_inr"])
    print("value_caught_by_m0_inr:", impact["value_caught_by_m0_inr"])
    print("value_caught_after_hardening_inr:", impact["value_caught_after_hardening_inr"])
    print("incremental_value_prevented_inr:", impact["incremental_value_prevented_inr"])
    val_m0 = impact["value_caught_by_m0_inr"]
    val_m1 = impact["value_caught_after_hardening_inr"]
    inc = impact["incremental_value_prevented_inr"]
    print("Recomputed exactly val_m1 - val_m0:", val_m1 - val_m0)
    print("Matches:", abs(inc - (val_m1 - val_m0)) < 1e-5)

check()
