import os
import sys
from pathlib import Path

# Fix python import resolution properly
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
BACKEND_DIR = PROJECT_ROOT / "backend"

if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from fastapi.testclient import TestClient
from app.main import app
def check():
    from app.api import endpoints
    endpoints.initialize_app_state(seed=42)
    client = TestClient(app)
    
    POPULATION_SIZE = 5
    GENERATIONS = 3
    client.post("/api/v1/arena/adaptive", json={"genome_id": "ATK-MS-001", "generations": GENERATIONS, "population_size": POPULATION_SIZE})

    lineage_res = client.get("/api/v1/observatory/lineage").json()
    if lineage_res["status"] != "ok":
        print("Run arena first!")
        return

    # GET /api/v1/observatory/lineage is a pure passthrough of the real
    # _LATEST_ADAPTIVE_RUN cache (see endpoints.py's api_observatory_lineage)
    # -- it already exposes the full "lineage" list (genome dict,
    # parent_attack_id, is_elite, is_best, novelty/impact/realism scores,
    # validity_status) directly, not just the flattened "trajectory". No
    # separate internal-cache inspection is needed: nothing here is hidden
    # from the public endpoint, confirmed by reading api_observatory_lineage
    # itself (it returns _LATEST_ADAPTIVE_RUN unmodified) and by a live
    # curl against a populated run. Every check below is API-level.
    lineage = lineage_res["lineage"]
    trajectory = lineage_res["trajectory"]
    
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
        # INCLUSIVE of target_gen itself: generation 0's own initial
        # mutations have their parent (the base genome) sitting IN
        # generation 0, not an earlier one. range(target_gen-1, -1, -1)
        # was the bug -- it always skips generation 0's own children,
        # producing (population_size - 1) false "missing parents" on every
        # run. Matches production's own real, already-verified logic
        # (frontend/src/app/observatory/page.tsx's getParentNodeId, which
        # searches `for (let g = targetGen; g >= 0; g--)` -- inclusive).
        for g in range(target_gen, -1, -1):
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
        
        assert pid != "parent_id_tracked", "Found placeholder parent_id_tracked"
        
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

    assert missing == 0, f"Found {missing} missing parents"
    assert self_loops == 0, f"Found {self_loops} non-root self loops"

    # "Node count match" was previously `total_records == len(lineage)` --
    # comparing a variable to itself, always True regardless of any real
    # bug. The real, meaningful cross-check available from this same
    # response: /observatory/lineage returns both "lineage" (full entries)
    # and "trajectory" (flattened), built from the same list in
    # endpoints.py's /arena/adaptive handler -- they must have equal length.
    expected_total = POPULATION_SIZE * GENERATIONS
    node_count_consistent = total_records == len(trajectory) == expected_total
    print(f"Node count match (lineage vs trajectory vs population*generations={expected_total}): {node_count_consistent}")
    assert node_count_consistent, f"Node count mismatch: lineage={total_records} trajectory={len(trajectory)} expected={expected_total}"

    # /observatory/impact is populated by /arena/run (via
    # _LATEST_ARENA_RUN/_LATEST_ARENA_IMPACT), NOT by /arena/adaptive --
    # the previous version of this script never called /arena/run, so
    # /observatory/impact was stuck on its honest "run_arena_first"
    # all-zero fallback the whole time, and "Matches: True" was really just
    # 0.0 == 0.0 -- a vacuous pass, not a real check of the formula.
    client.post("/api/v1/arena/run", json={"genome_id": "ATK-MS-001", "n_instances": 100})
    impact = client.get("/api/v1/observatory/impact").json()
    print("\nECONOMIC IMPACT REGRESSION")
    print("status:", impact["status"])
    
    assert impact["status"] == "ok", "/arena/run did not populate a real impact record -- cannot check the arithmetic meaningfully."
    
    # Never treat all-zero `run_arena_first` as a successful impact proof.
    assert impact["total_attack_value_inr"] > 0, "Impact value is 0, meaning it hit the fallback all-zero state"
    
    print("total_attack_value_inr:", impact["total_attack_value_inr"])
    print("value_caught_by_m0_inr:", impact["value_caught_by_m0_inr"])
    print("value_caught_after_hardening_inr:", impact["value_caught_after_hardening_inr"])
    print("incremental_value_prevented_inr:", impact["incremental_value_prevented_inr"])
    val_m0 = impact["value_caught_by_m0_inr"]
    val_m1 = impact["value_caught_after_hardening_inr"]
    inc = impact["incremental_value_prevented_inr"]
    print("Recomputed exactly val_m1 - val_m0:", val_m1 - val_m0)
    
    assert abs(inc - (val_m1 - val_m0)) < 1e-5, f"Impact formula mismatch! Expected {val_m1 - val_m0}, got {inc}"
    print("Matches:", True)

check()

