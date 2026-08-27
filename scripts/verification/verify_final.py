import json
import os
from fastapi.testclient import TestClient
from app.main import app
import app.api.endpoints as endpoints
import app.red_team.arena as arena
from app.red_team.attack_genomes import MICRO_STRUCTURING_GENOME

# Reset state
endpoints._APP_STATE.clear()
endpoints._LATEST_ARENA_RUN = None
endpoints._LATEST_ADAPTIVE_RUN = None
arena._LATEST_ARENA_IMPACT.clear()
endpoints.initialize_app_state(seed=42)

client = TestClient(app)

print("=== 1. FULL CLEAN START ===")
res_lin_clean = client.get("/api/v1/observatory/lineage")
print("Lineage Status:", res_lin_clean.json()["status"])

res_imp_clean = client.get("/api/v1/observatory/impact")
print("Impact Status:", res_imp_clean.json()["status"])

print("\n=== 2. REAL ADAPTIVE RUN ===")
req = {
    "genome_id": "ATK-MS-001",
    "population_size": 5,
    "generations": 3,
    "n_instances": 20
}
# We use /arena/adaptive instead of calling run_adaptive_arena to test API contract
client.post("/api/v1/arena/adaptive", json=req)

res_lin = client.get("/api/v1/observatory/lineage").json()
run_id = res_lin["run_id"]
base = res_lin["base_genome_id"]
lineage = res_lin["lineage"]

best = [n for n in lineage if n["is_best"]]
best_node = best[0] if best else None

print(f"RUN_ID: {run_id}")
print(f"BASE_GENOME_ID: {base}")
print(f"CREATED_AT: {res_lin['created_at']}")
print(f"NODE COUNT: {len(lineage)}")

edges = [n for n in lineage if n["parent_attack_id"] != n["genome"]["genome_id"]]
print(f"EDGE COUNT: {len(edges)}")
if best_node:
    print(f"BEST GENOME: {best_node['genome']['genome_id']}")
    print(f"BEST EVASION: {best_node['evasion_rate']}")
    print(f"BEST FITNESS: {best_node['total_fitness']}")

print("\n=== 3. LINEAGE API VERIFICATION ===")
genomes = {n["genome"]["genome_id"] for n in lineage}
print("Base genome present:", base in genomes)
print("Unique genome IDs:", len(genomes) == len(lineage))
missing_parents = [n["parent_attack_id"] for n in edges if n["parent_attack_id"] not in genomes]
print("Missing parents:", len(missing_parents) == 0)
print("parent_id_tracked present:", any("parent_id_tracked" in str(n["parent_attack_id"]) for n in lineage))

print("\n=== 4. ECONOMIC IMPACT VERIFICATION ===")
# Must run normal arena to get impact!
client.post("/api/v1/arena/run", json={"genome_id": "ATK-MS-001", "n_instances": 20})
imp = client.get("/api/v1/observatory/impact").json()

total_trans = imp["total_attack_transactions"]
total_val = imp["total_attack_value_inr"]
val_m0 = imp["value_caught_by_m0_inr"]
val_m1 = imp["value_caught_after_hardening_inr"]
inc_val = imp["incremental_value_prevented_inr"]

print(f"total_attack_transactions: {total_trans}")
print(f"total_attack_value_inr: {total_val}")
print(f"value_caught_by_m0_inr: {val_m0}")
print(f"value_caught_after_hardening_inr: {val_m1}")
print(f"incremental_value_prevented_inr: {inc_val}")
print(f"additional_transactions_caught: {imp['additional_transactions_caught']}")
print(f"m0_evasion_rate: {imp['m0_evasion_rate']}")
print(f"post_hardening_evasion_rate: {imp['post_hardening_evasion_rate']}")
print("Math check (val_m1 - val_m0 == inc_val):", abs(val_m1 - val_m0 - inc_val) < 0.01)

print("\n=== 7. STIX EXPORT END-TO-END ===")
stix_req = {"run_id": run_id, "genome_id": best_node['genome']['genome_id']}
stix_res = client.post("/api/v1/observatory/export", json=stix_req)
print(f"Valid request status: {stix_res.status_code}")
if stix_res.status_code == 200:
    bundle = stix_res.json()
    print("Type:", bundle.get("type"))
    print("Spec Version:", bundle.get("spec_version"))
    print("Objects > 0:", len(bundle.get("objects", [])) > 0)
    
    stix_res2 = client.post("/api/v1/observatory/export", json=stix_req)
    print("Deterministic match:", bundle == stix_res2.json())

stix_inv = client.post("/api/v1/observatory/export", json={"run_id": "invalid", "genome_id": "invalid"})
print(f"Invalid request status: {stix_inv.status_code}")

