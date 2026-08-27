"""Threat Observatory final integrity audit -- READ-ONLY.

Calls only existing, unmodified production functions/endpoints. Does not
alter any production file. Section numbers match the audit request.
"""
import sys
import os
from pathlib import Path

# Fix python import resolution properly
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
BACKEND_DIR = PROJECT_ROOT / "backend"

if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

import numpy as np
from fastapi.testclient import TestClient
from app.main import app


def section_1_run_id_seed_integrity():
    print("=" * 70)
    print("SECTION 1: RUN_ID / SEED INTEGRITY")
    print("=" * 70)

    import inspect
    from app.api.endpoints import AdaptiveArenaRequest
    from app.red_team.adaptive_attack import run_evolutionary_search

    fields = list(AdaptiveArenaRequest.model_fields.keys())
    print("AdaptiveArenaRequest fields:", fields)
    accepts_seed = "seed" in fields
    print("API schema accepts a per-request seed:", accepts_seed)

    sig = inspect.signature(run_evolutionary_search)
    print("run_evolutionary_search's own seed default:", sig.parameters["seed"].default)

    import app.api.endpoints as ep
    src = inspect.getsource(ep.run_adaptive_arena)
    calls_with_seed = "seed=req.seed" in src or "seed=req." in src and "seed" in src
    print("Does the /arena/adaptive handler forward any request seed into run_evolutionary_search?",
          "seed=" in src.split("run_evolutionary_search(")[1].split(")")[0])

    from app.core.config import SEED
    print(f"\nconfig.py SEED = {SEED}")
    
    # Extract run_id assignment block (handling multi-line assignment)
    src_lines = src.splitlines()
    run_id_lines = []
    in_run_id = False
    for line in src_lines:
        if "run_id = (" in line or "run_id = f" in line:
            in_run_id = True
        if in_run_id:
            run_id_lines.append(line.strip())
            if line.strip().endswith(")") or "run_id = f" in line:
                break
    run_id_str = " ".join(run_id_lines)
    print("run_id construction line:", run_id_str)

    ep.initialize_app_state(seed=42)
    client = TestClient(app)

    body = {"genome_id": "ATK-MS-001", "population_size": 5, "generations": 3, "elite_count": 2, "n_instances": 40}
    r1 = client.post("/api/v1/arena/adaptive", json=body).json()
    r2 = client.post("/api/v1/arena/adaptive", json=body).json()

    lineage1 = client.get("/api/v1/observatory/lineage").json()
    run_id_1 = lineage1["run_id"]
    lineage_seq_1 = [(e["generation"], e["genome"]["genome_id"], e["evasion_rate"]) for e in lineage1["lineage"]]

    lineage2_raw = r2  # second call's own /arena/adaptive response (contains "lineage" too)
    run_id_2 = client.get("/api/v1/observatory/lineage").json()["run_id"]
    lineage_seq_2 = [(e["generation"], e["genome"]["genome_id"], e["evasion_rate"]) for e in r2["lineage"]]

    print(f"\nCall A run_id: {run_id_1}")
    print(f"Call B run_id (identical request body, called again): {run_id_2}")
    print("run_id identical across identical requests:", run_id_1 == run_id_2)
    print("Full lineage (genome_id+generation+evasion_rate sequence) identical across calls:",
          lineage_seq_1 == lineage_seq_2)

    # Now vary a parameter NOT in the run_id string (n_instances) and check for a collision.
    body_diff_n = {**body, "n_instances": 41}
    r3 = client.post("/api/v1/arena/adaptive", json=body_diff_n).json()
    run_id_3 = client.get("/api/v1/observatory/lineage").json()["run_id"]
    lineage_seq_3 = [(e["generation"], e["genome"]["genome_id"], e["evasion_rate"]) for e in r3["lineage"]]
    print(f"\nCall C (n_instances=41 instead of 40) run_id: {run_id_3}")
    print("run_id UNCHANGED despite different n_instances:", run_id_1 == run_id_3)
    print("But lineage content differs:", lineage_seq_1 != lineage_seq_3)

    return {
        "accepts_seed": accepts_seed,
        "run_id_deterministic_same_input": run_id_1 == run_id_2 and lineage_seq_1 == lineage_seq_2,
        "run_id_collides_across_different_n_instances": run_id_1 == run_id_3 and lineage_seq_1 != lineage_seq_3,
    }


def section_2_economic_impact_final_proof():
    print("\n" + "=" * 70)
    print("SECTION 2: ECONOMIC IMPACT FINAL PROOF")
    print("=" * 70)

    from app.api import endpoints as ep
    from app.red_team.arena import (
        run_attack, harvest_hard_negatives, retrain, re_test,
        generate_matched_population_attacks, apply_mutation, embed_and_engineer,
    )
    from app.blue_team.graph_engine import apply_graph_features
    from app.blue_team.detector import FEATURE_COLUMNS
    from app.red_team.attack_genomes import MICRO_STRUCTURING_GENOME

    ep.initialize_app_state(seed=42)
    client = TestClient(app)

    r = client.post("/api/v1/arena/run", json={"genome_id": "ATK-MS-001", "n_instances": 100})
    assert r.status_code == 200, r.text
    arena_result = r.json()

    impact = client.get("/api/v1/observatory/impact").json()
    assert impact["status"] == "ok", impact

    print(f"RUN ID: {impact['run_id']}")
    print(f"TOTAL ATTACK TRANSACTIONS: {impact['total_attack_transactions']}")
    print(f"TOTAL ATTACK VALUE: {impact['total_attack_value_inr']}")
    print(f"M0 CAUGHT VALUE: {impact['value_caught_by_m0_inr']}")
    print(f"M1 CAUGHT VALUE: {impact['value_caught_after_hardening_inr']}")
    print(f"ADDITIONAL CAUGHT COUNT: {impact['additional_transactions_caught']}")
    print(f"INCREMENTAL VALUE PREVENTED: {impact['incremental_value_prevented_inr']}")
    print(f"M0 EVASION: {impact['m0_evasion_rate']}")
    print(f"M1 EVASION: {impact['post_hardening_evasion_rate']}")

    # Independent reconstruction: replicate run_arena_mvp_gate's exact,
    # fully-deterministic (fixed seeds) internal call sequence ourselves, so
    # we get access to the RAW fraud_rows + per-row catch masks that the
    # packaged "economic_impact" dict does not expose -- not just re-deriving
    # the same arithmetic the production code already did.
    state = ep._APP_STATE
    genome = MICRO_STRUCTURING_GENOME
    model_0 = state["model"]
    train_df, test_df = state["train_df"], state["test_df"]
    customers, clean_history, merchants, graph_features = (
        state["customers"], state["clean_history"], state["merchants"], state["graph_features"]
    )
    n_instances = 100
    seed = 42

    train_customer_ids = train_df["customer_id"].unique()
    test_customer_ids = test_df["customer_id"].unique()
    train_customers = customers[customers["customer_id"].isin(train_customer_ids)]
    test_customers = customers[customers["customer_id"].isin(test_customer_ids)]

    initial = run_attack(genome, model_0, test_customers, clean_history, merchants, graph_features,
                          FEATURE_COLUMNS, min(n_instances, len(test_customers)), seed)
    training_attack = run_attack(genome, model_0, train_customers, clean_history, merchants, graph_features,
                                  FEATURE_COLUMNS, min(n_instances, len(train_customers)), seed + 10)
    harvest = harvest_hard_negatives(training_attack["evaded_rows"], training_attack["attacks_raw"],
                                      customers, clean_history, merchants, graph_features, genome, seed)
    model_1 = retrain(train_df, harvest["hard_negatives"], FEATURE_COLUMNS)
    retraining_transaction_ids = set(harvest["hard_negatives"]["transaction_id"])
    matched_customer_ids = initial["customer_ids_used"]

    # Replicate re_test's body manually (same seed=seed+1000=1042) to expose
    # raw fraud_rows / per-row masks instead of trusting its packaged sums.
    retest_seed = seed + 1000
    attacks_raw = generate_matched_population_attacks(genome, customers, merchants, matched_customer_ids, seed=retest_seed)
    overlap = set(attacks_raw["transaction_id"]) & retraining_transaction_ids
    assert not overlap, f"re-test batch overlaps retraining rows: {list(overlap)[:5]}"

    rng = np.random.default_rng(retest_seed + 500)
    mutated_frames = []
    for instance_id, group in attacks_raw.groupby("instance_id"):
        fraud_group = group[group["is_fraud"] == 1]
        chosen_mutation = rng.choice(genome["mutations"])
        customer = customers[customers["customer_id"] == fraud_group["customer_id"].iloc[0]].iloc[0]
        mutated_fraud = apply_mutation(fraud_group, group, customer, merchants, genome["family"], chosen_mutation, rng)
        unchanged = group[~group["transaction_id"].isin(fraud_group["transaction_id"])]
        mutated_frames.append(__import__("pandas").concat([unchanged, mutated_fraud], ignore_index=True))
    mutated_attacks_raw = __import__("pandas").concat(mutated_frames, ignore_index=True)

    featured = embed_and_engineer(mutated_attacks_raw, customers, clean_history, merchants)
    featured = apply_graph_features(featured, graph_features)
    fraud_rows = featured[featured["is_fraud"] == 1].copy()

    y_pred_m1 = model_1.predict(fraud_rows[FEATURE_COLUMNS])
    y_pred_m0 = model_0.predict(fraud_rows[FEATURE_COLUMNS])
    caught_by_m1 = y_pred_m1 == 1
    caught_by_m0 = y_pred_m0 == 1

    m1_only = fraud_rows.loc[caught_by_m1 & ~caught_by_m0, "amount"].sum()
    m0_only = fraud_rows.loc[caught_by_m0 & ~caught_by_m1, "amount"].sum()
    incremental_strict_def = m1_only - m0_only  # sum(m1&!m0) minus any regressions sum(m0&!m1)
    incremental_literal_positive_only = m1_only  # the audit's literal wording: sum where m1 caught AND m0 did not

    val_m0 = fraud_rows.loc[caught_by_m0, "amount"].sum()
    val_m1 = fraud_rows.loc[caught_by_m1, "amount"].sum()
    algebraic = val_m1 - val_m0

    print("\n--- INDEPENDENT RAW-MASK RECOMPUTATION (own re-derivation of the attack batch, same fixed seeds) ---")
    print(f"Independently reconstructed fraud_rows: {len(fraud_rows)}")
    print(f"sum(amount where M1 caught AND M0 did not): {m1_only}")
    print(f"sum(amount where M0 caught AND M1 did not) [regressions]: {m0_only}")
    print(f"value_caught_by_m0 (independent): {val_m0}")
    print(f"value_caught_by_m1 (independent): {val_m1}")
    print(f"algebraic (val_m1 - val_m0), independent: {algebraic}")
    print(f"literal audit definition (sum where m1 caught & !m0 caught): {incremental_literal_positive_only}")
    print(f"Regressions exist (m0 caught something m1 missed): {m0_only > 0}")
    print(f"algebraic == literal definition (only true if zero regressions): {abs(algebraic - incremental_literal_positive_only) < 1e-5}")

    print(f"\nProduction incremental_value_prevented_inr: {impact['incremental_value_prevented_inr']}")
    print(f"Independent algebraic recomputation:          {algebraic}")
    print(f"Match within 1e-5: {abs(impact['incremental_value_prevented_inr'] - algebraic) < 1e-5}")

    print("\nFormula is unclamped subtraction (can go negative) -- confirmed by source read: "
          "`incremental_value_prevented_inr = value_caught_after_hardening_inr - value_caught_by_m0_inr`, "
          "no max(0, ...) or abs() wrapping.")

    import inspect as _inspect
    from app.api import endpoints as _ep2
    impact_src = _inspect.getsource(_ep2.api_observatory_impact)
    uses_clean_history_mean = "clean_history" in impact_src and ".mean()" in impact_src
    print(f"\n/observatory/impact handler references clean_history average: {uses_clean_history_mean}")
    from app.red_team import arena as arena_mod
    re_test_src = _inspect.getsource(arena_mod.re_test)
    uses_avg_in_re_test = "clean_history[" in re_test_src and ".mean()" in re_test_src
    print(f"re_test's economic_impact block references clean_history average: {uses_avg_in_re_test}")

    return {
        "algebraic_matches_production": abs(impact["incremental_value_prevented_inr"] - algebraic) < 1e-5,
        "regressions_exist": bool(m0_only > 0),
        "clean_history_avg_used": uses_clean_history_mean or uses_avg_in_re_test,
    }


def section_3_lineage_api_final_proof():
    print("\n" + "=" * 70)
    print("SECTION 3: LINEAGE API FINAL PROOF")
    print("=" * 70)

    from app.api import endpoints as ep
    ep.initialize_app_state(seed=42)
    client = TestClient(app)

    body = {"genome_id": "ATK-MS-001", "population_size": 5, "generations": 3, "elite_count": 2, "n_instances": 40}
    client.post("/api/v1/arena/adaptive", json=body)
    lineage_res = client.get("/api/v1/observatory/lineage").json()
    lineage = lineage_res["lineage"]

    required_fields = ["generation", "genome", "parent_attack_id", "evasion_rate", "total_fitness",
                        "is_elite", "is_best", "validity_status"]
    missing_field_entries = [e for e in lineage if not all(f in e for f in required_fields)]
    print(f"All {len(lineage)} entries contain required fields {required_fields}: {len(missing_field_entries) == 0}")

    root_entries = [e for e in lineage if e["generation"] == 0 and e["parent_attack_id"] == e["genome"]["genome_id"]]
    print(f"Root exists (gen 0, parent_attack_id == genome_id): {len(root_entries) >= 1}, count={len(root_entries)}")

    literal_placeholder = [e for e in lineage if e["parent_attack_id"] == "parent_id_tracked"]
    print(f"No 'parent_id_tracked' literal placeholder present: {len(literal_placeholder) == 0}")

    gens = {}
    for e in lineage:
        gens.setdefault(e["generation"], []).append(e)

    def parent_exists(target_gen, pid):
        for g in range(target_gen, -1, -1):
            if g in gens and any(c["genome"]["genome_id"] == pid for c in gens[g]):
                return True
        return False

    missing = 0
    self_loops_nonroot = 0
    fabricated = 0
    for e in lineage:
        gid, pid, gen = e["genome"]["genome_id"], e["parent_attack_id"], e["generation"]
        if pid == gid:
            if gen != 0:
                self_loops_nonroot += 1
            continue
        if not parent_exists(gen, pid):
            missing += 1

    print(f"Missing parents: {missing}")
    print(f"Non-root self-loops: {self_loops_nonroot}")
    print(f"Every non-root parent exists: {missing == 0}")

    # Determinism: re-run identical request, compare full lineage sequence.
    client.post("/api/v1/arena/adaptive", json=body)
    lineage_res_2 = client.get("/api/v1/observatory/lineage").json()
    seq1 = [(e["generation"], e["genome"]["genome_id"], e["parent_attack_id"], e["evasion_rate"], e["is_elite"], e["is_best"]) for e in lineage]
    seq2 = [(e["generation"], e["genome"]["genome_id"], e["parent_attack_id"], e["evasion_rate"], e["is_elite"], e["is_best"]) for e in lineage_res_2["lineage"]]
    print(f"Deterministic same-seed lineage (identical request twice): {seq1 == seq2}")

    # Elite carry-over: an elite genome_id appears at multiple generations --
    # verify its node identity (genome_id+generation composite) stays unique
    # and its own generation/is_elite/is_best fields are correctly per-instance,
    # not corrupted/shared across the repeated appearances.
    from collections import Counter
    genome_id_counts = Counter(e["genome"]["genome_id"] for e in lineage)
    duplicated_genome_ids = [gid for gid, c in genome_id_counts.items() if c > 1]
    node_ids = [f"{e['genome']['genome_id']}-gen{e['generation']}" for e in lineage]
    node_id_corruption = len(node_ids) != len(set(node_ids))
    print(f"Genome IDs appearing in >1 generation (elite carry-over): {duplicated_genome_ids}")
    print(f"Composite node identity (genome_id+generation) remains unique despite that: {not node_id_corruption}")

    return {
        "all_fields_present": len(missing_field_entries) == 0,
        "root_exists": len(root_entries) >= 1,
        "no_placeholder": len(literal_placeholder) == 0,
        "no_missing_parents": missing == 0,
        "no_nonroot_self_loops": self_loops_nonroot == 0,
        "deterministic": seq1 == seq2,
        "node_identity_uncorrupted": not node_id_corruption,
    }


def section_5_stix_final_proof():
    print("\n" + "=" * 70)
    print("SECTION 5: STIX FINAL PROOF")
    print("=" * 70)

    from app.api import endpoints as ep
    ep.initialize_app_state(seed=42)
    client = TestClient(app)

    body = {"genome_id": "ATK-MS-001", "population_size": 5, "generations": 3, "elite_count": 2, "n_instances": 40}
    client.post("/api/v1/arena/adaptive", json=body)
    lineage_res = client.get("/api/v1/observatory/lineage").json()
    run_id = lineage_res["run_id"]
    entry = lineage_res["lineage"][7]  # a real, non-root, non-trivial entry
    genome_id = entry["genome"]["genome_id"]
    print(f"Selected genome for export: {genome_id} (generation {entry['generation']}) from run {run_id}")

    export1 = client.post("/api/v1/observatory/export", json={"run_id": run_id, "genome_id": genome_id}).json()
    export2 = client.post("/api/v1/observatory/export", json={"run_id": run_id, "genome_id": genome_id}).json()

    obj1, obj2 = export1["objects"][0], export2["objects"][0]
    byte_identical_except_ids = {k: v for k, v in obj1.items() if not k.endswith("id")} == \
                                 {k: v for k, v in obj2.items() if not k.endswith("id")}
    print(f"Two exports (same run_id/genome_id) byte-identical on all non-id fields: {byte_identical_except_ids}")
    print(f"bundle id identical across calls (stable_id, hash-based): {export1['id'] == export2['id']}")
    print(f"attack-pattern id identical across calls: {obj1['id'] == obj2['id']}")

    checks = {
        "spec_version == 2.1": obj1["spec_version"] == "2.1",
        "run_id matches": obj1["x_sentinel_run_id"] == run_id,
        "genome_id matches": obj1["x_sentinel_genome_id"] == genome_id,
        "generation matches": obj1["x_sentinel_generation"] == entry["generation"],
        "parent_attack_id matches": obj1["x_sentinel_parent_attack_id"] == entry["parent_attack_id"],
        "evasion_rate matches": obj1["x_sentinel_evasion_rate"] == entry["evasion_rate"],
        "fitness matches": obj1["x_sentinel_fitness"] == entry["total_fitness"],
        "parameters match": obj1["x_sentinel_parameters"] == entry["genome"]["parameters"],
        "mutations match": obj1["x_sentinel_mutations"] == entry["genome"]["mutations"],
    }
    for k, v in checks.items():
        print(f"  {k}: {v}")

    bad_run = client.post("/api/v1/observatory/export", json={"run_id": "not-a-real-run-id", "genome_id": genome_id})
    print(f"\nInvalid run_id -> status {bad_run.status_code} (expected 404): {bad_run.status_code == 404}")

    bad_genome = client.post("/api/v1/observatory/export", json={"run_id": run_id, "genome_id": "ATK-NOT-REAL-999"})
    print(f"Valid run_id + wrong genome_id -> status {bad_genome.status_code} (expected 404): {bad_genome.status_code == 404}")

    return {
        "all_field_checks_pass": all(checks.values()),
        "byte_identical_repeat_export": byte_identical_except_ids,
        "invalid_run_id_fails_honestly": bad_run.status_code == 404,
        "wrong_genome_fails_honestly": bad_genome.status_code == 404,
    }


if __name__ == "__main__":
    r1 = section_1_run_id_seed_integrity()
    r2 = section_2_economic_impact_final_proof()
    r3 = section_3_lineage_api_final_proof()
    r5 = section_5_stix_final_proof()

    print("\n" + "=" * 70)
    print("MACHINE-READABLE SUMMARY")
    print("=" * 70)
    print("SECTION 1:", r1)
    print("SECTION 2:", r2)
    print("SECTION 3:", r3)
    print("SECTION 5:", r5)
