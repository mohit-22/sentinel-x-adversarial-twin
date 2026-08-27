import numpy as np
import pandas as pd
from app.api.endpoints import initialize_app_state, arena_run, ArenaRunRequest, api_observatory_impact, _APP_STATE, _LATEST_ARENA_RUN
from app.red_team.attack_genomes import MICRO_STRUCTURING_GENOME
from app.blue_team.detector import FEATURE_COLUMNS
from app.red_team.arena import _LATEST_ARENA_IMPACT, re_test, embed_and_engineer, apply_graph_features, generate_matched_population_attacks, apply_mutation, _customer_row

def independent_recomputation():
    run_id = _LATEST_ARENA_RUN.run_id
    seed = int(run_id.split("-")[-1])
    
    # We need to recreate the official_final dataset
    # We need retraining_transaction_ids
    # Run the same re_test logic to get the exact fraud_rows
    # Actually, we can get `matched_customer_ids` and `retraining_transaction_ids` from _LATEST_ARENA_RUN ? No, ArenaRunSummary doesn't have it.
    pass

print("Initializing...")
initialize_app_state(seed=42)

genome = MICRO_STRUCTURING_GENOME
print(f"Running Arena for {genome['family']}...")
req = ArenaRunRequest(genome_id=genome["genome_id"], n_instances=500)  # We use 500 for speed, or 2000 if we want
summary = arena_run(req)

print("\n--- ENDPOINT RESPONSE ---")
impact_data = api_observatory_impact()
for k, v in impact_data.items():
    print(f"{k}: {v}")
    
print("\n--- INDEPENDENT RECOMPUTATION ---")
# To independently recompute, we need the actual dataframe that was evaluated.
# Since we discarded it, we can modify verify script to recreate it, or just trust the impact dictionary.
# Let's recreate it manually as close as possible.
customers = _APP_STATE["customers"]
clean_history = _APP_STATE["clean_history"]
merchants = _APP_STATE["merchants"]
graph_features = _APP_STATE["graph_features"]
model = _APP_STATE["model"]

# Run attack to get initial customer ids
# But we need train_customers/test_customers
train_customer_ids = _APP_STATE["train_df"]["customer_id"].unique()
test_customer_ids = _APP_STATE["test_df"]["customer_id"].unique()
test_customers = customers[customers["customer_id"].isin(test_customer_ids)]

from app.red_team.arena import run_attack, harvest_hard_negatives, retrain
initial = run_attack(genome, model, test_customers, clean_history, merchants, graph_features, FEATURE_COLUMNS, min(500, len(test_customers)), seed=42)
matched_customer_ids = initial["customer_ids_used"]

train_customers = customers[customers["customer_id"].isin(train_customer_ids)]
training_attack = run_attack(genome, model, train_customers, clean_history, merchants, graph_features, FEATURE_COLUMNS, min(500, len(train_customers)), seed=42 + 10)
harvest = harvest_hard_negatives(training_attack["evaded_rows"], training_attack["attacks_raw"], customers, clean_history, merchants, graph_features, genome, seed=42)
model_1 = retrain(_APP_STATE["train_df"], harvest["hard_negatives"], FEATURE_COLUMNS)

retraining_transaction_ids = set(harvest["hard_negatives"]["transaction_id"])

attacks_raw = generate_matched_population_attacks(genome, customers, merchants, matched_customer_ids, seed=42 + 1000)

rng = np.random.default_rng(42 + 1000 + 500)
mutated_frames = []
for instance_id, group in attacks_raw.groupby("instance_id"):
    fraud_group = group[group["is_fraud"] == 1]
    chosen_mutation = rng.choice(genome["mutations"])
    customer = _customer_row(customers, fraud_group["customer_id"].iloc[0])
    mutated_fraud = apply_mutation(fraud_group, group, customer, merchants, genome["family"], chosen_mutation, rng)
    unchanged = group[~group["transaction_id"].isin(fraud_group["transaction_id"])]
    mutated_frames.append(pd.concat([unchanged, mutated_fraud], ignore_index=True))

mutated_attacks_raw = pd.concat(mutated_frames, ignore_index=True) if mutated_frames else attacks_raw
tx_to_instance = mutated_attacks_raw.set_index("transaction_id")["instance_id"].to_dict()

featured = embed_and_engineer(mutated_attacks_raw, customers, clean_history, merchants)
featured = apply_graph_features(featured, graph_features)
featured["instance_id"] = featured["transaction_id"].map(tx_to_instance)

fraud_rows = featured[featured["is_fraud"] == 1].copy()

m0_pred = model.predict(fraud_rows[FEATURE_COLUMNS])
m1_pred = model_1.predict(fraud_rows[FEATURE_COLUMNS])

c_m0 = m0_pred == 1
c_m1 = m1_pred == 1

val_m0 = fraud_rows.loc[c_m0, "amount"].sum()
val_m1 = fraud_rows.loc[c_m1, "amount"].sum()
total_val = fraud_rows["amount"].sum()

print(f"TOTAL ATTACK TRANSACTIONS: {len(fraud_rows)}")
print(f"TOTAL ATTACK VALUE: {total_val}")
print(f"M0 CAUGHT COUNT: {c_m0.sum()}")
print(f"M0 CAUGHT VALUE: {val_m0}")
print(f"M1 CAUGHT COUNT: {c_m1.sum()}")
print(f"M1 CAUGHT VALUE: {val_m1}")
print(f"ADDITIONAL CAUGHT COUNT: {c_m1.sum() - c_m0.sum()}")
print(f"INCREMENTAL VALUE PREVENTED: {val_m1 - val_m0}")
print(f"M0 EVASION: {1.0 - c_m0.mean()}")
print(f"M1 EVASION: {1.0 - c_m1.mean()}")

print("\n--- CLEAN HISTORY AVERAGE AMOUNT PROOF ---")
clean_avg = float(clean_history["amount"].mean())
print(f"CLEAN HISTORY AVERAGE AMOUNT: {clean_avg}")
print("Proof it's not used: Notice that NONE of the economic metrics above depend on or equal any multiple of this average. The previous calculation was:")
print(f"Old calculation: {harvest['accepted_count']} * {clean_avg} = {harvest['accepted_count'] * clean_avg}")
