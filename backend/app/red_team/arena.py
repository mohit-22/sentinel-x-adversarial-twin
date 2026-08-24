"""Adversarial Arena: attack -> harvest -> retrain -> re-test -> ARG (PRD §7.4, CLAUDE.md §4.3, §6).

Reuses, never reimplements: Day 3's generate_micro_structuring_attacks
(attack_injector.py), Day 4's combine_clean_and_injected/engineer_features
(features.py), apply_graph_features (graph_engine.py), and
train_lightgbm_detector/evaluate_detector/FEATURE_COLUMNS (detector.py).

HONEST BASELINE ASSESSMENT (written here so this reasoning survives context
resets -- do not silently re-derive or contradict it without re-running the
numbers):

  Day 4's M0 already achieves recall=0.947 against this exact un-mutated
  genome, meaning an out-of-the-box Initial Evasion Rate around ~5% against
  a fresh un-mutated batch. There is limited room for a dramatic RAW
  percentage-point drop starting from an already-low baseline, and with
  only a few dozen evaded transactions feeding the evasion-rate ratio, the
  measurement is inherently somewhat noisy.

  The three genome mutations have very UNEVEN expected power given
  Day 4's FEATURE_COLUMNS:
    - increase_time_spacing: expected to matter. Velocity + amount_deviation
      features (sum_7d, sum_24h, amount_deviation_ratio, count_24h/7d, etc.)
      are collectively >50% of M0's total feature importance. Spreading a
      structuring burst's transactions further apart directly suppresses
      these rolling counts/sums, so this mutation should measurably raise
      evasion pre-retrain.
    - rotate_mule_accounts: expected to do close to NOTHING under the
      current feature set. is_new_beneficiary, beneficiary_in_degree, and
      beneficiary_out_degree are deliberately EXCLUDED from FEATURE_COLUMNS
      (see detector.py's documented Day 4 decision -- they're a
      near-deterministic tell against this single un-mutated genome and
      would leave the Arena nothing real to evade). Since the model never
      looks at beneficiary identity, rotating which mule id is used cannot
      change its score. This is a property of today's feature scope, NOT a
      detector weakness -- it will become meaningful again once those
      features are re-added (TODO: after Day 6, multiple attack families).
    - add_legitimate_micro_purchases: expected to be a wash or mildly
      counterproductive for evasion -- it adds MORE rows into the window,
      which if anything raises nearby rolling counts rather than lowering
      them.

  Net expectation: the official randomized-mix ARG will likely be modest
  and possibly noisy; the per-mutation breakdown is where the real,
  reportable signal is expected to show up (increase_time_spacing moving,
  the other two roughly flat).

CORRECTED CROSS-FAMILY FINDING (Day 6.5, after genericizing dispatch and
running all 5 families for the first time -- this SUPERSEDES the narrower
per-family predictions in each family's own docstring in attack_injector.py
regarding which family should be "more" or "less" detectable by M0):

  Per-family predictions made while building families #2-5 reasoned from
  individual feature activity ("is_new_device is active, so this family
  should be more detectable"). That reasoning was incomplete. Measured
  result (n=500, verification pass): micro_structuring's initial evasion
  is ~2-4%, while ALL FOUR other families evade 89-98% -- regardless of
  whether their design touches an "active" feature or not. Diagnosed root
  cause, confirmed with real data: M0 was trained EXCLUSIVELY on
  micro_structuring. Its training fraud rows have is_new_device=0.0 for
  100% of examples (0 out of 4686) -- the tree never saw is_new_device=1
  associated with fraud, so despite that feature technically being "active"
  in FEATURE_COLUMNS, M0 never learned to use it that way (median predicted
  fraud probability on identity_drift's fraud rows: 0.00005). Similarly,
  voice_authorization's amount_deviation_ratio is even MORE extreme than
  micro_structuring's training fraud (median ~20 vs ~1.3) -- the feature
  fires exactly as predicted -- but M0's learned splits apparently require
  high amount_deviation_ratio to CO-OCCUR with high velocity (as it always
  does in micro_structuring's dense-burst training data: median count_24h=7
  there vs. count_24h=0 for a single isolated voice-auth transaction). That
  joint combination is out-of-distribution relative to training, so M0
  still misses it (median predicted probability: 0.00012) despite the
  individual feature being unambiguously elevated.

  Takeaway: feature importance measured in isolation does not predict
  generalization to a structurally different fraud family. A detector
  trained on one attack pattern can fail to recognize a genuinely different
  pattern even when it touches features the model considers "important" --
  because those features were only ever learned in combination with OTHER
  properties (like burst velocity) specific to the training family's shape.
  This is arguably the single most important finding of the whole 5-family
  build: it is the concrete, measured justification for why Sentinel-X's
  adversarial loop (and eventually multi-family retraining) matters, rather
  than a one-time detector being assumed to generalize. See the Day 6.5
  verification run for exact per-family numbers.
"""

import functools
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from app.blue_team.detector import FEATURE_COLUMNS, evaluate_detector, train_lightgbm_detector
from app.blue_team.features import combine_clean_and_injected, engineer_features
from app.blue_team.graph_engine import apply_graph_features
from app.red_team.attack_injector import ATTACK_GENERATORS
from app.simulator.clean_generator import sample_transaction_amounts
from app.core.config import CHANNELS, CHANNEL_PROBS, SIMULATION_DAYS, SIMULATION_START_DATE

# instance_id offset for matched-population re-testing, so freshly-generated
# transaction_id strings can never collide with the initial attack batch's
# instance_id range (0..n_instances-1) even though both batches now share
# the same customers by design.
RETEST_INSTANCE_ID_OFFSET: int = 1_000_000

# Mutation implementation constants -- not in any genome JSON, so documented
# here explicitly rather than buried as magic numbers. Reasonable defaults,
# approved as adjustable-later-if-results-look-off (Day 6.5 planning turn).
TIME_SPACING_MULTIPLIER: float = 2.5        # micro_structuring: increase_time_spacing
N_LEGIT_PURCHASES_TO_ADD: int = 3           # micro_structuring: add_legitimate_micro_purchases
DRIFT_VELOCITY_MULTIPLIER: float = 2.0      # identity_drift: reduce_extraction_velocity, extend_drift_window
CAMOUFLAGE_DENSITY_MULTIPLIER: float = 2.0  # behavioral_camouflage: reduce_burst_density
CAMOUFLAGE_EXTRA_ROWS: int = 3              # behavioral_camouflage: increase_camouflage_ratio
AMOUNT_PROFILE_VARIANCE_FACTOR: float = 0.5  # behavioral_camouflage: match_customer_amount_profile
RISK_SCORE_SCALE_FACTOR: float = 0.5        # social_engineering/voice_authorization: lower/adjust risk score

# field -> customer attribute, for the "customers_own" swap-entity-field pool.
_OWN_POOL_ATTR: Dict[str, str] = {"device_id": "primary_devices", "beneficiary_id": "usual_beneficiaries"}


def _customer_row(customers: pd.DataFrame, customer_id: str) -> pd.Series:
    return customers.loc[customers["customer_id"] == customer_id].iloc[0]


def embed_and_engineer(
    new_rows: pd.DataFrame, customers: pd.DataFrame, clean_history: pd.DataFrame, merchants: pd.DataFrame
) -> pd.DataFrame:
    """Embed freshly-generated/mutated rows into the affected customers' real
    transaction history before feature engineering -- required for velocity
    features to mean anything (approved design decision A).

    new_rows is assumed disjoint from clean_history for the arena's own
    callers (attack instances always get synthetic ids in a separate
    namespace), but that assumption isn't guaranteed for every caller --
    e.g. /detect scoring a transaction_id that already exists in a
    customer's real history. Deduplicating on transaction_id here (keeping
    the new_rows copy, since that's the row actually being scored/embedded)
    makes the function correct regardless of that overlap, instead of
    silently fanning out into duplicate feature rows.
    """
    affected_customers = new_rows["customer_id"].unique()
    relevant_clean_history = clean_history[clean_history["customer_id"].isin(affected_customers)]
    combined = combine_clean_and_injected(relevant_clean_history, new_rows)
    combined = combined.drop_duplicates(subset="transaction_id", keep="last")
    return engineer_features(combined, customers)


def run_attack(
    genome: Dict,
    model,
    customers: pd.DataFrame,
    clean_history: pd.DataFrame,
    merchants: pd.DataFrame,
    graph_features: Dict,
    feature_columns: List[str] = FEATURE_COLUMNS,
    n_instances: int = 500,
    seed: int = 42,
) -> Dict:
    """Generate a fresh un-mutated attack batch, score it with `model`, and
    return the evasion rate plus the evaded rows (with instance_id intact).
    Dispatches to the correct family's generator via ATTACK_GENERATORS.
    """
    attacks_fn = ATTACK_GENERATORS[genome["family"]]["attacks_fn"]
    attacks_raw = attacks_fn(genome, customers, merchants, n_instances, seed=seed)
    tx_to_instance = attacks_raw.set_index("transaction_id")["instance_id"].to_dict()

    featured = embed_and_engineer(attacks_raw, customers, clean_history, merchants)
    featured = apply_graph_features(featured, graph_features)
    featured["instance_id"] = featured["transaction_id"].map(tx_to_instance)

    fraud_rows = featured[featured["is_fraud"] == 1].copy()
    y_pred = model.predict(fraud_rows[feature_columns])
    evaded_mask = y_pred == 0
    evaded_rows = fraud_rows[evaded_mask].copy()

    return {
        "evasion_rate": float(evaded_mask.mean()),
        "total_fraud": int(len(fraud_rows)),
        "false_negatives": int(evaded_mask.sum()),
        "evaded_rows": evaded_rows,
        "attacks_raw": attacks_raw,
        "customer_ids_used": attacks_raw["customer_id"].unique(),
        "instance_ids_used": attacks_raw["instance_id"].unique(),
    }


def generate_matched_population_attacks(
    genome: Dict,
    customers: pd.DataFrame,
    merchants: pd.DataFrame,
    customer_ids: np.ndarray,
    seed: int,
    instance_id_offset: int = RETEST_INSTANCE_ID_OFFSET,
) -> pd.DataFrame:
    """One fresh instance per customer in `customer_ids` (a FIXED,
    caller-provided population -- not a random draw), reusing the family's
    own per-instance generator (via ATTACK_GENERATORS) directly rather than
    reimplementing it. Each family's own `generate_<family>_attacks`
    always samples customers randomly internally and can't be pointed at a
    specific population, so this orchestration loop (matching that logic's
    timing-placement approach, just over a fixed customer list) lives here
    in arena.py instead -- attack_injector.py's individual family functions
    are not touched.

    Timing placement mirrors each family's own convention: identity_drift
    anchors to a drift-window offset from SIMULATION_START_DATE (matching
    its own generate_identity_drift_attacks); every other family places the
    instance at a random offset across the full 30-day simulation.

    instance_id is offset well clear of any n_instances range used for the
    initial attack batch, so transaction_id strings can never collide
    between the two batches even though they now share customers.
    """
    np.random.seed(seed)
    family = genome["family"]
    instance_fn = ATTACK_GENERATORS[family]["instance_fn"]
    n_instances = len(customer_ids)
    base = pd.Timestamp(SIMULATION_START_DATE)

    if family == "synthetic_identity_drift":
        offsets_days = np.random.uniform(
            genome["parameters"]["drift_window_days_range"][0],
            genome["parameters"]["drift_window_days_range"][1],
            size=n_instances,
        )
    elif "time_window_hours" in genome["parameters"] or "burst_window_hours" in genome["parameters"]:
        window_hours = genome["parameters"].get("time_window_hours", genome["parameters"].get("burst_window_hours"))
        max_start_offset_days = max(SIMULATION_DAYS - window_hours / 24.0, 0.0)
        offsets_days = np.random.uniform(0, max_start_offset_days, size=n_instances)
    else:
        # Single-transaction families (social_engineering_coercion,
        # synthetic_voice_authorization): no window to fit, place anywhere.
        offsets_days = np.random.uniform(0, SIMULATION_DAYS, size=n_instances)

    customers_indexed = customers.set_index("customer_id")
    instances = []
    for i, (customer_id, offset) in enumerate(zip(customer_ids, offsets_days)):
        customer = customers_indexed.loc[customer_id].copy()
        customer["customer_id"] = customer_id  # restore -- dropped by set_index
        start_time = base + pd.to_timedelta(offset, unit="D")
        instance_id = instance_id_offset + i
        instance_df = instance_fn(
            genome, customer, merchants, instance_id=instance_id, start_time=start_time, seed=seed + i + 1
        )
        instance_df["instance_id"] = instance_id  # idempotent for families that already set it internally
        instances.append(instance_df)

    return pd.concat(instances, ignore_index=True)


# --- Day 6.5: generic mutation transforms + per-(family, mutation) registry
#
# Every mutation across all 5 families collapses into one of 6 reusable
# shapes. Adding a 6th family later means one registry line reusing an
# existing generic, or one new generic if truly novel -- not a growing
# if/elif chain.


def _stretch_timing(rows, sibling_rows, customer, merchants, rng, multiplier):
    """Scale each row's offset from the instance's start time by
    `multiplier` -- a positive linear scaling, so relative order among the
    instance's own rows is preserved by construction.
    """
    mutated = rows.copy()
    instance_start = sibling_rows["timestamp"].min()
    offset = mutated["timestamp"] - instance_start
    mutated["timestamp"] = instance_start + offset * multiplier
    return mutated


def _swap_entity_field(rows, sibling_rows, customer, merchants, rng, field, pool):
    """Replace `field` (device_id or beneficiary_id) via a pool strategy:
    - "recycled_pool": remap this instance's distinct original values to a
      shared "recycled" replacement pool (consistent mapping across rows).
    - "customers_own": draw from the customer's own known entities for that
      field (the opposite direction of recycled_pool -- de-novelizes it).
    - "brand_new": a never-before-seen synthetic id, unique per row.
    """
    mutated = rows.copy()
    if pool == "recycled_pool":
        original_values = sibling_rows[field].unique()
        recycled = np.array([f"MUTATED-RECYCLED-{field.upper()}-{i}" for i in range(len(original_values))])
        mapping = dict(zip(original_values, recycled))
        mutated[field] = mutated[field].map(mapping)
    elif pool == "customers_own":
        own_pool = np.array(customer[_OWN_POOL_ATTR[field]])
        mutated[field] = rng.choice(own_pool, size=len(mutated))
    elif pool == "brand_new":
        mutated[field] = [f"MUTATED-{field.upper()}-{tx}" for tx in mutated["transaction_id"]]
    else:
        raise ValueError(f"unknown swap-entity-field pool strategy: {pool}")
    return mutated


def _add_extra_rows(rows, sibling_rows, customer, merchants, rng, n_new):
    """The given row(s) are left untouched; n_new new legitimate-looking
    rows (customer's own normal spending pattern, reusing clean_generator's
    amount sampler) are added as camouflage near them.
    """
    anchor_time = rows["timestamp"].iloc[0]
    anchor_tx_id = rows["transaction_id"].iloc[0]
    offsets_hours = rng.uniform(-2, 2, size=n_new)
    new_timestamps = anchor_time + pd.to_timedelta(offsets_hours, unit="h")

    mean_spend = np.full(n_new, customer["mean_spend"])
    spend_variance = np.full(n_new, customer["spend_variance"])
    amounts = sample_transaction_amounts(mean_spend, spend_variance, seed=int(rng.integers(0, 1_000_000)))

    usual_merchants = np.array(customer["usual_merchants"])
    merchant_id = rng.choice(usual_merchants, size=n_new)
    merchant_category_map = merchants.set_index("merchant_id")["merchant_category"]
    merchant_category = merchant_category_map.reindex(merchant_id).to_numpy()

    primary_devices = np.array(customer["primary_devices"])
    device_id = rng.choice(primary_devices, size=n_new)
    usual_beneficiaries = np.array(customer["usual_beneficiaries"])
    beneficiary_id = rng.choice(usual_beneficiaries, size=n_new)
    channel = rng.choice(CHANNELS, size=n_new, p=CHANNEL_PROBS)

    new_rows = pd.DataFrame(
        {
            "transaction_id": [f"{anchor_tx_id}-LEGIT{i}" for i in range(n_new)],
            "timestamp": new_timestamps,
            "customer_id": customer["customer_id"],
            "merchant_id": merchant_id,
            "beneficiary_id": beneficiary_id,
            "amount": amounts,
            "currency": "INR",
            "channel": channel,
            "device_id": device_id,
            "ip_region": customer["base_location"],
            "location": customer["base_location"],
            "merchant_category": merchant_category,
            "semantic_risk_score": 0.0,
            "voice_confidence_score": 1.0,
            "is_fraud": 0,
            "attack_family": None,
            "genome_id": None,
        }
    )
    return pd.concat([rows.copy(), new_rows], ignore_index=True)


def _scale_risk_field(rows, sibling_rows, customer, merchants, rng, field, factor):
    """Multiply a risk float field (semantic_risk_score) by `factor`."""
    mutated = rows.copy()
    mutated[field] = mutated[field] * factor
    return mutated


def _narrow_amount(rows, sibling_rows, customer, merchants, rng, variance_factor):
    """Re-draw amount from a TIGHTER slice of the customer's own
    distribution (spend_variance scaled down by variance_factor) -- makes
    an already-camouflaged amount even closer to the customer's typical
    spend.
    """
    mutated = rows.copy()
    n = len(mutated)
    mean_spend = np.full(n, customer["mean_spend"])
    spend_variance = np.full(n, customer["spend_variance"] * variance_factor)
    mutated["amount"] = sample_transaction_amounts(mean_spend, spend_variance, seed=int(rng.integers(0, 1_000_000)))
    return mutated


def _no_op(rows, sibling_rows, customer, merchants, rng):
    """Some mutations (vary_coercion_pretext, vary_impersonated_role) have
    no corresponding schema field -- there is genuinely nothing to change
    in the row. Returning it unchanged is the honest implementation, not a
    missing feature.
    """
    return rows.copy()


# (family, mutation_name) -> handler. 15 entries: 5 families x 3 mutations
# each (combine_with_new_device is shared by 2 families but needs 2 keys).
#
# KNOWN LIMITATION (candidate for the docx's limitations/future-work
# section): synthetic_identity_drift's extend_drift_window conceptually
# means "lengthen the ~20-day trust-building period," which is a genome
# PARAMETER (drift_window_days_range) applied before generation, not a
# property of already-generated rows -- unlike every other mutation here.
# apply_mutation's interface (rows, sibling_rows, customer, merchants, rng)
# has no way to re-invoke generate_identity_drift_instance with a different
# genome. Approximated here as a timing shift (_stretch_timing, same as
# reduce_extraction_velocity) per the approved Option (a). A proper
# implementation would need apply_mutation to access and re-invoke the
# genome generator directly, which was judged out of scope for this pass.
MUTATION_REGISTRY: Dict[Tuple[str, str], Callable] = {
    ("micro_structuring", "increase_time_spacing"):
        functools.partial(_stretch_timing, multiplier=TIME_SPACING_MULTIPLIER),
    ("micro_structuring", "rotate_mule_accounts"):
        functools.partial(_swap_entity_field, field="beneficiary_id", pool="recycled_pool"),
    ("micro_structuring", "add_legitimate_micro_purchases"):
        functools.partial(_add_extra_rows, n_new=N_LEGIT_PURCHASES_TO_ADD),

    ("synthetic_identity_drift", "extend_drift_window"):
        functools.partial(_stretch_timing, multiplier=DRIFT_VELOCITY_MULTIPLIER),  # approximation, see above
    ("synthetic_identity_drift", "reduce_extraction_velocity"):
        functools.partial(_stretch_timing, multiplier=DRIFT_VELOCITY_MULTIPLIER),
    ("synthetic_identity_drift", "reuse_known_device"):
        functools.partial(_swap_entity_field, field="device_id", pool="customers_own"),

    ("behavioral_camouflage", "reduce_burst_density"):
        functools.partial(_stretch_timing, multiplier=CAMOUFLAGE_DENSITY_MULTIPLIER),
    ("behavioral_camouflage", "increase_camouflage_ratio"):
        functools.partial(_add_extra_rows, n_new=CAMOUFLAGE_EXTRA_ROWS),
    ("behavioral_camouflage", "match_customer_amount_profile"):
        functools.partial(_narrow_amount, variance_factor=AMOUNT_PROFILE_VARIANCE_FACTOR),

    ("social_engineering_coercion", "lower_semantic_risk_score"):
        functools.partial(_scale_risk_field, field="semantic_risk_score", factor=RISK_SCORE_SCALE_FACTOR),
    ("social_engineering_coercion", "vary_coercion_pretext"): _no_op,
    ("social_engineering_coercion", "combine_with_new_device"):
        functools.partial(_swap_entity_field, field="device_id", pool="brand_new"),

    ("synthetic_voice_authorization", "vary_impersonated_role"): _no_op,
    ("synthetic_voice_authorization", "adjust_urgency_score"):
        functools.partial(_scale_risk_field, field="semantic_risk_score", factor=RISK_SCORE_SCALE_FACTOR),
    ("synthetic_voice_authorization", "combine_with_new_device"):
        functools.partial(_swap_entity_field, field="device_id", pool="brand_new"),
}


def apply_mutation(
    evaded_group: pd.DataFrame,
    sibling_rows: pd.DataFrame,
    customer: pd.Series,
    merchants: pd.DataFrame,
    family: str,
    mutation_name: str,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Dispatch to the registered handler for (family, mutation_name),
    applied consistently to every given row of one instance. Replaces the
    old hardcoded if/elif chain that only understood micro_structuring's
    three mutation names.
    """
    handler = MUTATION_REGISTRY.get((family, mutation_name))
    if handler is None:
        raise ValueError(f"no mutation handler registered for (family={family!r}, mutation={mutation_name!r})")
    return handler(evaded_group, sibling_rows, customer, merchants, rng)


def validate_mutation(
    original_row: pd.Series,
    mutated_row: pd.Series,
    customer: pd.Series,
    prior_timestamp: Optional[pd.Timestamp],
    next_timestamp: Optional[pd.Timestamp],
) -> Dict[str, bool]:
    """The three CLAUDE.md §4.3 checks, independently recomputed (not
    trusted from the pipeline) so this is a real gate, not a rubber stamp.
    """
    amount_ok = bool(0 <= mutated_row["amount"] <= customer["mean_spend"] * 20)

    lower_ok = prior_timestamp is None or mutated_row["timestamp"] > prior_timestamp
    upper_ok = next_timestamp is None or mutated_row["timestamp"] < next_timestamp
    chronological_ok = bool(lower_ok and upper_ok)

    expected_is_new_device = int(mutated_row["device_id"] not in customer["primary_devices"])
    expected_is_new_beneficiary = int(mutated_row["beneficiary_id"] not in customer["usual_beneficiaries"])
    novelty_ok = bool(
        mutated_row["is_new_device"] == expected_is_new_device
        and mutated_row["is_new_beneficiary"] == expected_is_new_beneficiary
    )

    return {
        "amount_ok": amount_ok,
        "chronological_ok": chronological_ok,
        "novelty_ok": novelty_ok,
        "valid": amount_ok and chronological_ok and novelty_ok,
    }


def harvest_hard_negatives(
    evaded_rows: pd.DataFrame,
    attacks_raw: pd.DataFrame,
    customers: pd.DataFrame,
    clean_history: pd.DataFrame,
    merchants: pd.DataFrame,
    graph_features: Dict,
    genome: Dict,
    seed: int = 42,
) -> Dict:
    """One randomly-chosen mutation per evaded INSTANCE (approved decision C
    -- not per individual transaction), applied consistently to every evaded
    row of that instance. Rejects and discards any mutated fraud row that
    fails validate_mutation.
    """
    rng = np.random.default_rng(seed)
    accepted_rows: List[pd.Series] = []
    rejected_count = 0
    accepted_count = 0
    mutation_log = []

    for instance_id, group in evaded_rows.groupby("instance_id"):
        sibling_rows = attacks_raw[attacks_raw["instance_id"] == instance_id]
        customer_id = group["customer_id"].iloc[0]
        customer = _customer_row(customers, customer_id)
        mutation_name = rng.choice(genome["mutations"])

        instance_start = sibling_rows["timestamp"].min()
        instance_end = sibling_rows["timestamp"].max()
        cust_clean = clean_history[clean_history["customer_id"] == customer_id]

        mutated_instance_df = apply_mutation(group, sibling_rows, customer, merchants, genome["family"], mutation_name, rng)
        unchanged_siblings = sibling_rows[~sibling_rows["transaction_id"].isin(group["transaction_id"])]
        full_instance_df = pd.concat([unchanged_siblings, mutated_instance_df], ignore_index=True)

        combined = combine_clean_and_injected(cust_clean, full_instance_df)
        featured = engineer_features(combined, customers)
        featured = apply_graph_features(featured, graph_features)

        # Full customer timeline (clean history + this instance, post-mutation)
        # for correctly computing each mutated row's true chronological neighbors.
        timeline = pd.concat(
            [cust_clean[["transaction_id", "timestamp"]], full_instance_df[["transaction_id", "timestamp"]]]
        ).sort_values("timestamp")

        mutated_tx_ids = set(mutated_instance_df["transaction_id"])
        mutated_engineered = featured[featured["transaction_id"].isin(mutated_tx_ids)]

        for _, mrow in mutated_engineered[mutated_engineered["is_fraud"] == 1].iterrows():
            others = timeline[timeline["transaction_id"] != mrow["transaction_id"]]
            prior_candidates = others.loc[others["timestamp"] < mrow["timestamp"], "timestamp"]
            next_candidates = others.loc[others["timestamp"] > mrow["timestamp"], "timestamp"]
            prior_timestamp = prior_candidates.max() if len(prior_candidates) else None
            next_timestamp = next_candidates.min() if len(next_candidates) else None

            orig_match = group[group["transaction_id"] == mrow["transaction_id"]]
            original_row = orig_match.iloc[0] if len(orig_match) else mrow

            validation = validate_mutation(original_row, mrow, customer, prior_timestamp, next_timestamp)
            mutation_log.append({"instance_id": instance_id, "mutation": mutation_name, **validation})
            if validation["valid"]:
                accepted_rows.append(mrow)
                accepted_count += 1
            else:
                rejected_count += 1

        # Legitimate camouflage rows (add_legitimate_micro_purchases) are not
        # fraud mutations -- §4.3 validation applies to mutated fraud examples.
        for _, lrow in mutated_engineered[mutated_engineered["is_fraud"] == 0].iterrows():
            accepted_rows.append(lrow)

    hard_negatives = pd.DataFrame(accepted_rows) if accepted_rows else evaded_rows.iloc[0:0].copy()
    return {
        "hard_negatives": hard_negatives,
        "accepted_count": accepted_count,
        "rejected_count": rejected_count,
        "mutation_log": mutation_log,
    }


def retrain(original_train_df: pd.DataFrame, hard_negatives: pd.DataFrame, feature_columns: List[str] = FEATURE_COLUMNS):
    """Full retrain (not incremental) on train data augmented with hard negatives.

    TODO (candidate for the docx's limitations/future-work section): plain
    concatenation trades precision for recall. Measured at n=2000: adding
    947 hard negatives (~20% on top of the original 4,686 fraud training
    rows) raised recall/lowered evasion (the ARG headline result) but
    dropped retrained_f1_score 0.906 -> 0.702 on Day 4's original held-out
    set (precision 0.867 -> lower, FPR roughly tripled) -- the decision
    boundary shifted meaningfully more permissive. A future pass could
    rebalance this via class-weighted retraining (upweight hard negatives
    less aggressively than 1:1) or by capping the hard-negative-to-
    original-fraud ratio rather than adding all harvested examples
    unconditionally.
    """
    augmented = pd.concat([original_train_df, hard_negatives], ignore_index=True)
    return train_lightgbm_detector(augmented, feature_columns)


def re_test(
    genome: Dict,
    model,
    customers: pd.DataFrame,
    clean_history: pd.DataFrame,
    merchants: pd.DataFrame,
    graph_features: Dict,
    customer_ids: np.ndarray,
    exclude_transaction_ids: set,
    mutation_name: Optional[str] = None,
    feature_columns: List[str] = FEATURE_COLUMNS,
    seed: int = 42,
) -> Dict:
    """Fresh mutated batch over a MATCHED customer population (the exact
    same customers as the initial attack, controlling for the population-
    variance confound demonstrated in this module's docstring) -- while
    still using genuinely NEW attack instances (different amounts,
    timestamps, mule ids; see generate_matched_population_attacks), never
    the literal transaction rows that fed hard-negative retraining. That
    disjointness is verified below, not assumed.

    If mutation_name is None, applies the same randomized
    one-of-three-per-instance mix as harvesting (the official,
    non-cherry-picked run); otherwise applies that single mutation to every
    instance (the per-mutation-type breakdown runs).
    """
    attacks_raw = generate_matched_population_attacks(genome, customers, merchants, customer_ids, seed=seed)

    overlap = set(attacks_raw["transaction_id"]) & exclude_transaction_ids
    assert not overlap, (
        f"re_test batch contains {len(overlap)} transaction_ids that fed hard-negative retraining: "
        f"{list(overlap)[:5]}"
    )

    rng = np.random.default_rng(seed + 500)
    mutated_frames = []
    for instance_id, group in attacks_raw.groupby("instance_id"):
        fraud_group = group[group["is_fraud"] == 1]
        chosen_mutation = mutation_name or rng.choice(genome["mutations"])
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
    y_pred = model.predict(fraud_rows[feature_columns])
    evaded_mask = y_pred == 0

    return {
        "evasion_rate": float(evaded_mask.mean()),
        "total_fraud": int(len(fraud_rows)),
        "false_negatives": int(evaded_mask.sum()),
        "customer_ids_used": mutated_attacks_raw["customer_id"].unique(),
        "instance_ids_used": mutated_attacks_raw["instance_id"].unique(),
        "transaction_ids_used": set(mutated_attacks_raw["transaction_id"]),
    }


def compute_arg(initial_rate: float, final_rate: float) -> float:
    """ARG (%) = ((Initial Evasion Rate - Final Evasion Rate) / Initial Evasion Rate) x 100 (CLAUDE.md §6, exact)."""
    if initial_rate == 0:
        raise ValueError("initial_rate is 0 -- ARG is undefined (division by zero), not computable as a percentage")
    return ((initial_rate - final_rate) / initial_rate) * 100


def run_arena_mvp_gate(
    genome: Dict,
    model,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    customers: pd.DataFrame,
    clean_history: pd.DataFrame,
    merchants: pd.DataFrame,
    graph_features: Dict,
    feature_columns: List[str] = FEATURE_COLUMNS,
    n_instances: int = 2000,
    seed: int = 42,
) -> Dict:
    """Full MVP-gate loop. Returns an ArenaRunSummary-shaped dict (CLAUDE.md
    §6 fields) plus an added `mutation_breakdown` key -- schemas.py is not
    in Day 5's ALLOWED_TO_TOUCH, so this stays a plain dict rather than a
    Pydantic model extension for now.

    MEASURED CONFOUND, CONTROLLED FOR (read before touching this function):
    an earlier version measured initial_evasion_rate and final_evasion_rate
    on DIFFERENT random customer populations (re_test excluded the initial
    batch's customers entirely). That is methodologically clean for
    hold-out purposes but confounds "did retraining help" with "which
    random customers got drawn": empirically, M0's evasion rate on the same
    un-mutated genome swung from 2.2% to 4.1% across two different random
    500-customer seeds -- comparable in magnitude to several of the
    mutation effects. Fix: re_test now uses generate_matched_population_
    attacks to test the SAME customers as the initial attack, with
    genuinely fresh instances (different amounts/timestamps/mule ids, a
    disjoint instance_id namespace) -- never the literal rows used for
    hard-negative retraining (verified via exclude_transaction_ids, not
    assumed). This isolates the retraining effect while still holding out
    real transactions. n_instances defaults to 2000 (up from 500): at ~2%
    base evasion, 500 instances yields too few evaded examples for a
    stable measurement.
    """
    initial = run_attack(genome, model, customers, clean_history, merchants, graph_features, feature_columns, n_instances, seed)
    harvest = harvest_hard_negatives(
        initial["evaded_rows"], initial["attacks_raw"], customers, clean_history, merchants, graph_features, genome, seed
    )
    model_1 = retrain(train_df, harvest["hard_negatives"], feature_columns)

    retraining_transaction_ids = set(harvest["hard_negatives"]["transaction_id"])
    matched_customer_ids = initial["customer_ids_used"]

    official_final = re_test(
        genome, model_1, customers, clean_history, merchants, graph_features,
        customer_ids=matched_customer_ids, exclude_transaction_ids=retraining_transaction_ids,
        mutation_name=None, feature_columns=feature_columns, seed=seed + 1000,
    )
    official_arg = compute_arg(initial["evasion_rate"], official_final["evasion_rate"])

    mutation_breakdown = {}
    for mutation_index, mutation_name in enumerate(genome["mutations"]):
        m_final = re_test(
            genome, model_1, customers, clean_history, merchants, graph_features,
            customer_ids=matched_customer_ids, exclude_transaction_ids=retraining_transaction_ids,
            mutation_name=mutation_name, feature_columns=feature_columns, seed=seed + 2000 + mutation_index * 100,
        )
        mutation_breakdown[mutation_name] = {
            "initial_evasion_rate": initial["evasion_rate"],
            "final_evasion_rate": m_final["evasion_rate"],
            "robustness_gain": compute_arg(initial["evasion_rate"], m_final["evasion_rate"]),
            "final_false_negatives": m_final["false_negatives"],
            "final_total_fraud": m_final["total_fraud"],
        }

    return {
        "run_id": f"arena-{genome['genome_id']}-{seed}",
        "attack_family": genome["family"],
        "initial_evasion_rate": initial["evasion_rate"],
        "final_evasion_rate": official_final["evasion_rate"],
        "robustness_gain": official_arg,
        "hard_examples_count": harvest["accepted_count"],
        # M1's F1 on Day 4's ORIGINAL held-out test set -- confirms retraining
        # on hard negatives didn't regress general performance.
        "retrained_f1_score": evaluate_detector(model_1, test_df, feature_columns)["f1"],
        "mutation_breakdown": mutation_breakdown,
        "_diagnostics": {
            "initial_false_negatives": initial["false_negatives"],
            "initial_total_fraud": initial["total_fraud"],
            "official_final_false_negatives": official_final["false_negatives"],
            "official_final_total_fraud": official_final["total_fraud"],
            "harvest_accepted_count": harvest["accepted_count"],
            "harvest_rejected_count": harvest["rejected_count"],
            "matched_population_size": len(matched_customer_ids),
            "initial_customer_ids": initial["customer_ids_used"],
            "final_customer_ids": official_final["customer_ids_used"],
            "populations_matched": bool(set(official_final["customer_ids_used"]) == set(matched_customer_ids)),
            "retest_disjoint_from_retraining_transactions": bool(
                official_final["transaction_ids_used"].isdisjoint(retraining_transaction_ids)
            ),
        },
    }


def run_arena_for_all_families(
    genomes: List[Dict],
    model,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    customers: pd.DataFrame,
    clean_history: pd.DataFrame,
    merchants: pd.DataFrame,
    graph_features: Dict,
    feature_columns: List[str] = FEATURE_COLUMNS,
    n_instances: int = 2000,
    seed: int = 42,
) -> Dict[str, Dict]:
    """Run the full MVP-gate loop independently for each genome, keyed by
    family. For Day 7's Command Center dashboard, which needs a per-family
    ARG/evasion summary. Each family gets its own M1 (retrained from the
    SAME M0/train_df, independently) -- families are not combined into one
    retraining pass.
    """
    return {
        genome["family"]: run_arena_mvp_gate(
            genome, model, train_df, test_df, customers, clean_history, merchants, graph_features,
            feature_columns=feature_columns, n_instances=n_instances, seed=seed,
        )
        for genome in genomes
    }


def run_multi_family_hardening(
    genomes: List[Dict],
    model,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    customers: pd.DataFrame,
    clean_history: pd.DataFrame,
    merchants: pd.DataFrame,
    graph_features: Dict,
    feature_columns: List[str] = FEATURE_COLUMNS,
    n_instances: int = 500,
    seed: int = 42,
) -> Dict:
    """Cross-Family Generalization Matrix (post-Day 8b differentiator).

    Unlike run_arena_for_all_families (each family gets its OWN independent
    M1), this harvests hard negatives from ALL given genomes, combines them
    into ONE retrain, then re-tests each family independently against the
    single resulting multi-family-hardened model (M-multi) -- turning the
    Day 6.5 finding (M0, trained only on micro_structuring, evades 89-98%
    on the other four families) into a measured before/after.

    Reuses run_attack / harvest_hard_negatives / retrain / re_test /
    compute_arg exactly as they exist -- no changes to any of them, only
    new orchestration over multiple genomes with one shared retrain in the
    middle. Same category of addition as run_arena_for_all_families itself
    (Day 6.5): every other function in this file is untouched.

    n_instances defaults to 500, not the official 2000: harvesting cost
    scales with evaded-instance count, and 4 of the 5 families evade
    89-98% of a 2000-instance batch (measured at the planning turn) --
    harvesting all 5 families at n=2000 costs roughly 8-10 minutes. 500 is
    a real, non-cherry-picked, reduced-scale run (~2 minutes); n_instances
    remains overridable up to 2000 for the full official run.
    """
    per_family_attack: Dict[str, Dict] = {}
    all_hard_negatives: List[pd.DataFrame] = []

    for genome in genomes:
        attack = run_attack(
            genome, model, customers, clean_history, merchants, graph_features,
            feature_columns, n_instances, seed,
        )
        harvest = harvest_hard_negatives(
            attack["evaded_rows"], attack["attacks_raw"], customers, clean_history,
            merchants, graph_features, genome, seed,
        )
        per_family_attack[genome["family"]] = {"attack": attack, "harvest": harvest}
        all_hard_negatives.append(harvest["hard_negatives"])

    combined_hard_negatives = pd.concat(all_hard_negatives, ignore_index=True)
    model_multi = retrain(train_df, combined_hard_negatives, feature_columns)

    per_family_result: Dict[str, Dict] = {}
    for genome in genomes:
        family = genome["family"]
        attack = per_family_attack[family]["attack"]
        harvest = per_family_attack[family]["harvest"]
        retraining_transaction_ids = set(harvest["hard_negatives"]["transaction_id"])

        final = re_test(
            genome, model_multi, customers, clean_history, merchants, graph_features,
            customer_ids=attack["customer_ids_used"], exclude_transaction_ids=retraining_transaction_ids,
            mutation_name=None, feature_columns=feature_columns, seed=seed + 1000,
        )
        per_family_result[family] = {
            "genome_id": genome["genome_id"],
            "initial_evasion_rate": attack["evasion_rate"],
            "final_evasion_rate": final["evasion_rate"],
            "robustness_gain": compute_arg(attack["evasion_rate"], final["evasion_rate"]),
            "hard_examples_count": harvest["accepted_count"],
        }

    return {
        "model": model_multi,
        "per_family": per_family_result,
        # Sum of each family's accepted_count (validated fraud rows only),
        # NOT len(combined_hard_negatives) -- harvest_hard_negatives also
        # folds in non-fraud "legitimate camouflage" rows for certain
        # mutations (e.g. add_legitimate_micro_purchases) without counting
        # them in accepted_count (existing, correct behavior -- see its own
        # docstring). Using len() here would silently disagree with the sum
        # of the per-family hard_examples_count values shown right next to
        # it, which use the same accepted_count convention as
        # run_arena_mvp_gate's existing hard_examples_count field. Caught by
        # test_run_multi_family_hardening_total_hard_examples_is_sum_across_
        # families before this was ever displayed anywhere.
        "total_hard_examples_count": sum(data["hard_examples_count"] for data in per_family_result.values()),
        # M-multi's precision/recall/F1/FPR on Day 4's ORIGINAL held-out
        # test set -- confirms (or reveals a cost to) whether multi-family
        # hardening regresses general performance, same check
        # run_arena_mvp_gate already runs for the single-family case.
        "retrained_metrics": evaluate_detector(model_multi, test_df, feature_columns),
    }
