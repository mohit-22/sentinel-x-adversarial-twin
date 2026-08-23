"""Attack genomes (CLAUDE.md §4.1 — canonical, copied exactly where CLAUDE.md
gives an exact JSON; proposed and explicitly approved by the project owner
where it only gives a prose description, never silently invented).

All 5 attack families are now in scope: micro_structuring (Day 3),
synthetic_identity_drift, behavioral_camouflage, social_engineering_coercion,
and synthetic_voice_authorization (Day 6).
"""

MICRO_STRUCTURING_GENOME = {
    "genome_id": "ATK-MS-001",
    "family": "micro_structuring",
    "objective": "bypass_single_transaction_and_velocity_thresholds",
    "target_amount": 50000.0,
    "parameters": {
        "split_count_range": [10, 15],
        "amount_per_tx_range": [2500, 4800],
        "time_window_hours": 48,
        "recipient_count": 4,
        "inter_arrival_distribution": "exponential",
    },
    "behavioral_camouflage": {
        "interleave_legitimate_noise": True,
        "noise_ratio": 0.35,
        "merchant_diversity": "high",
    },
    "mutations": ["increase_time_spacing", "rotate_mule_accounts", "add_legitimate_micro_purchases"],
}

# CLAUDE.md §4.1 only describes this family in prose ("an account behaves
# normally for ~20 days, then executes a sudden high-velocity extraction via
# a new device and new payee") -- no exact JSON is given, unlike ATK-MS-001.
# Every parameter below was proposed and explicitly approved by the project
# owner (Day 6 planning turn) rather than invented silently.
SYNTHETIC_IDENTITY_DRIFT_GENOME = {
    "genome_id": "ATK-ID-001",
    "family": "synthetic_identity_drift",
    "objective": "bypass_behavioral_baseline_via_dormant_trust_building",
    "parameters": {
        "drift_window_days_range": [18, 22],
        "extraction_transaction_count_range": [3, 6],
        "extraction_window_hours": 4,
        "extraction_amount_multiplier_range": [10, 18],
        "recipient_count": 1,
        "device_count": 1,
    },
    "mutations": ["extend_drift_window", "reduce_extraction_velocity", "reuse_known_device"],
}

# CLAUDE.md §4.1 only describes this family in prose ("fraudulent
# transactions interleaved within authentic-looking spending bursts to
# corrupt short-term anomaly baselines") -- no exact JSON is given. Every
# parameter below was proposed and explicitly approved by the project owner
# (Day 6 planning turn) rather than invented silently.
BEHAVIORAL_CAMOUFLAGE_GENOME = {
    "genome_id": "ATK-BC-001",
    "family": "behavioral_camouflage",
    "objective": "corrupt_short_term_anomaly_baseline_via_authentic_looking_burst",
    "parameters": {
        "burst_window_hours": 12,
        "burst_transaction_count_range": [8, 15],
        "fraud_leg_ratio": 0.3,
        "reuse_customer_device": True,
        "use_real_merchant_and_channel": True,
    },
    "mutations": ["reduce_burst_density", "increase_camouflage_ratio", "match_customer_amount_profile"],
}

# CLAUDE.md §4.1 gives the exact genome_id ("ATK-SC-001") and real-world
# framing (UPI Collect Request scams) for this family but no exact JSON
# parameters -- family/objective/parameters/mutations below were proposed
# and explicitly approved by the project owner (Day 6 planning turn).
SOCIAL_ENGINEERING_COERCION_GENOME = {
    "genome_id": "ATK-SC-001",
    "family": "social_engineering_coercion",
    "objective": "bypass_victim_judgment_via_semantic_coercion",
    "parameters": {
        "coercion_pretext_options": ["kyc_verification", "refund_claim", "cashback_offer", "bill_payment_reminder"],
        "semantic_risk_score_range": [0.7, 0.95],
        "reuse_customer_device": True,
        "amount_source": "customer_normal_distribution",
    },
    "mutations": ["lower_semantic_risk_score", "vary_coercion_pretext", "combine_with_new_device"],
}

# CLAUDE.md §4.1 / SENTINEL_X_ADDENDUM.md Addition 1 give this genome's exact
# JSON verbatim -- copied without modification, unlike families #2-4.
SYNTHETIC_VOICE_AUTHORIZATION_GENOME = {
    "genome_id": "ATK-VD-001",
    "family": "synthetic_voice_authorization",
    "objective": "bypass_step_up_authentication_via_impersonated_voice_call",
    "parameters": {
        "impersonated_role": ["bank_agent", "executive", "family_member"],
        "urgency_score_range": [0.7, 0.95],
        "requests_verification_bypass": True,
        "channel": "voice_authorized",
    },
    "evasion_targets": ["step_up_challenge", "identity_verification"],
    "mutations": ["vary_impersonated_role", "adjust_urgency_score", "combine_with_new_device"],
}
