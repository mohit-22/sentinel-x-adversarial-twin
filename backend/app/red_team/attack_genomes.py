"""Attack genomes (CLAUDE.md §4.1 — canonical, copied exactly where CLAUDE.md
gives an exact JSON; proposed and explicitly approved by the project owner
where it only gives a prose description, never silently invented).

micro_structuring (Day 3) and synthetic_identity_drift (Day 6) are in scope
so far. The remaining 3 families are added one at a time in later phases.
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
