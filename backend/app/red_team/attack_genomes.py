"""Attack genomes (CLAUDE.md §4.1 — canonical, copied exactly, never paraphrased).

Only the micro_structuring genome is in scope for the current build phase
(Day 3). The other 4 families are added one at a time in later phases.
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
