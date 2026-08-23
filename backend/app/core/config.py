"""Single source of truth for seeds and dataset-size constants (CLAUDE.md §0.6, §9).

Only the constants needed by the current build phase (synthetic payment world)
are defined here. Decision thresholds and other later-phase config are added
when their owning phase starts.
"""

SEED: int = 42

N_CUSTOMERS: int = 10_000
N_MERCHANTS: int = 500
N_TRANSACTIONS: int = 50_000
SIMULATION_DAYS: int = 30
SIMULATION_START_DATE: str = "2026-01-01T00:00:00"

# Entity persistence floor per PRD §7.1 ("> 0.95"); 0.97 gives headroom above it.
ENTITY_PERSISTENCE_PROB: float = 0.97

# Income tiers: (population share, median monthly spend in INR).
INCOME_TIERS: dict = {
    "mass": {"share": 0.60, "median_spend": 800.0},
    "affluent": {"share": 0.30, "median_spend": 3000.0},
    "premium": {"share": 0.10, "median_spend": 12000.0},
}

MERCHANTS_PER_CUSTOMER_RANGE: tuple = (3, 10)
BENEFICIARIES_PER_CUSTOMER_RANGE: tuple = (2, 6)

# Fidelity target per PRD §7.1 ("Target > 90% similarity").
FIDELITY_SIMILARITY_TARGET: float = 0.90

# Customer-level transaction-amount coefficient of variation (spend_variance
# derived as (mean_spend * TRANSACTION_CV) ** 2). Not specified in the PRD;
# a documented modeling assumption.
TRANSACTION_CV: float = 0.5

# Diurnal hour-of-day sampling weight: daytime (09:00-21:00) vs overnight.
DIURNAL_DAY_HOURS: tuple = tuple(range(9, 21))
DIURNAL_DAY_WEIGHT: float = 4.0
DIURNAL_NIGHT_WEIGHT: float = 1.0

# Clean-transaction channel mix (voice_authorized is red-team-only, not
# generated here).
CHANNELS: tuple = ("POS", "WEB", "P2P")
CHANNEL_PROBS: tuple = (0.40, 0.35, 0.25)

# Probability a transaction's location deviates from the customer's base_location.
NOVEL_LOCATION_PROB: float = 0.05

MERCHANT_CATEGORIES: tuple = (
    "grocery", "electronics", "dining", "travel", "fuel",
    "utilities", "fashion", "pharmacy", "entertainment", "other",
)
