import time
from enum import Enum
from typing import Dict, List, Any, Optional
from pydantic import BaseModel, Field

class DifficultyProfile(str, Enum):
    EASY = "EASY"
    HARD = "HARD"
    UNKNOWN = "UNKNOWN"
    EXTREME = "EXTREME"

class ScenarioStatePhase(str, Enum):
    PREPARE = "PREPARE"
    ATTACK = "ATTACK"
    DETECT = "DETECT"
    ADAPT = "ADAPT"
    DISCOVER = "DISCOVER"
    ANALYZE = "ANALYZE"
    DEFEND = "DEFEND"
    SIMULATE = "SIMULATE"
    APPROVE = "APPROVE"
    REPLAY = "REPLAY"
    SCORE = "SCORE"

class JudgeScenario(BaseModel):
    scenario_id: str
    seed: int
    attack_family: str
    attack_scale: int = Field(le=500, description="Max 500 transactions to bound resources")
    difficulty: DifficultyProfile
    adaptive_red_team_enabled: bool
    zero_day_radar_enabled: bool
    defense_compiler_enabled: bool
    human_approval_required: bool
    evolution_generations: int = Field(le=5, description="Max 5 generations to bound runtime")
    created_at: float = Field(default_factory=time.time)

class Scorecard(BaseModel):
    attack_family: str
    initial_evasion: float
    best_evolved_evasion: float
    attack_generations: int
    attack_diversity: float
    
    precision: float
    recall: float
    f1: float
    fpr: float
    
    unknown_detection_rate: float
    false_unknown_rate: float
    cluster_count: int
    
    policy_generated: str
    policy_status: str
    evasion_before: float
    evasion_after: float
    evasion_reduction: float
    # None when a real post-policy simulation ran (a policy was compiled).
    # "no_policy_compiled" when evasion_after falls back to evolved_evasion
    # because no candidate policy exists to simulate -- honest, not silent.
    policy_simulation_note: Optional[str] = None
    
    clean_fpr_delta: float
    legitimate_block_rate: float
    customer_friction_proxy: float
    
    customer_leakage: int
    row_leakage: int
    reproducibility: bool
    
    total_runtime: float
    attack_generation_runtime: float
    detection_runtime: float
    policy_simulation_runtime: float
    
    defense_readiness_score: float

class ScenarioState(BaseModel):
    scenario: JudgeScenario
    current_phase: ScenarioStatePhase
    is_running: bool
    is_completed: bool
    scorecard: Optional[Scorecard] = None
    
    # Internal tracking for UI
    baseline_evasion: float = 0.0
    evolved_evasion: float = 0.0
    latest_genome_id: Optional[str] = None
    radar_novelty: float = 0.0
    radar_clusters: int = 0
    failure_cause: Optional[str] = None
    candidate_policy_id: Optional[str] = None
    policy_status: str = "NONE"
    simulated_evasion_after: float = 0.0
    # Set only when peak evolved evasion is genuinely ~equal to baseline
    # (evasion_difference < 0.005) -- an honest "no improvement found"
    # result, not a bug being papered over. None otherwise.
    evasion_note: Optional[str] = None
