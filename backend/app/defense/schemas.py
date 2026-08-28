from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any
from app.blue_team.defense_compiler import DefensePolicy
import time

class CertificationRequest(BaseModel):
    attack_family: str
    seed: int = 42
    rounds: int = Field(default=3, le=3, description="Max 3 rounds")
    generations_per_round: int = Field(default=2, le=3, description="Max 3 generations")
    population_size: int = Field(default=5, le=5, description="Max 5 population")
    attack_scale: int = Field(default=50, le=100, description="Max 100 instances")

class DefenseVersion(BaseModel):
    defense_id: str
    version: int
    parent_defense_id: Optional[str] = None
    source_attack_id: Optional[str] = None
    policies: List[DefensePolicy]
    status: str
    created_at: float = Field(default_factory=time.time)
    provenance: str

class DefenseRound(BaseModel):
    certification_id: str
    round_number: int
    defense_id: str
    attack_run_id: str
    attack_family: str
    evasion_rate: float
    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0
    fpr: float = 0.0
    clean_fpr_delta: float = 0.0
    novelty: float = 0.0
    impact_score: float = 0.0
    failure_cause: Optional[str] = None
    candidate_defense_id: Optional[str] = None
    new_defense_created: bool = False
    status: str

class CertificationResult(BaseModel):
    certification_id: str
    status: str
    starting_defense_id: str
    final_defense_id: str
    rounds_completed: int
    initial_evasion: float
    residual_evasion: float
    cumulative_robustness_gain: float
    defense_regression: bool
    clean_fpr_delta: float = 0.0
    f1_regression: float = 0.0
    new_weaknesses_found: List[str]
    customer_leakage: int = 0
    row_leakage: int = 0
    reproducibility_checked: bool = False
    certification_status: str
    rounds: List[DefenseRound]
