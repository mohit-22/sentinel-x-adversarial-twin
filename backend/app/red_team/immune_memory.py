import time
from typing import Dict, List, Optional
from pydantic import BaseModel, Field

class MemoryRecord(BaseModel):
    memory_id: str
    attack_family: str
    genome_id: str
    genome: Dict
    parent_attack_id: Optional[str]
    generation: int
    first_seen: float = Field(default_factory=time.time)
    last_seen: float = Field(default_factory=time.time)
    initial_evasion: float
    best_evasion: float
    defense_version: str
    current_status: str
    residual_evasion: float
    novelty_score: float
    realism_score: float
    provenance: str = "training" # 'training' vs 'evaluation'. Memory used for retraining MUST be 'training' only.

class ImmuneMemoryStore:
    def __init__(self):
        self.records: Dict[str, MemoryRecord] = {}
        
    def add_record(self, record: MemoryRecord):
        if record.memory_id in self.records:
            existing = self.records[record.memory_id]
            # Update best evasion if necessary, and last_seen
            if record.best_evasion > existing.best_evasion:
                existing.best_evasion = record.best_evasion
            existing.last_seen = time.time()
            existing.current_status = record.current_status
            existing.residual_evasion = record.residual_evasion
            existing.defense_version = record.defense_version
            self.records[record.memory_id] = existing
        else:
            self.records[record.memory_id] = record
            
    def get_all(self) -> List[MemoryRecord]:
        return list(self.records.values())
        
    def get_training_records(self) -> List[MemoryRecord]:
        return [r for r in self.records.values() if r.provenance == "training" and r.current_status != "RETIRED"]

    def update_status(self, memory_id: str, status: str, residual_evasion: float, defense_version: str):
        if memory_id in self.records:
            self.records[memory_id].current_status = status
            self.records[memory_id].residual_evasion = residual_evasion
            self.records[memory_id].defense_version = defense_version
            self.records[memory_id].last_seen = time.time()

    def get_by_family(self, family: str, limit: int = 3) -> List[MemoryRecord]:
        return [r for r in self.records.values()
                if r.attack_family == family][:limit]
