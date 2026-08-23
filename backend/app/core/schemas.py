"""Pydantic v2 data contracts for Sentinel-X (CLAUDE.md §6 — exact, do not modify)."""

from datetime import datetime
from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class CustomerProfile(BaseModel):
    """A synthetic payment-network customer and their stable behavioral baseline."""

    customer_id: str
    base_location: str
    primary_devices: List[str]
    mean_spend: float
    spend_variance: float
    usual_merchants: List[str]
    usual_beneficiaries: List[str]


class TransactionBase(BaseModel):
    """A single synthetic payment transaction (clean, pre-attack-injection)."""

    transaction_id: str
    timestamp: datetime
    customer_id: str
    merchant_id: str
    beneficiary_id: str
    amount: float
    currency: str = "INR"
    channel: str  # POS, WEB, P2P, voice_authorized
    device_id: str
    ip_region: str
    location: str
    merchant_category: str
    semantic_risk_score: float = 0.0  # [0.0, 1.0]
    voice_confidence_score: float = 1.0  # [0.0, 1.0], lower = more convincing deepfake


class InjectedTransaction(TransactionBase):
    """A transaction after red-team attack injection (clean or fraudulent)."""

    is_fraud: int = Field(default=0, ge=0, le=1)
    attack_family: Optional[str] = None
    genome_id: Optional[str] = None


class DetectionResult(BaseModel):
    """M0's risk score and decision for one scored transaction."""

    transaction_id: str
    risk_score: float
    decision: str  # ALLOW, STEP_UP, REVIEW, BLOCK
    reason_codes: List[Dict[str, str]]
    latency_ms: float


class ArenaRunSummary(BaseModel):
    """Summary of one Adversarial Arena run for a single attack family."""

    run_id: str
    attack_family: str
    initial_evasion_rate: float
    final_evasion_rate: float
    robustness_gain: float
    hard_examples_count: int
    retrained_f1_score: float
