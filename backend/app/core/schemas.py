"""Pydantic v2 data contracts for Sentinel-X (CLAUDE.md §6 — exact, do not modify)."""

from datetime import datetime
from typing import List

from pydantic import BaseModel


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
