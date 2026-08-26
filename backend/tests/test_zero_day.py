"""Tests for the Zero-Day Radar."""

from unittest.mock import patch
import pytest
from fastapi.testclient import TestClient
from app.main import app
from app.blue_team.zero_day import SAFE_NOVELTY_FEATURES

@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c

def _valid_transaction(customer_id="CUST-000000", transaction_id="TEST-TXN-0001", amount=4500.0):
    return {
        "transaction_id": transaction_id,
        "timestamp": "2026-01-15T12:00:00",
        "customer_id": customer_id,
        "merchant_id": "P2P-TRANSFER",
        "beneficiary_id": "MULE-9999-0",
        "amount": amount,
        "currency": "INR",
        "channel": "P2P",
        "device_id": "DEV-000000",
        "ip_region": "Nagpur",
        "location": "Nagpur",
        "merchant_category": "finance",
        "semantic_risk_score": 0.1,
        "voice_confidence_score": 1.0,
    }

def test_safe_novelty_features():
    assert "is_fraud" not in SAFE_NOVELTY_FEATURES
    assert "attack_family" not in SAFE_NOVELTY_FEATURES
    assert "genome_id" not in SAFE_NOVELTY_FEATURES
    # It should include core numeric features
    assert "amount" in SAFE_NOVELTY_FEATURES

def test_zero_day_scan_endpoint_deterministic(client):
    req_body = {
        "transactions": [
            _valid_transaction(transaction_id="TXN-1", amount=10.0),
            _valid_transaction(transaction_id="TXN-2", amount=90000.0) # Anomaly
        ]
    }
    resp1 = client.post("/api/v1/zero-day/scan", json=req_body)
    assert resp1.status_code == 200
    data1 = resp1.json()
    
    resp2 = client.post("/api/v1/zero-day/scan", json=req_body)
    data2 = resp2.json()
    
    # Must be perfectly deterministic
    assert data1["results"] == data2["results"]
    assert data1["aggregate_metrics"]["total_scanned"] == 2
    
    # TXN-2 should have a higher novelty score than TXN-1 because of the amount
    score1 = next(r["novelty_score"] for r in data1["results"] if r["transaction_id"] == "TXN-1")
    score2 = next(r["novelty_score"] for r in data1["results"] if r["transaction_id"] == "TXN-2")
    assert score2 > score1

@patch("app.api.endpoints.cluster_unknowns")
def test_zero_day_scan_clustering(mock_cluster, client):
    import pandas as pd
    
    # Mock the cluster_unknowns to return a single cluster for all unknowns
    def fake_cluster(df):
        df = df.copy()
        df["cluster_id"] = 0
        return df
    mock_cluster.side_effect = fake_cluster
    
    txns = [_valid_transaction(transaction_id=f"TXN-ANOM-{i}", amount=50000.0 + i) for i in range(10)]
    resp = client.post("/api/v1/zero-day/scan", json={"transactions": txns})
    data = resp.json()
    
    assert data["aggregate_metrics"]["total_scanned"] == 10
    assert data["aggregate_metrics"]["total_unknown"] >= 9
    assert data["aggregate_metrics"]["cluster_count"] == 1
    assert data["clusters"][0]["transaction_count"] >= 9
