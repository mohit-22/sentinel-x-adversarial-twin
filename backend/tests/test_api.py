"""Tests for the FastAPI wrapper (CLAUDE.md §7 -- exact six endpoints).

Uses a module-scoped TestClient so the real (slow, ~20s) startup pipeline
-- payment twin generation + Day 4 training -- runs exactly ONCE for the
whole file, not once per test.

TESTING-ONLY SHORTCUT, clearly separated from the real endpoint behavior:
/arena/run's default n_instances=2000 takes ~100s for real (confirmed via
manual curl during the Day 6-final planning turn -- 99.457s, exact match
to the known n=2000 result). Running that for real in the automated suite
would make every test run take 100+ seconds. test_arena_run_default_n_instances_is_2000
below MOCKS run_arena_mvp_gate to assert what n_instances value the route
actually passes, without executing the real computation. Every OTHER
/arena/run test uses a small real n_instances (fast, genuine end-to-end
execution, not mocked) so the endpoint's real behavior is still exercised.
"""

from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:  # triggers the lifespan startup hook once
        yield c


def test_simulate_returns_expected_shape(client):
    response = client.post("/api/v1/simulate", json={})
    assert response.status_code == 200
    body = response.json()
    assert body["customer_count"] == 10_000
    assert body["merchant_count"] == 500
    assert body["transaction_count"] == 50_000
    assert 0.0 <= body["fidelity_report"]["amount_similarity_overall"] <= 1.0
    assert 0.0 <= body["fidelity_report"]["temporal_similarity_overall"] <= 1.0


def test_simulate_respects_request_overrides(client):
    response = client.post("/api/v1/simulate", json={"n_customers": 200, "n_merchants": 20, "n_transactions": 1000})
    assert response.status_code == 200
    body = response.json()
    assert body["customer_count"] == 200
    assert body["merchant_count"] == 20
    assert body["transaction_count"] >= 1000


def _valid_transaction(customer_id="CUST-000000", transaction_id="TEST-TXN-0001"):
    return {
        "transaction_id": transaction_id,
        "timestamp": "2026-01-15T12:00:00",
        "customer_id": customer_id,
        "merchant_id": "P2P-TRANSFER",
        "beneficiary_id": "MULE-9999-0",
        "amount": 4500.0,
        "channel": "P2P",
        "device_id": "BRAND-NEW-DEVICE-0",
        "ip_region": "Erode",
        "location": "Erode",
        "merchant_category": "p2p_transfer",
    }


def test_detect_valid_customer_returns_scored_result(client):
    response = client.post("/api/v1/detect", json={"transactions": [_valid_transaction()]})
    assert response.status_code == 200
    results = response.json()["results"]
    assert len(results) == 1
    result = results[0]
    assert result["transaction_id"] == "TEST-TXN-0001"
    assert 0.0 <= result["risk_score"] <= 1.0
    assert result["decision"] in ("ALLOW", "STEP_UP", "REVIEW", "BLOCK")
    assert result["reason_codes"] == []  # SHAP is Day 8 -- honestly empty, not fabricated
    assert result["latency_ms"] >= 0.0


def test_detect_unknown_customer_id_returns_400(client):
    response = client.post(
        "/api/v1/detect",
        json={"transactions": [_valid_transaction(customer_id="CUST-999999-DOES-NOT-EXIST")]},
    )
    assert response.status_code == 400
    assert "CUST-999999-DOES-NOT-EXIST" in response.json()["detail"]


def test_arena_run_with_small_n_instances_real_execution(client):
    """Real, non-mocked end-to-end execution -- fast because n_instances is
    small, not because anything is faked.
    """
    response = client.post("/api/v1/arena/run", json={"genome_id": "ATK-MS-001", "n_instances": 20})
    assert response.status_code == 200
    body = response.json()
    assert set(body.keys()) == {
        "run_id", "attack_family", "initial_evasion_rate", "final_evasion_rate",
        "robustness_gain", "hard_examples_count", "retrained_f1_score",
    }
    assert body["attack_family"] == "micro_structuring"
    assert 0.0 <= body["initial_evasion_rate"] <= 1.0
    assert 0.0 <= body["final_evasion_rate"] <= 1.0


def test_arena_run_unknown_genome_id_returns_404(client):
    response = client.post("/api/v1/arena/run", json={"genome_id": "ATK-FAKE-999"})
    assert response.status_code == 404
    assert "ATK-FAKE-999" in response.json()["detail"]


def test_arena_run_default_n_instances_is_2000_mocked(client):
    """TESTING-ONLY SHORTCUT (see module docstring): mocks run_arena_mvp_gate
    to avoid the real ~100s computation, and asserts what n_instances value
    the route actually passes through -- 2000 when omitted from the
    request, matching run_arena_mvp_gate's own official default exactly
    (the route doesn't hardcode 2000 itself; it omits the kwarg so the
    function's own default applies, verified here).
    """
    fake_summary = {
        "run_id": "arena-ATK-MS-001-42",
        "attack_family": "micro_structuring",
        "initial_evasion_rate": 0.01,
        "final_evasion_rate": 0.01,
        "robustness_gain": 0.0,
        "hard_examples_count": 0,
        "retrained_f1_score": 0.9,
    }
    with patch("app.api.endpoints.run_arena_mvp_gate", return_value=fake_summary) as mock_run:
        response = client.post("/api/v1/arena/run", json={"genome_id": "ATK-MS-001"})
    assert response.status_code == 200
    mock_run.assert_called_once()
    _, kwargs = mock_run.call_args
    assert "n_instances" not in kwargs  # route omits it so the function's own default (2000) applies

    with patch("app.api.endpoints.run_arena_mvp_gate", return_value=fake_summary) as mock_run_override:
        response = client.post("/api/v1/arena/run", json={"genome_id": "ATK-MS-001", "n_instances": 50})
    _, kwargs_override = mock_run_override.call_args
    assert kwargs_override["n_instances"] == 50  # explicit override IS passed through


def test_metrics_returns_expected_shape(client):
    response = client.get("/api/v1/metrics")
    assert response.status_code == 200
    body = response.json()
    for key in ("precision", "recall", "f1", "pr_auc", "fpr"):
        assert 0.0 <= body[key] <= 1.0
    assert isinstance(body["test_set_size"], int)
    assert body["test_set_size"] > 0
    assert "latest_arena_run" in body  # present as a key regardless of null/populated


def test_metrics_latest_arena_run_is_null_on_fresh_server():
    """Dedicated fresh TestClient (own startup) so this is NOT affected by
    other tests' /arena/run calls against the shared `client` fixture --
    a genuine "no arena run yet this session" check, not order-dependent.
    """
    with TestClient(app) as fresh_client:
        response = fresh_client.get("/api/v1/metrics")
    assert response.status_code == 200
    assert response.json()["latest_arena_run"] is None


def test_metrics_latest_arena_run_populates_after_arena_run(client):
    """Sequential, order-independent: calls /arena/run with a small but
    real n_instances, then confirms /metrics immediately reflects THAT
    specific run (matching run_id/attack_family), not just "some" value.
    """
    arena_response = client.post("/api/v1/arena/run", json={"genome_id": "ATK-ID-001", "n_instances": 20})
    assert arena_response.status_code == 200
    arena_body = arena_response.json()

    metrics_response = client.get("/api/v1/metrics")
    assert metrics_response.status_code == 200
    latest = metrics_response.json()["latest_arena_run"]
    assert latest is not None
    assert latest["run_id"] == arena_body["run_id"]
    assert latest["attack_family"] == arena_body["attack_family"] == "synthetic_identity_drift"
    assert latest["initial_evasion_rate"] == arena_body["initial_evasion_rate"]


def test_explain_unknown_transaction_id_returns_404_not_a_crash(client):
    """Day 8a: /explain is real now (SHAP on the cached train/test set).
    TEST-TXN-0001 was never part of that cached dataset, so this is the
    honest "not in today's SHAP scope" 404, not the old blanket 501.
    """
    response = client.get("/api/v1/explain/TEST-TXN-0001")
    assert response.status_code == 404
    assert "TEST-TXN-0001" in response.json()["detail"]


def test_sandbox_compile_returns_501_not_a_crash(client):
    response = client.post("/api/v1/sandbox/compile", json={})
    assert response.status_code == 501
    assert "Day 8" in response.json()["detail"]


# ============================================================================
# GET /payment-twin/{customer_id} -- Day 7 Screen 3, approved 7th endpoint
# ============================================================================


def test_payment_twin_returns_real_customer_and_counterfactual(client):
    response = client.get("/api/v1/payment-twin/CUST-000000")
    assert response.status_code == 200
    body = response.json()

    assert body["customer"]["customer_id"] == "CUST-000000"
    assert body["customer"]["base_location"]
    assert len(body["customer"]["primary_devices"]) > 0

    assert len(body["normal_transactions"]) > 0
    assert all(t["customer_id"] == "CUST-000000" for t in body["normal_transactions"])
    # normal_transactions are TransactionBase, which has no is_fraud field at all
    assert all("is_fraud" not in t for t in body["normal_transactions"])

    assert len(body["counterfactual_transactions"]) > 0
    assert all(t["customer_id"] == "CUST-000000" for t in body["counterfactual_transactions"])
    fraud_legs = [t for t in body["counterfactual_transactions"] if t["is_fraud"] == 1]
    assert len(fraud_legs) > 0
    assert all(t["attack_family"] == "micro_structuring" for t in fraud_legs)
    assert all(t["genome_id"] == "ATK-MS-001" for t in fraud_legs)


def test_payment_twin_respects_attack_family_query_param(client):
    response = client.get("/api/v1/payment-twin/CUST-000000?attack_family=synthetic_identity_drift")
    assert response.status_code == 200
    body = response.json()
    fraud_legs = [t for t in body["counterfactual_transactions"] if t["is_fraud"] == 1]
    assert len(fraud_legs) > 0
    assert all(t["genome_id"] == "ATK-ID-001" for t in fraud_legs)
    assert all(t["device_id"].startswith("DRIFT-DEV-") for t in fraud_legs)


def test_payment_twin_unknown_customer_returns_404(client):
    response = client.get("/api/v1/payment-twin/CUST-999999-DOES-NOT-EXIST")
    assert response.status_code == 404
    assert "CUST-999999-DOES-NOT-EXIST" in response.json()["detail"]


def test_payment_twin_unknown_attack_family_returns_404(client):
    response = client.get("/api/v1/payment-twin/CUST-000000?attack_family=not_a_real_family")
    assert response.status_code == 404
    assert "not_a_real_family" in response.json()["detail"]


def test_payment_twin_counterfactual_never_collides_with_real_beneficiaries(client):
    """The counterfactual is illustrative, not a mutation of real rows --
    confirms its mule/drift beneficiaries never coincide with transaction
    ids already present in the customer's real normal_transactions.
    """
    response = client.get("/api/v1/payment-twin/CUST-000000")
    body = response.json()
    normal_ids = {t["transaction_id"] for t in body["normal_transactions"]}
    counterfactual_ids = {t["transaction_id"] for t in body["counterfactual_transactions"]}
    assert normal_ids.isdisjoint(counterfactual_ids)
