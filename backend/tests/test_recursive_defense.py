import pytest
import pandas as pd
from app.defense.schemas import CertificationRequest
from app.defense.recursive_engine import run_certification, _generate_cert_id
from app.api.endpoints import initialize_app_state, _APP_STATE

@pytest.fixture(scope="module", autouse=True)
def setup_state():
    initialize_app_state(seed=42)
    yield

def test_training_evaluation_customer_separation():
    train_df = _APP_STATE["train_df"]
    test_df = _APP_STATE["test_df"]
    
    train_cust = set(train_df["customer_id"].unique())
    test_cust = set(test_df["customer_id"].unique())
    
    assert len(train_cust.intersection(test_cust)) == 0

def test_training_evaluation_transaction_separation():
    train_df = _APP_STATE["train_df"]
    test_df = _APP_STATE["test_df"]
    
    train_tx = set(train_df["transaction_id"].unique())
    test_tx = set(test_df["transaction_id"].unique())
    
    assert len(train_tx.intersection(test_tx)) == 0

def test_deterministic_certification_id():
    req1 = CertificationRequest(attack_family="micro_structuring", seed=42)
    req2 = CertificationRequest(attack_family="micro_structuring", seed=42)
    req3 = CertificationRequest(attack_family="micro_structuring", seed=43)
    
    assert _generate_cert_id(req1) == _generate_cert_id(req2)
    assert _generate_cert_id(req1) != _generate_cert_id(req3)
    assert not _generate_cert_id(req1).startswith("CERT-UUID")

def test_certification_fail_due_to_evasion():
    req = CertificationRequest(
        attack_family="micro_structuring",
        seed=42,
        rounds=1,
        generations_per_round=1,
        population_size=2,
        attack_scale=20
    )
    result = run_certification(req)
    
    assert result.customer_leakage == 0
    assert result.row_leakage == 0
    assert result.reproducibility_checked is True
    
    if result.residual_evasion >= 0.05:
        assert result.certification_status == "FAILED"

def test_reproducibility_same_config():
    req = CertificationRequest(
        attack_family="micro_structuring",
        seed=101,
        rounds=1,
        generations_per_round=1,
        population_size=1,
        attack_scale=10
    )
    res1 = run_certification(req)
    res2 = run_certification(req)
    
    assert res1.certification_id == res2.certification_id
    assert res1.rounds[0].evasion_rate == res2.rounds[0].evasion_rate

def test_reproducibility_different_seed():
    req1 = CertificationRequest(
        attack_family="micro_structuring", seed=102, rounds=1, generations_per_round=1, population_size=1, attack_scale=10
    )
    req2 = CertificationRequest(
        attack_family="micro_structuring", seed=103, rounds=1, generations_per_round=1, population_size=1, attack_scale=10
    )
    res1 = run_certification(req1)
    res2 = run_certification(req2)
    
    assert res1.certification_id != res2.certification_id
    assert res1.rounds[0].evasion_rate != res2.rounds[0].evasion_rate

# --- row_leakage: real computation, not a hardcoded literal (audit finding A) ---


def test_row_leakage_is_actually_computed_not_hardcoded(monkeypatch):
    """Before this fix, CertificationResult.row_leakage was an unconditional
    literal 0 -- this test proves the gate can genuinely fail by forcing one
    real collision between an eval-attack transaction_id and a real train_df
    transaction_id, via the same generate_matched_population_attacks call
    recursive_engine itself makes (no production leakage-detection code is
    duplicated here)."""
    import app.defense.recursive_engine as recursive_engine

    train_df = _APP_STATE["train_df"]
    collision_id = train_df["transaction_id"].iloc[0]

    real_fn = recursive_engine.generate_matched_population_attacks

    def poisoned_fn(*args, **kwargs):
        df = real_fn(*args, **kwargs)
        df = df.copy()
        df.loc[df.index[0], "transaction_id"] = collision_id
        return df

    monkeypatch.setattr(recursive_engine, "generate_matched_population_attacks", poisoned_fn)

    req = CertificationRequest(
        attack_family="micro_structuring", seed=105, rounds=1, generations_per_round=1, population_size=1, attack_scale=10
    )
    result = run_certification(req)

    assert result.row_leakage > 0
    assert result.certification_status != "CERTIFIED"


def test_row_leakage_zero_across_multiple_rounds_despite_fixed_base_seed():
    """base_attacks_df intentionally reuses the same seed every round (a
    stable baseline probe for fair round-to-round comparison) -- that
    expected self-repetition must NOT be flagged as leakage. Only genuine
    collisions with train/test data or with a PRIOR round's evolved_attacks_df
    should count."""
    req = CertificationRequest(
        attack_family="micro_structuring", seed=106, rounds=2, generations_per_round=1, population_size=1, attack_scale=10
    )
    result = run_certification(req)
    assert result.row_leakage == 0


def test_no_policy_fabrication():
    req = CertificationRequest(
        attack_family="micro_structuring", seed=104, rounds=1, generations_per_round=1, population_size=1, attack_scale=10
    )
    res = run_certification(req)
    
    if res.rounds[0].candidate_defense_id == "NO_NEW_DEFENSE_GENERATED":
        assert res.rounds[0].new_defense_created is False
        assert res.final_defense_id == res.starting_defense_id
