"""Tests for the Judge Sandbox LLM genome compiler (Day 8b).

Real, deterministic pipeline logic (validation, merge, simulation) is
tested for real. The LLM call itself (compile_genome's Groq request) is
network-dependent, non-deterministic, and costs API credits -- mocked here
the same way Day 6-final's test_arena_run_default_n_instances_is_2000_mocked
mocks the one expensive/non-reproducible path while everything downstream
stays real. The real Groq call was verified manually during the Day 8b
planning turn (real connectivity, real auth, real structured tool-use
output) -- see the session's curl-equivalent verification, not repeated
here as an automated test.
"""

from unittest.mock import patch

import pytest

from app.core.config import N_CUSTOMERS, N_MERCHANTS, SEED
from app.red_team.attack_genomes import MICRO_STRUCTURING_GENOME, SYNTHETIC_VOICE_AUTHORIZATION_GENOME
from app.red_team.sandbox_compiler import (
    GenomeProposal,
    SandboxCompilerError,
    _validate_overrides,
    compile_genome,
    generate_sandbox_instance,
    merge_genome,
)
from app.simulator.clean_generator import generate_customer_profiles, generate_merchants


@pytest.fixture(scope="module")
def customers_and_merchants():
    merchants = generate_merchants(N_MERCHANTS, seed=SEED)
    customers = generate_customer_profiles(N_CUSTOMERS, merchants, seed=SEED)
    return customers, merchants


def test_validate_overrides_range_within_bounds_passes():
    _validate_overrides("micro_structuring", {"amount_per_tx_range": [2600, 3000]})


def test_validate_overrides_range_exceeds_canonical_bound_raises():
    with pytest.raises(ValueError, match="not fully within canonical range"):
        _validate_overrides("micro_structuring", {"amount_per_tx_range": [2000, 3000]})


def test_validate_overrides_range_not_two_element_list_raises():
    with pytest.raises(ValueError, match="must be a \\[low, high\\] pair"):
        _validate_overrides("micro_structuring", {"amount_per_tx_range": 3000})


def test_validate_overrides_enum_valid_member_passes():
    _validate_overrides("synthetic_voice_authorization", {"impersonated_role": "bank_agent"})


def test_validate_overrides_enum_invalid_member_raises():
    with pytest.raises(ValueError, match="must be one of"):
        _validate_overrides("synthetic_voice_authorization", {"impersonated_role": "random_stranger"})


def test_validate_overrides_unknown_key_raises():
    with pytest.raises(ValueError, match="not an overridable parameter"):
        _validate_overrides("micro_structuring", {"time_window_hours": 10})


def test_validate_overrides_unknown_family_raises():
    with pytest.raises(ValueError, match="unknown family"):
        _validate_overrides("not_a_real_family", {})


def test_merge_genome_does_not_mutate_canonical_genome():
    original_range = list(MICRO_STRUCTURING_GENOME["parameters"]["amount_per_tx_range"])
    proposal = GenomeProposal(
        family="micro_structuring",
        parameter_overrides={"amount_per_tx_range": [2600, 3000]},
        rationale="test",
    )
    genome = merge_genome(proposal)
    assert genome["parameters"]["amount_per_tx_range"] == [2600, 3000]
    assert MICRO_STRUCTURING_GENOME["parameters"]["amount_per_tx_range"] == original_range
    assert genome is not MICRO_STRUCTURING_GENOME


def test_merge_genome_wraps_enum_override_as_single_element_list():
    proposal = GenomeProposal(
        family="synthetic_voice_authorization",
        parameter_overrides={"impersonated_role": "executive"},
        rationale="test",
    )
    genome = merge_genome(proposal)
    assert genome["parameters"]["impersonated_role"] == ["executive"]
    assert SYNTHETIC_VOICE_AUTHORIZATION_GENOME["parameters"]["impersonated_role"] == [
        "bank_agent", "executive", "family_member",
    ]


def test_merge_genome_preserves_genome_id_and_family():
    proposal = GenomeProposal(family="micro_structuring", parameter_overrides={}, rationale="test")
    genome = merge_genome(proposal)
    assert genome["genome_id"] == "ATK-MS-001"
    assert genome["family"] == "micro_structuring"


def test_generate_sandbox_instance_real_execution(customers_and_merchants):
    """Real, non-mocked generation -- reuses the family's own existing
    top-level generator via ATTACK_GENERATORS, exactly as the arena does.
    """
    customers, merchants = customers_and_merchants
    proposal = GenomeProposal(
        family="synthetic_voice_authorization",
        parameter_overrides={"impersonated_role": "executive"},
        rationale="test",
    )
    genome = merge_genome(proposal)
    rows = generate_sandbox_instance(genome, customers, merchants)

    assert len(rows) >= 1
    assert (rows["attack_family"] == "synthetic_voice_authorization").all()
    assert (rows["genome_id"] == "ATK-VD-001").all()
    assert (rows["is_fraud"] == 1).all()
    assert (rows["device_id"].str.startswith("VOICE-") | rows["channel"].eq("voice_authorized")).all()


def test_compile_genome_retries_once_then_raises_on_persistent_invalid_output():
    """Mocks the LLM call itself (network/non-deterministic) to always
    return a structurally invalid proposal -- confirms the retry-once-
    then-fail contract without ever calling the real Groq API.
    """
    bad_proposal = {"family": "micro_structuring", "parameter_overrides": {"amount_per_tx_range": [999, 999]}, "rationale": "bad"}

    with patch("app.red_team.sandbox_compiler.Groq"):
        with patch(
            "app.red_team.sandbox_compiler._call_llm",
            side_effect=ValueError("amount_per_tx_range override [999, 999] is not fully within canonical range"),
        ) as mock_call:
            with pytest.raises(SandboxCompilerError, match="failed validation twice"):
                compile_genome("some free text")
            assert mock_call.call_count == 2


def test_compile_genome_succeeds_on_first_valid_llm_response():
    """Mocks only the LLM call (network/non-deterministic); everything
    downstream of a valid proposal is real.
    """
    valid_proposal = GenomeProposal(
        family="micro_structuring",
        parameter_overrides={"amount_per_tx_range": [2600, 3000]},
        rationale="mocked",
    )
    with patch("app.red_team.sandbox_compiler.Groq"):
        with patch("app.red_team.sandbox_compiler._call_llm", return_value=valid_proposal) as mock_call:
            result = compile_genome("some free text")
    assert result == valid_proposal
    mock_call.assert_called_once()
