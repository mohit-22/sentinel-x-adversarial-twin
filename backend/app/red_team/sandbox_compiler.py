"""Judge Sandbox LLM genome compiler (Day 8b).

CLAUDE.md §5's design rule: the LLM never scores a transaction and never
makes a fraud/allow/block decision -- it only ever produces a structured
Attack Genome (JSON, Pydantic-validated), matching §4.1's canonical shape.

Free text -> LLM selects the closest of the 5 canonical families and
proposes BOUNDED parameter overrides (a narrower sub-range within an
already-approved range, or one member of an already-approved enum) ->
Pydantic + bounds validated -> merged onto a deep copy of that family's
canonical genome (attack_genomes.py itself is never modified) -> simulated
via that family's own existing generate_<family>_attacks (Day 6.5's
ATTACK_GENERATORS registry -- reused, not reimplemented).

LLM provider: Groq (openai/gpt-oss-120b), chosen and approved at the
Day 8b planning turn. Verified empirically before this module was written:
real connectivity, real auth, and the exact tool-schema shape needed for
reliable structured output (an explicit, per-key typed sub-schema -- a
generic free-form `object` schema was tested first and caused the model to
collapse [low, high] ranges into a single scalar; typed per-key properties
fixed it).

GROQ_API_KEY is read from backend/.env, loaded here rather than in
main.py/config.py: neither is in Day 8b's ALLOWED_TO_TOUCH, and this is the
only module that needs the key.
"""

import json
import os
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from dotenv import load_dotenv
from groq import Groq
from pydantic import BaseModel, ValidationError

from app.core.config import SEED
from app.red_team.attack_genomes import (
    BEHAVIORAL_CAMOUFLAGE_GENOME,
    MICRO_STRUCTURING_GENOME,
    SOCIAL_ENGINEERING_COERCION_GENOME,
    SYNTHETIC_IDENTITY_DRIFT_GENOME,
    SYNTHETIC_VOICE_AUTHORIZATION_GENOME,
)
from app.red_team.attack_injector import ATTACK_GENERATORS

load_dotenv(Path(__file__).resolve().parents[2] / ".env")

GROQ_MODEL = "openai/gpt-oss-120b"

_CANONICAL_GENOMES: Dict[str, Dict] = {
    "micro_structuring": MICRO_STRUCTURING_GENOME,
    "synthetic_identity_drift": SYNTHETIC_IDENTITY_DRIFT_GENOME,
    "behavioral_camouflage": BEHAVIORAL_CAMOUFLAGE_GENOME,
    "social_engineering_coercion": SOCIAL_ENGINEERING_COERCION_GENOME,
    "synthetic_voice_authorization": SYNTHETIC_VOICE_AUTHORIZATION_GENOME,
}
_FAMILIES: List[str] = list(_CANONICAL_GENOMES.keys())

# Which parameters each family allows the LLM to narrow, and how:
# "range" -> a [low, high] sub-range that must be fully inside the
# canonical [low, high]; "enum" -> one member of the canonical list.
_OVERRIDABLE_PARAMS: Dict[str, Dict[str, str]] = {
    "micro_structuring": {"split_count_range": "range", "amount_per_tx_range": "range"},
    "synthetic_identity_drift": {
        "drift_window_days_range": "range",
        "extraction_transaction_count_range": "range",
        "extraction_amount_multiplier_range": "range",
    },
    "behavioral_camouflage": {"burst_transaction_count_range": "range"},
    "social_engineering_coercion": {"coercion_pretext_options": "enum"},
    "synthetic_voice_authorization": {"impersonated_role": "enum"},
}

_SYSTEM_PROMPT = """You compile a free-text fraud scenario description into a structured attack genome for a fraud-detection sandbox. You never score transactions or make fraud/allow/block decisions -- you only select the closest matching attack family and optionally narrow specific parameters within their ALREADY-APPROVED bounds.

Family descriptions:
- micro_structuring: splitting a large amount into many small transactions to mule accounts to stay under detection thresholds.
- synthetic_identity_drift: an account behaves normally for weeks (trust-building), then suddenly executes a high-velocity extraction via a new device/payee.
- behavioral_camouflage: fraudulent transactions interleaved within an authentic-looking spending burst.
- social_engineering_coercion: a structurally normal transaction driven by a coercive/urgent pretext message (e.g. fake KYC, refund, cashback).
- synthetic_voice_authorization: a phone call impersonating a trusted role (bank agent, executive, family member) to bypass step-up verification.

Only propose overrides for the parameters listed in the tool schema for whichever family you pick, using values strictly within their bounds/choices. Omit parameter_overrides entirely (empty object) if the scenario does not call for narrowing anything."""


class GenomeProposal(BaseModel):
    """The LLM's raw structured output, Pydantic-validated before any
    bounds check. Not the canonical genome itself -- see merge_genome.
    """

    family: str
    parameter_overrides: Dict[str, Any]
    rationale: str


class SandboxCompilerError(Exception):
    """Raised when the LLM's proposal fails structural or bounds validation
    twice (one retry). Callers surface this as a clean 4xx -- never silently
    coerce, clamp, or fabricate a plausible-looking genome.
    """


def _range_schema(canonical_range: List[float], family_note: str) -> Dict[str, Any]:
    return {
        "type": "array",
        "items": {"type": "number"},
        "minItems": 2,
        "maxItems": 2,
        "description": f"{family_note}; a [low, high] sub-range within canonical {canonical_range}",
    }


def _build_tools() -> List[Dict[str, Any]]:
    properties: Dict[str, Any] = {}
    for family, keys in _OVERRIDABLE_PARAMS.items():
        base_params = _CANONICAL_GENOMES[family]["parameters"]
        for key, kind in keys.items():
            if kind == "range":
                properties[key] = _range_schema(base_params[key], f"{family} only")
            else:  # enum
                properties[key] = {
                    "type": "string",
                    "enum": list(base_params[key]),
                    "description": f"{family} only",
                }

    return [
        {
            "type": "function",
            "function": {
                "name": "compile_genome",
                "description": "Compile a fraud scenario into a family selection plus bounded parameter overrides",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "family": {"type": "string", "enum": _FAMILIES},
                        "parameter_overrides": {
                            "type": "object",
                            "properties": properties,
                            "additionalProperties": False,
                        },
                        "rationale": {
                            "type": "string",
                            "description": "One or two sentences explaining the family choice and any overrides",
                        },
                    },
                    "required": ["family", "parameter_overrides", "rationale"],
                },
            },
        }
    ]


def _validate_overrides(family: str, overrides: Dict[str, Any]) -> None:
    if family not in _FAMILIES:
        raise ValueError(f"unknown family {family!r}. Known: {_FAMILIES}")

    allowed = _OVERRIDABLE_PARAMS[family]
    canonical_params = _CANONICAL_GENOMES[family]["parameters"]

    for key, value in overrides.items():
        if key not in allowed:
            raise ValueError(f"{key!r} is not an overridable parameter for family {family!r}")

        kind = allowed[key]
        canonical = canonical_params[key]

        if kind == "range":
            if not (isinstance(value, list) and len(value) == 2 and all(isinstance(v, (int, float)) for v in value)):
                raise ValueError(f"{key!r} must be a [low, high] pair, got {value!r}")
            low, high = value
            canon_low, canon_high = canonical
            if not (canon_low <= low <= high <= canon_high):
                raise ValueError(f"{key!r} override {value!r} is not fully within canonical range {canonical!r}")
        else:  # enum
            if value not in canonical:
                raise ValueError(f"{key!r} override {value!r} must be one of {canonical!r}")


def _call_llm(client: Groq, free_text: str, retry_error: Optional[str] = None) -> GenomeProposal:
    messages = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": free_text},
    ]
    if retry_error:
        messages.append(
            {
                "role": "user",
                "content": f"Your previous output was invalid: {retry_error}. Return a corrected call to compile_genome.",
            }
        )

    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=messages,
        tools=_build_tools(),
        tool_choice={"type": "function", "function": {"name": "compile_genome"}},
    )
    tool_calls = response.choices[0].message.tool_calls
    if not tool_calls:
        raise ValueError("LLM did not call the compile_genome tool")

    raw = json.loads(tool_calls[0].function.arguments)
    proposal = GenomeProposal.model_validate(raw)
    _validate_overrides(proposal.family, proposal.parameter_overrides)
    return proposal


def compile_genome(free_text: str) -> GenomeProposal:
    """Free text -> validated GenomeProposal. Retries the LLM call once on
    a structural/bounds validation failure, then raises SandboxCompilerError.
    """
    client = Groq(api_key=os.environ["GROQ_API_KEY"])

    try:
        return _call_llm(client, free_text)
    except (ValidationError, ValueError) as first_error:
        try:
            return _call_llm(client, free_text, retry_error=str(first_error))
        except (ValidationError, ValueError) as second_error:
            raise SandboxCompilerError(
                f"LLM proposal failed validation twice. First: {first_error}. Second: {second_error}."
            ) from second_error


def merge_genome(proposal: GenomeProposal) -> Dict:
    """Deep-copies the canonical genome (attack_genomes.py itself untouched)
    and merges validated overrides. Enum-type overrides are wrapped as a
    single-element list, matching how the generator functions already
    consume these fields (np.random.choice(params[key]) -- forcing that
    choice by making it the only element, not new selection logic).
    """
    genome = deepcopy(_CANONICAL_GENOMES[proposal.family])
    kinds = _OVERRIDABLE_PARAMS[proposal.family]
    for key, value in proposal.parameter_overrides.items():
        genome["parameters"][key] = [value] if kinds[key] == "enum" else value
    return genome


def generate_sandbox_instance(genome: Dict, customers: pd.DataFrame, merchants: pd.DataFrame) -> pd.DataFrame:
    """One freshly-generated instance for this genome, via the family's own
    existing top-level generator (Day 6.5 ATTACK_GENERATORS registry) --
    reused exactly as the arena and /payment-twin already reuse it.
    """
    attacks_fn = ATTACK_GENERATORS[genome["family"]]["attacks_fn"]
    return attacks_fn(genome, customers, merchants, n_instances=1, seed=SEED)
