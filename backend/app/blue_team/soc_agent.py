"""Autonomous SOC Agent: LLM-reasoned verdict over a flagged transaction's
real SHAP evidence + immune memory context (CLAUDE.md §3's design rule: the
LLM never scores a transaction or makes the fraud/allow/block decision --
that's M0 + explainability.py, both already deterministic. The LLM only
narrates a structured investigation over evidence it is given).

Groq client/model/error-handling pattern reused exactly from
sandbox_compiler.py (Day 8b, approved provider): tool-calling with a forced
function call (verified more reliable than response_format=json_object for
this project -- see sandbox_compiler.py's docstring), one retry on
structural validation failure, then a raised error -- never a silently
coerced/fabricated verdict. GROQ_API_KEY is loaded from backend/.env via
load_dotenv, same as sandbox_compiler.py.
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
from dotenv import load_dotenv
from groq import Groq
from pydantic import BaseModel, ValidationError

from app.blue_team.explainability import compute_reason_codes, find_cached_feature_row
from app.red_team.immune_memory import ImmuneMemoryStore

load_dotenv(Path(__file__).resolve().parents[2] / ".env")

GROQ_MODEL = "openai/gpt-oss-120b"

_ATTACK_FAMILIES = [
    "micro_structuring",
    "synthetic_identity_drift",
    "behavioral_camouflage",
    "social_engineering_coercion",
    "synthetic_voice_authorization",
    "unknown",
]
_ACTIONS = ["BLOCK", "ESCALATE", "MONITOR", "ALLOW"]

_SYSTEM_PROMPT = """You are an autonomous SOC analyst for a payment fraud detection system.
You investigate a flagged transaction using the SHAP evidence and immune-memory context you are given, and record a structured verdict via the record_verdict tool.
Base your analysis ONLY on the evidence provided. Never invent facts about the transaction, the customer, or past attacks beyond what is stated."""


class AgentVerdict(BaseModel):
    transaction_id: str
    hypothesis: str
    attack_family_suspected: str
    confidence_score: float
    evidence: List[str]
    recommended_action: str
    reasoning_chain: str
    audit_log_entry: str
    similar_past_attacks: int


class _VerdictProposal(BaseModel):
    """Raw LLM tool-call output, structurally validated before being merged
    into the full AgentVerdict (which also carries transaction_id and
    similar_past_attacks -- real values this module computed, not the LLM's).
    """

    hypothesis: str
    attack_family_suspected: str
    confidence_score: float
    evidence: List[str]
    recommended_action: str
    reasoning_chain: str
    audit_log_entry: str


class SOCAgentError(Exception):
    """Raised when the LLM's verdict fails structural/bounds validation
    twice (one retry). Callers surface this as a clean error -- never
    silently coerce or fabricate a plausible-looking verdict.
    """


def _validate_proposal(proposal: _VerdictProposal) -> None:
    if proposal.attack_family_suspected not in _ATTACK_FAMILIES:
        raise ValueError(f"attack_family_suspected {proposal.attack_family_suspected!r} must be one of {_ATTACK_FAMILIES}")
    if proposal.recommended_action not in _ACTIONS:
        raise ValueError(f"recommended_action {proposal.recommended_action!r} must be one of {_ACTIONS}")
    if not (0.0 <= proposal.confidence_score <= 1.0):
        raise ValueError(f"confidence_score {proposal.confidence_score!r} must be between 0.0 and 1.0")
    if len(proposal.evidence) != 3:
        raise ValueError(f"evidence must contain exactly 3 items, got {len(proposal.evidence)}")


def _build_tools() -> List[Dict]:
    return [
        {
            "type": "function",
            "function": {
                "name": "record_verdict",
                "description": "Record the structured SOC investigation verdict for this transaction",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "hypothesis": {
                            "type": "string",
                            "description": "One sentence explaining what fraud pattern this appears to be",
                        },
                        "attack_family_suspected": {"type": "string", "enum": _ATTACK_FAMILIES},
                        "confidence_score": {"type": "number", "description": "0.0 to 1.0"},
                        "evidence": {
                            "type": "array",
                            "items": {"type": "string"},
                            "minItems": 3,
                            "maxItems": 3,
                            "description": "Exactly 3 evidence points",
                        },
                        "recommended_action": {"type": "string", "enum": _ACTIONS},
                        "reasoning_chain": {
                            "type": "string",
                            "description": "2-3 sentences explaining the reasoning step by step",
                        },
                        "audit_log_entry": {
                            "type": "string",
                            "description": "Formal one-line entry suitable for a compliance audit log",
                        },
                    },
                    "required": [
                        "hypothesis",
                        "attack_family_suspected",
                        "confidence_score",
                        "evidence",
                        "recommended_action",
                        "reasoning_chain",
                        "audit_log_entry",
                    ],
                },
            },
        }
    ]


def _call_llm(client: Groq, user_prompt: str, retry_error: Optional[str] = None) -> _VerdictProposal:
    messages = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]
    if retry_error:
        messages.append(
            {
                "role": "user",
                "content": f"Your previous output was invalid: {retry_error}. Return a corrected call to record_verdict.",
            }
        )

    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=messages,
        tools=_build_tools(),
        tool_choice={"type": "function", "function": {"name": "record_verdict"}},
    )
    tool_calls = response.choices[0].message.tool_calls
    if not tool_calls:
        raise ValueError("LLM did not call the record_verdict tool")

    raw = json.loads(tool_calls[0].function.arguments)
    proposal = _VerdictProposal.model_validate(raw)
    _validate_proposal(proposal)
    return proposal


def run_soc_agent(
    transaction_id: str,
    model,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    immune_memory: ImmuneMemoryStore,
    customer_history_summary: Optional[str] = None,
) -> AgentVerdict:
    """Investigates one flagged transaction: real SHAP reason codes (Day 8a's
    TreeExplainer, cached feature rows only -- same disclosed scope as
    GET /explain/{transaction_id}) + real immune-memory context, narrated by
    the LLM into a structured verdict. Raises ValueError if the transaction
    isn't in the startup pipeline's cached train/test split.
    """
    row = find_cached_feature_row(transaction_id, train_df, test_df)
    if row is None:
        raise ValueError(f"Transaction {transaction_id} not in cached dataset")

    reason_codes = compute_reason_codes(row, model)

    shap_summary = "\n".join(
        f"- {rc['feature']}: {rc['contribution']} ({rc['description']})" for rc in reason_codes
    )

    all_memories = immune_memory.get_all()
    similar_count = len(all_memories)
    memory_summary = "No similar past attacks in memory."
    if all_memories:
        memory_summary = (
            f"{similar_count} past attack(s) in immune memory. "
            f"Most recent: {all_memories[-1].attack_family} with {all_memories[-1].best_evasion:.1%} evasion."
        )

    user_prompt = f"""Investigate this flagged transaction:

Transaction ID: {transaction_id}

SHAP Feature Attributions (top contributors to fraud score):
{shap_summary}

Customer History Context:
{customer_history_summary or "Not available for this session."}

Immune Memory (similar past attacks):
{memory_summary}

Call record_verdict with your verdict."""

    client = Groq(api_key=os.environ["GROQ_API_KEY"])

    try:
        proposal = _call_llm(client, user_prompt)
    except (ValidationError, ValueError) as first_error:
        try:
            proposal = _call_llm(client, user_prompt, retry_error=str(first_error))
        except (ValidationError, ValueError) as second_error:
            raise SOCAgentError(
                f"LLM verdict failed validation twice. First: {first_error}. Second: {second_error}."
            ) from second_error

    return AgentVerdict(
        transaction_id=transaction_id,
        hypothesis=proposal.hypothesis,
        attack_family_suspected=proposal.attack_family_suspected,
        confidence_score=proposal.confidence_score,
        evidence=proposal.evidence,
        recommended_action=proposal.recommended_action,
        reasoning_chain=proposal.reasoning_chain,
        audit_log_entry=proposal.audit_log_entry,
        similar_past_attacks=similar_count,
    )
