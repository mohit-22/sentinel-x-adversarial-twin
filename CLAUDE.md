# CLAUDE.md — Sentinel-X Project Knowledge & Build Instructions

**Read this file fully before responding to any task in this repository.**
This is the single source of truth for architecture, scope, schemas, and
execution order. The companion document `PRD_SENTINEL_X.md` contains the
full product rationale — read it once for context, but this file governs
day-to-day implementation decisions.

If any request conflicts with this file, **stop and ask before assuming.**
Do not silently resolve ambiguity in whatever direction seems most impressive
— resolve it in whatever direction this file specifies, and if this file
doesn't specify it, ask.

---

## 0. Non-negotiable working rules

1. **Never claim code works without running it in this session and pasting
   real output.** "This should work" is not a valid statement in this project.
2. **Never add a library, service, or architectural component not listed in
   §5 TECH STACK.** If you think something else would be better, say so and
   wait for explicit approval — do not add it and explain afterward.
3. **Never invent a metric formula, API contract, schema field, or attack
   parameter not defined in this file.** If you need one that isn't defined,
   stop and ask instead of guessing a reasonable-sounding value.
4. **Build only what the current task asks for.** Do not pre-build later
   phases "while you're at it," even if it seems efficient.
5. **Vectorized Pandas/NumPy only** for feature engineering and data
   generation — never row-wise Python `for` loops over a DataFrame.
6. **Fixed random seeds everywhere** (`numpy.random.seed(42)`,
   `Faker.seed(42)`) so every run is reproducible for judges.
7. **Full type hints and docstrings on every function.** Pydantic v2 models
   at every I/O boundary (API request/response, function inputs where
   structured data crosses a module boundary).
8. **After writing code, run it. Then report:** what you ran, the actual
   output, and what still needs verification before this is considered done.
9. **A suspiciously perfect result is a bug, not a win.** F1 ≈ 1.0 on first
   training, 0% evasion after one retrain, or fidelity scores of exactly
   100% should be treated as signals of a leak or a logic error — investigate
   before reporting them as success.
10. **Write a test alongside every core function**, even a minimal one. The
    test suite is not optional polish — it's how correctness gets verified
    across a multi-day build without re-deriving trust every session.

---

## 1. Project Identity

- **Name:** Sentinel-X — Autonomous Adversarial Payment Twin
- **Event:** Mastercard Innovation Challenge @ GFF 2026, "AI Defense Lab for
  Payment Security"
- **Deadline:** 31 August 2026, 11:59 PM IST
- **One-line pitch:** "Sentinel-X is an adversarial payment twin that
  continuously invents, simulates, and evolves GenAI-powered fraud attacks
  against its own detector, then learns from every successful evasion to
  automatically harden the defense."
- **Never pitch this as** "an AI fraud detector." It is a closed-loop
  attack/defend/learn system.
- **Hard data rule:** synthetic/sandboxed data only. Never real cardholder or
  production payment data. Never target any live system, even hypothetically
  in the prompt-injection attack family — that family only ever targets a
  *fictional, sandboxed* downstream parser.

---

## 2. Current Build Phase



> **Update this section at the start of every work session before sending
> the first prompt of that session.** This is the single most important
> anti-drift mechanism in this file — it tells Claude Code exactly what is
> and isn't in scope *right now*, regardless of what looks tempting to build.


CURRENT PHASE: Day 1-2 — Synthetic payment world
ALLOWED TO TOUCH: backend/app/simulator/, backend/app/core/schemas.py, backend/app/core/config.py, backend/tests/test_simulator.py
NOT ALLOWED TO TOUCH: red_team/, blue_team/, api/, frontend/


CURRENT PHASE: Day 3 — micro_structuring attack family ONLY
ALLOWED TO TOUCH: backend/app/red_team/attack_genomes.py, backend/app/red_team/attack_injector.py, backend/tests/test_red_team.py
NOT ALLOWED TO TOUCH: the other 4 attack families, arena.py, blue_team/, frontend/, simulator/ (already done, don't touch it again)




CURRENT PHASE: Day 4 — features.py + LightGBM baseline
ALLOWED TO TOUCH: backend/app/blue_team/features.py, backend/app/blue_team/detector.py, backend/app/blue_team/graph_engine.py, backend/tests/test_blue_team.py
NOT ALLOWED TO TOUCH: arena.py, remaining attack families, explainability.py, frontend/, simulator/ and red_team/ (already done, don't touch again)


CURRENT PHASE: Day 5 — Adversarial Arena — MVP GATE, do not rush
ALLOWED TO TOUCH: backend/app/red_team/arena.py, backend/tests/test_arena.py
NOT ALLOWED TO TOUCH: everything else — this is a single-file focus day





CURRENT PHASE: Day 6 — attack family #3 (Behavioral Camouflage) ONLY
ALLOWED TO TOUCH: backend/app/red_team/attack_genomes.py (add genome #3, do not touch #1/#2), backend/app/red_team/attack_injector.py, backend/tests/test_red_team.py
NOT ALLOWED TO TOUCH: arena.py, micro_structuring/identity_drift's existing code, blue_team/, frontend/



CURRENT PHASE: Day 6 — attack family #4 (Social Engineering / Semantic Coercion) ONLY
ALLOWED TO TOUCH: backend/app/red_team/attack_genomes.py (add genome #4, do not touch #1/#2/#3), backend/app/red_team/attack_injector.py, backend/tests/test_red_team.py
NOT ALLOWED TO TOUCH: arena.py, existing families' code, blue_team/, frontend/



CURRENT PHASE: Day 6 — attack family #5 (Synthetic Voice/Video Authorization Fraud) ONLY — LAST attack family
ALLOWED TO TOUCH: backend/app/red_team/attack_genomes.py (add genome #5, do not touch #1-4), backend/app/red_team/attack_injector.py, backend/app/core/schemas.py (add voice_confidence_score field per addendum), backend/tests/test_red_team.py
NOT ALLOWED TO TOUCH: arena.py, existing families' code, blue_team/, frontend/


CURRENT PHASE: Day 6.5 — Genericize arena.py across all 5 attack families
ALLOWED TO TOUCH: backend/app/red_team/arena.py, backend/app/red_team/attack_injector.py (only if a shared dispatcher/registry needs adding there), backend/tests/test_arena.py
NOT ALLOWED TO TOUCH: attack_genomes.py (genomes are final, don't edit), families' individual generator functions (reuse them, don't rewrite them), blue_team/, frontend/



CURRENT PHASE: Day 6 (final) — FastAPI wrapper around everything built so far
ALLOWED TO TOUCH: backend/app/api/endpoints.py, backend/app/api/websocket.py (if needed), backend/tests/test_api.py
NOT ALLOWED TO TOUCH: simulator/, red_team/, blue_team/ internals — call existing functions only, do not reimplement logic in route handlers; frontend/


CURRENT PHASE: Day 7 — Frontend Screen 1 of 5: Command Center ONLY
ALLOWED TO TOUCH: frontend/ (new directory), specifically frontend/src/app/, frontend/src/components/MetricCards.tsx, frontend/src/lib/api.ts
NOT ALLOWED TO TOUCH: the other 4 screens, backend/ (already done, don't touch)


CURRENT PHASE: Day 7-pre — extend MetricsResponse for Command Center KPIs
ALLOWED TO TOUCH: backend/app/api/endpoints.py ONLY (MetricsResponse model + /metrics handler + a small in-memory "last arena run" cache)
NOT ALLOWED TO TOUCH: everything else, including frontend/ (that resumes after this)

```
CURRENT PHASE: [not started]
ALLOWED TO TOUCH: [nothing yet — set this before Day 1]
NOT ALLOWED TO TOUCH: [everything else]
```

Example of how this should look mid-build:
```
CURRENT PHASE: Day 5 — Adversarial Arena (MVP gate)
ALLOWED TO TOUCH: backend/app/red_team/arena.py, backend/tests/test_arena.py
NOT ALLOWED TO TOUCH: frontend/, remaining attack families, API layer
```

---

## 3. Architecture

```
Payment Twin (clean synthetic data, statistically realistic)
        ↓
Red Team (Attack Genomes → injected fraud transactions)
        ↓
Feature Engineering Pipeline (vectorized: velocity + behavioral + graph)
        ↓
Blue Team (LightGBM risk engine + NetworkX graph features + SHAP)
        ↓
Adversarial Arena (measures evasion, harvests hard negatives, retrains)
        ↓
Auto-Hardening → Re-test → Adversarial Robustness Gain (ARG) reported
        ↓
loop closes, feeds back into Red Team
```

**Design rule that must never be violated:** the LLM never scores a
transaction and never makes a fraud/allow/block decision. It only ever
produces a structured Attack Genome (JSON, Pydantic-validated) or short
synthetic social-engineering text. All scoring is deterministic ML
(LightGBM + engineered features). This is what makes the system explainable
and audit-safe — it is a stated architectural strength, not a limitation to
work around.

---

## 4. Scope

### 4.1 In scope — build exactly this, nothing more, nothing less

- Synthetic payment world: 10,000 customers, 500 merchants, 50,000+
  transactions over 30 days
- **Exactly 5 attack families** (genomes below are canonical — copy exactly,
  do not paraphrase parameters):

  **1. Agentic Micro-Structuring**
  ```json
  {
    "genome_id": "ATK-MS-001",
    "family": "micro_structuring",
    "objective": "bypass_single_transaction_and_velocity_thresholds",
    "target_amount": 50000.0,
    "parameters": {
      "split_count_range": [10, 15],
      "amount_per_tx_range": [2500, 4800],
      "time_window_hours": 48,
      "recipient_count": 4,
      "inter_arrival_distribution": "exponential"
    },
    "behavioral_camouflage": {
      "interleave_legitimate_noise": true,
      "noise_ratio": 0.35,
      "merchant_diversity": "high"
    },
    "mutations": ["increase_time_spacing", "rotate_mule_accounts", "add_legitimate_micro_purchases"]
  }
  ```

  **2. Synthetic Identity Drift** — genome_id `ATK-ID-001`. Account behaves
  normally for ~20 days (build trust), then executes a sudden high-velocity
  extraction via a new device and new payee.

  **3. Behavioral Camouflage** — genome_id `ATK-BC-001`. Fraudulent
  transactions interleaved within authentic-looking spending bursts.

  **4. Social Engineering / Semantic Coercion (APP Fraud)** — genome_id
  `ATK-SC-001`. Structurally normal transaction, adversarial memo metadata
  (urgency, impersonation). Real-world analogue: UPI Collect Request scams.

  **5. Synthetic Voice/Video Authorization Fraud**
  ```json
  {
    "genome_id": "ATK-VD-001",
    "family": "synthetic_voice_authorization",
    "objective": "bypass_step_up_authentication_via_impersonated_voice_call",
    "parameters": {
      "impersonated_role": ["bank_agent", "executive", "family_member"],
      "urgency_score_range": [0.7, 0.95],
      "requests_verification_bypass": true,
      "channel": "voice_authorized"
    },
    "evasion_targets": ["step_up_challenge", "identity_verification"],
    "mutations": ["vary_impersonated_role", "adjust_urgency_score", "combine_with_new_device"]
  }
  ```
  No real audio/video is ever generated — this is pure metadata simulation.

- LightGBM detector, vectorized feature engineering
- Full adversarial loop with constraint-aware mutations (see §4.3)
- Web dashboard, exactly 5 screens (§7)
- SHAP top-3 reason codes on every flagged transaction
- Judge Sandbox: free text → LLM → Pydantic-validated genome → live simulation

### 4.2 OUT OF SCOPE — do not build, even if it seems like a clear improvement

- Kubernetes, Kafka, microservices architecture
- LangGraph / CrewAI / AutoGen — plain Python functions for orchestration only
  (`discover_attack()`, `compile_attack()`, `simulate_attack()`,
  `score_attack()`, `mutate_attack()`, `harden_defense()`)
- GNNs / PyTorch Geometric — NetworkX only
- CTGAN / diffusion models — rule-based NumPy/Faker generator only
- Real payment APIs, blockchain, computer vision, actual voice/video generation
- Reinforcement learning
- A 6th attack family
- More than one LLM provider
- The full Adversarial Robustness Toolbox, ONNX/gRPC serving, Polars
  migration, differential privacy, JA3 fingerprinting — these belong in the
  docx's Production Scale-Up section as *named future work*, never in the
  actual codebase for this submission

**If you catch yourself about to import a library not in §5, or write an
endpoint not in §6, stop and flag it instead of proceeding.**

### 4.3 Mutation constraint validation (applies to every genome mutation)

Before accepting any mutated genome for hard-negative generation, validate:
- mutated `amount` stays within `[0, customer.mean_spend * 20]`
- mutated timestamp sequence remains chronologically ordered per customer
- novelty flags (`is_new_device`, `is_new_beneficiary`) stay internally
  consistent with the mutation actually applied (don't mutate device identity
  but leave the flag unset)

This constraint-aware design is a real, citable technical differentiator:
naive perturbation-based attack generation produces physically impossible
transactions; Sentinel-X's mutations remain realizable. State this explicitly
in code comments and the docx.

---

## 5. Tech Stack — locked, no substitutions

| Layer | Technology |
|---|---|
| Frontend | Next.js (App Router) + TypeScript + Tailwind + shadcn/ui + Recharts |
| Backend API | FastAPI (Python 3.11+), Pydantic v2, WebSockets where needed |
| ML Engine | LightGBM + scikit-learn |
| Explainability | SHAP (`TreeExplainer`) |
| Graph modeling | NetworkX |
| Data generation | NumPy, Pandas, Faker |
| Storage | SQLite / Parquet / in-memory DataFrames |
| Deploy | Vercel (frontend) + Render or Railway (backend) |

`requirements.txt` allowlist: `fastapi`, `pydantic`, `uvicorn`, `lightgbm`,
`shap`, `scikit-learn`, `networkx`, `pandas`, `numpy`, `faker`, `pytest`,
`python-multipart`, `scipy` (for KS-statistic fidelity scoring). Do not add
anything else without explicit approval.

---

## 6. Data Schemas (Pydantic v2 — exact, do not modify field names or types)

```python
from pydantic import BaseModel, Field
from typing import Optional, List, Dict
from datetime import datetime

class CustomerProfile(BaseModel):
    customer_id: str
    base_location: str
    primary_devices: List[str]
    mean_spend: float
    spend_variance: float
    usual_merchants: List[str]
    usual_beneficiaries: List[str]

class TransactionBase(BaseModel):
    transaction_id: str
    timestamp: datetime
    customer_id: str
    merchant_id: str
    beneficiary_id: str
    amount: float
    currency: str = "INR"
    channel: str                          # POS, WEB, P2P, voice_authorized
    device_id: str
    ip_region: str
    location: str
    merchant_category: str
    semantic_risk_score: float = 0.0      # [0.0, 1.0]
    voice_confidence_score: float = 1.0   # [0.0, 1.0], lower = more convincing deepfake

class InjectedTransaction(TransactionBase):
    is_fraud: int = Field(default=0, ge=0, le=1)
    attack_family: Optional[str] = None
    genome_id: Optional[str] = None

class DetectionResult(BaseModel):
    transaction_id: str
    risk_score: float
    decision: str                         # ALLOW, STEP_UP, REVIEW, BLOCK
    reason_codes: List[Dict[str, str]]
    latency_ms: float

class ArenaRunSummary(BaseModel):
    run_id: str
    attack_family: str
    initial_evasion_rate: float
    final_evasion_rate: float
    robustness_gain: float
    hard_examples_count: int
    retrained_f1_score: float
```

**Feature engineering formulas (exact — do not approximate):**
- `amount_deviation_ratio = amount / (avg_amount_7d + 1e-6)`
- `Initial Evasion Rate = False Negatives / Total Fraud Transactions`
- `ARG (%) = ((Initial Evasion Rate − Final Evasion Rate) / Initial Evasion Rate) × 100`

**Decision thresholds (configurable at runtime, never hardcoded as constants
scattered through the codebase — one config source of truth):**
```
[0.00 – 0.35)  → ALLOW
[0.35 – 0.65)  → STEP-UP
[0.65 – 0.85)  → REVIEW
[0.85 – 1.00]  → BLOCK
```

---

## 7. API Contracts

- `POST /api/v1/simulate` — generate the payment twin dataset
- `POST /api/v1/detect` — score a batch of transactions
- `POST /api/v1/arena/run` — execute the full adversarial loop for one attack family
- `GET /api/v1/explain/{transaction_id}` — SHAP reason codes
- `GET /api/v1/metrics` — current global model metrics
- `POST /api/v1/sandbox/compile` — free text → validated genome → live simulation

No additional endpoints without approval. Endpoint handlers call existing
service-layer functions — never reimplement business logic in the route layer.

---

## 8. Frontend — Exactly 5 Screens

1. **Command Center** — KPI header (volume, F1, FPR, ARG), LIVE/SANDBOX status
2. **Red Team Lab** — attack family selector, scale slider, hardening trigger,
   live stream, Judge Sandbox input
3. **Payment Twin** — normal vs. counterfactual-attacked customer comparison
4. **Blue Team SOC** — precision/recall/F1/FPR/latency, color-coded feed,
   click-through SHAP modal
5. **Adversarial Arena** — live loop visualization, evasion-drop chart, ARG headline

Theme: dark mode SOC aesthetic, neon green (ALLOW) / neon red (BLOCK). No 6th
screen. Every number displayed must come from a real API call, never mock data.

---

## 9. Repository Structure (strict contract)

```
sentinel-x/
├── CLAUDE.md                        # this file
├── PRD_SENTINEL_X.md
├── backend/
│   ├── app/
│   │   ├── main.py
│   │   ├── core/
│   │   │   ├── config.py            # decision thresholds, seeds, single source of truth
│   │   │   └── schemas.py           # §6 Pydantic models, exact
│   │   ├── simulator/
│   │   │   ├── clean_generator.py
│   │   │   └── fidelity.py          # KS statistic + JS divergence
│   │   ├── red_team/
│   │   │   ├── attack_genomes.py    # all 5 genomes, exact JSON from §4.1
│   │   │   ├── attack_injector.py
│   │   │   └── arena.py             # the MVP-gate loop
│   │   ├── blue_team/
│   │   │   ├── features.py          # vectorized only
│   │   │   ├── graph_engine.py
│   │   │   ├── detector.py
│   │   │   └── explainability.py
│   │   └── api/
│   │       ├── endpoints.py         # §7 routes only
│   │       └── websocket.py
│   ├── requirements.txt
│   └── tests/
│       └── test_*.py                # one per core module, non-optional
├── frontend/
│   └── src/
│       ├── app/
│       ├── components/
│       └── lib/api.ts
└── README.md
```

---

## 10. Non-Functional Requirements (Security, Scalability, Code Quality)

- **Security/privacy:** zero real PII anywhere in repo or demo data. State
  this explicitly as compliance-by-design.
- **Reproducibility:** fixed seeds everywhere — identical results on every
  fresh clone.
- **Code quality:** full type hints, docstrings, Pydantic validation at every
  I/O boundary, modular separation per §9, a test alongside every core function.
- **Performance:** report actual measured `/detect` latency — never an
  estimated or assumed number.
- **Auditability:** every BLOCK decision produces a SHAP-backed reason code.
- **Demo reliability:** the full 5-screen flow must run error-free end to end
  — this matters more than any single feature's visual polish.

Production-scale techniques (feature stores, Kafka, GNNs, ART, Polars, ONNX)
are **named in the docx's Production Scale-Up section only** — never built
here. See `PRD_SENTINEL_X.md` §13 for exact wording to use.

---

## 11. Build Order — strict, sequential, do not reorder

1. Synthetic customer + normal transaction generator (fidelity-scored)
2. ONE attack family (micro-structuring) generator, sum-of-splits verified
3. Feature engineering (vectorized) + LightGBM baseline detector
4. **Full adversarial loop, end-to-end — MVP gate.** Do not proceed past this
   step until evasion rate demonstrably drops after retraining, verified
   against a held-out mutated set (not training rows).
5. Remaining 4 attack families, one at a time, each individually verified
   through the arena loop before starting the next
6. FastAPI wrapper around everything built so far
7. Frontend, one screen at a time, wired to the real running API — never mock data
8. SHAP explainability + Judge Sandbox
9. Deploy, README, docx, rehearse demo

**Never build frontend before the backend loop works. Never build multiple
attack families before one full loop is proven end-to-end.**

---

## 12. Verification Discipline (apply to every task, every session)

After any code is written:
1. Run it. Paste the real terminal output — not a description of expected output.
2. Sanity-check the output yourself before declaring the task done:
   - Row counts match what was requested?
   - Distributions look plausible (not uniform, not identical, not all-zero)?
   - F1/evasion numbers look real (not suspiciously perfect)?
3. If output doesn't match expectations, **diagnose before patching**: state
   what you think went wrong and why, referencing the exact function/file,
   then propose a fix and wait for confirmation before applying it.
4. Commit to git after each verified unit of work:
   `git add -A && git commit -m "<what was verified>"`

---

## 13. Working Style Rules

- Build in small, clearly-scoped stages — one task per response, even if
  asked to "build everything." Reference §2 CURRENT BUILD PHASE.
- If a request would violate §4.2 (out of scope), flag it rather than
  silently complying, and explain what the in-scope alternative is.
- If a request is ambiguous, ask one specific clarifying question rather than
  guessing across the whole architecture.
- Keep "why" explanations short — this file and the PRD already cover
  reasoning. Focus responses on implementation and verification output.
- Every number that ends up in the demo or docx must trace back to a real
  terminal output from this session — never round, estimate, or "typical
  range" a metric.

---

**This file does not change mid-build.** If a build decision seems to require
changing something here, stop and get explicit re-approval rather than
drifting from it.
