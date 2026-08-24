# PRODUCT REQUIREMENTS DOCUMENT — SENTINEL-X
### Autonomous Adversarial Payment Twin

**Competition:** Mastercard Innovation Challenge @ Global Fintech Fest (GFF) 2026 — "AI Defense Lab for Payment Security"
**Deadline:** 31 August 2026, 11:59 PM IST
**Status:** FINAL — v3 (locked)
**Document owner:** You. Claude Code implements against this; it does not revise it.

---

## 1. Vision & Problem Statement

Generative AI has collapsed the cost and skill barrier to sophisticated payment
fraud: automated social engineering, synthetic identities, deepfake-authorized
transfers, and structuring attacks that adapt faster than static, rule-based
defenses can be manually updated. Real-time payment rails (UPI, FedNow, RTP)
have simultaneously compressed the fraud-review window to under a few hundred
milliseconds, eliminating the human-in-the-loop review that legacy systems
relied on.

**Sentinel-X** is a closed-loop adversarial AI defense system. It does not ship
a static classifier. It ships a system that continuously synthesizes a
realistic payment network, programs GenAI-style fraud attacks against itself,
measures how much fraud slips through, and automatically hardens its own
detector against exactly the fraud it just missed — then proves the improvement
with a real number.

**One-line pitch (use this verbatim in the docx and demo):**

> "Sentinel-X is an adversarial payment twin that continuously invents,
> simulates, and evolves GenAI-powered fraud attacks against its own detector,
> then learns from every successful evasion to automatically harden the
> defense."

Never pitch this as "an AI fraud detector." It is an attack/defend/learn loop.
That framing is the entire competitive differentiator — say it explicitly to
judges, don't make them infer it.

---

## 2. Objective

Ship a working, hosted, judge-operable prototype demonstrating one complete
closed loop, end to end, with real (not simulated-for-the-demo) numbers:

```
attack → detect → some evade → evaded cases become training data →
retrain → re-attack (mutated) → evasion rate measurably drops → report the gain
```

### Core differentiators (in priority order — build in this order if time runs short)

1. **Adversarial Arena** — the live attack→evade→harden→retest loop. This is
   the single highest-value feature. It directly answers "novelty," "detection
   efficacy," and "real-world feasibility" simultaneously.
2. **Attack Genomes** — every fraud strategy is structured, mutation-capable
   JSON, never hardcoded logic. Proves the system is programmable, not scripted.
3. **Explainable decisions** — SHAP-derived reason codes on every flagged
   transaction. This is what makes the system deployable, not just accurate.
4. **Judge Sandbox** — a judge types a plain-English attack idea; the system
   compiles it to a genome, simulates it, and scores it live. This is your
   proof that the system generalizes rather than replaying a canned script.

---

## 3. Success Criteria — Mapped to the Actual Judging Rubric

| Judging criterion | How Sentinel-X addresses it |
|---|---|
| **Diversity of attacks identified** | 5 distinct attack families (4 core + 1 voice/deepfake), each a structured genome with defined mutation variants — not ad hoc examples |
| **Fidelity of attacks in simulation** | Synthetic transactions built from realistic per-customer statistical distributions; fidelity is *measured* via KS statistic and Jensen–Shannon divergence, never asserted |
| **Detection algorithm efficacy** | LightGBM classifier with measured precision/recall/F1/PR-AUC/FPR on a held-out set that includes attack variants never seen in training |
| **Novelty of the solution** | The closed adversarial loop + live Judge Sandbox — a system that tests and improves itself, not a static classifier wrapped in a dashboard |
| **Real-world feasibility** | Explainable decisions (SHAP), configurable risk thresholds, measured inference latency, and an explicit Production Scale-Up section acknowledging what a real deployment needs |

---

## 4. Users / Audience

- **Primary:** Hackathon judges operating the live prototype and reading the
  docx walkthrough. Optimize every screen for a 5-minute unguided demo.
- **Framing audience (for the "real-world feasibility" narrative):** a payments
  risk team at a bank/fintech running this internally as a red-team-as-a-service
  tool that continuously stress-tests their production fraud model.

---

## 5. System Architecture

```
Payment Twin (clean synthetic data, statistically realistic)
        ↓
Red Team (Attack Genomes → injected fraud transactions)
        ↓
Feature Engineering Pipeline (vectorized, velocity + behavioral + graph)
        ↓
Blue Team (LightGBM risk engine + NetworkX graph features + SHAP)
        ↓
Adversarial Arena (measures evasion, harvests hard negatives, retrains)
        ↓
Auto-Hardening → Re-test → Adversarial Robustness Gain reported
        ↓
loop closes, feeds back into Red Team
```

**Non-negotiable design rule:** the LLM never scores a transaction and never
makes a fraud decision. It only ever produces a structured Attack Genome (JSON)
or short synthetic social-engineering text. All scoring is deterministic
ML (LightGBM + engineered features), which is what makes the system audit-safe
and explainable — say this explicitly to judges, it's a real architectural
strength, not a limitation.

---

## 6. Scope

### 6.1 In scope (MVP — must ship, no exceptions)

- Synthetic payment world: 10,000 customers, 500 merchants, 50,000+
  transactions over a 30-day window, statistically realistic (log-normal
  spend, diurnal timing, >95% device persistence per customer)
- **5 attack families**, each a structured, mutation-capable Attack Genome,
  injected at ~2–3% of total transaction volume:
  1. Agentic Micro-Structuring (Smurfing)
  2. Synthetic Identity Drift
  3. Behavioral Camouflage
  4. Social Engineering / Semantic Coercion (APP Fraud)
  5. Synthetic Voice/Video Authorization Fraud (deepfake-triggered)
- LightGBM detector with vectorized behavioral + graph feature engineering
- Full adversarial loop: attack → score → measure evasion → harvest hard
  negatives → retrain → re-score against mutated attack → report Adversarial
  Robustness Gain (ARG)
- Constraint-aware Mutation Engine (mutations must respect realistic bounds —
  see §7.4)
- Web dashboard, exactly 5 screens (Command Center, Red Team Lab, Payment
  Twin, Blue Team SOC, Adversarial Arena)
- SHAP-based explanation (top-3 reason codes) for every flagged transaction
- Judge Sandbox: free-text attack description → LLM compiles to a validated
  genome → live simulation → live score
- India/UPI-specific framing throughout (see §7.2) — free credibility given
  the event's market

### 6.2 Explicitly OUT of scope — do not build even if it seems like an improvement

- Kubernetes, Kafka, microservices architecture
- LangGraph / CrewAI / AutoGen as dependencies — plain Python orchestration only
- GNNs / PyTorch Geometric — NetworkX is sufficient for the graph layer
- CTGAN / diffusion models for data generation — rule-based NumPy/Faker only
- Real payment APIs, blockchain, computer vision, actual voice/video generation
  (the voice-fraud attack family is *metadata-simulated*, never real audio/video)
- Reinforcement learning training loops
- More than 5 attack families
- Multiple LLM providers — one LLM, used only for genome-generation text
- The full Adversarial Robustness Toolbox (ART), ONNX/gRPC serving, Polars
  migration, differential privacy training, JA3 fingerprinting — these are
  legitimate production techniques and belong in the docx's Production
  Scale-Up section (§13), named correctly, but never built for this prototype

---

## 7. Functional Specifications

### 7.1 Pillar 1 — Payment Twin (Synthetic Data Simulator)

**Goal:** a hyper-realistic, statistically sound payment network containing
zero real PII.

Fidelity requirements:
- Amount distributions: log-normal spend curves, differentiated by 3 customer
  income tiers
- Temporal distributions: diurnal pattern (heavy 09:00–21:00, light overnight)
- Entity persistence: each customer sticks to 1–2 primary devices and a
  recurring set of merchants/beneficiaries (persistence probability > 0.95)
- Fidelity is **measured**, not asserted: Kolmogorov–Smirnov statistic on
  amount distributions, Jensen–Shannon divergence on temporal distributions.
  Target > 90% similarity to the intended reference distribution.

### 7.2 Pillar 2 — Red Team (Attack Genomes)

Every attack is a structured JSON genome, never hardcoded logic. An LLM may
help *author* new genomes (for the Judge Sandbox), but a deterministic Python
simulator always converts genome → transactions. India/UPI framing is applied
to the descriptions (not the underlying mechanics) as follows:

1. **Agentic Micro-Structuring** — a large transfer sliced into sub-threshold,
   log-normal transactions routed across mule accounts to evade velocity/AML
   triggers. Frame explicitly as modeling **UPI-style structuring across
   multiple payment apps**, not generic card structuring.

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

2. **Synthetic Identity Drift** — an account behaves normally for ~20 days,
   then executes a sudden high-velocity extraction via a new device and new
   payee. Frame as modeling **dormant-account reactivation**, a documented
   Indian mule-network technique (financial-inclusion accounts opened, later
   reactivated for layering).

3. **Behavioral Camouflage** — fraudulent transactions interleaved within
   authentic-looking spending bursts to corrupt short-term anomaly baselines.

4. **Social Engineering / Semantic Coercion (APP Fraud)** — a structurally
   normal transaction paired with adversarial memo metadata (urgency,
   impersonation, or simulated prompt-injection payloads aimed at a
   *hypothetical* downstream LLM dispute parser — sandboxed only, never
   targets any real system). Frame explicitly as modeling **UPI Collect
   Request scams**, a well-documented real fraud pattern.

5. **Synthetic Voice/Video Authorization Fraud** — models a deepfake-triggered
   step-up authentication bypass. No real audio/video is generated — this is
   pure metadata simulation.

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

   `TransactionBase.channel` supports the value `"voice_authorized"`; add a
   field `voice_confidence_score: float = 1.0` (lower = more convincing
   simulated deepfake), which feeds into `semantic_risk_score`.

Each genome supports mutation and can be re-run post-mutation. See §7.4 for
mutation validity constraints.

### 7.3 Pillar 3 — Blue Team (Detector & Explainability)

**Goal:** detect fraud with low latency and explain every decision.

Vectorized feature engineering (Pandas/NumPy `groupby().rolling()` — never
row-wise Python loops):
- Velocity: rolling transaction counts and sums (5m / 1h / 24h / 7d windows)
- Behavioral: `amount_deviation_ratio = amount / (avg_amount_7d + ε)`
- Graph (NetworkX): beneficiary in-degree/out-degree, shared-device count
  across distinct customers, 2-hop neighbor risk
- Novelty flags: `is_new_device`, `is_new_location`, `is_new_beneficiary`
- Semantic risk score `[0.0, 1.0]` from a lightweight signal pass over
  synthetic transaction memos (urgency/impersonation/coercion keywords)

Detection engine: LightGBM binary classifier predicting `is_fraud`, output a
risk score in `[0, 1]`.

Decision matrix (thresholds must be configurable at runtime, never hardcoded):
```
[0.00 – 0.35)  → ALLOW
[0.35 – 0.65)  → STEP-UP
[0.65 – 0.85)  → REVIEW
[0.85 – 1.00]  → BLOCK
```

Explainability: SHAP `TreeExplainer` providing the top-3 local feature
attributions per flagged transaction:
```json
{
  "transaction_id": "tx-84920",
  "risk_score": 0.89,
  "decision": "BLOCK",
  "reason_codes": [
    {"feature": "velocity_1h_count", "contribution": "+0.28", "description": "High transaction frequency within 1 hour"},
    {"feature": "is_new_device", "contribution": "+0.22", "description": "Transaction initiated from an unrecognized device"},
    {"feature": "beneficiary_in_degree", "contribution": "+0.19", "description": "Beneficiary receiving funds from unusual entity cluster"}
  ]
}
```

### 7.4 Pillar 4 — Adversarial Arena (Self-Hardening Loop)

1. Execute a specific Attack Genome against the baseline Blue Team model M₀.
2. Measure `Initial Evasion Rate = False Negatives / Total Fraud Transactions`.
3. Isolate evaded transactions. Apply bounded, **constraint-aware** mutations
   to synthesize hard-negative training examples. A mutation is only accepted
   if it passes validation:
   - mutated amount stays within `[0, customer.mean_spend * 20]`
   - mutated timestamp sequence remains chronologically ordered per customer
   - novelty flags (`is_new_device`, etc.) stay internally consistent with
     the mutation actually applied
   - This constraint-aware design is a real, citable differentiator: naive
     perturbation-based attack generation produces mathematically impossible
     transactions; Sentinel-X's mutations remain physically realizable.
4. Retrain M₀ → M₁ using hard-negative-enriched training data.
5. Re-test against **mutated** variants of the same genome (never the exact
   rows used in retraining — that would be leakage). Measure
   `Final Evasion Rate`.
6. Report **Adversarial Robustness Gain (ARG)**, the headline metric:

   ```
   ARG (%) = ((Initial Evasion Rate − Final Evasion Rate) / Initial Evasion Rate) × 100
   ```

---

## 8. Data Schemas (Pydantic v2 — exact contract, do not deviate)

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
    channel: str                      # POS, WEB, P2P, voice_authorized
    device_id: str
    ip_region: str
    location: str
    merchant_category: str
    semantic_risk_score: float = 0.0  # [0.0, 1.0]
    voice_confidence_score: float = 1.0  # [0.0, 1.0], lower = more convincing deepfake

class InjectedTransaction(TransactionBase):
    is_fraud: int = Field(default=0, ge=0, le=1)
    attack_family: Optional[str] = None
    genome_id: Optional[str] = None

class DetectionResult(BaseModel):
    transaction_id: str
    risk_score: float
    decision: str                     # ALLOW, STEP_UP, REVIEW, BLOCK
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

---

## 9. Tech Stack (locked — no substitutions without explicit approval)

| Layer | Technology | Constraint |
|---|---|---|
| Frontend | Next.js (App Router) + TypeScript + Tailwind + shadcn/ui | Dark-mode SOC aesthetic; charts via Recharts |
| Backend API | FastAPI (Python 3.11+), Pydantic v2 | Fully async, strict type validation, modular services |
| ML Engine | LightGBM + scikit-learn | Fast training, optimized for tabular data |
| Explainability | SHAP (TreeExplainer) | Fast local attribution, real-time capable |
| Graph modeling | NetworkX | In-memory degree/centrality — no GNN |
| Data generation | NumPy, Pandas, Faker | Deterministic, vectorized — no CTGAN/diffusion |
| Storage | SQLite / Parquet / in-memory DataFrames | Zero complex DB overhead for MVP |
| Deploy | Vercel (frontend) + Render/Railway (backend) | |

Coding directives:
- No Kafka, Kubernetes, microservices, or external GNNs.
- All feature transformations vectorized (Pandas/NumPy) — no per-row loops.
- Fixed random seeds (NumPy, Faker) everywhere — 100% reproducible for judges.
- Full type hints + docstrings on every function; Pydantic v2 on every I/O boundary.

---

## 10. API Contracts (REST / FastAPI)

- `POST /api/v1/simulate` — `{"total_customers": 10000, "total_transactions": 50000, "inject_fraud": true, "attack_families": [...]}` → summary counts + fidelity score
- `POST /api/v1/detect` — batch of `TransactionBase` → array of `DetectionResult`
- `POST /api/v1/arena/run` — `{"attack_family": "micro_structuring", "mutation_intensity": "medium"}` → full `ArenaRunSummary`
- `GET /api/v1/explain/{transaction_id}` → SHAP reason codes for the frontend modal
- `GET /api/v1/metrics` — current model metrics (F1, precision, recall, PR-AUC, FPR, latency p95, ARG)
- `POST /api/v1/sandbox/compile` — `{"description": "free text"}` → LLM-compiled, Pydantic-validated genome → live simulation result

---

## 11. Frontend — Exactly 5 Screens

Theme: dark mode, cybersecurity SOC aesthetic — deep grays, neon green (ALLOW),
neon red (BLOCK/attack).

1. **Command Center** — KPI header strip (total volume, F1, FPR, ARG), system
   status indicator (LIVE/SANDBOX)
2. **Red Team Lab** — attack family dropdown (5 families), scale slider,
   "Trigger Adversarial Hardening" button, live transaction stream, Judge
   Sandbox free-text input
3. **Payment Twin** — pick a synthetic customer, compare normal behavior vs.
   an attacked (counterfactual) version side by side
4. **Blue Team SOC** — precision/recall/F1/FPR/latency, color-coded live feed,
   click any BLOCKED row → SHAP reason-code modal
5. **Adversarial Arena** — the money screen: live attack→evade→harden→retest
   sequence, evasion-rate-drop chart, ARG headline number

No sixth screen. No feature added here that isn't in §6.1.

---

## 12. Requirements

### Functional
- Generate ≥50,000 synthetic normal transactions with realistic per-customer behavior
- Generate fraud for all 5 attack families, each parameterized by a genome
- Support genome mutation and re-running the mutated attack
- Score every transaction in `[0,1]`, map to a decision via configurable thresholds
- Report precision/recall/F1/PR-AUC/FPR on a held-out set including unseen attack variants
- Run a full adversarial cycle and report before/after evasion rate + ARG
- Explain any flagged transaction with SHAP top-3 contributing factors
- Accept free-text attack description → validated genome → live simulation → live score
- All data and testing fully sandboxed — no real cardholder data, no live-system targets

### Non-functional (Security, Scalability, Code Quality)
- **Latency:** report actual measured `/detect` inference latency (target: low
  double-digit ms per transaction batch)
- **Reproducibility:** fixed seeds everywhere; a fresh clone must reproduce
  identical demo numbers
- **Code quality:** full type hints, docstrings, Pydantic validation at every
  I/O boundary, modular separation (`simulator/`, `red_team/`, `blue_team/`,
  `api/`), unit tests alongside every core function
- **Data privacy:** zero real PII anywhere in the repo, dataset, or demo —
  state this explicitly in the docx as a compliance-by-design choice
- **Demo reliability:** the frontend must run error-free through a full live
  demo — this matters more than any single feature's visual polish
- **Auditability:** every BLOCK decision must produce a SHAP-backed reason
  code — this is what makes the system regulator-facing, not just accurate

---

## 13. Production Scale-Up Path (docx section — writing only, zero code)

One page in the docx. This is the highest-leverage, lowest-cost section in
the entire deliverable — it's what separates "we built a working demo" from
"we understand what this needs to become production-grade at Mastercard
scale." Name these techniques correctly; do not claim any of them were built:

- **Feature store:** the prototype's in-memory feature computation would move
  to a dedicated low-latency store (e.g. Redis-backed) to guarantee
  train/serve consistency at production transaction volumes.
- **Streaming ingestion:** the batch generator would be replaced by
  event-streaming ingestion (Kafka or equivalent) for real transaction throughput.
- **Faster feature computation:** at production scale, sliding-window
  aggregation would benefit from a columnar, multithreaded processing layer
  (e.g. Polars) rather than single-threaded Pandas.
- **Graph detection at scale:** the NetworkX-based graph features demonstrate
  the mule-network detection concept; at production scale this extends to
  Graph Neural Networks (GraphSAGE/GAT-style) for multi-hop laundering
  detection across millions of accounts.
- **Model hardening at scale:** the Adversarial Arena is a working prototype
  of a concept that in production would run continuously via a dedicated
  adversarial-training toolkit (e.g. IBM's Adversarial Robustness Toolbox)
  rather than the custom loop built for this demo.
- **Explainability/compliance:** the SHAP reason codes already double as an
  audit trail, satisfying explainability requirements regulators expect from
  production credit/fraud decision systems (relevant given RBI's 2026
  risk-based authentication mandate for Indian payment rails).
- **Network-level intelligence:** production deployment would benefit from
  integration with shared fraud-signal registries across institutions
  (conceptually similar to RBI Innovation Hub's DPIP initiative), rather than
  relying on single-institution data in isolation.

### 13.1 Known implementation approximations (disclose in docx limitations)

Running list of build-time approximations/known gaps surfaced during
implementation — each already flagged and approved in-session at the time,
tracked here so they land in the docx's limitations/future-work section
rather than being forgotten by submission time. Not fixed as part of this
list; each entry names the actual approved scope decision.

- **`extend_drift_window` mutation (Day 6.5, `arena.py` `MUTATION_REGISTRY`):**
  `synthetic_identity_drift`'s `extend_drift_window` mutation conceptually
  needs to regenerate the attack instance with a longer `drift_window_days`
  (a genome-parameter change), not transform existing rows. Approximated as
  a post-hoc timing shift via the generic `_stretch_timing` transform
  (approved Option (a) at the time) rather than proper regeneration, which
  would require `apply_mutation` to re-invoke the instance generator with a
  modified genome. Documented in code as a known limitation at the point of
  implementation.
- **`/payment-twin/{customer_id}` counterfactual `transaction_id` uniqueness
  (Day 7 Screen 4 planning):** the one freshly-generated counterfactual
  instance returned per call is only guaranteed unique *within that single
  call*. Calling the endpoint separately for multiple customers (as Screen
  4's Blue Team SOC feed does) can produce the same `transaction_id` for
  different customers, since instance-id numbering isn't scoped per caller/
  customer. Surfaced via a real React duplicate-key warning during Screen 4's
  CDP-driven verification; worked around on the frontend with a composite
  `customer_id + transaction_id` key. The underlying id-generation behavior
  itself was left untouched (out of that session's frontend-only scope).

---

## 14. Deliverables

1. Public GitHub repo, named exactly `TeamName`, organized per the repo
   structure in `CLAUDE.md` §Repository Structure, with a clear README
2. `TeamName.docx` — problem framing, attack taxonomy, generation methodology,
   detection architecture + real metrics, adversarial loop results (§7.4
   numbers), real-world feasibility (§13), limitations/future work
3. Hosted, working web prototype (Vercel + Render/Railway)

---

## 15. Milestones (9-day build, starting 22 Aug 2026)

| Day | Milestone |
|---|---|
| 1–2 | Synthetic payment world + normal transaction generator, fidelity-scored |
| 3 | First attack family (micro-structuring) generating labeled fraud, verified |
| 4 | LightGBM baseline detector with real precision/recall/F1 |
| 5 | **Full adversarial loop working end-to-end — MVP gate** |
| 6 | Remaining 4 attack families (including voice/deepfake) + FastAPI wrapper |
| 7 | Frontend: all 5 screens wired to live API, one at a time |
| 8 | SHAP explainability + Judge Sandbox |
| 9 | Deploy, polish, docx, rehearse demo, submit |

---

## 16. Risks & Mitigations

| Risk | Mitigation |
|---|---|
| Adversarial loop doesn't show a clean before/after improvement | Tune retraining on micro-structuring first (most controllable); treat Day 5 as the highest-priority engineering day, don't rush it |
| A "perfect" F1 score (≈1.0) on first training run | Almost always label leakage (e.g. `genome_id` left in features) — investigate before celebrating, never present an unverified perfect score |
| Frontend takes longer than planned | Backend/API must work and be demoable standalone via curl/Postman as fallback |
| Live demo fails during judging | Record a backup demo video by Day 8 |
| Scope creep from mid-build feature ideas | Anything not in §6.1 requires explicit re-approval before building — check `CLAUDE.md`'s out-of-scope list first |
| Inference latency creeps up | Keep the feature pipeline vectorized; profile `/detect` on Day 4, not Day 9 |

---

## 17. Open Questions (resolve before Day 1)

- Final team name (must match Kaggle team name exactly — used for repo and docx naming)
- Final LLM provider/API choice for genome-generation calls
- Confirm SQLite is sufficient for demo dataset size (it should be — ~50-60k rows)

---

**END OF PRD — this document does not change mid-build. If a build decision
seems to require changing something here, stop and re-approve explicitly
rather than drifting.**
