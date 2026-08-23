# SENTINEL-X — ADDENDUM (Research-Backed Additions)

Read this alongside SENTINEL_X_MASTER_SPEC.md. This addendum adds four low-cost, high-credibility upgrades identified from deeper research. Nothing here changes the locked build order, timeline, or tech stack in the master spec. If any instruction below conflicts with the master spec's §5.4 (out of scope) or §14 (build order), the master spec wins.

---

## What NOT to add, despite the research suggesting it

A large external research document was reviewed covering GraphSAGE/GAT graph neural networks, the full Adversarial Robustness Toolbox (ART), ONNX+gRPC model serving, Kafka event streaming, differential privacy training, JA3 TLS fingerprinting, and a full migration from Pandas to Polars. **None of these are added.** Each is either too heavy to implement reliably in the remaining build days, or requires infrastructure explicitly excluded in the master spec (§5.4/§6). They are legitimate production techniques and can be *mentioned by name* in the docx's "Production Scale-Up Path" section (below) to show awareness — but none get built. Do not let future prompts talk you into implementing them under deadline pressure.

---

## Addition 1: Attack family #5 — Synthetic Voice/Video Authorization Fraud

This is the one net-new attack family worth adding, because deepfake/voice-cloning fraud is the fastest-growing 2026 fraud vector and no other hackathon team is likely to model it. It reuses the existing Attack Genome pattern from §5.1 of the master spec — no new architecture needed, just a new genome + a few new transaction fields.

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

Add two fields to `TransactionBase`: `channel` already supports a new value `"voice_authorized"`; add `voice_confidence_score: float = 1.0` (synthetic — lower values simulate a more convincing deepfake voice, feeds into `semantic_risk_score`).

This family only becomes buildable **after** the MVP gate (master spec §14 step 4) is done — treat it as replacing one of your "remaining 3 attack families" slots in step 5, not an addition on top of the existing 4.

---

## Addition 2: Constraint-aware mutations — a real academic grounding for the Mutation Engine

Research on adversarial attacks against tabular ML models (the kind LightGBM handles) makes a useful distinction: real attackers are constrained by **immutable features** (a customer's account age or date of birth can't be changed) and **inter-feature dependencies** (a transaction amount can't exceed available balance; a velocity count can't go negative). An attacker who ignores these constraints produces an unrealistic, physically impossible transaction.

Apply this to your existing Mutation Engine (master spec §5.1, `mutations` field) with one small addition: before accepting a mutated genome, validate it against a short constraint list:

- Mutated transaction amount must stay within `[0, customer.mean_spend * 20]` (a rough sanity bound)
- Mutated timestamp sequence must remain chronologically ordered per customer
- `is_new_device` / `is_new_beneficiary` flags must be internally consistent with the mutation applied (don't mutate device but leave the flag unset)

This costs almost nothing to add (a validation function called before a mutated genome is accepted) but lets you say in the docx: "unlike naive perturbation-based attack generation, our Mutation Engine is constraint-aware — every mutated attack remains a physically realizable transaction, not just a mathematical adversarial example." That's a real differentiator judges with ML backgrounds will recognize.

---

## Addition 3: India/UPI-specific framing — free credibility given the event location

The event is hosted in Mumbai, judged partly by an Indian fintech ecosystem, and your data already uses ₹ (INR). Make this explicit rather than incidental. This costs **zero new code** — it's descriptive framing on genomes and attack family names you already have:

- **Micro-structuring** → explicitly describe as modeling UPI-style structuring across multiple payment apps and AML reporting-threshold evasion, not generic card structuring.
- **Synthetic Identity Drift** → mention this maps to dormant-account reactivation patterns, a known Indian mule-network technique (low-income accounts opened for financial inclusion, later reactivated for fund layering).
- **Social Engineering / Semantic Coercion** → explicitly reference UPI Collect Request scams as the real-world analogue — a well-documented, extremely common fraud pattern in India specifically.
- In the docx's problem framing section, one sentence: "While the architecture is rail-agnostic, attack parameters and thresholds in this prototype are tuned to reflect patterns observed in India's real-time payment ecosystem (UPI)."

This turns your existing 5 attack families into "we understand the specific market this event is about," which is a distinct axis from raw technical sophistication and costs you nothing.

---

## Addition 4: "Production Scale-Up Path" section for the docx (writing only)

One page, zero code, highest leverage-per-minute item in this whole addendum. Explicitly acknowledge the gap between your SQLite/in-memory prototype and a real deployment, naming the real techniques (including the ones you correctly did NOT build):

- **Feature store**: note the prototype's in-memory feature computation would move to a dedicated low-latency feature store (e.g. Redis-backed) to guarantee train/serve consistency at production transaction volumes.
- **Streaming ingestion**: batch generator → event-streaming ingestion (Kafka or equivalent) for real transaction throughput.
- **Faster feature computation**: mention that at production scale, sliding-window aggregation (5m/1h/24h counts) would benefit from a columnar, multithreaded processing layer rather than single-threaded Pandas — name Polars specifically as the kind of tool this problem calls for, without claiming you migrated to it.
- **Graph detection at scale**: your NetworkX-based graph features demonstrate the *concept* of mule-network detection via account/device/beneficiary topology; at production scale this concept extends to graph neural networks (GraphSAGE/GAT-style) for multi-hop laundering detection across millions of accounts.
- **Model hardening**: your Adversarial Arena is a working prototype of a concept that, in production, would run continuously via a dedicated adversarial-training toolkit (e.g. IBM's Adversarial Robustness Toolbox) rather than the custom loop built for this demo.
- **Explainability/compliance**: your SHAP reason codes already double as an audit trail; note this satisfies the explainability requirements regulators expect from any production credit/fraud decision system.

Each bullet takes one sentence to write and directly answers "how would this actually work at Mastercard scale" — the single question most likely to trip up teams that only demo a working prototype without acknowledging what's next.

---

## Summary: what changes in the build order

Nothing in the master spec's §14 build order changes. The only concrete additions are:
1. One new attack genome (Addition 1) — slots into step 5 (remaining attack families)
2. One small validation function on the existing Mutation Engine (Addition 2) — near-zero cost, add whenever the Mutation Engine is touched
3. Text/copy changes to existing attack descriptions (Addition 3) — do this during step 11 (polish) or whenever writing UI copy
4. A new docx section (Addition 4) — part of step 11 (docx walkthrough)
