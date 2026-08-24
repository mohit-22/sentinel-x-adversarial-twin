/**
 * Typed client for the Sentinel-X backend (CLAUDE.md §7).
 *
 * Types below mirror the backend's ACTUAL response shapes exactly
 * (backend/app/api/endpoints.py's MetricsResponse, backend/app/core/schemas.py's
 * ArenaRunSummary) -- not an idealized/aspirational contract. If the backend
 * response shape changes, update these types to match, not the other way
 * around.
 */

const API_BASE_URL =
  process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://127.0.0.1:8000/api/v1";

/** Mirrors backend/app/core/schemas.py's ArenaRunSummary exactly. */
export interface ArenaRunSummary {
  run_id: string;
  attack_family: string;
  initial_evasion_rate: number;
  final_evasion_rate: number;
  robustness_gain: number;
  hard_examples_count: number;
  retrained_f1_score: number;
}

/**
 * Mirrors backend/app/api/endpoints.py's MetricsResponse exactly, including
 * test_set_size and latest_arena_run (added in the Day 7 metrics-extension
 * follow-up). latest_arena_run is null until at least one /arena/run call
 * has happened in the backend's current process lifetime -- an honest
 * "not computed yet" state, never a fabricated value.
 */
export interface MetricsResponse {
  precision: number;
  recall: number;
  f1: number;
  pr_auc: number;
  fpr: number;
  test_set_size: number;
  latest_arena_run: ArenaRunSummary | null;
}

export class ApiError extends Error {
  constructor(
    message: string,
    public readonly status?: number,
  ) {
    super(message);
    this.name = "ApiError";
  }
}

/**
 * GET /api/v1/metrics -- current global model metrics, held-out test set
 * size, and the most recent arena run (if any this session).
 */
export async function fetchMetrics(): Promise<MetricsResponse> {
  let response: Response;
  try {
    response = await fetch(`${API_BASE_URL}/metrics`, { cache: "no-store" });
  } catch {
    throw new ApiError(
      `Could not reach backend at ${API_BASE_URL}/metrics -- is the server running?`,
    );
  }
  if (!response.ok) {
    throw new ApiError(
      `GET /metrics returned ${response.status} ${response.statusText}`,
      response.status,
    );
  }
  return (await response.json()) as MetricsResponse;
}

/**
 * The 5 canonical attack genomes (CLAUDE.md §4.1) -- genome_id/family pairs
 * exactly matching the backend's own _GENOME_REGISTRY. NOT fabricated data:
 * there is no "list genomes" endpoint in the locked §7 API contract, so this
 * static reference list is what the family dropdown is built from. If a 6th
 * family is ever added to attack_genomes.py, this list must be updated too.
 */
export interface KnownAttackGenome {
  genome_id: string;
  family: string;
  label: string;
}

export const KNOWN_ATTACK_GENOMES: KnownAttackGenome[] = [
  { genome_id: "ATK-MS-001", family: "micro_structuring", label: "Agentic Micro-Structuring" },
  { genome_id: "ATK-ID-001", family: "synthetic_identity_drift", label: "Synthetic Identity Drift" },
  { genome_id: "ATK-BC-001", family: "behavioral_camouflage", label: "Behavioral Camouflage" },
  {
    genome_id: "ATK-SC-001",
    family: "social_engineering_coercion",
    label: "Social Engineering / Semantic Coercion",
  },
  {
    genome_id: "ATK-VD-001",
    family: "synthetic_voice_authorization",
    label: "Synthetic Voice/Video Authorization Fraud",
  },
];

/**
 * POST /api/v1/arena/run -- executes the full adversarial loop for one
 * attack family. n_instances omitted means the backend's own official
 * default (2000) applies; passing it explicitly requests a faster,
 * non-official run. This call can take ~100+ seconds at the official
 * default -- callers must show a loading state for the full duration.
 */
export async function triggerArenaRun(
  genomeId: string,
  nInstances?: number,
): Promise<ArenaRunSummary> {
  const body: { genome_id: string; n_instances?: number } = {
    genome_id: genomeId,
  };
  if (nInstances !== undefined) {
    body.n_instances = nInstances;
  }

  let response: Response;
  try {
    response = await fetch(`${API_BASE_URL}/arena/run`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
  } catch {
    throw new ApiError(
      `Could not reach backend at ${API_BASE_URL}/arena/run -- is the server running?`,
    );
  }
  if (!response.ok) {
    let detail = response.statusText;
    try {
      const errorBody = await response.json();
      if (errorBody?.detail) detail = errorBody.detail;
    } catch {
      // response body wasn't JSON -- fall back to statusText
    }
    throw new ApiError(`POST /arena/run returned ${response.status}: ${detail}`, response.status);
  }
  return (await response.json()) as ArenaRunSummary;
}

/**
 * POST /api/v1/sandbox/compile -- free text -> validated genome -> live
 * simulation. Backend implementation is Day 8 work; today this ALWAYS
 * returns 501. Callers should check `error.status === 501` to show the
 * honest "not yet available" state, distinct from a genuine network/other
 * failure.
 */
/** Mirrors backend/app/core/schemas.py's CustomerProfile exactly. */
export interface CustomerProfile {
  customer_id: string;
  base_location: string;
  primary_devices: string[];
  mean_spend: number;
  spend_variance: number;
  usual_merchants: string[];
  usual_beneficiaries: string[];
}

/** Mirrors backend/app/core/schemas.py's TransactionBase exactly. */
export interface TransactionBase {
  transaction_id: string;
  timestamp: string;
  customer_id: string;
  merchant_id: string;
  beneficiary_id: string;
  amount: number;
  currency: string;
  channel: string;
  device_id: string;
  ip_region: string;
  location: string;
  merchant_category: string;
  semantic_risk_score: number;
  voice_confidence_score: number;
}

/** Mirrors backend/app/core/schemas.py's InjectedTransaction exactly. */
export interface InjectedTransaction extends TransactionBase {
  is_fraud: number;
  attack_family: string | null;
  genome_id: string | null;
}

/**
 * Mirrors backend/app/api/endpoints.py's PaymentTwinResponse exactly
 * (the approved 7th endpoint, Day 7 Screen 3 -- see CLAUDE.md §7).
 */
export interface PaymentTwinResponse {
  customer: CustomerProfile;
  normal_transactions: TransactionBase[];
  counterfactual_transactions: InjectedTransaction[];
}

/**
 * GET /api/v1/payment-twin/{customer_id} -- one customer's real clean
 * transaction history plus one freshly-generated counterfactual attack
 * instance for the requested attack_family.
 */
export async function fetchPaymentTwin(
  customerId: string,
  attackFamily: string,
): Promise<PaymentTwinResponse> {
  let response: Response;
  const url = `${API_BASE_URL}/payment-twin/${encodeURIComponent(customerId)}?attack_family=${encodeURIComponent(attackFamily)}`;
  try {
    response = await fetch(url, { cache: "no-store" });
  } catch {
    throw new ApiError(`Could not reach backend at ${url} -- is the server running?`);
  }
  if (!response.ok) {
    let detail = response.statusText;
    try {
      const errorBody = await response.json();
      if (errorBody?.detail) detail = errorBody.detail;
    } catch {
      // response body wasn't JSON -- fall back to statusText
    }
    throw new ApiError(`GET /payment-twin/${customerId} returned ${response.status}: ${detail}`, response.status);
  }
  return (await response.json()) as PaymentTwinResponse;
}

export async function compileSandbox(freeText: string): Promise<unknown> {
  let response: Response;
  try {
    response = await fetch(`${API_BASE_URL}/sandbox/compile`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text: freeText }),
    });
  } catch {
    throw new ApiError(
      `Could not reach backend at ${API_BASE_URL}/sandbox/compile -- is the server running?`,
    );
  }
  if (!response.ok) {
    let detail = response.statusText;
    try {
      const errorBody = await response.json();
      if (errorBody?.detail) detail = errorBody.detail;
    } catch {
      // response body wasn't JSON -- fall back to statusText
    }
    throw new ApiError(
      `POST /sandbox/compile returned ${response.status}: ${detail}`,
      response.status,
    );
  }
  return response.json();
}

/** Mirrors backend/app/core/schemas.py's DetectionResult exactly. */
export interface DetectionResult {
  transaction_id: string;
  risk_score: number;
  decision: string;
  reason_codes: Record<string, string>[];
  latency_ms: number;
}

/**
 * POST /api/v1/detect -- score a batch of transactions with the cached,
 * already-trained detector. reason_codes is honestly [] on every result --
 * SHAP is Day 8 work, not faked here.
 */
export async function detectTransactions(
  transactions: TransactionBase[],
): Promise<DetectionResult[]> {
  let response: Response;
  try {
    response = await fetch(`${API_BASE_URL}/detect`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ transactions }),
    });
  } catch {
    throw new ApiError(
      `Could not reach backend at ${API_BASE_URL}/detect -- is the server running?`,
    );
  }
  if (!response.ok) {
    let detail = response.statusText;
    try {
      const errorBody = await response.json();
      if (errorBody?.detail) detail = errorBody.detail;
    } catch {
      // response body wasn't JSON -- fall back to statusText
    }
    throw new ApiError(`POST /detect returned ${response.status}: ${detail}`, response.status);
  }
  const body = (await response.json()) as { results: DetectionResult[] };
  return body.results;
}

/**
 * GET /api/v1/explain/{transaction_id} -- SHAP reason codes. Backend
 * implementation is Day 8 work; today this ALWAYS returns 501. Callers
 * should check `error.status === 501` to show the honest "not yet
 * available" state, same pattern as compileSandbox.
 */
export async function explainTransaction(transactionId: string): Promise<unknown> {
  let response: Response;
  const url = `${API_BASE_URL}/explain/${encodeURIComponent(transactionId)}`;
  try {
    response = await fetch(url, { cache: "no-store" });
  } catch {
    throw new ApiError(`Could not reach backend at ${url} -- is the server running?`);
  }
  if (!response.ok) {
    let detail = response.statusText;
    try {
      const errorBody = await response.json();
      if (errorBody?.detail) detail = errorBody.detail;
    } catch {
      // response body wasn't JSON -- fall back to statusText
    }
    throw new ApiError(`GET /explain/${transactionId} returned ${response.status}: ${detail}`, response.status);
  }
  return response.json();
}
