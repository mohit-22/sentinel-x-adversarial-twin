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
