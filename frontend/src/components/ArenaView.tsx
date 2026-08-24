"use client";

import { useEffect, useRef, useState } from "react";
import { Bar, BarChart, CartesianGrid, Cell, ResponsiveContainer, XAxis, YAxis } from "recharts";

import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import {
  ApiError,
  KNOWN_ATTACK_GENOMES,
  fetchMetrics,
  triggerArenaRun,
  type ArenaRunSummary,
} from "@/lib/api";

function formatPercent(value: number, digits = 2): string {
  return `${(value * 100).toFixed(digits)}%`;
}

function familyLabel(family: string): string {
  return KNOWN_ATTACK_GENOMES.find((g) => g.family === family)?.label ?? family;
}

// --- Cross-Family Generalization Matrix (post-Day 8b differentiator) ------
//
// /arena/multi-family-run is a brand-new endpoint with no lib/api.ts client
// function -- api.ts isn't in this phase's ALLOWED_TO_TOUCH (only this new
// section of ArenaView.tsx is), so the fetch is self-contained here, same
// pattern ShapModal.tsx/RedTeamControls.tsx used for locally-typed response
// shapes when api.ts couldn't be touched.

const MATRIX_API_BASE_URL = process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://127.0.0.1:8000/api/v1";

interface FamilyMatrixResult {
  genome_id: string;
  initial_evasion_rate: number;
  final_evasion_rate: number;
  robustness_gain: number;
  hard_examples_count: number;
}

interface MultiFamilyRunResponse {
  per_family: Record<string, FamilyMatrixResult>;
  total_hard_examples_count: number;
  retrained_precision: number;
  retrained_recall: number;
  retrained_f1: number;
  retrained_fpr: number;
}

async function fetchMultiFamilyRun(nInstances?: number): Promise<MultiFamilyRunResponse> {
  const url = `${MATRIX_API_BASE_URL}/arena/multi-family-run`;
  let response: Response;
  try {
    response = await fetch(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(nInstances !== undefined ? { n_instances: nInstances } : {}),
    });
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
    throw new ApiError(`POST /arena/multi-family-run returned ${response.status}: ${detail}`, response.status);
  }
  return (await response.json()) as MultiFamilyRunResponse;
}

/** Evasion rate (0-1) -> a red(high)/green(low) heatmap color, reusing the
 * existing neon-red/neon-green tokens via color-mix (same technique already
 * used in BlueTeamSOC.tsx), not new hardcoded hex values.
 */
function evasionHeatColor(rate: number): string {
  const pct = Math.round(Math.min(Math.max(rate, 0), 1) * 100);
  return `color-mix(in oklch, var(--neon-red) ${pct}%, var(--neon-green))`;
}

const QUICK_N_INSTANCES = 500;
const OFFICIAL_MATRIX_N_INSTANCES = 2000;

/**
 * Builds the two-point "Before Hardening" / "After Hardening" chart data.
 * This is a 2-point comparison BY CONSTRUCTION -- /arena/run returns exactly
 * one initial/final evasion-rate pair, never a time series. A line chart or
 * anything implying a continuous curve or intermediate steps would
 * misrepresent what the backend actually measured, so this stays a bar
 * chart with exactly two categories.
 */
function chartData(result: ArenaRunSummary) {
  return [
    { stage: "Before Hardening", evasionRate: result.initial_evasion_rate * 100 },
    { stage: "After Hardening", evasionRate: result.final_evasion_rate * 100 },
  ];
}

/**
 * Real returned numbers only -- no templated placeholder text that could
 * be mistaken for live data if a field were ever missing.
 */
function narrativeSentence(result: ArenaRunSummary): string {
  return (
    `${familyLabel(result.attack_family)} initially evaded defenses on ` +
    `${formatPercent(result.initial_evasion_rate)} of held-out fraud attempts. ` +
    `After harvesting ${result.hard_examples_count.toLocaleString()} hard negatives and retraining, ` +
    `evasion dropped to ${formatPercent(result.final_evasion_rate)} -- ` +
    `a ${result.robustness_gain.toFixed(2)}% robustness gain.`
  );
}

/**
 * Screen 5: Adversarial Arena (CLAUDE.md §8) -- the demo narrative surface,
 * distinct in purpose from Screen 2's Red Team Lab (engineer configuration:
 * family + scale slider + quick-test mode + a Run Report data table). This
 * screen has no scale slider -- always the official n=2000 run -- and leads
 * with the ARG headline + narrative sentence + before/after chart; the
 * Run ID/family/hard-examples detail Screen 2 foregrounds is demoted to a
 * small footer here.
 *
 * Featured view is sourced ONLY from a real /metrics call
 * (latest_arena_run) on mount -- never a hardcoded baseline number, even as
 * a "well-intentioned" fallback, per §8's "every number from a real API
 * call, never mock data" rule. If no arena run has happened yet this
 * backend session, shows a plain honest empty state, same pattern as
 * Screen 1's MetricCards ARG card.
 */
export function ArenaView() {
  const [genomeId, setGenomeId] = useState(KNOWN_ATTACK_GENOMES[0].genome_id);
  const [result, setResult] = useState<ArenaRunSummary | null>(null);
  const [isFeaturedLoading, setIsFeaturedLoading] = useState(true);
  const [isRunning, setIsRunning] = useState(false);
  const [elapsedSeconds, setElapsedSeconds] = useState(0);
  const [error, setError] = useState<string | null>(null);

  const timerRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const [matrixResult, setMatrixResult] = useState<MultiFamilyRunResponse | null>(null);
  const [isMatrixRunning, setIsMatrixRunning] = useState(false);
  const [matrixElapsedSeconds, setMatrixElapsedSeconds] = useState(0);
  const [matrixError, setMatrixError] = useState<string | null>(null);
  const [useOfficialMatrixScale, setUseOfficialMatrixScale] = useState(false);
  const matrixTimerRef = useRef<ReturnType<typeof setInterval> | null>(null);

  useEffect(() => {
    return () => {
      if (matrixTimerRef.current) clearInterval(matrixTimerRef.current);
    };
  }, []);

  async function handleRunMatrix() {
    setIsMatrixRunning(true);
    setMatrixElapsedSeconds(0);
    setMatrixError(null);

    matrixTimerRef.current = setInterval(() => {
      setMatrixElapsedSeconds((s) => s + 1);
    }, 1000);

    try {
      const nInstances = useOfficialMatrixScale ? OFFICIAL_MATRIX_N_INSTANCES : QUICK_N_INSTANCES;
      const response = await fetchMultiFamilyRun(nInstances);
      setMatrixResult(response);
    } catch (err) {
      setMatrixError(err instanceof ApiError ? err.message : String(err));
    } finally {
      if (matrixTimerRef.current) clearInterval(matrixTimerRef.current);
      setIsMatrixRunning(false);
    }
  }

  useEffect(() => {
    fetchMetrics()
      .then((metrics) => {
        if (metrics.latest_arena_run) {
          setResult(metrics.latest_arena_run);
          setGenomeId(
            KNOWN_ATTACK_GENOMES.find((g) => g.family === metrics.latest_arena_run?.attack_family)
              ?.genome_id ?? KNOWN_ATTACK_GENOMES[0].genome_id,
          );
        }
      })
      .catch((err) => {
        setError(err instanceof ApiError ? err.message : String(err));
      })
      .finally(() => setIsFeaturedLoading(false));

    return () => {
      if (timerRef.current) clearInterval(timerRef.current);
    };
  }, []);

  async function handleTriggerRun() {
    setIsRunning(true);
    setElapsedSeconds(0);
    setError(null);

    timerRef.current = setInterval(() => {
      setElapsedSeconds((s) => s + 1);
    }, 1000);

    try {
      const summary = await triggerArenaRun(genomeId); // n_instances omitted -> official 2000
      setResult(summary);
    } catch (err) {
      setError(err instanceof ApiError ? err.message : String(err));
    } finally {
      if (timerRef.current) clearInterval(timerRef.current);
      setIsRunning(false);
    }
  }

  return (
    <div className="min-h-screen bg-background p-8 text-foreground">
      <div className="mx-auto max-w-4xl space-y-6">
        <header className="border-b border-border pb-4">
          <h1 className="text-xl font-semibold tracking-tight">Adversarial Arena</h1>
          <p className="text-sm text-muted-foreground">
            Attack finds a gap &rarr; defense hardens &rarr; gap closes
          </p>
        </header>

        <Card>
          <CardHeader>
            <CardTitle>Run Configuration</CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="flex flex-col gap-4 sm:flex-row sm:items-end">
              <div className="flex-1 space-y-1.5">
                <label htmlFor="arena-genome-select" className="text-sm font-medium text-muted-foreground">
                  Attack family
                </label>
                <select
                  id="arena-genome-select"
                  value={genomeId}
                  disabled={isRunning}
                  onChange={(e) => setGenomeId(e.target.value)}
                  className="w-full rounded-md border border-border bg-input/30 px-3 py-2 text-sm text-foreground disabled:opacity-50"
                >
                  {KNOWN_ATTACK_GENOMES.map((g) => (
                    <option key={g.genome_id} value={g.genome_id}>
                      {g.label} ({g.genome_id})
                    </option>
                  ))}
                </select>
              </div>
              <Button onClick={handleTriggerRun} disabled={isRunning}>
                {isRunning
                  ? `Running... (${elapsedSeconds}s elapsed)`
                  : "Trigger Official Run (n=2,000)"}
              </Button>
            </div>
            <p className="text-xs text-muted-foreground">
              Always the official n=2,000 methodology &mdash; no quick-test mode on this
              screen (that&apos;s Screen 2&apos;s job). A full run takes ~100s.
            </p>
            {isRunning && (
              <div className="space-y-1">
                <div className="h-1.5 w-full overflow-hidden rounded-full bg-muted">
                  <div className="h-full w-1/3 animate-pulse bg-primary" />
                </div>
                <p className="text-xs text-muted-foreground">
                  /arena/run is a single blocking call (no WebSocket in the locked tech
                  stack) &mdash; this is an elapsed-time indicator, not live
                  per-stage progress.
                </p>
              </div>
            )}
          </CardContent>
        </Card>

        {error && (
          <div className="rounded-lg border border-destructive/40 bg-destructive/10 p-3 text-sm text-destructive">
            {error}
          </div>
        )}

        {!isFeaturedLoading && !result && !isRunning && (
          <Card>
            <CardContent>
              <p className="border-l-2 border-dashed border-muted-foreground/50 pl-3 text-base font-medium text-muted-foreground">
                Run Adversarial Arena to compute
              </p>
              <p className="mt-1 text-xs text-muted-foreground">
                No arena run has completed in this session yet
              </p>
            </CardContent>
          </Card>
        )}

        {result && !isRunning && (
          <>
            <Card>
              <CardContent className="space-y-4 py-2">
                <div>
                  <p className="text-xs text-muted-foreground">Adversarial Robustness Gain</p>
                  <p
                    className="text-5xl font-bold tabular-nums"
                    style={{
                      color:
                        result.robustness_gain > 0
                          ? "var(--neon-green)"
                          : result.robustness_gain < 0
                            ? "var(--neon-red)"
                            : undefined,
                    }}
                  >
                    {result.robustness_gain > 0 ? "+" : ""}
                    {result.robustness_gain.toFixed(2)}%
                  </p>
                </div>
                <p className="text-sm leading-relaxed text-foreground">{narrativeSentence(result)}</p>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle>Evasion Rate: Before vs. After Hardening</CardTitle>
                <p className="text-xs text-muted-foreground">
                  Exactly two measured points -- /arena/run returns one initial/final
                  evasion-rate pair, not a time series. No intermediate steps are
                  measured or implied.
                </p>
              </CardHeader>
              <CardContent>
                <div className="h-64 w-full">
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart data={chartData(result)} margin={{ top: 8, right: 16, left: 0, bottom: 0 }}>
                      <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
                      <XAxis dataKey="stage" stroke="var(--muted-foreground)" fontSize={12} />
                      <YAxis
                        stroke="var(--muted-foreground)"
                        fontSize={12}
                        unit="%"
                        domain={[0, (max: number) => Math.max(10, Math.ceil(max * 1.2))]}
                      />
                      <Bar dataKey="evasionRate" radius={[4, 4, 0, 0]}>
                        <Cell fill="var(--neon-red)" />
                        <Cell fill="var(--neon-green)" />
                      </Bar>
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              </CardContent>
            </Card>

            <Card size="sm">
              <CardContent>
                <dl className="grid grid-cols-2 gap-x-4 gap-y-2 text-xs sm:grid-cols-4">
                  <div>
                    <dt className="text-muted-foreground">Run ID</dt>
                    <dd className="font-mono">{result.run_id}</dd>
                  </div>
                  <div>
                    <dt className="text-muted-foreground">Attack family</dt>
                    <dd>{familyLabel(result.attack_family)}</dd>
                  </div>
                  <div>
                    <dt className="text-muted-foreground">Hard examples harvested</dt>
                    <dd className="tabular-nums">{result.hard_examples_count.toLocaleString()}</dd>
                  </div>
                  <div>
                    <dt className="text-muted-foreground">Retrained F1</dt>
                    <dd className="tabular-nums">{formatPercent(result.retrained_f1_score)}</dd>
                  </div>
                </dl>
              </CardContent>
            </Card>
          </>
        )}

        <Card>
          <CardHeader>
            <CardTitle>Cross-Family Generalization Matrix</CardTitle>
            <p className="text-xs text-muted-foreground">
              M0 was trained only on micro_structuring. Harvests hard negatives from ALL
              5 families into one combined retrain (M-multi), then measures each family&apos;s
              evasion rate against that single model &mdash; not five independent retrains.
            </p>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
              <label className="flex items-center gap-2 text-xs text-muted-foreground">
                <input
                  type="checkbox"
                  checked={useOfficialMatrixScale}
                  disabled={isMatrixRunning}
                  onChange={(e) => setUseOfficialMatrixScale(e.target.checked)}
                  className="accent-primary"
                />
                Use official scale (n=2,000, ~8&ndash;10 min) instead of the quick run
              </label>
              <Button onClick={handleRunMatrix} disabled={isMatrixRunning} variant="secondary">
                {isMatrixRunning
                  ? `Running... (${matrixElapsedSeconds}s elapsed)`
                  : useOfficialMatrixScale
                    ? "Run Cross-Family Hardening (official, n=2,000)"
                    : `Run Cross-Family Hardening (quick, n=${QUICK_N_INSTANCES})`}
              </Button>
            </div>
            {!useOfficialMatrixScale && !isMatrixRunning && (
              <p className="text-xs" style={{ color: "var(--status-sandbox)" }}>
                QUICK TEST MODE &mdash; n={QUICK_N_INSTANCES}, a real reduced-scale run
                (~2 min), not the official n=2,000 methodology (~8&ndash;10 min).
              </p>
            )}
            {isMatrixRunning && (
              <div className="space-y-1">
                <div className="h-1.5 w-full overflow-hidden rounded-full bg-muted">
                  <div className="h-full w-1/3 animate-pulse bg-secondary-foreground/40" />
                </div>
                <p className="text-xs text-muted-foreground">
                  Harvests hard negatives from all 5 families, one combined retrain, then
                  re-tests each family &mdash; a single blocking call, elapsed-time indicator
                  only, same honesty pattern as the run above.
                </p>
              </div>
            )}

            {matrixError && (
              <div className="rounded-lg border border-destructive/40 bg-destructive/10 p-3 text-sm text-destructive">
                {matrixError}
              </div>
            )}

            {!matrixResult && !isMatrixRunning && !matrixError && (
              <p className="border-l-2 border-dashed border-muted-foreground/50 pl-3 text-sm text-muted-foreground">
                Run Cross-Family Hardening to compute
              </p>
            )}

            {matrixResult && !isMatrixRunning && (
              <>
                <div className="overflow-x-auto">
                  <table className="w-full text-sm">
                    <thead>
                      <tr className="border-b border-border text-left text-xs text-muted-foreground">
                        <th className="py-1.5 pr-4 font-medium">Attack family</th>
                        <th className="py-1.5 pr-4 font-medium">
                          M0 evasion (single-family baseline)
                        </th>
                        <th className="py-1.5 pr-4 font-medium">M-multi evasion (this run)</th>
                        <th className="py-1.5 font-medium">Robustness gain</th>
                      </tr>
                    </thead>
                    <tbody>
                      {Object.entries(matrixResult.per_family).map(([family, data]) => (
                        <tr key={family} className="border-b border-border/50">
                          <td className="py-1.5 pr-4">{familyLabel(family)}</td>
                          <td
                            className="py-1.5 pr-4 font-semibold tabular-nums"
                            style={{ color: evasionHeatColor(data.initial_evasion_rate) }}
                          >
                            {formatPercent(data.initial_evasion_rate)}
                          </td>
                          <td
                            className="py-1.5 pr-4 font-semibold tabular-nums"
                            style={{ color: evasionHeatColor(data.final_evasion_rate) }}
                          >
                            {formatPercent(data.final_evasion_rate)}
                          </td>
                          <td
                            className="py-1.5 tabular-nums"
                            style={{
                              color:
                                data.robustness_gain > 0
                                  ? "var(--neon-green)"
                                  : data.robustness_gain < 0
                                    ? "var(--neon-red)"
                                    : undefined,
                            }}
                          >
                            {data.robustness_gain > 0 ? "+" : ""}
                            {data.robustness_gain.toFixed(2)}%
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
                <p className="text-xs text-muted-foreground">
                  A negative robustness gain means evasion went up for that family under
                  M-multi, not down &mdash; reported as measured, not hidden.
                </p>

                <dl className="grid grid-cols-2 gap-x-4 gap-y-2 border-t border-border pt-3 text-xs sm:grid-cols-5">
                  <div>
                    <dt className="text-muted-foreground">Total hard examples</dt>
                    <dd className="tabular-nums">{matrixResult.total_hard_examples_count.toLocaleString()}</dd>
                  </div>
                  <div>
                    <dt className="text-muted-foreground">M-multi precision</dt>
                    <dd className="tabular-nums">{formatPercent(matrixResult.retrained_precision)}</dd>
                  </div>
                  <div>
                    <dt className="text-muted-foreground">M-multi recall</dt>
                    <dd className="tabular-nums">{formatPercent(matrixResult.retrained_recall)}</dd>
                  </div>
                  <div>
                    <dt className="text-muted-foreground">M-multi F1</dt>
                    <dd className="tabular-nums">{formatPercent(matrixResult.retrained_f1)}</dd>
                  </div>
                  <div>
                    <dt className="text-muted-foreground">M-multi FPR</dt>
                    <dd className="tabular-nums">{formatPercent(matrixResult.retrained_fpr)}</dd>
                  </div>
                </dl>
                <p className="text-xs text-muted-foreground">
                  Precision/recall/F1/FPR above are M-multi&apos;s real performance on Day 4&apos;s
                  original held-out test set &mdash; shown alongside the evasion-rate gains so
                  any cost to general performance isn&apos;t hidden.
                </p>
              </>
            )}
          </CardContent>
        </Card>
      </div>
    </div>
  );
}
