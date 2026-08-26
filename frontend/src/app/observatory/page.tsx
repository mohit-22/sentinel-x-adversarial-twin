"use client";

import { useEffect, useState } from "react";
import { Line, LineChart, CartesianGrid, ResponsiveContainer, XAxis, YAxis } from "recharts";

import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { ApiError, KNOWN_ATTACK_GENOMES } from "@/lib/api";

const API_BASE_URL = process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://127.0.0.1:8000/api/v1";

// --- Types mirror backend/app/api/endpoints.py's 3 Observatory routes
// exactly -- api.ts isn't touched for this screen (one new file only), so
// fetches are self-contained here, same pattern as ArenaView.tsx's
// Cross-Family Generalization Matrix section.

interface LineageEntry {
  generation: number;
  genome_id: string;
  evasion_rate: number;
  fitness: number;
}

interface LineageResponse {
  status: string;
  trajectory: LineageEntry[];
}

interface ImpactResponse {
  status: string;
  hard_negatives: number;
  avg_amount: number;
  fraud_prevented_inr: number;
  detection_rate: number;
  transactions_protected: number;
}

async function fetchLineage(): Promise<LineageResponse> {
  const url = `${API_BASE_URL}/observatory/lineage`;
  let response: Response;
  try {
    response = await fetch(url, { cache: "no-store" });
  } catch {
    throw new ApiError(`Could not reach backend at ${url} -- is the server running?`);
  }
  if (!response.ok) throw new ApiError(`GET /observatory/lineage returned ${response.status}`, response.status);
  return (await response.json()) as LineageResponse;
}

async function fetchImpact(): Promise<ImpactResponse> {
  const url = `${API_BASE_URL}/observatory/impact`;
  let response: Response;
  try {
    response = await fetch(url, { cache: "no-store" });
  } catch {
    throw new ApiError(`Could not reach backend at ${url} -- is the server running?`);
  }
  if (!response.ok) throw new ApiError(`GET /observatory/impact returned ${response.status}`, response.status);
  return (await response.json()) as ImpactResponse;
}

async function exportGenome(genomeId: string): Promise<unknown> {
  const url = `${API_BASE_URL}/observatory/export`;
  let response: Response;
  try {
    response = await fetch(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      // run_id is accepted by the backend but not yet used to look up
      // per-run state (no multi-run store exists yet) -- sent for contract
      // completeness, not because it changes the response.
      body: JSON.stringify({ run_id: "current-session", genome_id: genomeId }),
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
    throw new ApiError(`POST /observatory/export returned ${response.status}: ${detail}`, response.status);
  }
  return response.json();
}

function formatPercent(value: number, digits = 2): string {
  return `${(value * 100).toFixed(digits)}%`;
}

/** Evasion rate (0-1) -> red(high)/green(low), same color-mix technique
 * ArenaView.tsx/BlueTeamSOC.tsx already use for evasion heat coloring.
 */
function evasionHeatColor(rate: number): string {
  const pct = Math.round(Math.min(Math.max(rate, 0), 1) * 100);
  return `color-mix(in oklch, var(--neon-red) ${pct}%, var(--neon-green))`;
}

function triggerJsonDownload(data: unknown, filename: string) {
  const blob = new Blob([JSON.stringify(data, null, 2)], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = filename;
  document.body.appendChild(anchor);
  anchor.click();
  document.body.removeChild(anchor);
  URL.revokeObjectURL(url);
}

/**
 * Threat Observatory -- new screen, not one of CLAUDE.md §8's 5 canonical
 * screens. Reuses only already-cached, already-verified backend data
 * (_LATEST_ADAPTIVE_RUN, _LATEST_ARENA_RUN, _GENOME_REGISTRY) -- no new
 * computation happens here beyond what the 3 new endpoints return.
 */
export default function ObservatoryPage() {
  const [lineage, setLineage] = useState<LineageResponse | null>(null);
  const [isLineageLoading, setIsLineageLoading] = useState(true);
  const [lineageError, setLineageError] = useState<string | null>(null);

  const [impact, setImpact] = useState<ImpactResponse | null>(null);
  const [isImpactLoading, setIsImpactLoading] = useState(true);
  const [impactError, setImpactError] = useState<string | null>(null);

  const [exportGenomeId, setExportGenomeId] = useState(KNOWN_ATTACK_GENOMES[0].genome_id);
  const [isExporting, setIsExporting] = useState(false);
  const [exportError, setExportError] = useState<string | null>(null);

  function loadLineage() {
    setIsLineageLoading(true);
    setLineageError(null);
    fetchLineage()
      .then(setLineage)
      .catch((err) => setLineageError(err instanceof ApiError ? err.message : String(err)))
      .finally(() => setIsLineageLoading(false));
  }

  function loadImpact() {
    setIsImpactLoading(true);
    setImpactError(null);
    fetchImpact()
      .then(setImpact)
      .catch((err) => setImpactError(err instanceof ApiError ? err.message : String(err)))
      .finally(() => setIsImpactLoading(false));
  }

  useEffect(() => {
    loadLineage();
    loadImpact();
  }, []);

  async function handleExport() {
    setIsExporting(true);
    setExportError(null);
    try {
      const data = await exportGenome(exportGenomeId);
      triggerJsonDownload(data, `sentinel-x-threat-intel-${exportGenomeId}.json`);
    } catch (err) {
      setExportError(err instanceof ApiError ? err.message : String(err));
    } finally {
      setIsExporting(false);
    }
  }

  const chartData = (lineage?.trajectory ?? []).map((t) => ({
    generation: t.generation,
    evasionRate: t.evasion_rate * 100,
  }));

  return (
    <div className="min-h-screen bg-background p-8 text-foreground">
      <div className="mx-auto max-w-4xl space-y-6">
        <header className="border-b border-border pb-4">
          <h1 className="text-xl font-semibold tracking-tight">Threat Observatory</h1>
          <p className="text-sm text-muted-foreground">
            Fraud DNA lineage, economic impact, and threat-intel export -- all sourced from
            this session&apos;s cached arena/adaptive-search results, never mock data.
          </p>
        </header>

        {/* SECTION 1: Fraud DNA Evolution Tree */}
        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0">
            <CardTitle>Fraud DNA Evolution Tree</CardTitle>
            <Button variant="secondary" onClick={loadLineage} disabled={isLineageLoading}>
              Refresh
            </Button>
          </CardHeader>
          <CardContent className="space-y-4">
            {lineageError && (
              <div className="rounded-lg border border-destructive/40 bg-destructive/10 p-3 text-sm text-destructive">
                {lineageError}
              </div>
            )}

            {!isLineageLoading && !lineageError && (!lineage || lineage.trajectory.length === 0) && (
              <p className="border-l-2 border-dashed border-muted-foreground/50 pl-3 text-base font-medium text-muted-foreground">
                Run Arena Adaptive first
              </p>
            )}

            {lineage && lineage.trajectory.length > 0 && (
              <>
                <div className="h-64 w-full">
                  <ResponsiveContainer width="100%" height="100%">
                    <LineChart data={chartData} margin={{ top: 8, right: 16, left: 0, bottom: 0 }}>
                      <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
                      <XAxis
                        dataKey="generation"
                        type="number"
                        allowDuplicatedCategory={false}
                        stroke="var(--muted-foreground)"
                        fontSize={12}
                      />
                      <YAxis stroke="var(--muted-foreground)" fontSize={12} unit="%" />
                      <Line
                        type="monotone"
                        dataKey="evasionRate"
                        stroke="var(--neon-green)"
                        strokeWidth={2}
                        dot={{ r: 3 }}
                      />
                    </LineChart>
                  </ResponsiveContainer>
                </div>

                <div className="grid grid-cols-1 gap-3 sm:grid-cols-2 lg:grid-cols-3">
                  {lineage.trajectory.map((entry, idx) => (
                    <Card key={`${entry.generation}-${entry.genome_id}-${idx}`} size="sm">
                      <CardContent className="space-y-1 py-2">
                        <p className="text-xs text-muted-foreground">Generation {entry.generation}</p>
                        <p className="truncate font-mono text-xs" title={entry.genome_id}>
                          {entry.genome_id}
                        </p>
                        <div className="flex items-center justify-between">
                          <span className="text-xs text-muted-foreground">Evasion</span>
                          <span
                            className="font-semibold tabular-nums"
                            style={{ color: evasionHeatColor(entry.evasion_rate) }}
                          >
                            {formatPercent(entry.evasion_rate)}
                          </span>
                        </div>
                        <div className="flex items-center justify-between">
                          <span className="text-xs text-muted-foreground">Fitness</span>
                          <span className="tabular-nums">{entry.fitness.toFixed(3)}</span>
                        </div>
                      </CardContent>
                    </Card>
                  ))}
                </div>
              </>
            )}
          </CardContent>
        </Card>

        {/* SECTION 2: Economic Impact */}
        <Card>
          <CardHeader>
            <CardTitle>Economic Impact</CardTitle>
          </CardHeader>
          <CardContent>
            {impactError && (
              <div className="rounded-lg border border-destructive/40 bg-destructive/10 p-3 text-sm text-destructive">
                {impactError}
              </div>
            )}

            {!isImpactLoading && !impactError && impact?.status === "run_arena_first" && (
              <>
                <p className="border-l-2 border-dashed border-muted-foreground/50 pl-3 text-base font-medium text-muted-foreground">
                  Run Adversarial Arena to compute
                </p>
                <p className="mt-1 text-xs text-muted-foreground">
                  No arena run has completed in this session yet
                </p>
              </>
            )}

            {impact && impact.status === "ok" && (
              <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-4">
                <Card size="sm">
                  <CardHeader>
                    <CardTitle className="text-sm font-medium text-muted-foreground">
                      Fraud Prevented
                    </CardTitle>
                  </CardHeader>
                  <CardContent>
                    <p className="text-3xl font-semibold tabular-nums" style={{ color: "var(--neon-green)" }}>
                      &#8377;{impact.fraud_prevented_inr.toLocaleString("en-IN", { maximumFractionDigits: 0 })}
                    </p>
                    <p className="mt-1 text-xs text-muted-foreground">
                      hard negatives &times; avg transaction amount
                    </p>
                  </CardContent>
                </Card>

                <Card size="sm">
                  <CardHeader>
                    <CardTitle className="text-sm font-medium text-muted-foreground">
                      Hard Negatives Caught
                    </CardTitle>
                  </CardHeader>
                  <CardContent>
                    <p className="text-3xl font-semibold tabular-nums">
                      {impact.hard_negatives.toLocaleString()}
                    </p>
                  </CardContent>
                </Card>

                <Card size="sm">
                  <CardHeader>
                    <CardTitle className="text-sm font-medium text-muted-foreground">
                      Transactions Protected
                    </CardTitle>
                  </CardHeader>
                  <CardContent>
                    <p className="text-3xl font-semibold tabular-nums">
                      {impact.transactions_protected.toLocaleString()}
                    </p>
                  </CardContent>
                </Card>

                <Card size="sm">
                  <CardHeader>
                    <CardTitle className="text-sm font-medium text-muted-foreground">
                      Detection Rate
                    </CardTitle>
                  </CardHeader>
                  <CardContent>
                    <p className="text-3xl font-semibold tabular-nums">
                      {(impact.detection_rate * 100).toFixed(1)}%
                    </p>
                    <p className="mt-1 text-xs text-muted-foreground">F1 on held-out test set</p>
                  </CardContent>
                </Card>
              </div>
            )}
          </CardContent>
        </Card>

        {/* SECTION 3: Threat Intelligence Export */}
        <Card>
          <CardHeader>
            <CardTitle>Threat Intelligence Export</CardTitle>
            <p className="text-xs text-muted-foreground">
              STIX 2.1-shaped bundle for one known genome, built as a plain dict server-side --
              no new library added.
            </p>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="flex flex-col gap-4 sm:flex-row sm:items-end">
              <div className="flex-1 space-y-1.5">
                <label htmlFor="export-genome-select" className="text-sm font-medium text-muted-foreground">
                  Attack family
                </label>
                <select
                  id="export-genome-select"
                  value={exportGenomeId}
                  disabled={isExporting}
                  onChange={(e) => setExportGenomeId(e.target.value)}
                  className="w-full rounded-md border border-border bg-input/30 px-3 py-2 text-sm text-foreground disabled:opacity-50"
                >
                  {KNOWN_ATTACK_GENOMES.map((g) => (
                    <option key={g.genome_id} value={g.genome_id}>
                      {g.label} ({g.genome_id})
                    </option>
                  ))}
                </select>
              </div>
              <Button onClick={handleExport} disabled={isExporting}>
                {isExporting ? "Exporting..." : "Export STIX 2.1"}
              </Button>
            </div>

            {exportError && (
              <div className="rounded-lg border border-destructive/40 bg-destructive/10 p-3 text-sm text-destructive">
                {exportError}
              </div>
            )}
          </CardContent>
        </Card>
      </div>
    </div>
  );
}
