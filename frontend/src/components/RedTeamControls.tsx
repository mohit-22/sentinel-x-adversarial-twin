"use client";

import { useEffect, useRef, useState } from "react";

import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import {
  ApiError,
  KNOWN_ATTACK_GENOMES,
  compileSandbox,
  triggerArenaRun,
  type ArenaRunSummary,
} from "@/lib/api";

const OFFICIAL_N_INSTANCES = 2000;
const MIN_N_INSTANCES = 100;

function formatPercent(value: number, digits = 2): string {
  return `${(value * 100).toFixed(digits)}%`;
}

/**
 * Screen 2: Red Team Lab (CLAUDE.md §8). Attack family selector, scale
 * slider, hardening trigger, "live stream" (honestly reinterpreted below),
 * and the Judge Sandbox free-text input.
 *
 * "Live transaction stream" note: /arena/run is a single blocking HTTP
 * call -- there is no WebSocket in the locked tech stack, and the backend
 * never emits per-transaction events to stream even in principle. This
 * screen shows an elapsed-time progress indicator while the call is in
 * flight, then the completed result as a "Run Report" -- not a literal
 * live feed. That's a stated design decision, not a shortcut.
 */
export function RedTeamControls() {
  const [genomeId, setGenomeId] = useState(KNOWN_ATTACK_GENOMES[0].genome_id);
  const [nInstances, setNInstances] = useState(OFFICIAL_N_INSTANCES);

  const [isRunning, setIsRunning] = useState(false);
  const [elapsedSeconds, setElapsedSeconds] = useState(0);
  const [runResult, setRunResult] = useState<ArenaRunSummary | null>(null);
  const [runResultNInstances, setRunResultNInstances] = useState<number | null>(null);
  const [runError, setRunError] = useState<string | null>(null);

  const [sandboxText, setSandboxText] = useState("");
  const [sandboxState, setSandboxState] = useState<
    "idle" | "loading" | "unavailable" | "error" | "success"
  >("idle");
  const [sandboxMessage, setSandboxMessage] = useState<string | null>(null);

  const timerRef = useRef<ReturnType<typeof setInterval> | null>(null);

  useEffect(() => {
    return () => {
      if (timerRef.current) clearInterval(timerRef.current);
    };
  }, []);

  const isOfficialRun = nInstances >= OFFICIAL_N_INSTANCES;

  async function handleTriggerRun() {
    setIsRunning(true);
    setElapsedSeconds(0);
    setRunError(null);
    setRunResult(null);

    timerRef.current = setInterval(() => {
      setElapsedSeconds((s) => s + 1);
    }, 1000);

    const requestedNInstances = isOfficialRun ? undefined : nInstances;

    try {
      const result = await triggerArenaRun(genomeId, requestedNInstances);
      setRunResult(result);
      setRunResultNInstances(requestedNInstances ?? OFFICIAL_N_INSTANCES);
    } catch (err) {
      setRunError(err instanceof ApiError ? err.message : String(err));
    } finally {
      if (timerRef.current) clearInterval(timerRef.current);
      setIsRunning(false);
    }
  }

  async function handleSandboxSubmit() {
    setSandboxState("loading");
    setSandboxMessage(null);
    try {
      await compileSandbox(sandboxText);
      setSandboxState("success");
    } catch (err) {
      if (err instanceof ApiError && err.status === 501) {
        setSandboxState("unavailable");
        setSandboxMessage(err.message);
      } else {
        setSandboxState("error");
        setSandboxMessage(err instanceof ApiError ? err.message : String(err));
      }
    }
  }

  return (
    <div className="min-h-screen bg-background p-8 text-foreground">
      <div className="mx-auto max-w-4xl space-y-6">
        <header className="border-b border-border pb-4">
          <h1 className="text-xl font-semibold tracking-tight">Red Team Lab</h1>
          <p className="text-sm text-muted-foreground">
            Attack Genomes &rarr; Adversarial Arena
          </p>
        </header>

        <Card>
          <CardHeader>
            <CardTitle>Attack Configuration</CardTitle>
          </CardHeader>
          <CardContent className="space-y-5">
            <div className="space-y-1.5">
              <label htmlFor="genome-select" className="text-sm font-medium text-muted-foreground">
                Attack family
              </label>
              <select
                id="genome-select"
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

            <div className="space-y-1.5">
              <div className="flex items-center justify-between">
                <label htmlFor="scale-slider" className="text-sm font-medium text-muted-foreground">
                  Scale (n_instances)
                </label>
                <span className="text-sm tabular-nums text-foreground">
                  {nInstances.toLocaleString()}
                </span>
              </div>
              <input
                id="scale-slider"
                type="range"
                min={MIN_N_INSTANCES}
                max={OFFICIAL_N_INSTANCES}
                step={100}
                value={nInstances}
                disabled={isRunning}
                onChange={(e) => setNInstances(Number(e.target.value))}
                className="w-full accent-primary disabled:opacity-50"
              />
              {isOfficialRun ? (
                <p className="text-xs text-muted-foreground">
                  Official run (n=2000, the documented standard) &mdash; ~100s+
                </p>
              ) : (
                <p className="text-xs" style={{ color: "var(--status-sandbox)" }}>
                  QUICK TEST MODE &mdash; n={nInstances}, NOT the official run. Move the
                  slider to {OFFICIAL_N_INSTANCES.toLocaleString()} for the real result.
                </p>
              )}
            </div>

            <Button
              onClick={handleTriggerRun}
              disabled={isRunning}
              variant={isOfficialRun ? "default" : "outline"}
              className="w-full"
              style={!isOfficialRun ? { borderColor: "var(--status-sandbox)" } : undefined}
            >
              {isRunning
                ? `Running... (${elapsedSeconds}s elapsed)`
                : isOfficialRun
                  ? "Trigger Adversarial Hardening (official run)"
                  : `Trigger Adversarial Hardening (QUICK TEST, n=${nInstances})`}
            </Button>

            {isRunning && (
              <div className="space-y-1">
                <div className="h-1.5 w-full overflow-hidden rounded-full bg-muted">
                  <div className="h-full w-1/3 animate-pulse bg-primary" />
                </div>
                <p className="text-xs text-muted-foreground">
                  /arena/run is a single blocking call (no WebSocket in the locked
                  tech stack) &mdash; this is a progress indicator, not a live
                  per-transaction feed. Result will render as a Run Report on
                  completion.
                </p>
              </div>
            )}

            {runError && (
              <div className="rounded-lg border border-destructive/40 bg-destructive/10 p-3 text-sm text-destructive">
                {runError}
              </div>
            )}

            {runResult && !isRunning && (
              <RunReport result={runResult} nInstancesUsed={runResultNInstances} />
            )}
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>Judge Sandbox</CardTitle>
          </CardHeader>
          <CardContent className="space-y-3">
            <p className="text-xs text-muted-foreground">
              Free text &rarr; LLM &rarr; validated genome &rarr; live simulation. The
              compiler backend is Day 8 work and is not built yet.
            </p>
            <textarea
              value={sandboxText}
              onChange={(e) => setSandboxText(e.target.value)}
              placeholder="Describe a fraud scenario in plain English..."
              rows={3}
              disabled={sandboxState === "loading"}
              className="w-full rounded-md border border-border bg-input/30 px-3 py-2 text-sm text-foreground placeholder:text-muted-foreground disabled:opacity-50"
            />
            <Button
              onClick={handleSandboxSubmit}
              disabled={sandboxState === "loading" || sandboxText.trim().length === 0}
              variant="secondary"
            >
              {sandboxState === "loading" ? "Compiling..." : "Compile Genome"}
            </Button>

            {sandboxState === "unavailable" && (
              <div
                className="rounded-lg border border-dashed p-3 text-sm text-muted-foreground"
                style={{ borderColor: "var(--status-sandbox)" }}
              >
                <p className="font-medium" style={{ color: "var(--status-sandbox)" }}>
                  Sandbox not yet available
                </p>
                <p className="mt-1">{sandboxMessage}</p>
              </div>
            )}
            {sandboxState === "error" && (
              <div className="rounded-lg border border-destructive/40 bg-destructive/10 p-3 text-sm text-destructive">
                {sandboxMessage}
              </div>
            )}
            {sandboxState === "success" && (
              <div className="rounded-lg border border-primary/40 bg-primary/10 p-3 text-sm text-primary">
                Genome compiled -- this path isn&apos;t expected to succeed until Day 8.
              </div>
            )}
          </CardContent>
        </Card>
      </div>
    </div>
  );
}

function RunReport({
  result,
  nInstancesUsed,
}: {
  result: ArenaRunSummary;
  nInstancesUsed: number | null;
}) {
  const official = nInstancesUsed !== null && nInstancesUsed >= OFFICIAL_N_INSTANCES;

  return (
    <div className="space-y-3 rounded-lg border border-border bg-muted/30 p-4">
      <div className="flex items-center justify-between">
        <h3 className="text-sm font-medium">Run Report</h3>
        <Badge variant={official ? "default" : "outline"}>
          n={nInstancesUsed?.toLocaleString()} {official ? "(official)" : "(quick test)"}
        </Badge>
      </div>
      <dl className="grid grid-cols-2 gap-x-4 gap-y-2 text-sm sm:grid-cols-3">
        <ReportField label="Run ID" value={result.run_id} mono />
        <ReportField label="Attack family" value={result.attack_family} mono />
        <ReportField label="Hard examples" value={result.hard_examples_count.toLocaleString()} />
        <ReportField label="Initial evasion" value={formatPercent(result.initial_evasion_rate)} />
        <ReportField label="Final evasion" value={formatPercent(result.final_evasion_rate)} />
        <ReportField
          label="Robustness gain (ARG)"
          value={`${result.robustness_gain.toFixed(2)}%`}
          color={
            result.robustness_gain > 0
              ? "var(--neon-green)"
              : result.robustness_gain < 0
                ? "var(--neon-red)"
                : undefined
          }
        />
        <ReportField
          label="Retrained F1"
          value={formatPercent(result.retrained_f1_score)}
        />
      </dl>
    </div>
  );
}

function ReportField({
  label,
  value,
  mono,
  color,
}: {
  label: string;
  value: string;
  mono?: boolean;
  color?: string;
}) {
  return (
    <div>
      <dt className="text-xs text-muted-foreground">{label}</dt>
      <dd
        className={`font-medium tabular-nums ${mono ? "font-mono text-xs" : ""}`}
        style={color ? { color } : undefined}
      >
        {value}
      </dd>
    </div>
  );
}
