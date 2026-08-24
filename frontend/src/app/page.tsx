"use client";

import { useEffect, useState } from "react";

import { MetricCards } from "@/components/MetricCards";
import { ApiError, fetchMetrics, type MetricsResponse } from "@/lib/api";

/**
 * Screen 1: Command Center (CLAUDE.md §8). KPI header + LIVE/SANDBOX/error
 * status. Every displayed number comes from a real GET /api/v1/metrics
 * call made on mount -- nothing here is mocked or hardcoded.
 *
 * Status semantics: LIVE = the last /metrics call succeeded. ERROR = it
 * failed (backend unreachable or returned a non-2xx response). There is
 * no real "SANDBOX" data source yet -- that depends on the Day 8
 * /sandbox/compile feature, which doesn't exist. Rather than invent a
 * fake trigger condition for a SANDBOX state, this screen only ever shows
 * LIVE or ERROR today; SANDBOX will get a real meaning once that feature
 * is built.
 */
export default function Home() {
  const [metrics, setMetrics] = useState<MetricsResponse | null>(null);
  const [status, setStatus] = useState<"loading" | "live" | "error">(
    "loading",
  );
  const [errorMessage, setErrorMessage] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;

    fetchMetrics()
      .then((data) => {
        if (cancelled) return;
        setMetrics(data);
        setStatus("live");
      })
      .catch((err) => {
        if (cancelled) return;
        setStatus("error");
        setErrorMessage(err instanceof ApiError ? err.message : String(err));
      });

    return () => {
      cancelled = true;
    };
  }, []);

  return (
    <div className="min-h-screen bg-background p-8 text-foreground">
      <div className="mx-auto max-w-6xl space-y-6">
        <header className="flex items-center justify-between border-b border-border pb-4">
          <div>
            <h1 className="text-xl font-semibold tracking-tight">
              Sentinel-X Command Center
            </h1>
            <p className="text-sm text-muted-foreground">
              Autonomous Adversarial Payment Twin
            </p>
          </div>
          <StatusBadge status={status} />
        </header>

        {status === "loading" && (
          <p className="text-sm text-muted-foreground">
            Loading metrics from the backend...
          </p>
        )}

        {status === "error" && (
          <div className="rounded-lg border border-destructive/40 bg-destructive/10 p-4 text-sm text-destructive">
            <p className="font-medium">Could not load metrics.</p>
            <p className="mt-1 text-destructive/80">{errorMessage}</p>
          </div>
        )}

        {status === "live" && metrics && <MetricCards metrics={metrics} />}
      </div>
    </div>
  );
}

function StatusBadge({ status }: { status: "loading" | "live" | "error" }) {
  const config = {
    loading: { label: "CONNECTING", color: "var(--muted-foreground)" },
    live: { label: "LIVE", color: "var(--status-live)" },
    error: { label: "ERROR", color: "var(--status-error)" },
  }[status];

  return (
    <span
      className="inline-flex items-center gap-2 rounded-full border border-border px-3 py-1 text-xs font-medium tracking-wide"
      style={{ color: config.color }}
    >
      <span
        className="h-2 w-2 rounded-full"
        style={{ backgroundColor: config.color }}
      />
      {config.label}
    </span>
  );
}
