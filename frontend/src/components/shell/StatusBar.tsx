"use client";

import { useEffect, useState } from "react";
import { ApiError, fetchMetrics, type MetricsResponse } from "@/lib/api";

/**
 * Global Sentinel-X status bar. Every value here is either a real fetched
 * backend value (system/model status, F1, latest arena run) or a static
 * capability label for a feature that genuinely exists and is callable
 * this session (READY/ACTIVE badges) -- never a fabricated "live" number.
 * "OFFLINE" is a real, honestly-reachable state if the backend is down,
 * not a state that's hidden or silently retried forever.
 */
export function StatusBar() {
  const [metrics, setMetrics] = useState<MetricsResponse | null>(null);
  const [systemState, setSystemState] = useState<"checking" | "online" | "offline">("checking");

  useEffect(() => {
    let cancelled = false;
    fetchMetrics()
      .then((m) => {
        if (cancelled) return;
        setMetrics(m);
        setSystemState("online");
      })
      .catch(() => {
        if (!cancelled) setSystemState("offline");
      });
    return () => {
      cancelled = true;
    };
  }, []);

  const systemColor =
    systemState === "online" ? "var(--neon-green)" : systemState === "offline" ? "var(--neon-red)" : "var(--muted-foreground)";
  const systemLabel = systemState === "online" ? "ONLINE" : systemState === "offline" ? "OFFLINE" : "CHECKING";

  return (
    <div className="flex flex-wrap items-center justify-between gap-x-4 gap-y-1.5 border-b border-[var(--surface-glass-border)] bg-[var(--surface-glass)] px-4 py-2 backdrop-blur-md text-xs">
      <div className="flex items-center gap-3">
        <span className="text-sm font-semibold tracking-wide text-foreground">
          SENTINEL<span style={{ color: "var(--neon-green)" }}>-X</span>
        </span>
        <span className="hidden text-muted-foreground sm:inline">Autonomous Adversarial Payment Defense</span>
      </div>

      <div className="flex flex-wrap items-center gap-2">
        <StatusPill label="SYSTEM" value={systemLabel} color={systemColor} pulse={systemState === "online"} />
        {metrics && (
          <>
            <StatusPill label="MODEL" value={`F1 ${(metrics.f1 * 100).toFixed(1)}%`} color="var(--neon-green)" />
            <StatusPill
              label="ARENA"
              value={metrics.latest_arena_run ? metrics.latest_arena_run.attack_family : "no run yet"}
              color={metrics.latest_arena_run ? "var(--neon-cyan)" : "var(--muted-foreground)"}
            />
          </>
        )}
        <StatusPill label="ZERO-DAY RADAR" value="READY" color="var(--neon-blue)" />
        <StatusPill label="ADVERSARIAL ENGINE" value="READY" color="var(--neon-amber)" />
        <StatusPill label="AI INVESTIGATOR" value="ACTIVE" color="var(--neon-green)" />
      </div>
    </div>
  );
}

function StatusPill({ label, value, color, pulse }: { label: string; value: string; color: string; pulse?: boolean }) {
  return (
    <span className="flex items-center gap-1.5 rounded-full border border-[var(--surface-glass-border)] bg-black/20 px-2 py-1 text-[10px] font-medium">
      <span
        className={`h-1.5 w-1.5 rounded-full ${pulse ? "animate-pulse" : ""}`}
        style={{ backgroundColor: color, boxShadow: `0 0 6px ${color}` }}
      />
      <span className="text-muted-foreground">{label}:</span>
      <span style={{ color }}>{value}</span>
    </span>
  );
}
