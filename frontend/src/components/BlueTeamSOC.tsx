"use client";

import { useState } from "react";

import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { ShapModal } from "@/components/ShapModal";
import {
  ApiError,
  KNOWN_ATTACK_GENOMES,
  detectTransactions,
  fetchMetrics,
  fetchPaymentTwin,
  type DetectionResult,
  type InjectedTransaction,
  type MetricsResponse,
  type TransactionBase,
} from "@/lib/api";

const KNOWN_FAMILIES = Array.from(new Set(KNOWN_ATTACK_GENOMES.map((g) => g.family)));

// No "list customers" endpoint exists (same constraint as Screen 3), so the
// feed is built from a fixed, deterministic slice of customer ids
// (CUST-000000..CUST-000011 -- all real, all guaranteed to exist for the
// default 10,000-customer twin). Rotating the attack family per customer
// gives a demo feed that spans all 5 families, not just one.
const FEED_CUSTOMER_COUNT = 12;

function feedCustomerIds(): string[] {
  return Array.from({ length: FEED_CUSTOMER_COUNT }, (_, i) => `CUST-${String(i).padStart(6, "0")}`);
}

interface FeedRow extends DetectionResult {
  customer_id: string;
  amount: number;
  channel: string;
  is_fraud: number | null; // null for normal_transactions -- TransactionBase has no such field
  attack_family: string | null;
}

function formatPercent(value: number, digits = 1): string {
  return `${(value * 100).toFixed(digits)}%`;
}

function decisionColor(decision: string): string {
  if (decision === "ALLOW") return "var(--neon-green)";
  if (decision === "BLOCK") return "var(--neon-red)";
  return "var(--status-sandbox)"; // STEP_UP, REVIEW -- amber
}

/**
 * Screen 4: Blue Team SOC (CLAUDE.md §8). precision/recall/F1/FPR (from
 * /metrics, real), latency (from /detect responses -- the only endpoint
 * that measures it, per the Screen 4 planning investigation), a
 * color-coded feed, and a click-through SHAP modal.
 *
 * "Live feed" note, same honesty pattern as Screen 2's "live stream": this
 * is a real, one-shot Detection Feed built entirely from real backend
 * calls (/payment-twin/{id} for real transactions -> /detect for real
 * scores), not a literal push/streaming feed -- there's no WebSocket
 * wiring for this in the locked tech stack. "Refresh Feed" re-runs it.
 */
export function BlueTeamSOC() {
  const [metrics, setMetrics] = useState<MetricsResponse | null>(null);
  const [feed, setFeed] = useState<FeedRow[]>([]);
  const [avgLatencyMs, setAvgLatencyMs] = useState<number | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [selectedTransactionId, setSelectedTransactionId] = useState<string | null>(null);

  async function handleLoadFeed() {
    setIsLoading(true);
    setError(null);
    try {
      const metricsResult = await fetchMetrics();
      setMetrics(metricsResult);

      const customerIds = feedCustomerIds();
      const latencies: number[] = [];
      const rows: FeedRow[] = [];

      for (const [i, customerId] of customerIds.entries()) {
        const attackFamily = KNOWN_FAMILIES[i % KNOWN_FAMILIES.length];
        const twin = await fetchPaymentTwin(customerId, attackFamily);
        const batch: TransactionBase[] = [...twin.normal_transactions, ...twin.counterfactual_transactions];
        if (batch.length === 0) continue;

        const results = await detectTransactions(batch);
        latencies.push(...results.map((r) => r.latency_ms));

        const byId = new Map(batch.map((t) => [t.transaction_id, t]));
        for (const result of results) {
          const source = byId.get(result.transaction_id);
          if (!source) continue;
          const injected = "is_fraud" in source ? (source as InjectedTransaction) : null;
          rows.push({
            ...result,
            customer_id: source.customer_id,
            amount: source.amount,
            channel: source.channel,
            is_fraud: injected ? injected.is_fraud : null,
            attack_family: injected ? injected.attack_family : null,
          });
        }
      }

      rows.sort((a, b) => b.risk_score - a.risk_score);
      setFeed(rows);
      setAvgLatencyMs(latencies.length > 0 ? latencies.reduce((a, b) => a + b, 0) / latencies.length : null);
    } catch (err) {
      setError(err instanceof ApiError ? err.message : String(err));
    } finally {
      setIsLoading(false);
    }
  }

  return (
    <div className="min-h-screen bg-background p-8 text-foreground">
      <div className="mx-auto max-w-6xl space-y-6">
        <header className="flex items-center justify-between border-b border-border pb-4">
          <div>
            <p className="text-xs font-semibold uppercase tracking-widest" style={{ color: "var(--neon-green)" }}>
              Fraud Investigation Workstation
            </p>
            <h1 className="text-3xl font-bold tracking-tight">Blue Team SOC</h1>
            <p className="text-sm text-muted-foreground mt-1">
              Detection performance and a real, color-coded transaction feed
            </p>
          </div>
          <Button onClick={handleLoadFeed} disabled={isLoading}>
            {isLoading ? "Loading..." : feed.length > 0 ? "Refresh Feed" : "Load Detection Feed"}
          </Button>
        </header>

        {error && (
          <div className="rounded-lg border border-destructive/40 bg-destructive/10 p-3 text-sm text-destructive">
            {error}
          </div>
        )}

        {metrics && (
          <div className="grid grid-cols-2 gap-4 sm:grid-cols-5">
            <KpiCard label="Precision" value={formatPercent(metrics.precision)} />
            <KpiCard label="Recall" value={formatPercent(metrics.recall)} />
            <KpiCard label="F1" value={formatPercent(metrics.f1)} />
            <KpiCard label="FPR" value={formatPercent(metrics.fpr, 2)} />
            <KpiCard
              label="Latency (measured)"
              value={avgLatencyMs !== null ? `${avgLatencyMs.toFixed(2)} ms` : "—"}
              hint={avgLatencyMs !== null ? `avg over ${feed.length} real /detect results` : "load the feed to measure"}
            />
          </div>
        )}

        <Card>
          <CardHeader>
            <CardTitle>Detection Feed</CardTitle>
            <p className="text-xs text-muted-foreground">
              {feed.length > 0
                ? `${feed.length} real scored transactions across ${FEED_CUSTOMER_COUNT} customers, sorted by risk score. Click a row for its SHAP reason codes.`
                : "No feed loaded yet -- click \"Load Detection Feed\" to fetch real transactions and score them."}
            </p>
          </CardHeader>
          <CardContent>
            <div className="max-h-[560px] space-y-1.5 overflow-y-auto">
              {feed.map((row) => (
                <button
                  key={`${row.customer_id}-${row.transaction_id}`}
                  onClick={() => setSelectedTransactionId(row.transaction_id)}
                  className="flex w-full items-center justify-between rounded-md border p-2.5 text-left text-xs transition-colors hover:bg-muted/50"
                  style={{ borderColor: decisionColor(row.decision), borderLeftWidth: 3 }}
                >
                  <div className="flex min-w-0 flex-1 items-center gap-3">
                    <span className="shrink-0 font-mono text-muted-foreground">{row.transaction_id}</span>
                    <span className="shrink-0 text-muted-foreground">{row.customer_id}</span>
                    <span className="truncate font-medium tabular-nums">₹{row.amount.toFixed(2)}</span>
                    <Badge variant="outline" className="shrink-0 text-[10px]">
                      {row.channel}
                    </Badge>
                    {row.is_fraud === 1 && (
                      <Badge variant="destructive" className="shrink-0 text-[10px]">
                        ACTUAL: FRAUD ({row.attack_family})
                      </Badge>
                    )}
                  </div>
                  <div className="flex shrink-0 items-center gap-3">
                    <span className="tabular-nums text-muted-foreground">
                      risk {row.risk_score.toFixed(3)}
                    </span>
                    <span
                      className="w-16 text-right font-semibold"
                      style={{ color: decisionColor(row.decision) }}
                    >
                      {row.decision}
                    </span>
                  </div>
                </button>
              ))}
            </div>
          </CardContent>
        </Card>
      </div>

      <ShapModal
        key={selectedTransactionId}
        transactionId={selectedTransactionId}
        onClose={() => setSelectedTransactionId(null)}
      />
    </div>
  );
}

function KpiCard({ label, value, hint }: { label: string; value: string; hint?: string }) {
  return (
    <Card size="sm">
      <CardContent>
        <p className="text-xs text-muted-foreground">{label}</p>
        <p className="text-xl font-semibold tabular-nums">{value}</p>
        {hint && <p className="mt-0.5 text-[10px] text-muted-foreground">{hint}</p>}
      </CardContent>
    </Card>
  );
}
