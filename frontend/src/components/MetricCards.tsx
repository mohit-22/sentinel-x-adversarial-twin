import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import type { MetricsResponse } from "@/lib/api";

interface MetricCardsProps {
  metrics: MetricsResponse;
}

function formatPercent(value: number, digits = 1): string {
  return `${(value * 100).toFixed(digits)}%`;
}

/** KPI header strip (CLAUDE.md §8: "volume, F1, FPR, ARG"). Every value
 * comes directly from the MetricsResponse the caller fetched from
 * GET /api/v1/metrics -- no field here is computed, guessed, or hardcoded.
 */
export function MetricCards({ metrics }: MetricCardsProps) {
  const arena = metrics.latest_arena_run;

  return (
    <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-4">
      <Card>
        <CardHeader>
          <CardTitle className="text-sm font-medium text-muted-foreground">
            Held-out test set size
          </CardTitle>
        </CardHeader>
        <CardContent>
          <p className="text-3xl font-semibold tabular-nums">
            {metrics.test_set_size.toLocaleString()}
          </p>
          <p className="mt-1 text-xs text-muted-foreground">
            Transactions in M0&apos;s held-out evaluation set (not the full
            50,000-row generated dataset)
          </p>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle className="text-sm font-medium text-muted-foreground">
            F1 Score
          </CardTitle>
        </CardHeader>
        <CardContent>
          <p className="text-3xl font-semibold tabular-nums">
            {formatPercent(metrics.f1)}
          </p>
          <p className="mt-1 text-xs text-muted-foreground">
            M0 on the held-out test set
          </p>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle className="text-sm font-medium text-muted-foreground">
            False Positive Rate
          </CardTitle>
        </CardHeader>
        <CardContent>
          <p className="text-3xl font-semibold tabular-nums">
            {formatPercent(metrics.fpr, 2)}
          </p>
          <p className="mt-1 text-xs text-muted-foreground">
            M0 on the held-out test set
          </p>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle className="text-sm font-medium text-muted-foreground">
            Adversarial Robustness Gain
          </CardTitle>
        </CardHeader>
        <CardContent>
          {arena ? (
            <>
              <p
                className="text-3xl font-semibold tabular-nums"
                style={{
                  color:
                    arena.robustness_gain > 0
                      ? "var(--neon-green)"
                      : arena.robustness_gain < 0
                        ? "var(--neon-red)"
                        : undefined,
                }}
              >
                {arena.robustness_gain.toFixed(2)}%
              </p>
              <p className="mt-1 text-xs text-muted-foreground">
                {arena.attack_family} (run {arena.run_id})
              </p>
            </>
          ) : (
            <>
              <p className="border-l-2 border-dashed border-muted-foreground/50 pl-2 text-lg font-medium text-muted-foreground">
                Run Adversarial Arena to compute
              </p>
              <p className="mt-1 text-xs text-muted-foreground">
                No arena run has completed in this session yet
              </p>
            </>
          )}
        </CardContent>
      </Card>
    </div>
  );
}
