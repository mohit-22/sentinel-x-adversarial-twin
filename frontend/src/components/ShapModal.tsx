"use client";

import { Dialog } from "@base-ui/react/dialog";
import { useEffect, useState } from "react";

import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { ApiError, explainTransaction } from "@/lib/api";

interface ShapModalProps {
  transactionId: string | null;
  onClose: () => void;
}

/** Mirrors backend/app/api/endpoints.py's ExplainResponse exactly (Day 8a). */
interface ReasonCode {
  feature: string;
  contribution: string;
  description: string;
}
interface ExplainResponse {
  transaction_id: string;
  reason_codes: ReasonCode[];
}

/**
 * Click-through SHAP reason-code modal (CLAUDE.md §8, Screen 4; real logic
 * as of Day 8a). Calls the real GET /api/v1/explain/{transaction_id}.
 *
 * Honest scope: /explain only covers transactions already in M0's cached
 * train/test dataset (SHAP needs the exact engineered feature row, which
 * can't be recomputed from a bare id). A transaction generated fresh for
 * another view (e.g. Screen 4's feed rows sourced from /payment-twin's
 * counterfactual instances) gets a 404 -- shown here as a stated scope
 * boundary, not an error, since that's exactly the row type (high-risk
 * BLOCK, disproportionately injected fraud legs) a judge is most likely
 * to click. See PRD_SENTINEL_X §13.1 for the full disclosure.
 */
export function ShapModal({ transactionId, onClose }: ShapModalProps) {
  const [state, setState] = useState<"loading" | "success" | "scope-boundary" | "error">("loading");
  const [reasonCodes, setReasonCodes] = useState<ReasonCode[]>([]);
  const [message, setMessage] = useState<string | null>(null);

  useEffect(() => {
    if (!transactionId) return;
    explainTransaction(transactionId)
      .then((body) => {
        setReasonCodes((body as ExplainResponse).reason_codes);
        setState("success");
      })
      .catch((err) => {
        if (err instanceof ApiError && err.status === 404) {
          setState("scope-boundary");
          setMessage(
            "SHAP explanations are available for M0's original evaluation dataset. " +
              "This specific transaction was generated fresh for this view and isn't in that cached set.",
          );
        } else {
          setState("error");
          setMessage(err instanceof ApiError ? err.message : String(err));
        }
      });
  }, [transactionId]);

  return (
    <Dialog.Root open={transactionId !== null} onOpenChange={(open) => !open && onClose()}>
      <Dialog.Portal>
        <Dialog.Backdrop className="fixed inset-0 bg-black/60" />
        <Dialog.Popup className="fixed top-1/2 left-1/2 w-full max-w-md -translate-x-1/2 -translate-y-1/2 rounded-lg border border-border bg-card p-5 text-card-foreground shadow-xl">
          <Dialog.Title className="text-sm font-semibold">
            SHAP Reason Codes &mdash; {transactionId}
          </Dialog.Title>

          <div className="mt-4">
            {state === "loading" && (
              <p className="text-sm text-muted-foreground">Calling /explain/{transactionId}...</p>
            )}

            {state === "success" && (
              <ul className="space-y-2.5">
                {reasonCodes.map((code) => {
                  const isPositive = code.contribution.startsWith("+");
                  return (
                    <li key={code.feature} className="rounded-md border border-border p-2.5 text-xs">
                      <div className="flex items-center justify-between">
                        <span className="font-mono font-medium">{code.feature}</span>
                        <Badge
                          variant="outline"
                          className="tabular-nums"
                          style={{ color: isPositive ? "var(--neon-red)" : "var(--neon-green)" }}
                        >
                          {code.contribution}
                        </Badge>
                      </div>
                      <p className="mt-1 text-muted-foreground">{code.description}</p>
                    </li>
                  );
                })}
              </ul>
            )}

            {state === "scope-boundary" && (
              <div
                className="rounded-lg border border-dashed p-3 text-sm text-muted-foreground"
                style={{ borderColor: "var(--status-sandbox)" }}
              >
                <p className="font-medium" style={{ color: "var(--status-sandbox)" }}>
                  Outside today&apos;s SHAP scope
                </p>
                <p className="mt-1">{message}</p>
              </div>
            )}

            {state === "error" && (
              <div className="rounded-lg border border-destructive/40 bg-destructive/10 p-3 text-sm text-destructive">
                {message}
              </div>
            )}
          </div>

          <div className="mt-5 flex justify-end">
            <Dialog.Close render={<Button variant="secondary">Close</Button>} />
          </div>
        </Dialog.Popup>
      </Dialog.Portal>
    </Dialog.Root>
  );
}
