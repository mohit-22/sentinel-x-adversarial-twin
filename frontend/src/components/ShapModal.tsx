"use client";

import { Dialog } from "@base-ui/react/dialog";
import { useEffect, useState } from "react";

import { Button } from "@/components/ui/button";
import { ApiError, explainTransaction } from "@/lib/api";

interface ShapModalProps {
  transactionId: string | null;
  onClose: () => void;
}

/**
 * Click-through SHAP reason-code modal (CLAUDE.md §8, Screen 4). Calls the
 * real GET /api/v1/explain/{transaction_id} endpoint -- today that ALWAYS
 * returns 501 (SHAP wiring is Day 8 work, per endpoints.py). This modal
 * shows that real 501 honestly, same "not yet available" pattern as
 * Screen 2's Judge Sandbox -- never fabricated reason codes.
 */
export function ShapModal({ transactionId, onClose }: ShapModalProps) {
  const [state, setState] = useState<"loading" | "unavailable" | "error">("loading");
  const [message, setMessage] = useState<string | null>(null);

  useEffect(() => {
    if (!transactionId) return;
    explainTransaction(transactionId)
      .then(() => {
        // Not expected to succeed until Day 8 -- if it ever does, still
        // show something honest rather than assuming a shape.
        setState("unavailable");
        setMessage("/explain succeeded, unexpectedly -- Day 8 SHAP wiring may now exist.");
      })
      .catch((err) => {
        if (err instanceof ApiError && err.status === 501) {
          setState("unavailable");
          setMessage(err.message);
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
            {state === "unavailable" && (
              <div
                className="rounded-lg border border-dashed p-3 text-sm text-muted-foreground"
                style={{ borderColor: "var(--status-sandbox)" }}
              >
                <p className="font-medium" style={{ color: "var(--status-sandbox)" }}>
                  SHAP explainability &mdash; Day 8 work, not yet available
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
