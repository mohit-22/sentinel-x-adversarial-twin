"use client";

import { Dialog } from "@base-ui/react/dialog";
import { useEffect, useState } from "react";

import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { ApiError, explainTransaction, investigateTransaction, type AgentVerdict } from "@/lib/api";

interface ShapModalProps {
  transactionId: string | null;
  onClose: () => void;
}

type ModalTab = "shap" | "soc";

function actionColor(action: string): string {
  if (action === "BLOCK") return "var(--neon-red)";
  if (action === "ESCALATE") return "var(--status-sandbox)";
  if (action === "ALLOW") return "var(--neon-green)";
  return "#3b82f6"; // MONITOR -- blue, same ad hoc token ObservatoryNode already uses for its own non-CSS-var accents
}

function confidenceColor(score: number): string {
  if (score > 0.7) return "var(--neon-red)";
  if (score >= 0.4) return "var(--status-sandbox)";
  return "var(--neon-green)";
}

/**
 * SOC Agent Investigation tab: real POST /soc/investigate/{transaction_id}
 * on first activation of this tab (not on modal open -- the Groq call takes
 * ~2-3s and shouldn't fire for a judge who only wants the SHAP tab).
 */
function SocAgentTab({ transactionId }: { transactionId: string }) {
  const [state, setState] = useState<"loading" | "success" | "scope-boundary" | "error">("loading");
  const [verdict, setVerdict] = useState<AgentVerdict | null>(null);
  const [message, setMessage] = useState<string | null>(null);

  useEffect(() => {
    setState("loading");
    investigateTransaction(transactionId)
      .then((v) => {
        setVerdict(v);
        setState("success");
      })
      .catch((err) => {
        if (err instanceof ApiError && err.status === 404) {
          setState("scope-boundary");
          setMessage(
            "The SOC Agent shares /explain's cached-dataset scope: this transaction wasn't in M0's " +
              "cached train/test set, so there's no SHAP evidence to investigate from.",
          );
        } else {
          setState("error");
          setMessage(err instanceof ApiError ? err.message : String(err));
        }
      });
  }, [transactionId]);

  if (state === "loading") {
    return <p className="text-sm text-muted-foreground">Calling the SOC Agent (Groq)... this can take 2-3s.</p>;
  }

  if (state === "scope-boundary") {
    return (
      <div
        className="rounded-lg border border-dashed p-3 text-sm text-muted-foreground"
        style={{ borderColor: "var(--status-sandbox)" }}
      >
        <p className="font-medium" style={{ color: "var(--status-sandbox)" }}>
          Outside today&apos;s SOC Agent scope
        </p>
        <p className="mt-1">{message}</p>
      </div>
    );
  }

  if (state === "error") {
    return (
      <div className="rounded-lg border border-destructive/40 bg-destructive/10 p-3 text-sm text-destructive">
        {message}
      </div>
    );
  }

  if (!verdict) return null;

  return (
    <div className="space-y-3 text-sm">
      <div
        className="inline-block rounded-md border px-2.5 py-1 text-xs font-bold tracking-wide"
        style={{ borderColor: actionColor(verdict.recommended_action), color: actionColor(verdict.recommended_action) }}
      >
        {verdict.recommended_action}
      </div>

      <p>{verdict.hypothesis}</p>

      <div>
        <div className="flex items-center justify-between text-xs text-muted-foreground">
          <span>Confidence</span>
          <span className="tabular-nums">{(verdict.confidence_score * 100).toFixed(0)}%</span>
        </div>
        <div className="mt-1 h-1.5 w-full overflow-hidden rounded-full bg-muted">
          <div
            className="h-full"
            style={{
              width: `${Math.min(Math.max(verdict.confidence_score, 0), 1) * 100}%`,
              backgroundColor: confidenceColor(verdict.confidence_score),
            }}
          />
        </div>
      </div>

      <div>
        <p className="text-xs font-medium text-muted-foreground">Evidence</p>
        <ul className="mt-1 list-disc space-y-1 pl-4">
          {verdict.evidence.map((e, i) => (
            <li key={i}>{e}</li>
          ))}
        </ul>
      </div>

      <div>
        <p className="text-xs font-medium text-muted-foreground">Reasoning</p>
        <p className="mt-0.5">{verdict.reasoning_chain}</p>
      </div>

      <div className="flex items-center justify-between text-xs">
        <span className="text-muted-foreground">Suspected Family</span>
        <span className="font-medium">{verdict.attack_family_suspected}</span>
      </div>
      <div className="flex items-center justify-between text-xs">
        <span className="text-muted-foreground">Similar Past Attacks</span>
        <span className="font-medium tabular-nums">{verdict.similar_past_attacks}</span>
      </div>

      <div className="border-t border-border pt-2">
        <p className="text-xs font-medium text-muted-foreground">Audit Log Entry</p>
        <p className="mt-1 rounded bg-muted/40 p-2 font-mono text-[11px] text-muted-foreground">
          {verdict.audit_log_entry}
        </p>
      </div>
    </div>
  );
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
  const [tab, setTab] = useState<ModalTab>("shap");
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
            Transaction Investigation &mdash; {transactionId}
          </Dialog.Title>

          <div className="mt-3 flex gap-1 border-b border-border">
            <button
              onClick={() => setTab("shap")}
              className={`px-3 py-1.5 text-xs font-medium transition-colors ${
                tab === "shap"
                  ? "border-b-2 border-foreground text-foreground"
                  : "text-muted-foreground hover:text-foreground"
              }`}
            >
              SHAP Analysis
            </button>
            <button
              onClick={() => setTab("soc")}
              className={`px-3 py-1.5 text-xs font-medium transition-colors ${
                tab === "soc"
                  ? "border-b-2 border-foreground text-foreground"
                  : "text-muted-foreground hover:text-foreground"
              }`}
            >
              SOC Agent Investigation
            </button>
          </div>

          <div className="mt-4 max-h-[60vh] overflow-y-auto">
            {tab === "soc" && transactionId && <SocAgentTab transactionId={transactionId} />}

            {tab === "shap" && state === "loading" && (
              <p className="text-sm text-muted-foreground">Calling /explain/{transactionId}...</p>
            )}

            {tab === "shap" && state === "success" && (
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

            {tab === "shap" && state === "scope-boundary" && (
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

            {tab === "shap" && state === "error" && (
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
