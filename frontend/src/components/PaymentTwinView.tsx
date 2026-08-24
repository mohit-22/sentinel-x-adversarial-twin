"use client";

import { useState } from "react";

import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import {
  ApiError,
  KNOWN_ATTACK_GENOMES,
  fetchPaymentTwin,
  type InjectedTransaction,
  type PaymentTwinResponse,
  type TransactionBase,
} from "@/lib/api";

const KNOWN_FAMILIES = Array.from(new Set(KNOWN_ATTACK_GENOMES.map((g) => g.family)));
const FAMILY_LABELS: Record<string, string> = Object.fromEntries(
  KNOWN_ATTACK_GENOMES.map((g) => [g.family, g.label]),
);

const DEFAULT_CUSTOMER_ID = "CUST-000000";

function formatTime(isoTimestamp: string): string {
  return new Date(isoTimestamp).toLocaleString(undefined, {
    month: "short",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  });
}

/**
 * Screen 3: Payment Twin (CLAUDE.md §8). Normal vs. counterfactual-attacked
 * customer comparison. Every field comes from a real
 * GET /api/v1/payment-twin/{customer_id} call -- the approved 7th endpoint
 * (see CLAUDE.md §7's documented exception, added specifically because
 * neither /simulate nor /arena/run exposes any per-customer detail).
 *
 * customer_id is a free-text input, not a dropdown -- there is no "list
 * customers" endpoint (deliberately out of scope for the minimal 7th
 * endpoint), but ids are deterministic (CUST-000000..CUST-009999 for the
 * default 10,000-customer twin), so a judge/user can type any real one.
 */
export function PaymentTwinView() {
  const [customerId, setCustomerId] = useState(DEFAULT_CUSTOMER_ID);
  const [attackFamily, setAttackFamily] = useState(KNOWN_FAMILIES[0]);

  const [isLoading, setIsLoading] = useState(false);
  const [data, setData] = useState<PaymentTwinResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  async function handleLoad() {
    setIsLoading(true);
    setError(null);
    try {
      const result = await fetchPaymentTwin(customerId.trim(), attackFamily);
      setData(result);
    } catch (err) {
      setData(null);
      setError(err instanceof ApiError ? err.message : String(err));
    } finally {
      setIsLoading(false);
    }
  }

  return (
    <div className="min-h-screen bg-background p-8 text-foreground">
      <div className="mx-auto max-w-6xl space-y-6">
        <header className="border-b border-border pb-4">
          <h1 className="text-xl font-semibold tracking-tight">Payment Twin</h1>
          <p className="text-sm text-muted-foreground">
            Normal behavior vs. counterfactual-attacked comparison
          </p>
        </header>

        <Card>
          <CardHeader>
            <CardTitle>Select Customer</CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="flex flex-col gap-4 sm:flex-row sm:items-end">
              <div className="flex-1 space-y-1.5">
                <label htmlFor="customer-id-input" className="text-sm font-medium text-muted-foreground">
                  Customer ID
                </label>
                <input
                  id="customer-id-input"
                  type="text"
                  value={customerId}
                  onChange={(e) => setCustomerId(e.target.value)}
                  placeholder="e.g. CUST-000000"
                  className="w-full rounded-md border border-border bg-input/30 px-3 py-2 font-mono text-sm text-foreground placeholder:text-muted-foreground"
                />
              </div>
              <div className="flex-1 space-y-1.5">
                <label htmlFor="family-select" className="text-sm font-medium text-muted-foreground">
                  Counterfactual attack family
                </label>
                <select
                  id="family-select"
                  value={attackFamily}
                  onChange={(e) => setAttackFamily(e.target.value)}
                  className="w-full rounded-md border border-border bg-input/30 px-3 py-2 text-sm text-foreground"
                >
                  {KNOWN_FAMILIES.map((family) => (
                    <option key={family} value={family}>
                      {FAMILY_LABELS[family]}
                    </option>
                  ))}
                </select>
              </div>
              <Button onClick={handleLoad} disabled={isLoading || customerId.trim().length === 0}>
                {isLoading ? "Loading..." : "Load Comparison"}
              </Button>
            </div>
            <p className="text-xs text-muted-foreground">
              There is no &quot;list customers&quot; endpoint, so this is a free-text
              lookup, not a browsable dropdown &mdash; ids are deterministic
              (CUST-000000 through CUST-009999 for the default twin).
            </p>
          </CardContent>
        </Card>

        {error && (
          <div className="rounded-lg border border-destructive/40 bg-destructive/10 p-3 text-sm text-destructive">
            {error}
          </div>
        )}

        {data && (
          <>
            <Card>
              <CardHeader>
                <CardTitle>{data.customer.customer_id}</CardTitle>
              </CardHeader>
              <CardContent>
                <dl className="grid grid-cols-2 gap-x-4 gap-y-2 text-sm sm:grid-cols-4">
                  <Field label="Base location" value={data.customer.base_location} />
                  <Field label="Mean spend" value={`₹${data.customer.mean_spend.toFixed(2)}`} />
                  <Field label="Primary devices" value={data.customer.primary_devices.join(", ")} />
                  <Field label="Usual beneficiaries" value={String(data.customer.usual_beneficiaries.length)} />
                </dl>
              </CardContent>
            </Card>

            <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
              <TransactionColumn
                title="Normal Behavior"
                subtitle={`${data.normal_transactions.length} real transactions from this customer's clean history`}
                accentColor="var(--neon-green)"
                transactions={data.normal_transactions}
              />
              <TransactionColumn
                title={`Counterfactual — ${FAMILY_LABELS[attackFamily]}`}
                subtitle={`${data.counterfactual_transactions.length} rows from one freshly-generated attack instance for this same customer`}
                accentColor="var(--neon-red)"
                transactions={data.counterfactual_transactions}
              />
            </div>
          </>
        )}
      </div>
    </div>
  );
}

function Field({ label, value }: { label: string; value: string }) {
  return (
    <div>
      <dt className="text-xs text-muted-foreground">{label}</dt>
      <dd className="font-medium">{value}</dd>
    </div>
  );
}

function TransactionColumn({
  title,
  subtitle,
  accentColor,
  transactions,
}: {
  title: string;
  subtitle: string;
  accentColor: string;
  transactions: (TransactionBase | InjectedTransaction)[];
}) {
  const sorted = [...transactions].sort((a, b) => a.timestamp.localeCompare(b.timestamp));

  return (
    <Card>
      <CardHeader>
        <CardTitle style={{ color: accentColor }}>{title}</CardTitle>
        <p className="text-xs text-muted-foreground">{subtitle}</p>
      </CardHeader>
      <CardContent>
        <div className="max-h-[480px] space-y-2 overflow-y-auto">
          {sorted.map((t) => {
            const isFraud = "is_fraud" in t && t.is_fraud === 1;
            return (
              <div
                key={t.transaction_id}
                className="rounded-md border p-2.5 text-xs"
                style={{
                  borderColor: isFraud ? "var(--neon-red)" : "var(--border)",
                  backgroundColor: isFraud ? "color-mix(in oklch, var(--neon-red) 8%, transparent)" : undefined,
                }}
              >
                <div className="flex items-center justify-between">
                  <span className="font-mono text-muted-foreground">{t.transaction_id}</span>
                  <span className="text-muted-foreground">{formatTime(t.timestamp)}</span>
                </div>
                <div className="mt-1 flex items-center justify-between">
                  <span className="font-medium tabular-nums">₹{t.amount.toFixed(2)}</span>
                  <div className="flex items-center gap-1.5">
                    <Badge variant="outline" className="text-[10px]">
                      {t.channel}
                    </Badge>
                    {isFraud && (
                      <Badge variant="destructive" className="text-[10px]">
                        FRAUD
                      </Badge>
                    )}
                  </div>
                </div>
                <div className="mt-1 text-muted-foreground">
                  {t.merchant_category} &middot; {t.device_id} &middot; &rarr; {t.beneficiary_id}
                </div>
              </div>
            );
          })}
        </div>
      </CardContent>
    </Card>
  );
}
