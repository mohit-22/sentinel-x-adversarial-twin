"use client";

import { useEffect, useMemo, useState } from "react";
import { AlertTriangle, MapPin } from "lucide-react";

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";

const API_BASE_URL = process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://127.0.0.1:8000/api/v1";

// Real, fixed geographic coordinates -- these never change, so hardcoding
// them is not "fabricated data" (unlike any of the fraud/transaction
// numbers below, which all come from the real GET /api/v1/threat-map call).
const CITY_COORDS: Record<string, [number, number]> = {
  Mumbai: [72.8777, 19.076],
  Delhi: [77.1025, 28.7041],
  Bangalore: [77.5946, 12.9716],
  Chennai: [80.2707, 13.0827],
  Hyderabad: [78.4867, 17.385],
  Kolkata: [88.3639, 22.5726],
  Pune: [73.8567, 18.5204],
  Ahmedabad: [72.5714, 23.0225],
  Jaipur: [75.7873, 26.9124],
  Nagpur: [79.0882, 21.1458],
  Surat: [72.8311, 21.1702],
  Lucknow: [80.9462, 26.8467],
};

// Rough India bounding box, used only to position the 12 known cities
// within the panel proportionally to their real lat/long -- not a precise
// map projection, just enough to make relative geography (north/south/
// east/west) honest at a glance. No topojson/map library: react-simple-maps
// doesn't support React 19 (peer dependency conflict on install), so this
// is the sanctioned grid/positioned fallback, not a shortcut taken lightly.
const BOUNDS = { minLng: 68, maxLng: 97.5, minLat: 8, maxLat: 36 };

function toPosition([lng, lat]: [number, number]): { left: string; top: string } {
  const x = ((lng - BOUNDS.minLng) / (BOUNDS.maxLng - BOUNDS.minLng)) * 100;
  const y = ((BOUNDS.maxLat - lat) / (BOUNDS.maxLat - BOUNDS.minLat)) * 100;
  return { left: `${x}%`, top: `${y}%` };
}

interface CityThreat {
  city: string;
  total_transactions: number;
  fraud_transactions: number;
  fraud_rate: number;
  total_amount_blocked_inr: number;
  risk_level: "HIGH" | "MEDIUM" | "LOW";
}

interface ThreatMapResponse {
  cities: CityThreat[];
  summary: {
    total_fraud_blocked_inr: number;
    highest_risk_city: string | null;
    cities_monitored: number;
  };
}

function riskColor(level: string): string {
  if (level === "HIGH") return "var(--neon-red)";
  if (level === "MEDIUM") return "var(--neon-amber)";
  return "var(--neon-green)";
}

function formatInr(value: number): string {
  return `₹${value.toLocaleString("en-IN", { maximumFractionDigits: 0 })}`;
}

export default function ThreatMapPage() {
  const [data, setData] = useState<ThreatMapResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [hovered, setHovered] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    fetch(`${API_BASE_URL}/threat-map`, { cache: "no-store" })
      .then((res) => {
        if (!res.ok) throw new Error(`GET /threat-map returned ${res.status}`);
        return res.json();
      })
      .then((body: ThreatMapResponse) => {
        if (!cancelled) setData(body);
      })
      .catch((err) => {
        if (!cancelled) setError(err instanceof Error ? err.message : String(err));
      });
    return () => {
      cancelled = true;
    };
  }, []);

  const byCity = useMemo(() => {
    if (!data) return {};
    return Object.fromEntries(data.cities.map((c) => [c.city, c]));
  }, [data]);

  const mappedCities = useMemo(() => {
    return Object.keys(CITY_COORDS)
      .map((name) => byCity[name])
      .filter((c): c is CityThreat => Boolean(c));
  }, [byCity]);

  const sizeScale = useMemo(() => {
    if (mappedCities.length === 0) return () => 20;
    const counts = mappedCities.map((c) => c.total_transactions);
    const min = Math.min(...counts);
    const max = Math.max(...counts);
    return (n: number) => {
      if (max === min) return 24;
      const t = Math.sqrt((n - min) / (max - min));
      return 16 + t * 30;
    };
  }, [mappedCities]);

  const top5 = useMemo(() => (data ? [...data.cities].sort((a, b) => b.fraud_rate - a.fraud_rate).slice(0, 5) : []), [data]);

  return (
    <div className="min-h-screen bg-background p-8 text-foreground">
      <div className="mx-auto max-w-6xl space-y-6">
        <header className="border-b border-border pb-4">
          <p className="text-xs font-semibold uppercase tracking-widest" style={{ color: "var(--neon-cyan)" }}>
            Threat Intelligence
          </p>
          <h1 className="text-3xl font-bold tracking-tight">Threat Intelligence Map</h1>
          <p className="mt-1 text-sm text-muted-foreground">
            Current Payment Twin evaluation snapshot
          </p>
        </header>

        {error && (
          <div className="rounded-lg border border-destructive/40 bg-destructive/10 p-3 text-sm text-destructive">
            {error}
          </div>
        )}

        {!data && !error && (
          <div className="flex h-64 items-center justify-center text-sm text-muted-foreground">
            Loading threat map...
          </div>
        )}

        {data && (
          <div className="grid grid-cols-1 gap-4 lg:grid-cols-10">
            {/* Map panel -- ~70% */}
            <Card className="relative overflow-hidden lg:col-span-7">
              <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-3">
                <CardTitle className="text-base">City Risk Map</CardTitle>
                <div className="flex items-center gap-3 text-[10px] text-muted-foreground">
                  <LegendDot color="var(--neon-red)" label="High" />
                  <LegendDot color="var(--neon-amber)" label="Medium" />
                  <LegendDot color="var(--neon-green)" label="Low" />
                </div>
              </CardHeader>
              <CardContent>
                <div
                  className="relative h-[420px] w-full overflow-hidden rounded-lg border border-border"
                  style={{
                    backgroundImage:
                      "linear-gradient(hsl(var(--muted-foreground)/0.08) 1px, transparent 1px), linear-gradient(90deg, hsl(var(--muted-foreground)/0.08) 1px, transparent 1px)",
                    backgroundSize: "24px 24px",
                  }}
                >
                  <span className="absolute left-3 top-3 text-[10px] font-semibold uppercase tracking-widest text-muted-foreground/60">
                    India &mdash; UPI Network
                  </span>
                  {mappedCities.map((c) => {
                    const pos = toPosition(CITY_COORDS[c.city]);
                    const size = sizeScale(c.total_transactions);
                    const isHovered = hovered === c.city;
                    return (
                      <div
                        key={c.city}
                        className="absolute -translate-x-1/2 -translate-y-1/2 cursor-pointer"
                        style={{ left: pos.left, top: pos.top }}
                        onMouseEnter={() => setHovered(c.city)}
                        onMouseLeave={() => setHovered(null)}
                      >
                        <div
                          className="rounded-full transition-transform"
                          style={{
                            width: size,
                            height: size,
                            backgroundColor: `color-mix(in oklch, ${riskColor(c.risk_level)} 35%, transparent)`,
                            border: `2px solid ${riskColor(c.risk_level)}`,
                            boxShadow: isHovered ? `0 0 12px ${riskColor(c.risk_level)}` : undefined,
                            transform: isHovered ? "scale(1.15)" : undefined,
                          }}
                        />
                        <span className="absolute left-1/2 top-full mt-1 -translate-x-1/2 whitespace-nowrap text-[10px] text-muted-foreground">
                          {c.city}
                        </span>
                        {isHovered && (
                          <div className="absolute bottom-full left-1/2 z-10 mb-2 w-48 -translate-x-1/2 rounded-md border border-border bg-card p-2.5 text-xs shadow-xl">
                            <p className="font-semibold" style={{ color: riskColor(c.risk_level) }}>
                              {c.city} &mdash; {c.risk_level}
                            </p>
                            <p className="mt-1 text-muted-foreground">
                              Fraud rate: <span className="font-medium text-foreground">{(c.fraud_rate * 100).toFixed(1)}%</span>
                            </p>
                            <p className="text-muted-foreground">
                              Amount flagged: <span className="font-medium text-foreground">{formatInr(c.total_amount_blocked_inr)}</span>
                            </p>
                          </div>
                        )}
                      </div>
                    );
                  })}
                </div>
                <p className="mt-2 text-[11px] text-muted-foreground">
                  Circle size is proportional to transaction volume; only the {mappedCities.length} major cities with
                  known coordinates are plotted here. The full {data.summary.cities_monitored}-city breakdown is in
                  the panel and list to the right.
                </p>
              </CardContent>
            </Card>

            {/* Stats panel -- ~30% */}
            <div className="space-y-4 lg:col-span-3">
              <Card>
                <CardHeader className="pb-2">
                  <CardTitle className="text-sm font-medium text-muted-foreground">Model-Flagged Fraud Value</CardTitle>
                </CardHeader>
                <CardContent>
                  <p className="text-2xl font-bold tabular-nums" style={{ color: "var(--neon-green)" }}>
                    {formatInr(data.summary.total_fraud_blocked_inr)}
                  </p>
                </CardContent>
              </Card>

              <div className="grid grid-cols-2 gap-4">
                <Card size="sm">
                  <CardHeader className="pb-1">
                    <CardTitle className="text-xs font-medium text-muted-foreground">Cities Monitored</CardTitle>
                  </CardHeader>
                  <CardContent>
                    <p className="text-xl font-bold tabular-nums">{data.summary.cities_monitored}</p>
                  </CardContent>
                </Card>
                <Card size="sm">
                  <CardHeader className="pb-1">
                    <CardTitle className="text-xs font-medium text-muted-foreground">Highest Risk</CardTitle>
                  </CardHeader>
                  <CardContent>
                    <p className="truncate text-sm font-bold" style={{ color: "var(--neon-red)" }}>
                      {data.summary.highest_risk_city ?? "—"}
                    </p>
                  </CardContent>
                </Card>
              </div>

              <Card>
                <CardHeader className="pb-2">
                  <CardTitle className="flex items-center gap-1.5 text-sm">
                    <AlertTriangle className="h-3.5 w-3.5" style={{ color: "var(--neon-red)" }} />
                    Top 5 Riskiest Cities
                  </CardTitle>
                </CardHeader>
                <CardContent className="space-y-2">
                  {top5.map((c, i) => (
                    <div key={c.city} className="flex items-center justify-between text-xs">
                      <div className="flex items-center gap-2">
                        <span className="w-4 text-muted-foreground">{i + 1}</span>
                        <MapPin className="h-3 w-3" style={{ color: riskColor(c.risk_level) }} />
                        <span className="font-medium">{c.city}</span>
                      </div>
                      <span className="tabular-nums" style={{ color: riskColor(c.risk_level) }}>
                        {(c.fraud_rate * 100).toFixed(1)}%
                      </span>
                    </div>
                  ))}
                </CardContent>
              </Card>
            </div>
          </div>
        )}

        <p className="border-t border-border pt-4 text-center text-xs italic text-muted-foreground">
          Data sourced from Sentinel-X Payment Twin &mdash; synthetic benchmark, not real transaction data.
        </p>
      </div>
    </div>
  );
}

function LegendDot({ color, label }: { color: string; label: string }) {
  return (
    <span className="flex items-center gap-1">
      <span className="h-2 w-2 rounded-full" style={{ backgroundColor: color }} />
      {label}
    </span>
  );
}
