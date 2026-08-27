"use client";

import { useEffect, useState, useMemo, useCallback } from "react";
import {
  ReactFlow,
  Controls,
  Background,
  MiniMap,
  Handle,
  Position,
  Node,
  Edge,
  useNodesState,
  useEdgesState,
  MarkerType,
} from "@xyflow/react";
import "@xyflow/react/dist/style.css";

import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { ApiError, KNOWN_ATTACK_GENOMES } from "@/lib/api";
import { Badge } from "@/components/ui/badge";

const API_BASE_URL = process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://127.0.0.1:8000/api/v1";

interface GenomeData {
  genome_id: string;
  family: string;
  objective?: string;
  mutations?: string[];
}

interface LineageNodeEntry {
  generation: number;
  genome: GenomeData;
  parent_attack_id: string;
  evasion_rate: number;
  novelty_score?: number;
  impact_score?: number;
  realism_score?: number;
  total_fitness: number;
  validity_status: string;
  is_elite: boolean;
  is_best: boolean;
}

interface TrajectoryEntry {
  generation: number;
  genome_id: string;
  evasion_rate: number;
  fitness: number;
}

interface LineageResponse {
  status: string;
  run_id?: string;
  base_genome_id?: string;
  lineage: LineageNodeEntry[];
  trajectory: TrajectoryEntry[];
}

interface ImpactResponse {
  status: string;
  run_id?: string;
  attack_family?: string;
  total_attack_transactions?: number;
  total_attack_value_inr?: number;
  value_caught_by_m0_inr?: number;
  value_caught_after_hardening_inr?: number;
  incremental_value_prevented_inr?: number;
  additional_transactions_caught?: number;
  m0_evasion_rate?: number;
  post_hardening_evasion_rate?: number;
  methodology?: string;
}

async function fetchLineage(): Promise<LineageResponse> {
  const url = `${API_BASE_URL}/observatory/lineage`;
  let response: Response;
  try {
    response = await fetch(url, { cache: "no-store" });
  } catch {
    throw new ApiError(`Could not reach backend at ${url} -- is the server running?`);
  }
  if (!response.ok) throw new ApiError(`GET /observatory/lineage returned ${response.status}`, response.status);
  return (await response.json()) as LineageResponse;
}

async function fetchImpact(): Promise<ImpactResponse> {
  const url = `${API_BASE_URL}/observatory/impact`;
  let response: Response;
  try {
    response = await fetch(url, { cache: "no-store" });
  } catch {
    throw new ApiError(`Could not reach backend at ${url} -- is the server running?`);
  }
  if (!response.ok) throw new ApiError(`GET /observatory/impact returned ${response.status}`, response.status);
  return (await response.json()) as ImpactResponse;
}

async function exportGenome(runId: string, genomeId: string): Promise<unknown> {
  const url = `${API_BASE_URL}/observatory/export`;
  let response: Response;
  try {
    response = await fetch(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ run_id: runId, genome_id: genomeId }),
    });
  } catch {
    throw new ApiError(`Could not reach backend at ${url} -- is the server running?`);
  }
  if (!response.ok) {
    let detail = response.statusText;
    try {
      const errorBody = await response.json();
      if (errorBody?.detail) detail = errorBody.detail;
    } catch {
      // ignore
    }
    throw new ApiError(`POST /observatory/export returned ${response.status}: ${detail}`, response.status);
  }
  return response.json();
}

function evasionHeatColor(rate: number): string {
  const pct = Math.round(Math.min(Math.max(rate, 0), 1) * 100);
  return `color-mix(in oklch, var(--neon-red) ${pct}%, var(--neon-green))`;
}

function triggerJsonDownload(data: unknown, filename: string) {
  const blob = new Blob([JSON.stringify(data, null, 2)], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = filename;
  document.body.appendChild(anchor);
  anchor.click();
  document.body.removeChild(anchor);
  URL.revokeObjectURL(url);
}

// --- React Flow Custom Node ---
const ObservatoryNode = ({ data }: { data: any }) => {
  const { isBase, isElite, isBest, isInvalid, genomeId, generation, evasionRate } = data;
  
  let borderColor = "var(--border)";
  if (isBase) borderColor = "var(--neon-blue)";
  else if (isInvalid) borderColor = "var(--destructive)";
  else if (isBest) borderColor = "var(--neon-green)";
  else if (isElite) borderColor = "var(--neon-cyan)";

  return (
    <div
      className={`relative min-w-[150px] rounded-lg border-2 bg-card p-3 shadow-sm transition-transform hover:scale-[1.02]`}
      style={{ borderColor, backgroundColor: "hsl(var(--card))", color: "hsl(var(--card-foreground))" }}
    >
      <Handle type="target" position={Position.Top} className="!bg-muted-foreground !border-none !w-2 !h-2" />
      
      <div className="absolute -top-3 left-1/2 flex -translate-x-1/2 gap-1">
        {isBase && <span className="whitespace-nowrap rounded bg-blue-500/20 px-1.5 py-0.5 text-[10px] font-bold text-blue-500">BASE</span>}
        {isBest && <span className="whitespace-nowrap rounded bg-green-500/20 px-1.5 py-0.5 text-[10px] font-bold text-green-500">★ BEST</span>}
        {isElite && !isBest && <span className="whitespace-nowrap rounded bg-cyan-500/20 px-1.5 py-0.5 text-[10px] font-bold text-cyan-500">ELITE</span>}
        {isInvalid && <span className="whitespace-nowrap rounded bg-red-500/20 px-1.5 py-0.5 text-[10px] font-bold text-red-500">INVALID</span>}
      </div>

      <div className="mt-1 space-y-1">
        <p className="text-xs text-muted-foreground">Gen {generation}</p>
        <p className="truncate font-mono text-xs font-semibold" title={genomeId}>
          {genomeId}
        </p>
        <p className="text-xs text-muted-foreground">
          Evasion: <span className="font-medium" style={{ color: evasionHeatColor(evasionRate) }}>{(evasionRate * 100).toFixed(1)}%</span>
        </p>
      </div>
      
      <Handle type="source" position={Position.Bottom} className="!bg-muted-foreground !border-none !w-2 !h-2" />
    </div>
  );
};

export default function ObservatoryPage() {
  const [lineage, setLineage] = useState<LineageResponse | null>(null);
  const [isLineageLoading, setIsLineageLoading] = useState(true);
  const [lineageError, setLineageError] = useState<string | null>(null);

  const [impact, setImpact] = useState<ImpactResponse | null>(null);
  const [isImpactLoading, setIsImpactLoading] = useState(true);
  const [impactError, setImpactError] = useState<string | null>(null);

  const [exportGenomeId, setExportGenomeId] = useState("");
  const [isExporting, setIsExporting] = useState(false);
  const [exportError, setExportError] = useState<string | null>(null);

  const [nodes, setNodes, onNodesChange] = useNodesState([]);
  const [edges, setEdges, onEdgesChange] = useEdgesState([]);
  const [selectedNodeData, setSelectedNodeData] = useState<LineageNodeEntry | null>(null);

  const nodeTypes = useMemo(() => ({ observatoryNode: ObservatoryNode }), []);

  function loadLineage() {
    setIsLineageLoading(true);
    setLineageError(null);
    fetchLineage()
      .then((data) => {
        setLineage(data);
        if (data?.trajectory?.length > 0) {
          setExportGenomeId(data.trajectory[0].genome_id);
        }
        if (data?.lineage) {
          buildGraph(data.lineage);
        }
      })
      .catch((err) => setLineageError(err instanceof ApiError ? err.message : String(err)))
      .finally(() => setIsLineageLoading(false));
  }

  function loadImpact() {
    setIsImpactLoading(true);
    setImpactError(null);
    fetchImpact()
      .then(setImpact)
      .catch((err) => setImpactError(err instanceof ApiError ? err.message : String(err)))
      .finally(() => setIsImpactLoading(false));
  }

  useEffect(() => {
    loadLineage();
    loadImpact();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  function buildGraph(lineageEntries: LineageNodeEntry[]) {
    const newNodes: Node[] = [];
    const newEdges: Edge[] = [];
    
    const gens: Record<number, LineageNodeEntry[]> = {};
    lineageEntries.forEach((entry) => {
      if (!gens[entry.generation]) gens[entry.generation] = [];
      gens[entry.generation].push(entry);
    });
    
    const X_SPACING = 220;
    const Y_SPACING = 150;
    
    Object.entries(gens).forEach(([genStr, entries]) => {
      const gen = parseInt(genStr, 10);
      const totalWidth = (entries.length - 1) * X_SPACING;
      const startX = -totalWidth / 2;
      
      entries.forEach((entry, idx) => {
        const isBase = gen === 0 && entry.parent_attack_id === entry.genome.genome_id;
        
        newNodes.push({
          id: entry.genome.genome_id,
          position: { x: startX + idx * X_SPACING, y: gen * Y_SPACING },
          type: "observatoryNode",
          data: {
            genomeId: entry.genome.genome_id,
            generation: entry.generation,
            evasionRate: entry.evasion_rate,
            isBase,
            isElite: entry.is_elite,
            isBest: entry.is_best,
            isInvalid: entry.validity_status !== "VALID",
            fullEntry: entry,
          },
        });
        
        if (entry.parent_attack_id && entry.parent_attack_id !== entry.genome.genome_id) {
          newEdges.push({
            id: `e-${entry.parent_attack_id}-${entry.genome.genome_id}`,
            source: entry.parent_attack_id,
            target: entry.genome.genome_id,
            type: "smoothstep",
            animated: entry.is_elite || entry.is_best,
            style: { stroke: (entry.is_elite || entry.is_best) ? "var(--neon-green)" : "hsl(var(--muted-foreground))" },
            markerEnd: {
              type: MarkerType.ArrowClosed,
              color: (entry.is_elite || entry.is_best) ? "var(--neon-green)" : "hsl(var(--muted-foreground))",
            },
          });
        }
      });
    });
    
    setNodes(newNodes);
    setEdges(newEdges);
    setSelectedNodeData(null);
  }

  const onNodeClick = useCallback((_: React.MouseEvent, node: Node) => {
    setSelectedNodeData(node.data.fullEntry as LineageNodeEntry);
  }, []);

  async function handleExport() {
    if (!lineage?.run_id || !exportGenomeId) return;
    setIsExporting(true);
    setExportError(null);
    try {
      const data = await exportGenome(lineage.run_id, exportGenomeId);
      triggerJsonDownload(data, `sentinel-x-threat-intel-${exportGenomeId}.json`);
    } catch (err) {
      setExportError(err instanceof ApiError ? err.message : String(err));
    } finally {
      setIsExporting(false);
    }
  }
  
  const hasLineageData = lineage && lineage.status === "ok" && lineage.lineage?.length > 0;

  // Compute summary metrics
  const variantCount = lineage?.lineage?.length ?? 0;
  let maxGen = 0;
  let bestEvasion = 0;
  let bestFitness = 0;
  if (hasLineageData) {
    lineage.lineage.forEach(n => {
      if (n.generation > maxGen) maxGen = n.generation;
      if (n.is_best) {
        bestEvasion = n.evasion_rate;
        bestFitness = n.total_fitness;
      }
    });
  }

  return (
    <div className="min-h-screen bg-background p-8 text-foreground">
      <div className="mx-auto max-w-5xl space-y-6">
        <header className="border-b border-border pb-4">
          <h1 className="text-xl font-semibold tracking-tight">Threat Observatory</h1>
          <p className="text-sm text-muted-foreground">
            Fraud DNA lineage, economic impact, and threat-intel export -- all sourced from
            this session&apos;s cached arena/adaptive-search results, never mock data.
          </p>
        </header>

        {/* SECTION 1: Fraud DNA Evolution Tree */}
        <Card className="overflow-hidden">
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-4 border-b border-border bg-muted/20">
            <CardTitle>Fraud DNA Evolution Tree</CardTitle>
            <Button variant="secondary" onClick={loadLineage} disabled={isLineageLoading}>
              Refresh
            </Button>
          </CardHeader>
          <CardContent className="p-0">
            {lineageError && (
              <div className="m-4 rounded-lg border border-destructive/40 bg-destructive/10 p-3 text-sm text-destructive">
                {lineageError}
              </div>
            )}

            {!isLineageLoading && !lineageError && !hasLineageData && (
              <div className="p-8">
                <p className="border-l-2 border-dashed border-muted-foreground/50 pl-3 text-base font-medium text-muted-foreground">
                  Run Adaptive Arena first
                </p>
              </div>
            )}

            {isLineageLoading && !hasLineageData && (
              <div className="flex h-64 items-center justify-center">
                <p className="text-sm text-muted-foreground animate-pulse">Loading...</p>
              </div>
            )}

            {hasLineageData && (
              <div className="flex h-[600px] w-full flex-col md:flex-row">
                {/* Main Graph Area */}
                <div className="flex-1 relative border-r border-border bg-dot-pattern">
                  {/* Summary Bar */}
                  <div className="absolute top-2 left-2 z-10 flex gap-2">
                    <Badge variant="outline" className="bg-background/80 backdrop-blur">
                      Run: <span className="font-mono ml-1 font-normal">{lineage.run_id}</span>
                    </Badge>
                    <Badge variant="outline" className="bg-background/80 backdrop-blur">
                      Gens: <span className="ml-1 font-normal">{maxGen + 1}</span>
                    </Badge>
                    <Badge variant="outline" className="bg-background/80 backdrop-blur">
                      Variants: <span className="ml-1 font-normal">{variantCount}</span>
                    </Badge>
                  </div>
                  
                  <ReactFlow
                    nodes={nodes}
                    edges={edges}
                    onNodesChange={onNodesChange}
                    onEdgesChange={onEdgesChange}
                    onNodeClick={onNodeClick}
                    nodeTypes={nodeTypes}
                    fitView
                    minZoom={0.2}
                    maxZoom={2}
                    className="w-full h-full"
                    colorMode="dark"
                  >
                    <Background gap={24} size={2} color="hsl(var(--muted-foreground))" />
                    <Controls className="!bg-card !border-border !fill-foreground" />
                    <MiniMap 
                      className="!bg-card !border-border"
                      nodeColor={(n: any) => {
                        if (n.data?.isInvalid) return "hsl(var(--destructive))";
                        if (n.data?.isBest) return "#22c55e"; // neon-green
                        if (n.data?.isElite) return "#06b6d4"; // neon-cyan
                        if (n.data?.isBase) return "#3b82f6"; // blue
                        return "hsl(var(--muted-foreground))";
                      }} 
                    />
                  </ReactFlow>
                </div>

                {/* Detail Panel */}
                <div className="w-full md:w-80 bg-muted/10 p-4 overflow-y-auto">
                  <h3 className="mb-4 font-semibold text-sm text-muted-foreground uppercase tracking-wider">Node Details</h3>
                  {selectedNodeData ? (
                    <div className="space-y-4 text-sm">
                      <div>
                        <p className="text-xs text-muted-foreground">Genome ID</p>
                        <p className="font-mono break-all">{selectedNodeData.genome.genome_id}</p>
                      </div>
                      
                      {selectedNodeData.parent_attack_id && (
                        <div>
                          <p className="text-xs text-muted-foreground">Parent Attack ID</p>
                          <p className="font-mono break-all">
                            {selectedNodeData.parent_attack_id === selectedNodeData.genome.genome_id 
                              ? "Self (Root Base)" 
                              : selectedNodeData.parent_attack_id}
                          </p>
                        </div>
                      )}
                      
                      <div className="grid grid-cols-2 gap-2">
                        <div>
                          <p className="text-xs text-muted-foreground">Generation</p>
                          <p>{selectedNodeData.generation}</p>
                        </div>
                        <div>
                          <p className="text-xs text-muted-foreground">Family</p>
                          <p className="truncate" title={selectedNodeData.genome.family}>{selectedNodeData.genome.family}</p>
                        </div>
                      </div>
                      
                      <div className="grid grid-cols-2 gap-2">
                        <div>
                          <p className="text-xs text-muted-foreground">Evasion Rate</p>
                          <p className="font-medium" style={{ color: evasionHeatColor(selectedNodeData.evasion_rate) }}>
                            {(selectedNodeData.evasion_rate * 100).toFixed(2)}%
                          </p>
                        </div>
                        <div>
                          <p className="text-xs text-muted-foreground">Fitness</p>
                          <p>{selectedNodeData.total_fitness.toFixed(3)}</p>
                        </div>
                      </div>
                      
                      <div className="grid grid-cols-2 gap-2">
                        {selectedNodeData.novelty_score !== undefined && (
                          <div>
                            <p className="text-xs text-muted-foreground">Novelty Score</p>
                            <p>{selectedNodeData.novelty_score.toFixed(3)}</p>
                          </div>
                        )}
                        {selectedNodeData.impact_score !== undefined && (
                          <div>
                            <p className="text-xs text-muted-foreground">Impact Score</p>
                            <p>{selectedNodeData.impact_score.toFixed(0)}</p>
                          </div>
                        )}
                      </div>

                      <div className="space-y-2 pt-2 border-t border-border">
                        <div className="flex items-center justify-between">
                          <span className="text-xs text-muted-foreground">Status</span>
                          <span className={selectedNodeData.validity_status === "VALID" ? "text-green-500" : "text-red-500"}>
                            {selectedNodeData.validity_status}
                          </span>
                        </div>
                        <div className="flex items-center justify-between">
                          <span className="text-xs text-muted-foreground">Elite</span>
                          <span>{selectedNodeData.is_elite ? "Yes" : "No"}</span>
                        </div>
                        <div className="flex items-center justify-between">
                          <span className="text-xs text-muted-foreground">Best Overall</span>
                          <span>{selectedNodeData.is_best ? "Yes" : "No"}</span>
                        </div>
                      </div>

                      {selectedNodeData.genome.mutations && selectedNodeData.genome.mutations.length > 0 && (
                        <div className="pt-2 border-t border-border">
                          <p className="text-xs text-muted-foreground mb-1">Mutations</p>
                          <ul className="list-disc pl-4 text-xs space-y-1">
                            {selectedNodeData.genome.mutations.map((m, i) => (
                              <li key={i}>{m}</li>
                            ))}
                          </ul>
                        </div>
                      )}
                    </div>
                  ) : (
                    <div className="flex h-full flex-col items-center justify-center text-center text-muted-foreground space-y-2 opacity-60">
                      <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                        <circle cx="12" cy="12" r="10" />
                        <path d="M12 16v-4" />
                        <path d="M12 8h.01" />
                      </svg>
                      <p className="text-sm">Select a node in the graph to view genome details</p>
                    </div>
                  )}
                </div>
              </div>
            )}
          </CardContent>
        </Card>

        {/* SECTION 2: Economic Impact */}
        <Card>
          <CardHeader>
            <CardTitle>Economic Impact</CardTitle>
          </CardHeader>
          <CardContent>
            {impactError && (
              <div className="rounded-lg border border-destructive/40 bg-destructive/10 p-3 text-sm text-destructive">
                {impactError}
              </div>
            )}

            {!isImpactLoading && !impactError && impact?.status === "run_arena_first" && (
              <>
                <p className="border-l-2 border-dashed border-muted-foreground/50 pl-3 text-base font-medium text-muted-foreground">
                  Run Adversarial Arena to compute
                </p>
                <p className="mt-1 text-xs text-muted-foreground">
                  No arena run has completed in this session yet
                </p>
              </>
            )}

            {impact && impact.status === "ok" && (
              <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-4">
                <Card size="sm">
                  <CardHeader>
                    <CardTitle className="text-sm font-medium text-muted-foreground">
                      Total Attack Value
                    </CardTitle>
                  </CardHeader>
                  <CardContent>
                    <p className="text-3xl font-semibold tabular-nums">
                      &#8377;{(impact.total_attack_value_inr ?? 0).toLocaleString("en-IN", { maximumFractionDigits: 0 })}
                    </p>
                  </CardContent>
                </Card>

                <Card size="sm">
                  <CardHeader>
                    <CardTitle className="text-sm font-medium text-muted-foreground">
                      Value Caught by M0
                    </CardTitle>
                  </CardHeader>
                  <CardContent>
                    <p className="text-3xl font-semibold tabular-nums">
                      &#8377;{(impact.value_caught_by_m0_inr ?? 0).toLocaleString("en-IN", { maximumFractionDigits: 0 })}
                    </p>
                  </CardContent>
                </Card>

                <Card size="sm">
                  <CardHeader>
                    <CardTitle className="text-sm font-medium text-muted-foreground">
                      Value Caught After Hardening
                    </CardTitle>
                  </CardHeader>
                  <CardContent>
                    <p className="text-3xl font-semibold tabular-nums">
                      &#8377;{(impact.value_caught_after_hardening_inr ?? 0).toLocaleString("en-IN", { maximumFractionDigits: 0 })}
                    </p>
                  </CardContent>
                </Card>

                <Card size="sm">
                  <CardHeader>
                    <CardTitle className="text-sm font-medium text-muted-foreground">
                      Synthetic Fraud Value Prevented
                    </CardTitle>
                  </CardHeader>
                  <CardContent>
                    <p className="text-3xl font-semibold tabular-nums" style={{ color: "var(--neon-green)" }}>
                      &#8377;{(impact.incremental_value_prevented_inr ?? 0).toLocaleString("en-IN", { maximumFractionDigits: 0 })}
                    </p>
                  </CardContent>
                </Card>

                <Card size="sm">
                  <CardHeader>
                    <CardTitle className="text-sm font-medium text-muted-foreground">
                      Additional Transactions Caught
                    </CardTitle>
                  </CardHeader>
                  <CardContent>
                    <p className="text-3xl font-semibold tabular-nums">
                      {(impact.additional_transactions_caught ?? 0).toLocaleString()}
                    </p>
                  </CardContent>
                </Card>

                <Card size="sm">
                  <CardHeader>
                    <CardTitle className="text-sm font-medium text-muted-foreground">
                      M0 Evasion
                    </CardTitle>
                  </CardHeader>
                  <CardContent>
                    <p className="text-3xl font-semibold tabular-nums">
                      {((impact.m0_evasion_rate ?? 0) * 100).toFixed(1)}%
                    </p>
                  </CardContent>
                </Card>

                <Card size="sm">
                  <CardHeader>
                    <CardTitle className="text-sm font-medium text-muted-foreground">
                      Post-Hardening Evasion
                    </CardTitle>
                  </CardHeader>
                  <CardContent>
                    <p className="text-3xl font-semibold tabular-nums">
                      {((impact.post_hardening_evasion_rate ?? 0) * 100).toFixed(1)}%
                    </p>
                  </CardContent>
                </Card>
              </div>
            )}
            
            {impact && impact.status === "ok" && impact.methodology && (
              <p className="mt-4 text-xs italic text-muted-foreground">
                {impact.methodology}
              </p>
            )}
          </CardContent>
        </Card>

        {/* SECTION 3: Threat Intelligence Export */}
        <Card>
          <CardHeader>
            <CardTitle>Threat Intelligence Export</CardTitle>
            <p className="text-xs text-muted-foreground">
              STIX 2.1-shaped bundle for one known genome, built as a plain dict server-side --
              no new library added.
            </p>
          </CardHeader>
          <CardContent className="space-y-4">
            {!lineage?.run_id ? (
              <p className="border-l-2 border-dashed border-muted-foreground/50 pl-3 text-base font-medium text-muted-foreground">
                Run Adaptive Arena first
              </p>
            ) : (
              <div className="flex flex-col gap-4 sm:flex-row sm:items-end">
                <div className="flex-1 space-y-1.5">
                  <label htmlFor="export-genome-select" className="text-sm font-medium text-muted-foreground">
                    Attack family
                  </label>
                  <select
                    id="export-genome-select"
                    value={exportGenomeId}
                    disabled={isExporting || lineage.trajectory.length === 0}
                    onChange={(e) => setExportGenomeId(e.target.value)}
                    className="w-full rounded-md border border-border bg-input/30 px-3 py-2 text-sm text-foreground disabled:opacity-50"
                  >
                    {lineage.trajectory.map((t) => (
                      <option key={t.genome_id} value={t.genome_id}>
                        Generation {t.generation} ({t.genome_id})
                      </option>
                    ))}
                  </select>
                </div>
                <Button onClick={handleExport} disabled={isExporting || lineage.trajectory.length === 0}>
                  {isExporting ? "Exporting..." : "Export STIX 2.1"}
                </Button>
              </div>
            )}

            {exportError && (
              <div className="rounded-lg border border-destructive/40 bg-destructive/10 p-3 text-sm text-destructive">
                {exportError}
              </div>
            )}
          </CardContent>
        </Card>
      </div>
    </div>
  );
}
