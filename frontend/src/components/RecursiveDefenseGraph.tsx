import { useMemo, useCallback } from "react";
import {
  ReactFlow,
  Controls,
  Background,
  MiniMap,
  Handle,
  Position,
  Node,
  Edge,
  MarkerType,
} from "@xyflow/react";
import "@xyflow/react/dist/style.css";
import { CertificationResult } from "@/lib/api";

interface RecursiveDefenseGraphProps {
  result: CertificationResult | null;
}

const DefenseNode = ({ data }: { data: any }) => {
  const { id, parentId, f1, fpr, evasion, status, isFinal } = data;
  return (
    <div className="relative min-w-[200px] rounded-lg border-2 bg-card p-3 shadow-sm border-[var(--neon-cyan)] transition-transform hover:scale-[1.02]">
      <Handle type="target" position={Position.Top} className="!bg-muted-foreground !border-none !w-2 !h-2" />
      <div className="absolute -top-3 left-1/2 flex -translate-x-1/2 gap-1">
        <span className="whitespace-nowrap rounded bg-cyan-500/20 px-1.5 py-0.5 text-[10px] font-bold text-[var(--neon-cyan)]">DEFENSE</span>
      </div>
      <div className="mt-1 space-y-1">
        <p className="font-mono text-sm font-bold text-[var(--neon-cyan)]">{id}</p>
        {parentId && <p className="text-xs text-muted-foreground">Parent: <span className="font-mono">{parentId}</span></p>}
        <div className="grid grid-cols-2 gap-2 mt-2 pt-2 border-t border-border/50 text-xs">
          <div><span className="text-muted-foreground">Residual Evasion:</span> <span className="font-medium">{(evasion * 100).toFixed(1)}%</span></div>
          <div><span className="text-muted-foreground">Status:</span> <span className="font-medium">{status}</span></div>
          <div><span className="text-muted-foreground">F1:</span> <span className="font-medium">{f1.toFixed(3)}</span></div>
          <div><span className="text-muted-foreground">FPR:</span> <span className="font-medium">{fpr.toFixed(3)}</span></div>
        </div>
      </div>
      <Handle type="source" position={Position.Bottom} className="!bg-muted-foreground !border-none !w-2 !h-2" />
    </div>
  );
};

const AttackNode = ({ data }: { data: any }) => {
  const { id, target, evasion } = data;
  return (
    <div className="relative min-w-[180px] rounded-lg border-2 bg-card p-3 shadow-sm border-[var(--neon-red)] transition-transform hover:scale-[1.02]">
      <Handle type="target" position={Position.Top} className="!bg-muted-foreground !border-none !w-2 !h-2" />
      <div className="absolute -top-3 left-1/2 flex -translate-x-1/2 gap-1">
        <span className="whitespace-nowrap rounded bg-red-500/20 px-1.5 py-0.5 text-[10px] font-bold text-[var(--neon-red)]">ATTACK</span>
      </div>
      <div className="mt-1 space-y-1">
        <p className="font-mono text-xs font-semibold text-foreground truncate" title={id}>{id}</p>
        <p className="text-xs text-muted-foreground">Target: <span className="font-mono">{target}</span></p>
        <p className="text-xs text-muted-foreground">Evasion: <span className="font-medium text-[var(--neon-red)]">{(evasion * 100).toFixed(1)}%</span></p>
      </div>
      <Handle type="source" position={Position.Bottom} className="!bg-muted-foreground !border-none !w-2 !h-2" />
    </div>
  );
};

const PolicyNode = ({ data }: { data: any }) => {
  const { cause, noDefense, policyId } = data;
  
  if (noDefense) {
    return (
      <div className="relative min-w-[150px] rounded-lg border-2 border-dashed border-muted-foreground bg-card/50 p-3 shadow-sm">
        <Handle type="target" position={Position.Top} className="!bg-muted-foreground !border-none !w-2 !h-2" />
        <div className="text-center text-xs font-medium text-muted-foreground py-2">NO NEW DEFENSE GENERATED</div>
        <Handle type="source" position={Position.Bottom} className="!bg-muted-foreground !border-none !w-2 !h-2" />
      </div>
    );
  }

  return (
    <div className="relative min-w-[180px] rounded-lg border-2 bg-card p-3 shadow-sm border-amber-500 transition-transform hover:scale-[1.02]">
      <Handle type="target" position={Position.Top} className="!bg-muted-foreground !border-none !w-2 !h-2" />
      <div className="absolute -top-3 left-1/2 flex -translate-x-1/2 gap-1">
        <span className="whitespace-nowrap rounded bg-amber-500/20 px-1.5 py-0.5 text-[10px] font-bold text-amber-500">WEAKNESS / POLICY</span>
      </div>
      <div className="mt-1 space-y-1">
        {policyId && <p className="font-mono text-xs text-amber-500">{policyId}</p>}
        <p className="text-xs font-mono font-medium">{cause}</p>
      </div>
      <Handle type="source" position={Position.Bottom} className="!bg-muted-foreground !border-none !w-2 !h-2" />
    </div>
  );
};

const ResultNode = ({ data }: { data: any }) => {
  const { status, finalEvasion } = data;
  const isPass = status === "PASSED";
  const color = isPass ? "var(--neon-green)" : "var(--neon-red)";
  const bg = isPass ? "bg-green-500/20" : "bg-red-500/20";
  
  return (
    <div className="relative min-w-[150px] rounded-lg border-2 bg-card p-4 shadow-sm" style={{ borderColor: color }}>
      <Handle type="target" position={Position.Top} className="!bg-muted-foreground !border-none !w-2 !h-2" />
      <div className="text-center space-y-2">
        <div className={`mx-auto w-fit rounded px-2 py-1 text-xs font-bold ${bg}`} style={{ color }}>{status}</div>
        <div className="text-xs text-muted-foreground">Final Residual Evasion: <span className="font-medium text-foreground">{(finalEvasion * 100).toFixed(1)}%</span></div>
      </div>
    </div>
  );
};

const nodeTypes = {
  defenseNode: DefenseNode,
  attackNode: AttackNode,
  policyNode: PolicyNode,
  resultNode: ResultNode,
};

export default function RecursiveDefenseGraph({ result }: RecursiveDefenseGraphProps) {
  const { nodes, edges } = useMemo(() => {
    if (!result || !result.rounds || result.rounds.length === 0) {
      return { nodes: [], edges: [] };
    }

    const n: Node[] = [];
    const e: Edge[] = [];
    const X_POS = 200;
    const Y_STEP = 160;
    let currentY = 50;

    let previousNodeId: string | null = null;

    result.rounds.forEach((round, i) => {
      // 1. Defense Node
      const dNodeId = `def-${round.defense_id}-${i}`;
      n.push({
        id: dNodeId,
        position: { x: X_POS, y: currentY },
        type: "defenseNode",
        data: {
          id: round.defense_id,
          parentId: i > 0 ? result.rounds[i-1].candidate_defense_id : "None",
          evasion: round.evasion_rate,
          status: round.status,
          f1: round.f1,
          fpr: round.fpr,
        },
      });
      if (previousNodeId) {
        e.push({
          id: `e-${previousNodeId}-${dNodeId}`,
          source: previousNodeId,
          target: dNodeId,
          type: "smoothstep",
          style: { stroke: "var(--neon-cyan)" },
          markerEnd: { type: MarkerType.ArrowClosed, color: "var(--neon-cyan)" },
        });
      }
      currentY += Y_STEP;

      // 2. Attack Node
      const aNodeId = `atk-${round.attack_run_id}-${i}`;
      n.push({
        id: aNodeId,
        position: { x: X_POS, y: currentY },
        type: "attackNode",
        data: {
          id: round.attack_run_id,
          target: round.defense_id,
          evasion: round.evasion_rate,
        },
      });
      e.push({
        id: `e-${dNodeId}-${aNodeId}`,
        source: dNodeId,
        target: aNodeId,
        type: "smoothstep",
        style: { stroke: "var(--neon-red)" },
        markerEnd: { type: MarkerType.ArrowClosed, color: "var(--neon-red)" },
      });
      currentY += Y_STEP;

      // 3. Policy / Weakness Node
      const pNodeId = `pol-${i}`;
      n.push({
        id: pNodeId,
        position: { x: X_POS, y: currentY },
        type: "policyNode",
        data: {
          noDefense: !round.new_defense_created,
          cause: round.failure_cause || "UNKNOWN",
          policyId: round.candidate_defense_id ? `Generated: ${round.candidate_defense_id}` : null,
        },
      });
      e.push({
        id: `e-${aNodeId}-${pNodeId}`,
        source: aNodeId,
        target: pNodeId,
        type: "smoothstep",
        style: { stroke: "var(--neon-amber)", strokeDasharray: round.new_defense_created ? "none" : "5 5" },
        markerEnd: { type: MarkerType.ArrowClosed, color: "var(--neon-amber)" },
      });
      currentY += Y_STEP;
      
      previousNodeId = pNodeId;
    });

    // Final Certification Result Node
    const resNodeId = "result-node";
    n.push({
      id: resNodeId,
      position: { x: X_POS, y: currentY },
      type: "resultNode",
      data: {
        status: result.certification_status || result.status,
        finalEvasion: result.residual_evasion,
      },
    });
    if (previousNodeId) {
      e.push({
        id: `e-${previousNodeId}-${resNodeId}`,
        source: previousNodeId,
        target: resNodeId,
        type: "smoothstep",
        style: { stroke: "hsl(var(--muted-foreground))" },
        markerEnd: { type: MarkerType.ArrowClosed, color: "hsl(var(--muted-foreground))" },
      });
    }

    return { nodes: n, edges: e };
  }, [result]);

  if (!result) {
    return (
      <div className="flex h-[500px] w-full flex-col items-center justify-center text-center text-muted-foreground opacity-60">
        <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="mb-2">
          <circle cx="12" cy="12" r="10" />
          <path d="M12 16v-4" />
          <path d="M12 8h.01" />
        </svg>
        <p className="text-sm font-medium">Awaiting recursive defense run</p>
      </div>
    );
  }

  return (
    <div className="h-[600px] w-full border border-border rounded-lg overflow-hidden bg-dot-pattern">
      <ReactFlow
        nodes={nodes}
        edges={edges}
        nodeTypes={nodeTypes}
        fitView
        minZoom={0.2}
        maxZoom={1.5}
        colorMode="dark"
      >
        <Background gap={24} size={2} color="hsl(var(--muted-foreground))" />
        <Controls className="!bg-card !border-border !fill-foreground" />
      </ReactFlow>
    </div>
  );
}
