"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import { Canvas, useFrame } from "@react-three/fiber";
import { OrbitControls, Html } from "@react-three/drei";
import * as THREE from "three";

import { ApiError, fetchImmuneMemory, type ImmuneMemoryRecord } from "@/lib/api";

/**
 * "Payment Threat Universe" -- Sentinel-X's signature 3D visualization.
 *
 * Real data only: GET /api/v1/immune-memory, the accumulated record of
 * every attack genome this server session has discovered across ALL
 * families/runs (Arena, adaptive search, recursive certification). This is
 * deliberately NOT the same dataset as Threat Observatory's Fraud DNA tree
 * (one adaptive run's generations) -- it's the cross-run, cross-family
 * memory store, a genuinely different real dataset, not a re-skin.
 *
 * Layout (all derived from real fields, nothing fabricated):
 * - angle: attack_family (clustered into wedges around the circle)
 * - radius: generation (deeper evolution = further from center)
 * - height: best_evasion (how far the attack got past the defense)
 * - color: current_status / residual_evasion severity
 *
 * Starts genuinely empty until the user runs Red Team / Arena / adaptive
 * search / certification -- shown honestly as an empty state, not a
 * fabricated idle animation.
 */

const FAMILY_ORDER = [
  "micro_structuring",
  "synthetic_identity_drift",
  "behavioral_camouflage",
  "social_engineering_coercion",
  "synthetic_voice_authorization",
];

function familyAngle(family: string): number {
  const idx = FAMILY_ORDER.indexOf(family);
  const slot = idx === -1 ? FAMILY_ORDER.length : idx;
  const total = FAMILY_ORDER.length + 1;
  return (slot / total) * Math.PI * 2;
}

function severityColor(record: ImmuneMemoryRecord): string {
  if (record.current_status === "RETIRED") return "#3b82f6"; // neutralized -- neutral blue
  if (record.residual_evasion > 0.3) return "#ff3b5c"; // still highly evasive -- red
  if (record.residual_evasion > 0.05) return "#f5c04a"; // partially caught -- amber
  return "#39ff88"; // fully caught -- green
}

interface ThreatNodeProps {
  record: ImmuneMemoryRecord;
  position: [number, number, number];
  isSelected: boolean;
  onSelect: () => void;
}

function ThreatNode({ record, position, isSelected, onSelect }: ThreatNodeProps) {
  const meshRef = useRef<THREE.Mesh>(null);
  const [hovered, setHovered] = useState(false);
  const color = severityColor(record);

  useFrame((_, delta) => {
    if (meshRef.current) {
      meshRef.current.rotation.y += delta * 0.15;
    }
  });

  const scale = 0.35 + Math.min(record.best_evasion, 1) * 0.5;

  return (
    <group position={position}>
      <mesh
        ref={meshRef}
        scale={hovered || isSelected ? scale * 1.3 : scale}
        onPointerOver={(e) => {
          e.stopPropagation();
          setHovered(true);
        }}
        onPointerOut={() => setHovered(false)}
        onClick={(e) => {
          e.stopPropagation();
          onSelect();
        }}
      >
        <icosahedronGeometry args={[1, 1]} />
        <meshStandardMaterial
          color={color}
          emissive={color}
          emissiveIntensity={isSelected ? 1.1 : hovered ? 0.7 : 0.35}
          roughness={0.35}
          metalness={0.2}
        />
      </mesh>
      {(hovered || isSelected) && (
        <Html center distanceFactor={10} style={{ pointerEvents: "none" }}>
          <div className="whitespace-nowrap rounded-md border border-border bg-card/95 px-2 py-1 text-[11px] font-mono text-foreground shadow-lg">
            {record.genome_id}
          </div>
        </Html>
      )}
    </group>
  );
}

function Scene({
  records,
  selectedId,
  onSelect,
}: {
  records: ImmuneMemoryRecord[];
  selectedId: string | null;
  onSelect: (id: string) => void;
}) {
  const positioned = useMemo(() => {
    return records.map((r) => {
      const angle = familyAngle(r.attack_family);
      const radius = 2.5 + r.generation * 1.4;
      const jitter = (parseInt(r.genome_id.replace(/\D/g, "").slice(-3) || "0", 10) % 100) / 100;
      const theta = angle + (jitter - 0.5) * 0.5;
      const x = Math.cos(theta) * radius;
      const z = Math.sin(theta) * radius;
      const y = r.best_evasion * 4 - 1;
      return { record: r, position: [x, y, z] as [number, number, number] };
    });
  }, [records]);

  return (
    <>
      <ambientLight intensity={0.4} />
      <pointLight position={[10, 10, 10]} intensity={1.2} color="#39ff88" />
      <pointLight position={[-10, -5, -10]} intensity={0.6} color="#3b82f6" />
      <gridHelper args={[24, 24, "#23262b", "#1a1d21"]} position={[0, -1.5, 0]} />
      {positioned.map(({ record, position }) => (
        <ThreatNode
          key={record.memory_id}
          record={record}
          position={position}
          isSelected={selectedId === record.memory_id}
          onSelect={() => onSelect(record.memory_id)}
        />
      ))}
      <OrbitControls enablePan={false} minDistance={4} maxDistance={30} autoRotate autoRotateSpeed={0.4} />
    </>
  );
}

function LegendRow({ color, label }: { color: string; label: string }) {
  return (
    <div className="flex items-center gap-1.5">
      <span className="h-2 w-2 rounded-full" style={{ backgroundColor: color, boxShadow: `0 0 4px ${color}` }} />
      <span className="text-muted-foreground">{label}</span>
    </div>
  );
}

export function ThreatUniverse() {
  const [records, setRecords] = useState<ImmuneMemoryRecord[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [selectedId, setSelectedId] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    fetchImmuneMemory()
      .then((body) => {
        if (!cancelled) setRecords(body.records);
      })
      .catch((err) => {
        if (!cancelled) setError(err instanceof ApiError ? err.message : String(err));
      });
    return () => {
      cancelled = true;
    };
  }, []);

  const selected = records?.find((r) => r.memory_id === selectedId) ?? null;

  return (
    <div className="relative h-[420px] w-full overflow-hidden rounded-lg border border-border bg-black/40">
      {error && (
        <div className="absolute inset-0 z-10 flex items-center justify-center bg-background/90 p-4 text-center text-sm text-destructive">
          {error}
        </div>
      )}

      {!error && records !== null && records.length === 0 && (
        <div className="absolute inset-0 z-10 flex flex-col items-center justify-center gap-1 bg-background/70 text-center">
          <p className="text-sm font-medium text-muted-foreground">No threats discovered yet</p>
          <p className="max-w-xs text-xs text-muted-foreground">
            Run Red Team Lab or the Adversarial Arena to populate the threat universe with real
            discovered attack genomes.
          </p>
        </div>
      )}

      {records === null && !error && (
        <div className="absolute inset-0 z-10 flex items-center justify-center text-sm text-muted-foreground">
          Loading threat universe...
        </div>
      )}

      {records !== null && records.length > 0 && (
        <>
          <Canvas camera={{ position: [0, 3, 12], fov: 50 }} dpr={[1, 1.5]}>
            <Scene records={records} selectedId={selectedId} onSelect={setSelectedId} />
          </Canvas>
          <div className="pointer-events-none absolute right-3 top-3 z-10 flex flex-col gap-1 rounded-md border border-border bg-card/80 p-2 text-[10px] backdrop-blur">
            <LegendRow color="#ff3b5c" label="Still evasive (>30%)" />
            <LegendRow color="#f5c04a" label="Partially caught" />
            <LegendRow color="#39ff88" label="Fully caught" />
            <LegendRow color="#3b82f6" label="Retired" />
          </div>
        </>
      )}

      {selected && (
        <div className="absolute bottom-3 left-3 right-3 z-10 rounded-md border border-border bg-card/95 p-3 text-xs backdrop-blur">
          <div className="flex flex-wrap items-center gap-x-4 gap-y-1">
            <span className="font-mono font-semibold">{selected.genome_id}</span>
            <span className="text-muted-foreground">family: {selected.attack_family}</span>
            <span className="text-muted-foreground">generation: {selected.generation}</span>
            <span className="text-muted-foreground">best evasion: {(selected.best_evasion * 100).toFixed(1)}%</span>
            <span className="text-muted-foreground">residual: {(selected.residual_evasion * 100).toFixed(1)}%</span>
            <span className="text-muted-foreground">status: {selected.current_status}</span>
          </div>
        </div>
      )}
    </div>
  );
}
