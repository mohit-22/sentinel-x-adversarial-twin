"use client";

import { useEffect, useState, useRef } from "react";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Activity, ShieldAlert, Crosshair, ArrowRight, Save, Play, CheckCircle2, AlertTriangle, FileSearch } from "lucide-react";
import { 
  ApiError,
  JudgeScenario, 
  ScenarioState, 
  createJudgeScenario, 
  runJudgeScenario, 
  getJudgeScenario, 
  resetJudgeScenario, 
  approveJudgeScenario 
} from "@/lib/api";

const PROFILES = {
  EASY: {
    difficulty: "EASY" as const,
    attack_scale: 100,
    adaptive_red_team_enabled: false,
    zero_day_radar_enabled: false,
    defense_compiler_enabled: false,
    human_approval_required: false,
    evolution_generations: 0,
  },
  HARD: {
    difficulty: "HARD" as const,
    attack_scale: 300,
    adaptive_red_team_enabled: true,
    zero_day_radar_enabled: false,
    defense_compiler_enabled: true,
    human_approval_required: true,
    evolution_generations: 3,
  },
  UNKNOWN: {
    difficulty: "UNKNOWN" as const,
    attack_scale: 200,
    adaptive_red_team_enabled: false,
    zero_day_radar_enabled: true,
    defense_compiler_enabled: true,
    human_approval_required: true,
    evolution_generations: 0,
  },
  EXTREME: {
    difficulty: "EXTREME" as const,
    attack_scale: 500,
    adaptive_red_team_enabled: true,
    zero_day_radar_enabled: true,
    defense_compiler_enabled: true,
    human_approval_required: true,
    evolution_generations: 5,
  }
};

const FAMILIES = [
  "micro_structuring",
  "social_engineering",
  "account_takeover",
  "synthetic_identity",
  "behavioral_camouflage"
];

export default function JudgeModePage() {
  const [profile, setProfile] = useState<keyof typeof PROFILES>("HARD");
  const [family, setFamily] = useState(FAMILIES[0]);
  const [scenarioId, setScenarioId] = useState<string | null>(null);
  const [state, setState] = useState<ScenarioState | null>(null);
  const [approveError, setApproveError] = useState<string | null>(null);
  const pollTimer = useRef<NodeJS.Timeout | null>(null);

  // Stop polling when unmounted
  useEffect(() => {
    return () => {
      if (pollTimer.current) clearInterval(pollTimer.current);
    };
  }, []);

  const pollState = async (id: string) => {
    try {
      const st = await getJudgeScenario(id);
      setState(st);
      if (st.is_completed || st.current_phase === "APPROVE") {
        if (pollTimer.current) clearInterval(pollTimer.current);
      }
    } catch (err) {
      console.error(err);
    }
  };

  const handleLaunch = async () => {
    try {
      const id = `judge-run-${Date.now()}`;
      const conf = PROFILES[profile];
      const payload: JudgeScenario = {
        scenario_id: id,
        seed: 42,
        attack_family: family,
        ...conf
      };
      
      const st = await createJudgeScenario(payload);
      setScenarioId(id);
      setState(st);
      
      await runJudgeScenario(id);
      
      // Start polling
      if (pollTimer.current) clearInterval(pollTimer.current);
      pollTimer.current = setInterval(() => pollState(id), 1000);
    } catch (err) {
      console.error("Launch failed", err);
    }
  };

  const handleApprove = async () => {
    if (!scenarioId) return;
    setApproveError(null);
    try {
      await approveJudgeScenario(scenarioId);
      // Resume polling
      if (pollTimer.current) clearInterval(pollTimer.current);
      pollTimer.current = setInterval(() => pollState(scenarioId), 1000);
    } catch (err: any) {
      console.error(err);
      if (err instanceof ApiError && err.status === 501) {
        setApproveError(err.message);
      } else {
        setApproveError("An error occurred during approval.");
      }
    }
  };

  const handleReset = async () => {
    if (scenarioId) await resetJudgeScenario(scenarioId);
    setScenarioId(null);
    setState(null);
    if (pollTimer.current) clearInterval(pollTimer.current);
  };

  const phases = ["PREPARE", "ATTACK", "DETECT", "ADAPT", "DISCOVER", "ANALYZE", "DEFEND", "SIMULATE", "APPROVE", "REPLAY", "SCORE"];
  const currentPhaseIndex = state 
    ? (state.is_completed ? phases.length : phases.indexOf(state.current_phase)) 
    : -1;

  return (
    <div className="min-h-screen bg-background p-8 text-foreground pb-24">
      <div className="mx-auto max-w-5xl space-y-8">
        
        {/* SAFETY BANNER */}
        <div className="bg-destructive/10 border-l-4 border-destructive p-4 flex items-center justify-between">
          <div className="flex items-center gap-3 text-destructive">
            <AlertTriangle className="h-6 w-6" />
            <div>
              <p className="font-bold tracking-widest uppercase">SYNTHETIC SANDBOX</p>
              <p className="text-sm">No live payment systems. No real cardholder data. No production transactions.</p>
            </div>
          </div>
          <ShieldAlert className="h-8 w-8 text-destructive/20" />
        </div>

        <header className="border-b border-border pb-4">
          <h1 className="text-3xl font-bold tracking-tight">Sentinel-X Judge Mode</h1>
          <p className="text-muted-foreground mt-1 text-lg">Fraud Cyber Range</p>
        </header>

        {/* TOP SECTION: CONFIGURATION */}
        <Card className="border-border shadow-sm">
          <CardHeader className="bg-muted/30 border-b">
            <CardTitle className="text-lg flex items-center gap-2">
              <Crosshair className="h-5 w-5" />
              CHALLENGE CONFIGURATION
            </CardTitle>
          </CardHeader>
          <CardContent className="pt-6">
            <div className="grid md:grid-cols-2 gap-8">
              <div className="space-y-4">
                <div>
                  <label className="text-sm font-medium text-muted-foreground block mb-2">Attack Family</label>
                  <select 
                    className="w-full bg-background border rounded p-2 text-sm focus:ring-1 focus:ring-[var(--neon-purple)]"
                    value={family}
                    onChange={(e) => setFamily(e.target.value)}
                    disabled={!!state}
                  >
                    {FAMILIES.map(f => <option key={f} value={f}>{f.replace("_", " ").toUpperCase()}</option>)}
                  </select>
                </div>
                <div>
                  <label className="text-sm font-medium text-muted-foreground block mb-2">Difficulty Profile</label>
                  <div className="flex gap-2">
                    {Object.keys(PROFILES).map((p) => (
                      <Button 
                        key={p} 
                        variant={profile === p ? "default" : "outline"}
                        onClick={() => setProfile(p as any)}
                        disabled={!!state}
                        size="sm"
                        className={profile === p ? "bg-[var(--neon-purple)]" : ""}
                      >
                        {p}
                      </Button>
                    ))}
                  </div>
                </div>
              </div>

              <div className="bg-muted/10 p-4 rounded-lg border font-mono text-xs space-y-2">
                <div className="flex justify-between border-b border-border/50 pb-1">
                  <span className="text-muted-foreground">Attack Scale</span>
                  <span>{PROFILES[profile].attack_scale} TXs</span>
                </div>
                <div className="flex justify-between border-b border-border/50 pb-1">
                  <span className="text-muted-foreground">Adaptive Red Team</span>
                  <span className={PROFILES[profile].adaptive_red_team_enabled ? "text-[var(--neon-red)]" : ""}>
                    {PROFILES[profile].adaptive_red_team_enabled ? "ENABLED" : "DISABLED"}
                  </span>
                </div>
                <div className="flex justify-between border-b border-border/50 pb-1">
                  <span className="text-muted-foreground">Zero-Day Radar</span>
                  <span className={PROFILES[profile].zero_day_radar_enabled ? "text-[var(--neon-yellow)]" : ""}>
                    {PROFILES[profile].zero_day_radar_enabled ? "ENABLED" : "DISABLED"}
                  </span>
                </div>
                <div className="flex justify-between border-b border-border/50 pb-1">
                  <span className="text-muted-foreground">Defense Compiler</span>
                  <span className={PROFILES[profile].defense_compiler_enabled ? "text-[var(--neon-blue)]" : ""}>
                    {PROFILES[profile].defense_compiler_enabled ? "ENABLED" : "DISABLED"}
                  </span>
                </div>
                <div className="flex justify-between">
                  <span className="text-muted-foreground">Human Approval</span>
                  <span className={PROFILES[profile].human_approval_required ? "text-[var(--neon-green)]" : ""}>
                    {PROFILES[profile].human_approval_required ? "REQUIRED" : "AUTO"}
                  </span>
                </div>
              </div>
            </div>

            <div className="mt-6 flex justify-end gap-4">
              <Button variant="outline" onClick={handleReset} disabled={!state}>
                RESET
              </Button>
              <Button onClick={handleLaunch} disabled={!!state} className="bg-[var(--neon-green)] text-black hover:bg-[var(--neon-green)]/90 disabled:opacity-50 disabled:pointer-events-none">
                <Play className="h-4 w-4 mr-2" />
                START CHALLENGE
              </Button>
            </div>
          </CardContent>
        </Card>

        {/* LIVE RUN LIFECYCLE */}
        {state && (
          <div className="animate-in fade-in slide-in-from-bottom-4 duration-500 space-y-8">
            <Card>
              <CardContent className="py-6">
                <div className="flex items-center justify-between text-xs font-mono tracking-tighter sm:tracking-normal sm:text-sm text-muted-foreground">
                  {phases.map((p, i) => {
                    let color = "text-muted-foreground";
                    if (i < currentPhaseIndex) color = "text-[var(--neon-green)]";
                    if (i === currentPhaseIndex) color = "text-[var(--neon-purple)] animate-pulse";
                    return (
                      <div key={p} className="flex flex-col items-center gap-1">
                        <div className={`h-3 w-3 rounded-full ${i <= currentPhaseIndex ? 'bg-current' : 'bg-muted'} ${color}`} />
                        <span className={color}>{p}</span>
                      </div>
                    );
                  })}
                </div>
              </CardContent>
            </Card>

            <div className="grid md:grid-cols-2 gap-8">
              
              {/* ATTACK NARRATIVE */}
              <div className="space-y-4">
                {currentPhaseIndex >= 2 && (
                  <Card className="border-[var(--neon-red)]/50 bg-[var(--neon-red)]/5">
                    <CardHeader className="pb-2">
                      <CardTitle className="text-sm text-[var(--neon-red)]">THE BLUE TEAM WAS CHALLENGED</CardTitle>
                    </CardHeader>
                    <CardContent>
                      <p className="text-2xl font-bold">{(state.baseline_evasion * 100).toFixed(1)}% Evasion</p>
                      <p className="text-xs text-muted-foreground">Initial Base Attack</p>
                    </CardContent>
                  </Card>
                )}
                
                {currentPhaseIndex >= 3 && state.scenario.adaptive_red_team_enabled && (
                  <Card className="border-destructive/50 bg-destructive/5">
                    <CardHeader className="pb-2">
                      <CardTitle className="text-sm text-destructive">THE RED TEAM EVOLVED</CardTitle>
                    </CardHeader>
                    <CardContent>
                      <p className="text-2xl font-bold">{(state.evolved_evasion * 100).toFixed(1)}% Evasion</p>
                      <p className="text-xs text-muted-foreground">Gen {state.scenario.evolution_generations} - {state.latest_genome_id}</p>
                    </CardContent>
                  </Card>
                )}

                {currentPhaseIndex >= 4 && state.scenario.zero_day_radar_enabled && (
                  <Card className="border-[var(--neon-yellow)]/50 bg-[var(--neon-yellow)]/5">
                    <CardHeader className="pb-2">
                      <CardTitle className="text-sm text-[var(--neon-yellow)]">
                        {state.scenario.difficulty === "UNKNOWN" ? "UNKNOWN BEHAVIOUR SURFACED" : "RADAR ANALYSIS"}
                      </CardTitle>
                    </CardHeader>
                    <CardContent>
                      <p className="text-xl font-bold">{state.radar_clusters} Unknown Clusters</p>
                      <p className="text-xs text-muted-foreground">Novelty Score: {state.radar_novelty.toFixed(3)}</p>
                    </CardContent>
                  </Card>
                )}
              </div>

              {/* DEFENSE NARRATIVE */}
              <div className="space-y-4">
                {currentPhaseIndex >= 5 && state.failure_cause && (
                  <Card className="border-amber-500/50 bg-amber-500/5">
                    <CardHeader className="pb-2">
                      <CardTitle className="text-sm text-amber-500 flex items-center gap-2">
                        <FileSearch className="h-4 w-4" /> ROOT CAUSE
                      </CardTitle>
                    </CardHeader>
                    <CardContent>
                      <p className="text-lg font-mono font-bold">{state.failure_cause}</p>
                    </CardContent>
                  </Card>
                )}

                {currentPhaseIndex >= 7 && state.candidate_policy_id && (
                  <Card className="border-[var(--neon-purple)]/50 bg-[var(--neon-purple)]/5">
                    <CardHeader className="pb-2">
                      <CardTitle className="text-sm text-[var(--neon-purple)] flex items-center gap-2">
                        <ShieldAlert className="h-4 w-4" /> THE DEFENSE RESPONDED
                      </CardTitle>
                    </CardHeader>
                    <CardContent className="space-y-4">
                      <p className="text-sm font-mono text-muted-foreground">{state.candidate_policy_id}</p>
                      
                      <div className="grid grid-cols-2 gap-4 pt-2 border-t border-[var(--neon-purple)]/20">
                        <div>
                          <p className="text-xs text-muted-foreground">Before Policy</p>
                          <p className="text-lg font-bold text-destructive">{(state.evolved_evasion * 100).toFixed(1)}%</p>
                        </div>
                        <div>
                          <p className="text-xs text-muted-foreground">After Policy</p>
                          <p className="text-lg font-bold text-[var(--neon-green)]">{(state.simulated_evasion_after * 100).toFixed(1)}%</p>
                        </div>
                      </div>

                      {state.current_phase === "APPROVE" && (
                        <div className="pt-4 border-t border-[var(--neon-purple)]/20 flex flex-col gap-2">
                           {approveError && (
                             <div className="w-full mb-2 p-2 rounded bg-destructive/10 border border-destructive text-destructive text-xs font-semibold flex items-center gap-2">
                               <ShieldAlert className="h-4 w-4" />
                               <span>NOT IMPLEMENTED: {approveError}</span>
                             </div>
                           )}
                           <Button className="w-full bg-[var(--neon-purple)] text-white hover:bg-[var(--neon-purple)]/90" onClick={handleApprove}>
                             APPROVE & RE-ATTACK
                           </Button>
                        </div>
                      )}
                    </CardContent>
                  </Card>
                )}
              </div>
            </div>
          </div>
        )}

        {/* FINAL SCORECARD */}
        {state?.scorecard && (
          <div className="mt-12 animate-in fade-in slide-in-from-bottom-8 duration-700">
            <h2 className="text-2xl font-bold tracking-tight mb-6 flex items-center gap-2">
              <CheckCircle2 className="h-6 w-6 text-[var(--neon-green)]" />
              DEFENSE READINESS SCORECARD
            </h2>
            
            <div className="grid md:grid-cols-3 gap-6">
              <Card className="md:col-span-1 border-[var(--neon-green)]/50 bg-muted/10 shadow-lg">
                <CardHeader>
                  <CardTitle className="text-center text-muted-foreground text-sm uppercase tracking-widest">Composite Score</CardTitle>
                </CardHeader>
                <CardContent className="flex flex-col items-center justify-center py-8">
                  <div className="text-7xl font-bold text-[var(--neon-green)]">{Math.round(state.scorecard.defense_readiness_score)}</div>
                  <p className="text-xs text-muted-foreground mt-4 text-center px-4">
                    Based on evasion reduction, safety margins, and anomaly detection.
                  </p>
                </CardContent>
              </Card>

              <Card className="md:col-span-2 shadow-sm border-border">
                <CardContent className="p-0">
                  <div className="grid sm:grid-cols-2 divide-y sm:divide-y-0 sm:divide-x divide-border">
                    <div className="p-6 space-y-4 text-sm">
                      <h3 className="font-semibold text-[var(--neon-purple)] uppercase tracking-wider text-xs border-b pb-2">Component: Adversarial Evasion</h3>
                      <div className="flex justify-between"><span className="text-muted-foreground">Initial Evasion</span><span className="font-mono">{(state.scorecard.initial_evasion*100).toFixed(1)}%</span></div>
                      <div className="flex justify-between"><span className="text-muted-foreground">Peak Red Team Evasion</span><span className="font-mono text-destructive">{(state.scorecard.best_evolved_evasion*100).toFixed(1)}%</span></div>
                      <div className="flex justify-between"><span className="text-muted-foreground">Final Post-Policy Evasion</span><span className="font-mono text-[var(--neon-green)] font-bold">{(state.scorecard.evasion_after*100).toFixed(1)}%</span></div>
                      <div className="flex justify-between"><span className="text-muted-foreground">Generations Survived</span><span className="font-mono">{state.scorecard.attack_generations}</span></div>
                    </div>
                    
                    <div className="p-6 space-y-4 text-sm">
                      <h3 className="font-semibold text-blue-400 uppercase tracking-wider text-xs border-b pb-2">Component: Safety & Validity</h3>
                      <div className="flex justify-between"><span className="text-muted-foreground">Clean FPR Penalty</span><span className="font-mono">{(state.scorecard.clean_fpr_delta*100).toFixed(2)}%</span></div>
                      <div className="flex justify-between"><span className="text-muted-foreground">Unknown Detection</span><span className="font-mono">{state.scorecard.cluster_count > 0 ? "YES" : "NO"}</span></div>
                      <div className="flex justify-between"><span className="text-muted-foreground">Customer Leakage</span><span className="font-mono text-[var(--neon-green)]">{state.scorecard.customer_leakage}</span></div>
                      <div className="flex justify-between"><span className="text-muted-foreground">Reproducible Run</span><span className="font-mono text-[var(--neon-green)]">{state.scorecard.reproducibility ? "YES" : "NO"}</span></div>
                    </div>
                  </div>
                </CardContent>
              </Card>
            </div>
            
            {/* PROVENANCE */}
            <div className="mt-8 bg-muted/10 border p-4 rounded-lg flex items-center justify-between text-xs font-mono text-muted-foreground overflow-x-auto whitespace-nowrap gap-4">
              <span className="font-bold text-foreground">PROVENANCE</span>
              <span>{state.scenario.attack_family}</span>
              <ArrowRight className="h-3 w-3" />
              <span>{state.failure_cause || "NO_FAILURE"}</span>
              <ArrowRight className="h-3 w-3" />
              <span>{state.candidate_policy_id || "NO_POLICY"}</span>
              <ArrowRight className="h-3 w-3" />
              <span className={state.policy_status === "ACTIVE" ? "text-[var(--neon-green)]" : ""}>{state.policy_status}</span>
              <ArrowRight className="h-3 w-3" />
              <span>{(state.scorecard.total_runtime).toFixed(2)}s Runtime</span>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
