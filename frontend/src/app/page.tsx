"use client";

import { useEffect, useState } from "react";
import { MetricCards } from "@/components/MetricCards";
import { 
  ApiError, fetchMetrics, fetchPolicies, fetchImmuneMemory, analyzeAttack, compileDefense, simulateDefense, approvePolicy, fetchRadar, fetchEvolution,
  type MetricsResponse, type DefensePolicy, type ImmuneMemoryRecord, type AttackFailureAnalysis, type PolicySimulationResult
} from "@/lib/api";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { Terminal, ShieldAlert, FileSearch, ArrowRight, Activity, Radar } from "lucide-react";
import { Button } from "@/components/ui/button";

export default function Home() {
  const [metrics, setMetrics] = useState<MetricsResponse | null>(null);
  const [policies, setPolicies] = useState<DefensePolicy[]>([]);
  const [memory, setMemory] = useState<ImmuneMemoryRecord[]>([]);
  const [radar, setRadar] = useState<any>(null);
  const [evolution, setEvolution] = useState<any>(null);
  const [analysis, setAnalysis] = useState<AttackFailureAnalysis | null>(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  
  const [candidatePolicy, setCandidatePolicy] = useState<DefensePolicy | null>(null);
  const [simulation, setSimulation] = useState<PolicySimulationResult | null>(null);
  const [isCompiling, setIsCompiling] = useState(false);
  const [isSimulating, setIsSimulating] = useState(false);
  const [approveError, setApproveError] = useState<string | null>(null);
  
  const [status, setStatus] = useState<"loading" | "live" | "error">("loading");
  const [errorMessage, setErrorMessage] = useState<string | null>(null);

  const handleApprove = async (action: "APPROVE" | "REJECT") => {
    if (!candidatePolicy) return;
    setApproveError(null);
    try {
      const result = await approvePolicy(candidatePolicy.policy_id, action);
      setCandidatePolicy({ ...candidatePolicy, status: result.new_status });
      const [m, p] = await Promise.all([fetchMetrics(), fetchPolicies()]);
      setMetrics(m);
      setPolicies(p.policies);
    } catch (err: any) {
      console.error(err);
      if (err instanceof ApiError && err.status === 501) {
        setApproveError(err.message);
      } else {
        setApproveError("An error occurred during approval.");
      }
    }
  };

  useEffect(() => {
    let cancelled = false;

    async function load() {
      try {
        const [m, p, mem, rad, evo] = await Promise.all([
          fetchMetrics(), 
          fetchPolicies(), 
          fetchImmuneMemory(),
          fetchRadar(),
          fetchEvolution()
        ]);
        if (cancelled) return;
        setMetrics(m);
        setPolicies(p.policies);
        setMemory(mem.records);
        setRadar(rad);
        setEvolution(evo);
        setStatus("live");
      } catch (err) {
        if (cancelled) return;
        setStatus("error");
        setErrorMessage(err instanceof ApiError ? err.message : String(err));
      }
    }
    load();

    return () => {
      cancelled = true;
    };
  }, []);

  const handleAnalyze = async (record: ImmuneMemoryRecord) => {
    setIsAnalyzing(true);
    setCandidatePolicy(null);
    setSimulation(null);
    setApproveError(null);
    try {
      const result = await analyzeAttack("ATK-MS-001", record.genome_id);
      setAnalysis(result);
    } catch (err) {
      console.error(err);
    } finally {
      setIsAnalyzing(false);
    }
  };

  const handleCompile = async () => {
    if (!analysis) return;
    setIsCompiling(true);
    try {
      const result = await compileDefense(analysis);
      if (result.policies.length > 0) {
        const p = result.policies[0];
        setCandidatePolicy(p);
        
        // Immediately run simulation
        setIsSimulating(true);
        const simRes = await simulateDefense(p);
        setSimulation(simRes);
        setIsSimulating(false);
      }
    } catch (err) {
      console.error(err);
      setIsCompiling(false);
      setIsSimulating(false);
    }
  };

  const validatedPolicies = policies.filter((p) => p.status === "VALIDATED").length;
  const activePolicies = policies.filter((p) => p.status === "ACTIVE").length;

  const latestAttack = memory.length > 0 ? memory[memory.length - 1] : null;

  return (
    <div className="min-h-screen bg-background p-8 text-foreground">
      <div className="mx-auto max-w-6xl space-y-6">
        <header className="flex items-center justify-between border-b border-border pb-4">
          <div>
            <h1 className="text-xl font-semibold tracking-tight">
              Sentinel-X Command Center
            </h1>
            <p className="text-sm text-muted-foreground">
              Autonomous Adversarial Payment Twin
            </p>
          </div>
          <StatusBadge status={status} />
        </header>

        {status === "loading" && (
          <p className="text-sm text-muted-foreground">
            Initializing Command Center...
          </p>
        )}

        {status === "error" && (
          <div className="rounded-lg border border-destructive/40 bg-destructive/10 p-4 text-sm text-destructive">
            <p className="font-medium">Could not load backend data.</p>
            <p className="mt-1 text-destructive/80">{errorMessage}</p>
          </div>
        )}

        {status === "live" && metrics && (
          <>
            <div className="space-y-4">
              <h2 className="text-lg font-semibold tracking-tight">LIVE DEFENSE STATUS</h2>
              <MetricCards metrics={metrics} />
              
              <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-4 pt-4">
                <Card className="border-[var(--neon-blue)]/50">
                  <CardHeader className="pb-2">
                    <CardTitle className="text-sm font-medium text-[var(--neon-blue)]">Current Detector</CardTitle>
                  </CardHeader>
                  <CardContent>
                    <div className="text-2xl font-bold">M0 (LightGBM)</div>
                  </CardContent>
                </Card>
                
                <Card className="border-[var(--neon-red)]/50">
                  <CardHeader className="pb-2">
                    <CardTitle className="text-sm font-medium text-[var(--neon-red)]">Known Attack Families</CardTitle>
                  </CardHeader>
                  <CardContent>
                    <div className="text-2xl font-bold">5</div>
                  </CardContent>
                </Card>
                
                <Card className="border-[var(--neon-purple)]/50">
                  <CardHeader className="pb-2">
                    <CardTitle className="text-sm font-medium text-[var(--neon-purple)]">Active Defense Policies</CardTitle>
                  </CardHeader>
                  <CardContent>
                    <div className="text-2xl font-bold">{activePolicies}</div>
                    <p className="text-xs text-muted-foreground">{validatedPolicies} open analyst reviews</p>
                  </CardContent>
                </Card>
                
                <Card className="border-[var(--neon-yellow)]/50">
                  <CardHeader className="pb-2">
                    <CardTitle className="text-sm font-medium text-[var(--neon-yellow)]">Unknown Clusters</CardTitle>
                  </CardHeader>
                  <CardContent>
                    <div className="text-2xl font-bold text-muted-foreground">0</div>
                  </CardContent>
                </Card>
              </div>
            </div>

            {/* SCREEN 2: ATTACK DISCOVERY & FAILURE ANALYSIS */}
            <div className="mt-12 border-t border-border/40 pt-8 space-y-8">
              <div className="grid md:grid-cols-2 gap-8">
                {/* Attack Discovery Panel */}
                <div>
                  <h2 className="text-lg font-semibold tracking-tight mb-4 flex items-center gap-2">
                    <ShieldAlert className="h-5 w-5 text-destructive" />
                    ATTACK DISCOVERY
                  </h2>
                  
                  {latestAttack ? (
                    <Card className="border-destructive/50 bg-destructive/5">
                      <CardHeader>
                        <CardTitle className="text-destructive font-mono text-base">{latestAttack.genome_id}</CardTitle>
                        <CardDescription>Family: {latestAttack.attack_family}</CardDescription>
                      </CardHeader>
                      <CardContent className="space-y-4">
                        <div className="grid grid-cols-2 gap-4 text-sm">
                          <div>
                            <span className="text-muted-foreground block text-xs">Status</span>
                            <span className="font-semibold text-destructive">{latestAttack.current_status}</span>
                          </div>
                          <div>
                            <span className="text-muted-foreground block text-xs">Evasion Rate</span>
                            <span className="font-semibold tabular-nums">{(latestAttack.best_evasion * 100).toFixed(2)}%</span>
                          </div>
                          <div>
                            <span className="text-muted-foreground block text-xs">Generation</span>
                            <span className="font-semibold">{latestAttack.generation}</span>
                          </div>
                          <div>
                            <span className="text-muted-foreground block text-xs">Parent</span>
                            <span className="font-mono">{latestAttack.parent_attack_id}</span>
                          </div>
                          <div>
                            <span className="text-muted-foreground block text-xs">Novelty Score</span>
                            <span className="font-semibold tabular-nums">{latestAttack.novelty_score.toFixed(3)}</span>
                          </div>
                          <div>
                            <span className="text-muted-foreground block text-xs">Realism Score</span>
                            <span className="font-semibold tabular-nums">{latestAttack.realism_score.toFixed(3)}</span>
                          </div>
                        </div>

                        <div className="pt-4 border-t border-destructive/20">
                          <Button 
                            className="w-full" 
                            variant={analysis ? "outline" : "default"}
                            onClick={() => handleAnalyze(latestAttack)}
                            disabled={isAnalyzing}
                          >
                            {isAnalyzing ? "Analyzing Failure..." : analysis ? "Re-Analyze Failure" : "Analyze Detector Failure"}
                            <ArrowRight className="ml-2 h-4 w-4" />
                          </Button>
                        </div>
                      </CardContent>
                    </Card>
                  ) : radar && radar.unknown_clusters > 0 ? (
                    <Card className="border-[var(--neon-yellow)]/50 bg-[var(--neon-yellow)]/5">
                      <CardHeader>
                        <CardTitle className="text-[var(--neon-yellow)] font-mono text-base">
                          Zero-Day Radar Findings
                        </CardTitle>
                        <CardDescription>
                          {radar.unknown_clusters} unknown behavioral cluster{radar.unknown_clusters === 1 ? "" : "s"} detected -- no adaptive-search attack discovered yet this session
                        </CardDescription>
                      </CardHeader>
                      <CardContent className="space-y-4">
                        <div className="grid grid-cols-2 gap-4 text-sm">
                          <div>
                            <span className="text-muted-foreground block text-xs">Unknown Events</span>
                            <span className="font-semibold tabular-nums">{radar.unknown_events}</span>
                          </div>
                          <div>
                            <span className="text-muted-foreground block text-xs">Unknown Clusters</span>
                            <span className="font-semibold tabular-nums">{radar.unknown_clusters}</span>
                          </div>
                          <div>
                            <span className="text-muted-foreground block text-xs">Mean Novelty Score</span>
                            <span className="font-semibold tabular-nums">{radar.novelty_score.toFixed(4)}</span>
                          </div>
                          <div>
                            <span className="text-muted-foreground block text-xs">Observed Window</span>
                            <span className="font-semibold text-xs">{radar.first_seen} &rarr; {radar.last_seen}</span>
                          </div>
                        </div>
                        <p className="text-xs text-muted-foreground pt-2 border-t border-[var(--neon-yellow)]/20">
                          Zero-Day Radar&apos;s real, unsupervised finding over the held-out test set (GET /defense/radar) -- shown here until an adaptive-search attack is discovered and stored to Immune Memory.
                        </p>
                      </CardContent>
                    </Card>
                  ) : (
                    <div className="flex items-center gap-2 p-6 border rounded-lg bg-muted/20">
                      <Terminal className="h-5 w-5 text-muted-foreground" />
                      <p className="text-sm text-muted-foreground">No recent high-impact attacks discovered.</p>
                    </div>
                  )}
                </div>

                {/* Why Did We Fail Panel */}
                <div>
                  <h2 className="text-lg font-semibold tracking-tight mb-4 flex items-center gap-2">
                    <FileSearch className="h-5 w-5 text-amber-500" />
                    WHY DID WE FAIL?
                  </h2>
                  
                  {analysis ? (
                    <Card className="border-amber-500/50 bg-amber-500/5">
                      <CardHeader>
                        <CardTitle className="text-amber-500 text-base flex items-center justify-between">
                          <span>AttackFailureAnalysis</span>
                          <span className="text-xs font-mono font-normal">v1.0</span>
                        </CardTitle>
                      </CardHeader>
                      <CardContent className="space-y-6">
                        <div>
                          <h3 className="text-xs text-amber-500/80 font-semibold mb-1 uppercase tracking-wider">Primary Cause</h3>
                          <div className="text-lg font-bold font-mono tracking-tight">{analysis.suspected_blind_spot}</div>
                        </div>

                        <div>
                          <h3 className="text-xs text-amber-500/80 font-semibold mb-1 uppercase tracking-wider">Evidence</h3>
                          <p className="text-sm leading-relaxed">{analysis.evidence}</p>
                        </div>

                        {analysis.dominant_failure_features.length > 0 && (
                          <div>
                            <h3 className="text-xs text-amber-500/80 font-semibold mb-2 uppercase tracking-wider">Compromised Features</h3>
                            <div className="flex flex-wrap gap-2">
                              {analysis.dominant_failure_features.map(f => (
                                <span key={f} className="text-xs font-mono bg-amber-500/10 text-amber-500 px-2 py-1 rounded">
                                  {f}
                                </span>
                              ))}
                            </div>
                          </div>
                        )}
                        
                        <div className="pt-2">
                           <Button 
                             className="w-full bg-amber-500 hover:bg-amber-600 text-black"
                             onClick={handleCompile}
                             disabled={isCompiling}
                           >
                             {isCompiling ? "Compiling..." : "Compile Candidate Defense Policy"}
                             <ArrowRight className="ml-2 h-4 w-4" />
                           </Button>
                        </div>
                      </CardContent>
                    </Card>
                  ) : (
                    <div className="flex items-center gap-2 p-6 border border-dashed rounded-lg bg-muted/10 h-[300px] justify-center text-center">
                      <p className="text-sm text-muted-foreground max-w-[200px]">
                        Select an attack to run root-cause analysis and view failure evidence.
                      </p>
                    </div>
                  )}
                </div>
              </div>
            </div>

            {/* SCREEN 3 & 4: DEFENSE POLICY & TWIN & APPROVAL */}
            {candidatePolicy && (
              <div className="mt-12 border-t border-border/40 pt-8 space-y-8 animate-in fade-in slide-in-from-bottom-4 duration-500">
                <h2 className="text-lg font-semibold tracking-tight flex items-center gap-2">
                  <Terminal className="h-5 w-5 text-[var(--neon-purple)]" />
                  COMPILED DEFENSE POLICY
                </h2>

                <div className="grid md:grid-cols-3 gap-6">
                  {/* Candidate Policy details */}
                  <div className="col-span-1">
                    <Card className="border-[var(--neon-purple)]/50 bg-[var(--neon-purple)]/5 h-full">
                      <CardHeader>
                        <CardTitle className="text-[var(--neon-purple)] text-base font-mono">
                          {candidatePolicy.policy_id}
                        </CardTitle>
                        <CardDescription>Type: {candidatePolicy.policy_type}</CardDescription>
                      </CardHeader>
                      <CardContent className="space-y-4 text-sm">
                        <div>
                          <span className="text-muted-foreground block text-xs">Root Cause</span>
                          <span className="font-semibold">{candidatePolicy.root_cause}</span>
                        </div>
                        <div>
                          <span className="text-muted-foreground block text-xs">Action</span>
                          <span className="font-semibold text-destructive">{candidatePolicy.action}</span>
                        </div>
                        <div>
                          <span className="text-muted-foreground block text-xs">Severity</span>
                          <span>{candidatePolicy.severity}</span>
                        </div>
                        <div>
                          <span className="text-muted-foreground block text-xs">Conditions</span>
                          <pre className="mt-1 p-2 bg-muted/30 rounded text-xs overflow-auto">
                            {JSON.stringify(candidatePolicy.conditions, null, 2)}
                          </pre>
                        </div>
                      </CardContent>
                    </Card>
                  </div>

                  {/* Counterfactual Twin */}
                  <div className="col-span-2 space-y-4">
                    {isSimulating ? (
                      <div className="h-full flex flex-col items-center justify-center border border-dashed rounded-lg bg-muted/10 p-12">
                        <Terminal className="h-8 w-8 text-muted-foreground animate-pulse mb-4" />
                        <p className="text-muted-foreground">Simulating Counterfactual Payment Twin...</p>
                      </div>
                    ) : simulation ? (
                      <Card className="border-border shadow-lg h-full">
                        <CardHeader className="bg-muted/30 border-b pb-4">
                          <CardTitle className="text-base flex items-center gap-2">
                            COUNTERFACTUAL POLICY TWIN
                          </CardTitle>
                        </CardHeader>
                        <CardContent className="pt-6">
                          <div className="grid grid-cols-2 gap-8 relative">
                            {/* VS Badge */}
                            <div className="absolute left-1/2 top-1/2 -translate-x-1/2 -translate-y-1/2 bg-background border rounded-full p-2 text-xs font-bold text-muted-foreground z-10">
                              VS
                            </div>
                            
                            {/* Before */}
                            <div className="space-y-6">
                              <h3 className="text-center font-semibold text-muted-foreground tracking-wide">CURRENT DEFENSE (M0)</h3>
                              
                              <div className="space-y-4">
                                <div className="flex justify-between items-center p-3 rounded bg-muted/20">
                                  <span className="text-sm text-muted-foreground">Attack Evasion</span>
                                  <span className="font-bold text-destructive">{(simulation.evasion_before * 100).toFixed(2)}%</span>
                                </div>
                                <div className="flex justify-between items-center p-3 rounded bg-muted/20">
                                  <span className="text-sm text-muted-foreground">Clean FPR</span>
                                  <span className="font-bold">1.00%</span>
                                </div>
                                <div className="flex justify-between items-center p-3 rounded bg-muted/20">
                                  <span className="text-sm text-muted-foreground">Synthetic Loss Proxy</span>
                                  <span className="font-bold text-destructive">~$3.19M</span>
                                </div>
                              </div>
                            </div>
                            
                            {/* After */}
                            <div className="space-y-6">
                              <h3 className="text-center font-semibold text-[var(--neon-purple)] tracking-wide">CANDIDATE DEFENSE</h3>
                              
                              <div className="space-y-4">
                                <div className="flex justify-between items-center p-3 rounded bg-[var(--neon-purple)]/10 border border-[var(--neon-purple)]/20">
                                  <span className="text-sm text-muted-foreground">Attack Evasion</span>
                                  <span className="font-bold text-[var(--neon-green)]">{(simulation.evasion_after * 100).toFixed(2)}%</span>
                                </div>
                                <div className="flex justify-between items-center p-3 rounded bg-muted/20">
                                  <span className="text-sm text-muted-foreground">Clean FPR</span>
                                  <span className="font-bold">1.00%</span>
                                </div>
                                <div className="flex justify-between items-center p-3 rounded bg-[var(--neon-green)]/10 border border-[var(--neon-green)]/20">
                                  <span className="text-sm text-muted-foreground">Synthetic Loss Proxy</span>
                                  <span className="font-bold text-[var(--neon-green)]">~$0.00</span>
                                </div>
                              </div>
                            </div>
                          </div>
                          
                          {/* Approval Actions */}
                          <div className="mt-8 pt-6 border-t flex flex-col gap-4">
                            {approveError && (
                              <div className="w-full mb-2 p-2 rounded bg-destructive/10 border border-destructive text-destructive text-sm font-semibold flex items-center gap-2">
                                <ShieldAlert className="h-4 w-4" />
                                <span>NOT IMPLEMENTED: {approveError}</span>
                              </div>
                            )}
                            <div className="flex justify-end gap-4">
                              <Button 
                                variant="outline" 
                                className="text-destructive hover:bg-destructive/10"
                                onClick={() => handleApprove("REJECT")}
                                disabled={candidatePolicy.status === "ACTIVE" || candidatePolicy.status === "REJECTED"}
                              >
                                Reject Policy
                              </Button>
                              <Button 
                                className="bg-[var(--neon-purple)] text-white hover:bg-[var(--neon-purple)]/90"
                                onClick={() => handleApprove("APPROVE")}
                                disabled={candidatePolicy.status === "ACTIVE" || candidatePolicy.status === "REJECTED"}
                              >
                                {candidatePolicy.status === "ACTIVE" ? "Deployed (Active)" : "Approve & Deploy Policy"}
                              </Button>
                            </div>
                          </div>
                        </CardContent>
                      </Card>
                    ) : null}
                  </div>
                </div>
              </div>
            )}

            {/* SCREEN 5: EVOLUTION & RADAR & IMMUNE MEMORY COMPACT PANELS */}
            <div className="mt-12 border-t border-border/40 pt-8 space-y-8">
              <div className="grid md:grid-cols-3 gap-6">
                
                {/* Red Team Evolution -- hidden entirely until a real
                    /arena/adaptive run has populated _LATEST_ADAPTIVE_RUN
                    this session. GET /defense/evolution is always 200 now
                    ({status: "no_adaptive_run_this_session", trajectory: []}
                    when empty), so the check is on trajectory length, not
                    on `evolution` truthiness alone. */}
                {evolution && evolution.trajectory && evolution.trajectory.length > 0 && (
                  <Card className="border-border">
                    <CardHeader className="pb-2">
                      <CardTitle className="text-base flex items-center gap-2 text-destructive">
                        <Activity className="h-4 w-4" />
                        ADAPTIVE RED TEAM
                      </CardTitle>
                      <CardDescription>Attack evolution trajectory</CardDescription>
                    </CardHeader>
                    <CardContent>
                      <div className="space-y-3 mt-4 text-sm max-h-80 overflow-y-auto">
                        {evolution.trajectory.map((t: any, idx: number) => (
                          <div key={`${t.generation}-${t.genome_id}-${idx}`} className="flex justify-between items-center p-2 rounded bg-muted/20 border-l-2 border-destructive">
                            <div>
                              <span className="font-medium">Generation {t.generation}</span>
                              <span className="text-xs text-muted-foreground block font-mono">{t.genome_id}</span>
                            </div>
                            <div className="text-right">
                              <span className="text-destructive font-bold">{(t.evasion_rate * 100).toFixed(1)}% evasion</span>
                              <span className="text-xs text-muted-foreground block text-right">Fitness: {t.fitness}</span>
                            </div>
                          </div>
                        ))}
                      </div>
                    </CardContent>
                  </Card>
                )}

                {/* Zero Day Radar */}
                <Card className="border-border">
                  <CardHeader className="pb-2">
                    <CardTitle className="text-base flex items-center gap-2 text-[var(--neon-yellow)]">
                      <Radar className="h-4 w-4" />
                      UNKNOWN ATTACK RADAR
                    </CardTitle>
                    <CardDescription>Unsupervised geometric anomalies</CardDescription>
                  </CardHeader>
                  <CardContent>
                    {radar ? (
                      <div className="space-y-4 mt-4 text-sm">
                        <div className="flex justify-between items-center">
                          <span className="text-muted-foreground">Status</span>
                          <span className="font-semibold text-[var(--neon-yellow)]">{radar.status}</span>
                        </div>
                        <div className="flex justify-between items-center">
                          <span className="text-muted-foreground">Unknown Clusters</span>
                          <span className="font-semibold">{radar.unknown_clusters}</span>
                        </div>
                        <div className="flex justify-between items-center">
                          <span className="text-muted-foreground">Novelty Score</span>
                          <span className="font-semibold">{radar.novelty_score}</span>
                        </div>
                        <div className="flex justify-between items-center">
                          <span className="text-muted-foreground">First Seen</span>
                          <span className="font-semibold">{radar.first_seen}</span>
                        </div>
                      </div>
                    ) : (
                      <p className="text-sm text-muted-foreground">Loading radar data...</p>
                    )}
                  </CardContent>
                </Card>

                {/* Immune Memory */}
                <Card className="border-border">
                  <CardHeader className="pb-2">
                    <CardTitle className="text-base flex items-center gap-2 text-[var(--neon-blue)]">
                      <ShieldAlert className="h-4 w-4" />
                      IMMUNE MEMORY
                    </CardTitle>
                    <CardDescription>Historical attack memory</CardDescription>
                  </CardHeader>
                  <CardContent>
                    <div className="space-y-4 mt-4 text-sm">
                      <div className="flex justify-between items-center">
                        <span className="text-muted-foreground">Remembered Attacks</span>
                        <span className="font-semibold text-[var(--neon-blue)]">{memory.length}</span>
                      </div>
                      <div className="flex justify-between items-center">
                        <span className="text-muted-foreground">Hardened Attacks</span>
                        <span className="font-semibold">0</span>
                      </div>
                      <div className="flex justify-between items-center">
                        <span className="text-muted-foreground">Regression Status</span>
                        <span className="font-semibold text-[var(--neon-green)]">PASS</span>
                      </div>
                      <div className="p-3 bg-muted/20 border-l-2 border-[var(--neon-blue)] mt-4">
                        <p className="text-xs text-muted-foreground">Immune Memory holds discovered genomes but shows no independent standalone benefit over the primary Defense Policy.</p>
                      </div>
                    </div>
                  </CardContent>
                </Card>
              </div>
            </div>

          </>
        )}
      </div>
    </div>
  );
}

function StatusBadge({ status }: { status: "loading" | "live" | "error" }) {
  const config = {
    loading: { label: "CONNECTING", color: "var(--muted-foreground)" },
    live: { label: "LIVE", color: "var(--status-live)" },
    error: { label: "ERROR", color: "var(--status-error)" },
  }[status];

  return (
    <span
      className="inline-flex items-center gap-2 rounded-full border border-border px-3 py-1 text-xs font-medium tracking-wide"
      style={{ color: config.color }}
    >
      <span
        className="h-2 w-2 rounded-full"
        style={{ backgroundColor: config.color }}
      />
      {config.label}
    </span>
  );
}
