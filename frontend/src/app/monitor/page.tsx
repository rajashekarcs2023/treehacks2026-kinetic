"use client";

import { Suspense, useState, useEffect, useRef } from "react";
import { useSearchParams } from "next/navigation";
import { Card, CardContent } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import {
  Eye,
  EyeOff,
  Shield,
  AlertTriangle,
  Activity,
  Users,
  Send,
  Video,
  Bell,
  Clock,
  Loader2,
  CheckCircle,
  XCircle,
  Wrench,
  Brain,
  Terminal,
} from "lucide-react";

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

interface Alert {
  message: string;
  timestamp: number;
  type?: string;
  sent?: boolean;
}

const GOALS = [
  { id: "elderly_care", name: "Elderly Care Guardian", icon: "👴", desc: "Fall detection + inactivity alerts → Telegram", color: "red" },
  { id: "desk_watch", name: "Desk Guardian", icon: "🖥️", desc: "Alert when someone approaches your desk", color: "blue" },
  { id: "posture_coach", name: "Posture Monitor", icon: "🧘", desc: "Watch for bad posture and remind to correct", color: "green" },
  { id: "driver_monitor", name: "Driver Alertness", icon: "🚗", desc: "Drowsiness + distraction detection", color: "amber" },
  { id: "study_focus", name: "Study Focus", icon: "📚", desc: "Track focus time, alert on distraction", color: "purple" },
  { id: "general", name: "General Monitoring", icon: "👁️", desc: "Watch the space, report noteworthy events", color: "gray" },
];

const CLINICAL_GOALS = [
  { id: "elderly_care", name: "Fall Detection", icon: "🚨", desc: "Detect patient falls and alert nurse station immediately with photo", color: "red" },
  { id: "bed_exit", name: "Bed Exit Alert", icon: "🛏️", desc: "Patient attempting to leave bed unassisted — high fall risk", color: "red" },
  { id: "immobility", name: "Immobility / Pressure Ulcer", icon: "⏱️", desc: "Patient hasn't repositioned in 2+ hours — bedsore prevention", color: "amber" },
  { id: "line_pulling", name: "Line & Tube Safety", icon: "💉", desc: "Patient reaching for IV lines, catheters, or oxygen tubes", color: "orange" },
  { id: "post_op", name: "Post-Op Distress", icon: "🏥", desc: "Unusual agitation, restlessness, or distress signs after surgery", color: "purple" },
  { id: "wandering", name: "Wandering / Elopement", icon: "🚪", desc: "Confused or dementia patient leaving bed or room unsupervised", color: "blue" },
];

export default function MonitorPage() {
  return (
    <Suspense fallback={<div className="flex h-screen items-center justify-center bg-black text-muted-foreground">Loading monitor...</div>}>
      <MonitorContent />
    </Suspense>
  );
}

function MonitorContent() {
  const searchParams = useSearchParams();
  const initialGoal = searchParams.get("goal") || "general";
  const mode = searchParams.get("mode");
  const isClinicalMode = mode === "clinical" || ["bed_exit", "immobility", "line_pulling", "post_op", "wandering"].includes(initialGoal);
  const activeGoals = isClinicalMode ? CLINICAL_GOALS : GOALS;
  const [selectedGoal, setSelectedGoal] = useState(initialGoal);
  const [isMonitoring, setIsMonitoring] = useState(false);
  const [isStarting, setIsStarting] = useState(false);
  const [alerts, setAlerts] = useState<Alert[]>([]);
  const [personsDetected, setPersonsDetected] = useState(0);
  const [toolCalls, setToolCalls] = useState<{name: string; args: string; timestamp: number; result?: string}[]>([]);
  const [decisions, setDecisions] = useState<{text: string; timestamp: number}[]>([]);
  const frameRef = useRef<HTMLImageElement>(null);
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const toolPollRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const framePollRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const startMonitoring = async () => {
    setIsStarting(true);
    try {
      const res = await fetch(`${API_BASE}/api/monitoring/start`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ goal_id: selectedGoal }),
      });
      const data = await res.json();
      if (data.status === "monitoring_started") {
        setIsMonitoring(true);
        setAlerts([]);
      }
    } catch (err) {
      console.error("Failed to start monitoring:", err);
    } finally {
      setIsStarting(false);
    }
  };

  const stopMonitoring = async () => {
    try {
      await fetch(`${API_BASE}/api/monitoring/stop`, { method: "POST" });
    } catch { /* ignore */ }
    setIsMonitoring(false);
  };

  // Poll monitoring status + tool calls
  useEffect(() => {
    if (!isMonitoring) {
      if (pollRef.current) clearInterval(pollRef.current);
      if (toolPollRef.current) clearInterval(toolPollRef.current);
      return;
    }

    const poll = async () => {
      try {
        const res = await fetch(`${API_BASE}/api/monitoring/status`);
        const data = await res.json();
        if (data.alerts) setAlerts(data.alerts);
        if (data.persons_detected !== undefined) setPersonsDetected(data.persons_detected);
      } catch { /* ignore */ }
    };

    const pollTools = async () => {
      try {
        const [toolRes, decRes] = await Promise.all([
          fetch(`${API_BASE}/api/logs/tools`),
          fetch(`${API_BASE}/api/logs/decisions`),
        ]);
        const toolData = await toolRes.json();
        const decData = await decRes.json();
        if (toolData.tools) {
          const mapped = toolData.tools.map((t: Record<string, string>) => ({
            name: t.tool || t.name || "unknown",
            args: t.input_preview || t.args || "",
            timestamp: t.timestamp,
            result: t.result,
          }));
          setToolCalls(mapped.slice(-30));
        }
        if (decData.decisions) {
          const mappedDec = decData.decisions.map((d: Record<string, string>) => ({
            text: d.tool ? `${d.tool}(${d.input_preview || ""})` : d.text || "",
            timestamp: d.timestamp,
          }));
          setDecisions(mappedDec.slice(-10));
        }
      } catch { /* ignore */ }
    };

    poll();
    pollTools();
    pollRef.current = setInterval(poll, 3000);
    toolPollRef.current = setInterval(pollTools, 2000);
    return () => {
      if (pollRef.current) clearInterval(pollRef.current);
      if (toolPollRef.current) clearInterval(toolPollRef.current);
    };
  }, [isMonitoring]);

  // Poll backend camera frames for live feed
  useEffect(() => {
    if (!isMonitoring) {
      if (framePollRef.current) clearInterval(framePollRef.current);
      return;
    }
    const updateFrame = () => {
      if (frameRef.current) {
        frameRef.current.src = `${API_BASE}/api/frame?t=${Date.now()}`;
      }
    };
    updateFrame();
    framePollRef.current = setInterval(updateFrame, 500); // 2 FPS
    return () => {
      if (framePollRef.current) clearInterval(framePollRef.current);
    };
  }, [isMonitoring]);

  const goalInfo = activeGoals.find((g) => g.id === selectedGoal) || GOALS.find((g) => g.id === selectedGoal);
  const elapsed = alerts.length > 0 ? Math.round((Date.now() / 1000) - alerts[0].timestamp) : 0;

  return (
    <div className="flex-1 flex flex-col bg-background p-6 overflow-y-auto">
      <div className="max-w-5xl mx-auto w-full space-y-6">
        {/* Header */}
        <div>
          <h1 className="text-2xl font-bold flex items-center gap-2">
            <Eye className={`h-6 w-6 ${isClinicalMode ? "text-red-500" : "text-blue-500"}`} />
            {isClinicalMode ? "Clinical Patient Safety" : "Goal-Based Intelligence"}
          </h1>
          <p className="text-sm text-muted-foreground mt-1">
            {isClinicalMode
              ? "Hospital-grade patient monitoring — fall detection, bed exit alerts, nurse notifications via Telegram."
              : "Give Kinetic a goal. It watches, reasons, and acts autonomously — alerts via Telegram."}
          </p>
        </div>

        {!isMonitoring ? (
          /* ── SETUP ── */
          <div className="space-y-6">
            {/* Goal Selection */}
            <div className="space-y-3">
              <h2 className="text-lg font-semibold">{isClinicalMode ? "Select Patient Safety Protocol" : "Choose a Monitoring Goal"}</h2>
              <div className="grid grid-cols-2 md:grid-cols-3 gap-3">
                {activeGoals.map((goal) => (
                  <button
                    key={goal.id}
                    onClick={() => setSelectedGoal(goal.id)}
                    className={`relative flex flex-col items-start gap-2 p-4 rounded-xl border-2 transition-all text-left ${
                      selectedGoal === goal.id
                        ? isClinicalMode ? "border-red-500/60 bg-red-500/5 shadow-lg" : "border-blue-500/60 bg-blue-500/5 shadow-lg"
                        : isClinicalMode ? "border-border hover:border-red-500/30" : "border-border hover:border-blue-500/30"
                    }`}
                  >
                    {selectedGoal === goal.id && (
                      <Badge className={`absolute -top-2 -right-2 text-white text-[9px] px-1.5 ${isClinicalMode ? "bg-red-500" : "bg-blue-500"}`}>Selected</Badge>
                    )}
                    <span className="text-2xl">{goal.icon}</span>
                    <span className="text-sm font-semibold">{goal.name}</span>
                    <span className="text-[11px] text-muted-foreground leading-tight">{goal.desc}</span>
                  </button>
                ))}
              </div>
            </div>

            {/* Start Button */}
            <Button
              onClick={startMonitoring}
              disabled={isStarting}
              className={`w-full gap-2 py-6 text-lg rounded-xl text-white ${isClinicalMode ? "bg-red-600 hover:bg-red-700" : "bg-blue-600 hover:bg-blue-700"}`}
              size="lg"
            >
              {isStarting ? (
                <><Loader2 className="h-5 w-5 animate-spin" /> Starting Monitor...</>
              ) : (
                <><Shield className="h-5 w-5" /> Start Autonomous Monitoring</>
              )}
            </Button>
          </div>
        ) : (
          /* ── MONITORING ACTIVE — full-screen layout like coaching ── */
          <div className="fixed inset-0 flex bg-black z-50">
            {/* ── LEFT: Big Camera Feed ── */}
            <div className="flex-1 relative">
              <img
                ref={frameRef}
                alt="Live camera feed"
                className="w-full h-full object-cover"
                onError={(e) => { (e.target as HTMLImageElement).style.opacity = '0.3'; }}
                onLoad={(e) => { (e.target as HTMLImageElement).style.opacity = '1'; }}
              />

              {/* Top bar overlay */}
              <div className="absolute top-0 left-0 right-0 flex items-center justify-between p-4 bg-gradient-to-b from-black/80 to-transparent">
                <div className="flex items-center gap-3">
                  <div className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-red-600/90 backdrop-blur-sm">
                    <div className="h-2 w-2 rounded-full bg-white animate-pulse" />
                    <span className="text-xs font-bold text-white uppercase tracking-wider">LIVE</span>
                  </div>
                  <Badge className={`text-xs ${isClinicalMode ? "bg-red-500/20 text-red-300 border-red-500/40" : "bg-blue-500/20 text-blue-300 border-blue-500/40"}`}>
                    {goalInfo?.icon} {goalInfo?.name}
                  </Badge>
                </div>
                <div className="flex items-center gap-3">
                  <Badge variant="outline" className="border-white/20 text-white/80 bg-black/40 backdrop-blur-sm">
                    <Users className="h-3 w-3 mr-1" /> {personsDetected} detected
                  </Badge>
                  <Badge variant="outline" className="border-white/20 text-white/80 bg-black/40 backdrop-blur-sm">
                    <Bell className="h-3 w-3 mr-1" /> {alerts.length} alerts
                  </Badge>
                  <Button onClick={stopMonitoring} variant="destructive" size="sm" className="gap-1 bg-red-600 hover:bg-red-700">
                    <EyeOff className="h-4 w-4" /> Stop
                  </Button>
                </div>
              </div>

              {/* Bottom overlay — latest alert flash */}
              {alerts.length > 0 && (
                <div className="absolute bottom-0 left-0 right-0 p-4 bg-gradient-to-t from-black/80 to-transparent">
                  <div className={`p-3 rounded-lg border backdrop-blur-sm ${
                    alerts[alerts.length - 1]?.type === "fall"
                      ? "bg-red-500/20 border-red-500/40"
                      : alerts[alerts.length - 1]?.type === "check"
                      ? "bg-purple-500/20 border-purple-500/40"
                      : "bg-amber-500/20 border-amber-500/40"
                  }`}>
                    <p className="text-sm text-white font-medium">{alerts[alerts.length - 1]?.message}</p>
                    <p className="text-[10px] text-white/50 mt-1">
                      {new Date((alerts[alerts.length - 1]?.timestamp || 0) * 1000).toLocaleTimeString()}
                      {alerts[alerts.length - 1]?.sent && " · ✓ Sent to Telegram"}
                    </p>
                  </div>
                </div>
              )}

              {/* No feed placeholder */}
              {!frameRef.current?.src && (
                <div className="absolute inset-0 flex items-center justify-center">
                  <div className="text-center text-white/40">
                    <Video className="h-16 w-16 mx-auto mb-3 opacity-30" />
                    <p className="text-lg">Connecting to camera...</p>
                  </div>
                </div>
              )}
            </div>

            {/* ── RIGHT: Sidebar — Alerts + Tool Calls ── */}
            <div className="w-80 border-l border-white/10 bg-background flex flex-col overflow-hidden">
              {/* Alert Feed */}
              <div className="flex-1 overflow-y-auto border-b border-border">
                <div className="sticky top-0 bg-background z-10 flex items-center gap-2 p-3 border-b border-border">
                  <AlertTriangle className="h-4 w-4 text-amber-500" />
                  <span className="text-sm font-semibold">Alert Feed</span>
                  <span className="text-xs text-muted-foreground ml-auto">{alerts.length}</span>
                </div>
                <div className="p-3 space-y-2">
                  {alerts.length === 0 ? (
                    <div className="text-center py-12 text-muted-foreground">
                      <Shield className="h-10 w-10 mx-auto mb-3 opacity-30" />
                      <p className="text-sm font-medium">Monitoring active</p>
                      <p className="text-xs mt-1">No alerts yet — AI is watching</p>
                      <p className="text-[10px] mt-3 text-muted-foreground/60">
                        {isClinicalMode ? "Fall detection, bed exit, vitals monitoring..." : "Watching for events matching your goal..."}
                      </p>
                    </div>
                  ) : (
                    [...alerts].reverse().map((alert, i) => (
                      <div
                        key={i}
                        className={`p-3 rounded-lg border text-sm ${
                          alert.type === "fall" || alert.type === "elopement"
                            ? "bg-red-500/10 border-red-500/30 text-red-300"
                            : alert.type === "check"
                            ? "bg-purple-500/10 border-purple-500/30 text-purple-300"
                            : "bg-secondary/30 border-border/50 text-muted-foreground"
                        }`}
                      >
                        <div className="flex items-start gap-2">
                          {alert.type === "fall" || alert.type === "elopement" ? (
                            <AlertTriangle className="h-4 w-4 mt-0.5 shrink-0 text-red-500" />
                          ) : alert.type === "check" ? (
                            <Brain className="h-4 w-4 mt-0.5 shrink-0 text-purple-400" />
                          ) : (
                            <Activity className="h-4 w-4 mt-0.5 shrink-0" />
                          )}
                          <div className="flex-1 min-w-0">
                            <p className="text-xs leading-relaxed">{alert.message}</p>
                            <div className="flex items-center gap-2 mt-1">
                              <span className="text-[10px] opacity-60">
                                {new Date(alert.timestamp * 1000).toLocaleTimeString()}
                              </span>
                              {alert.sent && (
                                <span className="flex items-center gap-0.5 text-[10px] text-green-500">
                                  <CheckCircle className="h-2.5 w-2.5" /> Telegram
                                </span>
                              )}
                            </div>
                          </div>
                        </div>
                      </div>
                    ))
                  )}
                </div>
              </div>

              {/* Tool Calls Panel */}
              <div className="h-[40%] overflow-y-auto">
                <div className="sticky top-0 bg-background z-10 flex items-center gap-2 p-3 border-b border-border">
                  <Terminal className="h-4 w-4 text-purple-500" />
                  <span className="text-sm font-semibold">Agent Activity</span>
                  <span className="text-xs text-muted-foreground ml-auto">{toolCalls.length}</span>
                </div>
                <div className="p-3 space-y-1.5 font-mono">
                  {toolCalls.length === 0 && decisions.length === 0 ? (
                    <div className="text-center py-8 text-muted-foreground">
                      <Brain className="h-8 w-8 mx-auto mb-2 opacity-30" />
                      <p className="text-xs">Agent reasoning...</p>
                      <p className="text-[10px] mt-1">MCP tool calls appear here</p>
                    </div>
                  ) : (
                    <>
                      {decisions.length > 0 && (
                        <div className="mb-2 space-y-1">
                          {[...decisions].reverse().slice(0, 3).map((dec, i) => (
                            <div key={`d-${i}`} className="p-2 rounded bg-purple-500/10 border border-purple-500/20">
                              <div className="flex items-center gap-1 mb-0.5">
                                <Brain className="h-3 w-3 text-purple-400" />
                                <span className="text-[10px] font-semibold text-purple-400">REASONING</span>
                              </div>
                              <p className="text-[10px] text-muted-foreground leading-relaxed">{dec.text?.slice(0, 150)}</p>
                            </div>
                          ))}
                        </div>
                      )}
                      {[...toolCalls].reverse().map((tc, i) => (
                        <div key={`t-${i}`} className="p-2 rounded bg-secondary/40 border border-border/50">
                          <div className="flex items-center gap-1.5">
                            <Wrench className="h-3 w-3 text-blue-400 shrink-0" />
                            <span className="text-[10px] font-semibold text-blue-400 truncate">{tc.name}</span>
                          </div>
                          {tc.args && (
                            <p className="text-[9px] text-muted-foreground mt-0.5 truncate">
                              {typeof tc.args === 'string' ? tc.args.slice(0, 80) : JSON.stringify(tc.args).slice(0, 80)}
                            </p>
                          )}
                          {tc.result && (
                            <p className="text-[9px] text-green-500/70 mt-0.5 truncate">
                              → {typeof tc.result === 'string' ? tc.result.slice(0, 60) : JSON.stringify(tc.result).slice(0, 60)}
                            </p>
                          )}
                        </div>
                      ))}
                    </>
                  )}
                </div>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
