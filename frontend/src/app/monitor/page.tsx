"use client";

import { Suspense, useState, useEffect, useRef, useCallback } from "react";
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
  VideoOff,
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
  const initialGoal = searchParams.get("goal") || "elderly_care";
  const isClinicalMode = ["elderly_care", "bed_exit", "immobility", "line_pulling", "post_op", "wandering"].includes(initialGoal);
  const activeGoals = isClinicalMode ? CLINICAL_GOALS : GOALS;
  const [selectedGoal, setSelectedGoal] = useState(initialGoal);
  const [isMonitoring, setIsMonitoring] = useState(false);
  const [isStarting, setIsStarting] = useState(false);
  const [alerts, setAlerts] = useState<Alert[]>([]);
  const [personsDetected, setPersonsDetected] = useState(0);
  const [cameraActive, setCameraActive] = useState(false);
  const [toolCalls, setToolCalls] = useState<{name: string; args: string; timestamp: number; result?: string}[]>([]);
  const [decisions, setDecisions] = useState<{text: string; timestamp: number}[]>([]);
  const videoRef = useRef<HTMLVideoElement>(null);
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const toolPollRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const startCamera = useCallback(async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: 640, height: 480, facingMode: "user" },
      });
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        setCameraActive(true);
      }
    } catch {
      console.error("Camera access denied");
    }
  }, []);

  const stopCamera = useCallback(() => {
    if (videoRef.current?.srcObject) {
      const stream = videoRef.current.srcObject as MediaStream;
      stream.getTracks().forEach((t) => t.stop());
      videoRef.current.srcObject = null;
      setCameraActive(false);
    }
  }, []);

  const startMonitoring = async () => {
    setIsStarting(true);
    try {
      if (!cameraActive) await startCamera();

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
        if (toolData.tools) setToolCalls(toolData.tools.slice(-30));
        if (decData.decisions) setDecisions(decData.decisions.slice(-10));
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

  // Stream video frames to backend when monitoring
  useEffect(() => {
    if (!isMonitoring || !cameraActive || !videoRef.current) return;

    let ws: WebSocket;
    try {
      const wsBase = API_BASE.replace(/^http/, "ws");
      ws = new WebSocket(`${wsBase}/ws/video`);

      const sendFrame = () => {
        if (!videoRef.current || !ws || ws.readyState !== WebSocket.OPEN) return;
        const canvas = document.createElement("canvas");
        canvas.width = 640;
        canvas.height = 480;
        const ctx = canvas.getContext("2d");
        if (!ctx) return;
        ctx.drawImage(videoRef.current, 0, 0, 640, 480);
        const dataUrl = canvas.toDataURL("image/jpeg", 0.6);
        const base64 = dataUrl.split(",")[1];
        ws.send(JSON.stringify({ type: "frame", data: base64 }));
      };

      const frameInterval = setInterval(sendFrame, 500); // 2 FPS for monitoring

      return () => {
        clearInterval(frameInterval);
        if (ws && ws.readyState <= 1) ws.close();
      };
    } catch { /* backend unavailable */ }
  }, [isMonitoring, cameraActive]);

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
                        ? "border-red-500/60 bg-red-500/5 shadow-lg"
                        : "border-border hover:border-red-500/30"
                    }`}
                  >
                    {selectedGoal === goal.id && (
                      <Badge className="absolute -top-2 -right-2 bg-red-500 text-white text-[9px] px-1.5">Selected</Badge>
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
              className="w-full gap-2 py-6 text-lg rounded-xl bg-red-600 hover:bg-red-700 text-white"
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
          /* ── MONITORING ACTIVE ── */
          <div className="space-y-4">
            {/* Status Bar */}
            <div className="flex items-center justify-between p-4 rounded-xl bg-red-500/10 border border-red-500/30">
              <div className="flex items-center gap-3">
                <div className="h-3 w-3 rounded-full bg-red-500 animate-pulse" />
                <div>
                  <p className="text-sm font-bold text-red-400">MONITORING ACTIVE</p>
                  <p className="text-xs text-muted-foreground">{goalInfo?.icon} {goalInfo?.name}</p>
                </div>
              </div>
              <div className="flex items-center gap-4 text-xs text-muted-foreground">
                <span className="flex items-center gap-1"><Users className="h-3 w-3" /> {personsDetected} detected</span>
                <span className="flex items-center gap-1"><Bell className="h-3 w-3" /> {alerts.filter(a => a.type === "fall").length} alerts</span>
              </div>
              <Button onClick={stopMonitoring} variant="destructive" size="sm" className="gap-1">
                <EyeOff className="h-4 w-4" /> Stop
              </Button>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              {/* Camera Feed */}
              <Card className="border-border/50">
                <CardContent className="p-3">
                  <div className="flex items-center gap-2 mb-2">
                    {cameraActive ? <Video className="h-4 w-4 text-green-500" /> : <VideoOff className="h-4 w-4 text-red-500" />}
                    <span className="text-sm font-semibold">Live Feed</span>
                  </div>
                  <div className="relative rounded-lg overflow-hidden bg-black aspect-video">
                    <video
                      ref={videoRef}
                      autoPlay
                      playsInline
                      muted
                      className="w-full h-full object-cover"
                    />
                    {!cameraActive && (
                      <div className="absolute inset-0 flex items-center justify-center text-muted-foreground">
                        <p className="text-sm">Camera starting...</p>
                      </div>
                    )}
                    {/* Overlay */}
                    <div className="absolute top-2 left-2 flex items-center gap-1.5 px-2 py-1 rounded-md bg-red-600/80 backdrop-blur-sm">
                      <div className="h-1.5 w-1.5 rounded-full bg-white animate-pulse" />
                      <span className="text-[10px] font-bold text-white uppercase tracking-wider">LIVE</span>
                    </div>
                  </div>
                </CardContent>
              </Card>

              {/* Alert Feed */}
              <Card className="border-border/50">
                <CardContent className="p-3">
                  <div className="flex items-center gap-2 mb-2">
                    <AlertTriangle className="h-4 w-4 text-amber-500" />
                    <span className="text-sm font-semibold">Alert Feed</span>
                    <span className="text-xs text-muted-foreground ml-auto">{alerts.length} events</span>
                  </div>
                  <div className="space-y-2 max-h-[320px] overflow-y-auto">
                    {alerts.length === 0 ? (
                      <div className="text-center py-8 text-muted-foreground">
                        <Shield className="h-8 w-8 mx-auto mb-2 opacity-30" />
                        <p className="text-sm">Monitoring... no alerts yet</p>
                        <p className="text-xs mt-1">AI is watching the scene</p>
                      </div>
                    ) : (
                      [...alerts].reverse().map((alert, i) => (
                        <div
                          key={i}
                          className={`p-3 rounded-lg border text-sm ${
                            alert.type === "fall"
                              ? "bg-red-500/10 border-red-500/30 text-red-300"
                              : "bg-secondary/30 border-border/50 text-muted-foreground"
                          }`}
                        >
                          <div className="flex items-start gap-2">
                            {alert.type === "fall" ? (
                              <AlertTriangle className="h-4 w-4 mt-0.5 shrink-0 text-red-500" />
                            ) : (
                              <Activity className="h-4 w-4 mt-0.5 shrink-0" />
                            )}
                            <div className="flex-1 min-w-0">
                              <p className="text-xs leading-relaxed">{alert.message}</p>
                              <div className="flex items-center gap-2 mt-1">
                                <span className="text-[10px] opacity-60">
                                  {new Date(alert.timestamp * 1000).toLocaleTimeString()}
                                </span>
                                {alert.sent ? (
                                  <span className="flex items-center gap-0.5 text-[10px] text-green-500">
                                    <CheckCircle className="h-2.5 w-2.5" /> Sent to Telegram
                                  </span>
                                ) : (
                                  <span className="flex items-center gap-0.5 text-[10px] text-muted-foreground">
                                    <XCircle className="h-2.5 w-2.5" /> Logged only
                                  </span>
                                )}
                              </div>
                            </div>
                          </div>
                        </div>
                      ))
                    )}
                  </div>
                </CardContent>
              </Card>
              {/* Tool Calls Panel */}
              <Card className="border-border/50 md:row-span-1">
                <CardContent className="p-3">
                  <div className="flex items-center gap-2 mb-2">
                    <Terminal className="h-4 w-4 text-purple-500" />
                    <span className="text-sm font-semibold">Agent Tool Calls</span>
                    <span className="text-xs text-muted-foreground ml-auto">{toolCalls.length}</span>
                  </div>
                  <div className="space-y-1.5 max-h-[320px] overflow-y-auto font-mono">
                    {toolCalls.length === 0 && decisions.length === 0 ? (
                      <div className="text-center py-6 text-muted-foreground">
                        <Brain className="h-6 w-6 mx-auto mb-2 opacity-30" />
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
                </CardContent>
              </Card>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
