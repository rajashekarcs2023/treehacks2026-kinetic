"use client";

import { Suspense, useCallback, useEffect, useRef, useState } from "react";
import { useSearchParams } from "next/navigation";
import { Card, CardContent } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { ScoreRing } from "@/components/score-ring";
import { createCoachingWS, createVideoWS, startCoaching, stopCoaching } from "@/lib/api";
import {
  Video,
  VideoOff,
  Mic,
  MicOff,
  Play,
  Square,
  RotateCcw,
  Maximize2,
  ChevronDown,
  Activity,
  Gauge,
  Timer,
  TrendingUp,
  Sparkles,
  AlertTriangle,
} from "lucide-react";

interface JointFeedback {
  name: string;
  status: "good" | "warning" | "bad";
  angle: number;
  target: number;
  message: string;
}

const SKILL_CATEGORIES: Record<string, string[]> = {
  "Fitness": ["Squat", "Deadlift", "Push-up", "Lunge", "Plank", "Burpee"],
  "Dance": ["Salsa Basic", "Hip-hop Groove", "Ballet Plié", "Bachata Step"],
  "Sports": ["Tennis Serve", "Golf Swing", "Boxing Jab", "Batting Stance"],
  "Yoga": ["Warrior Pose", "Tree Pose", "Downward Dog", "Crow Pose"],
  "Martial Arts": ["Front Kick", "Roundhouse", "Jab-Cross", "Horse Stance"],
  "PT & Rehab": ["Knee Extension", "Shoulder Raise", "Glute Bridge", "Wall Sit"],
  "Music": ["Guitar Posture", "Piano Hands", "Drum Grip", "Violin Bow"],
  "Sign Language": ["ASL Alphabet", "Common Signs", "Finger Spelling"],
};

const ALL_SKILLS = Object.values(SKILL_CATEGORIES).flat();

export default function CoachPage() {
  return (
    <Suspense fallback={<div className="flex h-screen items-center justify-center bg-black text-muted-foreground">Loading coach...</div>}>
      <CoachContent />
    </Suspense>
  );
}

function CoachContent() {
  const searchParams = useSearchParams();
  const initialSkill = searchParams.get("skill") || "Squat";

  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [cameraActive, setCameraActive] = useState(false);
  const [micActive, setMicActive] = useState(false);
  const [isCoaching, setIsCoaching] = useState(false);
  const [selectedSkill, setSelectedSkill] = useState(initialSkill);
  const [showSkillPicker, setShowSkillPicker] = useState(false);
  const [isFullscreen, setIsFullscreen] = useState(false);

  // Coaching state
  const [currentScore, setCurrentScore] = useState(0);
  const [repCount, setRepCount] = useState(0);
  const [phase, setPhase] = useState("Ready");
  const [elapsed, setElapsed] = useState(0);
  const [avgScore, setAvgScore] = useState(0);
  const [bestScore, setBestScore] = useState(0);
  const [currentFeedback, setCurrentFeedback] = useState("");
  const [scores, setScores] = useState<number[]>([]);

  const [joints, setJoints] = useState<JointFeedback[]>([
    { name: "Left Knee", status: "good", angle: 92, target: 90, message: "Good depth" },
    { name: "Right Knee", status: "good", angle: 88, target: 90, message: "Good depth" },
    { name: "Hip", status: "warning", angle: 78, target: 85, message: "Push hips back more" },
    { name: "Torso", status: "bad", angle: 38, target: 55, message: "Chest up!" },
    { name: "Left Ankle", status: "good", angle: 72, target: 70, message: "Stable" },
    { name: "Right Ankle", status: "good", angle: 71, target: 70, message: "Stable" },
  ]);

  const [quality, setQuality] = useState({
    smoothness: 78,
    symmetry: 85,
    rangeOfMotion: 72,
    tempoConsistency: 90,
  });

  const wsRef = useRef<WebSocket | null>(null);
  const videoWsRef = useRef<WebSocket | null>(null);
  const [backendConnected, setBackendConnected] = useState(false);

  // Timer
  useEffect(() => {
    if (!isCoaching) return;
    const interval = setInterval(() => {
      setElapsed((e) => e + 1);
    }, 1000);
    return () => clearInterval(interval);
  }, [isCoaching]);

  // Connect coaching WebSocket for real-time data
  useEffect(() => {
    if (!isCoaching) return;

    let ws: WebSocket;
    try {
      ws = createCoachingWS();
      wsRef.current = ws;

      ws.onopen = () => {
        setBackendConnected(true);
        console.log("[Kinetic] Coaching WebSocket connected");
      };

      ws.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          if (data.score !== undefined) {
            setCurrentScore(data.score);
            setScores((prev) => {
              const updated = [...prev, data.score];
              setAvgScore(updated.reduce((a, b) => a + b, 0) / updated.length);
              setBestScore(Math.max(...updated));
              return updated;
            });
          }
          if (data.reps !== undefined) setRepCount(data.reps);
          if (data.phase) setPhase(data.phase);
          if (data.feedback) setCurrentFeedback(data.feedback);
          if (data.joints) {
            setJoints(data.joints.map((j: { name: string; status: string; angle: number; target: number; message: string }) => ({
              name: j.name,
              status: j.status as "good" | "warning" | "bad",
              angle: j.angle,
              target: j.target,
              message: j.message,
            })));
          }
          if (data.quality) {
            setQuality({
              smoothness: data.quality.smoothness ?? 0,
              symmetry: data.quality.symmetry ?? 0,
              rangeOfMotion: data.quality.range_of_motion ?? 0,
              tempoConsistency: data.quality.tempo_consistency ?? 0,
            });
          }
        } catch {
          // ignore malformed messages
        }
      };

      ws.onerror = () => {
        setBackendConnected(false);
        console.warn("[Kinetic] Coaching WS error — falling back to simulation");
      };

      ws.onclose = () => {
        setBackendConnected(false);
        wsRef.current = null;
      };
    } catch {
      setBackendConnected(false);
    }

    return () => {
      if (ws && ws.readyState <= 1) ws.close();
      wsRef.current = null;
    };
  }, [isCoaching]);

  // Fallback simulation when backend is not connected
  useEffect(() => {
    if (!isCoaching || backendConnected) return;
    const interval = setInterval(() => {
      const newScore = 65 + Math.random() * 30;
      setCurrentScore(newScore);
      setScores((prev) => {
        const updated = [...prev, newScore];
        setAvgScore(updated.reduce((a, b) => a + b, 0) / updated.length);
        setBestScore(Math.max(...updated));
        return updated;
      });

      if (Math.random() > 0.85) {
        setRepCount((r) => r + 1);
        setPhase("Peak");
        setTimeout(() => setPhase("Recovery"), 800);
        setTimeout(() => setPhase("Preparation"), 1600);
      }

      const feedbacks = [
        "Keep your chest up — torso angle is dropping",
        "Great depth! Hold that bottom position",
        "Push knees out — tracking slightly inward",
        "Nice tempo! Stay consistent",
        "Drive through your heels on the way up",
        "Excellent form on that rep!",
      ];
      setCurrentFeedback(feedbacks[Math.floor(Math.random() * feedbacks.length)]);

      setJoints((prev) =>
        prev.map((j) => ({
          ...j,
          status: (Math.random() > 0.3 ? "good" : Math.random() > 0.5 ? "warning" : "bad") as "good" | "warning" | "bad",
          angle: j.target + Math.floor(Math.random() * 20 - 10),
        }))
      );

      setQuality({
        smoothness: 70 + Math.random() * 25,
        symmetry: 75 + Math.random() * 20,
        rangeOfMotion: 65 + Math.random() * 30,
        tempoConsistency: 80 + Math.random() * 15,
      });
    }, 2000);
    return () => clearInterval(interval);
  }, [isCoaching, backendConnected]);

  // Stream webcam frames to backend
  useEffect(() => {
    if (!isCoaching || !cameraActive || !videoRef.current) return;

    let videoWs: WebSocket;
    try {
      videoWs = createVideoWS();
      videoWsRef.current = videoWs;

      const sendFrame = () => {
        if (!videoRef.current || !videoWs || videoWs.readyState !== WebSocket.OPEN) return;
        const canvas = document.createElement("canvas");
        canvas.width = 640;
        canvas.height = 480;
        const ctx = canvas.getContext("2d");
        if (!ctx) return;
        ctx.drawImage(videoRef.current, 0, 0, 640, 480);
        canvas.toBlob(
          (blob) => {
            if (blob && videoWs.readyState === WebSocket.OPEN) {
              blob.arrayBuffer().then((buf) => videoWs.send(buf));
            }
          },
          "image/jpeg",
          0.7
        );
      };

      const frameInterval = setInterval(sendFrame, 100); // ~10 FPS

      videoWs.onerror = () => console.warn("[Kinetic] Video WS error");
      videoWs.onclose = () => { videoWsRef.current = null; };

      return () => {
        clearInterval(frameInterval);
        if (videoWs && videoWs.readyState <= 1) videoWs.close();
        videoWsRef.current = null;
      };
    } catch {
      // backend not available
    }
  }, [isCoaching, cameraActive]);

  // Camera setup
  const startCamera = useCallback(async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: 1280, height: 720, facingMode: "user" },
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

  const toggleCoaching = async () => {
    if (isCoaching) {
      setIsCoaching(false);
      setPhase("Finished");
      try { await stopCoaching(); } catch { /* backend may be offline */ }
    } else {
      setIsCoaching(true);
      setRepCount(0);
      setScores([]);
      setElapsed(0);
      setPhase("Preparation");
      setCurrentFeedback("Starting coaching session...");
      if (!cameraActive) startCamera();
      try { await startCoaching(selectedSkill); } catch { /* backend may be offline — simulation will kick in */ }
    }
  };

  const formatTime = (s: number) => {
    const m = Math.floor(s / 60);
    const sec = s % 60;
    return `${m}:${sec.toString().padStart(2, "0")}`;
  };

  const getStatusColor = (status: string) => {
    if (status === "good") return "text-green-500";
    if (status === "warning") return "text-yellow-500";
    return "text-red-500";
  };

  const getStatusBg = (status: string) => {
    if (status === "good") return "bg-green-500/10 border-green-500/20";
    if (status === "warning") return "bg-yellow-500/10 border-yellow-500/20";
    return "bg-red-500/10 border-red-500/20";
  };

  return (
    <div className={`flex h-screen ${isFullscreen ? "" : ""}`}>
      {/* Main Camera View */}
      <div className="flex-1 flex flex-col relative bg-black">
        {/* Top Bar */}
        <div className="absolute top-0 left-0 right-0 z-10 flex items-center justify-between p-4 bg-gradient-to-b from-black/80 to-transparent">
          <div className="flex items-center gap-3">
            <Badge variant="outline" className="border-primary/40 text-primary">
              {selectedSkill}
            </Badge>
            {isCoaching && (
              <>
                <Badge variant="outline" className="border-green-500/40 text-green-500 animate-pulse">
                  LIVE
                </Badge>
                <span className="text-sm text-white/70 tabular-nums">{formatTime(elapsed)}</span>
              </>
            )}
          </div>
          <div className="flex items-center gap-2">
            <Button
              variant="ghost"
              size="icon"
              className="text-white/70 hover:text-white hover:bg-white/10"
              onClick={() => setIsFullscreen(!isFullscreen)}
            >
              <Maximize2 className="h-4 w-4" />
            </Button>
          </div>
        </div>

        {/* Video Feed */}
        <div className="flex-1 flex items-center justify-center relative">
          <video
            ref={videoRef}
            autoPlay
            playsInline
            muted
            className="h-full w-full object-cover"
            style={{ transform: "scaleX(-1)" }}
          />
          <canvas ref={canvasRef} className="absolute inset-0 h-full w-full" />

          {!cameraActive && (
            <div className="absolute inset-0 flex flex-col items-center justify-center bg-background/90 gap-4">
              <div className="h-24 w-24 rounded-full border-2 border-dashed border-muted-foreground/30 flex items-center justify-center">
                <Video className="h-10 w-10 text-muted-foreground/50" />
              </div>
              <p className="text-muted-foreground text-sm">Camera feed will appear here</p>
              <Button onClick={startCamera} className="gap-2">
                <Video className="h-4 w-4" />
                Enable Camera
              </Button>
            </div>
          )}

          {/* Real-time Feedback Overlay */}
          {isCoaching && currentFeedback && (
            <div className="absolute bottom-24 left-1/2 -translate-x-1/2 max-w-md">
              <div className="bg-black/70 backdrop-blur-sm rounded-xl px-5 py-3 border border-white/10">
                <div className="flex items-center gap-2">
                  <Sparkles className="h-4 w-4 text-primary shrink-0" />
                  <p className="text-sm text-white">{currentFeedback}</p>
                </div>
              </div>
            </div>
          )}
        </div>

        {/* Bottom Controls */}
        <div className="absolute bottom-0 left-0 right-0 z-10 p-4 bg-gradient-to-t from-black/80 to-transparent">
          <div className="flex items-center justify-center gap-4">
            <Button
              variant="outline"
              size="icon"
              className="rounded-full h-12 w-12 border-white/20 text-white/70 hover:text-white hover:bg-white/10"
              onClick={() => (cameraActive ? stopCamera() : startCamera())}
            >
              {cameraActive ? <VideoOff className="h-5 w-5" /> : <Video className="h-5 w-5" />}
            </Button>

            <Button
              size="icon"
              className={`rounded-full h-16 w-16 ${
                isCoaching
                  ? "bg-red-500 hover:bg-red-600"
                  : "bg-primary hover:bg-primary/90"
              }`}
              onClick={toggleCoaching}
            >
              {isCoaching ? <Square className="h-6 w-6" /> : <Play className="h-6 w-6 ml-0.5" />}
            </Button>

            <Button
              variant="outline"
              size="icon"
              className="rounded-full h-12 w-12 border-white/20 text-white/70 hover:text-white hover:bg-white/10"
              onClick={() => setMicActive(!micActive)}
            >
              {micActive ? <Mic className="h-5 w-5" /> : <MicOff className="h-5 w-5" />}
            </Button>
          </div>
        </div>

        {/* Score HUD (top right during coaching) */}
        {isCoaching && (
          <div className="absolute top-16 right-4 z-10">
            <ScoreRing score={currentScore} size={80} strokeWidth={6} />
          </div>
        )}

        {/* Rep Counter (top left during coaching) */}
        {isCoaching && (
          <div className="absolute top-16 left-4 z-10">
            <div className="bg-black/60 backdrop-blur-sm rounded-xl p-3 border border-white/10 text-center">
              <p className="text-3xl font-bold text-white tabular-nums">{repCount}</p>
              <p className="text-[10px] text-white/50 uppercase tracking-wider">Reps</p>
              <Badge variant="outline" className="mt-1 text-[10px] border-primary/40 text-primary">
                {phase}
              </Badge>
            </div>
          </div>
        )}
      </div>

      {/* Right Panel — Coaching Data */}
      {!isFullscreen && (
        <div className="w-[340px] border-l border-border bg-sidebar overflow-y-auto">
          {/* Skill Picker */}
          <div className="p-4 border-b border-border">
            <button
              className="w-full flex items-center justify-between p-3 rounded-lg bg-secondary/50 hover:bg-secondary transition-colors"
              onClick={() => setShowSkillPicker(!showSkillPicker)}
            >
              <div className="flex items-center gap-2">
                <Activity className="h-4 w-4 text-primary" />
                <span className="text-sm font-medium">{selectedSkill}</span>
              </div>
              <ChevronDown
                className={`h-4 w-4 text-muted-foreground transition-transform ${
                  showSkillPicker ? "rotate-180" : ""
                }`}
              />
            </button>
            {showSkillPicker && (
              <div className="mt-2 max-h-64 overflow-y-auto space-y-3">
                {Object.entries(SKILL_CATEGORIES).map(([category, skills]) => (
                  <div key={category}>
                    <p className="text-[10px] font-medium text-muted-foreground uppercase tracking-wider mb-1.5">{category}</p>
                    <div className="grid grid-cols-2 gap-1">
                      {skills.map((skill) => (
                        <button
                          key={skill}
                          className={`text-xs p-1.5 rounded-md transition-colors text-left ${
                            skill === selectedSkill
                              ? "bg-primary/15 text-primary border border-primary/30"
                              : "bg-secondary/30 text-muted-foreground hover:bg-secondary/50 hover:text-foreground border border-transparent"
                          }`}
                          onClick={() => {
                            setSelectedSkill(skill);
                            setShowSkillPicker(false);
                          }}
                        >
                          {skill}
                        </button>
                      ))}
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>

          {/* Session Stats */}
          <div className="p-4 border-b border-border">
            <p className="text-xs text-muted-foreground mb-3 font-medium">Session</p>
            <div className="grid grid-cols-3 gap-3">
              <div className="text-center">
                <p className="text-lg font-bold tabular-nums">{repCount}</p>
                <p className="text-[10px] text-muted-foreground">Reps</p>
              </div>
              <div className="text-center">
                <p className="text-lg font-bold tabular-nums">{Math.round(avgScore)}</p>
                <p className="text-[10px] text-muted-foreground">Avg Score</p>
              </div>
              <div className="text-center">
                <p className="text-lg font-bold tabular-nums">{Math.round(bestScore)}</p>
                <p className="text-[10px] text-muted-foreground">Best</p>
              </div>
            </div>
          </div>

          {/* Joint Feedback */}
          <div className="p-4 border-b border-border">
            <p className="text-xs text-muted-foreground mb-3 font-medium">Joint Analysis</p>
            <div className="space-y-2">
              {joints.map((joint) => (
                <div
                  key={joint.name}
                  className={`flex items-center gap-3 p-2 rounded-lg border ${getStatusBg(joint.status)}`}
                >
                  <div
                    className={`h-2 w-2 rounded-full ${
                      joint.status === "good"
                        ? "bg-green-500"
                        : joint.status === "warning"
                        ? "bg-yellow-500"
                        : "bg-red-500"
                    }`}
                  />
                  <div className="flex-1 min-w-0">
                    <div className="flex items-center justify-between">
                      <span className="text-xs font-medium">{joint.name}</span>
                      <span className={`text-xs tabular-nums ${getStatusColor(joint.status)}`}>
                        {joint.angle}°
                      </span>
                    </div>
                    <p className="text-[10px] text-muted-foreground truncate">{joint.message}</p>
                  </div>
                </div>
              ))}
            </div>
          </div>

          {/* Movement Quality */}
          <div className="p-4 border-b border-border">
            <p className="text-xs text-muted-foreground mb-3 font-medium">Movement Quality</p>
            <div className="space-y-2.5">
              {[
                { label: "Smoothness", value: quality.smoothness, icon: TrendingUp },
                { label: "Symmetry", value: quality.symmetry, icon: Activity },
                { label: "Range of Motion", value: quality.rangeOfMotion, icon: Gauge },
                { label: "Tempo", value: quality.tempoConsistency, icon: Timer },
              ].map((metric) => (
                <div key={metric.label}>
                  <div className="flex items-center justify-between mb-1">
                    <div className="flex items-center gap-1.5">
                      <metric.icon className="h-3 w-3 text-muted-foreground" />
                      <span className="text-xs">{metric.label}</span>
                    </div>
                    <span className="text-xs text-muted-foreground tabular-nums">
                      {Math.round(metric.value)}%
                    </span>
                  </div>
                  <div className="h-1.5 rounded-full bg-secondary">
                    <div
                      className="h-full rounded-full transition-all duration-500"
                      style={{
                        width: `${metric.value}%`,
                        backgroundColor:
                          metric.value >= 80
                            ? "#06b6d4"
                            : metric.value >= 60
                            ? "#22c55e"
                            : "#f59e0b",
                      }}
                    />
                  </div>
                </div>
              ))}
            </div>
          </div>

          {/* Score History Mini Chart */}
          {scores.length > 1 && (
            <div className="p-4">
              <p className="text-xs text-muted-foreground mb-3 font-medium">Score Trend</p>
              <div className="flex items-end gap-[2px] h-16">
                {scores.slice(-20).map((s, i) => (
                  <div
                    key={i}
                    className="flex-1 rounded-sm transition-all duration-300"
                    style={{
                      height: `${(s / 100) * 100}%`,
                      backgroundColor:
                        s >= 80 ? "#06b6d4" : s >= 60 ? "#22c55e" : s >= 40 ? "#f59e0b" : "#ef4444",
                      opacity: 0.5 + (i / scores.slice(-20).length) * 0.5,
                    }}
                  />
                ))}
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

// video feed
// controls
// coaching state
// joint feedback
// quality metrics
// score trend
// broader skills
// coaching ws
