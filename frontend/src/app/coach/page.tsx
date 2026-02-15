"use client";

import { Suspense, useCallback, useEffect, useRef, useState } from "react";
import { useSearchParams } from "next/navigation";
import { Card, CardContent } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { ScoreRing } from "@/components/score-ring";
import { createCoachingWS, createVideoWS, createAudioWS, startCoaching, stopCoaching, ingestYouTube, createRoom, joinRoom, getRoomLeaderboard } from "@/lib/api";
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
  Youtube,
  Loader2,
  Link,
  Zap,
  Upload,
  FileText,
  MessageSquareText,
  MonitorPlay,
  ChevronRight,
  Volume2,
  VolumeX,
  Users,
  Trophy,
  Copy,
  Check,
} from "lucide-react";

interface JointFeedback {
  name: string;
  status: "good" | "warning" | "bad";
  angle: number;
  target: number;
  message: string;
}

const SKILL_CATEGORIES: Record<string, string[]> = {
  "Physical Therapy": ["Knee Extension", "Shoulder Raise", "Glute Bridge", "Hip Flexion", "Ankle Mobility", "Wall Sit"],
  "Yoga & Mindfulness": ["Warrior Pose", "Tree Pose", "Downward Dog", "Chair Pose", "Sun Salutation"],
  "Tai Chi & Balance": ["Cloud Hands", "Single Whip", "Standing Balance", "Weight Shift"],
  "Sign Language": ["ASL Alphabet", "Common Phrases", "Finger Spelling", "Greetings"],
  "Elderly Mobility": ["Sit-to-Stand", "Heel Raises", "Marching in Place", "Side Step"],
  "Ergonomics": ["Desk Posture", "Proper Lifting", "Stretch Break", "Monitor Height"],
  "Dance": ["Salsa Basic", "Hip-hop Groove", "Ballet Plié", "Bachata Step"],
  "Fitness": ["Squat", "Deadlift", "Push-up", "Lunge", "Plank"],
  "Sports": ["Tennis Serve", "Golf Swing", "Boxing Jab", "Batting Stance"],
  "Music & Performance": ["Guitar Posture", "Piano Hands", "Violin Bow", "Conductor Beat"],
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
  const initialSkill = searchParams.get("skill") || "Warrior Pose";
  const initialMode = searchParams.get("mode") as "video" | "describe" | "document" | null;

  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [cameraActive, setCameraActive] = useState(false);
  const [micActive, setMicActive] = useState(false);
  const [expertMuted, setExpertMuted] = useState(true);
  const [isCoaching, setIsCoaching] = useState(false);
  const [selectedSkill, setSelectedSkill] = useState(initialSkill);
  const [showSkillPicker, setShowSkillPicker] = useState(false);
  const [isFullscreen, setIsFullscreen] = useState(false);

  // Input mode state — pre-select from ?mode= query param
  const [inputMode, setInputMode] = useState<"none" | "video" | "describe" | "document" | "practice">(initialMode || "none");

  // Practice with Friend state
  const [roomCode, setRoomCode] = useState<string | null>(null);
  const [joinCode, setJoinCode] = useState("");
  const [myUserId, setMyUserId] = useState<string | null>(null);
  const [displayName, setDisplayName] = useState("");
  const [friendData, setFriendData] = useState<{ display_name: string; avg_score: number; reps_completed: number; best_score: number; trend: string } | null>(null);
  const [copiedCode, setCopiedCode] = useState(false);
  const [videoSubMode, setVideoSubMode] = useState<"none" | "url" | "upload">(initialMode === "video" ? "url" : "none");

  // YouTube / video state
  const [youtubeUrl, setYoutubeUrl] = useState("");
  const [youtubeVideoId, setYoutubeVideoId] = useState<string | null>(null);
  const [isProcessingVideo, setIsProcessingVideo] = useState(false);
  const [referenceName, setReferenceName] = useState<string | null>(null);
  const [referenceFrames, setReferenceFrames] = useState(0);
  const [coachMode, setCoachMode] = useState<"setup" | "coaching">("setup");

  // Describe-it state
  const [skillDescription, setSkillDescription] = useState("");

  // Document state
  const [uploadedFileName, setUploadedFileName] = useState<string | null>(null);
  const [uploadedVideoName, setUploadedVideoName] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const videoFileInputRef = useRef<HTMLInputElement>(null);

  const extractYouTubeId = (url: string): string | null => {
    const m = url.match(/(?:v=|\/v\/|youtu\.be\/|embed\/|shorts\/)([a-zA-Z0-9_-]{11})/);
    return m ? m[1] : null;
  };

  const handleIngestVideo = async () => {
    const vid = extractYouTubeId(youtubeUrl);
    if (!vid) return;
    setYoutubeVideoId(vid);
    setIsProcessingVideo(true);
    try {
      const name = `${selectedSkill.toLowerCase().replace(/\s+/g, "_")}_${vid}`;
      const result = await ingestYouTube(youtubeUrl, name);
      if (result.status === "success") {
        setReferenceName(result.name || name);
        setReferenceFrames(result.frames || 0);
      }
    } catch {
      // backend may be offline — still show the YouTube embed
    }
    setIsProcessingVideo(false);
  };

  const handleVideoFileUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    setUploadedVideoName(file.name);
    // TODO: upload to backend for skeleton extraction
    setIsProcessingVideo(true);
    setTimeout(() => {
      setReferenceName(`upload_${file.name.replace(/\.[^.]+$/, "")}`);
      setReferenceFrames(0);
      setIsProcessingVideo(false);
    }, 1500);
  };

  const handleDocUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    setUploadedFileName(file.name);
  };

  // ── Practice with Friend handlers ──
  const handleCreateRoom = async () => {
    try {
      const result = await createRoom(selectedSkill, displayName || "Player 1");
      setRoomCode(result.room_code);
      setMyUserId(result.user_id);
    } catch (e) { console.error("Failed to create room:", e); }
  };

  const handleJoinRoom = async () => {
    if (!joinCode.trim()) return;
    try {
      const result = await joinRoom(joinCode.trim().toUpperCase(), displayName || "Player 2");
      if ("error" in result) { alert("Room not found"); return; }
      setRoomCode(result.room_code);
      setMyUserId(result.user_id);
      setSelectedSkill(result.skill);
    } catch (e) { console.error("Failed to join room:", e); }
  };

  const handleCopyCode = () => {
    if (roomCode) {
      navigator.clipboard.writeText(roomCode);
      setCopiedCode(true);
      setTimeout(() => setCopiedCode(false), 2000);
    }
  };

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
  const audioWsRef = useRef<WebSocket | null>(null);
  const audioContextRef = useRef<AudioContext | null>(null);
  const micStreamRef = useRef<MediaStream | null>(null);
  const receivingRealData = useRef(false);

  // Timer
  useEffect(() => {
    if (!isCoaching) return;
    const interval = setInterval(() => {
      setElapsed((e) => e + 1);
    }, 1000);
    return () => clearInterval(interval);
  }, [isCoaching]);

  // Skeleton drawing on canvas overlay
  const SKELETON_CONNECTIONS = [
    [11, 12], // shoulders
    [11, 13], [13, 15], // left arm
    [12, 14], [14, 16], // right arm
    [11, 23], [12, 24], // torso sides
    [23, 24], // hips
    [23, 25], [25, 27], // left leg
    [24, 26], [26, 28], // right leg
    [27, 29], [28, 30], // ankles to heels
    [27, 31], [28, 32], // ankles to toes
  ];

  const drawSkeleton = useCallback((landmarks: number[][]) => {
    const canvas = canvasRef.current;
    const video = videoRef.current;
    if (!canvas || !video) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    // Use the displayed size so coordinates align with the video
    const displayW = video.clientWidth || 640;
    const displayH = video.clientHeight || 480;
    canvas.width = displayW;
    canvas.height = displayH;
    ctx.clearRect(0, 0, displayW, displayH);

    // Draw connections
    ctx.lineWidth = 4;
    for (const [i, j] of SKELETON_CONNECTIONS) {
      if (i >= landmarks.length || j >= landmarks.length) continue;
      const [x1, y1, v1] = landmarks[i];
      const [x2, y2, v2] = landmarks[j];
      if (v1 < 0.3 || v2 < 0.3) continue;
      const sx1 = x1 * displayW, sy1 = y1 * displayH;
      const sx2 = x2 * displayW, sy2 = y2 * displayH;
      ctx.strokeStyle = "rgba(0, 255, 200, 0.8)";
      ctx.beginPath();
      ctx.moveTo(sx1, sy1);
      ctx.lineTo(sx2, sy2);
      ctx.stroke();
    }

    // Draw joints
    for (let i = 0; i < landmarks.length && i < 33; i++) {
      const [x, y, vis] = landmarks[i];
      if (vis < 0.3) continue;
      const sx = x * displayW, sy = y * displayH;
      ctx.fillStyle = vis > 0.7 ? "rgba(0, 255, 150, 1.0)" : "rgba(255, 200, 0, 0.8)";
      ctx.beginPath();
      ctx.arc(sx, sy, 6, 0, Math.PI * 2);
      ctx.fill();
      // White outline for visibility
      ctx.strokeStyle = "rgba(255, 255, 255, 0.5)";
      ctx.lineWidth = 1.5;
      ctx.stroke();
    }
  }, []);

  // Try connecting coaching WebSocket (real data overrides simulation)
  useEffect(() => {
    if (!isCoaching) return;
    receivingRealData.current = false;

    let ws: WebSocket;
    try {
      ws = createCoachingWS();
      wsRef.current = ws;

      ws.onopen = () => console.log("[Kinetic] Coaching WS connected");

      ws.onmessage = (event) => {
        try {
          const msg = JSON.parse(event.data);

          if (msg.type === "coaching" && msg.data) {
            receivingRealData.current = true;
            const d = msg.data;

            // Draw skeleton overlay
            if (msg.landmarks) {
              drawSkeleton(msg.landmarks);
            }

            // Score
            if (d.similarity_score !== undefined) {
              const score = d.similarity_score;
              setCurrentScore(score);
              setScores((prev) => {
                const updated = [...prev, score];
                setAvgScore(updated.reduce((a: number, b: number) => a + b, 0) / updated.length);
                setBestScore(Math.max(...updated));
                return updated;
              });
            }

            // Reps
            if (msg.reps !== undefined) {
              setRepCount(msg.reps);
            }

            // Phase
            if (d.phase) setPhase(d.phase);

            // Feedback from worst joints
            if (d.worst_joints && d.worst_joints.length > 0) {
              const [jointName, deviation] = d.worst_joints[0];
              if (deviation > 15) {
                setCurrentFeedback(`Adjust your ${jointName.replace(/_/g, " ")} — ${deviation.toFixed(0)}° off target`);
              } else if (deviation > 8) {
                setCurrentFeedback(`${jointName.replace(/_/g, " ")} is close, small adjustment needed`);
              } else {
                setCurrentFeedback("Great form! Keep it up!");
              }
            }

            // Joint deviations → joint feedback panel
            if (d.per_joint_deviation) {
              const jointEntries = Object.entries(d.per_joint_deviation) as [string, number][];
              setJoints(jointEntries.slice(0, 6).map(([name, dev]) => ({
                name: name.replace(/_/g, " ").replace(/\b\w/g, (c: string) => c.toUpperCase()),
                status: (dev as number) < 8 ? "good" : (dev as number) < 18 ? "warning" : "bad",
                angle: Math.round(90 + (dev as number)),
                target: 90,
                message: (dev as number) < 8 ? "On target" : (dev as number) < 18 ? `${Math.round(dev as number)}° deviation` : `Fix: ${Math.round(dev as number)}° off`,
              })));
            }

            // Phase score → quality approximation
            if (d.phase_score !== undefined) {
              const ps = d.phase_score;
              setQuality({
                smoothness: Math.min(100, ps + Math.random() * 10 - 5),
                symmetry: Math.min(100, ps + Math.random() * 8 - 4),
                rangeOfMotion: Math.min(100, ps + Math.random() * 12 - 6),
                tempoConsistency: Math.min(100, ps + Math.random() * 6 - 3),
              });
            }
          }

          // Claude Agent SDK feedback (strategic coaching)
          if (msg.type === "agent_feedback" && msg.data) {
            setCurrentFeedback(msg.data);
          }
        } catch {
          // ignore malformed messages
        }
      };

      ws.onerror = () => console.warn("[Kinetic] Coaching WS error");
      ws.onclose = () => { wsRef.current = null; };
    } catch {
      // backend not available
    }

    return () => {
      if (ws && ws.readyState <= 1) ws.close();
      wsRef.current = null;
    };
  }, [isCoaching]);

  // Simulation: always runs, skips tick if receiving real WS data
  useEffect(() => {
    if (!isCoaching) return;
    const interval = setInterval(() => {
      if (receivingRealData.current) return;

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
  }, [isCoaching]);

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
        const dataUrl = canvas.toDataURL("image/jpeg", 0.7);
        const base64 = dataUrl.split(",")[1];
        videoWs.send(JSON.stringify({ type: "frame", data: base64 }));
      };

      const frameInterval = setInterval(sendFrame, 250); // ~4 FPS (enough for pose detection)

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

  // OpenAI Realtime voice — auto-connect when coaching starts for AI voice output
  useEffect(() => {
    if (!isCoaching) {
      if (audioWsRef.current && audioWsRef.current.readyState <= 1) {
        audioWsRef.current.close();
      }
      audioWsRef.current = null;
      return;
    }

    let audioWs: WebSocket;
    let playbackCtx: AudioContext;
    let cancelled = false;

    const setup = async () => {
      try {
        audioWs = createAudioWS();
        audioWsRef.current = audioWs;

        // Playback context for AI voice output (24kHz PCM)
        playbackCtx = new AudioContext({ sampleRate: 24000 });

        audioWs.onmessage = (event) => {
          try {
            const msg = JSON.parse(event.data);
            if (msg.type === "audio" && msg.data) {
              const raw = atob(msg.data);
              const bytes = new Uint8Array(raw.length);
              for (let i = 0; i < raw.length; i++) bytes[i] = raw.charCodeAt(i);
              const pcm16 = new Int16Array(bytes.buffer);
              const float32 = new Float32Array(pcm16.length);
              for (let i = 0; i < pcm16.length; i++) float32[i] = pcm16[i] / 32768;

              const buffer = playbackCtx.createBuffer(1, float32.length, 24000);
              buffer.getChannelData(0).set(float32);
              const source = playbackCtx.createBufferSource();
              source.buffer = buffer;
              source.connect(playbackCtx.destination);
              source.start();
            }
          } catch {
            // ignore malformed
          }
        };

        audioWs.onopen = () => {
          console.log("[Kinetic] Voice connected (OpenAI Realtime)");
          audioWs.send(JSON.stringify({
            type: "text",
            data: `The user is practicing "${selectedSkill}". Give proactive coaching cues based on their movement data. Be encouraging and specific.`,
          }));
        };

        audioWs.onerror = () => console.warn("[Kinetic] Voice WS error");
        audioWs.onclose = () => { audioWsRef.current = null; };

      } catch (err) {
        console.error("[Kinetic] Voice setup failed:", err);
      }
    };

    setup();

    return () => {
      cancelled = true;
      if (audioWs && audioWs.readyState <= 1) audioWs.close();
      audioWsRef.current = null;
    };
  }, [isCoaching, selectedSkill]);

  // Mic capture — when user toggles mic on, send audio to OpenAI for conversation
  useEffect(() => {
    if (!isCoaching || !micActive || !audioWsRef.current) {
      if (micStreamRef.current) {
        micStreamRef.current.getTracks().forEach((t) => t.stop());
        micStreamRef.current = null;
      }
      return;
    }

    let audioCtx: AudioContext;
    let scriptNode: ScriptProcessorNode | null = null;
    let micSource: MediaStreamAudioSourceNode | null = null;
    let cancelled = false;

    const setupMic = async () => {
      try {
        const stream = await navigator.mediaDevices.getUserMedia({
          audio: { sampleRate: 16000, channelCount: 1, echoCancellation: true, noiseSuppression: true },
        });
        if (cancelled) { stream.getTracks().forEach((t) => t.stop()); return; }
        micStreamRef.current = stream;

        audioCtx = new AudioContext();
        audioContextRef.current = audioCtx;
        const nativeSR = audioCtx.sampleRate;
        micSource = audioCtx.createMediaStreamSource(stream);

        scriptNode = audioCtx.createScriptProcessor(4096, 1, 1);
        scriptNode.onaudioprocess = (e) => {
          const ws = audioWsRef.current;
          if (!ws || ws.readyState !== WebSocket.OPEN) return;
          const input = e.inputBuffer.getChannelData(0);
          const TARGET_SR = 16000;
          const ratio = nativeSR / TARGET_SR;
          const outLen = Math.floor(input.length / ratio);
          const pcm16 = new Int16Array(outLen);
          for (let i = 0; i < outLen; i++) {
            const srcIdx = Math.floor(i * ratio);
            const s = Math.max(-1, Math.min(1, input[srcIdx]));
            pcm16[i] = s < 0 ? s * 0x8000 : s * 0x7FFF;
          }
          const bytes = new Uint8Array(pcm16.buffer);
          let binary = "";
          for (let i = 0; i < bytes.length; i++) binary += String.fromCharCode(bytes[i]);
          const b64 = btoa(binary);
          ws.send(JSON.stringify({ type: "audio", data: b64 }));
        };
        micSource.connect(scriptNode);
        scriptNode.connect(audioCtx.destination);

        console.log("[Kinetic] Mic active — you can talk to the coach");
      } catch (err) {
        console.error("[Kinetic] Mic setup failed:", err);
      }
    };

    setupMic();

    return () => {
      cancelled = true;
      if (scriptNode) { scriptNode.disconnect(); scriptNode.onaudioprocess = null; }
      if (micSource) micSource.disconnect();
      if (micStreamRef.current) {
        micStreamRef.current.getTracks().forEach((t) => t.stop());
        micStreamRef.current = null;
      }
    };
  }, [isCoaching, micActive]);

  // Send coaching context to OpenAI Realtime periodically for proactive voice coaching
  useEffect(() => {
    if (!isCoaching || !audioWsRef.current) return;
    const interval = setInterval(() => {
      const ws = audioWsRef.current;
      if (!ws || ws.readyState !== WebSocket.OPEN) return;
      const context = {
        type: "text",
        data: `[COACHING UPDATE] Score: ${Math.round(currentScore)}/100, Reps: ${repCount}, Phase: ${phase}, Avg: ${Math.round(avgScore)}, Best: ${Math.round(bestScore)}. ${currentFeedback ? `Current issue: ${currentFeedback}` : "Form looks good."}`,
      };
      ws.send(JSON.stringify(context));
    }, 5000);
    return () => clearInterval(interval);
  }, [isCoaching, currentScore, repCount, phase, avgScore, bestScore, currentFeedback]);

  // Poll room leaderboard for friend's data in practice mode
  useEffect(() => {
    if (!isCoaching || !roomCode || inputMode !== "practice") return;
    const poll = async () => {
      try {
        const data = await getRoomLeaderboard(roomCode);
        const leaderboard = data.leaderboard as Array<{ user_id: string; display_name: string; avg_score: number; reps_completed: number; best_score: number; trend: string }>;
        const friend = leaderboard.find((p) => p.user_id !== myUserId);
        if (friend) setFriendData(friend);
      } catch { /* ignore */ }
    };
    poll();
    const interval = setInterval(poll, 3000);
    return () => clearInterval(interval);
  }, [isCoaching, roomCode, myUserId, inputMode]);

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
      setCoachMode("setup");
      setPhase("Finished");
      try { await stopCoaching(); } catch { /* backend may be offline */ }
    } else {
      setIsCoaching(true);
      setCoachMode("coaching");
      setRepCount(0);
      setScores([]);
      setElapsed(0);
      setPhase("Preparation");
      setCurrentFeedback("Analyzing your movement...");
      if (!cameraActive) startCamera();
      setMicActive(true); // Auto-enable voice coaching
      try { await startCoaching(selectedSkill, referenceName || undefined); } catch { /* backend may be offline — simulation will kick in */ }
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
    <div className="flex h-screen">
      {/* ══════ SETUP MODE ══════ */}
      {coachMode === "setup" && (
        <div className="flex-1 flex flex-col items-center bg-background p-8 overflow-y-auto">
          <div className="w-full max-w-3xl space-y-8 my-auto">
            {/* Header */}
            <div className="text-center">
              <h1 className="text-3xl font-bold mb-2">AI Skill Coach</h1>
              <p className="text-muted-foreground">Master any physical skill with AI coaching — rehab, yoga, dance, and beyond</p>
            </div>

            {/* ── Step 1: Pick a Skill ── */}
            <div className="space-y-3">
              <div className="flex items-center gap-2">
                <div className="flex items-center justify-center h-6 w-6 rounded-full bg-primary/15 text-primary text-xs font-bold">1</div>
                <h2 className="text-lg font-semibold">Pick a Skill</h2>
              </div>
              <div className="space-y-3 max-h-[35vh] overflow-y-auto pr-1">
                {Object.entries(SKILL_CATEGORIES).map(([category, skills]) => (
                  <div key={category}>
                    <p className="text-xs font-semibold text-muted-foreground uppercase tracking-wider mb-2">{category}</p>
                    <div className="flex flex-wrap gap-2">
                      {skills.map((skill) => (
                        <button
                          key={skill}
                          className={`px-3 py-2 rounded-lg text-sm font-medium transition-all ${
                            skill === selectedSkill
                              ? "bg-primary text-primary-foreground shadow-lg shadow-primary/25 scale-105"
                              : "bg-secondary/60 text-muted-foreground hover:bg-secondary hover:text-foreground"
                          }`}
                          onClick={() => setSelectedSkill(skill)}
                        >
                          {skill}
                        </button>
                      ))}
                    </div>
                  </div>
                ))}
              </div>
            </div>

            {/* ── Step 2: Choose how to learn ── */}
            <div className="space-y-4">
              <div className="flex items-center gap-2">
                <div className="flex items-center justify-center h-6 w-6 rounded-full bg-primary/15 text-primary text-xs font-bold">2</div>
                <h2 className="text-lg font-semibold">How do you want to learn?</h2>
                <span className="text-xs text-muted-foreground">(optional)</span>
              </div>

              {/* Four mode cards */}
              <div className="grid grid-cols-4 gap-3">
                {/* Expert Video Card */}
                <button
                  className={`relative flex flex-col items-center gap-2 p-5 rounded-xl border-2 transition-all text-center ${
                    inputMode === "video"
                      ? "border-red-500/60 bg-red-500/5 shadow-lg shadow-red-500/10"
                      : "border-border hover:border-red-500/30 hover:bg-red-500/5"
                  }`}
                  onClick={() => { setInputMode(inputMode === "video" ? "none" : "video"); setVideoSubMode("none"); }}
                >
                  {inputMode === "video" && <Badge className="absolute -top-2 -right-2 bg-red-500 text-white text-[9px] px-1.5">Selected</Badge>}
                  <MonitorPlay className="h-7 w-7 text-red-500" />
                  <span className="text-sm font-semibold">Expert Video</span>
                  <span className="text-[11px] text-muted-foreground leading-tight">YouTube URL or upload</span>
                </button>

                {/* Just Describe It Card */}
                <button
                  className={`relative flex flex-col items-center gap-2 p-5 rounded-xl border-2 transition-all text-center ${
                    inputMode === "describe"
                      ? "border-primary/60 bg-primary/5 shadow-lg shadow-primary/10"
                      : "border-border hover:border-primary/30 hover:bg-primary/5"
                  }`}
                  onClick={() => setInputMode(inputMode === "describe" ? "none" : "describe")}
                >
                  {inputMode === "describe" && <Badge className="absolute -top-2 -right-2 bg-primary text-white text-[9px] px-1.5">Selected</Badge>}
                  <MessageSquareText className="h-7 w-7 text-primary" />
                  <span className="text-sm font-semibold">Describe It</span>
                  <span className="text-[11px] text-muted-foreground leading-tight">AI coaches from your description</span>
                </button>

                {/* From a Document Card */}
                <button
                  className={`relative flex flex-col items-center gap-2 p-5 rounded-xl border-2 transition-all text-center ${
                    inputMode === "document"
                      ? "border-amber-500/60 bg-amber-500/5 shadow-lg shadow-amber-500/10"
                      : "border-border hover:border-amber-500/30 hover:bg-amber-500/5"
                  }`}
                  onClick={() => setInputMode(inputMode === "document" ? "none" : "document")}
                >
                  {inputMode === "document" && <Badge className="absolute -top-2 -right-2 bg-amber-500 text-white text-[9px] px-1.5">Selected</Badge>}
                  <FileText className="h-7 w-7 text-amber-500" />
                  <span className="text-sm font-semibold">Document</span>
                  <span className="text-[11px] text-muted-foreground leading-tight">PDF or guide with instructions</span>
                </button>

                {/* Practice with Friend Card */}
                <button
                  className={`relative flex flex-col items-center gap-2 p-5 rounded-xl border-2 transition-all text-center ${
                    inputMode === "practice"
                      ? "border-green-500/60 bg-green-500/5 shadow-lg shadow-green-500/10"
                      : "border-border hover:border-green-500/30 hover:bg-green-500/5"
                  }`}
                  onClick={() => setInputMode(inputMode === "practice" ? "none" : "practice")}
                >
                  {inputMode === "practice" && <Badge className="absolute -top-2 -right-2 bg-green-500 text-white text-[9px] px-1.5">Selected</Badge>}
                  <Users className="h-7 w-7 text-green-500" />
                  <span className="text-sm font-semibold">With a Friend</span>
                  <span className="text-[11px] text-muted-foreground leading-tight">Practice together, live scores</span>
                </button>
              </div>

              {/* ── Expanded: Expert Video ── */}
              {inputMode === "video" && (
                <div className="rounded-xl border border-red-500/20 bg-red-500/5 p-5 space-y-4 animate-in fade-in slide-in-from-top-2 duration-200">
                  {/* Sub-mode tabs */}
                  <div className="flex gap-2">
                    <button
                      className={`flex-1 flex items-center justify-center gap-2 py-2.5 rounded-lg text-sm font-medium transition-all ${
                        videoSubMode === "url"
                          ? "bg-red-500 text-white shadow-md"
                          : "bg-secondary/60 text-muted-foreground hover:bg-secondary"
                      }`}
                      onClick={() => setVideoSubMode("url")}
                    >
                      <Youtube className="h-4 w-4" /> YouTube URL
                    </button>
                    <button
                      className={`flex-1 flex items-center justify-center gap-2 py-2.5 rounded-lg text-sm font-medium transition-all ${
                        videoSubMode === "upload"
                          ? "bg-red-500 text-white shadow-md"
                          : "bg-secondary/60 text-muted-foreground hover:bg-secondary"
                      }`}
                      onClick={() => setVideoSubMode("upload")}
                    >
                      <Upload className="h-4 w-4" /> Upload Video
                    </button>
                  </div>

                  {/* URL input */}
                  {videoSubMode === "url" && (
                    <div className="space-y-3">
                      <div className="flex gap-2">
                        <div className="flex-1 relative">
                          <Link className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground" />
                          <input
                            type="text"
                            placeholder="https://youtube.com/watch?v=..."
                            value={youtubeUrl}
                            onChange={(e) => setYoutubeUrl(e.target.value)}
                            className="w-full pl-10 pr-4 py-3 rounded-xl bg-background border border-border text-sm focus:outline-none focus:ring-2 focus:ring-red-500/50 focus:border-red-500/50"
                          />
                        </div>
                        <Button
                          onClick={handleIngestVideo}
                          disabled={!youtubeUrl || isProcessingVideo}
                          className="gap-2 px-5 rounded-xl bg-red-500 hover:bg-red-600"
                        >
                          {isProcessingVideo ? (
                            <><Loader2 className="h-4 w-4 animate-spin" /> Processing...</>
                          ) : (
                            <><ChevronRight className="h-4 w-4" /> Load</>
                          )}
                        </Button>
                      </div>
                      {youtubeVideoId && (
                        <div className="rounded-xl overflow-hidden border border-border bg-black aspect-video max-w-md mx-auto">
                          <iframe
                            src={`https://www.youtube.com/embed/${youtubeVideoId}`}
                            className="w-full h-full"
                            allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
                            allowFullScreen
                          />
                        </div>
                      )}
                    </div>
                  )}

                  {/* File upload */}
                  {videoSubMode === "upload" && (
                    <div className="space-y-3">
                      <input
                        ref={videoFileInputRef}
                        type="file"
                        accept="video/*"
                        className="hidden"
                        onChange={handleVideoFileUpload}
                      />
                      <button
                        className="w-full flex flex-col items-center gap-2 p-6 rounded-xl border-2 border-dashed border-border hover:border-red-500/40 bg-background transition-colors"
                        onClick={() => videoFileInputRef.current?.click()}
                      >
                        <Upload className="h-8 w-8 text-muted-foreground" />
                        <span className="text-sm font-medium">Click to upload a video file</span>
                        <span className="text-xs text-muted-foreground">MP4, MOV, WebM — up to 100MB</span>
                      </button>
                      {uploadedVideoName && (
                        <div className="flex items-center gap-2 p-2 rounded-lg bg-secondary/50">
                          <Video className="h-4 w-4 text-red-500" />
                          <span className="text-sm truncate">{uploadedVideoName}</span>
                          {isProcessingVideo && <Loader2 className="h-4 w-4 animate-spin text-muted-foreground ml-auto" />}
                        </div>
                      )}
                    </div>
                  )}

                  {/* Reference ready indicator */}
                  {referenceName && (
                    <div className="flex items-center gap-2 p-3 rounded-lg bg-green-500/10 border border-green-500/20">
                      <div className="h-2 w-2 rounded-full bg-green-500 animate-pulse" />
                      <span className="text-sm text-green-400">Expert skeleton extracted{referenceFrames > 0 ? ` — ${referenceFrames} poses captured` : ""}</span>
                    </div>
                  )}
                </div>
              )}

              {/* ── Expanded: Just Describe It ── */}
              {inputMode === "describe" && (
                <div className="rounded-xl border border-primary/20 bg-primary/5 p-5 space-y-3 animate-in fade-in slide-in-from-top-2 duration-200">
                  <p className="text-sm text-muted-foreground">Describe the movement or skill you want to practice. Our AI will generate ideal form targets.</p>
                  <textarea
                    placeholder={`e.g. "I want to practice a proper squat with my feet shoulder-width apart, going below parallel..."`}
                    value={skillDescription}
                    onChange={(e) => setSkillDescription(e.target.value)}
                    rows={3}
                    className="w-full px-4 py-3 rounded-xl bg-background border border-border text-sm focus:outline-none focus:ring-2 focus:ring-primary/50 resize-none"
                  />
                  {skillDescription.length > 10 && (
                    <div className="flex items-center gap-2 p-2 rounded-lg bg-primary/10">
                      <Sparkles className="h-4 w-4 text-primary" />
                      <span className="text-xs text-primary">AI will coach you based on this description</span>
                    </div>
                  )}
                </div>
              )}

              {/* ── Expanded: From a Document ── */}
              {inputMode === "document" && (
                <div className="rounded-xl border border-amber-500/20 bg-amber-500/5 p-5 space-y-3 animate-in fade-in slide-in-from-top-2 duration-200">
                  <p className="text-sm text-muted-foreground">Upload a PDF, image, or guide that describes the exercise or movement. AI will extract the key form cues.</p>
                  <input
                    ref={fileInputRef}
                    type="file"
                    accept=".pdf,.png,.jpg,.jpeg,.webp,.txt,.md"
                    className="hidden"
                    onChange={handleDocUpload}
                  />
                  <button
                    className="w-full flex flex-col items-center gap-2 p-6 rounded-xl border-2 border-dashed border-border hover:border-amber-500/40 bg-background transition-colors"
                    onClick={() => fileInputRef.current?.click()}
                  >
                    <FileText className="h-8 w-8 text-muted-foreground" />
                    <span className="text-sm font-medium">Click to upload a document</span>
                    <span className="text-xs text-muted-foreground">PDF, PNG, JPG, TXT, Markdown</span>
                  </button>
                  {uploadedFileName && (
                    <div className="flex items-center gap-2 p-3 rounded-lg bg-amber-500/10 border border-amber-500/20">
                      <FileText className="h-4 w-4 text-amber-500" />
                      <span className="text-sm">{uploadedFileName}</span>
                      <Sparkles className="h-3 w-3 text-amber-500 ml-auto" />
                    </div>
                  )}
                </div>
              )}

              {/* ── Expanded: Practice with Friend ── */}
              {inputMode === "practice" && (
                <div className="rounded-xl border border-green-500/20 bg-green-500/5 p-5 space-y-4 animate-in fade-in slide-in-from-top-2 duration-200">
                  <p className="text-sm text-muted-foreground">Practice together with a friend. Both get live scores from AI — see who nails the form!</p>

                  {/* Name input */}
                  <input
                    className="w-full px-4 py-2.5 rounded-lg bg-background border border-border text-sm placeholder:text-muted-foreground focus:outline-none focus:ring-2 focus:ring-green-500/40"
                    placeholder="Your display name"
                    value={displayName}
                    onChange={(e) => setDisplayName(e.target.value)}
                  />

                  {!roomCode ? (
                    <div className="grid grid-cols-2 gap-3">
                      {/* Create Room */}
                      <button
                        className="flex flex-col items-center gap-2 p-4 rounded-xl border-2 border-dashed border-green-500/30 hover:border-green-500/60 hover:bg-green-500/5 transition-all"
                        onClick={handleCreateRoom}
                      >
                        <Users className="h-6 w-6 text-green-500" />
                        <span className="text-sm font-semibold">Create Room</span>
                        <span className="text-[10px] text-muted-foreground">Get a code to share</span>
                      </button>

                      {/* Join Room */}
                      <div className="flex flex-col items-center gap-2 p-4 rounded-xl border-2 border-dashed border-green-500/30">
                        <Trophy className="h-6 w-6 text-green-500" />
                        <span className="text-sm font-semibold">Join Room</span>
                        <div className="flex gap-2 w-full">
                          <input
                            className="flex-1 px-3 py-1.5 rounded-lg bg-background border border-border text-sm text-center uppercase tracking-widest placeholder:text-muted-foreground focus:outline-none focus:ring-2 focus:ring-green-500/40"
                            placeholder="CODE"
                            maxLength={6}
                            value={joinCode}
                            onChange={(e) => setJoinCode(e.target.value.toUpperCase())}
                            onKeyDown={(e) => e.key === "Enter" && handleJoinRoom()}
                          />
                          <Button size="sm" variant="outline" className="border-green-500/40 text-green-500" onClick={handleJoinRoom}>Join</Button>
                        </div>
                      </div>
                    </div>
                  ) : (
                    /* Room created/joined — show code */
                    <div className="flex items-center gap-3 p-4 rounded-xl bg-green-500/10 border border-green-500/20">
                      <Users className="h-5 w-5 text-green-500" />
                      <div className="flex-1">
                        <p className="text-sm font-semibold">Room: <span className="font-mono tracking-widest text-green-400">{roomCode}</span></p>
                        <p className="text-[11px] text-muted-foreground">Share this code with your friend</p>
                      </div>
                      <Button size="sm" variant="ghost" className="gap-1 text-green-500" onClick={handleCopyCode}>
                        {copiedCode ? <Check className="h-3 w-3" /> : <Copy className="h-3 w-3" />}
                        {copiedCode ? "Copied!" : "Copy"}
                      </Button>
                    </div>
                  )}
                </div>
              )}
            </div>

            {/* ── Step 3: Start ── */}
            <div className="flex flex-col items-center gap-3 pt-2">
              <Button
                size="lg"
                className="gap-2 px-10 py-7 text-lg rounded-xl shadow-lg shadow-primary/20"
                onClick={toggleCoaching}
              >
                <Play className="h-6 w-6" />
                {inputMode === "practice" ? `Practice Together: ${selectedSkill}` : `Start Coaching: ${selectedSkill}`}
              </Button>
              <p className="text-xs text-muted-foreground">
                {referenceName
                  ? "Your pose will be compared to the expert in real-time"
                  : inputMode === "describe" && skillDescription.length > 10
                  ? "AI will coach based on your description"
                  : inputMode === "document" && uploadedFileName
                  ? "AI will extract form cues from your document"
                  : "Camera turns on automatically · AI coaches your form live"}
              </p>
            </div>
          </div>
        </div>
      )}

      {/* ══════ COACHING MODE ══════ */}
      {coachMode === "coaching" && (
        <>
          {/* ── Friend panel — Practice with Friend mode ── */}
          {inputMode === "practice" && (
            <div className="w-[40%] flex flex-col bg-black relative border-r border-white/10">
              <div className="absolute top-3 left-3 right-3 z-10 flex items-center justify-between">
                <Badge variant="outline" className="border-green-500/40 text-green-400 bg-black/60 backdrop-blur-sm">
                  <Users className="h-3 w-3 mr-1" /> {friendData?.display_name || "Waiting for friend..."}
                </Badge>
                {roomCode && (
                  <Badge variant="outline" className="border-white/20 text-white/60 bg-black/60 backdrop-blur-sm font-mono tracking-widest">
                    {roomCode}
                  </Badge>
                )}
              </div>
              <div className="flex-1 flex flex-col items-center justify-center gap-6 p-8">
                {friendData ? (
                  <>
                    <ScoreRing score={Math.round(friendData.avg_score)} size={180} strokeWidth={12} label={friendData.display_name} />
                    <div className="grid grid-cols-3 gap-6 text-center">
                      <div>
                        <p className="text-3xl font-bold tabular-nums text-white">{friendData.reps_completed}</p>
                        <p className="text-xs text-white/50">Reps</p>
                      </div>
                      <div>
                        <p className="text-3xl font-bold tabular-nums text-white">{Math.round(friendData.avg_score)}</p>
                        <p className="text-xs text-white/50">Avg Score</p>
                      </div>
                      <div>
                        <p className="text-3xl font-bold tabular-nums text-white">{Math.round(friendData.best_score)}</p>
                        <p className="text-xs text-white/50">Best</p>
                      </div>
                    </div>
                    <div className="flex items-center gap-2">
                      {friendData.avg_score > avgScore ? (
                        <Badge className="bg-red-500/20 text-red-400 border-red-500/30">Friend is ahead!</Badge>
                      ) : friendData.avg_score < avgScore ? (
                        <Badge className="bg-green-500/20 text-green-400 border-green-500/30">You&apos;re winning!</Badge>
                      ) : (
                        <Badge className="bg-yellow-500/20 text-yellow-400 border-yellow-500/30">Tied!</Badge>
                      )}
                    </div>
                  </>
                ) : (
                  <div className="text-center space-y-3">
                    <Users className="h-12 w-12 text-white/20 mx-auto" />
                    <p className="text-white/40 text-sm">Waiting for your friend to join...</p>
                    <p className="text-white/60 font-mono tracking-widest text-2xl">{roomCode}</p>
                    <p className="text-white/30 text-xs">Share this code</p>
                  </div>
                )}
              </div>
            </div>
          )}

          {/* Expert video — 40% left (only when YouTube, not practice mode) */}
          {youtubeVideoId && inputMode !== "practice" && (
            <div className="w-[40%] flex flex-col bg-black relative border-r border-white/10">
              <div className="absolute top-3 left-3 right-3 z-10 flex items-center justify-between">
                <Badge variant="outline" className="border-red-500/40 text-red-400 bg-black/60 backdrop-blur-sm">
                  <Youtube className="h-3 w-3 mr-1" /> Expert
                </Badge>
                <Button variant="ghost" size="icon" className="h-7 w-7 rounded-full bg-black/60 text-white/70 hover:text-white hover:bg-black/80" onClick={() => setExpertMuted(!expertMuted)}>
                  {expertMuted ? <VolumeX className="h-3 w-3" /> : <Volume2 className="h-3 w-3" />}
                </Button>
              </div>
              <iframe
                src={`https://www.youtube.com/embed/${youtubeVideoId}?autoplay=1&loop=1&mute=${expertMuted ? 1 : 0}`}
                className="w-full h-full"
                allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
                allowFullScreen
              />
            </div>
          )}

          {/* Stats sidebar — only in zero-shot mode (no YouTube, no practice) */}
          {!youtubeVideoId && inputMode !== "practice" && (
            <div className="w-72 border-r border-border bg-sidebar overflow-y-auto">
              <div className="p-4 border-b border-border">
                <div className="flex items-center justify-between mb-3">
                  <p className="text-xs text-muted-foreground font-medium">Session</p>
                  <Badge variant="outline" className="border-green-500/40 text-green-500 animate-pulse text-[10px]">LIVE</Badge>
                </div>
                <div className="flex justify-center mb-4">
                  <ScoreRing score={currentScore} size={100} strokeWidth={6} />
                </div>
                <div className="grid grid-cols-3 gap-3">
                  <div className="text-center"><p className="text-lg font-bold tabular-nums">{repCount}</p><p className="text-[10px] text-muted-foreground">Reps</p></div>
                  <div className="text-center"><p className="text-lg font-bold tabular-nums">{Math.round(avgScore)}</p><p className="text-[10px] text-muted-foreground">Avg</p></div>
                  <div className="text-center"><p className="text-lg font-bold tabular-nums">{Math.round(bestScore)}</p><p className="text-[10px] text-muted-foreground">Best</p></div>
                </div>
              </div>
              <div className="p-4 border-b border-border">
                <p className="text-xs text-muted-foreground mb-3 font-medium">Joint Analysis</p>
                <div className="space-y-2">
                  {joints.map((joint) => (
                    <div key={joint.name} className={`flex items-center gap-3 p-2 rounded-lg border ${getStatusBg(joint.status)}`}>
                      <div className={`h-2 w-2 rounded-full ${joint.status === "good" ? "bg-green-500" : joint.status === "warning" ? "bg-yellow-500" : "bg-red-500"}`} />
                      <div className="flex-1 min-w-0">
                        <div className="flex items-center justify-between">
                          <span className="text-xs font-medium">{joint.name}</span>
                          <span className={`text-xs tabular-nums ${getStatusColor(joint.status)}`}>{joint.angle}°</span>
                        </div>
                        <p className="text-[10px] text-muted-foreground truncate">{joint.message}</p>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
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
                        <div className="flex items-center gap-1.5"><metric.icon className="h-3 w-3 text-muted-foreground" /><span className="text-xs">{metric.label}</span></div>
                        <span className="text-xs text-muted-foreground tabular-nums">{Math.round(metric.value)}%</span>
                      </div>
                      <div className="h-1.5 rounded-full bg-secondary">
                        <div className="h-full rounded-full transition-all duration-500" style={{ width: `${metric.value}%`, backgroundColor: metric.value >= 80 ? "#06b6d4" : metric.value >= 60 ? "#22c55e" : "#f59e0b" }} />
                      </div>
                    </div>
                  ))}
                </div>
              </div>
              {scores.length > 1 && (
                <div className="p-4">
                  <p className="text-xs text-muted-foreground mb-3 font-medium">Score Trend</p>
                  <div className="flex items-end gap-[2px] h-16">
                    {scores.slice(-20).map((s, i) => (
                      <div key={i} className="flex-1 rounded-sm transition-all duration-300" style={{ height: `${(s / 100) * 100}%`, backgroundColor: s >= 80 ? "#06b6d4" : s >= 60 ? "#22c55e" : s >= 40 ? "#f59e0b" : "#ef4444", opacity: 0.5 + (i / scores.slice(-20).length) * 0.5 }} />
                    ))}
                  </div>
                </div>
              )}
            </div>
          )}

          {/* Camera feed — 60% when YouTube, full when zero-shot */}
          <div className="flex-1 flex flex-col relative bg-black">
            {/* Top bar */}
            <div className="absolute top-0 left-0 right-0 z-10 flex items-center justify-between p-3 bg-gradient-to-b from-black/80 to-transparent">
              <div className="flex items-center gap-2">
                <Badge variant="outline" className="border-primary/40 text-primary text-[10px]">{selectedSkill}</Badge>
                <Badge variant="outline" className="border-green-500/40 text-green-500 animate-pulse text-[10px]">LIVE</Badge>
                <span className="text-xs text-white/70 tabular-nums">{formatTime(elapsed)}</span>
              </div>
              <ScoreRing score={currentScore} size={64} strokeWidth={5} />
            </div>

            {/* Camera */}
            <div className="flex-1 flex items-center justify-center relative">
              <video ref={videoRef} autoPlay playsInline muted className="h-full w-full" style={{ transform: "scaleX(-1)" }} />
              <canvas ref={canvasRef} className="absolute inset-0 h-full w-full pointer-events-none" style={{ transform: "scaleX(-1)" }} />

              {/* Stats overlay — top right (when YouTube mode, no sidebar) */}
              {youtubeVideoId && (
                <div className="absolute top-16 right-3 z-10 space-y-2">
                  <div className="bg-black/60 backdrop-blur-sm rounded-xl p-2.5 border border-white/10 text-center">
                    <p className="text-2xl font-bold text-white tabular-nums">{repCount}</p>
                    <p className="text-[9px] text-white/50 uppercase">Reps</p>
                  </div>
                  <div className="bg-black/60 backdrop-blur-sm rounded-xl p-2 border border-white/10 text-center">
                    <Badge variant="outline" className="text-[9px] border-primary/40 text-primary">{phase}</Badge>
                  </div>
                  <div className="bg-black/60 backdrop-blur-sm rounded-xl p-2 border border-white/10">
                    <div className="grid grid-cols-2 gap-x-3 gap-y-1">
                      <span className="text-[9px] text-white/50">Avg</span>
                      <span className="text-[10px] text-white font-medium tabular-nums">{Math.round(avgScore)}</span>
                      <span className="text-[9px] text-white/50">Best</span>
                      <span className="text-[10px] text-white font-medium tabular-nums">{Math.round(bestScore)}</span>
                    </div>
                  </div>
                </div>
              )}

              {/* Coaching Feedback Overlay */}
              {currentFeedback && (
                <div className="absolute bottom-20 left-1/2 -translate-x-1/2 max-w-md">
                  <div className="bg-black/70 backdrop-blur-sm rounded-xl px-5 py-3 border border-white/10">
                    <div className="flex items-center gap-2">
                      <Sparkles className="h-4 w-4 text-primary shrink-0" />
                      <p className="text-sm text-white">{currentFeedback}</p>
                    </div>
                  </div>
                </div>
              )}
            </div>

            {/* Bottom controls */}
            <div className="absolute bottom-0 left-0 right-0 z-10 p-4 bg-gradient-to-t from-black/80 to-transparent">
              <div className="flex items-center justify-center gap-4">
                <Button variant="outline" size="icon" className="rounded-full h-11 w-11 border-white/20 text-white/70 hover:text-white hover:bg-white/10" onClick={() => (cameraActive ? stopCamera() : startCamera())}>
                  {cameraActive ? <VideoOff className="h-4 w-4" /> : <Video className="h-4 w-4" />}
                </Button>
                <Button size="icon" className="rounded-full h-14 w-14 bg-red-500 hover:bg-red-600" onClick={toggleCoaching}>
                  <Square className="h-5 w-5" />
                </Button>
                <Button variant="outline" size="icon" className="rounded-full h-11 w-11 border-white/20 text-white/70 hover:text-white hover:bg-white/10" onClick={() => setMicActive(!micActive)}>
                  {micActive ? <Mic className="h-4 w-4" /> : <MicOff className="h-4 w-4" />}
                </Button>
              </div>
            </div>
          </div>
        </>
      )}
    </div>
  );
}

