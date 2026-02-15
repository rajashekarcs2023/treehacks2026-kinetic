"use client";

import { useState, useEffect } from "react";
import Link from "next/link";
import { Button } from "@/components/ui/button";
import { ArrowLeft } from "lucide-react";

const CompanyBadge = ({ name, color, logo }: { name: string; color: string; logo?: string }) => (
  <span
    style={{
      display: "inline-flex",
      alignItems: "center",
      gap: 4,
      background: color + "18",
      border: `1px solid ${color}44`,
      borderRadius: 4,
      padding: "1px 6px",
      fontSize: 9,
      fontWeight: 700,
      color: color,
      letterSpacing: "0.03em",
      textTransform: "uppercase" as const,
      whiteSpace: "nowrap" as const,
    }}
  >
    {logo && <span style={{ fontSize: 11 }}>{logo}</span>}
    {name}
  </span>
);

const TechCard = ({ title, subtitle, company, companyColor, companyLogo, specs, accent, small, icon }: {
  title: string; subtitle?: string; company?: string; companyColor?: string; companyLogo?: string;
  specs?: string[]; accent?: string; small?: boolean; icon?: string;
}) => (
  <div
    style={{
      background: "#0a0a0f",
      border: `1px solid ${accent || "#333"}55`,
      borderLeft: `3px solid ${accent || "#555"}`,
      borderRadius: 8,
      padding: small ? "8px 10px" : "10px 14px",
      display: "flex",
      flexDirection: "column",
      gap: 4,
      minWidth: small ? 140 : 170,
      flex: small ? "0 0 auto" : "1 1 0",
      transition: "border-color 0.2s, box-shadow 0.2s",
    }}
  >
    <div style={{ display: "flex", alignItems: "center", gap: 6, flexWrap: "wrap" }}>
      {icon && <span style={{ fontSize: 14 }}>{icon}</span>}
      <span style={{ fontSize: small ? 11 : 12.5, fontWeight: 700, color: "#f0f0f5", lineHeight: 1.2 }}>
        {title}
      </span>
    </div>
    {subtitle && (
      <span style={{ fontSize: 10, color: "#9ca3af", lineHeight: 1.3 }}>{subtitle}</span>
    )}
    <div style={{ display: "flex", flexWrap: "wrap", gap: 3, marginTop: 2 }}>
      {company && <CompanyBadge name={company} color={companyColor || accent || "#888"} logo={companyLogo} />}
      {specs &&
        specs.map((s, i) => (
          <span
            key={i}
            style={{
              fontSize: 9,
              background: "#ffffff0a",
              border: "1px solid #ffffff15",
              borderRadius: 3,
              padding: "1px 5px",
              color: "#a0a0b0",
              whiteSpace: "nowrap" as const,
            }}
          >
            {s}
          </span>
        ))}
    </div>
  </div>
);

const LayerHeader = ({ label, color, number }: { label: string; color: string; number: string }) => (
  <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 12, marginTop: 4 }}>
    <div
      style={{
        width: 28, height: 28, borderRadius: "50%",
        background: `${color}22`, border: `2px solid ${color}`,
        display: "flex", alignItems: "center", justifyContent: "center",
        fontSize: 13, fontWeight: 800, color: color, flexShrink: 0,
      }}
    >
      {number}
    </div>
    <div style={{ fontSize: 13, fontWeight: 800, color, letterSpacing: "0.08em", textTransform: "uppercase" as const }}>
      {label}
    </div>
    <div style={{ flex: 1, height: 1, background: `${color}33` }} />
  </div>
);

const FlowArrow = ({ label, color }: { label: string; color?: string }) => (
  <div style={{ display: "flex", flexDirection: "column", alignItems: "center", gap: 2, padding: "6px 0" }}>
    <div style={{ width: 2, height: 18, background: color || "#555" }} />
    {label && (
      <span style={{ fontSize: 8.5, color: color || "#777", fontWeight: 600, textAlign: "center", maxWidth: 120, lineHeight: 1.2 }}>
        {label}
      </span>
    )}
    <div style={{ width: 0, height: 0, borderLeft: "4px solid transparent", borderRight: "4px solid transparent", borderTop: `6px solid ${color || "#555"}` }} />
  </div>
);

const SectionBox = ({ title, children, accent, wide }: { title: string; children: React.ReactNode; accent: string; wide?: boolean }) => (
  <div
    style={{
      border: `1px solid ${accent}33`, borderRadius: 10,
      padding: "10px 12px 12px", background: `${accent}08`,
      flex: wide ? "1 1 100%" : "1 1 0", minWidth: 0,
    }}
  >
    <div style={{ fontSize: 10, fontWeight: 700, color: accent, letterSpacing: "0.06em", textTransform: "uppercase" as const, marginBottom: 8, paddingBottom: 4, borderBottom: `1px solid ${accent}22` }}>
      {title}
    </div>
    <div style={{ display: "flex", flexWrap: "wrap", gap: 8 }}>{children}</div>
  </div>
);

export default function ArchitecturePage() {
  const [loaded, setLoaded] = useState(false);
  useEffect(() => { setTimeout(() => setLoaded(true), 100); }, []);

  return (
    <div style={{ minHeight: "100vh", background: "#06060b", color: "#e8e8f0", fontFamily: "'JetBrains Mono', 'SF Mono', monospace", padding: "24px 16px", overflowX: "auto" }}>
      {/* Back button */}
      <div style={{ maxWidth: 1100, margin: "0 auto 16px", display: "flex", gap: 8 }}>
        <Link href="/"><Button variant="ghost" size="sm" className="gap-2 text-white/60 hover:text-white"><ArrowLeft className="h-4 w-4" /> Dashboard</Button></Link>
      </div>

      {/* Header */}
      <div style={{ textAlign: "center", marginBottom: 28, opacity: loaded ? 1 : 0, transform: loaded ? "translateY(0)" : "translateY(-10px)", transition: "all 0.6s ease" }}>
        <div style={{ display: "flex", alignItems: "center", justifyContent: "center", gap: 12, marginBottom: 6 }}>
          <div style={{ width: 40, height: 40, borderRadius: 10, background: "linear-gradient(135deg, #f97316, #ef4444, #8b5cf6)", display: "flex", alignItems: "center", justifyContent: "center", fontSize: 20, fontWeight: 900, color: "#fff", boxShadow: "0 0 30px #f9731644" }}>K</div>
          <h1 style={{ fontSize: 32, fontWeight: 900, letterSpacing: "-0.02em", background: "linear-gradient(135deg, #fff 0%, #a0a0c0 100%)", WebkitBackgroundClip: "text", WebkitTextFillColor: "transparent", margin: 0 }}>Kinetic.ai</h1>
        </div>
        <p style={{ fontSize: 12, color: "#777", fontWeight: 500, letterSpacing: "0.12em", textTransform: "uppercase" as const, margin: 0 }}>AI Skill Coach — Full System Architecture</p>
        <p style={{ fontSize: 10, color: "#555", marginTop: 4 }}>Edge + Cloud GPU · Multi-Agent AI · Real-Time Computer Vision · Voice Coaching</p>
      </div>

      <div style={{ maxWidth: 1100, margin: "0 auto", opacity: loaded ? 1 : 0, transition: "opacity 0.8s ease 0.2s" }}>
        {/* Layer 1: User */}
        <LayerHeader label="User Interface" color="#94a3b8" number="1" />
        <div style={{ display: "flex", gap: 10, flexWrap: "wrap", marginBottom: 6 }}>
          <TechCard title="Camera" subtitle="30fps video capture" icon="📷" accent="#94a3b8" small />
          <TechCard title="Microphone" subtitle="PCM 16kHz audio input" icon="🎤" accent="#94a3b8" small />
          <TechCard title="Speaker" subtitle="24kHz voice output" icon="🔊" accent="#94a3b8" small />
          <TechCard title="Screen" subtitle="Visual scores + feedback" icon="🖥️" accent="#94a3b8" small />
        </div>
        <FlowArrow label="30fps video + 16kHz audio" color="#94a3b8" />

        {/* Layer 2: Frontend */}
        <LayerHeader label="Frontend" color="#e5e5e5" number="2" />
        <div style={{ display: "flex", gap: 10, flexWrap: "wrap", marginBottom: 6 }}>
          <TechCard title="Next.js 14" subtitle="React + TailwindCSS + shadcn/ui" company="Vercel" companyColor="#fff" accent="#e5e5e5" specs={["SSR", "TypeScript"]} icon="⚡" />
          <TechCard title="WebSocket /ws/video" subtitle="30fps camera stream → backend" accent="#e5e5e5" specs={["base64 frames"]} icon="📡" small />
          <TechCard title="WebSocket /ws/audio" subtitle="Bidirectional voice stream" accent="#e5e5e5" specs={["PCM 16kHz"]} icon="📡" small />
          <TechCard title="WebSocket /ws/coaching" subtitle="Live scores + feedback display" accent="#e5e5e5" specs={["JSON events"]} icon="📡" small />
          <TechCard title="Score Ring + Joint Analysis + Rep Counter" subtitle="Real-time visual coaching UI" accent="#e5e5e5" icon="📊" small />
        </div>
        <FlowArrow label="base64 frames via WebSocket" color="#2563eb" />

        {/* Layer 3: Backend */}
        <LayerHeader label="Backend Intelligence — FastAPI" color="#3b82f6" number="3" />
        <div style={{ border: "1px solid #3b82f633", borderRadius: 12, padding: 14, background: "#3b82f606", marginBottom: 6 }}>
          <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 12 }}>
            <TechCard title="FastAPI Backend" subtitle="Python 3.12 — 44 REST routes, 3 WebSocket endpoints" company="Tiangolo" companyColor="#009688" accent="#3b82f6" specs={["uvicorn", "async"]} icon="🖥️" />
          </div>
          <div style={{ display: "flex", gap: 10, flexWrap: "wrap" }}>
            <SectionBox title="🔍 Computer Vision Pipeline" accent="#22c55e">
              <TechCard title="YOLO11n" subtitle="Person detection · 5.4MB · 15 FPS" company="Ultralytics" companyColor="#a855f7" accent="#a855f7" small />
              <TechCard title="MediaPipe Pose" subtitle="33 body landmarks · 5.6MB · 30 FPS" company="Google" companyColor="#4285F4" accent="#4285F4" small />
              <TechCard title="MediaPipe Hands" subtitle="21 hand landmarks per hand · 30 FPS" company="Google" companyColor="#4285F4" accent="#4285F4" small />
              <TechCard title="ByteTrack" subtitle="Multi-person tracking" company="ByteDance" companyColor="#00f2ea" accent="#00f2ea" small />
            </SectionBox>
            <SectionBox title="🎯 Triple-Metric Pose Scoring" accent="#f97316">
              <TechCard title="Gaussian Kernel Scoring" subtitle="16 joint angles scored individually" accent="#f97316" small />
              <TechCard title="Cosine Spatial Similarity" subtitle="Global body orientation match" accent="#f97316" small />
              <TechCard title="COCO OKS" subtitle="Industry-standard keypoint similarity" accent="#f97316" small />
              <TechCard title="Final Score" subtitle="0.5 Gaussian + 0.3 Cosine + 0.2 OKS" accent="#f97316" specs={["DTW alignment", "phase detect", "rep count"]} small />
            </SectionBox>
          </div>
          <div style={{ display: "flex", gap: 10, flexWrap: "wrap", marginTop: 10 }}>
            <SectionBox title="✨ AI Expert Generation (4-Tier)" accent="#a855f7">
              <TechCard title="Tier 1" subtitle="Semantic alias lookup · 53 aliases · 0ms" accent="#a855f7" small />
              <TechCard title="Tier 2" subtitle="Claude semantic mapping · 0.5s" company="Anthropic" companyColor="#f97316" accent="#f97316" small />
              <TechCard title="Tier 3" subtitle="Claude angle generation · 1-2s" company="Anthropic" companyColor="#f97316" accent="#f97316" small />
              <TechCard title="Tier 4" subtitle="DGX Spark + Modal A100 motion gen · 5-15s" company="NVIDIA" companyColor="#76b900" accent="#76b900" small />
            </SectionBox>
            <SectionBox title="🔁 Coaching Loop (every 10s)" accent="#ef4444">
              <TechCard title="Score Aggregation" subtitle="Gather scores + reps + trend + corrections" accent="#ef4444" small />
              <TechCard title="Prompt Builder" subtitle="Punchy coaching prompt · max 15 words" accent="#ef4444" small />
            </SectionBox>
          </div>
        </div>

        <div style={{ display: "flex", justifyContent: "space-around", flexWrap: "wrap", gap: 8, padding: "4px 0" }}>
          <FlowArrow label="coaching prompt + data" color="#f97316" />
          <FlowArrow label="voice cues" color="#22c55e" />
          <FlowArrow label="generate motion (HTTP)" color="#76b900" />
        </div>

        {/* Layer 4: AI Services */}
        <LayerHeader label="AI Services" color="#f97316" number="4" />
        <div style={{ display: "flex", gap: 12, flexWrap: "wrap", marginBottom: 6 }}>
          <div style={{ flex: "1 1 350px", minWidth: 0 }}>
            <SectionBox title="🧠 Anthropic Claude — AI Brain" accent="#f97316">
              <TechCard title="Claude Sonnet 4" subtitle="Main orchestrator — routes tasks to sub-agents" company="Anthropic" companyColor="#f97316" accent="#f97316" specs={["Agent SDK"]} icon="🧠" />
              <TechCard title="Perception Sub-Agent" subtitle="11 MCP tools — spatial analysis, pose check" accent="#f97316" specs={["MCP"]} icon="👁️" small />
              <TechCard title="Coach Sub-Agent" subtitle="14 MCP tools — form comparison, quality" accent="#f97316" specs={["MCP"]} icon="🏋️" small />
              <TechCard title="Progress Sub-Agent" subtitle="10 MCP tools — goals, memory, plans" accent="#f97316" specs={["MCP"]} icon="📈" small />
              <TechCard title="44 MCP Tools" subtitle="Model Context Protocol — agent ↔ tool" company="Anthropic" companyColor="#f97316" accent="#f97316" icon="🔧" small />
              <TechCard title="3 Agent Hooks" subtitle="Safety guard · Audit log · Session summary" accent="#f97316" icon="🛡️" small />
            </SectionBox>
          </div>
          <div style={{ flex: "1 1 300px", minWidth: 0 }}>
            <SectionBox title="🎙️ OpenAI — Voice AI" accent="#22c55e">
              <TechCard title="GPT-4o Realtime Preview" subtitle="Bidirectional voice coaching · alloy voice" company="OpenAI" companyColor="#22c55e" accent="#22c55e" specs={["PCM 16kHz in", "24kHz out"]} icon="🎙️" />
              <TechCard title="3-Layer Interruption System" subtitle="" accent="#22c55e" small icon="⚡" />
              <TechCard title="Layer 1: Server VAD" subtitle="50ms speech detection" accent="#22c55e" small />
              <TechCard title="Layer 2: State Machine" subtitle="No overlap — clean turn-taking" accent="#22c55e" small />
              <TechCard title="Layer 3: Single Voice Source" subtitle="Proactive + reactive coaching merged" accent="#22c55e" small />
              <TechCard title="TTS Fallback" subtitle="Browser speechSynthesis backup" accent="#22c55e" small icon="🔊" />
            </SectionBox>
          </div>
        </div>

        <FlowArrow label="generate squat motion → edge → cloud GPU" color="#76b900" />

        {/* Layer 5: GPU Compute */}
        <LayerHeader label="GPU Compute" color="#16a34a" number="5" />
        <div style={{ display: "flex", gap: 12, flexWrap: "wrap", marginBottom: 6 }}>
          <div style={{ flex: "1 1 280px", minWidth: 0 }}>
            <SectionBox title="⚡ NVIDIA DGX Spark — Edge AI" accent="#76b900">
              <TechCard title="GB10 Superchip" subtitle="Grace ARM CPU (20 cores) + Blackwell GPU" company="NVIDIA" companyColor="#76b900" accent="#76b900" icon="🟢" />
              <TechCard title="YOLOv8n-pose" subtitle="17-keypoint pose estimation on device" company="Ultralytics" companyColor="#a855f7" accent="#a855f7" small />
              <TechCard title="POST /predict" subtitle="Real-time pose from camera frames" accent="#76b900" small />
              <TechCard title="POST /generate_motion" subtitle="Proxies to Modal A100 cloud GPU" accent="#76b900" small />
              <div style={{ fontSize: 9, color: "#76b900", fontWeight: 600, padding: "4px 0" }}>⚡ Edge inference — low latency, on-premise</div>
            </SectionBox>
          </div>
          <div style={{ display: "flex", alignItems: "center", justifyContent: "center", minWidth: 40 }}>
            <div style={{ display: "flex", flexDirection: "column", alignItems: "center", gap: 4 }}>
              <span style={{ fontSize: 8, color: "#76b900", fontWeight: 700 }}>HTTP</span>
              <div style={{ display: "flex", alignItems: "center", gap: 2 }}>
                <div style={{ width: 30, height: 2, background: "#76b900" }} />
                <div style={{ width: 0, height: 0, borderTop: "5px solid transparent", borderBottom: "5px solid transparent", borderLeft: "8px solid #76b900" }} />
              </div>
              <div style={{ display: "flex", alignItems: "center", gap: 2, transform: "scaleX(-1)" }}>
                <div style={{ width: 30, height: 2, background: "#3b82f6" }} />
                <div style={{ width: 0, height: 0, borderTop: "5px solid transparent", borderBottom: "5px solid transparent", borderLeft: "8px solid #3b82f6" }} />
              </div>
              <span style={{ fontSize: 8, color: "#3b82f6", fontWeight: 700 }}>3D skeleton</span>
            </div>
          </div>
          <div style={{ flex: "1 1 320px", minWidth: 0 }}>
            <SectionBox title="☁️ Modal + NVIDIA A100 — Cloud GPU" accent="#3b82f6">
              <TechCard title="NVIDIA A100" subtitle="40-80GB VRAM · Serverless cloud GPU" company="Modal" companyColor="#3b82f6" accent="#3b82f6" specs={["$530 credits"]} icon="☁️" />
              <TechCard title="HY-Motion 1.0-Lite" subtitle="SOTA text→3D motion · Dec 2025 · DiT + Flow Matching" company="Tencent" companyColor="#3b82f6" accent="#3b82f6" specs={["0.46B params", "1.84GB"]} icon="✨" />
              <TechCard title="Pipeline" subtitle="Text prompt → SMPL 22-joint 3D → MediaPipe 33-point 2D" accent="#3b82f6" small />
              <div style={{ fontSize: 9, color: "#3b82f6", fontWeight: 600, padding: "4px 0" }}>☁️ Serverless — scales to zero, pay per use</div>
            </SectionBox>
          </div>
        </div>

        {/* Legend */}
        <div style={{ marginTop: 24, border: "1px solid #ffffff15", borderRadius: 12, padding: 16, background: "#ffffff06" }}>
          <div style={{ fontSize: 11, fontWeight: 800, color: "#888", letterSpacing: "0.08em", textTransform: "uppercase" as const, marginBottom: 12 }}>
            Technology Stack — Companies & Models
          </div>
          <div style={{ display: "flex", flexWrap: "wrap", gap: 8 }}>
            {[
              { name: "Anthropic", color: "#f97316", detail: "Claude Sonnet 4 · Agent SDK · 44 MCP Tools" },
              { name: "OpenAI", color: "#22c55e", detail: "GPT-4o Realtime · Voice API" },
              { name: "NVIDIA", color: "#76b900", detail: "DGX Spark GB10 · A100 GPU" },
              { name: "Modal", color: "#3b82f6", detail: "Serverless A100 Cloud GPU" },
              { name: "Tencent", color: "#6366f1", detail: "HY-Motion 1.0-Lite (0.46B)" },
              { name: "Google", color: "#4285F4", detail: "MediaPipe Pose + Hands" },
              { name: "Ultralytics", color: "#a855f7", detail: "YOLO11n · YOLOv8n-pose" },
            ].map((c) => (
              <div key={c.name} style={{ display: "flex", alignItems: "center", gap: 6, background: `${c.color}0a`, border: `1px solid ${c.color}33`, borderRadius: 6, padding: "5px 10px" }}>
                <div style={{ width: 8, height: 8, borderRadius: "50%", background: c.color, flexShrink: 0 }} />
                <div>
                  <span style={{ fontSize: 10, fontWeight: 700, color: c.color }}>{c.name}</span>
                  <span style={{ fontSize: 9, color: "#777", marginLeft: 6 }}>{c.detail}</span>
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* 6 Pillars */}
        <div style={{ marginTop: 16, display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(160px, 1fr))", gap: 8 }}>
          {[
            { n: "1", label: "NVIDIA DGX Spark", detail: "Edge AI · GB10 Superchip", color: "#76b900" },
            { n: "2", label: "Modal + A100", detail: "Cloud GPU · HY-Motion 1.0", color: "#3b82f6" },
            { n: "3", label: "Anthropic Claude", detail: "3 Agents · 44 MCP Tools", color: "#f97316" },
            { n: "4", label: "OpenAI Realtime", detail: "GPT-4o Voice · 3-Layer", color: "#22c55e" },
            { n: "5", label: "Google + Ultralytics", detail: "MediaPipe + YOLO CV", color: "#a855f7" },
            { n: "6", label: "Tencent HY-Motion", detail: "0.46B · SOTA Motion Gen", color: "#6366f1" },
          ].map((p) => (
            <div key={p.n} style={{ display: "flex", alignItems: "center", gap: 8, padding: "8px 10px", border: `1px solid ${p.color}33`, borderRadius: 8, background: `${p.color}08` }}>
              <div style={{ width: 22, height: 22, borderRadius: "50%", background: p.color, display: "flex", alignItems: "center", justifyContent: "center", fontSize: 11, fontWeight: 800, color: "#000", flexShrink: 0 }}>{p.n}</div>
              <div>
                <div style={{ fontSize: 10, fontWeight: 700, color: p.color }}>{p.label}</div>
                <div style={{ fontSize: 8.5, color: "#777" }}>{p.detail}</div>
              </div>
            </div>
          ))}
        </div>

        <div style={{ textAlign: "center", marginTop: 20, fontSize: 9, color: "#444" }}>
          Kinetic.ai Architecture v1.0 — AI Skill Coach with Edge + Cloud GPU
        </div>
      </div>
    </div>
  );
}
