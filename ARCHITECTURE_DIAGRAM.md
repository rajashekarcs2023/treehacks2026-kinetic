# Kinetic Architecture Diagram — For Eraser.io

## Paste this into Eraser.io (Diagram from Code mode):

```
// KINETIC — AI Skill Coach with Edge + Cloud GPU Architecture
// 5 major infrastructure blocks, clearly labeled

// ═══════════════════════════════════════
// LAYER 1: USER INTERFACE
// ═══════════════════════════════════════

User [icon: user] {
  Camera [icon: camera]
  Microphone [icon: mic]
  Speaker [icon: speaker]
  Screen [icon: monitor]
}

Next.js 14 Frontend (Vercel) [icon: nextjs, color: black] {
  React + TailwindCSS + shadcn/ui [icon: react]
  WebSocket /ws/video - 30fps camera stream [icon: wifi]
  WebSocket /ws/audio - bidirectional voice [icon: wifi]
  WebSocket /ws/coaching - live scores + feedback [icon: wifi]
  Score Ring + Joint Analysis + Rep Counter [icon: bar-chart]
}

// ═══════════════════════════════════════
// LAYER 2: BACKEND INTELLIGENCE (FastAPI)
// ═══════════════════════════════════════

FastAPI Backend - Python 3.12 (44 routes, 3 WebSockets) [icon: server, color: blue] {

  Computer Vision Pipeline [icon: eye, color: green] {
    YOLO11n - Person Detection (Ultralytics, 5.4MB, 15 FPS) [icon: box]
    Google MediaPipe Pose - 33 Body Landmarks (5.6MB, 30 FPS) [icon: activity]
    Google MediaPipe Hands - 21 Hand Landmarks per hand (30 FPS) [icon: hand]
    ByteTrack - Multi-Person Tracking (ByteDance) [icon: users]
    Depth Anything V2 - Monocular Depth (ByteDance, 22 FPS) [icon: layers]
  }

  Triple-Metric Pose Scoring Engine [icon: activity, color: orange] {
    16 Joint Angles with Gaussian Kernel Scoring [icon: trending-up]
    Cosine Spatial Similarity [icon: minimize]
    COCO OKS - Industry Standard Keypoint Similarity [icon: award]
    Final = 0.5 Gaussian + 0.3 Cosine + 0.2 OKS [icon: target]
    DTW Temporal Alignment + Phase Detection [icon: clock]
    Rep Counting + Compensation Detection [icon: repeat]
  }

  AI Expert Generation Pipeline [icon: sparkles, color: purple] {
    Tier 1 - Semantic Alias Lookup (53 aliases, 0ms) [icon: search]
    Tier 2 - Claude Semantic Mapping (0.5s) [icon: brain]
    Tier 3 - Claude Angle Generation (1-2s) [icon: cpu]
    Tier 4 - NVIDIA DGX Spark + Modal A100 Motion Gen (5-15s) [icon: zap]
  }

  Coaching Intelligence Loop - every 10s [icon: repeat, color: red] {
    Gather scores + reps + trend + corrections [icon: bar-chart]
    Build punchy coaching prompt - max 15 words [icon: message-circle]
  }

  Local 1D CNN Scorer (14K params, NumPy, 0.15ms) [icon: cpu, color: yellow]
  Skill Graph DAG + PageRank Recommendations [icon: git-branch]
  JSONL Session Storage + Reference Skeleton Store [icon: database]
}

// ═══════════════════════════════════════
// LAYER 3: ANTHROPIC CLAUDE (AI Brain)
// ═══════════════════════════════════════

Anthropic Claude Agent SDK [icon: brain, color: orange] {
  Main Orchestrator - Claude Sonnet 4 [icon: brain]
  Perception Sub-Agent - 11 MCP tools (spatial, pose, hands) [icon: eye]
  Coach Sub-Agent - 14 MCP tools (comparison, quality, form) [icon: dumbbell]
  Progress Sub-Agent - 10 MCP tools (goals, memory, training) [icon: trending-up]
  44 MCP Tools via Model Context Protocol (Anthropic) [icon: tool]
  3 Agent Hooks - Safety Guard + Audit Log + Session Summary [icon: shield]
}

// ═══════════════════════════════════════
// LAYER 4: OPENAI (Voice AI)
// ═══════════════════════════════════════

OpenAI Realtime Voice API [icon: headphones, color: green] {
  GPT-4o Realtime Preview - alloy voice [icon: mic]
  PCM 16kHz input + PCM 24kHz output [icon: volume-2]
  3-Layer Interruption System [icon: zap] {
    Layer 1 - Server-Side VAD (50ms speech detection) [icon: radio]
    Layer 2 - Response State Machine (no overlap) [icon: toggle-right]
    Layer 3 - Single Voice Source (proactive + reactive) [icon: volume-2]
  }
  Browser speechSynthesis TTS Fallback [icon: speaker]
}

// ═══════════════════════════════════════
// LAYER 5: NVIDIA DGX SPARK (Edge AI)
// ═══════════════════════════════════════

NVIDIA DGX Spark - GB10 Superchip [icon: cpu, color: green] {
  Grace ARM CPU (20 cores) + Blackwell GPU [icon: cpu]
  YOLOv8n-pose - 17 Keypoint Pose Estimation [icon: activity]
  POST /predict - Real-time pose from camera frames [icon: camera]
  POST /generate_motion - Proxies to Modal A100 GPU [icon: arrow-right]
  GET /health - System + model status [icon: heart]
  Edge inference - low latency, on-premise [icon: zap]
}

// ═══════════════════════════════════════
// LAYER 6: MODAL A100 (Cloud GPU)
// ═══════════════════════════════════════

Modal - NVIDIA A100 GPU (40-80GB VRAM) [icon: cloud, color: blue] {
  HY-Motion 1.0-Lite (Tencent, SOTA Dec 2025) [icon: sparkles]
  0.46 Billion Parameters - DiT + Flow Matching [icon: layers]
  Trained on 3000+ hours of 3D motion data [icon: database]
  3-Stage Training - Pretrain + Finetune + RLHF [icon: git-branch]
  Text Prompt → SMPL 22-Joint 3D Skeleton [icon: activity]
  SMPL → MediaPipe 33-Point 2D Conversion [icon: maximize]
  1.84 GB Model Checkpoint [icon: hard-drive]
  Serverless - scales to zero, pay per use ($530 credits) [icon: dollar-sign]
}

// ═══════════════════════════════════════
// DATA FLOW CONNECTIONS
// ═══════════════════════════════════════

// User → Frontend
User.Camera --> Next.js 14 Frontend (Vercel).WebSocket /ws/video - 30fps camera stream: 30fps video frames
User.Microphone --> Next.js 14 Frontend (Vercel).WebSocket /ws/audio - bidirectional voice: PCM 16kHz audio

// Frontend → Backend CV
Next.js 14 Frontend (Vercel).WebSocket /ws/video - 30fps camera stream --> FastAPI Backend - Python 3.12 (44 routes, 3 WebSockets).Computer Vision Pipeline: base64 frames

// CV → Scoring → Coaching Loop
FastAPI Backend - Python 3.12 (44 routes, 3 WebSockets).Computer Vision Pipeline --> FastAPI Backend - Python 3.12 (44 routes, 3 WebSockets).Triple-Metric Pose Scoring Engine: 33 landmarks + depth map
FastAPI Backend - Python 3.12 (44 routes, 3 WebSockets).Triple-Metric Pose Scoring Engine --> FastAPI Backend - Python 3.12 (44 routes, 3 WebSockets).Coaching Intelligence Loop - every 10s: score + reps + corrections

// Coaching Loop → Claude Agent SDK
FastAPI Backend - Python 3.12 (44 routes, 3 WebSockets).Coaching Intelligence Loop - every 10s --> Anthropic Claude Agent SDK: coaching prompt with inline data

// Claude delegates to sub-agents
Anthropic Claude Agent SDK.Main Orchestrator - Claude Sonnet 4 --> Anthropic Claude Agent SDK.Coach Sub-Agent - 14 MCP tools (comparison, quality, form): form analysis
Anthropic Claude Agent SDK.Main Orchestrator - Claude Sonnet 4 --> Anthropic Claude Agent SDK.Perception Sub-Agent - 11 MCP tools (spatial, pose, hands): spatial check
Anthropic Claude Agent SDK.Main Orchestrator - Claude Sonnet 4 --> Anthropic Claude Agent SDK.Progress Sub-Agent - 10 MCP tools (goals, memory, training): goal tracking

// Claude → Voice + UI
Anthropic Claude Agent SDK --> OpenAI Realtime Voice API: speak() coaching cue
Anthropic Claude Agent SDK --> Next.js 14 Frontend (Vercel).WebSocket /ws/coaching - live scores + feedback: text + scores

// Voice → User
OpenAI Realtime Voice API --> User.Speaker: natural voice coaching (24kHz)
Next.js 14 Frontend (Vercel).Score Ring + Joint Analysis + Rep Counter --> User.Screen: visual feedback

// Frontend audio → OpenAI
Next.js 14 Frontend (Vercel).WebSocket /ws/audio - bidirectional voice --> OpenAI Realtime Voice API: bidirectional PCM audio

// AI Expert → DGX Spark → Modal (THE KEY GPU PIPELINE)
FastAPI Backend - Python 3.12 (44 routes, 3 WebSockets).AI Expert Generation Pipeline --> NVIDIA DGX Spark - GB10 Superchip: "generate squat motion" (HTTP POST)
NVIDIA DGX Spark - GB10 Superchip.POST /generate_motion - Proxies to Modal A100 GPU --> Modal - NVIDIA A100 GPU (40-80GB VRAM): proxy to cloud GPU (HTTP)
Modal - NVIDIA A100 GPU (40-80GB VRAM) --> NVIDIA DGX Spark - GB10 Superchip: 3D skeleton (SMPL → MediaPipe 33pt)
NVIDIA DGX Spark - GB10 Superchip --> FastAPI Backend - Python 3.12 (44 routes, 3 WebSockets).AI Expert Generation Pipeline: expert reference skeleton

// AI Expert also uses Claude for Tier 2 + 3
FastAPI Backend - Python 3.12 (44 routes, 3 WebSockets).AI Expert Generation Pipeline --> Anthropic Claude Agent SDK: semantic mapping + angle generation
```

---

## Key Technologies to show with company logos:

| Technology | Company | Where in Pipeline | Logo Color |
|---|---|---|---|
| **Claude Sonnet 4 + Agent SDK** | Anthropic | AI brain — orchestration, coaching, expert gen | Orange |
| **GPT-4o Realtime** | OpenAI | Voice AI — bidirectional coaching | Green |
| **DGX Spark (GB10 Superchip)** | NVIDIA | Edge AI — pose estimation + motion proxy | Green |
| **A100 GPU (Modal)** | NVIDIA + Modal | Cloud GPU — HY-Motion 1.0 inference | Blue |
| **HY-Motion 1.0-Lite (0.46B)** | Tencent | SOTA text-to-3D motion generation | Blue |
| **YOLO11n / YOLOv8n-pose** | Ultralytics | Person detection + pose estimation | Purple |
| **MediaPipe Pose + Hands** | Google | 33 body + 21 hand landmarks | Blue/Red/Yellow/Green |
| **Depth Anything V2** | ByteDance | Monocular depth estimation | Blue |
| **ByteTrack** | ByteDance | Multi-person tracking | Blue |
| **MCP Protocol** | Anthropic | 44 tools for agent-tool communication | Orange |
| **Next.js 14 + React** | Vercel | Frontend UI | Black |
| **TailwindCSS + shadcn/ui** | Tailwind Labs | Styling + components | Blue |
| **FastAPI** | Tiangolo | Backend server (44 routes) | Green |

---

## 6 Infrastructure Pillars (for slide/poster):

1. **NVIDIA DGX Spark** — Edge AI inference (GB10 Superchip, YOLOv8n-pose, 17 keypoints)
2. **Modal + NVIDIA A100** — Cloud GPU for HY-Motion 1.0-Lite (0.46B param SOTA motion gen)
3. **Anthropic Claude Agent SDK** — Multi-agent orchestration (3 sub-agents, 44 MCP tools, 3 hooks)
4. **OpenAI Realtime API** — Bidirectional voice coaching (GPT-4o, 3-layer interruption)
5. **Google MediaPipe + Ultralytics YOLO** — Real-time computer vision (33 body + 21 hand landmarks)
6. **Tencent HY-Motion 1.0** — State-of-the-art text-to-3D motion (DiT + Flow Matching, Dec 2025)
