# Kinetic — Real-Time Physical Movement Intelligence

> **"One camera. One AI. Coaches your squat, detects a fall, guides PT rehab, monitors a hospital room."**

Kinetic is a unified platform for **physical movement intelligence** — AI skill coaching, physical therapy rehab, autonomous space monitoring, and clinical patient safety — all from the same CV + agent + voice stack.

4 modes. 46 MCP tools. 10 ML models. 17,000+ lines. Built solo in 20 hours at TreeHacks 2026.

---

## Demo

🎥 [Loom Demo Video](https://youtu.be/xTkgjQ2uvHc) <!-- TODO: Add Loom link -->
🌐 [Live Frontend](https://frontend-qh6de6ixx-rajashekarvs-projects.vercel.app)

**Infrastructure:**
- 🖥️ **DGX Spark (Edge)**: Real-time pose, fall detection, monitoring — all on-device, zero cloud dependency

> ⚡ **Why both edge and cloud?** DGX Spark's Blackwell GPU doesn't have PyTorch/CUDA wheels for ARM yet — so safety-critical inference (pose, falls) runs on-device at sub-50ms, while one-time heavy generation (0.46B params) offloads to A100. Result is cached locally; after that, coaching is 100% edge.

- 🚀 **Modal A100 (Cloud)**: HY-Motion 1.0-Lite text-to-3D motion generation — **Endpoint**: `POST https://rajashekarvennavelli--aegis-motion-generate-endpoint.modal.run`
- 🤖 **46 MCP Tools**: Full stack exposed via HTTP for external AI agents (Poke)

---

## 4 Modes, One Stack

```
┌─────────────────────────────────────────────────────────┐
│                    KINETIC PLATFORM                      │
├──────────────┬──────────────┬──────────┬────────────────┤
│  🏋️ Coaching  │  🩺 PT Rehab  │ 🎯 Goals │  🏥 Clinical   │
│              │              │          │                │
│ Any skill    │ Knee/shoulder│ Fall     │ Fall detection │
│ Voice coach  │ Safe ROM     │ Posture  │ Bed exit alert │
│ AI expert    │ Rep counting │ Security │ Immobility     │
│ Scoring      │ Voice rehab  │ Focus    │ Wandering      │
├──────────────┴──────────────┴──────────┴────────────────┤
│  CV Pipeline: YOLO + MediaPipe + ByteTrack + Depth      │
│  Agent: Claude SDK (3 sub-agents, 46 MCP tools, hooks)  │
│  Voice: OpenAI Realtime (bidirectional, interruption)   │
│  Edge AI: DGX Spark + Modal A100 (HY-Motion 1.0)       │
│  Alerts: Telegram + Voice + Frontend dashboard          │
└─────────────────────────────────────────────────────────┘
```

---

## 6 Infrastructure Pillars

### 1. 🟢 NVIDIA DGX Spark — Edge AI Inference
- **GB10 Superchip** (Grace ARM CPU + Blackwell GPU)
- **YOLOv8n-pose**: 17-keypoint real-time pose estimation
- Runs on-premise with low latency — no cloud roundtrip for pose
- Endpoints: `POST /predict`, `GET /health`

### 2. 🔵 Modal + NVIDIA A100 — Cloud GPU for Motion Generation
- **HY-Motion 1.0-Lite** (Tencent, SOTA Dec 2025)
- 0.46B parameters, DiT + Flow Matching architecture
- Trained on 3,000+ hours of 3D motion data (pretrain → finetune → RLHF)
- Text prompt → SMPL 22-joint 3D skeleton → MediaPipe 33-point conversion
- **Endpoint**: `POST https://rajashekarvennavelli--aegis-motion-generate-endpoint.modal.run`
- Serverless A100, scales to zero, $530 credits available

### 3. 🟠 Anthropic Claude Agent SDK — AI Orchestration
- **Claude Sonnet 4** as main orchestrator
- **3 Sub-Agents**: Perception (11 tools), Coach (14 tools), Progress (10 tools)
- **44 MCP Tools** via Model Context Protocol
- **3 Agent Hooks**: Safety guard, audit log, session summary
- Handles: coaching decisions, expert generation, form analysis, goal tracking

### 4. 🎙️ OpenAI Realtime API — Voice Coaching
- **GPT-4o Realtime Preview** with `alloy` voice
- **3-Layer Interruption System**:
  - Layer 1: Server-side VAD (50ms speech detection)
  - Layer 2: Response state machine (prevents audio overlap)
  - Layer 3: Single voice source (proactive coaching + reactive Q&A)
- Punchy prompts: max 15 words, no filler
- Browser `speechSynthesis` TTS fallback for offline use

### 5. 👁️ Computer Vision Pipeline
| Model | Purpose | Size | Speed |
|---|---|---|---|
| YOLO11n | Person detection | 5.4 MB | 15 FPS |
| MediaPipe Pose | 33 body landmarks | 5.6 MB | 30 FPS |
| MediaPipe Hands | 21 hand landmarks/hand | ~5 MB | 30 FPS |
| Depth Anything V2 | Monocular depth | ~50 MB | 22 FPS |
| ByteTrack | Multi-person tracking | — | 15 FPS |

### 6. 📊 Triple-Metric Pose Scoring
| Metric | Weight | What it measures |
|---|---|---|
| **Gaussian Angle Scoring** (16 joints) | 50% | Per-joint angular accuracy with tuned σ |
| **Cosine Spatial Similarity** | 30% | Overall pose shape matching |
| **COCO OKS** | 20% | Industry-standard keypoint similarity |

Plus: DTW temporal alignment, phase detection, rep counting, compensation detection.

---

## AI Expert Generation — No Video Required

Kinetic can coach **any skill** without a reference video:

| Tier | Method | Latency | How |
|---|---|---|---|
| 1 | Semantic Alias Lookup | 0ms | 53 aliases → 10 canonical exercises |
| 2 | Claude Semantic Mapping | ~0.5s | Claude maps novel names to known exercises |
| 3 | Claude Angle Generation | ~1-2s | Claude generates biomechanically correct angles |
| 4 | HY-Motion 3D Generation | ~5-15s | Full 3D motion on Modal A100 GPU |

**10 built-in exercises**: squat, pushup, lunge, deadlift, shoulder press, bicep curl, plank, jumping jack, warrior II, tree pose

---

## Complete Model Zoo

| Model | Company | Purpose | Location |
|---|---|---|---|
| **YOLOv8n-pose** | Ultralytics | 17-keypoint pose | DGX Spark |
| **YOLO11n** | Ultralytics | Person detection | Local |
| **MediaPipe Pose Lite** | Google | 33 body landmarks | Local |
| **MediaPipe Hands** | Google | 21 hand landmarks | Local |
| **Depth Anything V2** | ByteDance | Monocular depth | Local (MPS) |
| **ByteTrack** | ByteDance | Multi-person tracking | Local |
| **HY-Motion 1.0-Lite** | Tencent | Text → 3D motion (SOTA) | Modal A100 |
| **Claude Sonnet 4** | Anthropic | Agent orchestration | Cloud API |
| **GPT-4o Realtime** | OpenAI | Voice coaching | Cloud API |
| **Custom 1D CNN** | Built in-house | Pose scoring (14K params) | Local (NumPy) |

---

## Tech Stack

| Layer | Technology |
|---|---|
| **Frontend** | Next.js 14 + React + TailwindCSS + shadcn/ui |
| **Backend** | FastAPI + Python 3.12 (44 routes, 3 WebSockets) |
| **AI Orchestration** | Claude Agent SDK (3 sub-agents, 44 MCP tools, 3 hooks) |
| **Voice AI** | OpenAI Realtime API (GPT-4o) + browser TTS fallback |
| **Edge AI** | NVIDIA DGX Spark (GB10 Superchip, YOLOv8n-pose) |
| **Cloud GPU** | Modal + NVIDIA A100 (HY-Motion 1.0-Lite, 0.46B params) |
| **Computer Vision** | YOLO11n + MediaPipe Pose/Hands + ByteTrack + Depth Anything V2 |
| **Pose Scoring** | Gaussian angles + Cosine spatial + COCO OKS |
| **Protocol** | MCP (Model Context Protocol) — 44 tools |

---

## Quick Start

### Prerequisites
- Python 3.12+
- Node.js 18+
- API Keys: `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`

### Backend
```bash
cd aegis
pip install -r requirements.txt
python run_server.py
# Server starts at http://localhost:8000
```

### Frontend
```bash
cd frontend
npm install
npm run dev
# UI at http://localhost:3000
```

### Test Motion Generation (Modal A100)
```bash
curl -X POST https://rajashekarvennavelli--aegis-motion-generate-endpoint.modal.run \
  -H "Content-Type: application/json" \
  -d '{"prompt": "a person doing a squat", "num_frames": 60}'
```

### DGX Spark Edge Inference
```bash
# On DGX Spark (gx10-eb94)
cd dgx
pip install -r requirements.txt
python inference_server.py
# Pose endpoint at http://<dgx-ip>:8080/predict
```

---

## Project Structure

```
kinetic/
├── aegis/                    # Backend (internal codename)
│   ├── server.py             # FastAPI server (44 routes, 3 WebSockets)
│   ├── spatial_engine.py     # CV pipeline (YOLO + MediaPipe + ByteTrack + Depth)
│   ├── pose_comparison.py    # Triple-metric scoring (Gaussian + Cosine + OKS)
│   ├── ai_expert.py          # AI expert generation (aliases + Claude + HY-Motion)
│   ├── sdk_agent.py          # Claude Agent SDK (3 sub-agents, hooks)
│   ├── mcp_server.py         # 44 MCP tools (fastmcp)
│   ├── openai_voice.py       # OpenAI Realtime voice (3-layer interruption)
│   ├── gemini_bridge.py      # Gemini Live fallback voice
│   ├── skill_graph.py        # Skill DAG + PageRank recommendations
│   ├── data_collector.py     # JSONL session storage
│   └── run_server.py         # Orchestrator
├── dgx/                      # NVIDIA DGX Spark deployment
│   ├── inference_server.py   # Edge AI: YOLOv8n-pose + health endpoints
│   └── modal_motion.py       # Modal A100: HY-Motion 1.0-Lite deployment
├── frontend/                 # Next.js 14 UI
│   └── app/coach/page.tsx    # Main coaching interface
└── README.md
```

---

## Codebase: ~17,000+ lines across 20+ modules

---

## Built at TreeHacks 2026 🌲

Solo-built in 20 hours. 17,000+ lines. 10 models. 4 modes. 46 MCP tools.

**Powered by:**
- **Anthropic** Claude Sonnet 4 + Agent SDK (3 sub-agents, hooks, MCP)
- **OpenAI** GPT-4o Realtime API (bidirectional voice coaching)
- **NVIDIA** DGX Spark GB10 Superchip (edge pose estimation)
- **Modal** A100 GPU Cloud (HY-Motion 1.0-Lite, text-to-3D motion)
- **Google** MediaPipe Pose + Hands (33 body + 21 hand landmarks)
- **Poke** MCP integration (46 tools exposed via HTTP/SSE)
