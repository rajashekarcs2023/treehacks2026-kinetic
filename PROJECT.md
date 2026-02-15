# Kinetic — Real-Time AI Skill Coach with Expert Motion Transfer

> **Learn any physical skill from any expert — in real-time, through voice.**

---

## The Problem

Learning physical skills — yoga, physical therapy, weightlifting, dance, sports — is broken:

- **Personal trainers cost $60-150/hour** and aren't available 24/7
- **YouTube tutorials can't see you** — you watch, try, and hope you're doing it right
- **Physical therapy compliance is 30-50%** — patients go home, do exercises wrong, get re-injured
- **Beginners don't know what "wrong" feels like** — they need real-time correction, not post-hoc review

The core problem: **there's no way to get expert-quality, real-time, personalized movement coaching at scale.**

---

## What Kinetic Does

Kinetic is an AI system that:

1. **Watches an expert perform a movement** (live, video, or YouTube)
2. **Extracts their skeleton and joint angles** as a reference
3. **Watches YOU perform the same movement** via camera
4. **Coaches you in real-time through voice** — telling you exactly what to fix, in plain language
5. **Learns and improves** — the more it coaches, the better its scoring model gets

It's like having a world-class coach who can see your body, understands biomechanics, and never gets tired.

---

## How It Works — Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        KINETIC                               │
│                                                              │
│  ┌──────────┐    ┌──────────────┐    ┌──────────────────┐   │
│  │   EYES   │    │    BRAIN     │    │      VOICE       │   │
│  │          │───▶│              │───▶│                  │   │
│  │ YOLO11n  │    │ Claude Agent │    │ OpenAI Realtime  │   │
│  │ MediaPipe│    │ 46 MCP Tools │    │ GPT-4o Voice     │   │
│  │  Pose+   │    │ 3 Subagents  │    │ Bidirectional    │   │
│  │  Hands   │    │ Hooks+Skills │    │ + Interruption   │   │
│  │ ByteTrack│    │              │    │                  │   │
│  │ Depth V2 │    │              │    │                  │   │
│  └──────────┘    └──────────────┘    └──────────────────┘   │
│        │                │                                    │
│        ▼                ▼                                    │
│  ┌──────────┐    ┌──────────────┐    ┌──────────────────┐   │
│  │ LEARNING │    │  COACHING    │    │   PROGRESSION    │   │
│  │          │    │              │    │                  │   │
│  │ PyTorch  │    │ DTW Align    │    │ Skill DAG        │   │
│  │ Scorer   │    │ Phase Detect │    │ PageRank Recs    │   │
│  │ Activity │    │ Rep Counter  │    │ Proficiency      │   │
│  │ CNN      │    │ Quality Score│    │ Tracking         │   │
│  └──────────┘    └──────────────┘    └──────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

### The Eyes — Computer Vision Pipeline
- **YOLO11n** — person detection at 15 FPS
- **MediaPipe PoseLandmarker** — 33 skeletal landmarks, async at 30 FPS
- **MediaPipe HandLandmarker** — 21 landmarks per hand, finger-level tracking for sign language
- **ByteTrack** — multi-person tracking with persistent IDs
- **Depth Anything V2** — monocular depth estimation on MPS GPU
- **Activity Classifier** — ML temporal CNN (11 activity classes) + heuristic fallback
- **NVIDIA DGX Spark** — RTMPose-WholeBody (133 keypoints: body+hands+face) via GB10 GPU
- **Camera Coverage Detection** — checks landmark visibility/occlusion, warns user to reposition

### The Brain — Multi-Agent System via Claude Agent SDK
This is NOT a single agent with all tools. It's a **multi-agent architecture**:

```
┌─────────────────────────────────────────────┐
│           MAIN ORCHESTRATOR AGENT           │
│  Decides what to do, delegates via "Task"   │
│  Has access to all 43 tools + Task tool     │
├─────────┬──────────────┬────────────────────┤
│  TASK   │     TASK     │       TASK         │
│   ↓     │      ↓       │        ↓           │
│ ┌─────┐ │ ┌──────────┐ │ ┌──────────────┐  │
│ │PERC.│ │ │  COACH   │ │ │  PROGRESS    │  │
│ │AGENT│ │ │  AGENT   │ │ │  AGENT       │  │
│ │     │ │ │          │ │ │              │  │
│ │10   │ │ │14 tools  │ │ │10 tools      │  │
│ │tools│ │ │coaching  │ │ │goals, memory │  │
│ │scene│ │ │compare   │ │ │graphs, train │  │
│ └─────┘ │ └──────────┘ │ └──────────────┘  │
└─────────┴──────────────┴────────────────────┘
```

- **Orchestrator** — decides intent, delegates to specialized subagents
- **Perception Agent** (10 tools) — reads scene, pose, hands, activity, objects, landmark visibility
- **Coach Agent** (14 tools) — pose comparison, corrections, rep counting, coaching sessions
- **Progress Agent** (10 tools) — skill graphs, goal management, memory, model training
- **46 total MCP tools** across 13 categories
- **Hooks** — PreToolUse (safety guardrails: blocks dangerous tool calls), PostToolUse (audit log: every tool call recorded), Stop (session summary generation)
- **Skills** — markdown-defined domain expertise (`.claude/skills/coaching.md`, `perception.md`, `progress.md`)

**Why multi-agent?** Each subagent has **restricted tool access** — the Coach Agent can't modify goals, the Perception Agent can't alter training data. This is a real safety architecture, not just for show.

### The Voice — OpenAI Realtime API (GPT-4o)
- **Bidirectional audio** — user speaks naturally, AI responds with natural voice
- **3-Layer Interruption System** — solves the "proactive vs reactive" voice challenge:
  - Layer 1: Server-side VAD detects user speech → pauses coaching, clears audio queue
  - Layer 2: Response state machine prevents overlapping speech
  - Layer 3: Single voice source architecture (only `speak()` triggers output)
- **Emotional Tone Adaptation** — detects score trends and adapts coaching style:
  - Scores dropping → gentle, supportive tone
  - New personal best → big celebration
  - User sounds frustrated → empathetic approach
- **Proactive Check-ins** — 3+ declining reps → fatigue check, 10-rep milestones, coverage warnings
- **Multi-Language Support** — "Coach me in Spanish" → switches language seamlessly
- **Session Memory** — loads past session data per-skill: "Last time you averaged 78, knee alignment was the issue"
- **WebSocket streaming** — low-latency audio via /ws/audio
- Gemini Live kept as fallback if OpenAI unavailable

### The Learning — Self-Improving Models
- **SkillScorer (PyTorch)** — 1D CNN that scores movement quality (0-100)
  - Trains on real coaching data collected during sessions
  - GPU-accelerated on MPS/CUDA/DGX
  - Falls back to NumPy evolutionary strategy on CPU
- **ActivityNet (PyTorch)** — temporal 1D CNN for activity classification
  - 11 activity classes, 36K parameters
  - Bootstraps from synthetic biomechanical data
  - 100% validation accuracy on synthetic, ~90%+ on real data
- **HybridScorer** — blends local model (60%) + Claude analysis (40%)
  - Auto-collects training data from every coaching rep
  - Auto-retrains after 20 new samples
  - Bootstrap from zero with synthetic data generation

### The Coaching Engine — Pose Comparison
- **Skeleton normalization** — camera/distance-invariant comparison
- **10 joint angles** — computed from 33 landmarks (elbows, shoulders, hips, knees, ankles)
- **DTW alignment** — temporal matching between expert and user sequences
- **Phase detection** — preparation → execution → peak → recovery
- **Rep counting** — automatic from joint angle oscillation patterns
- **Movement quality metrics** — smoothness, symmetry, range of motion, tempo consistency
- **Compensation detection** — spots when you cheat (e.g., leaning forward during squats)

---

## Technical Depth — What Makes This Real

### Not a Wrapper Around an LLM

Kinetic is **not** "send a photo to GPT-4 and ask it to coach you." Here's what's actually happening:

| Component | What It Really Does |
|-----------|-------------------|
| Pose Comparison | Real biomechanics: joint angles in degrees, skeleton normalization for camera invariance, Dynamic Time Warping for temporal alignment |
| Movement Quality | Computes smoothness (jerk minimization), symmetry (L/R deviation), range of motion, tempo consistency — real signal processing |
| Compensation Detection | Cross-joint correlation analysis to detect when you're cheating (e.g., hip shift during unilateral exercises) |
| Data Augmentation | Time warping, noise injection, L/R mirroring, score jitter — proper ML training techniques |
| Skill Progression | Directed acyclic graph with prerequisites, proficiency decay, PageRank-inspired recommendations |
| Training Pipeline | Validation (NaN/Inf/range checks), augmentation (4 types), synthetic data generation, proper train/val/test splits |

### The Self-Improving Loop

```
User does a rep
    → CV pipeline extracts skeleton
    → Joint angles computed (10 angles × T frames)
    → Compared to expert reference via DTW
    → Score + corrections generated
    → Claude provides deep coaching feedback via voice
    → Rep data auto-saved to training pipeline
    → After 20 reps, model auto-retrains
    → Next session: faster, more accurate scoring
```

Every coaching session makes the system better. This is real online learning.

### Zero-Shot Coaching

Don't have an expert reference? Just describe the skill:
> "Coach me on proper deadlift form"

Claude uses its biomechanics knowledge + the 10 joint angles from your live pose to coach you without ever seeing an expert do it.

---

## Use Cases

### 1. Physical Therapy Rehabilitation
- Patient goes home after knee surgery
- Sets up Kinetic with PT-prescribed exercises
- System coaches them through each rep with correct form
- Progress tracked on skill graph, reported to PT

### 2. Fitness Training
- User wants to learn proper squat form
- Records an expert (or uses built-in reference)
- Kinetic compares every rep to the expert
- Voice: "Your left knee is caving in 12 degrees — push it out"

### 3. Dance / Choreography
- Watch a dance tutorial, Kinetic extracts the skeleton
- Practice with real-time comparison to the choreographer
- Phase-by-phase breakdown: "You're behind on the second count"

### 4. Sports Coaching
- Tennis serve, golf swing, boxing technique
- Frame-by-frame comparison to professional athletes
- Identifies specific joints that deviate from expert form

### 5. Yoga / Meditation
- Precise pose alignment coaching
- Holds timed with breathing guidance
- Progressive difficulty through skill graph

### 6. Elderly Care / Fall Prevention
- Activity monitoring with fallen/lying_down detection (safety-critical, priority-boosted)
- Exercise compliance tracking
- Telegram alerts to caregivers

### 7. Sign Language Learning
- 21 landmarks per hand via MediaPipe HandLandmarker
- Finger-level tracking for ASL alphabet and phrases
- DGX Spark provides 42 hand keypoints at higher accuracy

### 8. Practice with a Friend
- Create a room → get a 6-character code → share
- Both practice simultaneously with live scores
- Real-time leaderboard updates every 3 seconds
- AI compares both players and provides competitive coaching

---

## Demo Script — What We Show

### Demo Flow (2-3 minutes)

**Scene 1: "Meet Kinetic" (30 sec)**
- Camera on, person standing
- System recognizes: "I see 1 person, standing, good posture"
- Voice: "Hi! I'm Kinetic, your AI skill coach. What would you like to work on today?"
- User: "Help me with my squat form"

**Scene 2: "Expert Reference" (20 sec)**
- Load a pre-recorded expert squat reference
- Show the skeleton overlay of the expert
- "I've loaded the expert squat reference. I'll compare your form to this. Let's start."

**Scene 3: "Real-Time Coaching" (60 sec)**
- User performs squats
- Live scoring: score bar updates per-frame
- Rep counter increments automatically
- Voice corrections: "Knees tracking over toes — good. Now push your hips back more. Left knee is 15 degrees too narrow."
- Joint-by-joint color coding: green (good), yellow (warning), red (fix this)

**Scene 4: "Self-Improving" (20 sec)**
- Show training data auto-collecting
- "I've collected 5 reps of training data. The model is getting smarter."
- Show model accuracy improving

**Scene 5: "Progression" (20 sec)**
- Show skill graph: squat unlocked → can now progress to lunge, Bulgarian split squat
- "Based on your squat score of 82, I recommend working on lunges next."
- Show proficiency tracking over time

**Scene 6: "Zero-Shot" (20 sec)**
- User: "Can you coach me on a tennis serve?"
- No expert reference needed
- System uses biomechanics knowledge to coach from description
- "Extend your arm fully at the top — your elbow is at 140°, should be 170°+"

### Technical Skills We Demonstrate

1. **Real-time Computer Vision** — YOLO + Pose + Hands + Tracking at 15 FPS, zero frame storage (privacy-first)
2. **Claude Agent SDK** — 46 MCP tools, 3 subagents, hooks, skills, continuous conversation
3. **Expert Motion Transfer** — skeleton extraction → normalization → DTW comparison → coaching
4. **Self-Improving ML** — models that learn from every coaching session (1D CNN + hybrid scoring)
5. **Multimodal Interaction** — camera input + voice output + visual overlay
6. **Biomechanical Analysis** — real joint angles, compensation detection, movement quality metrics
7. **OpenAI Realtime Voice** — natural bidirectional audio with 3-layer interruption system
8. **Skill Progression System** — DAG with prerequisites, PageRank recommendations, proficiency tracking
9. **NVIDIA DGX Spark** — 133-keypoint whole-body pose via GPU inference
10. **Camera Coverage Intelligence** — detects out-of-frame/occluded landmarks, guides user to reposition
11. **Session Memory** — per-skill JSONL history for personalized cross-session coaching
12. **Emotional AI** — adapts coaching tone based on performance trends and user affect

---

## Prize Alignment

| Prize | How Kinetic Fits |
|-------|------------------|
| **Anthropic Human Flourishing** | AI that makes physical health accessible to everyone — PT rehab, fitness, elderly care |
| **Best Use of Claude Agent SDK** | Deep integration: 46 MCP tools, 3 subagents, hooks (safety, audit, summary), skills, continuous conversation |
| **Greylock Best Multi-Turn Agent** | Coaching is inherently multi-turn: observe → correct → re-observe → progress over sessions |
| **Decagon Best Conversation** | 3-layer interruption system, emotional tone adaptation, session memory, multi-language — best-in-class conversational AI |
| **Neo Most Likely to Become a Product** | Clear market (PT, fitness, dance), clear business model (subscription), massive TAM |
| **Most Technically Complex** | Real-time CV + agent reasoning + voice + ML training + biomechanics + DGX — all integrated |
| **Most Impactful** | Physical therapy compliance → fewer re-injuries. Accessible coaching → health equity. |
| **Healthcare Track** | PT rehab is a direct healthcare application with measurable outcomes |
| **Education Track** | Teaching physical skills is education — applicable to any movement-based learning |
| **NVIDIA Edge AI** | DGX Spark with RTMPose-WholeBody (133 keypoints) for high-accuracy pose inference |

---

## API Surface

- **50+ REST endpoints** — coaching, references, graphs, training, model, memory, activity, rooms, DGX
- **3 WebSocket streams** — /ws/video (camera), /ws/audio (OpenAI Realtime voice), /ws/coaching (real-time data)
- **46 MCP tools** — 13 categories from perception to skill intelligence
- **Multiplayer rooms** — create/join/leaderboard APIs
- **Full FastAPI auto-docs** at /docs

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Detection | YOLO11n (5.4MB, 15 FPS) |
| Pose | MediaPipe PoseLandmarker Lite (5.6MB, 30 FPS async) |
| Tracking | ByteTrack (persistent IDs) |
| Depth | Depth Anything V2 Small (MPS GPU) |
| Hands | MediaPipe HandLandmarker (21 landmarks/hand, 30 FPS) |
| DGX Inference | NVIDIA DGX Spark — RTMPose-WholeBody (133 keypoints, GB10 GPU) |
| Agent | Claude Agent SDK (3 subagents, hooks, 46 MCP tools) |
| Voice | OpenAI Realtime API (GPT-4o, bidirectional, 3-layer interruption) |
| ML Training | 1D CNN (NumPy, 14K params) + Claude hybrid scoring |
| Server | FastAPI + uvicorn (Python 3.12) |
| Frontend | Next.js 14 + React + TailwindCSS + shadcn/ui |
| Communication | Telegram Bot API |
| Data | JSONL (training + session memory), JSON (skill graphs) |
| Codebase | ~15,000 lines of code |

---

## Team

Built at **TreeHacks 2026** @ Stanford University.

---

*"The best coach isn't the one who knows the most — it's the one who can see you, understand you, and guide you in the moment. Kinetic is that coach."*
