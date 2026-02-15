# AEGIS — Prizes & Architecture Reference

## Target Prizes (9 opt-in + 4 auto-eligible = 13 total)

### Auto-Eligible (FREE — don't count toward 10)

| Prize | What judges look for |
|---|---|
| **Grand Prize** ($12K / $8K / iPhone 17 Pro) | Innovation, functionality, overall excellence |
| **Most Creative** (Pioneer DJ) | Originality, unconventional thinking |
| **Most Impactful** (JBL Partybox) | Positive societal change, accessibility |
| **Most Technically Complex** (DJI Flip) | Technical sophistication, advanced tech mastery |

### Opt-In (9 of 10 slots used)

| # | Sponsor | Prize Name | Our Angle | Prize |
|---|---|---|---|---|
| 1 | **Anthropic** | Human Flourishing Track | AI makes PT rehab, yoga, elderly care, sign language accessible to everyone | Tungsten cubes + Claude Pro |
| 2 | **Anthropic** | Best Use of Claude Agent SDK | 3 sub-agents + 44 MCP tools + hooks + multi-agent orchestration | $2,500 credits |
| 3 | **Y Combinator** | Build Iconic YC Company with AI | Reimagine pre-2022 PT/coaching company — AI-native physical therapy | **Guaranteed YC interview** |
| 4 | **Greylock** | Best Multi-Turn Agent | Claude agent reasons across turns: pose→feedback→adjustment→progress tracking | $10K Warriors tickets + office hours |
| 5 | **Zoom** | Education Track | AI skill coaching IS education: yoga, PT exercises, sign language, elderly mobility | $1K + Ray-Bans |
| 6 | **Neo** | Most Likely to Become a Product | $50B PT market, working MVP, clear product-market fit | Neo retreat + mentors |
| 7 | **Human Capital** | Fellowship Prize | Solo engineer, ambitious problem, working product | **$50K equity-free** |
| 8 | **Decagon** | Best Conversation Assistant | Gemini Live voice takes turns, understands context, guides physical tasks naturally | Switch 2 + interview |
| 9 | **Google** | Cloud AI Track | Gemini 2.5 Flash Native Audio for real-time voice coaching | Pixel phones |

---

## Complete Technical Architecture

### High-Level System Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                        FRONTEND (Next.js)                       │
│                     localhost:3000 / Vercel                      │
│                                                                 │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌───────────────┐  │
│  │ Camera   │  │ Mic      │  │ YouTube  │  │ Practice w/   │  │
│  │ Feed     │  │ Capture  │  │ Expert   │  │ Friend (Room) │  │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘  └───────┬───────┘  │
│       │              │             │                │           │
│       │ WebSocket    │ WebSocket   │ HTTP API       │ HTTP API  │
│       │ /ws/video    │ /ws/audio   │ /api/coaching  │ /api/rooms│
└───────┼──────────────┼─────────────┼────────────────┼───────────┘
        │              │             │                │
        ▼              ▼             ▼                ▼
┌─────────────────────────────────────────────────────────────────┐
│                     BACKEND (FastAPI + Python)                   │
│                        localhost:8000                            │
│                                                                 │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │                    FastAPI Server (server.py)               │ │
│  │  44 REST routes + 3 WebSocket endpoints                    │ │
│  │                                                            │ │
│  │  WebSockets:                                               │ │
│  │  • /ws/video — receives camera frames, returns skeleton    │ │
│  │  • /ws/audio — bidirectional audio with Gemini Live        │ │
│  │  • /ws/coaching — sends coaching feedback + scores to UI   │ │
│  │                                                            │ │
│  │  REST APIs:                                                │ │
│  │  • /api/coaching/* — start, stop, progress, score, quality │ │
│  │  • /api/rooms/* — create, join, leaderboard, compare       │ │
│  │  • /api/references/* — record, load, list expert skeletons │ │
│  │  • /api/graphs/* — skill progression DAGs                  │ │
│  │  • /api/training/* — model training data                   │ │
│  │  • /api/model/* — local CNN inference                      │ │
│  └────────────────────────────────────────────────────────────┘ │
│                                                                 │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐  │
│  │ Spatial      │  │ Gemini Live  │  │ Claude Agent SDK     │  │
│  │ Engine       │  │ Bridge       │  │ Agent                │  │
│  │              │  │              │  │                      │  │
│  │ • YOLO11n    │  │ • Native     │  │ • 3 Sub-Agents      │  │
│  │ • MediaPipe  │  │   Audio      │  │ • 44 MCP Tools      │  │
│  │ • ByteTrack  │  │ • Voice:Kore │  │ • 3 Hooks           │  │
│  │ • Depth Any  │  │ • Bidir.     │  │ • Multi-turn        │  │
│  │   V2         │  │   streaming  │  │   reasoning         │  │
│  └──────┬───────┘  └──────┬───────┘  └──────────┬───────────┘  │
│         │                 │                      │              │
│         ▼                 ▼                      ▼              │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │              Pose Comparison Engine                       │   │
│  │  • Skeleton normalization (hip-centered, scale-invariant) │   │
│  │  • 10 joint angles (shoulders, elbows, hips, knees, etc) │   │
│  │  • DTW temporal alignment                                │   │
│  │  • Phase detection (prep → execute → peak → recovery)    │   │
│  │  • Rep counting (automatic via peak detection)           │   │
│  │  • Movement quality (smoothness, symmetry, ROM)          │   │
│  │  • Compensation detection (injury risk)                  │   │
│  │  • Similarity scoring (0-100)                            │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                 │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐  │
│  │ Skill Graph  │  │ Data         │  │ Room Manager         │  │
│  │              │  │ Collector    │  │ (Multiplayer)        │  │
│  │ • DAG with   │  │              │  │                      │  │
│  │   prereqs    │  │ • JSONL      │  │ • Room codes         │  │
│  │ • PageRank   │  │   storage    │  │ • Dual coaching      │  │
│  │   recommend. │  │ • Export for │  │   sessions           │  │
│  │ • Fitness,   │  │   training   │  │ • Leaderboard        │  │
│  │   yoga, PT   │  │ • T×10 angle │  │ • Claude comparison  │  │
│  │   skill trees│  │   matrices   │  │ • Final verdict      │  │
│  └──────────────┘  └──────────────┘  └──────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Claude Agent SDK — Deep Architecture

### Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                   CLAUDE AGENT SDK (sdk_agent.py)                │
│                                                                 │
│  ClaudeSDKClient                                                │
│  ├── Model: claude-sonnet-4-20250514                            │
│  ├── Permission: bypassPermissions                              │
│  ├── Max turns: 15                                              │
│  ├── Allowed tools: ["Task", "mcp__aegis__*"]                   │
│  │                                                              │
│  ├── MCP Server: "aegis" (in-process, 44 tools)                 │
│  │   └── create_sdk_mcp_server(name="aegis", tools=ALL_TOOLS)   │
│  │                                                              │
│  ├── Sub-Agents (via AgentDefinition + Task tool):              │
│  │   ├── perception-agent (sonnet, 10 tools)                    │
│  │   ├── coach-agent (sonnet, 14 tools)                         │
│  │   └── progress-agent (haiku, 10 tools)                       │
│  │                                                              │
│  └── Hooks (via HookMatcher):                                   │
│      ├── PreToolUse  → safety_hook (blocks dangerous actions)   │
│      ├── PostToolUse → audit_hook (logs all tool calls)         │
│      └── Stop        → stop_hook (session summary)              │
└─────────────────────────────────────────────────────────────────┘
```

### Sub-Agent Detail

```
┌─────────────────────────────────────────────────────────────┐
│                    MAIN ORCHESTRATOR AGENT                    │
│  Model: claude-sonnet-4-20250514                             │
│  Role: Routes user intent to the right sub-agent via Task    │
│  Tools: Task (delegates), all 44 MCP tools as fallback       │
│                                                              │
│  "User wants form check" ──→ delegates to coach-agent        │
│  "User asks about scene" ──→ delegates to perception-agent   │
│  "User asks about goals" ──→ delegates to progress-agent     │
└──────┬──────────────────┬─────────────────────┬──────────────┘
       │                  │                     │
       ▼                  ▼                     ▼
┌──────────────┐  ┌──────────────────┐  ┌────────────────────┐
│ PERCEPTION   │  │ COACH AGENT      │  │ PROGRESS AGENT     │
│ AGENT        │  │                  │  │                    │
│ Model:sonnet │  │ Model: sonnet    │  │ Model: haiku       │
│              │  │                  │  │                    │
│ 10 tools:    │  │ 14 tools:        │  │ 10 tools:          │
│              │  │                  │  │                    │
│ Spatial:     │  │ Comparison:      │  │ References:        │
│ • get_state  │  │ • compare_to_ref │  │ • list_references  │
│ • get_summary│  │ • joint_deviation│  │ • load_reference   │
│ • person_det │  │ • quality_analy. │  │ • parse_skill_doc  │
│ • scene_chg  │  │ • compensation   │  │                    │
│ • objects    │  │ • full_movement  │  │ Goals:             │
│ • count_obj  │  │                  │  │ • get_current_goal │
│              │  │ Coaching:        │  │ • update_goal      │
│ Posture:     │  │ • start_session  │  │ • get_presets      │
│ • analyze    │  │ • get_progress   │  │                    │
│ • landmarks  │  │ • get_rep_count  │  │ Memory:            │
│ • alignment  │  │ • end_session    │  │ • save_observation │
│              │  │ • analyze_desc.  │  │ • get_observations │
│ Activity:    │  │ • collect_rep    │  │                    │
│ • timeline   │  │                  │  │ Training:          │
│              │  │ Voice:           │  │ • bootstrap_model  │
│              │  │ • speak_to_user  │  │ • generate_data    │
│              │  │                  │  │                    │
│              │  │ References:      │  │                    │
│              │  │ • record_start   │  │                    │
│              │  │ • record_stop    │  │                    │
└──────────────┘  └──────────────────┘  └────────────────────┘
```

### Hook System

```
┌──────────────────────────────────────────────────────┐
│                    HOOK PIPELINE                       │
│                                                       │
│  Every tool call flows through:                       │
│                                                       │
│  ┌─────────────┐    ┌──────────────┐    ┌─────────┐  │
│  │ PreToolUse  │───▶│  Tool Exec   │───▶│PostTool │  │
│  │ safety_hook │    │              │    │audit_hk │  │
│  └─────────────┘    └──────────────┘    └─────────┘  │
│                                                       │
│  PreToolUse (safety_hook):                            │
│  • Blocks shell commands during coaching              │
│  • Returns {"decision": "block", "message": "..."}    │
│  • Matcher: "*" (all tools)                           │
│                                                       │
│  PostToolUse (audit_hook):                            │
│  • Logs tool name, args, result to decision log       │
│  • Tracks tool usage patterns                         │
│  • Matcher: "*" (all tools)                           │
│                                                       │
│  Stop (stop_hook):                                    │
│  • Generates session summary                          │
│  • Saves to memory store                              │
│  • Triggered when agent completes turn                │
└──────────────────────────────────────────────────────┘
```

---

## Conversational Voice Architecture (OpenAI Realtime API)

```
┌──────────────────────────────────────────────────────────────────┐
│              OPENAI REALTIME VOICE PIPELINE (GPT-4o)              │
│                                                                   │
│  Model: gpt-4o-realtime-preview    Voice: alloy                  │
│  Protocol: WebSocket (wss://api.openai.com/v1/realtime)          │
│  Audio: PCM 16kHz in, PCM 24kHz out, server-side VAD             │
│                                                                   │
│  BROWSER                   SERVER                    OPENAI      │
│  ┌──────┐  /ws/audio    ┌──────────┐  WebSocket   ┌──────────┐ │
│  │ Mic  │──base64 PCM──▶│  Voice   │──PCM 16kHz──▶│ Realtime │ │
│  │16kHz │               │  Bridge  │              │   API    │ │
│  └──────┘               │          │              │ (GPT-4o) │ │
│                          │          │◀─PCM 24kHz──│          │ │
│  ┌──────┐  /ws/audio    │          │  audio out   │          │ │
│  │Audio │◀─base64 PCM──│          │              │          │ │
│  │ Play │  24kHz        └──────────┘              └──────────┘ │
│  └──────┘                    ▲                         ▲        │
│                              │                         │        │
│  PROACTIVE COACHING:         │    SILENT CONTEXT:      │        │
│  Claude coaching loop ───speak()──▶ response.create    │        │
│  (every 10s)                 │                         │        │
│                              │    inject_coaching_context()      │
│  Frontend context ──────text msg──▶ (NO response.create)        │
│  (every 5s, scores/reps)    │    GPT-4o absorbs silently        │
│                              │                                   │
│  ═══════════════════════════════════════════════════════════════ │
│                                                                   │
│  TECHNICAL CHALLENGE: Proactive vs Reactive Voice                │
│  ──────────────────────────────────────────────────────────────  │
│                                                                   │
│  Problem: AI must coach proactively (count reps, fix form)       │
│  BUT also pause instantly when the user asks a question.          │
│                                                                   │
│  Solution: 3-Layer Interruption System                           │
│                                                                   │
│  Layer 1: Server-Side VAD (Voice Activity Detection)             │
│    OpenAI detects user speech start → speech_started event       │
│    → _user_speaking = True                                       │
│    → All queued audio CLEARED (old cues don't play)              │
│    → speak() returns False (coaching pauses)                     │
│    → inject_coaching_context() skips (no context flooding)       │
│                                                                   │
│  Layer 2: Response State Machine                                 │
│    response.created → _response_in_progress = True               │
│    → speak() and inject() blocked during AI response             │
│    response.done → _response_in_progress = False                 │
│    → Coaching resumes on next cycle                              │
│                                                                   │
│  Layer 3: Single Voice Source                                    │
│    Only speak() triggers response.create (voice output)          │
│    inject_coaching_context() is SILENT (background info only)    │
│    → Prevents triple-speak bug (3 sources → 3 voices)            │
│    → One consistent natural voice at all times                   │
│                                                                   │
│  Result: User says "is this correct?" →                          │
│    1. VAD detects speech (50ms)                                  │
│    2. Coaching audio cleared                                     │
│    3. OpenAI listens + has full coaching context                 │
│    4. Responds naturally: "Yeah, push your knees out more"       │
│    5. Coaching resumes after response.done                       │
│                                                                   │
│  ═══════════════════════════════════════════════════════════════ │
│                                                                   │
│  WHY OPENAI REALTIME OVER GEMINI LIVE:                           │
│  - Gemini reinterprets text (garbled coaching output)            │
│  - OpenAI Realtime: STT + reasoning + TTS in single API         │
│  - GPT-4o reasons about coaching data before speaking            │
│  - Native interruption via server-side VAD                       │
│  - Natural voice quality with alloy/echo/nova/shimmer options    │
│  - Gemini kept as fallback if OpenAI key unavailable             │
└──────────────────────────────────────────────────────────────────┘
```

---

## Computer Vision Pipeline

```
┌─────────────────────────────────────────────────────────────┐
│                    CV PIPELINE (spatial_engine.py)            │
│                                                              │
│  Camera Frame (1280×720 @ 30fps)                             │
│       │                                                      │
│       ▼                                                      │
│  ┌──────────────┐                                            │
│  │   YOLO11n    │  Person detection                          │
│  │   (5.4 MB)   │  ~15 FPS on M4 Pro                        │
│  │   0.80 conf  │  Bounding boxes + class labels             │
│  └──────┬───────┘                                            │
│         │                                                    │
│         ▼                                                    │
│  ┌──────────────┐                                            │
│  │  MediaPipe   │  33 pose landmarks per person              │
│  │  PoseLand-   │  ~30 FPS async (LIVE_STREAM mode)          │
│  │  marker Lite │  99.8% detection rate                      │
│  │  (5.6 MB)    │  3D world coordinates                      │
│  └──────┬───────┘                                            │
│         │                                                    │
│         ▼                                                    │
│  ┌──────────────┐                                            │
│  │  ByteTrack   │  Multi-person tracking                     │
│  │              │  Persistent IDs across frames              │
│  │              │  Re-identification on re-entry             │
│  └──────┬───────┘                                            │
│         │                                                    │
│         ▼                                                    │
│  ┌──────────────┐                                            │
│  │ Depth Any.   │  Monocular depth estimation                │
│  │ V2 Small     │  22 FPS on MPS GPU                         │
│  │              │  Relative depth map                        │
│  └──────┬───────┘                                            │
│         │                                                    │
│         ▼                                                    │
│  STRUCTURED SPATIAL DATA                                     │
│  {persons: [{id, bbox, landmarks[33], depth, activity}]}     │
│  → Frames discarded (privacy-first, no video storage)        │
└─────────────────────────────────────────────────────────────┘
```

---

## Pose Comparison Engine (pose_comparison.py — 1048 lines)

```
┌─────────────────────────────────────────────────────────────┐
│              POSE COMPARISON ENGINE                           │
│                                                              │
│  INPUT: User skeleton (33 landmarks) + Expert reference      │
│                                                              │
│  Step 1: NORMALIZE                                           │
│  • Center on hip midpoint                                    │
│  • Scale to unit torso length                                │
│  • Makes comparison position/size invariant                  │
│                                                              │
│  Step 2: JOINT ANGLES (10 angles)                            │
│  • Left/Right shoulder flexion                               │
│  • Left/Right elbow flexion                                  │
│  • Left/Right hip flexion                                    │
│  • Left/Right knee flexion                                   │
│  • Torso lean                                                │
│  • Neck angle                                                │
│                                                              │
│  Step 3: DTW ALIGNMENT                                       │
│  • Dynamic Time Warping matches user timing to expert        │
│  • Handles different speeds of execution                     │
│  • Cost matrix: angular distance across all joints           │
│                                                              │
│  Step 4: SIMILARITY SCORING (0-100)                          │
│  • Weighted joint angle comparison                           │
│  • Penalty for large deviations                              │
│  • Bonus for smooth transitions                              │
│                                                              │
│  Step 5: PHASE DETECTION                                     │
│  • Preparation → Execution → Peak → Recovery                 │
│  • Uses velocity + angle thresholds                          │
│                                                              │
│  Step 6: REP COUNTING                                        │
│  • Peak detection on primary joint angle                     │
│  • Automatic rep boundary identification                     │
│                                                              │
│  Step 7: MOVEMENT QUALITY                                    │
│  • Smoothness (jerk minimization)                            │
│  • Symmetry (left vs right comparison)                       │
│  • Range of Motion (% of target ROM achieved)                │
│                                                              │
│  Step 8: COMPENSATION DETECTION                              │
│  • Identifies when user compensates with wrong muscles       │
│  • Example: leaning torso during knee extension              │
│  • Flags injury risk                                         │
│                                                              │
│  OUTPUT: {score, reps, phase, quality, corrections[]}        │
└─────────────────────────────────────────────────────────────┘
```

---

## Multiplayer Practice Mode

```
┌─────────────────────────────────────────────────────────────┐
│               MULTIPLAYER ROOM ARCHITECTURE                  │
│                                                              │
│  Player A (Creator)              Player B (Joiner)           │
│  ┌──────────────┐               ┌──────────────┐            │
│  │ POST /api/   │               │ POST /api/   │            │
│  │ rooms/create │               │ rooms/join   │            │
│  │              │               │              │            │
│  │ Gets room    │◀─── code ───▶│ Enters code  │            │
│  │ code: JAB42X │               │ JAB42X       │            │
│  └──────┬───────┘               └──────┬───────┘            │
│         │                              │                    │
│         ▼                              ▼                    │
│  ┌─────────────────────────────────────────────────┐        │
│  │              ROOM (rooms.py)                     │        │
│  │                                                  │        │
│  │  room_code: "JAB42X"                             │        │
│  │  skill: "Warrior Pose"                           │        │
│  │  status: "active"                                │        │
│  │                                                  │        │
│  │  participants:                                   │        │
│  │  ├── Player A: {coaching_session, scores, reps}  │        │
│  │  └── Player B: {coaching_session, scores, reps}  │        │
│  │                                                  │        │
│  │  Each player has INDEPENDENT:                    │        │
│  │  • CoachingSession (pose comparison)             │        │
│  │  • Score tracking (avg, best, trend)             │        │
│  │  • Rep counting                                  │        │
│  └──────────────────────┬──────────────────────────┘        │
│                         │                                    │
│         ┌───────────────┼───────────────┐                    │
│         ▼               ▼               ▼                    │
│  ┌────────────┐  ┌────────────┐  ┌──────────────┐           │
│  │ Leaderboard│  │ Compare    │  │ Close Room   │           │
│  │ GET /rooms │  │ POST /rooms│  │ POST /rooms  │           │
│  │ /{code}/   │  │ /{code}/   │  │ /{code}/     │           │
│  │ leaderboard│  │ compare    │  │ close        │           │
│  │            │  │            │  │              │           │
│  │ Real-time  │  │ Claude     │  │ Claude final │           │
│  │ scores for │  │ analyzes   │  │ verdict:     │           │
│  │ both       │  │ both and   │  │ "Player A    │           │
│  │ players    │  │ compares   │  │ had better   │           │
│  │ (polled    │  │ strengths/ │  │ form on..."  │           │
│  │ every 3s)  │  │ weaknesses │  │              │           │
│  └────────────┘  └────────────┘  └──────────────┘           │
└─────────────────────────────────────────────────────────────┘
```

---

## Coaching Intelligence Loop

```
┌─────────────────────────────────────────────────────────────┐
│          COACHING INTELLIGENCE LOOP (every 10s)              │
│                                                              │
│  1. GATHER DATA                                              │
│     coaching_session.get_progress() → {                      │
│       reps_completed, avg_score, best_score,                 │
│       trend, top_corrections, latest_score                   │
│     }                                                        │
│                                                              │
│  2. BUILD PROMPT (inline data, no MCP tool calls)            │
│     "[COACHING DATA]                                         │
│      Skill: Warrior Pose                                     │
│      Reps: 5 | Avg: 78 | Best: 85 | Trend: improving        │
│      Top Corrections: hip_flexion, knee_alignment            │
│      ---                                                     │
│      Respond with ONLY a short coaching sentence."           │
│                                                              │
│  3. CLAUDE AGENT SDK processes prompt                        │
│     → Returns clean coaching text:                           │
│       "Great progress! Focus on keeping your front knee      │
│        directly over your ankle."                            │
│                                                              │
│  4. DISTRIBUTE FEEDBACK                                      │
│     ├──▶ WebSocket /ws/coaching → frontend shows text        │
│     └──▶ Gemini Live bridge.speak(text) → voice output       │
│                                                              │
│  5. USER HEARS + SEES FEEDBACK                               │
│     • Voice: Gemini speaks naturally (Kore voice)            │
│     • UI: feedback text + score ring + joint analysis         │
└─────────────────────────────────────────────────────────────┘
```

---

## Full Data Flow (End-to-End)

```
USER WITH CAMERA
      │
      │ video frames (WebSocket /ws/video)
      ▼
SPATIAL ENGINE (YOLO + MediaPipe + ByteTrack + Depth)
      │
      │ structured landmarks {33 joints × 3D coords}
      ▼
POSE COMPARISON ENGINE
      │
      │ {score: 78, reps: 5, phase: "execution", corrections: [...]}
      ▼
COACHING INTELLIGENCE LOOP (every 10s)
      │
      │ prompt with inline coaching data
      ▼
CLAUDE AGENT SDK (orchestrator → coach-agent)
      │
      │ "Focus on keeping your front knee over your ankle"
      ├──────────────────────────────┐
      ▼                              ▼
WEBSOCKET /ws/coaching         GEMINI LIVE BRIDGE
(text to frontend UI)          (text → natural voice)
      │                              │
      ▼                              ▼
USER SEES FEEDBACK             USER HEARS COACHING
(score ring, corrections)      (Kore voice, real-time)
```

---

## Codebase Stats

| Module | File | Lines | Purpose |
|---|---|---|---|
| **Server** | `aegis/server.py` | 1,418 | FastAPI, 44 routes, 3 WebSockets |
| **MCP Tools** | `aegis/mcp_server.py` | 1,675 | 46 MCP tools (fastmcp) — incl. hands, visibility, DGX |
| **SDK Tools** | `aegis/sdk_tools.py` | 493 | 45 tools wrapped for Claude Agent SDK |
| **SDK Agent** | `aegis/sdk_agent.py` | 411 | Claude Agent SDK client + sub-agents + hooks |
| **Pose Engine** | `aegis/pose_comparison.py` | 1,048 | Skeleton comparison, DTW, scoring, reps |
| **Skill Graph** | `aegis/skill_graph.py` | 648 | Skill DAG, PageRank recommendations |
| **OpenAI Voice** | `aegis/openai_voice.py` | 356 | OpenAI Realtime voice + interruption + emotional tone |
| **Gemini Bridge** | `aegis/gemini_bridge.py` | 319 | Gemini Live fallback voice |
| **Data Collector** | `aegis/data_collector.py` | 587 | JSONL training data, export |
| **Goals** | `aegis/goals.py` | 472 | 12 goal presets, dynamic goals |
| **Rooms** | `aegis/rooms.py` | 166 | Multiplayer room management |
| **Spatial Engine** | `aegis/spatial_engine.py` | 470 | YOLO + MediaPipe Pose + **MediaPipe Hands** + ByteTrack |
| **Scorer** | `aegis/skill_scorer.py` | 384 | Local 1D CNN (14K params, NumPy) |
| **Hybrid Scorer** | `aegis/hybrid_scorer.py` | 437 | Local 60% + Claude 40% scoring |
| **Frontend** | `coach/page.tsx` | 1,261 | Coaching UI, practice mode, setup |
| **API Client** | `api.ts` | 350 | Frontend API helpers |
| **DGX Client** | `aegis/dgx_client.py` | ~120 | NVIDIA DGX Spark inference client |
| **TOTAL** | | **~15,000** | |

---

## Tech Stack Summary

| Layer | Technology | Why |
|---|---|---|
| **AI Orchestration** | Claude Agent SDK (claude-agent-sdk v0.1.36) | Sub-agents, hooks, MCP, multi-turn reasoning |
| **Voice AI** | OpenAI Realtime API (GPT-4o) | Bidirectional voice, 3-layer interruption, emotional tone, multi-language |
| **Computer Vision** | YOLO11n + MediaPipe Pose + **MediaPipe Hands** + ByteTrack + Depth Anything V2 | Real-time skeleton (33 body + 21/hand) + tracking |
| **Backend** | FastAPI + Python 3.12 | Async WebSockets, REST APIs |
| **Frontend** | Next.js 14 + React + TailwindCSS + shadcn/ui | Modern UI, responsive, dark mode |
| **DGX Inference** | NVIDIA DGX Spark (RTMPose-WholeBody, GB10 GPU) | 133-keypoint whole-body pose estimation |
| **Protocol** | MCP (Model Context Protocol) | 46 tools for agent-tool communication |
| **Scoring** | Custom 1D CNN (NumPy, 14K params) + Claude hybrid | 0.15ms local inference + Claude for nuance |
| **Skills** | Skill Graph (DAG) + PageRank recommendations | Progression tracking with prerequisites |

---

## Known Gaps & Fixes Needed

### Gap 1: Coaching Loop Bypasses Sub-Agents — ✅ FIXED
**Problem:** Coaching prompts said "NO tool calls" which prevented sub-agent delegation.
**Fix:** Rewrote prompts to encourage tool use: "Use your coaching tools to analyze..."
**Status:** DONE — coaching loop now triggers perception-agent and coach-agent tool calls.

### Gap 2: No Finger-Level Tracking (Sign Language) — ✅ FIXED
**Problem:** MediaPipe Pose only tracks wrists (2 points). ASL needs individual finger joints.
**Fix:** Added MediaPipe HandLandmarker (21 landmarks/hand, async LIVE_STREAM, 30 FPS).
**Status:** DONE — integrated into spatial_engine.py + new MCP tool `get_hand_landmarks` + SDK tool wrapper.

### Gap 3: Document Upload Not Functional
**Problem:** Frontend has upload UI but backend doesn't parse PDFs into movement cues.
**Fix:** Low priority — can describe skills instead. Remove from demo claims if not fixed.

### Gap 4: Hybrid Scoring Not in Live Pipeline
**Problem:** `hybrid_scorer.py` exists but isn't called during coaching loop.
**Fix:** Wire into coaching session scoring. Medium effort.

### Gap 5: Compensation Detection Not Live
**Problem:** `detect_compensation_patterns()` exists in pose_comparison.py but isn't called during the coaching intelligence loop.
**Fix:** Add to coaching loop data gathering. Small effort.

### Gap 6: End-to-End Flow Untested
**Problem:** Never done a full run: camera → skeleton → score → Claude → OpenAI voice → user hears feedback.
**Fix:** Must test before demo.

### Gap 7: Camera Coverage → ✅ FIXED
**Problem:** Camera can't always see full body (legs cut off, too close/far).
**Fix:** Added `check_landmark_visibility` MCP tool. Coaching loop checks every 30s and warns user via voice.
**Status:** DONE — warns "step back from camera" when ankles/knees out of frame.

### Gap 8: Voice Double/Triple Speak → ✅ FIXED
**Problem:** Gemini Live + browser TTS + backend speech all active simultaneously.
**Fix:** Removed Gemini Live + browser TTS. OpenAI Realtime is sole voice. Silent context injection. 3-layer interruption.
**Status:** DONE — single consistent voice with natural interruption handling.

### Gap 9: No Session Memory → ✅ FIXED
**Problem:** Each session starts fresh, no awareness of past performance.
**Fix:** `DataCollector.get_last_session_summary()` loads per-skill JSONL. Injected into OpenAI on coaching start.
**Status:** DONE — "Last time you averaged 78 on squats, knee alignment was the issue."

---

## CV Pipeline Upgrade Options

### Option A: MediaPipe Hands (RECOMMENDED — Safe)
- Add alongside existing pose detection
- 21 landmarks per hand (all finger joints)
- 30+ FPS on M4 Pro CPU
- Same mediapipe ecosystem, same API pattern
- Enables: ASL alphabet, finger spelling, greetings
- Effort: 30 min
- Risk: Zero

### Option B: DWPose (Risky)
- Replaces MediaPipe Pose entirely
- Body (133 keypoints) + Hands (42 points) + Face (68 points) = 243 total
- HuggingFace model, needs ONNX or PyTorch
- Needs GPU (MPS untested)
- Effort: 2-3 hours
- Risk: High — could break entire pipeline

### Option C: RTMPose-WholeBody (Risky)
- MMPose ecosystem, same capabilities as DWPose
- Needs mmpose, mmdet, mmengine dependencies
- Effort: 2-3 hours
- Risk: High — heavy dependency chain

### Decision: Go with Option A (MediaPipe Hands) — safe, fast, enables sign language.


---

## MCP Architecture Deep Dive (Client ↔ Server)

```
┌─────────────────────────────────────────────────────────────┐
│                    HOW MCP WORKS IN AEGIS                     │
│                                                              │
│  There is NO separate MCP server deployment.                 │
│  Everything runs IN-PROCESS within the Python backend.       │
│                                                              │
│  STARTUP (run_server.py):                                    │
│    1. engine = SpatialEngine()        → starts CV pipeline   │
│    2. agent = AegisSDKAgent()         → creates Claude client│
│    3. mcp_module.init(engine=engine)  → gives tools access   │
│                                          to the CV engine    │
│                                                              │
│  TOOL CALL FLOW:                                             │
│                                                              │
│  Claude decides: "I need to check the user's hand landmarks" │
│       │                                                      │
│       ▼                                                      │
│  Claude Agent SDK (internal MCP client)                      │
│       │  MCP protocol (in-process, zero network latency)     │
│       ▼                                                      │
│  create_sdk_mcp_server(name="aegis", tools=ALL_TOOLS)        │
│       │  Routes to matching @tool function                   │
│       ▼                                                      │
│  sdk_tools.py: get_hand_landmarks()                          │
│       │  Direct Python function call                         │
│       ▼                                                      │
│  mcp_server.py: get_hand_landmarks()                         │
│       │  Reads from _engine (set during init)                │
│       ▼                                                      │
│  spatial_engine.get_state()["hands"]                         │
│       │                                                      │
│       ▼                                                      │
│  Returns: {hands_detected: 2, hands: [{handedness: "Right",  │
│            landmarks: [{name: "thumb_tip", x: 0.45, ...}]}   │
│                                                              │
│  Same pattern for ALL 45 tools.                              │
└─────────────────────────────────────────────────────────────┘
```

### Why Two Layers? (mcp_server.py vs sdk_tools.py)

| Layer | File | Purpose |
|---|---|---|
| **mcp_server.py** | 45 `@mcp.tool()` functions (fastmcp) | Core tool logic + used by REST API endpoints in server.py |
| **sdk_tools.py** | 45 `@tool` wrappers (claude_agent_sdk) | Same functions re-wrapped for Claude Agent SDK's MCP format |
| **create_aegis_mcp_server()** | Bundles all @tool functions | Creates in-process MCP server the SDK client connects to |

`mcp_server.py` is the source of truth for tool logic. `sdk_tools.py` is a thin wrapper that makes the same tools available to Claude via the Agent SDK's MCP protocol.

### Sub-Agent Tool Restrictions

Each sub-agent only sees a subset of the 45 tools:

| Sub-Agent | Tools Available | Purpose |
|---|---|---|
| **perception-agent** | 11 tools | Spatial state, pose landmarks, hand landmarks, body alignment, activity |
| **coach-agent** | 13 tools | Compare to reference, joint deviation, movement quality, compensation, coaching session |
| **progress-agent** | 9 tools | Skill graph, recommendations, training data, model predict |

The main orchestrator agent can use ALL 45 tools + the `Task` tool to delegate to sub-agents.

---

## Recent Updates (Session 2)

### 1. MediaPipe Hands Integration
- Added `HandLandmarker` to `spatial_engine.py` (async LIVE_STREAM mode)
- 21 landmarks per hand (all finger joints: thumb, index, middle, ring, pinky)
- Runs alongside existing pose detection at 30 FPS
- New MCP tool: `get_hand_landmarks` in both mcp_server.py and sdk_tools.py
- Enables: ASL alphabet coaching, finger spelling, hand gesture analysis

### 2. Sub-Agent Delegation Fix
- Coaching prompts in `server.py` now encourage tool use instead of blocking it
- Three prompt types: session_start, rep_feedback, correction
- Agent orchestration metadata (action type, tools used, sub-agents invoked) sent to frontend via WebSocket

### 3. Gemini Live Conversation Upgrade
- System instruction rewritten for deep conversational coaching
- Multi-turn context: tracks improvement across reps, builds on corrections
- Skill-specific coaching styles: PT (gentle), yoga (calm), sign language (precise), dance (energetic)
- Natural coaching behaviors: counts reps, celebrates milestones, adapts to user requests

