# AEGIS — Build Status & Inventory

> **One device. Any goal. Any space.**
> Last updated: Feb 13, 2026, 2:55 PM PST

---

## Quick Stats

| Metric | Value |
|--------|-------|
| Total lines of code | ~6,400 |
| Python modules | 21 files |
| Frontend files | 3 (HTML + JS) |
| MCP tools | 25 |
| Goal presets | 6 |
| API endpoints | 13 REST + 2 WebSocket |
| CV models | 3 (YOLO11n, MediaPipe Pose, Depth Anything V2) |
| Experiments passed | 8/8 |

---

## Architecture Overview

```
┌────────────────────────────────────────────────────────┐
│                    AEGIS Device                         │
│                                                        │
│  ┌──────────┐   ┌──────────────┐   ┌───────────────┐  │
│  │  Camera   │──▶│  CV Pipeline  │──▶│ Spatial State  │  │
│  │  (YOLO +  │   │  (15 FPS)    │   │  (JSON)       │  │
│  │  Pose +   │   └──────────────┘   └───────┬───────┘  │
│  │  Track)   │                              │          │
│  └──────────┘                              ▼          │
│                                    ┌──────────────┐   │
│  ┌──────────┐                      │  MCP Server   │   │
│  │ Gemini   │◀─── voice ──────────│  (25 tools)   │   │
│  │ Live     │                      └──────┬───────┘   │
│  │ (VOICE)  │                             │           │
│  └──────────┘                             ▼           │
│                                    ┌──────────────┐   │
│  ┌──────────┐                      │ Claude Agent  │   │
│  │ Telegram │◀─── alerts ─────────│  (BRAIN)      │   │
│  └──────────┘                      └──────────────┘   │
│                                                        │
│  ┌──────────────────────────────────────────────────┐  │
│  │  FastAPI Server (port 8000)                       │  │
│  │  /ws/video  /ws/audio  /dashboard  /api/*         │  │
│  └──────────────────────────────────────────────────┘  │
└────────────────────────────────────────────────────────┘
```

---

## Component Inventory

### 1. CV Pipeline (the EYES) — ✅ BUILT & TESTED

Real-time computer vision running at ~15 FPS on M4 Pro.

| Component | File | Lines | Status |
|-----------|------|-------|--------|
| Person detection | `src/perception.py` | 319 | ✅ YOLO11n, 80 COCO classes |
| Pose estimation | `src/perception.py` | (included) | ✅ MediaPipe 33 landmarks |
| Multi-person tracking | `src/perception.py` | (included) | ✅ ByteTrack, persistent IDs |
| Motion modeling | `src/motion.py` | 83 | ✅ Velocity, direction, prediction |
| Risk estimation | `src/risk.py` | 106 | ✅ TTC, danger zones, risk scores |
| Activity recognition | `aegis/activity.py` | 169 | ✅ 8 activities from pose geometry |
| Depth estimation | `src/perception.py` | (included) | ✅ Depth Anything V2 (optional) |
| Data models | `src/models.py` | 124 | ✅ TrackedPerson, BBox, RiskEvent, etc. |

**Activities detected:** standing, sitting, walking, running, fallen, lying_down, waving, reaching, crouching

**Objects detected:** All 80 COCO classes — person, bicycle, car, chair, laptop, cell phone, cup, bottle, book, backpack, etc.

**Experiments (all passed):**
- `experiments/exp1-exp8` — detection, tracking, pose, depth, combined pipeline
- Results logged in `experiments_log.md`

---

### 2. Spatial Engine — ✅ BUILT & TESTED

| Component | File | Lines | Status |
|-----------|------|-------|--------|
| Engine core | `aegis/spatial_engine.py` | 408 | ✅ Background thread, JSON state |
| External frame input | (included) | — | ✅ `push_frame()` for phone/device |
| Snapshot capture | (included) | — | ✅ Save annotated/raw JPEGs |
| Display frame | (included) | — | ✅ Bounding boxes, overlays, HUD |
| Event callbacks | (included) | — | ✅ Risk event notifications |
| Tracked persons | (included) | — | ✅ Raw pose landmark access |

**Modes:** Local camera (`--camera`) or external source (WebSocket from phone/device)

**Output:** Structured JSON with persons, objects, zones, risks — updated every frame.

---

### 3. MCP Server (aegis-spatial) — ✅ BUILT & TESTED

25 tools exposed via FastMCP, auto-discovered by the Claude agent.

| File | Lines | Package |
|------|-------|---------|
| `aegis/mcp_server.py` | 975 | `fastmcp==2.14.5` |

**Tool Inventory (25 tools, 8 categories):**

| # | Category | Tool | Description |
|---|----------|------|-------------|
| 1 | Perception | `get_spatial_state` | Full JSON — persons, objects, zones, risks |
| 2 | Perception | `get_spatial_summary` | Concise human-readable scene description |
| 3 | Perception | `get_person_detail` | Deep detail on a specific tracked person |
| 4 | Perception | `get_scene_changes` | Diff: new people, left, activity changes, objects |
| 5 | Perception | `get_objects_in_scene` | List detected objects, optional class filter |
| 6 | Perception | `count_objects` | Count specific object class (e.g. "cell phone") |
| 7 | Pose | `analyze_posture` | Shoulder/spine/head/knee metrics from skeleton |
| 8 | Pose | `get_pose_landmarks` | Raw 33 MediaPipe landmarks with names |
| 9 | Pose | `check_body_alignment` | Deviation checks + exercise-specific analysis |
| 10 | Activity | `get_activity_timeline` | Activity segments over time per person |
| 11 | Activity | `get_time_in_activity` | Duration in specific activity (e.g. "sitting") |
| 12 | Activity | `get_session_stats` | Session duration, people seen, alerts, goal info |
| 13 | Zones | `set_watch_zone` | Define rectangular monitoring zone (pixels) |
| 14 | Zones | `clear_watch_zones` | Remove all zones |
| 15 | Zones | `check_zone_status` | Who's in/near zones + active risks |
| 16 | Alerts | `send_telegram_alert` | Message user via Telegram, optional photo |
| 17 | Alerts | `speak_to_user` | Voice output (Gemini Live or macOS TTS fallback) |
| 18 | Alerts | `capture_photo` | Save snapshot (annotated or raw) |
| 19 | Memory | `save_observation` | Persist a note with tags for later recall |
| 20 | Memory | `get_observations` | Retrieve saved observations, filter by tag |
| 21 | Knowledge | `web_search` | Tavily search API (if configured) |
| 22 | Knowledge | `get_current_time` | Date, time, day of week |
| 23 | Goals | `get_current_goal` | Active goal info |
| 24 | Goals | `update_goal` | Change goal via natural language |
| 25 | Goals | `get_goal_presets` | List all 6 preset goals |

**Connection:** Agent connects via in-process `fastmcp.Client(mcp)` — zero network overhead.

---

### 4. Claude Agent (the BRAIN) — ✅ BUILT & TESTED

| Component | File | Lines | Status |
|-----------|------|-------|--------|
| Agent core | `aegis/agent.py` | 362 | ✅ MCP tool discovery + execution |
| Goal system | `aegis/goals.py` | 259 | ✅ 6 presets + dynamic custom goals |
| Proactive monitor | `aegis/monitor.py` | 117 | ✅ Fall detection, risk events, heartbeat |

**Model:** `claude-sonnet-4-20250514` (configurable via `AEGIS_MODEL` env)

**How it works:**
1. Agent discovers all 25 tools from MCP server on first use
2. Auto-converts MCP schemas → Anthropic tool format
3. Tool calls go through MCP: `Agent → Client(mcp) → FastMCP → tool function`
4. Goal-driven system prompt adapts reasoning style per goal
5. Conversation history maintained with auto-trimming (last 20 turns)

**Agent features:**
- Goal-driven reasoning (different system prompts per goal)
- Proactive event handling (called by Monitor on fall/risk detection)
- User message handling (from Telegram or console)
- Periodic checks (heartbeat loop)
- Tool call logging + decision logging (for dashboard)

---

### 5. Goal System — ✅ BUILT & TESTED

| Preset | ID | Category | Key Behavior |
|--------|----|----------|-------------|
| Desk Guardian | `desk_watch` | security | Alert on approach, track intruders |
| Posture Coach | `posture_coach` | wellness | Analyze skeleton, give form corrections |
| Driver Monitor | `driver_monitor` | safety | Detect drowsiness, escalating alerts |
| Study Focus | `study_focus` | productivity | Track distraction, phone detection, Pomodoro |
| Elderly Care | `elderly_care` | healthcare | Fall detection (#1 priority), inactivity alerts |
| Spatial Awareness | `general` | general | Report everything noteworthy |

**Dynamic goals:** Any natural language goal is accepted. If no preset matches, a custom goal is created with a rich prompt that tells Claude to interpret freely using all 25 tools.

**Examples of custom goals that work:**
- "Count how many people walk through the hallway"
- "Make sure my cat doesn't jump on the counter"
- "Remind me to take a break every 20 minutes"
- "Watch for deliveries at the front door"

---

### 6. Gemini Live Voice (the VOICE) — ✅ BUILT

| Component | File | Lines | Status |
|-----------|------|-------|--------|
| Server-side bridge | `aegis/gemini_bridge.py` | 316 | ✅ Python SDK, async |
| Client-side voice | `aegis/static/app.js` | 691 | ✅ Browser → Gemini direct |
| Legacy TTS narrator | `aegis/voice.py` | 210 | ✅ macOS say / espeak fallback |

**Model:** `gemini-2.5-flash-preview-native-audio-dialog`
**Voice:** Kore (configurable)

**Server-side bridge features:**
- Connects to Gemini Live via `google.genai` Python SDK
- Goal-aware system instructions (adapts when goal changes)
- Spatial context injection (periodic state updates as text)
- Proactive narration via `bridge.speak()` (used by `speak_to_user` tool)
- Audio I/O queues for WebSocket proxying

**Client-side (phone app) features:**
- Direct browser → Gemini WebSocket (low latency)
- Mic capture (16kHz PCM) + audio playback (24kHz PCM)
- Spatial state injected as text context every 3 seconds
- Interruption handling (clears playback queue)

---

### 7. Web Server & API — ✅ BUILT

| Component | File | Lines | Status |
|-----------|------|-------|--------|
| FastAPI server | `aegis/server.py` | 298 | ✅ REST + WebSocket |
| Server runner | `aegis/run_server.py` | 98 | ✅ Orchestrates everything |
| Phone web app | `aegis/static/index.html` | 248 | ✅ Camera + overlay + voice |
| Phone app JS | `aegis/static/app.js` | 691 | ✅ VideoStream + SpatialOverlay + GeminiVoice |
| Demo dashboard | `aegis/static/dashboard.html` | 445 | ✅ Goals + state + tool log + decisions |

**REST Endpoints:**

| Method | Path | Purpose |
|--------|------|---------|
| GET | `/` | Phone web app |
| GET | `/dashboard` | Demo dashboard |
| GET | `/api/state` | Current spatial state JSON |
| GET | `/api/summary` | Human-readable summary |
| GET | `/api/config` | Client config (Gemini key, model) |
| GET | `/api/goals` | All goals + active goal |
| POST | `/api/goals/{id}` | Set active goal |
| GET | `/api/logs/tools` | Recent tool call log |
| GET | `/api/logs/decisions` | Recent decision log |
| GET | `/api/agent/status` | Agent status + goal info |
| GET | `/api/voice/status` | Gemini Live bridge status |

**WebSocket Endpoints:**

| Path | Purpose |
|------|---------|
| `/ws/video` | Phone sends camera frames, receives spatial state |
| `/ws/audio` | Phone sends mic audio, receives Gemini audio (server-proxied) |

---

### 8. Telegram Bot — ✅ BUILT

| File | Lines | Status |
|------|-------|--------|
| `aegis/telegram_bot.py` | 256 | ✅ Commands + message forwarding |

**Commands:**
- `/start` — Welcome message
- `/status` — Current goal + spatial summary
- `/goals` — List all goals with switch buttons
- `/goal_<id>` — Quick goal switch (e.g. `/goal_desk_watch`)
- `/help` — Command list
- Any text → forwarded to Claude agent

---

### 9. Configuration — ✅ READY

| File | Lines | Purpose |
|------|-------|---------|
| `aegis/config.py` | 51 | All settings, env var overrides |
| `.env` | — | API keys (gitignored) |

**Required env vars:**
- `ANTHROPIC_API_KEY` — Claude agent
- `GEMINI_API_KEY` — Gemini Live voice

**Optional env vars:**
- `TELEGRAM_BOT_TOKEN` + `TELEGRAM_CHAT_ID` — Telegram alerts
- `TAVILY_API_KEY` — Web search tool
- `AEGIS_MODEL` — Override Claude model
- `AEGIS_CAMERA` — Camera index
- `AEGIS_HOST` / `AEGIS_PORT` — Server bind

---

## Entry Points

| Command | What it does |
|---------|-------------|
| `python -m aegis.run_server` | Full server (phone/device mode, dashboard, API) |
| `python -m aegis.run_server --camera` | Server with local webcam |
| `python -m aegis.run_server --no-agent` | CV pipeline only, no Claude |
| `python -m aegis.main` | Standalone mode (camera + Telegram + monitor) |
| `python -m aegis.main --show-camera` | With live camera window (macOS) |
| `python -m aegis.main --voice` | With TTS narration |
| `python -m aegis.main --engine-only` | CV pipeline only, prints state |
| `python aegis/mcp_server.py` | Standalone MCP server (stdio) |

---

## Dependencies

**Python packages (key):**
- `ultralytics` — YOLO11n
- `mediapipe` — Pose estimation
- `opencv-python` — Frame processing
- `torch` + `transformers` — Depth Anything V2
- `anthropic` — Claude API
- `google-genai` — Gemini Live API
- `fastmcp` — MCP server framework
- `fastapi` + `uvicorn` — Web server
- `python-dotenv` — Environment management

**ML Models:**
- `yolo11n.pt` — 5.4 MB, auto-downloaded
- `models/pose_landmarker_lite.task` — 5.6 MB
- Depth Anything V2 Small — downloaded on first use

---

## What's NOT Built Yet

| Item | Priority | Notes |
|------|----------|-------|
| End-to-end demo with 3 goals | High | Need to test full loop: goal set → CV → agent reasons → alerts |
| Demo choreography | Medium | 2-min scripted walkthrough for judges |
| Claude Agent SDK migration | Optional | Current direct API works; SDK would add hooks, subagents |
| Production error handling | Low | Hackathon-grade is fine for now |
| requirements.txt update | Low | Missing fastmcp, google-genai, fastapi, uvicorn |

---

## File Tree

```
project-hoover/
├── aegis/                          # Main AEGIS package
│   ├── __init__.py
│   ├── config.py                   # All configuration (51 lines)
│   ├── spatial_engine.py           # CV pipeline runner (408 lines)
│   ├── activity.py                 # Activity recognition from pose (169 lines)
│   ├── mcp_server.py               # 25 MCP tools via FastMCP (975 lines)
│   ├── agent.py                    # Claude agent + MCP bridge (362 lines)
│   ├── goals.py                    # 6 presets + dynamic goals (259 lines)
│   ├── gemini_bridge.py            # Gemini Live server-side voice (316 lines)
│   ├── voice.py                    # Legacy TTS narrator (210 lines)
│   ├── monitor.py                  # Proactive heartbeat loop (117 lines)
│   ├── server.py                   # FastAPI backend (298 lines)
│   ├── run_server.py               # Server orchestrator (98 lines)
│   ├── main.py                     # Standalone entry point (253 lines)
│   ├── telegram_bot.py             # Telegram interface (256 lines)
│   └── static/
│       ├── index.html              # Phone web app (248 lines)
│       ├── app.js                  # Phone app JS (691 lines)
│       └── dashboard.html          # Demo dashboard (445 lines)
├── src/                            # CV pipeline internals
│   ├── models.py                   # Data models (124 lines)
│   ├── perception.py               # YOLO + Pose + Depth (319 lines)
│   ├── motion.py                   # Velocity + prediction (83 lines)
│   ├── risk.py                     # TTC + risk scoring (106 lines)
│   ├── decision.py                 # Decision engine (84 lines)
│   ├── intervention.py             # Alert actions (209 lines)
│   └── dashboard.py                # Legacy Flask dashboard (295 lines)
├── experiments/                    # 8 CV experiments (all passed)
├── models/                         # ML model files
├── data/snapshots/                 # Captured images
├── .env                            # API keys (gitignored)
├── requirements.txt                # Python dependencies
├── DESIGN.md                       # Product design doc
├── STATUS.md                       # ← THIS FILE
└── yolo11n.pt                      # YOLO model weights
```
