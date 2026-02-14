# AEGIS — Design Document

> **One device. Any goal. Any space.**
> Goal-directed spatial AI that you place anywhere, give a goal, and it watches, understands, and acts autonomously.

---

## Product Vision

AEGIS is a small, smart camera device you place on any surface — desk, car dashboard, yoga mat, study area, elderly's room. You tell it what to watch for (in natural language), and it:

1. **Sees** — CV pipeline detects people, objects, poses, activities in real-time
2. **Thinks** — Claude reasons about what's happening relative to your goal
3. **Acts** — Alerts you via Telegram, speaks through the device, coaches you in real-time

### Use Cases

| Placement | Goal Example | What AEGIS Does |
|-----------|-------------|-----------------|
| Desk | "Alert if someone approaches my desk" | Monitors zone, detects intruders, sends Telegram alert with photo |
| Car dashboard | "Watch me for drowsiness" | Analyzes head tilt, eye closure, alerts with escalating urgency |
| Yoga mat | "Coach my warrior pose" | Analyzes pose landmarks, gives real-time voice corrections |
| Student desk | "Help me stay focused" | Detects phone pickup, distraction, tracks focus time |
| Elderly's room | "Watch for falls" | Fall detection → immediate alert to caregiver with photo |
| Anywhere | _Any custom goal_ | Claude interprets the goal dynamically and decides what to monitor |

---

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                    AEGIS DEVICE                      │
│  Camera + Mic + Speaker (or Jetson/Pi/Webcam)       │
└──────────────┬──────────────────┬───────────────────┘
               │ frames           │ audio
               ▼                  ▼
┌──────────────────────┐  ┌───────────────────────┐
│  CV PIPELINE (Eyes)  │  │  GEMINI LIVE (Voice)  │
│  YOLO11n detection   │  │  Real-time speech I/O │
│  MediaPipe Pose      │  │  Bidirectional audio  │
│  ByteTrack tracking  │  │  VAD + proactive      │
│  Activity recognition│  └───────────┬───────────┘
│  80-class objects     │              │
│  Depth estimation    │              │
└──────────┬───────────┘              │
           │ structured spatial state  │
           ▼                          │
┌──────────────────────────────────────────────────────┐
│              MCP SERVER (aegis-spatial)               │
│                                                      │
│  Tools exposed via Model Context Protocol:           │
│  - Perception: spatial state, scene changes, objects │
│  - Pose: posture analysis, landmarks, alignment      │
│  - Activity: timeline, focus tracking, statistics    │
│  - Zones: set/check watched areas                    │
│  - Alerts: Telegram, voice, photo capture            │
│  - Memory: save/recall observations                  │
│  - Knowledge: web search, time                       │
│  - Goals: get/set/update monitoring goal             │
└──────────────────┬───────────────────────────────────┘
                   │ MCP protocol
                   ▼
┌──────────────────────────────────────────────────────┐
│              CLAUDE AGENT (Brain)                     │
│                                                      │
│  Claude Agent SDK with:                              │
│  - MCP server connection (tools auto-discovered)     │
│  - Dynamic goal interpretation (no static matching)  │
│  - Hooks: PreToolUse, PostToolUse, audit logging     │
│  - Subagents: perception, safety, communicator       │
│  - Session memory across monitoring period           │
└──────────────────────────────────────────────────────┘
```

---

## MCP Server: Complete Tool Inventory

### Category 1: PERCEPTION — Reading the Scene

| # | Tool | Parameters | Returns | Description |
|---|------|-----------|---------|-------------|
| 1 | `get_spatial_state` | _none_ | Full JSON | Everything: persons, objects, activities, velocities, predictions, risks, zones |
| 2 | `get_spatial_summary` | _none_ | Text | Human-readable 1-paragraph scene description |
| 3 | `get_person_detail` | `track_id: int` | JSON | Deep info on one person: pose landmarks, activity, velocity, time in scene |
| 4 | `get_scene_changes` | `seconds_back: int = 10` | JSON | What changed recently: new people, people left, activity changes, new objects |
| 5 | `get_objects_in_scene` | `class_filter: str? = null` | JSON list | All detected objects, optionally filtered by class (e.g., "phone", "laptop") |
| 6 | `count_objects` | `class_name: str` | int | Count specific objects (e.g., "how many chairs?") |

### Category 2: POSE ANALYSIS — Body Understanding

| # | Tool | Parameters | Returns | Description |
|---|------|-----------|---------|-------------|
| 7 | `analyze_posture` | `track_id: int` | JSON | Posture analysis: shoulder alignment, spine angle, head tilt, weight distribution |
| 8 | `get_pose_landmarks` | `track_id: int` | JSON | Raw 33 landmark coordinates (x, y, z, visibility) for custom analysis |
| 9 | `check_body_alignment` | `track_id: int, exercise: str?` | JSON | Compare current pose to ideal alignment, return deviations |

### Category 3: ACTIVITY & TIMELINE

| # | Tool | Parameters | Returns | Description |
|---|------|-----------|---------|-------------|
| 10 | `get_activity_timeline` | `track_id: int, last_minutes: int = 5` | JSON list | Timeline: [{activity, start, end, duration}] |
| 11 | `get_time_in_activity` | `track_id: int, activity: str` | float (seconds) | How long has person X been doing activity Y? |
| 12 | `get_session_stats` | _none_ | JSON | Total time, people seen, events, alerts sent, goal changes |

### Category 4: ZONES — Spatial Monitoring

| # | Tool | Parameters | Returns | Description |
|---|------|-----------|---------|-------------|
| 13 | `set_watch_zone` | `x1, y1, x2, y2: int, label: str` | zone_id | Define a rectangular zone to monitor |
| 14 | `clear_watch_zones` | _none_ | confirmation | Remove all defined zones |
| 15 | `check_zone_status` | `zone_id: str?` | JSON | Who/what is in a specific zone (or all zones) |

### Category 5: ALERTS — Communicating with User

| # | Tool | Parameters | Returns | Description |
|---|------|-----------|---------|-------------|
| 16 | `send_telegram_alert` | `message: str, include_photo: bool = false` | confirmation | Send alert to user via Telegram |
| 17 | `speak_to_user` | `message: str, urgency: str = "normal"` | confirmation | Speak through device speaker (Gemini Live TTS) |
| 18 | `capture_photo` | `annotated: bool = true` | file_path | Take snapshot (with or without bounding box overlays) |

### Category 6: MEMORY — Persistent Context

| # | Tool | Parameters | Returns | Description |
|---|------|-----------|---------|-------------|
| 19 | `save_observation` | `text: str, tags: list[str] = []` | observation_id | Save a note ("Person 1 keeps rubbing eyes — possible drowsiness") |
| 20 | `get_observations` | `tag_filter: str? = null, last_n: int = 20` | JSON list | Retrieve past observations |

### Category 7: KNOWLEDGE — External Info

| # | Tool | Parameters | Returns | Description |
|---|------|-----------|---------|-------------|
| 21 | `web_search` | `query: str` | JSON results | Search the web (e.g., "proper warrior II pose alignment") |
| 22 | `get_current_time` | _none_ | ISO string | Current date/time |

### Category 8: GOAL MANAGEMENT

| # | Tool | Parameters | Returns | Description |
|---|------|-----------|---------|-------------|
| 23 | `get_current_goal` | _none_ | JSON | Current goal: id, name, description |
| 24 | `update_goal` | `description: str` | JSON | Change goal to new natural language description |
| 25 | `get_goal_presets` | _none_ | JSON list | List available preset goal shortcuts |

**Total: 25 tools**

---

## Goal System Design

### How Goals Work

1. User describes a goal in natural language (voice or text): _"Watch me for drowsiness while driving"_
2. Goal text is stored as-is — NO keyword matching, NO static mapping
3. Goal text is injected into Claude's system prompt
4. Claude dynamically decides which tools to use based on the goal
5. Preset goals are just convenience shortcuts with pre-written descriptions

### Presets (shortcuts only, not the primary mechanism)

| ID | Name | Description |
|----|------|-------------|
| `desk_watch` | Desk Guardian | Watch my desk and alert if anyone approaches or touches my stuff |
| `posture_coach` | Posture & Form Coach | Coach my posture, yoga form, or exercise technique in real-time |
| `driver_monitor` | Driver Alertness | Watch the driver for signs of drowsiness or distraction |
| `study_focus` | Study Focus | Help me stay focused while studying — alert if I get distracted |
| `elderly_care` | Elderly Care | Watch for falls, inactivity, or unusual behavior. Alert caregiver immediately |
| `general` | Spatial Awareness | General spatial monitoring — describe what's happening |

### System Prompt Structure

```
You are AEGIS — a goal-directed spatial AI device.
[capabilities description]
[list of available tools — auto-discovered from MCP]

CURRENT GOAL: "{user's goal in their exact words}"

Interpret this goal and use your tools to:
1. Monitor the scene for anything relevant to this goal
2. Proactively alert when something important happens
3. Respond naturally when the user asks questions
```

---

## CV Pipeline (Already Built)

| Component | Model | FPS | Size |
|-----------|-------|-----|------|
| Person Detection + Tracking | YOLO11n + ByteTrack | 15 FPS | 5.4 MB |
| Pose Estimation | MediaPipe PoseLandmarker Lite | 30 FPS (async) | 5.6 MB |
| Depth Estimation | Depth Anything V2 Small | 22 FPS | — |
| Object Detection | YOLO11n (80 COCO classes) | shared model | — |
| Activity Recognition | Geometric pose analysis | zero overhead | — |

### 80 COCO Object Classes
person, bicycle, car, motorcycle, airplane, bus, train, truck, boat, traffic light, fire hydrant, stop sign, parking meter, bench, bird, cat, dog, horse, sheep, cow, elephant, bear, zebra, giraffe, backpack, umbrella, handbag, tie, suitcase, frisbee, skis, snowboard, sports ball, kite, baseball bat, baseball glove, skateboard, surfboard, tennis racket, bottle, wine glass, cup, fork, knife, spoon, bowl, banana, apple, sandwich, orange, broccoli, carrot, hot dog, pizza, donut, cake, chair, couch, potted plant, bed, dining table, toilet, tv, laptop, mouse, remote, keyboard, cell phone, microwave, oven, toaster, sink, refrigerator, book, clock, vase, scissors, teddy bear, hair drier, toothbrush

### Activity Classes (from pose landmarks)
standing, sitting, walking, running, fallen, lying_down, waving, reaching, crouching

---

## Interfaces

### Primary: Voice (Gemini Live)
- Real-time bidirectional audio via WebSocket
- User speaks goals, asks questions
- AEGIS speaks alerts, coaching, descriptions
- Voice Activity Detection (interruptions)
- Affective dialog (urgency matches situation)

### Secondary: Telegram Bot
- Async alerts with photos
- Remote control: /start, /stop, /goals
- Text-based interaction when voice isn't practical

### Dashboard (for demo)
- Real-time spatial state visualization
- Goal switching UI
- Tool call log (shows Claude's reasoning)
- Decision log (shows agent's choices)
- Session statistics

---

## Prize Strategy

| Priority | Prize | Our Angle |
|----------|-------|-----------|
| 1 | **Anthropic Human Flourishing** | Every use case = AI improving lives (elderly care, driver safety, posture, focus) |
| 2 | **Anthropic Best Use of Claude Agent SDK** | Deep: MCP server, hooks, subagents, sessions, tool use |
| 3 | **Greylock Best Multi-Turn Agent** | Goal-directed agent, persistent context, multi-turn reasoning |
| 4 | **Neo Most Likely to Become a Product** | Real product: hardware device + AI platform |
| 5 | **Google Cloud AI Track** | Gemini Live for voice interface |
| 6 | **Human Capital Fellowship** | $50K/member potential |
| Auto | Grand Prize, Most Creative, Most Impactful, Most Technically Complex | |

---

## File Structure

```
project-hoover/
├── aegis/
│   ├── __init__.py
│   ├── config.py           — All configuration
│   ├── main.py             — CLI entry point (local webcam mode)
│   ├── run_server.py       — FastAPI server entry point
│   ├── spatial_engine.py   — CV pipeline orchestrator
│   ├── agent.py            — Claude Agent (SDK-based)
│   ├── mcp_server.py       — MCP server with all 25 tools
│   ├── goals.py            — Goal presets + management
│   ├── activity.py         — Pose-based activity recognition
│   ├── voice.py            — Gemini Live voice integration
│   ├── telegram_bot.py     — Telegram interface
│   ├── monitor.py          — Proactive monitoring loop
│   ├── server.py           — FastAPI REST + WebSocket endpoints
│   └── static/
│       ├── index.html      — Device web app
│       ├── app.js          — Device app JS
│       └── dashboard.html  — Demo dashboard
├── src/
│   ├── models.py           — Data models (TrackedPerson, DetectedObject, etc.)
│   ├── perception.py       — YOLO + Pose + Depth + Object detection
│   ├── motion.py           — Velocity, trajectory prediction
│   └── risk.py             — Zone monitoring, TTC calculation
├── models/                 — ML model files
├── experiments/            — Experiment scripts (exp1-exp8)
├── data/                   — Runtime data (snapshots, etc.)
├── DESIGN.md               — This document
├── requirements.txt
└── .env                    — API keys (gitignored)
```

---

## Implementation Order

1. ~~CV Pipeline (experiments 1-8)~~ ✅
2. ~~Spatial Engine + Agent + Telegram~~ ✅
3. ~~Activity Recognition + Object Detection~~ ✅
4. ~~FastAPI Server + Dashboard~~ ✅
5. **MCP Server** (25 tools) ← NEXT
6. **Claude Agent SDK integration** (hooks, subagents)
7. **Gemini Live voice**
8. **End-to-end multi-goal demo**
9. **Demo polish**
