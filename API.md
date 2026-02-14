# AEGIS API Reference — For Next.js Frontend

**Base URL:** `http://localhost:8000`

---

## Core APIs

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/state` | Current spatial state (people, objects, poses) |
| GET | `/api/summary` | Human-readable spatial summary |
| GET | `/api/config` | Client config (Gemini API key, model, voice) |
| GET | `/api/voice/status` | Gemini Live voice bridge status |

## Goal Management

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/goals` | List all 12 goals + active goal ID |
| POST | `/api/goals/{goal_id}` | Set active goal (skill_coach, fitness_trainer, pt_rehab, etc.) |

**Goal IDs:** `desk_watch`, `posture_coach`, `driver_monitor`, `study_focus`, `elderly_care`, `general`, `skill_coach`, `pt_rehab`, `fitness_trainer`, `dance_teacher`, `sports_coach`, `zero_shot_coach`

## Agent

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/agent/message` | Send message to Claude agent. Body: `{ "message": "Coach me on squats" }` |
| GET | `/api/agent/status` | Agent status (goal, tool calls count, conversation turns) |
| GET | `/api/logs/tools` | Recent tool call log (last 50) |
| GET | `/api/logs/decisions` | Recent agent decisions (last 50) |

## Coaching Session

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/coaching/start` | Start session. Body: `{ "skill_name": "squat", "reference_name?": "perfect_squat", "primary_angle?": "left_knee" }` |
| POST | `/api/coaching/stop` | End session. Returns final summary with reps, scores, trend. |
| GET | `/api/coaching/status` | Is session active? Skill name, frame count, rep count. |
| GET | `/api/coaching/progress` | Detailed: reps, avg/best/worst score, trend, top corrections. |
| GET | `/api/coaching/score` | Current similarity score vs expert reference (single frame). |
| GET | `/api/coaching/quality` | Movement quality: smoothness, symmetry, tempo. |
| GET | `/api/coaching/compensation` | Compensation pattern detection (injury risk). |
| GET | `/api/coaching/angles` | All 10 joint angles + expert angles if reference loaded. |

**Available angles:** `left_elbow`, `right_elbow`, `left_shoulder`, `right_shoulder`, `left_hip`, `right_hip`, `left_knee`, `right_knee`, `left_ankle`, `right_ankle`

## Expert References

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/references` | List all stored expert references. |
| GET | `/api/references/{name}` | Get reference metadata (frames, duration, phases). |
| POST | `/api/references/record/start?name=squat` | Start recording expert reference from live camera. |
| POST | `/api/references/record/stop` | Stop recording & save. Body: `{ "key_angle?": "left_knee" }` |
| DELETE | `/api/references/{name}` | Delete a reference. |

## Skill Progression Graphs

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/graphs` | List all skill graphs (fitness, yoga, pt_rehab_knee, etc.) |
| GET | `/api/graphs/{name}` | Full skill tree: nodes + links + progress (for visualization). |
| GET | `/api/graphs/{name}/recommend?top_n=3` | AI-recommended next skills to practice. |
| GET | `/api/graphs/{name}/progress` | Overall progress: mastered, in_progress, completion %. |
| POST | `/api/graphs/{name}/skills/{skill_id}/update?score=85` | Update skill proficiency after session. |

**Built-in graphs:** `fitness` (13 skills), `yoga` (10), `pt_rehab_knee` (7), `pt_rehab_shoulder` (4)

## Training Data

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/training/stats` | Dataset stats: samples per skill, avg scores. |
| GET | `/api/training/export?skill=squat&pad_to=60` | Export for model training: X (N×T×10), y (N,). |
| DELETE | `/api/training/{skill}` | Clear training data for a skill. |

## Local Model & Hybrid Inference

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/model/train?skill=squat&epochs=50` | Train local 1D CNN from collected data. |
| GET | `/api/model/status` | Model status: trained, param count, training data stats. |
| GET | `/api/model/predict` | Instant local model prediction (~0.2ms). |
| GET | `/api/model/hybrid` | Hybrid score: 60% local + 40% Claude. |

## WebSockets

### `/ws/video` — Camera Frame Streaming
```
Client → Server: { "type": "frame", "data": "<base64 JPEG>" }
Server → Client: { "type": "state", "data": <spatial_state> }
```

### `/ws/audio` — Gemini Live Voice
```
Client → Server: { "type": "audio", "data": "<base64 PCM 16kHz>" }
Server → Client: { "type": "audio", "data": "<base64 PCM 24kHz>" }
Client → Server: { "type": "text", "data": "message" }
Client → Server: { "type": "spatial", "data": <state> }
```

### `/ws/coaching` — Real-Time Coaching Data
```
Server → Client: {
  "type": "coaching",
  "data": {
    "similarity_score": 85.2,
    "per_joint_deviation": { "left_knee": 12.3, ... },
    "worst_joints": [["left_knee", 12.3], ...],
    "best_joints": [["right_elbow", 1.2], ...]
  },
  "reps": 5,
  "frame": 142
}
```

---

## Typical Frontend Flow

### 1. Skill Selection Screen
```
GET /api/goals → show coaching goals as cards
GET /api/graphs → show available skill trees
GET /api/graphs/fitness/recommend → show "recommended next"
GET /api/references → show available expert references
```

### 2. Start Coaching
```
POST /api/goals/skill_coach → activate coaching goal
POST /api/coaching/start { skill_name, reference_name?, primary_angle? }
Connect to ws://localhost:8000/ws/coaching → real-time scores
Connect to ws://localhost:8000/ws/video → send camera frames
Connect to ws://localhost:8000/ws/audio → voice coaching
```

### 3. During Coaching (poll or use WebSocket)
```
ws/coaching → continuous score updates
GET /api/coaching/angles → joint angle display
GET /api/coaching/quality → smoothness/symmetry
GET /api/model/predict → instant local model score
```

### 4. End Session
```
POST /api/coaching/stop → get final summary
POST /api/graphs/fitness/skills/bodyweight_squat/update?score=85
GET /api/graphs/fitness → updated skill tree
GET /api/training/stats → check data collection
```

### 5. Train Local Model (after enough data)
```
GET /api/training/stats → check sample count
POST /api/model/train?epochs=50 → train CNN
GET /api/model/status → verify trained
```
