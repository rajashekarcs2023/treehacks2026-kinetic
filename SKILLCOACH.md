# AEGIS Skill Coach — The Complete Feature Map

> **"Show me any movement. I'll learn it. Then I'll teach you — in real-time, through voice. And I get smarter every rep."**

Last updated: Feb 14, 2026, 2:10 AM PST

---

## Core Layers (must-build)

### Layer 1: Expert Motion Transfer
**"Learn any movement from any source"**

- Record a live demo → extract skeleton sequence
- Upload a video file → process and extract skeleton
- Paste a YouTube URL → download, extract, store
- Normalize skeletons (position + scale invariant, centered on hip)
- Compute key joint angles (10+ angles: elbows, shoulders, hips, knees, spine)
- Detect movement phases (preparation → execution → peak → recovery)
- Store as named reference with metadata

### Layer 2: Real-Time Pose Comparison
**"See exactly how you differ from the expert"**

- DTW (Dynamic Time Warping) to align user movement to expert timing
- Per-joint angle deviation in degrees
- Weighted similarity score 0-100% (important joints weighted higher)
- Phase-by-phase scoring (you might nail the descent but botch the recovery)
- Compensation detection — "you're leaning left to avoid weak right knee"

### Layer 3: Claude Agent Coaching (the BRAIN)
**"AI that reasons about WHY your form is off, not just THAT it's off"**

- 40 MCP tools (25 existing spatial + 15 new skill tools)
- Claude sees: deviation data, similarity scores, phase info, skill graph
- Claude reasons: "Which correction has the biggest impact right now?"
- Prioritizes corrections — doesn't overwhelm with 5 things at once
- Adapts coaching style per skill (encouraging for beginners, precise for advanced)
- Remembers your history across reps and sessions

### Layer 4: Gemini Live Voice (the VOICE)
**"Speaks corrections while you move — hands-free, eyes-free"**

- Real-time bidirectional audio via Gemini Live
- Spatial context injected (your current pose, score, phase)
- Natural coaching tone, not robotic
- Interruption handling — stops speaking when you're mid-movement
- "Lift your elbow 15° — you're dropping it on the downswing. That's 83%, up from 74!"

---

## Innovation Layers (differentiators — what TherapEase couldn't do)

### Layer 5: Zero-Shot Skill Understanding
**"Coach any skill without training data or expert video"**

- User says: "Coach my tennis serve"
- Claude reasons from biomechanics knowledge: "I need shoulder rotation, elbow extension, wrist snap, weight transfer"
- Maps natural language → relevant joint angles to monitor
- Works for ANY described physical skill
- No reference video needed — Claude's knowledge IS the reference

### Layer 6: Document → Skill Conversion
**"Turn any PT protocol, yoga manual, or coaching guide into live coaching"**

- Upload PDF / paste URL / paste text of any physical skill documentation
- Claude extracts: target joint angles, phases, rep counts, safety boundaries
- Generates a "Virtual Expert Reference" — same format as video-extracted reference
- Works with PT protocols ("3 sets of 10 knee extensions, 45° ROM")
- Works with illustrated guides (future: VLM on DGX Spark reads diagrams)
- **Unlocks every documented physical skill ever written down**

### Layer 7: Self-Training Local Model
**"The system gets smarter as you use it"**

- Every coached rep = labeled training data:
  - skeleton sequence + quality score + specific corrections from Claude
- After ~50 reps, train a tiny 1D CNN (10KB, trains in 5 seconds on CPU)
- Local model provides INSTANT quality scoring at 15 FPS (no API call)
- Claude still does deep reasoning periodically, but local model handles continuous feedback
- **Demo moment:** "The first 5 reps, Claude coaches each one. By rep 6, the system trained itself. Now feedback is instant."
- Flywheel: more reps → more data → better model → better coaching → more reps

### Layer 8: Skill Progression Graph
**"Coaches in the right ORDER — fixes root causes, not symptoms"**

Inspired by OpenCortex's PageRank-based evaluation layer.

- Track correction dependencies:
  ```
  ankle_mobility → knee_tracking → squat_depth → full_mastery
  ```
- Claude updates the graph after each rep
- System prioritizes coaching the **foundation** first
- "Your knee cave is caused by ankle mobility. Let's fix that first."
- Visualize as a skill tree with scores per node
- Progress tracking: "Ankle mobility: 58% → 72% this session"
- Prevents the "random corrections" problem — coaching has a strategy

### Layer 9: Auto-Drill Generation
**"Detects your weak spots and creates targeted micro-exercises"**

Inspired by Paradigm's auto-workflow creation from observed patterns.

- System notices you consistently fail at the same point in a movement
- Auto-generates a focused drill: "I noticed you drop your elbow at the peak. Do just the peak portion 10 times."
- Isolates the problematic phase and creates a mini-exercise
- Tracks drill completion and re-tests the full movement
- **The AI becomes your personal trainer with a training plan, not just a mirror**

---

## Visual Feedback (browser/phone overlay)

### Side-by-Side Skeleton View
```
┌──────────────┐    ┌──────────────┐
│  YOUR POSE    │    │  EXPERT POSE  │
│  (live)       │    │  (reference)  │
│   🟢 head     │    │   ⚪ head     │
│  🟢  🟢 arms  │    │  ⚪  ⚪      │
│  🟡  🟡 hips  │    │  ⚪  ⚪      │
│  🔴  🔴 knees │    │  ⚪  ⚪      │
└──────────────┘    └──────────────┘
```

### Color-Coded Joints
- 🟢 Green: within 10° of expert
- 🟡 Yellow: 10-25° off
- 🔴 Red: >25° off

### Score Bar + Rep Counter
```
SIMILARITY: ████████░░ 81%     REP: 3/10
Focus: Push knees out over toes
Trend: 74% → 78% → 81% ↑ improving
```

### Skill Graph Visualization
```
Full Squat Mastery ░░░░░░░░░░ 68%
├── Depth         ████████░░ 88% ✅
├── Knee Tracking ██████░░░░ 71% ⚠️ ← working on this
│   └── Ankle Mobility ████░░░░░░ 58% ❌ ← root cause
└── Chest Position████████░░ 85% ✅
```

### Improvement Timeline
```
Session 1: ████░░░░░░ 42%
Session 2: ██████░░░░ 61%
Session 3: ████████░░ 78% ← you are here
```

---

## Three Input Paths (all lead to the same coaching experience)

```
PATH 1: Expert Video               PATH 2: Voice/Text          PATH 3: Document
"Learn from this video"             "Coach my squats"           "Follow this PT protocol"
        │                                   │                           │
        ▼                                   ▼                           ▼
   Extract skeleton              Claude reasons from              Claude extracts
   sequence from video           biomechanics knowledge           targets from text
        │                                   │                           │
        └───────────────┬───────────────────┘───────────────────────────┘
                        ▼
              Unified Reference Format
              (joint angles + phases + safety bounds)
                        │
                        ▼
              Same coaching experience:
              comparison → voice feedback → score → graph → self-training
```

---

## New MCP Tools (15 skill-specific)

### Reference Management (4)
| Tool | Description |
|------|-------------|
| `record_reference_start` | Start recording expert skeleton from live camera |
| `record_reference_stop` | Stop recording, normalize, detect phases, store |
| `load_reference_from_video` | Extract skeleton sequence from video file |
| `list_references` | List all stored references with metadata |

### Comparison & Analysis (5)
| Tool | Description |
|------|-------------|
| `compare_to_reference` | Single-frame comparison: deviations, score, worst joints |
| `compare_movement_sequence` | Full movement DTW comparison with phase scores |
| `get_joint_deviation` | Specific joint deviation from reference |
| `get_movement_quality` | Smoothness, symmetry, tempo consistency |
| `detect_compensation` | Check for compensatory movement patterns |

### Coaching State (4)
| Tool | Description |
|------|-------------|
| `start_coaching_session` | Begin session, init rep counter + score tracking |
| `get_coaching_progress` | Improvement trend, best/worst reps, per-rep scores |
| `count_reps` | Auto-detected rep count from phase cycles |
| `end_coaching_session` | Session summary: reps, avg score, improvement |

### Intelligence (2)
| Tool | Description |
|------|-------------|
| `parse_skill_document` | Extract skill definition from text/document |
| `get_skill_graph` | Current skill graph with dependencies and scores |

---

## Demo Script (2 minutes)

### Act 1: "I Can Learn" (30 sec)
- Play a video of a perfect squat (or record live)
- AEGIS: "Got it — 15 frames, 3 phases. I've learned this movement."
- Show the extracted skeleton on screen

### Act 2: "I Can Teach" (30 sec)
- User does a squat
- Side-by-side skeleton comparison, color-coded joints
- AEGIS speaks: "Good depth! Knees are caving 12° — push them out. 74% match."
- User does another: "Better! 83%. Knees tracking well. Watch your forward lean."

### Act 3: "I Get Smarter" (30 sec)
- "I've coached 6 reps. Let me train a model..."
- Switch to instant feedback — score updates on every frame, no API delay
- "The system just taught itself. Real-time feedback, zero latency."
- Show the skill graph: "Ankle mobility is your root cause. Let me generate a drill."

### Act 4: "Any Skill, Any Source" (30 sec)
- Voice: "Coach my dance move" → zero-shot, no video needed
- OR: Upload a PT protocol PDF → instant coaching from documentation
- Architecture reveal: show MCP tools, Claude reasoning, Gemini voice on dashboard
- **"Any expert. Any skill. Any person. Real-time. And it gets smarter every rep."**

---

## Prize Alignment

| Prize | How We Win It |
|-------|--------------|
| **Human Flourishing (Track)** | Democratizes expert coaching. Makes physical mastery accessible to everyone, everywhere. |
| **Claude Agent SDK (Sponsor)** | 40 MCP tools, dynamic skill reasoning, multi-turn coaching, memory, goal adaptation |
| **Greylock Multi-Turn Agent** | Coaching IS multi-turn. Tracks progress across reps, sessions, generates drills |
| **Neo Most Likely to Become a Product** | $50B PT + $100B fitness market. Clear product vision. "All you need is a webcam." |
| **Grand Prize** | Technical depth (DTW + self-training + skill graph) + emotional impact (accessibility) |
| **Most Impactful** | Expert coaching for free. PT compliance goes from 30% → 90%. |
| **Most Technically Complex** | DTW, pose normalization, self-training CNN, skill dependency graph, real-time voice |

---

## What Makes This Different from TherapEase (2024 winner)

| | TherapEase (2024) | AEGIS Skill Coach (2026) |
|--|---|---|
| Needs live trainer | **Yes** | **No** — learns from video/text/voice |
| Fixed exercises | Yes — pre-programmed PT only | **Any skill** — zero-shot |
| AI role | Chatbot (text Q&A) | **Autonomous agent** with 40 tools |
| Learns over time | No | **Yes** — self-training model |
| Coaching strategy | Random corrections | **Skill graph** — dependency-aware |
| Voice coaching | No | **Yes** — Gemini Live real-time |
| Expert transfer | No | **Yes** — from video, text, or voice |

---

## Build Priority Order

| # | Component | Est. Hours | Status |
|---|-----------|-----------|--------|
| 1 | Pose Comparison Engine (normalize + angles + similarity) | 2-3h | 🔲 |
| 2 | Reference Store (record + save + load) | 1-2h | 🔲 |
| 3 | DTW Alignment | 1-2h | 🔲 |
| 4 | 15 new MCP tools | 2-3h | 🔲 |
| 5 | Skill Progression Graph | 2h | 🔲 |
| 6 | Data Collection Pipeline | 1h | 🔲 |
| 7 | Local Model Training (SkillScorer CNN) | 2h | 🔲 |
| 8 | Hybrid Inference (local + Claude) | 1h | 🔲 |
| 9 | Visual Overlay (side-by-side, color-coded, score) | 2-3h | 🔲 |
| 10 | Goal Presets + Demo Script | 2h | 🔲 |
| | **Total** | **~16-20h** | |
