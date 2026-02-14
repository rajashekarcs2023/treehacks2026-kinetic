# AEGIS v4 — AI Skill Coach with Expert Motion Transfer

> **"Learn any physical skill from any expert — in real-time, through voice."**

---

## The Problem

Learning physical skills is broken:

| Problem | Impact |
|---------|--------|
| **Expert coaches cost $50-150/hour** | 90% of people can't afford regular coaching |
| **You can't see yourself** | Mirrors show you, but can't tell you what's wrong |
| **YouTube is one-directional** | You watch a tutorial, but it can't watch you back |
| **Fitness apps are pre-programmed** | Tempo knows squats, but can't coach a dance move or a golf swing |
| **Physical therapy compliance is ~30%** | People do exercises wrong at home, no one corrects them |
| **Skill transfer is inefficient** | An expert can show you, but articulating *what* makes their movement good is a separate skill most don't have |

## The Insight

Physical skill = **your skeleton doing the right thing at the right time.**

We have:
- 33-point skeleton tracking at 15 FPS ✅
- An LLM that can reason about joint angles and biomechanics ✅
- Real-time voice that can coach you while you move ✅

**The breakthrough:** Connect an expert's skeleton to your skeleton through AI reasoning. The AI watches both, understands the difference, and coaches you to close the gap — in real-time, for ANY skill.

---

## Product Vision

### How It Works

```
┌─────────────────────────────────────────────────────────┐
│                                                         │
│  STEP 1: Load an expert reference                       │
│  ┌──────────────────────────────────────────────────┐   │
│  │ • Record live demo: "Show me the movement"       │   │
│  │ • Upload a video: expert doing the skill          │   │
│  │ • YouTube URL: extract skeleton from any video    │   │
│  │ • Text only: "Coach my squat" (zero-shot)         │   │
│  └──────────────────────────────────────────────────┘   │
│                        ↓                                │
│  STEP 2: AI extracts skeleton sequence                  │
│  ┌──────────────────────────────────────────────────┐   │
│  │ • 33 landmarks per frame                          │   │
│  │ • Normalized (position/scale invariant)           │   │
│  │ • Key joint angles computed                       │   │
│  │ • Movement phases detected (start → peak → end)   │   │
│  └──────────────────────────────────────────────────┘   │
│                        ↓                                │
│  STEP 3: You perform the movement                       │
│  ┌──────────────────────────────────────────────────┐   │
│  │ • Live camera captures your skeleton              │   │
│  │ • Temporally aligned to expert via DTW            │   │
│  │ • Per-joint deviation computed in real-time        │   │
│  │ • Overall similarity score: 0-100%                │   │
│  └──────────────────────────────────────────────────┘   │
│                        ↓                                │
│  STEP 4: AI coaches you through voice                   │
│  ┌──────────────────────────────────────────────────┐   │
│  │ • Claude reasons about WHY the deviation matters  │   │
│  │ • Prioritizes most important correction first     │   │
│  │ • Gemini Live speaks it naturally in real-time    │   │
│  │ • "Lift your elbow 15° — you're dropping it on    │   │
│  │    the downswing"                                 │   │
│  │ • Tracks your improvement: "Better! 73% → 81%"   │   │
│  └──────────────────────────────────────────────────┘   │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

### The "Any Skill" Promise

Because Claude reasons from first principles about biomechanics (not from a training dataset), it works for skills it's never been explicitly programmed for:

| Skill Category | Examples | Key Landmarks |
|---|---|---|
| **Fitness** | Squats, deadlifts, pushups, planks, lunges | Hip-knee-ankle chain, spine angle |
| **Yoga** | Warrior, tree, downward dog, crow | Balance, symmetry, joint angles |
| **Dance** | Salsa, hip-hop, ballet positions | Full body coordination, rhythm |
| **Martial Arts** | Punches, kicks, stances, kata | Weight distribution, rotation |
| **Sports** | Tennis serve, golf swing, batting stance | Kinetic chain, rotation sequence |
| **Music** | Guitar posture, piano hand position, drumming | Wrist angle, finger spread, posture |
| **PT/Rehab** | Knee exercises, shoulder mobility, gait training | Specific joint ROM, compensation detection |
| **Ergonomics** | Typing posture, lifting form, desk setup | Spine, shoulders, wrist angles |
| **Sign Language** | ASL signs, finger spelling | Hand position, arm angles |

---

## Technical Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                     AEGIS SKILL COACH                         │
│                                                              │
│  ┌────────────┐   ┌──────────────────┐   ┌───────────────┐  │
│  │  Camera     │──▶│  Pose Pipeline    │──▶│ Live Skeleton  │  │
│  │  (15 FPS)   │   │  YOLO + MediaPipe │   │ 33 landmarks   │  │
│  └────────────┘   └──────────────────┘   └───────┬───────┘  │
│                                                   │          │
│  ┌────────────────────────────────────┐          │          │
│  │  Reference Skeleton Store          │          │          │
│  │  • Pre-recorded expert sequences   │          │          │
│  │  • Uploaded video extractions      │          │          │
│  │  • Phase-segmented movements       │          │          │
│  └──────────────┬─────────────────────┘          │          │
│                 │                                 │          │
│                 ▼                                 ▼          │
│  ┌──────────────────────────────────────────────────────┐   │
│  │              POSE COMPARISON ENGINE                    │   │
│  │                                                       │   │
│  │  1. Normalize both skeletons (translation + scale)    │   │
│  │  2. Temporal alignment via Dynamic Time Warping       │   │
│  │  3. Per-joint angle deviation (degrees)               │   │
│  │  4. Weighted similarity score (0-100%)                │   │
│  │  5. Phase detection (preparation → execution → follow)│   │
│  │  6. Movement quality metrics (smoothness, symmetry)   │   │
│  └──────────────────────────┬────────────────────────────┘   │
│                             │                                │
│                             ▼                                │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │                  CLAUDE AGENT (BRAIN)                    │ │
│  │                                                         │ │
│  │  • Receives: deviation data, similarity score, phase    │ │
│  │  • Reasons: "Which correction has biggest impact?"      │ │
│  │  • Decides: coaching priority, encouragement timing     │ │
│  │  • Remembers: past attempts, improvement trajectory     │ │
│  │  • 30+ MCP tools for full skill coaching stack          │ │
│  └────────────────────────┬────────────────────────────────┘ │
│                           │                                  │
│                           ▼                                  │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │               GEMINI LIVE (VOICE)                        │ │
│  │                                                         │ │
│  │  "Great depth on that squat! Now focus on keeping       │ │
│  │   your knees tracking over your toes — they're          │ │
│  │   caving in about 10 degrees. Push them out.            │ │
│  │   That's 78% match — up from 71% last rep!"            │ │
│  └─────────────────────────────────────────────────────────┘ │
│                                                              │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │  VISUAL FEEDBACK (Browser/Phone)                         │ │
│  │  • Side-by-side: your skeleton vs expert skeleton       │ │
│  │  • Joints color-coded: green=good, yellow=close, red=off│ │
│  │  • Similarity score overlay                             │ │
│  │  • Rep counter, improvement graph                       │ │
│  └─────────────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────────────┘
```

---

## Pose Comparison Engine — Technical Deep Dive

### 1. Skeleton Normalization

Raw MediaPipe landmarks are in pixel coordinates — we need position and scale invariance.

```python
# Normalize: translate to hip center, scale by torso length
def normalize_skeleton(landmarks_33):
    # 1. Center on hip midpoint (landmarks 23, 24)
    hip_center = midpoint(landmarks[23], landmarks[24])
    centered = [lm - hip_center for lm in landmarks]

    # 2. Scale by torso length (hip center to shoulder midpoint)
    shoulder_center = midpoint(centered[11], centered[12])
    torso_length = distance(centered[23], shoulder_center)
    scaled = [lm / torso_length for lm in centered]

    return scaled  # Now position/scale invariant
```

### 2. Joint Angle Computation

Raw landmark comparison is noisy. Joint angles are stable and meaningful.

```python
KEY_ANGLES = {
    "left_elbow":    (11, 13, 15),  # shoulder-elbow-wrist
    "right_elbow":   (12, 14, 16),
    "left_shoulder":  (13, 11, 23),  # elbow-shoulder-hip
    "right_shoulder": (14, 12, 24),
    "left_hip":      (11, 23, 25),  # shoulder-hip-knee
    "right_hip":     (12, 24, 26),
    "left_knee":     (23, 25, 27),  # hip-knee-ankle
    "right_knee":    (24, 26, 28),
    "spine":         (0, mid(11,12), mid(23,24)),  # head-shoulders-hips
    "neck_tilt":     (mid(11,12), 0, vertical),  # head tilt
}

def compute_angle(a, b, c):
    """Angle at point b formed by points a-b-c, in degrees."""
    ba = a - b
    bc = c - b
    cosine = dot(ba, bc) / (norm(ba) * norm(bc))
    return degrees(arccos(clip(cosine, -1, 1)))
```

### 3. Similarity Scoring

```python
def compute_similarity(user_angles, expert_angles, weights=None):
    """
    Compare user's joint angles to expert's.
    Returns:
      - overall_score: 0-100%
      - per_joint_deviation: dict of {joint: degrees_off}
      - worst_joints: sorted list of biggest deviations
    """
    deviations = {}
    for joint in KEY_ANGLES:
        diff = abs(user_angles[joint] - expert_angles[joint])
        deviations[joint] = diff

    # Weighted score: each joint contributes based on importance
    if weights is None:
        weights = {j: 1.0 for j in KEY_ANGLES}

    total_weight = sum(weights.values())
    weighted_error = sum(
        weights[j] * min(deviations[j] / 45.0, 1.0)  # 45° = max error
        for j in KEY_ANGLES
    )
    score = max(0, 100 * (1 - weighted_error / total_weight))

    worst = sorted(deviations.items(), key=lambda x: -x[1])

    return {
        "score": round(score, 1),
        "deviations": deviations,
        "worst_joints": worst[:3],  # top 3 issues
    }
```

### 4. Dynamic Time Warping (for movement sequences)

For movements that happen over time (a squat, a dance move), we need to align the user's timing to the expert's timing.

```python
import numpy as np
from scipy.spatial.distance import cdist

def dtw_align(user_sequence, expert_sequence):
    """
    Align two skeleton sequences using Dynamic Time Warping.

    user_sequence:   list of normalized skeletons (one per frame)
    expert_sequence: list of normalized skeletons (one per frame)

    Returns: list of (user_frame, expert_frame) pairs
    """
    n, m = len(user_sequence), len(expert_sequence)

    # Cost matrix: pairwise skeleton distance
    cost = np.zeros((n, m))
    for i in range(n):
        for j in range(m):
            cost[i, j] = skeleton_distance(user_sequence[i], expert_sequence[j])

    # DTW accumulation
    D = np.full((n + 1, m + 1), np.inf)
    D[0, 0] = 0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            D[i, j] = cost[i-1, j-1] + min(D[i-1, j], D[i, j-1], D[i-1, j-1])

    # Backtrack to get alignment path
    path = []
    i, j = n, m
    while i > 0 and j > 0:
        path.append((i-1, j-1))
        candidates = [D[i-1, j-1], D[i-1, j], D[i, j-1]]
        argmin = np.argmin(candidates)
        if argmin == 0:   i, j = i-1, j-1
        elif argmin == 1: i = i-1
        else:             j = j-1

    return list(reversed(path))
```

### 5. Movement Phase Detection

Segment a movement into phases for better coaching.

```python
PHASE_TYPES = ["preparation", "execution", "peak", "recovery", "rest"]

def detect_phases(skeleton_sequence, key_joint="left_knee"):
    """
    Detect movement phases based on a key joint's angle over time.
    E.g., for squats: knee angle goes 170° → 90° → 170°

    Returns: list of (start_frame, end_frame, phase_name)
    """
    angles = [compute_angle_for_joint(s, key_joint) for s in skeleton_sequence]

    # Find peaks and valleys
    # Peak angle = standing (preparation/recovery)
    # Valley angle = bottom of movement (peak of effort)
    # Transitions = execution phases

    phases = []
    # ... peak/valley detection logic
    return phases
```

---

## New MCP Tools (Skill Coaching)

In addition to the existing 25 tools, add these skill-specific tools:

### Reference Management (4 tools)

| # | Tool | Description |
|---|------|-------------|
| 1 | `record_reference_start` | Start recording a reference movement from live camera. Records skeleton sequence until stopped. |
| 2 | `record_reference_stop` | Stop recording. Normalizes, detects phases, stores as named reference. Returns frame count and duration. |
| 3 | `load_reference_from_video` | Extract skeleton sequence from a video file. Processes offline, stores as reference. |
| 4 | `list_references` | List all stored reference movements with metadata (name, duration, frame count, phases). |

### Comparison & Analysis (5 tools)

| # | Tool | Description |
|---|------|-------------|
| 5 | `compare_to_reference` | Compare current live pose to a stored reference (single frame). Returns per-joint deviations, similarity score, worst joints. |
| 6 | `compare_movement_sequence` | Compare a live movement sequence to a reference using DTW. Returns phase-by-phase scores. |
| 7 | `get_joint_deviation` | Get specific joint angle deviation from reference. E.g., "How far off is my left knee from expert?" |
| 8 | `get_movement_quality` | Compute smoothness, symmetry, tempo consistency metrics for the current movement. |
| 9 | `detect_compensation` | Check if user is compensating for weakness (e.g., leaning to one side during a squat to favor strong leg). |

### Coaching State (4 tools)

| # | Tool | Description |
|---|------|-------------|
| 10 | `start_coaching_session` | Begin a coaching session for a specific skill. Initializes rep counter, score tracking. |
| 11 | `get_coaching_progress` | Get improvement over current session: score per rep, trend, best/worst reps. |
| 12 | `count_reps` | Get current rep count for repetitive movements (auto-detected from phase cycles). |
| 13 | `end_coaching_session` | End session, return summary: total reps, average score, improvement, time spent. |

### Zero-Shot Skill (2 tools)

| # | Tool | Description |
|---|------|-------------|
| 14 | `analyze_skill_from_description` | Given a natural language skill description, Claude reasons about which joints/angles matter most and what "good form" looks like. No reference needed. |
| 15 | `get_skill_knowledge` | Search Claude's knowledge for biomechanics advice about a specific skill (e.g., "common mistakes in deadlift"). |

**Total tools: 25 (existing) + 15 (skill) = 40 tools**

Or — we can replace some existing tools that are less relevant now (desk-watch zone tools, etc.) and keep it tighter.

---

## Updated Goal Presets

| ID | Name | Use Case | Key Tools |
|---|---|---|---|
| `skill_coach` | **Skill Coach** | Real-time movement coaching with expert comparison | compare_to_reference, get_joint_deviation, speak_to_user |
| `pt_rehab` | **PT & Rehab** | Physical therapy exercise guidance and compliance | compare_movement_sequence, count_reps, detect_compensation |
| `fitness_trainer` | **Fitness Trainer** | Workout form correction, rep counting, encouragement | analyze_posture, count_reps, get_coaching_progress |
| `dance_teacher` | **Dance Teacher** | Learn choreography, match expert movements | compare_movement_sequence, get_movement_quality |
| `sports_coach` | **Sports Coach** | Technique analysis for any sport | compare_to_reference, analyze_skill_from_description |
| `zero_shot` | **Any Skill** | Describe any skill, AI figures it out | analyze_skill_from_description, get_pose_landmarks |

Plus keep: `general` for spatial awareness, `accessibility` for blind/low-vision guidance.

---

## The Demo (2 minutes)

### Scene 1: "Expert Transfer" (40 sec)

1. Show a video of an expert doing a perfect squat (or record one live)
2. AEGIS extracts the skeleton: "Got it — I've learned this movement. 15 frames, 3 phases."
3. User attempts the squat
4. Screen shows: **side-by-side skeletons**, joints color-coded green/yellow/red
5. AEGIS speaks: *"Good depth! But your knees are caving in 12 degrees — push them out over your toes. That was 74% match."*
6. User does another rep
7. AEGIS: *"Much better! 83%. Knees are tracking well now. Watch your forward lean — keep your chest up."*

### Scene 2: "Zero-Shot Skill" (30 sec)

1. User says: *"Coach my tennis serve"*
2. No reference video — just voice command
3. AEGIS: *"I'll watch for shoulder rotation, elbow extension, wrist snap, and weight transfer. Show me a serve."*
4. User mimes a tennis serve
5. AEGIS: *"Your shoulder rotation is limited — you're using too much arm. Rotate your torso more. Your toss arm is good but release the ball higher."*

### Scene 3: "Accessibility" (20 sec)

1. Switch to accessibility mode
2. User puts on blindfold
3. AEGIS narrates the room: *"Clear path ahead. Chair on your left, 4 feet. Person standing to your right, 6 feet."*
4. User navigates successfully

### Scene 4: "Architecture" (30 sec)

1. Show dashboard: real-time tool calls, Claude's reasoning, similarity scores
2. "40 MCP tools. Claude picks the right ones dynamically. Gemini Live speaks naturally."
3. "Works on a webcam. No special hardware. No training data per skill."
4. **"We built the first AI that can teach you any physical skill, from any expert, in real-time."**

---

## Visual Feedback (Browser Overlay)

```
┌────────────────────────────────────────────────┐
│                                                │
│   ┌──────────────┐    ┌──────────────┐         │
│   │  YOUR POSE    │    │  EXPERT POSE  │         │
│   │  (live)       │    │  (reference)  │         │
│   │              │    │              │         │
│   │   🟢 ← head  │    │   ⚪ ← head  │         │
│   │  🟢  🟢      │    │  ⚪  ⚪      │         │
│   │   │          │    │   │          │         │
│   │  🟡  🟡      │    │  ⚪  ⚪      │  ← arms │
│   │   │          │    │   │          │         │
│   │  🔴  🔴      │    │  ⚪  ⚪      │  ← knees│
│   │  │    │      │    │  │    │      │         │
│   └──────────────┘    └──────────────┘         │
│                                                │
│   SIMILARITY: ████████░░ 81%     REP: 3/10     │
│   Focus: Push knees out over toes              │
│   Trend: 74% → 78% → 81% ↑ improving          │
│                                                │
└────────────────────────────────────────────────┘
```

Joint colors:
- 🟢 Green: within 10° of expert
- 🟡 Yellow: 10-25° off
- 🔴 Red: >25° off

---

## Prize Alignment

| Prize | How This Wins |
|-------|--------------|
| **Human Flourishing (Anthropic)** | Democratizes expert coaching. A kid in rural India learns yoga from a world-class instructor's skeleton. Physical therapy at home actually works. |
| **Claude Agent SDK** | 40 MCP tools, dynamic tool selection, multi-turn coaching conversations, memory of your progress |
| **Greylock Multi-Turn Agent** | Coaching IS multi-turn. Agent remembers your weak points, adapts coaching over reps/sessions |
| **Neo Most Likely to Become a Product** | $50B physical therapy market. $100B fitness market. Clear product-market fit. |
| **Most Technically Complex** | DTW + pose normalization + zero-shot biomechanics reasoning + real-time voice coaching |
| **Most Impactful** | Makes expert coaching free and universal |

---

## What We Keep vs. What Changes

### Keep (already built, still essential)
- ✅ CV Pipeline (YOLO + MediaPipe + ByteTrack) — the eyes
- ✅ MCP Server framework — extends to 40 tools
- ✅ Claude Agent — the brain, now focused on coaching
- ✅ Gemini Live voice — the coach's voice
- ✅ FastAPI server — delivery platform
- ✅ Dynamic goal system — now with skill-focused presets
- ✅ Phone web app — becomes the coaching interface

### Build New
- 🔨 **Pose Comparison Engine** — normalize, angle computation, similarity scoring
- 🔨 **Reference Store** — record, store, load expert skeletons
- 🔨 **DTW Alignment** — temporal matching of movement sequences
- 🔨 **Phase Detection** — segment movements into phases
- 🔨 **15 new MCP tools** — skill coaching specific
- 🔨 **Visual overlay** — side-by-side skeleton + color-coded joints
- 🔨 **New goal presets** — skill_coach, pt_rehab, fitness_trainer, etc.

### Remove/Deprioritize
- ~~Desk Guardian goal~~ → not relevant
- ~~Driver Monitor goal~~ → not relevant
- ~~Zone management tools~~ → keep but lower priority
- ~~Telegram alerts~~ → keep for progress reports, not primary interface

---

## Implementation Order

| # | Task | Est. Time | Priority |
|---|------|-----------|----------|
| 1 | Pose Comparison Engine (normalize + angles + similarity) | 2-3 hours | **Critical** |
| 2 | Reference Store (record + save + load skeleton sequences) | 1-2 hours | **Critical** |
| 3 | 15 new MCP tools for skill coaching | 2-3 hours | **Critical** |
| 4 | Updated goal presets (skill_coach, pt_rehab, etc.) | 1 hour | **Critical** |
| 5 | Visual overlay (side-by-side, color-coded joints) | 2-3 hours | **High** |
| 6 | DTW alignment for movement sequences | 1-2 hours | **High** |
| 7 | Phase detection | 1 hour | **Medium** |
| 8 | Demo choreography and testing | 2-3 hours | **High** |
| **Total** | | **~12-17 hours** | |

---

## The One-Liner

> **AEGIS: The first AI that can watch an expert, understand their movement, and teach you to do it — in real-time, through voice, for any physical skill.**
