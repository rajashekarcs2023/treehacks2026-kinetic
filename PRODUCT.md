# AEGIS — AI Skill Coach

**Master any physical skill with AI — rehab recovery, yoga, sign language, and beyond.**

---

## The Problem

### Physical therapy is broken
- A single PT session costs **$150-$350**. The average recovery requires 20+ sessions.
- **60% of patients quit** their prescribed PT program before completing it — primarily due to cost and access.
- Rural and underserved communities often have **zero physical therapists** within 50 miles.
- Elderly patients (65+) face the highest risk: **1 in 4 falls each year** (36 million falls/year in the US), yet mobility exercises that prevent falls require consistent supervised practice most can't afford.

### Skill learning is gatekept by cost and geography
- A personal yoga instructor costs **$75-$150/hour**.
- Sign language tutors charge **$50-$100/hour** — and there aren't enough of them.
- Dance, sports, and music lessons require traveling to a studio, scheduling around an instructor's availability, and paying per session.
- YouTube tutorials show you what to do but **can't tell you what you're doing wrong**.

### The core gap
There is no affordable way to get **real-time, personalized, spoken feedback** on your physical form — the kind of feedback a human coach gives when they watch you move and say "straighten your knee" or "rotate your hip more."

---

## The Solution

AEGIS is an **AI-powered physical skill coach** that watches you through your phone or laptop camera and coaches you in real-time through voice — just like having a personal trainer, physical therapist, or instructor in the room.

### How it works

1. **Pick a skill** — Choose from 50+ skills across physical therapy, yoga, tai chi, sign language, elderly mobility, ergonomics, dance, fitness, sports, and music.

2. **Choose your learning mode:**
   - **Expert Video** — Paste a YouTube URL or upload a video. AI extracts the expert's skeleton and coaches you to match their form.
   - **Describe It** — Just tell the AI what you want to practice. "I'm recovering from ACL surgery and need to do knee extensions." AI generates the ideal form targets.
   - **From a Document** — Upload a PT exercise sheet, yoga guide, or instruction PDF. AI extracts the movement cues.
   - **Practice with a Friend** — Create a room, share a code. Both practice together with live scores. AI coaches both and compares performance.

3. **AI coaches you live:**
   - Camera captures your movement in real-time
   - Computer vision extracts your skeleton (33 body landmarks)
   - AI compares your form to the correct form
   - You hear spoken corrections: "Bend your front knee more" / "Great form, keep it up!"
   - You see your live score, rep count, and joint-by-joint feedback

---

## Features

### Real-Time Pose Analysis
- 33-point skeleton tracking via MediaPipe at 30 FPS
- **21-point hand tracking** via MediaPipe HandLandmarker (per hand, for sign language)
- Person detection and multi-person tracking (YOLO11n + ByteTrack)
- Joint angle computation (10 key angles: shoulders, elbows, hips, knees, torso, neck)
- Skeleton normalization for position/scale-invariant comparison
- **NVIDIA DGX Spark** — 133-keypoint whole-body pose (body+hands+face) via GB10 GPU
- **Camera coverage detection** — warns when landmarks are out of frame or occluded

### Intelligent Coaching (Claude Agent SDK)
- **3 specialized AI sub-agents** orchestrated by a main agent:
  - **Perception Agent** — Analyzes what it sees (posture, alignment, hands, scene, landmark visibility)
  - **Coach Agent** — Provides real-time movement corrections and rep tracking
  - **Progress Agent** — Manages goals, skill progression, and memory
- **46 MCP tools** the agents can call (pose comparison, quality analysis, compensation detection, hand landmarks, DGX inference, etc.)
- **Safety hooks** — blocks dangerous actions during coaching
- **Audit hooks** — logs every AI decision for transparency
- Multi-turn reasoning: the AI adapts its coaching based on your progress across reps

### Natural Voice Coaching (OpenAI Realtime API)
- Bidirectional real-time voice via OpenAI GPT-4o Realtime API
- Natural voice (alloy) — not robotic TTS
- **3-Layer Interruption System** — AI pauses coaching when you speak, resumes after
- **Emotional Tone Adaptation** — scores dropping → gentle tone; new best → celebration
- **Proactive Check-ins** — 3+ declining reps → "Want a break?", 10-rep milestones
- **Multi-Language** — "Coach me in Spanish" → switches seamlessly
- **Session Memory** — "Last time you averaged 78, knee alignment was the issue"
- **Camera Coverage Warnings** — "I can't see your legs, try stepping back"
- You can talk back: ask questions, request different exercises, say "slower please"

### Expert Motion Transfer
- Paste any YouTube tutorial → AI extracts the expert's skeleton
- Your form is compared to the expert frame-by-frame using Dynamic Time Warping (DTW)
- Works with any speed — DTW handles timing differences
- Color-coded skeleton overlay shows which joints need adjustment

### Similarity Scoring (0-100)
- Weighted joint angle comparison
- Automatic rep counting via peak detection
- Phase detection: preparation → execution → peak → recovery
- Movement quality metrics: smoothness, symmetry, range of motion
- Compensation detection: flags when you're using wrong muscles (injury risk)

### Practice with a Friend (Multiplayer)
- Create a room → get a 6-character code → share with friend
- Both practice the same skill simultaneously
- Split-screen view: friend's live score on the left, your camera on the right
- Real-time leaderboard updates every 3 seconds
- "Friend is ahead!" / "You're winning!" competitive badges
- AI compares both players and provides comparative analysis

### Skill Progression
- Directed Acyclic Graph (DAG) for skill prerequisites
- Built-in skill trees: fitness (13 skills), yoga (10), PT knee rehab (7), PT shoulder (4)
- PageRank-inspired skill recommendations
- Progress tracking with trend analysis

### Hybrid Scoring
- Local 1D CNN model (14K parameters, NumPy-only, 0.15ms inference) handles 60% of scoring
- Claude handles 40% for nuanced assessment (triggered smartly, not every frame)
- Self-improving: collects training data from coaching sessions

---

## Use Cases

### 1. Post-Surgery Rehab at Home
**User:** Patient recovering from knee surgery
**Skill:** Knee Extension, Hip Flexion, Glute Bridge
**How:** Doctor gives them exercise sheet → patient uploads it → AI coaches them through each exercise at home → tracks progress over weeks → reports to doctor

### 2. Yoga for Mental Health
**User:** Person dealing with anxiety, wants to start yoga
**Skill:** Warrior Pose, Tree Pose, Sun Salutation
**How:** Selects yoga category → AI guides them through poses with voice → corrects alignment in real-time → tracks improvement over sessions

### 3. Elderly Fall Prevention
**User:** 72-year-old living alone
**Skill:** Sit-to-Stand, Heel Raises, Standing Balance
**How:** Family member sets up the app → elderly parent practices daily mobility exercises → AI ensures safe form → family can check progress remotely

### 4. Sign Language Learning
**User:** Parent learning ASL to communicate with deaf child
**Skill:** ASL Alphabet, Common Phrases, Greetings
**How:** Selects sign language → AI watches hand/arm positions → provides feedback on sign accuracy → tracks which signs need more practice

### 5. Remote PT with Friends
**User:** Two friends recovering from sports injuries
**Skill:** Shoulder Raise, Ankle Mobility
**How:** Create a practice room → both join → practice together with live scores → AI coaches both → friendly competition motivates consistency

### 6. Workplace Ergonomics
**User:** Office worker with back pain
**Skill:** Desk Posture, Proper Lifting, Stretch Break
**How:** Opens app at desk → AI monitors posture → alerts when slouching → guides through micro-breaks with correct form

---

## Why This Matters (Human Flourishing)

Physical capability is foundational to human wellbeing. When people can't move well — due to injury, age, disability, or lack of access to instruction — their quality of life suffers dramatically.

AEGIS democratizes access to physical coaching. A patient in rural Montana gets the same quality of PT guidance as someone in Manhattan. A grandmother in a small town gets fall-prevention coaching that could save her life. A child learns sign language with real-time feedback that no textbook can provide.

**One phone. Zero cost. Any skill. Everyone deserves a coach.**

---

## Tech Stack

| Layer | Technology |
|---|---|
| AI Orchestration | Claude Agent SDK — 3 sub-agents, 46 MCP tools, hooks |
| Voice AI | OpenAI Realtime API (GPT-4o) — bidirectional voice, 3-layer interruption, emotional tone |
| Computer Vision | YOLO11n + MediaPipe Pose + MediaPipe Hands + ByteTrack + Depth Anything V2 |
| DGX Inference | NVIDIA DGX Spark — RTMPose-WholeBody (133 keypoints, GB10 GPU) |
| Pose Engine | Custom: normalization, 10 joint angles, DTW, phase detection, scoring |
| Backend | FastAPI + Python 3.12 — 50+ REST routes, 3 WebSockets |
| Frontend | Next.js 14 + React + TailwindCSS + shadcn/ui |
| Scoring | Hybrid: local 1D CNN (14K params) + Claude for nuance |
| Codebase | ~15,000 lines of code |
