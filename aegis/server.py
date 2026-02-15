"""
AEGIS Server — FastAPI backend for AI Skill Coach.

REST API Endpoints:
  GET  /              → Serves phone web app (static files)
  GET  /api/state     → Current spatial state JSON
  GET  /api/config    → Client config
  GET  /api/goals     → List all goals (spatial + skill coaching)
  POST /api/goals/{id}→ Set active goal

  Coaching APIs (for Next.js frontend):
  POST /api/coaching/start          → Start coaching session
  POST /api/coaching/stop           → End coaching session
  GET  /api/coaching/status         → Current coaching session status
  GET  /api/coaching/progress       → Detailed progress (reps, scores, trend)
  GET  /api/coaching/score          → Current similarity score vs reference
  GET  /api/coaching/quality        → Movement quality metrics
  GET  /api/coaching/compensation   → Compensation pattern detection
  GET  /api/references              → List expert references
  POST /api/references/record/start → Start recording expert reference
  POST /api/references/record/stop  → Stop recording, save reference
  GET  /api/references/{name}       → Get reference metadata

  WebSockets:
  WS   /ws/video      → Camera frames in, spatial state out
  WS   /ws/audio      → Mic audio proxy to Gemini Live voice
  WS   /ws/coaching    → Real-time coaching data stream (scores, reps, feedback)

Architecture:
  Camera → /ws/video → SpatialEngine → structured state + coaching data
  Mic → /ws/audio → Gemini Live bridge → voice coaching
  Next.js → REST APIs → MCP tools → pose comparison engine
"""

import asyncio
import base64
import json
import time
import os
import subprocess

import cv2
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
from pydantic import BaseModel
from typing import Optional

from aegis import config
from aegis.pose_comparison import (
    normalize_skeleton, compute_joint_angles, compare_poses, quick_compare,
    compare_sequences, compute_movement_quality, detect_compensation,
    ReferenceStore, RecordingSession, CoachingSession,
    KEY_ANGLES, ANGLE_NAMES,
)
from aegis.skill_graph import GraphStore, SkillGraph
from aegis.data_collector import DataCollector
from aegis.hybrid_scorer import HybridScorer
from aegis.memory import MemoryStore
from aegis.rooms import RoomManager
from aegis.ai_expert import (
    list_canonical_exercises, get_ai_expert, generate_expert_sequence,
    generate_expert_sequence_from_description, get_best_expert,
)
import re

app = FastAPI(title="AEGIS — AI Skill Coach")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Shared state (set by main before starting server) ─────────────────
engine = None  # SpatialEngine instance (set externally)
agent = None   # AegisSDKAgent instance (set externally)
gemini_bridge = None  # GeminiBridge instance (set externally)
openai_voice = None   # OpenAIVoiceBridge instance (set externally)
dgx_client = None     # DGXClient instance (set externally)
telegram_bot = None   # TelegramBot instance (set externally)
_monitoring_active = False
_monitoring_goal = None
_monitoring_task: asyncio.Task | None = None
_monitoring_alerts: list[dict] = []

# ── Coaching state ────────────────────────────────────────────────────
_reference_store = ReferenceStore()
_recording_session: RecordingSession | None = None
_coaching_session: CoachingSession | None = None
_coaching_ws_clients: list[WebSocket] = []  # connected coaching WS clients
_graph_store = GraphStore()
_data_collector = DataCollector()
_hybrid_scorer = HybridScorer()
_memory_store = MemoryStore()
_room_manager = RoomManager()

STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")

# ── Claude Agent SDK coaching intelligence loop ──────────────────────────
_coaching_intelligence_task: asyncio.Task | None = None
_last_agent_rep_count = 0

async def _run_coaching_intelligence():
    """Background loop: Claude Agent SDK analyzes coaching every ~5 seconds.
    
    Gathers coaching data directly from server state and passes it to Claude
    for intelligent analysis. Claude responds with ONLY the coaching feedback.
    Feedback is shown on screen + spoken via OpenAI Realtime voice.
    """
    global _last_agent_rep_count
    _last_agent_rep_count = 0
    check_count = 0
    
    # Wait for coaching to get going
    await asyncio.sleep(4)
    _last_coverage_warn_time = 0
    _consecutive_declining = 0
    _last_milestone_rep = 0
    
    while _coaching_session and agent:
        try:
            check_count += 1
            session = _coaching_session
            if not session:
                break
            
            # Gather coaching data directly from server state
            progress = session.get_progress()
            current_reps = progress.get("reps_completed", 0)
            avg_score = progress.get("avg_score", 0)
            best_score = progress.get("best_score", 0)
            trend = progress.get("trend", "stable")
            recent_scores = progress.get("scores_per_rep", [])
            top_corrections = progress.get("top_corrections", [])
            
            # Build prompt with data inline (no MCP tool dependency)
            corrections_str = ", ".join(c[0] for c in top_corrections[:2]) if top_corrections else "none yet"
            data_block = (
                f"Skill: {session.skill_name}\n"
                f"Reps: {current_reps}, Avg score: {avg_score:.0f}/100, "
                f"Best: {best_score:.0f}/100\n"
                f"Trend: {trend}, Top corrections needed: {corrections_str}\n"
                f"Recent scores: {[round(s) for s in recent_scores[-5:]]}"
            )
            
            # --- Camera coverage check (every 30s) ---
            now = time.time()
            if now - _last_coverage_warn_time > 30:
                try:
                    from aegis import mcp_server as mcp_mod
                    vis = mcp_mod.check_landmark_visibility(0)
                    if isinstance(vis, dict) and vis.get("warnings"):
                        _last_coverage_warn_time = now
                        coverage_warn = vis["warnings"][0]
                        if openai_voice and openai_voice.is_connected:
                            await openai_voice.speak(coverage_warn)
                except Exception:
                    pass

            # --- Proactive check-ins ---
            if trend == "declining":
                _consecutive_declining += 1
            else:
                _consecutive_declining = 0

            # Inject coaching context into OpenAI voice (via session.update — no conversation pollution)
            if openai_voice and openai_voice.is_connected:
                try:
                    await openai_voice.inject_coaching_context(data_block)
                except Exception:
                    pass

            if check_count == 1:
                # Set skill + session memory on first check
                if openai_voice and openai_voice.is_connected:
                    openai_voice.set_skill(session.skill_name)
                    try:
                        history = _data_collector.get_last_session_summary(session.skill_name)
                        if history:
                            memory_text = (
                                f"Last time on {history['skill']}: "
                                f"{history['total_reps']} reps, avg {history['avg_score']}/100, "
                                f"best {history['best_score']}/100. "
                                f"Top issues: {', '.join(history['top_corrections']) if history['top_corrections'] else 'none'}."
                            )
                            openai_voice.set_session_history(memory_text)
                    except Exception:
                        pass
                prompt = (
                    f"[COACHING SESSION STARTED]\n{data_block}\n\n"
                    "Output ONLY the exact sentence to say aloud — nothing else. "
                    "Give a brief, warm opening like 'Alright, let's get started! Show me what you've got.'"
                )
                agent_action = "session_start"
            elif current_reps > 0 and current_reps % 10 == 0 and current_reps != _last_milestone_rep:
                _last_milestone_rep = current_reps
                prompt = (
                    f"[MILESTONE: {current_reps} REPS]\n{data_block}\n\n"
                    "Output ONLY the exact sentence to say aloud — nothing else. "
                    f"Celebrate {current_reps} reps! Example: 'That's {current_reps}! Great set, want to keep going?'"
                )
                agent_action = "milestone"
            elif _consecutive_declining >= 3:
                prompt = (
                    f"[SCORES DECLINING 3+ REPS]\n{data_block}\n\n"
                    "Output ONLY the exact sentence to say aloud — nothing else. "
                    "Be gentle. Example: 'Hey, let's slow it down a bit. Take a breath.'"
                )
                agent_action = "fatigue_checkin"
                _consecutive_declining = 0
            elif current_reps > _last_agent_rep_count:
                _last_agent_rep_count = current_reps
                prompt = (
                    f"[REP {current_reps} COMPLETED]\n{data_block}\n\n"
                    "Output ONLY the exact sentence to say aloud — nothing else. No analysis, no reasoning. "
                    "Give specific body cue feedback. Example: 'Rep 5! Push your knees out more on the way down.'"
                )
                agent_action = "rep_feedback"
            else:
                if trend == "declining":
                    prompt = (
                        f"[SCORES DECLINING]\n{data_block}\n\n"
                        "Output ONLY the exact sentence to say aloud — nothing else. "
                        "Give a motivating correction. Example: 'Focus on your form — chest up, core tight.'"
                    )
                    agent_action = "correction"
                else:
                    # Skip periodic if nothing interesting
                    await asyncio.sleep(5)
                    continue
            
            response = await agent.send_message(prompt)
            
            # Clean response — extract only the spoken coaching line
            if response:
                speech_text = response.strip()
                # Remove reasoning prefixes Claude adds
                meta_prefixes = ["Here's", "Sure,", "Okay,", "I'll", "Let me", "Based on", 
                                 "I can see", "Looking at", "After analyzing", "I see",
                                 "I notice", "I want to", "I'm going", "I need"]
                for prefix in meta_prefixes:
                    if speech_text.startswith(prefix) and ":" in speech_text[:80]:
                        speech_text = speech_text.split(":", 1)[1].strip()
                # Split into sentences (keep ! as sentence ender)
                sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', speech_text) if s.strip()]
                # Filter out meta-sentences (about tools, analysis, etc)
                meta_words = ['tool', 'analyze', 'technical', "i'll", 'let me', 'i can see',
                              'i will', 'experiencing', 'i see there', 'i notice', 'i want to',
                              'i\'m going', 'coaching tool', 'perception', 'starting position']
                coaching_sentences = [s for s in sentences if not any(w in s.lower() for w in meta_words)]
                if coaching_sentences:
                    speech_text = ' '.join(coaching_sentences[-2:])
                elif sentences:
                    # All sentences are meta — use a simple fallback
                    if agent_action == "session_start":
                        speech_text = "Alright, let's get started! Show me what you've got."
                    elif agent_action == "rep_feedback":
                        speech_text = f"That's rep {current_reps}! Keep it going."
                    elif agent_action == "milestone":
                        speech_text = f"Nice, {current_reps} reps done! Great work."
                    else:
                        speech_text = "Looking good, keep that form tight!"
                # Remove quotes if wrapped
                speech_text = speech_text.strip('"').strip("'")
                # Limit length
                speech_text = speech_text[:200]
                
                if speech_text and len(speech_text) > 5:
                    print(f"[Agent] Coach says: {speech_text}")
                    
                    # Get recent tool calls from agent for UI visibility
                    recent_tools = agent.get_tool_log(last_n=5) if hasattr(agent, 'get_tool_log') else []
                    tool_names = [t.get("tool", "") for t in recent_tools][-3:]
                    
                    # Send to frontend as text overlay with agent orchestration metadata
                    if _coaching_ws_clients:
                        agent_msg = json.dumps({
                            "type": "agent_feedback",
                            "data": speech_text,
                            "check": check_count,
                            "agent_action": agent_action,
                            "tools_used": tool_names,
                            "sub_agents": ["perception-agent", "coach-agent"] if any("compare" in t or "posture" in t or "alignment" in t for t in tool_names) else ["coach-agent"],
                        })
                        for client in list(_coaching_ws_clients):
                            try:
                                await client.send_text(agent_msg)
                            except Exception:
                                pass
                    
                    # Voice: OpenAI Realtime (primary) → macOS TTS (fallback)
                    spoken = False
                    if openai_voice and openai_voice.is_connected:
                        try:
                            spoken = await openai_voice.speak(speech_text)
                        except Exception:
                            pass
                    if not spoken and _coaching_ws_clients:
                        # Fallback: browser speechSynthesis (works offline, any device)
                        fallback_msg = json.dumps({
                            "type": "tts_fallback",
                            "data": speech_text,
                        })
                        for client in list(_coaching_ws_clients):
                            try:
                                await client.send_text(fallback_msg)
                            except Exception:
                                pass
                        print("[Voice] Fallback: browser TTS")
            
        except Exception as e:
            print(f"[Agent] Coaching intelligence error: {e}")
        
        # Wait before next check
        await asyncio.sleep(5)

def _start_coaching_intelligence():
    """Start the Claude coaching intelligence background task."""
    global _coaching_intelligence_task
    if agent is None:
        return
    if _coaching_intelligence_task and not _coaching_intelligence_task.done():
        _coaching_intelligence_task.cancel()
    _coaching_intelligence_task = asyncio.create_task(_run_coaching_intelligence())
    print("[Agent] Coaching intelligence loop started")

def _stop_coaching_intelligence():
    """Stop the Claude coaching intelligence background task."""
    global _coaching_intelligence_task
    if _coaching_intelligence_task and not _coaching_intelligence_task.done():
        _coaching_intelligence_task.cancel()
        _coaching_intelligence_task = None
        print("[Agent] Coaching intelligence loop stopped")

# ── Direct pose detector for coaching (avoids engine async race condition) ──
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

_POSE_MODEL = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", "pose_landmarker_lite.task")
_pose_landmarker = None

def _get_or_create_landmarker():
    global _pose_landmarker
    if _pose_landmarker is None and os.path.exists(_POSE_MODEL):
        base_options = mp_python.BaseOptions(model_asset_path=_POSE_MODEL)
        options = mp_vision.PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=mp_vision.RunningMode.IMAGE,
        )
        _pose_landmarker = mp_vision.PoseLandmarker.create_from_options(options)
    return _pose_landmarker

def _detect_pose_direct(frame_rgb: np.ndarray):
    """Run pose detection directly on a frame. Returns list of (x,y,vis) or None."""
    landmarker = _get_or_create_landmarker()
    if landmarker is None:
        return None
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
    result = landmarker.detect(mp_image)
    if result.pose_landmarks and len(result.pose_landmarks) > 0:
        return [(lm.x, lm.y, lm.visibility) for lm in result.pose_landmarks[0]]
    return None


# ── Pydantic request/response models ─────────────────────────────────

class CoachingStartRequest(BaseModel):
    skill_name: str
    reference_name: Optional[str] = None
    primary_angle: Optional[str] = "left_knee"

class RecordStopRequest(BaseModel):
    key_angle: Optional[str] = "left_knee"


# ── REST endpoints ────────────────────────────────────────────────────

@app.get("/")
async def index():
    """Serve the phone web app."""
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))


@app.get("/api/state")
async def get_state():
    """Current spatial state as JSON."""
    if engine is None:
        return {"error": "Engine not started"}
    return engine.get_state() or {}


@app.get("/api/summary")
async def get_summary():
    """Human-readable spatial summary."""
    if engine is None:
        return {"summary": "Engine not started"}
    return {"summary": engine.get_summary()}


@app.get("/api/config")
async def get_config():
    """Client-side config (API keys for Gemini Live client-side connection)."""
    return {
        "gemini_api_key": config.GEMINI_API_KEY,
        "gemini_model": config.GEMINI_MODEL,
        "gemini_voice": config.GEMINI_VOICE,
    }


# ── Goal management ───────────────────────────────────────────────

@app.get("/api/goals")
async def get_goals():
    """List all available goals and the active one."""
    from aegis.goals import get_all_goals
    goals = get_all_goals()
    active_id = agent.goal_id if agent else "general"
    return {
        "goals": [
            {
                "goal_id": g.goal_id,
                "name": g.name,
                "description": g.description,
                "icon": g.icon,
                "category": g.category,
                "active": g.goal_id == active_id,
            }
            for g in goals
        ],
        "active_goal_id": active_id,
    }


@app.post("/api/goals/{goal_id}")
async def set_goal(goal_id: str):
    """Set the active goal by ID."""
    if agent is None:
        return {"error": "Agent not started"}
    goal = agent.set_goal_by_id(goal_id)
    return {
        "goal_id": goal.goal_id,
        "name": goal.name,
        "description": goal.description,
        "icon": goal.icon,
    }


# ── Demo dashboard APIs (tool calls, decisions) ────────────────────

@app.get("/api/logs/tools")
async def get_tool_logs():
    """Recent tool call log for demo dashboard."""
    if agent is None:
        return {"tools": []}
    return {"tools": agent.get_tool_log(50)}


@app.get("/api/logs/decisions")
async def get_decision_logs():
    """Recent decision/reasoning log for demo dashboard."""
    if agent is None:
        return {"decisions": []}
    return {"decisions": agent.get_decision_log(50)}


@app.get("/api/agent/status")
async def get_agent_status():
    """Full agent status for dashboard."""
    if agent is None:
        return {"error": "Agent not started"}
    goal = agent.active_goal
    return {
        "goal": {
            "goal_id": goal.goal_id,
            "name": goal.name,
            "description": goal.description,
            "icon": goal.icon,
        },
        "tool_calls_count": len(agent.get_tool_log()),
        "decisions_count": len(agent.get_decision_log()),
        "conversation_turns": len(agent.conversation_history),
    }


class AgentMessageRequest(BaseModel):
    message: str

@app.post("/api/agent/message")
async def agent_message(req: AgentMessageRequest):
    """Send a message to the Claude agent. Returns the agent's response.

    Use this for coaching commands like:
    - "Coach me on squats"
    - "How's my form?"
    - "Start a workout session"
    """
    if agent is None:
        return {"error": "Agent not started"}
    response = await agent.send_message(req.message)
    return {
        "response": response,
        "goal": agent.active_goal.name,
        "goal_id": agent.active_goal.goal_id,
    }


@app.get("/dashboard")
async def dashboard():
    """Serve the demo dashboard."""
    return FileResponse(os.path.join(STATIC_DIR, "dashboard.html"))


# ═══════════════════════════════════════════════════════════════════════
# COACHING APIs — for Next.js frontend
# ═══════════════════════════════════════════════════════════════════════

@app.post("/api/coaching/start")
async def coaching_start(req: CoachingStartRequest):
    """Start a coaching session.

    Body: { skill_name, reference_name?, primary_angle? }

    If no reference_name provided, auto-generates an AI expert reference
    using canonical templates, Claude generation, or DGX motion generation.
    """
    global _coaching_session

    ref = None
    ref_source = None

    if req.reference_name:
        ref = _reference_store.load(req.reference_name)
        if ref is None:
            return {"error": f"Reference '{req.reference_name}' not found"}
        ref_source = "recorded_video"
    else:
        # Auto-generate AI expert — no video needed!
        ref = await get_best_expert(req.skill_name, dgx_client=dgx_client)
        if ref:
            ref_source = ref.metadata.get("source", "ai_generated") if ref.metadata else "ai_generated"

        # Get primary angle from template if available
        template = get_ai_expert(req.skill_name)
        if template and not req.primary_angle:
            req.primary_angle = template.get("primary_angle")

    _coaching_session = CoachingSession(skill_name=req.skill_name, reference=ref)
    _coaching_session.set_primary_angle(req.primary_angle or "left_knee")

    # Start Claude Agent SDK coaching intelligence (if agent available)
    _start_coaching_intelligence()

    return {
        "status": "session_started",
        "skill": req.skill_name,
        "reference_loaded": req.reference_name or (f"ai_expert_{req.skill_name}" if ref else None),
        "reference_source": ref_source,
        "primary_angle": req.primary_angle,
        "available_angles": list(KEY_ANGLES.keys()),
        "agent_active": agent is not None,
        "coaching_cues": (get_ai_expert(req.skill_name) or {}).get("coaching_cues", []),
    }


@app.post("/api/coaching/stop")
async def coaching_stop():
    """End the current coaching session. Returns final summary."""
    global _coaching_session
    if _coaching_session is None:
        return {"error": "No active coaching session"}

    # Stop Claude coaching intelligence
    _stop_coaching_intelligence()

    summary = _coaching_session.end()

    # Have Claude generate a session summary (if agent available)
    if agent:
        try:
            agent_summary = await agent.send_message(
                f"Coaching session for '{summary.get('skill', 'unknown')}' just ended. "
                f"Stats: {summary.get('total_reps', 0)} reps, "
                f"avg score {summary.get('avg_score', 0):.0f}, "
                f"best score {summary.get('best_score', 0):.0f}. "
                "Use get_coaching_progress for full details. "
                "Give a brief motivating session summary (2-3 sentences). "
                "Then update the user's skill proficiency if possible."
            )
            summary["agent_summary"] = agent_summary
        except Exception as e:
            print(f"[Agent] Session summary error: {e}")

    _coaching_session = None
    return summary


@app.get("/api/coaching/status")
async def coaching_status():
    """Get current coaching session status (is session active, what skill, etc.)."""
    if _coaching_session is None:
        return {
            "active": False,
            "recording_active": _recording_session is not None and _recording_session.active,
        }

    return {
        "active": True,
        "skill": _coaching_session.skill_name,
        "frame_count": _coaching_session.frame_count,
        "reps": _coaching_session.get_rep_count(),
        "has_reference": _coaching_session.reference is not None,
        "recording_active": _recording_session is not None and _recording_session.active,
    }


@app.get("/api/coaching/progress")
async def coaching_progress():
    """Detailed coaching progress: reps, scores per rep, trend, corrections."""
    if _coaching_session is None:
        return {"error": "No active coaching session"}
    return _coaching_session.get_progress()


@app.get("/api/coaching/score")
async def coaching_score():
    """Current similarity score vs expert reference (single frame comparison)."""
    if _coaching_session is None:
        return {"error": "No active coaching session"}
    if _coaching_session.reference is None:
        return {"error": "No reference loaded — using zero-shot mode"}

    # Get current pose from engine
    user_points = _get_current_pose_points()
    if user_points is None:
        return {"error": "No person detected"}

    result = quick_compare(user_points, _coaching_session.reference.skeletons[0].points)
    return result.to_dict()


@app.get("/api/coaching/quality")
async def coaching_quality():
    """Movement quality metrics: smoothness, symmetry, tempo."""
    if _coaching_session is None:
        return {"error": "No active coaching session"}

    frames = _coaching_session.current_rep_frames
    if len(frames) < 3:
        return {"error": "Need more frames — keep moving"}

    return compute_movement_quality(frames)


@app.get("/api/coaching/compensation")
async def coaching_compensation():
    """Detect compensation patterns (asymmetric deviations)."""
    user_points = _get_current_pose_points()
    if user_points is None:
        return {"error": "No person detected"}

    user_angles = compute_joint_angles(user_points)

    if _coaching_session and _coaching_session.reference:
        mid = len(_coaching_session.reference.skeletons) // 2
        expert_angles = _coaching_session.reference.skeletons[mid].joint_angles
        return {"patterns": detect_compensation(user_angles, expert_angles)}

    # No reference — check symmetry
    mirror = {}
    for joint, angle in user_angles.items():
        if "left_" in joint:
            right = joint.replace("left_", "right_")
            if right in user_angles:
                mirror[joint] = user_angles[right]
                mirror[right] = angle
    return {"patterns": detect_compensation(user_angles, mirror) if mirror else []}


@app.get("/api/coaching/angles")
async def coaching_angles():
    """Get all current joint angles for the detected person."""
    user_points = _get_current_pose_points()
    if user_points is None:
        return {"error": "No person detected"}

    angles = compute_joint_angles(user_points)
    result = {
        "angles": {k: round(v, 1) for k, v in angles.items()},
        "angle_names": ANGLE_NAMES,
    }

    # Include expert angles if reference loaded
    if _coaching_session and _coaching_session.reference:
        ref_skels = _coaching_session.reference.skeletons
        if ref_skels:
            mid = len(ref_skels) // 2
            result["expert_angles"] = {
                k: round(v, 1) for k, v in ref_skels[mid].joint_angles.items()
            }

    return result


# ═══════════════════════════════════════════════════════════════════════
# ROOM APIs — Multiplayer coaching rooms
# ═══════════════════════════════════════════════════════════════════════

class RoomCreateRequest(BaseModel):
    skill_name: str
    display_name: str = "Player 1"

class RoomJoinRequest(BaseModel):
    room_code: str
    display_name: str = "Player 2"

@app.post("/api/rooms/create")
async def create_room(req: RoomCreateRequest):
    """Create a new coaching room. Returns room code to share."""
    room = _room_manager.create_room(req.skill_name)
    participant = room.add_participant("host", req.display_name)
    return {
        "room_code": room.code,
        "skill": room.skill_name,
        "user_id": "host",
        "participants": room.participant_count,
    }

@app.post("/api/rooms/join")
async def join_room(req: RoomJoinRequest):
    """Join an existing room by code."""
    import uuid
    user_id = f"user_{uuid.uuid4().hex[:6]}"
    participant = _room_manager.join_room(req.room_code, user_id, req.display_name)
    if not participant:
        return {"error": f"Room '{req.room_code}' not found or closed"}
    room = _room_manager.get_room(req.room_code)
    return {
        "room_code": req.room_code,
        "skill": room.skill_name,
        "user_id": user_id,
        "participants": room.participant_count,
        "leaderboard": room.get_leaderboard(),
    }

@app.get("/api/rooms/{code}")
async def get_room(code: str):
    """Get room status and leaderboard."""
    room = _room_manager.get_room(code)
    if not room:
        return {"error": "Room not found"}
    return room.get_comparison_data()

@app.get("/api/rooms/{code}/leaderboard")
async def room_leaderboard(code: str):
    """Get live leaderboard for a room."""
    room = _room_manager.get_room(code)
    if not room:
        return {"error": "Room not found"}
    return {"leaderboard": room.get_leaderboard()}

@app.post("/api/rooms/{code}/compare")
async def room_compare(code: str):
    """Use Claude Agent SDK to compare all participants' performance.
    
    This is the multi-agent orchestration: Claude analyzes each user's
    coaching data and generates comparative feedback.
    """
    room = _room_manager.get_room(code)
    if not room:
        return {"error": "Room not found"}
    
    comparison = room.get_comparison_data()
    
    if not agent:
        return {**comparison, "agent_analysis": "Agent not available"}
    
    # Build comparison prompt for Claude
    participants_text = ""
    for i, p in enumerate(comparison["leaderboard"], 1):
        participants_text += (
            f"\n#{i} {p['display_name']}: "
            f"{p.get('reps_completed', 0)} reps, "
            f"avg {p.get('avg_score', 0):.0f}/100, "
            f"best {p.get('best_score', 0):.0f}/100, "
            f"trend: {p.get('trend', 'n/a')}"
        )
    
    try:
        analysis = await agent.send_message(
            f"You are judging a friendly coaching competition.\n"
            f"Skill: {comparison['skill']}\n"
            f"Duration: {comparison['duration']:.0f}s\n"
            f"Participants:{participants_text}\n\n"
            "Give a fun, encouraging comparison (3-4 sentences). "
            "Highlight each person's strength. Declare a winner if clear. "
            "Keep it friendly and motivating.\n"
            "NO reasoning, JUST the comparison text."
        )
        comparison["agent_analysis"] = analysis
    except Exception as e:
        comparison["agent_analysis"] = f"Analysis unavailable: {e}"
    
    return comparison

@app.post("/api/rooms/{code}/close")
async def close_room(code: str):
    """Close a room and get final comparison with Claude analysis."""
    result = _room_manager.close_room(code)
    if not result:
        return {"error": "Room not found"}
    
    # Get Claude's final analysis
    if agent and len(result.get("leaderboard", [])) > 1:
        try:
            participants_text = ""
            for i, p in enumerate(result["leaderboard"], 1):
                participants_text += (
                    f"\n#{i} {p['display_name']}: "
                    f"{p.get('reps_completed', 0)} reps, "
                    f"avg {p.get('avg_score', 0):.0f}/100, "
                    f"best {p.get('best_score', 0):.0f}/100"
                )
            
            analysis = await agent.send_message(
                f"Multiplayer coaching session ended!\n"
                f"Skill: {result['skill']}\n"
                f"Results:{participants_text}\n\n"
                "Give the final verdict (2-3 sentences). "
                "Declare a winner, compliment both players, suggest what to work on next.\n"
                "NO reasoning, JUST the verdict."
            )
            result["agent_verdict"] = analysis
        except Exception:
            pass
    
    return result

@app.get("/api/rooms")
async def list_rooms():
    """List all active rooms."""
    return {"rooms": _room_manager.list_rooms()}


# ═══════════════════════════════════════════════════════════════════════
# AI EXPERT APIs — Generate expert references without video
# ═══════════════════════════════════════════════════════════════════════

@app.get("/api/ai-expert/exercises")
async def ai_expert_list():
    """List all canonical exercises the AI can generate expert form for."""
    return {"exercises": list_canonical_exercises()}


@app.get("/api/ai-expert/{exercise}")
async def ai_expert_get(exercise: str):
    """Get AI expert template for an exercise (angles, phases, coaching cues)."""
    template = get_ai_expert(exercise)
    if template is None:
        return {"error": f"No template for '{exercise}'", "available": [e["id"] for e in list_canonical_exercises()]}
    return {"exercise": exercise, **template}


class AIExpertGenerateRequest(BaseModel):
    skill_description: str
    use_dgx: bool = True


@app.post("/api/ai-expert/generate")
async def ai_expert_generate(req: AIExpertGenerateRequest):
    """Generate expert reference for ANY skill using AI.

    Priority: DGX motion generation → canonical template → Claude generation.
    """
    seq = await get_best_expert(
        req.skill_description,
        dgx_client=dgx_client if req.use_dgx else None,
    )
    if seq is None:
        return {"error": f"Could not generate expert for '{req.skill_description}'"}

    # Serialize skeleton keypoints for frontend rendering
    keyframes = []
    for skel in seq.skeletons:
        frame_pts = []
        for pt in skel.points:
            if isinstance(pt, (list, tuple)) and len(pt) >= 3:
                frame_pts.append([float(pt[0]), float(pt[1]), float(pt[2])])
            elif isinstance(pt, (list, tuple)) and len(pt) >= 2:
                frame_pts.append([float(pt[0]), float(pt[1]), 0.95])
            else:
                frame_pts.append([0.0, 0.0, 0.0])
        keyframes.append(frame_pts)

    return {
        "skill": req.skill_description,
        "source": seq.metadata.get("source", "unknown") if seq.metadata else "unknown",
        "frame_count": seq.frame_count,
        "phases": seq.metadata.get("phases", []) if seq.metadata else [],
        "coaching_cues": seq.metadata.get("coaching_cues", []) if seq.metadata else [],
        "keyframes": keyframes,
        "generation_ms": seq.metadata.get("generation_ms", 0) if seq.metadata else 0,
        "model": seq.metadata.get("model", "") if seq.metadata else "",
    }


# ═══════════════════════════════════════════════════════════════════════
# REFERENCE APIs — Expert reference management
# ═══════════════════════════════════════════════════════════════════════

@app.get("/api/references")
async def list_references():
    """List all stored expert references."""
    return {"references": _reference_store.list_references()}


class YouTubeIngestRequest(BaseModel):
    url: str
    name: str
    key_angle: Optional[str] = "left_knee"


@app.post("/api/references/from_youtube")
async def ingest_from_youtube(req: YouTubeIngestRequest):
    """Download a YouTube video, extract skeleton, save as coaching reference.

    Body: { url, name, key_angle? }
    Returns: reference metadata + frame count.
    """
    try:
        from aegis.video_ingest import ingest_youtube
        ref = await ingest_youtube(
            url=req.url,
            name=req.name,
            key_angle=req.key_angle or "left_knee",
            reference_store=_reference_store,
        )
        return {
            "status": "success",
            "name": ref.name,
            "frames": len(ref.skeletons),
            "phases": len(ref.phases),
            "duration": ref.skeletons[-1].timestamp if ref.skeletons else 0,
            "metadata": ref.metadata,
        }
    except Exception as e:
        return {"status": "error", "error": str(e)}


@app.get("/api/references/{name}")
async def get_reference(name: str):
    """Get metadata for a specific reference."""
    ref = _reference_store.load(name)
    if ref is None:
        return {"error": f"Reference '{name}' not found"}
    return {
        "name": ref.name,
        "frame_count": ref.frame_count,
        "duration": round(ref.duration, 1),
        "phases": [{"name": p.name, "start": p.start_frame, "end": p.end_frame} for p in ref.phases],
        "angle_keys": list(ref.skeletons[0].joint_angles.keys()) if ref.skeletons else [],
    }


@app.post("/api/references/record/start")
async def reference_record_start(name: str):
    """Start recording an expert reference from live camera."""
    global _recording_session
    if _recording_session and _recording_session.active:
        return {"error": "Recording already in progress"}
    _recording_session = RecordingSession(name=name)
    return {"status": "recording_started", "name": name}


@app.post("/api/references/record/stop")
async def reference_record_stop(req: RecordStopRequest = RecordStopRequest()):
    """Stop recording and save the expert reference."""
    global _recording_session
    if not _recording_session or not _recording_session.active:
        return {"error": "No recording in progress"}

    sequence = _recording_session.stop(key_angle=req.key_angle or "left_knee")
    filepath = _reference_store.save(sequence)
    _recording_session = None

    return {
        "status": "reference_saved",
        "name": sequence.name,
        "frame_count": sequence.frame_count,
        "duration": round(sequence.duration, 1),
        "phases": len(sequence.phases),
        "file": filepath,
    }


@app.delete("/api/references/{name}")
async def delete_reference(name: str):
    """Delete a stored reference."""
    deleted = _reference_store.delete(name)
    if not deleted:
        return {"error": f"Reference '{name}' not found"}
    return {"status": "deleted", "name": name}


# ═══════════════════════════════════════════════════════════════════════
# SKILL GRAPH APIs — Progression tracking
# ═══════════════════════════════════════════════════════════════════════

@app.get("/api/graphs")
async def list_skill_graphs():
    """List all available skill graphs with progress summaries."""
    return {"graphs": _graph_store.list_graphs()}


@app.get("/api/graphs/{name}")
async def get_skill_graph(name: str):
    """Get full skill tree for visualization (nodes + links + progress)."""
    graph = _graph_store.get_or_create(name)
    return graph.get_skill_tree()


@app.get("/api/graphs/{name}/recommend")
async def get_recommendations(name: str, top_n: int = 3):
    """Get recommended next skills to practice."""
    graph = _graph_store.get_or_create(name)
    return {"recommendations": graph.get_next_recommended(top_n)}


@app.post("/api/graphs/{name}/skills/{skill_id}/update")
async def update_skill_score(name: str, skill_id: str, score: float):
    """Update a skill's proficiency after a coaching session."""
    graph = _graph_store.get_or_create(name)
    if skill_id not in graph.skills:
        return {"error": f"Skill '{skill_id}' not found in graph '{name}'"}
    graph.update_skill_proficiency(skill_id, score)
    _graph_store.save_graph(graph)
    return graph.skills[skill_id].to_dict()


@app.get("/api/graphs/{name}/progress")
async def get_graph_progress(name: str):
    """Get overall progress summary for a skill graph."""
    graph = _graph_store.get_or_create(name)
    return graph.get_progress_summary()


# ═══════════════════════════════════════════════════════════════════════
# DATA COLLECTION APIs — Training data management
# ═══════════════════════════════════════════════════════════════════════

@app.get("/api/training/stats")
async def training_data_stats():
    """Get statistics about collected training data."""
    return _data_collector.get_dataset_stats()


@app.get("/api/training/export")
async def export_training_data(skill: str = None, pad_to: int = 60):
    """Export training data for model training.

    Returns angle sequences (X) and scores (y) ready for 1D CNN training.
    """
    return _data_collector.export_for_training(skill=skill, pad_to=pad_to)


@app.delete("/api/training/{skill}")
async def clear_training_data(skill: str):
    """Clear training data for a specific skill."""
    deleted = _data_collector.clear_skill_data(skill)
    if not deleted:
        return {"error": f"No training data for '{skill}'"}
    return {"status": "cleared", "skill": skill}


# ═══════════════════════════════════════════════════════════════════════
# MODEL & HYBRID INFERENCE APIs
# ═══════════════════════════════════════════════════════════════════════

@app.post("/api/model/train")
async def train_model(skill: str = None, epochs: int = 50, augment: bool = True):
    """Train the local SkillScorer model from collected coaching data.

    Uses data augmentation by default (5x multiplier) for robustness.
    Requires at least 5 training samples. Training takes a few seconds.
    After training, the model provides instant (~0.2ms) scoring.
    """
    result = _hybrid_scorer.train_from_collected_data(
        skill=skill, epochs=epochs, augment=augment,
    )
    return result


@app.post("/api/model/bootstrap")
async def bootstrap_model(skill: str = "squat", n_synthetic: int = 50, epochs: int = 80):
    """Bootstrap a model with synthetic data when no real data exists.

    Generates synthetic training data based on biomechanically plausible
    angle ranges for the given skill, then trains the model immediately.
    The model improves as real coaching data comes in.
    """
    result = _hybrid_scorer.generate_and_train(
        skill=skill, n_synthetic=n_synthetic, epochs=epochs,
    )
    return result


@app.post("/api/model/train-split")
async def train_model_split(skill: str = None, epochs: int = 80):
    """Train with proper train/val/test split for rigorous evaluation.

    Only augments the training set. Val and test use real data only.
    Returns separate metrics for each split.
    """
    result = _hybrid_scorer.train_with_split(skill=skill, epochs=epochs)
    return result


@app.post("/api/training/generate")
async def generate_synthetic(skill: str = "squat", n_samples: int = 50):
    """Generate synthetic training data for a skill.

    Uses biomechanically plausible angle ranges with varying quality
    levels (good/medium/poor form). Useful for bootstrapping.
    """
    result = _hybrid_scorer.data_collector.generate_synthetic_data(
        skill=skill, n_samples=n_samples,
    )
    return result


@app.get("/api/model/status")
async def model_status():
    """Get local model and hybrid scorer status."""
    return _hybrid_scorer.get_status()


@app.get("/api/model/predict")
async def model_predict():
    """Get instant local model prediction for current pose.

    Returns the local model's score (~0.2ms) without calling Claude.
    Requires a trained model and active coaching session.
    """
    if not _hybrid_scorer.local_model_available:
        return {"error": "Local model not trained yet. Use POST /api/model/train first."}

    if _coaching_session is None or not _coaching_session.current_rep_frames:
        return {"error": "No coaching session or no frames captured yet."}

    # Build angle sequence from recent frames
    from aegis.data_collector import ANGLE_KEYS
    frames = _coaching_session.current_rep_frames[-60:]
    angles_seq = []
    for skel in frames:
        row = [skel.joint_angles.get(k, 0.0) for k in ANGLE_KEYS]
        angles_seq.append(row)

    score = _hybrid_scorer.score_local(angles_seq)
    return {
        "score": round(score, 1) if score is not None else None,
        "source": "local_model",
        "frames_used": len(angles_seq),
    }


@app.get("/api/model/hybrid")
async def hybrid_score():
    """Get hybrid score combining local model + last Claude analysis.

    Blends local model (60%) with Claude's last score (40%) for
    best-of-both-worlds accuracy.
    """
    if _coaching_session is None or not _coaching_session.current_rep_frames:
        return {"error": "No coaching session with frames."}

    from aegis.data_collector import ANGLE_KEYS
    frames = _coaching_session.current_rep_frames[-60:]
    angles_seq = [[skel.joint_angles.get(k, 0.0) for k in ANGLE_KEYS] for skel in frames]

    # Get Claude's last score from coaching progress if available
    claude_score = None
    if _coaching_session.reps:
        claude_score = _coaching_session.reps[-1].similarity_score

    result = _hybrid_scorer.get_hybrid_score(
        angles_seq,
        claude_score=claude_score,
    )
    return result.to_dict()


# ═══════════════════════════════════════════════════════════════════════
# ACTIVITY ML MODEL APIs
# ═══════════════════════════════════════════════════════════════════════

@app.post("/api/activity/train")
async def train_activity_model(n_per_class: int = 150, epochs: int = 60):
    """Train the ML activity classifier from synthetic biomechanical data.

    Generates realistic pose sequences for each activity class and trains
    a temporal 1D CNN. After training, the classifier uses ML predictions
    with heuristic fallback.
    """
    from aegis.activity import _classifier
    result = _classifier.train_ml_model(n_per_class=n_per_class, epochs=epochs)
    return result


@app.get("/api/activity/stats")
async def activity_stats():
    """Get activity classifier stats: ML vs heuristic prediction ratio."""
    from aegis.activity import _classifier
    return _classifier.get_stats()


# ═══════════════════════════════════════════════════════════════════════
# MEMORY APIs — Observations, user profile, recall
# ═══════════════════════════════════════════════════════════════════════

class ObservationRequest(BaseModel):
    content: str
    category: str = "general"

class ProfileUpdateRequest(BaseModel):
    name: Optional[str] = None
    fitness_level: Optional[str] = None
    coaching_style: Optional[str] = None
    dominant_side: Optional[str] = None
    injury_history: Optional[list[str]] = None
    limitations: Optional[list[str]] = None

@app.post("/api/memory/observe")
async def add_observation(req: ObservationRequest):
    """Store an observation in memory."""
    obs = _memory_store.add_observation(req.content, category=req.category)
    return {"status": "stored", "category": req.category}


@app.get("/api/memory/recall")
async def recall_memory(query: str, top_k: int = 5, category: str = None):
    """Retrieve relevant memories using semantic search."""
    results = _memory_store.recall(query, top_k=top_k, category=category)
    return {
        "query": query,
        "results": [o.to_dict() for o in results],
    }


@app.get("/api/memory/recent")
async def recent_memories(n: int = 10, category: str = None):
    """Get most recent observations."""
    results = _memory_store.get_recent(n=n, category=category)
    return {"observations": [o.to_dict() for o in results]}


@app.get("/api/memory/stats")
async def memory_stats():
    """Memory system statistics."""
    return _memory_store.get_stats()


@app.get("/api/profile")
async def get_user_profile():
    """Get the user profile."""
    return _memory_store.user_profile.to_dict()


@app.patch("/api/profile")
async def update_user_profile(req: ProfileUpdateRequest):
    """Update user profile fields."""
    updates = {k: v for k, v in req.dict().items() if v is not None}
    _memory_store.update_profile(**updates)
    return _memory_store.user_profile.to_dict()


# ═══════════════════════════════════════════════════════════════════════
# HELPER: Get current pose from engine
# ═══════════════════════════════════════════════════════════════════════

def _get_current_pose_points() -> list | None:
    """Extract current pose landmark points from the engine's tracked persons."""
    if engine is None:
        return None
    persons = engine._tracked_persons if hasattr(engine, '_tracked_persons') else []
    for p in persons:
        if p.pose and p.pose.points:
            return p.pose.points
    return None


# ── Zero-shot coaching helpers ─────────────────────────────────────────

# Ideal angle ranges for common exercises (min, ideal, max)
IDEAL_ANGLES = {
    "Squat": {"left_knee": (70, 90, 110), "right_knee": (70, 90, 110), "left_hip": (70, 85, 100), "right_hip": (70, 85, 100)},
    "Deadlift": {"left_knee": (140, 165, 180), "right_knee": (140, 165, 180), "left_hip": (80, 100, 120)},
    "Push-up": {"left_elbow": (80, 90, 100), "right_elbow": (80, 90, 100), "left_shoulder": (40, 60, 80)},
    "Lunge": {"left_knee": (75, 90, 105), "right_knee": (75, 90, 105), "left_hip": (80, 100, 120)},
    "Plank": {"left_elbow": (160, 175, 180), "left_hip": (165, 175, 180), "left_knee": (165, 175, 180)},
    "Warrior Pose": {"left_knee": (85, 95, 110), "right_knee": (160, 175, 180), "left_hip": (80, 90, 100)},
    "Tree Pose": {"left_knee": (160, 175, 180), "right_knee": (40, 60, 90), "left_hip": (165, 175, 180)},
    "Front Kick": {"left_knee": (150, 170, 180), "left_hip": (60, 80, 100)},
    "Boxing Jab": {"left_elbow": (155, 170, 180), "left_shoulder": (70, 85, 100)},
}

# Fallback: general good posture
_DEFAULT_IDEALS = {"left_knee": (90, 120, 170), "right_knee": (90, 120, 170), "left_hip": (90, 120, 170), "right_hip": (90, 120, 170), "left_elbow": (90, 130, 170), "right_elbow": (90, 130, 170)}


def _zero_shot_score(skill_name: str, angles: dict) -> tuple[float, dict]:
    """Score current angles against ideal ranges for the skill. Returns (score, deviations)."""
    ideals = IDEAL_ANGLES.get(skill_name, _DEFAULT_IDEALS)
    deviations = {}
    scores = []

    for joint, angle_val in angles.items():
        if joint in ideals:
            lo, ideal, hi = ideals[joint]
        else:
            lo, ideal, hi = _DEFAULT_IDEALS.get(joint, (90, 135, 180))

        if lo <= angle_val <= hi:
            dev = abs(angle_val - ideal)
            score = max(0, 100 - dev * 2)
        elif angle_val < lo:
            dev = lo - angle_val
            score = max(0, 70 - dev * 3)
        else:
            dev = angle_val - hi
            score = max(0, 70 - dev * 3)

        deviations[joint] = abs(angle_val - ideal)
        scores.append(score)

    overall = sum(scores) / len(scores) if scores else 50.0
    return overall, deviations


def _detect_simple_phase(angle_history: dict, primary_angle: str) -> str:
    """Simple phase detection from angle history."""
    history = angle_history.get(primary_angle, [])
    if len(history) < 5:
        return "Preparation"

    recent = history[-5:]
    trend = recent[-1] - recent[0]

    if trend < -5:
        return "Descending"
    elif trend > 5:
        return "Ascending"
    elif recent[-1] < 110:
        return "Bottom"
    else:
        return "Standing"


# ── WebSocket: Video frames from phone ────────────────────────────────

@app.websocket("/ws/video")
async def ws_video(websocket: WebSocket):
    """
    Phone sends camera frames as base64 JPEG.
    Server processes them and sends back spatial state.
    
    Protocol:
      Phone → Server: {"type": "frame", "data": "<base64 jpeg>"}
      Server → Phone: {"type": "state", "data": <spatial_state_dict>}
    """
    await websocket.accept()
    print("[Server] Phone connected (video)")

    frame_count = 0
    try:
        while True:
            msg = await websocket.receive_text()
            data = json.loads(msg)

            if data.get("type") == "frame":
                # Decode base64 JPEG to numpy array
                jpg_bytes = base64.b64decode(data["data"])
                np_arr = np.frombuffer(jpg_bytes, dtype=np.uint8)
                frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

                if frame is not None:
                    if engine is not None:
                        engine.push_frame(frame)
                    frame_count += 1

                    # Run pose detection directly on this frame (sync, no race condition)
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    pose_points = _detect_pose_direct(rgb)
                    if pose_points:
                        if _recording_session and _recording_session.active:
                            _recording_session.add_frame(pose_points)
                        if _coaching_session and _coaching_ws_clients:
                            coaching_result = _coaching_session.add_frame(pose_points)
                            # Convert pose points for frontend skeleton drawing
                            landmarks = [[float(p[0]), float(p[1]), float(p[2]) if len(p) > 2 else 1.0] for p in pose_points[:33]]

                            if coaching_result:
                                # Reference-based coaching
                                coaching_msg = json.dumps({
                                    "type": "coaching",
                                    "data": coaching_result.to_dict(),
                                    "reps": _coaching_session.get_rep_count(),
                                    "frame": _coaching_session.frame_count,
                                    "landmarks": landmarks,
                                })
                            else:
                                # Zero-shot coaching: score from joint angles
                                skel = normalize_skeleton(pose_points)
                                angles = skel.joint_angles
                                score, devs = _zero_shot_score(
                                    _coaching_session.skill_name, angles
                                )
                                worst = sorted(devs.items(), key=lambda x: -x[1])[:3]
                                best = sorted(devs.items(), key=lambda x: x[1])[:3]
                                phase = _detect_simple_phase(
                                    _coaching_session._angle_history,
                                    _coaching_session._primary_angle,
                                )
                                coaching_msg = json.dumps({
                                    "type": "coaching",
                                    "landmarks": landmarks,
                                    "data": {
                                        "similarity_score": round(score, 1),
                                        "per_joint_deviation": {
                                            k: round(v, 1) for k, v in devs.items()
                                        },
                                        "worst_joints": [
                                            (j, round(d, 1)) for j, d in worst
                                        ],
                                        "best_joints": [
                                            (j, round(d, 1)) for j, d in best
                                        ],
                                        "phase": phase,
                                        "phase_score": round(score, 1),
                                    },
                                    "reps": _coaching_session.get_rep_count(),
                                    "frame": _coaching_session.frame_count,
                                })
                            for client in list(_coaching_ws_clients):
                                try:
                                    await client.send_text(coaching_msg)
                                except Exception:
                                    _coaching_ws_clients.remove(client)

                    # Send back spatial state every frame
                    state = engine.get_state()
                    if state:
                        await websocket.send_text(json.dumps({
                            "type": "state",
                            "data": state,
                        }))

            elif data.get("type") == "ping":
                await websocket.send_text(json.dumps({"type": "pong"}))

    except WebSocketDisconnect:
        print(f"[Server] Phone disconnected (video) after {frame_count} frames")
    except Exception as e:
        print(f"[Server] Video WebSocket error: {e}")


# ── WebSocket: Audio proxy (OpenAI Realtime or Gemini Live) ──────────

@app.websocket("/ws/audio")
async def ws_audio(websocket: WebSocket):
    """
    Bidirectional voice coaching via OpenAI Realtime API (preferred) or Gemini Live.
    
    Protocol:
      Client → Server: {"type":"audio","data":"<base64 pcm 16kHz>"} — mic audio
      Server → Client: {"type":"audio","data":"<base64 pcm 24kHz>"} — AI voice
      Client → Server: {"type":"text","data":"<text>"} — coaching context
      Client → Server: {"type":"ping"} → {"type":"pong"}
    """
    await websocket.accept()
    print("[Server] Audio client connected")

    # Pick voice bridge: OpenAI Realtime preferred
    bridge = openai_voice or gemini_bridge
    bridge_name = "OpenAI Realtime" if openai_voice else "Gemini Live"

    if bridge is None:
        await websocket.send_text(json.dumps({"type": "error", "data": "No voice bridge configured"}))
        await websocket.close()
        return

    # Connect if not already
    if not bridge.is_connected:
        ok = await bridge.connect()
        if not ok:
            await websocket.send_text(json.dumps({"type": "error", "data": f"{bridge_name} connection failed"}))
            await websocket.close()
            return

    print(f"[Server] Voice bridge: {bridge_name}")

    # Task: forward AI audio output → client
    async def send_audio_to_client():
        try:
            while True:
                audio_bytes = await bridge.get_audio_output(timeout=0.1)
                if audio_bytes:
                    b64 = base64.b64encode(audio_bytes).decode()
                    await websocket.send_text(json.dumps({
                        "type": "audio",
                        "data": b64,
                    }))
        except asyncio.CancelledError:
            pass
        except Exception as e:
            print(f"[Server] Audio send error: {e}")

    send_task = asyncio.create_task(send_audio_to_client())

    try:
        while True:
            msg = await websocket.receive_text()
            data = json.loads(msg)

            if data.get("type") == "audio":
                pcm_bytes = base64.b64decode(data["data"])
                await bridge.send_audio(pcm_bytes)

            elif data.get("type") == "text":
                # Coaching context injection
                if openai_voice and openai_voice.is_connected:
                    await openai_voice.inject_coaching_context(data["data"])
                elif gemini_bridge and gemini_bridge.is_connected:
                    await gemini_bridge.send_text(data["data"])

            elif data.get("type") == "spatial":
                if gemini_bridge and gemini_bridge.is_connected:
                    await gemini_bridge.inject_spatial_state(data["data"])

            elif data.get("type") == "ping":
                await websocket.send_text(json.dumps({"type": "pong"}))

    except WebSocketDisconnect:
        print("[Server] Audio client disconnected")
    except Exception as e:
        print(f"[Server] Audio WebSocket error: {e}")
    finally:
        send_task.cancel()
        try:
            await send_task
        except asyncio.CancelledError:
            pass


@app.get("/api/dgx/status")
async def dgx_status():
    """NVIDIA DGX Spark inference server status."""
    if dgx_client is None:
        return {"available": False, "note": "Start server with --dgx URL to enable"}
    health = await dgx_client.check_health()
    return {
        "available": health,
        "stats": dgx_client.get_stats(),
        "model": "RTMPose-WholeBody (133 keypoints)",
        "gpu": "NVIDIA GB10",
    }


# ═══════════════════════════════════════════════════════════════════════
# MONITORING MODE — Autonomous spatial intelligence with goal-based alerts
# ═══════════════════════════════════════════════════════════════════════

class MonitoringStartRequest(BaseModel):
    goal_id: str = "elderly_care"
    custom_goal: str = ""


class AlertRequest(BaseModel):
    message: str


@app.post("/api/monitoring/start")
async def monitoring_start(req: MonitoringStartRequest):
    """Start autonomous monitoring mode with a goal."""
    global _monitoring_active, _monitoring_goal, _monitoring_task

    _monitoring_active = True
    _monitoring_goal = req.goal_id
    _monitoring_alerts.clear()

    # Set agent goal if available
    if agent:
        try:
            agent.set_goal_by_id(req.goal_id)
        except Exception:
            pass

    # Start monitoring background task
    if _monitoring_task is None or _monitoring_task.done():
        _monitoring_task = asyncio.create_task(_run_monitoring_loop())

    return {
        "status": "monitoring_started",
        "goal": req.goal_id,
        "agent_active": agent is not None,
        "telegram_configured": telegram_bot is not None and telegram_bot.is_configured if telegram_bot else False,
    }


@app.post("/api/monitoring/stop")
async def monitoring_stop():
    """Stop autonomous monitoring."""
    global _monitoring_active, _monitoring_task
    _monitoring_active = False
    if _monitoring_task and not _monitoring_task.done():
        _monitoring_task.cancel()
    _monitoring_task = None
    return {"status": "monitoring_stopped", "alerts_sent": len(_monitoring_alerts)}


@app.get("/api/monitoring/status")
async def monitoring_status():
    """Get monitoring status and recent alerts."""
    return {
        "active": _monitoring_active,
        "goal": _monitoring_goal,
        "alerts": _monitoring_alerts[-20:],
        "persons_detected": len(engine._tracked_persons) if engine and hasattr(engine, '_tracked_persons') else 0,
    }


@app.get("/api/frame")
async def get_frame(annotated: bool = True):
    """Return the current camera frame as JPEG with optional pose overlays."""
    from fastapi.responses import Response
    if engine is None:
        return Response(content=b"", media_type="image/jpeg", status_code=204)
    frame = engine.get_frame()
    if frame is None:
        return Response(content=b"", media_type="image/jpeg", status_code=204)

    if annotated:
        frame = _annotate_frame(frame)

    _, jpg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
    return Response(content=jpg.tobytes(), media_type="image/jpeg")


# MediaPipe pose connections for skeleton drawing
_POSE_CONNECTIONS = [
    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),  # arms
    (11, 23), (12, 24), (23, 24),  # torso
    (23, 25), (25, 27), (24, 26), (26, 28),  # legs
    (0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8),  # face
    (15, 17), (15, 19), (16, 18), (16, 20),  # hands
    (27, 29), (27, 31), (28, 30), (28, 32),  # feet
]

_ACTIVITY_COLORS = {
    "fallen": (0, 0, 255),       # red
    "lying_down": (0, 80, 255),  # orange
    "walking": (0, 255, 0),      # green
    "running": (0, 255, 255),    # yellow
    "sitting": (255, 200, 0),    # cyan
    "standing": (200, 200, 200), # gray
    "crouching": (255, 100, 0),  # blue
    "reaching": (255, 0, 255),   # magenta
}


def _annotate_frame(frame: np.ndarray) -> np.ndarray:
    """Draw pose skeletons, bounding boxes, activity labels on the frame."""
    out = frame.copy()
    h, w = out.shape[:2]

    if engine is None or not hasattr(engine, '_tracked_persons'):
        return out

    persons = engine._tracked_persons
    state = engine.get_state()

    for p in persons:
        # Bounding box
        color = _ACTIVITY_COLORS.get(p.activity, (200, 200, 200))
        x1, y1, x2, y2 = int(p.bbox.x1), int(p.bbox.y1), int(p.bbox.x2), int(p.bbox.y2)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)

        # Activity label + ID
        label = f"P{p.track_id} {p.activity or 'unknown'}"
        if p.speed > 1:
            label += f" {p.speed:.0f}px/s"
        label_bg_y = max(y1 - 28, 0)
        cv2.rectangle(out, (x1, label_bg_y), (x1 + len(label) * 10 + 8, label_bg_y + 24), color, -1)
        cv2.putText(out, label, (x1 + 4, label_bg_y + 17), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        cv2.putText(out, label, (x1 + 4, label_bg_y + 17), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Confidence bar
        conf_w = int((x2 - x1) * p.activity_confidence)
        cv2.rectangle(out, (x1, y2 + 2), (x1 + conf_w, y2 + 6), color, -1)

        # Pose skeleton — p.pose is a PoseLandmarks with .points [(x_px, y_px, vis), ...]
        pose_points = getattr(p.pose, 'points', None) if p.pose is not None else None
        if pose_points is not None and len(pose_points) >= 17:
            pts = []
            for lm in pose_points:
                if isinstance(lm, (list, tuple)) and len(lm) >= 2:
                    px, py = int(lm[0]), int(lm[1])
                else:
                    px, py = 0, 0
                pts.append((px, py))

            # Draw connections
            for i, j in _POSE_CONNECTIONS:
                if i < len(pts) and j < len(pts):
                    p1, p2 = pts[i], pts[j]
                    if p1[0] > 0 and p1[1] > 0 and p2[0] > 0 and p2[1] > 0:
                        cv2.line(out, p1, p2, color, 2, cv2.LINE_AA)

            # Draw keypoints
            for pt in pts:
                if pt[0] > 0 and pt[1] > 0:
                    cv2.circle(out, pt, 4, (255, 255, 255), -1)
                    cv2.circle(out, pt, 4, color, 1)

    # FPS + person count overlay
    if state:
        info = f"FPS: {state.get('fps', 0):.0f} | Persons: {len(persons)} | Frame: {state.get('frame_number', 0)}"
        cv2.putText(out, info, (10, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 3)
        cv2.putText(out, info, (10, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 200), 1)

    return out


@app.post("/api/alert")
async def send_alert(req: AlertRequest):
    """Send an alert via Telegram (and log it)."""
    alert_entry = {"message": req.message, "timestamp": time.time(), "sent": False}

    if telegram_bot and telegram_bot.is_configured:
        try:
            telegram_bot.send_message(f"🚨 KINETIC ALERT\n\n{req.message}")
            alert_entry["sent"] = True
        except Exception as e:
            alert_entry["error"] = str(e)
    else:
        alert_entry["note"] = "Telegram not configured — alert logged only"

    _monitoring_alerts.append(alert_entry)
    return {"status": "sent" if alert_entry.get("sent") else "logged", **alert_entry}


async def _run_monitoring_loop():
    """Background loop: monitor the scene and trigger alerts autonomously."""
    last_fall_alert = 0
    last_activity_check = 0
    last_bed_exit_alert = 0
    last_immobility_alert = 0
    last_line_alert = 0
    last_wandering_alert = 0
    last_agitation_alert = 0
    immobility_start: dict[int, float] = {}  # person_id → timestamp when immobility began
    prev_activities: dict[int, str] = {}  # person_id → last known activity

    while _monitoring_active:
        try:
            await asyncio.sleep(3)
            if not _monitoring_active or engine is None:
                continue

            persons = engine._tracked_persons if hasattr(engine, '_tracked_persons') else []
            now = time.time()
            goal = _monitoring_goal or "elderly_care"

            for person in persons:
                activity = person.activity if hasattr(person, 'activity') else None
                pid = person.track_id if hasattr(person, 'track_id') else 0
                speed = person.speed if hasattr(person, 'speed') else 0
                prev_act = prev_activities.get(pid)

                # ── FALL DETECTION (all clinical + elderly_care) ──
                if activity in ("fallen", "lying_down") and (now - last_fall_alert) > 30:
                    if goal in ("elderly_care", "bed_exit", "post_op", "wandering", "general"):
                        last_fall_alert = now
                        alert_msg = f"⚠️ FALL DETECTED — Person appears to have fallen. Activity: {activity}. Immediate attention needed."
                        _monitoring_alerts.append({"message": alert_msg, "timestamp": now, "type": "fall", "sent": False})
                        await _send_clinical_alert(alert_msg, "fall", now)

                # ── BED EXIT DETECTION ──
                if goal == "bed_exit" and (now - last_bed_exit_alert) > 20:
                    if prev_act in ("lying_down", "sitting") and activity in ("standing", "walking"):
                        last_bed_exit_alert = now
                        severity = "🚨🚨 CRITICAL" if activity == "walking" else "🚨 ALERT"
                        alert_msg = f"{severity}: Bed exit detected — patient went from {prev_act} to {activity}. Fall risk!"
                        _monitoring_alerts.append({"message": alert_msg, "timestamp": now, "type": "bed_exit", "sent": False})
                        await _send_clinical_alert(alert_msg, "bed_exit", now)

                # ── IMMOBILITY / PRESSURE ULCER ──
                if goal == "immobility":
                    if activity in ("lying_down", "sitting") and speed is not None and speed < 0.5:
                        if pid not in immobility_start:
                            immobility_start[pid] = now
                        elapsed_min = (now - immobility_start[pid]) / 60
                        if elapsed_min > 120 and (now - last_immobility_alert) > 300:
                            last_immobility_alert = now
                            alert_msg = f"🚨 REPOSITIONING NEEDED: Patient immobile for {int(elapsed_min)} minutes. Pressure ulcer risk. Activity: {activity}."
                            _monitoring_alerts.append({"message": alert_msg, "timestamp": now, "type": "immobility", "sent": False})
                            await _send_clinical_alert(alert_msg, "immobility", now)
                        elif elapsed_min > 60 and (now - last_immobility_alert) > 600:
                            last_immobility_alert = now
                            alert_msg = f"⏱️ Immobility notice: Patient in same position for {int(elapsed_min)} minutes. Consider repositioning."
                            _monitoring_alerts.append({"message": alert_msg, "timestamp": now, "type": "immobility_warning", "sent": False})
                            await _send_clinical_alert(alert_msg, "immobility_warning", now)
                    else:
                        if pid in immobility_start:
                            del immobility_start[pid]

                # ── LINE & TUBE SAFETY (agitation-based) ──
                if goal == "line_pulling" and (now - last_line_alert) > 20:
                    if activity in ("exercising",) or (activity == "lying_down" and speed is not None and speed > 2.0):
                        last_line_alert = now
                        alert_msg = f"⚠️ Line safety concern: Patient showing agitated movement (activity: {activity}, speed: {speed:.1f}). Check IV/tube integrity."
                        _monitoring_alerts.append({"message": alert_msg, "timestamp": now, "type": "line_pulling", "sent": False})
                        await _send_clinical_alert(alert_msg, "line_pulling", now)

                # ── POST-OP DISTRESS ──
                if goal == "post_op" and (now - last_agitation_alert) > 30:
                    if speed is not None and speed > 3.0 and activity in ("lying_down", "sitting"):
                        last_agitation_alert = now
                        alert_msg = f"⚠️ Post-op agitation: Patient showing elevated movement (speed: {speed:.1f}) while {activity}. Pain or distress assessment needed."
                        _monitoring_alerts.append({"message": alert_msg, "timestamp": now, "type": "post_op_agitation", "sent": False})
                        await _send_clinical_alert(alert_msg, "post_op", now)
                    elif prev_act in ("sitting", "lying_down") and activity == "fallen":
                        last_agitation_alert = now
                        alert_msg = f"🚨🚨 EMERGENCY: Post-op patient has fallen! Activity: {activity}. Surgical site at risk. Immediate response."
                        _monitoring_alerts.append({"message": alert_msg, "timestamp": now, "type": "post_op_fall", "sent": False})
                        await _send_clinical_alert(alert_msg, "post_op_fall", now)

                # ── WANDERING / ELOPEMENT ──
                if goal == "wandering" and (now - last_wandering_alert) > 20:
                    if prev_act in ("lying_down", "sitting") and activity in ("standing", "walking"):
                        last_wandering_alert = now
                        severity = "🚨 WANDERING" if activity == "walking" else "⚠️ Out of bed"
                        alert_msg = f"{severity}: Patient is now {activity} (was {prev_act}). Elopement risk — check on patient."
                        _monitoring_alerts.append({"message": alert_msg, "timestamp": now, "type": "wandering", "sent": False})
                        await _send_clinical_alert(alert_msg, "wandering", now)

                # Track previous activity for transition detection
                if activity:
                    prev_activities[pid] = activity

            # ── WANDERING: person disappeared from view ──
            if goal == "wandering" and (now - last_wandering_alert) > 20:
                if len(persons) == 0 and prev_activities:
                    last_wandering_alert = now
                    alert_msg = "🚨🚨 ELOPEMENT ALERT: Patient has left the monitored area. No person detected. Locate patient NOW."
                    _monitoring_alerts.append({"message": alert_msg, "timestamp": now, "type": "elopement", "sent": False})
                    await _send_clinical_alert(alert_msg, "elopement", now)

            # Periodic scene check via agent (every 30s)
            if agent and (now - last_activity_check) > 30 and persons:
                last_activity_check = now
                activities = [getattr(p, 'activity', 'unknown') for p in persons]
                try:
                    response = await agent.send_message(
                        f"Monitoring check ({goal} mode): {len(persons)} person(s) detected. "
                        f"Activities: {', '.join(activities)}. "
                        "Assess per your goal instructions. If concerning, alert via Telegram."
                    )
                    _monitoring_alerts.append({
                        "message": f"[Agent Check] {response[:200]}",
                        "timestamp": now,
                        "type": "check",
                        "sent": False,
                    })
                except Exception:
                    pass

        except asyncio.CancelledError:
            break
        except Exception as e:
            print(f"[Monitor] Error in monitoring loop: {e}")
            await asyncio.sleep(5)


async def _send_clinical_alert(alert_msg: str, alert_type: str, now: float):
    """Send alert via Telegram (with photo) + voice + agent analysis."""
    # Telegram with photo
    if telegram_bot and telegram_bot.is_configured:
        snapshot_path = None
        if engine and hasattr(engine, '_last_frame') and engine._last_frame is not None:
            snapshot_path = os.path.join(config.SNAPSHOTS_DIR, f"{alert_type}_{int(now)}.jpg")
            try:
                cv2.imwrite(snapshot_path, engine._last_frame)
            except Exception:
                snapshot_path = None
        try:
            telegram_bot.send_message(alert_msg, photo_path=snapshot_path)
            if _monitoring_alerts:
                _monitoring_alerts[-1]["sent"] = True
        except Exception:
            pass

    # Voice alert for critical
    if openai_voice and openai_voice.is_connected:
        try:
            voice_msg = alert_msg.replace("🚨🚨", "").replace("🚨", "").replace("⚠️", "").replace("⏱️", "").strip()
            await openai_voice.speak(f"Clinical alert: {voice_msg[:150]}")
        except Exception:
            pass

    # Agent analysis
    if agent:
        try:
            await agent.send_message(
                f"CLINICAL ALERT ({alert_type}): {alert_msg}. "
                "Assess the situation per your goal instructions. Use speak_to_user if needed."
            )
        except Exception:
            pass


@app.get("/api/voice/status")
async def voice_status():
    """Voice bridge status (OpenAI Realtime or Gemini Live)."""
    if openai_voice is not None:
        return {
            "available": True,
            "engine": "OpenAI Realtime (GPT-4o)",
            "connected": openai_voice.is_connected,
            "voice": "alloy",
        }
    return {
        "available": gemini_bridge is not None,
        "engine": "Gemini Live",
        "connected": gemini_bridge.is_connected if gemini_bridge else False,
        "model": config.GEMINI_MODEL,
        "voice": config.GEMINI_VOICE,
    }


# ── WebSocket: Real-time coaching data stream ─────────────────────────

@app.websocket("/ws/coaching")
async def ws_coaching(websocket: WebSocket):
    """
    Real-time coaching data stream for Next.js frontend.

    Server → Client messages:
      {"type": "coaching", "data": {score, reps, joint_deviations, ...}}
      {"type": "angles", "data": {angles, expert_angles}}
      {"type": "rep_complete", "data": {rep_number, score}}

    Client → Server messages:
      {"type": "ping"} → {"type": "pong"}
    """
    await websocket.accept()
    _coaching_ws_clients.append(websocket)
    print(f"[Server] Coaching WS client connected ({len(_coaching_ws_clients)} total)")

    try:
        while True:
            msg = await websocket.receive_text()
            data = json.loads(msg)
            if data.get("type") == "ping":
                await websocket.send_text(json.dumps({"type": "pong"}))
    except WebSocketDisconnect:
        pass
    except Exception as e:
        print(f"[Server] Coaching WS error: {e}")
    finally:
        if websocket in _coaching_ws_clients:
            _coaching_ws_clients.remove(websocket)
        print(f"[Server] Coaching WS client disconnected ({len(_coaching_ws_clients)} remaining)")


# ── MCP over HTTP (for Poke integration) ──────────────────────────
# Exposes the MCP server at /mcp so Poke (or any MCP client) can connect
# via Streamable HTTP / SSE transport.
try:
    from aegis.mcp_server import mcp as _mcp_server
    _mcp_http = _mcp_server.http_app()
    app.mount("/mcp", _mcp_http)
    print("[MCP/HTTP] Mounted at /mcp — Poke can connect to http://HOST:8000/mcp")
except Exception as _e:
    print(f"[MCP/HTTP] Could not mount: {_e}")

# ── Static files ────────────────────────────────────────────────────

os.makedirs(STATIC_DIR, exist_ok=True)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
