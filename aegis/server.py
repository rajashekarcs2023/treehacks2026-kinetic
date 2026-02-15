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

# ── Coaching state ────────────────────────────────────────────────────
_reference_store = ReferenceStore()
_recording_session: RecordingSession | None = None
_coaching_session: CoachingSession | None = None
_coaching_ws_clients: list[WebSocket] = []  # connected coaching WS clients
_graph_store = GraphStore()
_data_collector = DataCollector()
_hybrid_scorer = HybridScorer()
_memory_store = MemoryStore()

STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")

# ── Claude Agent SDK coaching intelligence loop ──────────────────────────
_coaching_intelligence_task: asyncio.Task | None = None
_last_agent_rep_count = 0

async def _run_coaching_intelligence():
    """Background loop: Claude Agent SDK analyzes coaching every ~10 seconds.
    
    Gathers coaching data directly from server state and passes it to Claude
    for intelligent analysis. Claude responds with ONLY the coaching feedback.
    Feedback is shown on screen + spoken via Gemini Live.
    """
    global _last_agent_rep_count
    _last_agent_rep_count = 0
    check_count = 0
    
    # Wait for coaching to get going
    await asyncio.sleep(6)
    
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
            
            if check_count == 1:
                prompt = (
                    f"You are a real-time AI skill coach. A session just started.\n"
                    f"{data_block}\n\n"
                    "Respond with ONLY a brief encouraging opening (1 sentence). "
                    "Example: 'Let's work on your boxing jab — I'll be watching your form!'\n"
                    "NO reasoning, NO tool calls, JUST the coaching sentence."
                )
            elif current_reps > _last_agent_rep_count:
                _last_agent_rep_count = current_reps
                prompt = (
                    f"You are a real-time AI skill coach. The user just finished rep {current_reps}.\n"
                    f"{data_block}\n\n"
                    "Give specific, actionable form feedback (1-2 sentences max). "
                    "Be encouraging but specific about what to fix.\n"
                    "Example: 'Good power on that jab! Try rotating your hips more for extra reach.'\n"
                    "NO reasoning, JUST the coaching feedback."
                )
            else:
                if trend == "declining":
                    prompt = (
                        f"You are a real-time AI skill coach. Scores are declining.\n"
                        f"{data_block}\n\n"
                        "Give a motivating correction (1 sentence). "
                        "Example: 'I notice your form dropping — focus on keeping your guard up.'\n"
                        "NO reasoning, JUST the coaching sentence."
                    )
                else:
                    # Skip periodic if nothing interesting
                    await asyncio.sleep(10)
                    continue
            
            response = await agent.send_message(prompt)
            
            # Clean response — extract only the coaching line
            if response:
                # Remove any reasoning prefixes Claude might add
                speech_text = response.strip()
                for prefix in ["Here's", "Sure,", "Okay,", "I'll", "Let me", "Based on"]:
                    if speech_text.startswith(prefix) and ":" in speech_text[:60]:
                        speech_text = speech_text.split(":", 1)[1].strip()
                # Remove quotes if wrapped
                speech_text = speech_text.strip('"').strip("'")
                # Limit length
                speech_text = speech_text[:180]
                
                if speech_text and len(speech_text) > 5:
                    print(f"[Agent] Coach says: {speech_text}")
                    
                    # Send to frontend as text overlay
                    if _coaching_ws_clients:
                        agent_msg = json.dumps({
                            "type": "agent_feedback",
                            "data": speech_text,
                            "check": check_count,
                        })
                        for client in list(_coaching_ws_clients):
                            try:
                                await client.send_text(agent_msg)
                            except Exception:
                                pass
                    
                    # Speak via Gemini Live (if connected) or macOS say fallback
                    if gemini_bridge and gemini_bridge.is_connected:
                        try:
                            await gemini_bridge.speak(speech_text)
                        except Exception:
                            pass
                    else:
                        try:
                            import subprocess
                            subprocess.Popen(
                                ["say", "-r", "185", speech_text],
                                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                            )
                        except Exception:
                            pass
            
        except Exception as e:
            print(f"[Agent] Coaching intelligence error: {e}")
        
        # Wait before next check
        await asyncio.sleep(10)

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
    Returns: session status with loaded reference info.
    """
    global _coaching_session

    ref = None
    if req.reference_name:
        ref = _reference_store.load(req.reference_name)
        if ref is None:
            return {"error": f"Reference '{req.reference_name}' not found"}

    _coaching_session = CoachingSession(skill_name=req.skill_name, reference=ref)
    _coaching_session.set_primary_angle(req.primary_angle or "left_knee")

    # Start Claude Agent SDK coaching intelligence (if agent available)
    _start_coaching_intelligence()

    return {
        "status": "session_started",
        "skill": req.skill_name,
        "reference_loaded": req.reference_name if ref else None,
        "primary_angle": req.primary_angle,
        "available_angles": list(KEY_ANGLES.keys()),
        "agent_active": agent is not None,
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


# ── WebSocket: Audio proxy to Gemini Live ────────────────────────────

@app.websocket("/ws/audio")
async def ws_audio(websocket: WebSocket):
    """
    Phone sends mic audio, server proxies to Gemini Live, returns audio.
    
    Protocol:
      Phone → Server: raw bytes (16kHz 16-bit PCM) or {"type":"audio","data":"<base64 pcm>"}
      Server → Phone: raw bytes (24kHz 16-bit PCM) or {"type":"audio","data":"<base64 pcm>"}
      Phone → Server: {"type":"text","data":"<text message>"}
      Phone → Server: {"type":"spatial","data":<state_dict>}
    """
    await websocket.accept()
    print("[Server] Audio client connected")

    if gemini_bridge is None:
        await websocket.send_text(json.dumps({"type": "error", "data": "Voice bridge not initialized"}))
        await websocket.close()
        return

    # Connect Gemini if not already
    if not gemini_bridge.is_connected:
        ok = await gemini_bridge.connect()
        if not ok:
            await websocket.send_text(json.dumps({"type": "error", "data": "Gemini connection failed"}))
            await websocket.close()
            return

    # Task: forward Gemini audio output → phone
    async def send_audio_to_phone():
        try:
            while True:
                audio_bytes = await gemini_bridge.get_audio_output(timeout=0.1)
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

    send_task = asyncio.create_task(send_audio_to_phone())

    try:
        while True:
            msg = await websocket.receive_text()
            data = json.loads(msg)

            if data.get("type") == "audio":
                # Base64-encoded PCM audio from phone
                pcm_bytes = base64.b64decode(data["data"])
                await gemini_bridge.send_audio(pcm_bytes)

            elif data.get("type") == "text":
                # Text message to Gemini
                await gemini_bridge.send_text(data["data"])

            elif data.get("type") == "spatial":
                # Spatial state injection
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


@app.get("/api/voice/status")
async def voice_status():
    """Gemini Live voice bridge status."""
    return {
        "available": gemini_bridge is not None,
        "connected": gemini_bridge.is_connected if gemini_bridge else False,
        "model": config.GEMINI_MODEL,
        "voice": config.GEMINI_VOICE,
        "api_key_set": bool(config.GEMINI_API_KEY),
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


# ── Static files ────────────────────────────────────────────────────

os.makedirs(STATIC_DIR, exist_ok=True)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
