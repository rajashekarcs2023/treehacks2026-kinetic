"""
AEGIS MCP Server — 40 tools for AI skill coaching + spatial intelligence.

Exposes the full spatial intelligence and skill coaching stack via MCP.
The Claude agent connects to this server (in-process or via HTTP) to
discover and execute tools dynamically based on the user's goal.

Categories:
  1. Perception   (6 tools) — reading the scene
  2. Pose         (3 tools) — body understanding
  3. Activity     (3 tools) — timeline & stats
  4. Zones        (3 tools) — spatial monitoring
  5. Alerts       (3 tools) — communicating with user
  6. Memory       (2 tools) — persistent observations
  7. Knowledge    (2 tools) — external info
  8. Goals        (3 tools) — goal management
  9. Skill Ref    (4 tools) — expert reference management
 10. Skill Compare(5 tools) — pose comparison & analysis
 11. Skill Coach  (4 tools) — coaching session management
 12. Skill Intel  (2 tools) — zero-shot & document parsing
"""

import json
import math
import time
import os
from datetime import datetime
from typing import Optional, Annotated

from fastmcp import FastMCP

from aegis import config
from aegis.pose_comparison import (
    normalize_skeleton, compute_joint_angles, compare_poses, quick_compare,
    compare_sequences, compute_movement_quality, detect_compensation,
    detect_phases, count_reps_from_angles, dtw_align,
    ReferenceStore, RecordingSession, CoachingSession,
    NormalizedSkeleton, SkeletonSequence, KEY_ANGLES, ANGLE_NAMES,
)

# ── FastMCP server instance ─────────────────────────────────────────────
mcp = FastMCP(
    name="aegis-spatial",
    instructions=(
        "AEGIS skill coaching + spatial intelligence server. Provides 40 tools for "
        "real-time skill coaching with expert motion transfer, pose comparison, "
        "movement quality analysis, coaching sessions, plus spatial perception, "
        "activity tracking, alerts, memory, knowledge, and goal management."
    ),
)

# ── Shared state (set by run_server.py before agent connects) ───────────
_engine = None          # SpatialEngine instance
_telegram_sender = None  # callable(message, photo_path=None)
_agent_ref = None        # AegisAgent back-reference (for goal management)

# ── Internal stores ─────────────────────────────────────────────────────
_observations: list[dict] = []      # saved observations (memory)
_activity_history: list[dict] = []  # activity timeline per person
_prev_state: dict | None = None     # previous state for scene changes
_prev_state_time: float = 0.0
_session_start: float = time.time()
_session_stats: dict = {
    "people_seen": set(),
    "alerts_sent": 0,
    "tool_calls": 0,
    "goal_changes": 0,
}

# ── Skill coaching state ──────────────────────────────────────────────
_reference_store: ReferenceStore = ReferenceStore()
_recording_session: RecordingSession | None = None
_coaching_session: CoachingSession | None = None


def init(engine, telegram_sender=None, agent=None):
    """Initialize shared state. Called once at startup."""
    global _engine, _telegram_sender, _agent_ref, _session_start
    _engine = engine
    _telegram_sender = telegram_sender
    _agent_ref = agent
    _session_start = time.time()


# ═══════════════════════════════════════════════════════════════════════
# CATEGORY 1: PERCEPTION — Reading the Scene (6 tools)
# ═══════════════════════════════════════════════════════════════════════

@mcp.tool(tags={"perception"}, annotations={"readOnlyHint": True})
def get_spatial_state() -> dict:
    """Get the full current spatial state as JSON.

    Returns everything: persons (with positions, velocities, predicted
    trajectories, activities), detected objects (80 COCO classes),
    danger zones, and active risk events.
    """
    _inc_tool_calls()
    if _engine is None:
        return {"error": "Engine not initialized"}
    state = _engine.get_state()
    _update_tracking(state)
    return state


@mcp.tool(tags={"perception"}, annotations={"readOnlyHint": True})
def get_spatial_summary() -> str:
    """Get a concise human-readable summary of the current scene.

    Returns a short paragraph describing: number of people, their
    activities, detected objects, active zones, and any risk events.
    Good for quick status checks without parsing full JSON.
    """
    _inc_tool_calls()
    if _engine is None:
        return "Engine not initialized"
    return _engine.get_summary()


@mcp.tool(tags={"perception"}, annotations={"readOnlyHint": True})
def get_person_detail(
    track_id: Annotated[int, "The persistent tracking ID of the person to inspect"],
) -> dict:
    """Get deep detail about a specific tracked person.

    Returns: bounding box, center position, velocity, speed, direction,
    predicted position, activity, activity confidence, pose availability,
    depth estimate, and time in scene.
    """
    _inc_tool_calls()
    if _engine is None:
        return {"error": "Engine not initialized"}
    state = _engine.get_state()
    for p in state.get("persons", []):
        if p["track_id"] == track_id:
            return p
    return {"error": f"Person with track_id={track_id} not found in current frame"}


@mcp.tool(tags={"perception"}, annotations={"readOnlyHint": True})
def get_scene_changes(
    seconds_back: Annotated[int, "How many seconds back to compare (default 10)"] = 10,
) -> dict:
    """Detect what changed in the scene recently.

    Returns: new people entered, people who left, activity changes,
    new objects, objects gone. Useful for monitoring without polling
    the full state constantly.
    """
    _inc_tool_calls()
    global _prev_state, _prev_state_time
    if _engine is None:
        return {"error": "Engine not initialized"}

    current = _engine.get_state()
    now = time.time()

    if _prev_state is None or (now - _prev_state_time) > seconds_back:
        _prev_state = current
        _prev_state_time = now
        return {
            "changes": "First check — no previous state to compare",
            "current_persons": len(current.get("persons", [])),
            "current_objects": len(current.get("objects", [])),
        }

    prev_ids = {p["track_id"] for p in _prev_state.get("persons", [])}
    curr_ids = {p["track_id"] for p in current.get("persons", [])}
    new_people = curr_ids - prev_ids
    left_people = prev_ids - curr_ids

    # Activity changes
    prev_activities = {p["track_id"]: p.get("activity") for p in _prev_state.get("persons", [])}
    activity_changes = []
    for p in current.get("persons", []):
        tid = p["track_id"]
        if tid in prev_activities and prev_activities[tid] != p.get("activity"):
            activity_changes.append({
                "track_id": tid,
                "from": prev_activities[tid],
                "to": p.get("activity"),
            })

    # Object changes
    prev_objs = {o["class_name"] for o in _prev_state.get("objects", [])}
    curr_objs = {o["class_name"] for o in current.get("objects", [])}

    result = {
        "new_people_entered": list(new_people),
        "people_left": list(left_people),
        "activity_changes": activity_changes,
        "new_object_classes": list(curr_objs - prev_objs),
        "gone_object_classes": list(prev_objs - curr_objs),
        "current_person_count": len(curr_ids),
        "current_object_count": len(current.get("objects", [])),
        "risk_events": current.get("risk_events", []),
    }

    _prev_state = current
    _prev_state_time = now
    return result


@mcp.tool(tags={"perception"}, annotations={"readOnlyHint": True})
def get_objects_in_scene(
    class_filter: Annotated[Optional[str], "Filter by object class name (e.g. 'cell phone', 'laptop'). None = all objects."] = None,
) -> list[dict]:
    """List all detected objects in the scene.

    YOLO detects 80 COCO classes including: cell phone, laptop, book,
    cup, bottle, chair, backpack, remote, keyboard, mouse, etc.
    Optionally filter by class name.
    """
    _inc_tool_calls()
    if _engine is None:
        return [{"error": "Engine not initialized"}]
    state = _engine.get_state()
    objects = state.get("objects", [])
    if class_filter:
        objects = [o for o in objects if class_filter.lower() in o["class_name"].lower()]
    return objects


@mcp.tool(tags={"perception"}, annotations={"readOnlyHint": True})
def count_objects(
    class_name: Annotated[str, "The object class to count (e.g. 'chair', 'cell phone', 'person')"],
) -> dict:
    """Count how many of a specific object class are visible.

    Supports all 80 COCO classes. Use 'person' to count people,
    'cell phone' for phones, 'laptop' for laptops, etc.
    """
    _inc_tool_calls()
    if _engine is None:
        return {"error": "Engine not initialized"}
    state = _engine.get_state()

    if class_name.lower() == "person":
        count = len(state.get("persons", []))
    else:
        count = sum(
            1 for o in state.get("objects", [])
            if class_name.lower() in o["class_name"].lower()
        )
    return {"class_name": class_name, "count": count}


# ═══════════════════════════════════════════════════════════════════════
# CATEGORY 2: POSE ANALYSIS — Body Understanding (3 tools)
# ═══════════════════════════════════════════════════════════════════════

@mcp.tool(tags={"pose"}, annotations={"readOnlyHint": True})
def analyze_posture(
    track_id: Annotated[int, "Tracking ID of the person to analyze"],
) -> dict:
    """Analyze a person's posture from their skeletal landmarks.

    Returns computed metrics: shoulder alignment (level vs tilted),
    spine angle, head tilt, knee angles, and an overall posture score.
    Useful for posture coaching, ergonomics, exercise form checking.
    """
    _inc_tool_calls()
    if _engine is None:
        return {"error": "Engine not initialized"}

    state = _engine.get_state()
    person = None
    for p in state.get("persons", []):
        if p["track_id"] == track_id:
            person = p
            break
    if person is None:
        return {"error": f"Person {track_id} not found"}
    if not person.get("has_pose"):
        return {"error": f"Person {track_id} has no pose data in current frame"}

    # Access raw pose from the engine's internal tracked persons
    landmarks = _get_raw_landmarks(track_id)
    if landmarks is None:
        return {"error": "Could not retrieve pose landmarks"}

    return _compute_posture_metrics(landmarks)


@mcp.tool(tags={"pose"}, annotations={"readOnlyHint": True})
def get_pose_landmarks(
    track_id: Annotated[int, "Tracking ID of the person"],
) -> dict:
    """Get the raw 33 MediaPipe pose landmarks for a person.

    Each landmark has: index, name, x (pixel), y (pixel), visibility.
    Use this for custom pose analysis that analyze_posture doesn't cover.

    Landmark indices: 0=nose, 11/12=shoulders, 13/14=elbows,
    15/16=wrists, 23/24=hips, 25/26=knees, 27/28=ankles.
    """
    _inc_tool_calls()
    if _engine is None:
        return {"error": "Engine not initialized"}

    landmarks = _get_raw_landmarks(track_id)
    if landmarks is None:
        return {"error": f"No pose landmarks for person {track_id}"}

    LANDMARK_NAMES = {
        0: "nose", 1: "left_eye_inner", 2: "left_eye", 3: "left_eye_outer",
        4: "right_eye_inner", 5: "right_eye", 6: "right_eye_outer",
        7: "left_ear", 8: "right_ear", 9: "mouth_left", 10: "mouth_right",
        11: "left_shoulder", 12: "right_shoulder", 13: "left_elbow",
        14: "right_elbow", 15: "left_wrist", 16: "right_wrist",
        17: "left_pinky", 18: "right_pinky", 19: "left_index",
        20: "right_index", 21: "left_thumb", 22: "right_thumb",
        23: "left_hip", 24: "right_hip", 25: "left_knee", 26: "right_knee",
        27: "left_ankle", 28: "right_ankle", 29: "left_heel",
        30: "right_heel", 31: "left_foot_index", 32: "right_foot_index",
    }

    result = []
    for i, (x, y, vis) in enumerate(landmarks):
        result.append({
            "index": i,
            "name": LANDMARK_NAMES.get(i, f"landmark_{i}"),
            "x": round(x, 1),
            "y": round(y, 1),
            "visibility": round(vis, 3),
        })
    return {"track_id": track_id, "landmarks": result}


@mcp.tool(tags={"pose"}, annotations={"readOnlyHint": True})
def get_hand_landmarks() -> dict:
    """Get hand landmarks for all detected hands (21 landmarks per hand).

    Each hand has 21 landmarks covering all finger joints:
    0=wrist, 1-4=thumb(CMC,MCP,IP,TIP), 5-8=index(MCP,PIP,DIP,TIP),
    9-12=middle, 13-16=ring, 17-20=pinky.
    Returns handedness (Left/Right) and all landmark positions.
    Use this for sign language coaching and hand gesture analysis.
    """
    _inc_tool_calls()
    if _engine is None:
        return {"error": "Engine not initialized"}

    state = _engine.get_state()
    hands = state.get("hands", [])
    if not hands:
        return {"hands_detected": 0, "hands": [], "note": "No hands detected in frame"}

    FINGER_NAMES = {
        0: "wrist",
        1: "thumb_cmc", 2: "thumb_mcp", 3: "thumb_ip", 4: "thumb_tip",
        5: "index_mcp", 6: "index_pip", 7: "index_dip", 8: "index_tip",
        9: "middle_mcp", 10: "middle_pip", 11: "middle_dip", 12: "middle_tip",
        13: "ring_mcp", 14: "ring_pip", 15: "ring_dip", 16: "ring_tip",
        17: "pinky_mcp", 18: "pinky_pip", 19: "pinky_dip", 20: "pinky_tip",
    }

    result = []
    for hand in hands:
        landmarks = []
        for i, lm in enumerate(hand.get("landmarks", [])):
            landmarks.append({
                "index": i,
                "name": FINGER_NAMES.get(i, f"landmark_{i}"),
                "x": lm["x"],
                "y": lm["y"],
                "z": lm["z"],
            })
        result.append({
            "handedness": hand.get("handedness", "unknown"),
            "landmarks": landmarks,
        })

    return {"hands_detected": len(result), "hands": result}


@mcp.tool(tags={"pose"}, annotations={"readOnlyHint": True})
def check_body_alignment(
    track_id: Annotated[int, "Tracking ID of the person"],
    exercise: Annotated[Optional[str], "Optional exercise name for context-specific checks (e.g. 'squat', 'warrior_ii', 'plank')"] = None,
) -> dict:
    """Check body alignment and return deviations from ideal posture.

    Analyzes symmetry (left vs right side), joint angles, and common
    misalignments. If an exercise name is provided, focuses on the
    key angles for that movement.
    """
    _inc_tool_calls()
    if _engine is None:
        return {"error": "Engine not initialized"}

    landmarks = _get_raw_landmarks(track_id)
    if landmarks is None:
        return {"error": f"No pose landmarks for person {track_id}"}

    metrics = _compute_posture_metrics(landmarks)
    issues = []

    # Check shoulder symmetry
    if abs(metrics.get("shoulder_tilt_deg", 0)) > 8:
        issues.append(f"Shoulders uneven — {metrics['shoulder_tilt_deg']:.1f}° tilt")

    # Check head forward position
    if metrics.get("head_forward_offset", 0) > 30:
        issues.append(f"Head is forward — {metrics['head_forward_offset']:.0f}px ahead of shoulders")

    # Check spine angle
    spine = metrics.get("spine_angle_deg", 0)
    if spine > 15:
        issues.append(f"Spine is leaning — {spine:.1f}° from vertical")

    # Exercise-specific checks
    if exercise:
        if exercise.lower() in ("squat", "squats"):
            l_knee = metrics.get("left_knee_angle")
            r_knee = metrics.get("right_knee_angle")
            if l_knee and l_knee > 100:
                issues.append(f"Left knee not bent enough ({l_knee:.0f}°, target ~90°)")
            if r_knee and r_knee > 100:
                issues.append(f"Right knee not bent enough ({r_knee:.0f}°, target ~90°)")

    return {
        "track_id": track_id,
        "exercise": exercise,
        "metrics": metrics,
        "issues": issues,
        "alignment_score": max(0, 100 - len(issues) * 20),
    }


# ═══════════════════════════════════════════════════════════════════════
# CATEGORY 3: ACTIVITY & TIMELINE (3 tools)
# ═══════════════════════════════════════════════════════════════════════

@mcp.tool(tags={"activity"}, annotations={"readOnlyHint": True})
def get_activity_timeline(
    track_id: Annotated[int, "Tracking ID of the person"],
    last_minutes: Annotated[int, "How many minutes of history to return"] = 5,
) -> list[dict]:
    """Get the activity timeline for a specific person.

    Returns a list of activity segments: [{activity, started_at,
    ended_at, duration_seconds}]. Shows what the person has been doing
    over time. Useful for focus tracking, break detection, etc.
    """
    _inc_tool_calls()
    cutoff = time.time() - (last_minutes * 60)
    timeline = [
        e for e in _activity_history
        if e["track_id"] == track_id and e["timestamp"] >= cutoff
    ]
    return timeline


@mcp.tool(tags={"activity"}, annotations={"readOnlyHint": True})
def get_time_in_activity(
    track_id: Annotated[int, "Tracking ID of the person"],
    activity: Annotated[str, "Activity to measure (standing, sitting, walking, running, fallen, etc.)"],
) -> dict:
    """Get how long a person has been doing a specific activity.

    Returns total seconds in that activity and whether they are
    currently doing it. Useful for: "How long has the student been
    sitting?" or "How long has the driver been still?"
    """
    _inc_tool_calls()
    total = 0.0
    currently = False

    state = _engine.get_state() if _engine else {}
    for p in state.get("persons", []):
        if p["track_id"] == track_id:
            currently = p.get("activity", "").lower() == activity.lower()
            break

    for e in _activity_history:
        if e["track_id"] == track_id and e["activity"].lower() == activity.lower():
            total += e.get("duration", 0)

    return {
        "track_id": track_id,
        "activity": activity,
        "total_seconds": round(total, 1),
        "currently_active": currently,
    }


@mcp.tool(tags={"activity"}, annotations={"readOnlyHint": True})
def get_session_stats() -> dict:
    """Get aggregate statistics for the current monitoring session.

    Returns: session duration, unique people seen, total alerts sent,
    total tool calls, goal changes, and current goal info.
    """
    _inc_tool_calls()
    elapsed = time.time() - _session_start
    goal_info = None
    if _agent_ref:
        g = _agent_ref.active_goal
        goal_info = {"goal_id": g.goal_id, "name": g.name, "icon": g.icon}

    return {
        "session_duration_seconds": round(elapsed, 1),
        "session_duration_human": _format_duration(elapsed),
        "unique_people_seen": len(_session_stats["people_seen"]),
        "alerts_sent": _session_stats["alerts_sent"],
        "tool_calls": _session_stats["tool_calls"],
        "goal_changes": _session_stats["goal_changes"],
        "current_goal": goal_info,
        "observations_saved": len(_observations),
    }


# ═══════════════════════════════════════════════════════════════════════
# CATEGORY 4: ZONES — Spatial Monitoring (3 tools)
# ═══════════════════════════════════════════════════════════════════════

@mcp.tool(tags={"zones"})
def set_watch_zone(
    x1: Annotated[int, "Left edge (pixels)"],
    y1: Annotated[int, "Top edge (pixels)"],
    x2: Annotated[int, "Right edge (pixels)"],
    y2: Annotated[int, "Bottom edge (pixels)"],
    label: Annotated[str, "Human-readable label for this zone"] = "Watch Zone",
) -> dict:
    """Define a rectangular zone to monitor for intrusions.

    When a person enters or approaches this zone, the CV pipeline
    generates risk events with time-to-collision estimates.
    Coordinates are in pixels (frame is typically 640x480).
    """
    _inc_tool_calls()
    if _engine is None:
        return {"error": "Engine not initialized"}
    zone_id = _engine.add_danger_zone(x1, y1, x2, y2, label)
    return {"zone_id": zone_id, "label": label, "bounds": {"x1": x1, "y1": y1, "x2": x2, "y2": y2}}


@mcp.tool(tags={"zones"})
def clear_watch_zones() -> dict:
    """Remove all defined watch/danger zones."""
    _inc_tool_calls()
    if _engine is None:
        return {"error": "Engine not initialized"}
    _engine.clear_danger_zones()
    return {"status": "All watch zones cleared"}


@mcp.tool(tags={"zones"}, annotations={"readOnlyHint": True})
def check_zone_status(
    zone_id: Annotated[Optional[str], "Specific zone ID to check, or None for all zones"] = None,
) -> dict:
    """Check who/what is in or near the watched zones.

    Returns zone info plus any people or objects currently inside
    the zone boundaries and any active risk events for the zone.
    """
    _inc_tool_calls()
    if _engine is None:
        return {"error": "Engine not initialized"}

    state = _engine.get_state()
    zones = state.get("danger_zones", [])
    persons = state.get("persons", [])
    risks = state.get("risk_events", [])

    if zone_id:
        zones = [z for z in zones if z["zone_id"] == zone_id]

    result = []
    for z in zones:
        zb = z["bbox"]
        people_in = []
        for p in persons:
            pc = p["center"]
            if zb["x1"] <= pc["x"] <= zb["x2"] and zb["y1"] <= pc["y"] <= zb["y2"]:
                people_in.append({
                    "track_id": p["track_id"],
                    "activity": p.get("activity"),
                    "speed": p.get("speed_px_per_sec", 0),
                })
        zone_risks = [r for r in risks if r.get("zone_id") == z["zone_id"]]
        result.append({
            "zone_id": z["zone_id"],
            "label": z["label"],
            "bounds": zb,
            "people_inside": people_in,
            "risk_events": zone_risks,
        })
    return {"zones": result}


# ═══════════════════════════════════════════════════════════════════════
# CATEGORY 5: ALERTS — Communicating with User (3 tools)
# ═══════════════════════════════════════════════════════════════════════

@mcp.tool(tags={"alerts"})
def send_telegram_alert(
    message: Annotated[str, "The alert message to send to the user"],
    include_photo: Annotated[bool, "Whether to capture and attach a photo"] = False,
) -> dict:
    """Send an alert message to the user via Telegram.

    Use for important updates, warnings, and answers to questions.
    Set include_photo=true to capture a snapshot and send it along
    with the message.
    """
    _inc_tool_calls()
    _session_stats["alerts_sent"] += 1
    photo_path = None
    if include_photo and _engine:
        photo_path = _engine.capture_snapshot()

    if _telegram_sender:
        _telegram_sender(message, photo_path)
        result = f"Alert sent: \"{message}\""
        if photo_path:
            result += " (with photo)"
        return {"status": "sent", "message": message, "photo": photo_path is not None}
    return {"status": "no_telegram", "message": message, "note": "Telegram not configured — message logged only"}


@mcp.tool(tags={"alerts"})
def speak_to_user(
    message: Annotated[str, "What to say to the user"],
    urgency: Annotated[str, "Urgency level: 'low', 'normal', 'high', 'critical'"] = "normal",
) -> dict:
    """Speak a message to the user through the device speaker.

    Uses Gemini Live for natural speech when available, with macOS say
    as fallback. Urgency affects speech rate (fallback) and prefixed
    urgency cue (Gemini).
    """
    _inc_tool_calls()

    # Try Gemini Live bridge first (natural voice)
    try:
        import aegis.server as srv
        if srv.gemini_bridge and srv.gemini_bridge.is_connected:
            import asyncio
            urgency_prefix = ""
            if urgency == "high":
                urgency_prefix = "Say this with urgency: "
            elif urgency == "critical":
                urgency_prefix = "Say this with extreme urgency, as a critical alert: "
            asyncio.run_coroutine_threadsafe(
                srv.gemini_bridge.speak(f"{urgency_prefix}{message}"),
                asyncio.get_event_loop(),
            )
            return {"status": "speaking_gemini", "message": message, "urgency": urgency}
    except Exception:
        pass

    # Fallback: macOS TTS
    rate_map = {"low": 160, "normal": 185, "high": 210, "critical": 240}
    rate = rate_map.get(urgency, 185)

    try:
        import subprocess
        subprocess.Popen(
            ["say", "-r", str(rate), message],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        return {"status": "speaking_tts", "message": message, "urgency": urgency}
    except Exception as e:
        return {"status": "tts_error", "error": str(e), "message": message}


@mcp.tool(tags={"alerts"})
def capture_photo(
    annotated: Annotated[bool, "If true, include bounding box overlays on the photo"] = True,
) -> dict:
    """Capture a snapshot from the camera.

    Returns the file path of the saved image. Use annotated=true
    to include bounding boxes and labels in the image.
    """
    _inc_tool_calls()
    if _engine is None:
        return {"error": "Engine not initialized"}

    if annotated:
        frame = _engine.get_display_frame()
    else:
        frame = _engine.get_frame()

    if frame is None:
        return {"error": "No frame available (camera may not be ready)"}

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filename = f"snapshot_{timestamp}.jpg"
    filepath = os.path.join(config.SNAPSHOTS_DIR, filename)

    import cv2
    cv2.imwrite(filepath, frame)
    return {"status": "captured", "file_path": filepath, "annotated": annotated}


# ═══════════════════════════════════════════════════════════════════════
# CATEGORY 6: MEMORY — Persistent Observations (2 tools)
# ═══════════════════════════════════════════════════════════════════════

@mcp.tool(tags={"memory"})
def save_observation(
    text: Annotated[str, "The observation to save (e.g. 'Person 1 keeps rubbing eyes')"],
    tags: Annotated[list[str], "Optional tags for categorization"] = [],
) -> dict:
    """Save a note/observation for later reference.

    Use this to build up context over time. For example:
    'Person 1 has been sitting for 45 minutes without moving'
    or 'Phone detected near student 3 times in last 10 minutes'.
    """
    _inc_tool_calls()
    obs = {
        "id": f"obs_{len(_observations)}",
        "text": text,
        "tags": tags,
        "timestamp": datetime.now().isoformat(),
        "epoch": time.time(),
    }
    _observations.append(obs)
    return {"status": "saved", "observation_id": obs["id"]}


@mcp.tool(tags={"memory"}, annotations={"readOnlyHint": True})
def get_observations(
    tag_filter: Annotated[Optional[str], "Filter by tag (e.g. 'drowsiness', 'posture')"] = None,
    last_n: Annotated[int, "Number of recent observations to return"] = 20,
) -> list[dict]:
    """Retrieve saved observations from this session.

    Returns the most recent observations, optionally filtered by tag.
    Use this to recall earlier notes and track patterns over time.
    """
    _inc_tool_calls()
    obs = _observations
    if tag_filter:
        obs = [o for o in obs if tag_filter.lower() in [t.lower() for t in o.get("tags", [])]]
    return obs[-last_n:]


# ═══════════════════════════════════════════════════════════════════════
# CATEGORY 7: KNOWLEDGE — External Info (2 tools)
# ═══════════════════════════════════════════════════════════════════════

@mcp.tool(tags={"knowledge"}, annotations={"readOnlyHint": True})
def web_search(
    query: Annotated[str, "Search query (e.g. 'proper warrior II pose alignment')"],
) -> dict:
    """Search the web for information.

    Useful for looking up: proper exercise form, medical symptoms,
    ergonomic guidelines, safety protocols, etc.
    Uses Tavily search API if configured, otherwise returns a note.
    """
    _inc_tool_calls()
    tavily_key = os.environ.get("TAVILY_API_KEY", "")
    if tavily_key:
        try:
            import httpx
            resp = httpx.post(
                "https://api.tavily.com/search",
                json={"api_key": tavily_key, "query": query, "max_results": 3},
                timeout=10,
            )
            data = resp.json()
            results = []
            for r in data.get("results", []):
                results.append({
                    "title": r.get("title"),
                    "url": r.get("url"),
                    "content": r.get("content", "")[:500],
                })
            return {"query": query, "results": results}
        except Exception as e:
            return {"query": query, "error": str(e)}
    return {
        "query": query,
        "note": "Web search not configured (set TAVILY_API_KEY). Use your built-in knowledge instead.",
    }


@mcp.tool(tags={"knowledge"}, annotations={"readOnlyHint": True})
def get_current_time() -> dict:
    """Get the current date, time, and timezone.

    Useful for time-based reasoning: break reminders, session duration,
    noting when events occurred, etc.
    """
    _inc_tool_calls()
    now = datetime.now()
    return {
        "iso": now.isoformat(),
        "date": now.strftime("%Y-%m-%d"),
        "time": now.strftime("%H:%M:%S"),
        "day_of_week": now.strftime("%A"),
        "unix_epoch": time.time(),
    }


# ═══════════════════════════════════════════════════════════════════════
# CATEGORY 8: GOAL MANAGEMENT (3 tools)
# ═══════════════════════════════════════════════════════════════════════

@mcp.tool(tags={"goals"}, annotations={"readOnlyHint": True})
def get_current_goal() -> dict:
    """Get the currently active monitoring goal.

    Returns: goal_id, name, description, icon, and category.
    """
    _inc_tool_calls()
    if _agent_ref is None:
        return {"error": "Agent not initialized"}
    g = _agent_ref.active_goal
    return {
        "goal_id": g.goal_id,
        "name": g.name,
        "description": g.description,
        "icon": g.icon,
        "category": g.category,
    }


@mcp.tool(tags={"goals"})
def update_goal(
    description: Annotated[str, "New goal in natural language (e.g. 'Watch me for drowsiness while driving')"],
) -> dict:
    """Change the monitoring goal to a new natural language description.

    The system will try to match it to a preset goal. If no match,
    a custom goal is created. The agent's reasoning and alerts will
    adapt to the new goal immediately.
    """
    _inc_tool_calls()
    _session_stats["goal_changes"] += 1
    if _agent_ref is None:
        return {"error": "Agent not initialized"}
    _agent_ref.set_goal(description)
    g = _agent_ref.active_goal
    return {
        "status": "goal_updated",
        "goal_id": g.goal_id,
        "name": g.name,
        "description": g.description,
        "icon": g.icon,
    }


@mcp.tool(tags={"goals"}, annotations={"readOnlyHint": True})
def get_goal_presets() -> list[dict]:
    """List all available preset goal shortcuts.

    Each preset has optimized system prompts, alert triggers, and
    reasoning styles. Users can also set any custom goal.
    """
    _inc_tool_calls()
    from aegis.goals import get_all_goals
    return [
        {
            "goal_id": g.goal_id,
            "name": g.name,
            "description": g.description,
            "icon": g.icon,
            "category": g.category,
        }
        for g in get_all_goals()
    ]


# ═══════════════════════════════════════════════════════════════════════
# INTERNAL HELPERS (not exposed as tools)
# ═══════════════════════════════════════════════════════════════════════

def _inc_tool_calls():
    """Increment the session tool call counter."""
    _session_stats["tool_calls"] += 1


def _update_tracking(state: dict):
    """Update internal tracking data from state."""
    for p in state.get("persons", []):
        _session_stats["people_seen"].add(p["track_id"])
        _record_activity(p["track_id"], p.get("activity", "unknown"))


def _record_activity(track_id: int, activity: str):
    """Record activity for timeline tracking."""
    now = time.time()
    # Check if last entry for this person has the same activity
    for entry in reversed(_activity_history):
        if entry["track_id"] == track_id:
            if entry["activity"] == activity:
                entry["last_seen"] = now
                entry["duration"] = now - entry["timestamp"]
                return
            break

    _activity_history.append({
        "track_id": track_id,
        "activity": activity,
        "timestamp": now,
        "last_seen": now,
        "duration": 0,
        "time_str": datetime.now().strftime("%H:%M:%S"),
    })

    # Trim old entries
    if len(_activity_history) > 2000:
        _activity_history[:] = _activity_history[-1000:]


def _get_raw_landmarks(track_id: int) -> list | None:
    """Get raw pose landmark tuples for a tracked person.

    Accesses the SpatialEngine's internal perception state to get
    the PoseLandmarks object for the given track_id.
    """
    if _engine is None:
        return None

    # The engine stores TrackedPerson objects in its pipeline
    # We need to access the latest tracked persons before they get serialized
    # For now, we check if the person has pose data in the state
    state = _engine.get_state()
    for p in state.get("persons", []):
        if p["track_id"] == track_id and p.get("has_pose"):
            # Access the engine's internal tracked persons list
            # The spatial_engine stores the raw TrackedPerson with pose
            try:
                with _engine._lock:
                    for tp in _get_tracked_persons():
                        if tp.track_id == track_id and tp.pose:
                            return tp.pose.points
            except Exception:
                pass
    return None


def _get_tracked_persons():
    """Try to get the raw TrackedPerson list from the engine.

    Falls back to an empty list if the engine doesn't expose this.
    """
    # The engine stores the latest state dict, but we need the raw objects
    # We'll add a method to SpatialEngine to expose tracked persons
    if hasattr(_engine, '_tracked_persons'):
        return _engine._tracked_persons
    return []


def _compute_posture_metrics(landmarks: list) -> dict:
    """Compute posture metrics from raw landmark points.

    landmarks: list of (x, y, visibility) tuples (33 points).
    """
    def pt(idx):
        if idx < len(landmarks) and landmarks[idx][2] > 0.4:
            return landmarks[idx]
        return None

    def angle_3pt(a, b, c):
        ba = (a[0] - b[0], a[1] - b[1])
        bc = (c[0] - b[0], c[1] - b[1])
        dot = ba[0] * bc[0] + ba[1] * bc[1]
        mag_ba = math.sqrt(ba[0]**2 + ba[1]**2) + 1e-6
        mag_bc = math.sqrt(bc[0]**2 + bc[1]**2) + 1e-6
        cos_a = max(-1, min(1, dot / (mag_ba * mag_bc)))
        return math.degrees(math.acos(cos_a))

    metrics = {}

    l_sh, r_sh = pt(11), pt(12)
    l_hip, r_hip = pt(23), pt(24)
    nose = pt(0)

    # Shoulder alignment
    if l_sh and r_sh:
        metrics["shoulder_tilt_deg"] = round(
            math.degrees(math.atan2(r_sh[1] - l_sh[1], r_sh[0] - l_sh[0])), 1
        )
        sh_mid = ((l_sh[0] + r_sh[0]) / 2, (l_sh[1] + r_sh[1]) / 2)
        metrics["shoulder_midpoint"] = {"x": round(sh_mid[0], 1), "y": round(sh_mid[1], 1)}

        # Head forward offset
        if nose:
            metrics["head_forward_offset"] = round(abs(nose[0] - sh_mid[0]), 1)
            metrics["head_tilt_y"] = round(nose[1] - sh_mid[1], 1)

    # Spine angle
    if l_sh and r_sh and l_hip and r_hip:
        sh_mid = ((l_sh[0] + r_sh[0]) / 2, (l_sh[1] + r_sh[1]) / 2)
        hip_mid = ((l_hip[0] + r_hip[0]) / 2, (l_hip[1] + r_hip[1]) / 2)
        dx = sh_mid[0] - hip_mid[0]
        dy = sh_mid[1] - hip_mid[1]
        spine_angle = abs(math.degrees(math.atan2(dx, -dy)))  # 0 = vertical
        metrics["spine_angle_deg"] = round(spine_angle, 1)

    # Hip alignment
    if l_hip and r_hip:
        metrics["hip_tilt_deg"] = round(
            math.degrees(math.atan2(r_hip[1] - l_hip[1], r_hip[0] - l_hip[0])), 1
        )

    # Knee angles
    l_knee, r_knee = pt(25), pt(26)
    l_ankle, r_ankle = pt(27), pt(28)

    if l_hip and l_knee and l_ankle:
        metrics["left_knee_angle"] = round(angle_3pt(l_hip, l_knee, l_ankle), 1)
    if r_hip and r_knee and r_ankle:
        metrics["right_knee_angle"] = round(angle_3pt(r_hip, r_knee, r_ankle), 1)

    # Elbow angles
    l_elbow, r_elbow = pt(13), pt(14)
    l_wrist, r_wrist = pt(15), pt(16)

    if l_sh and l_elbow and l_wrist:
        metrics["left_elbow_angle"] = round(angle_3pt(l_sh, l_elbow, l_wrist), 1)
    if r_sh and r_elbow and r_wrist:
        metrics["right_elbow_angle"] = round(angle_3pt(r_sh, r_elbow, r_wrist), 1)

    return metrics


def _format_duration(seconds: float) -> str:
    """Format seconds into human-readable duration."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    if h > 0:
        return f"{h}h {m}m {s}s"
    elif m > 0:
        return f"{m}m {s}s"
    return f"{s}s"


# ═══════════════════════════════════════════════════════════════════════
# CATEGORY 9: SKILL REFERENCE — Expert Reference Management (4 tools)
# ═══════════════════════════════════════════════════════════════════════

@mcp.tool(tags={"skill", "reference"})
def record_reference_start(name: str) -> dict:
    """Start recording an expert reference movement from the live camera.

    The system will capture skeleton data every frame until you call
    record_reference_stop. Have the expert perform the movement while recording.

    Args:
        name: Name for this reference (e.g., "perfect_squat", "warrior_pose")
    """
    _inc_tool_calls()
    global _recording_session
    if _recording_session and _recording_session.active:
        return {"error": "Recording already in progress. Stop it first."}
    _recording_session = RecordingSession(name=name)
    return {"status": "recording_started", "name": name, "message": f"Recording '{name}'. Perform the movement now."}


@mcp.tool(tags={"skill", "reference"})
def record_reference_stop(key_angle: str = "left_knee") -> dict:
    """Stop recording and save the expert reference.

    Normalizes all captured skeletons, detects movement phases, and
    stores the reference for future comparison.

    Args:
        key_angle: Which joint angle defines the movement phases.
                   Options: left_knee, right_knee, left_hip, right_hip,
                   left_elbow, right_elbow, left_shoulder, right_shoulder.
                   Use left_knee for squats/lunges, left_elbow for curls.
    """
    _inc_tool_calls()
    global _recording_session
    if not _recording_session or not _recording_session.active:
        return {"error": "No recording in progress."}

    # Capture remaining frames from engine
    if _engine:
        persons = _engine._tracked_persons if hasattr(_engine, '_tracked_persons') else []
        for p in persons:
            if p.pose and p.pose.points:
                _recording_session.add_frame(p.pose.points)

    sequence = _recording_session.stop(key_angle=key_angle)
    filepath = _reference_store.save(sequence)
    _recording_session = None

    return {
        "status": "reference_saved",
        "name": sequence.name,
        "frame_count": sequence.frame_count,
        "duration": round(sequence.duration, 1),
        "phases_detected": len(sequence.phases),
        "phases": [{"name": p.name, "frames": f"{p.start_frame}-{p.end_frame}"} for p in sequence.phases],
        "file": filepath,
    }


@mcp.tool(tags={"skill", "reference"}, annotations={"readOnlyHint": True})
def list_references() -> list[dict]:
    """List all stored expert reference movements.

    Returns metadata for each reference: name, frame count, duration, phases.
    """
    _inc_tool_calls()
    return _reference_store.list_references()


@mcp.tool(tags={"skill", "reference"})
def load_reference_from_current(name: str, num_frames: int = 30,
                                key_angle: str = "left_knee") -> dict:
    """Capture frames from the live camera right now and save as a reference.

    Useful for quick reference creation: have the expert stand in front of
    the camera and perform the movement. Captures num_frames at ~15 FPS.

    Args:
        name: Name for this reference
        num_frames: Number of frames to capture (30 = ~2 seconds at 15 FPS)
        key_angle: Joint angle for phase detection
    """
    _inc_tool_calls()
    if _engine is None:
        return {"error": "Engine not initialized"}

    session = RecordingSession(name=name)
    persons = _engine._tracked_persons if hasattr(_engine, '_tracked_persons') else []

    if not persons:
        return {"error": "No person detected in camera. Have the expert stand in view."}

    # Capture from the first detected person with pose
    for p in persons:
        if p.pose and p.pose.points:
            session.add_frame(p.pose.points)

    if session.frames:
        sequence = session.stop(key_angle=key_angle)
        filepath = _reference_store.save(sequence)
        return {
            "status": "reference_saved",
            "name": name,
            "frame_count": sequence.frame_count,
            "duration": round(sequence.duration, 1),
            "message": f"Captured {sequence.frame_count} frames. For better results, record over multiple seconds.",
        }

    return {"error": "Could not capture pose data. Ensure person is fully visible."}


# ═══════════════════════════════════════════════════════════════════════
# CATEGORY 10: SKILL COMPARE — Pose Comparison & Analysis (5 tools)
# ═══════════════════════════════════════════════════════════════════════

@mcp.tool(tags={"skill", "compare"}, annotations={"readOnlyHint": True})
def compare_to_reference(reference_name: str) -> dict:
    """Compare the user's CURRENT live pose to a stored expert reference.

    Returns per-joint deviations in degrees, overall similarity score (0-100%),
    and the worst joints that need correction.

    Args:
        reference_name: Name of the stored reference to compare against
    """
    _inc_tool_calls()
    if _engine is None:
        return {"error": "Engine not initialized"}

    ref = _reference_store.load(reference_name)
    if ref is None:
        return {"error": f"Reference '{reference_name}' not found. Use list_references to see available ones."}

    # Get current user pose
    persons = _engine._tracked_persons if hasattr(_engine, '_tracked_persons') else []
    user_points = None
    for p in persons:
        if p.pose and p.pose.points:
            user_points = p.pose.points
            break

    if user_points is None:
        return {"error": "No person with pose data detected. Ensure user is visible."}

    # Find best matching frame in reference
    user_skel = normalize_skeleton(user_points)
    best_idx = 0
    best_dist = float('inf')
    for i, ref_skel in enumerate(ref.skeletons):
        from aegis.pose_comparison import compute_skeleton_distance
        d = compute_skeleton_distance(user_skel, ref_skel)
        if d < best_dist:
            best_dist = d
            best_idx = i

    result = compare_poses(user_skel.joint_angles, ref.skeletons[best_idx].joint_angles)
    return {
        **result.to_dict(),
        "reference": reference_name,
        "matched_frame": best_idx,
        "total_reference_frames": ref.frame_count,
    }


@mcp.tool(tags={"skill", "compare"}, annotations={"readOnlyHint": True})
def get_joint_deviation(joint_name: str, reference_name: str = "") -> dict:
    """Get the deviation of a specific joint from the expert reference or ideal.

    Args:
        joint_name: Joint to check. Options: left_knee, right_knee, left_hip,
                    right_hip, left_shoulder, right_shoulder, left_elbow,
                    right_elbow, left_ankle, right_ankle
        reference_name: Reference to compare against. If empty, returns current angle only.
    """
    _inc_tool_calls()
    if _engine is None:
        return {"error": "Engine not initialized"}

    if joint_name not in KEY_ANGLES:
        return {"error": f"Unknown joint '{joint_name}'. Options: {list(KEY_ANGLES.keys())}"}

    persons = _engine._tracked_persons if hasattr(_engine, '_tracked_persons') else []
    user_points = None
    for p in persons:
        if p.pose and p.pose.points:
            user_points = p.pose.points
            break

    if user_points is None:
        return {"error": "No person with pose detected."}

    user_angles = compute_joint_angles(user_points)
    if joint_name not in user_angles:
        return {"error": f"Joint '{joint_name}' not visible in current pose."}

    result = {
        "joint": joint_name,
        "friendly_name": ANGLE_NAMES.get(joint_name, joint_name),
        "current_angle": round(user_angles[joint_name], 1),
    }

    if reference_name:
        ref = _reference_store.load(reference_name)
        if ref and ref.skeletons:
            # Average angle across reference
            ref_angles = [s.joint_angles.get(joint_name) for s in ref.skeletons if joint_name in s.joint_angles]
            if ref_angles:
                avg_ref = sum(ref_angles) / len(ref_angles)
                result["expert_angle"] = round(avg_ref, 1)
                result["deviation"] = round(abs(user_angles[joint_name] - avg_ref), 1)
                result["direction"] = "too_open" if user_angles[joint_name] > avg_ref else "too_closed"

    return result


@mcp.tool(tags={"skill", "compare"}, annotations={"readOnlyHint": True})
def get_movement_quality_analysis() -> dict:
    """Analyze the quality of the user's recent movement.

    Returns smoothness (jerk analysis), left-right symmetry,
    and tempo consistency scores (each 0-100).

    Requires an active coaching session with at least a few frames.
    """
    _inc_tool_calls()
    if _coaching_session is None:
        return {"error": "No active coaching session. Start one with start_coaching_session."}

    if len(_coaching_session.current_rep_frames) < 3:
        return {"error": "Need at least 3 frames. Keep moving and try again."}

    quality = compute_movement_quality(_coaching_session.current_rep_frames)

    return {
        "smoothness": quality["smoothness"],
        "smoothness_grade": "Excellent" if quality["smoothness"] > 80 else "Good" if quality["smoothness"] > 60 else "Needs work",
        "symmetry": quality["symmetry"],
        "symmetry_grade": "Excellent" if quality["symmetry"] > 80 else "Good" if quality["symmetry"] > 60 else "Asymmetric",
        "tempo_consistency": quality["tempo_consistency"],
        "tempo_grade": "Excellent" if quality["tempo_consistency"] > 80 else "Good" if quality["tempo_consistency"] > 60 else "Inconsistent",
        "frames_analyzed": len(_coaching_session.current_rep_frames),
    }


@mcp.tool(tags={"skill", "compare"}, annotations={"readOnlyHint": True})
def detect_compensation_patterns(reference_name: str = "") -> list[dict]:
    """Check if the user is compensating (favoring one side over the other).

    Detects asymmetric deviations that suggest the user is avoiding
    weakness on one side. Important for injury prevention.

    Args:
        reference_name: Reference to compare against. If empty, checks raw symmetry.
    """
    _inc_tool_calls()
    if _engine is None:
        return [{"error": "Engine not initialized"}]

    persons = _engine._tracked_persons if hasattr(_engine, '_tracked_persons') else []
    user_points = None
    for p in persons:
        if p.pose and p.pose.points:
            user_points = p.pose.points
            break

    if user_points is None:
        return [{"error": "No person detected."}]

    user_angles = compute_joint_angles(user_points)

    if reference_name:
        ref = _reference_store.load(reference_name)
        if ref and ref.skeletons:
            mid = len(ref.skeletons) // 2
            expert_angles = ref.skeletons[mid].joint_angles
            return detect_compensation(user_angles, expert_angles)

    # No reference — check symmetry against mirror
    mirror_angles = {}
    for joint, angle in user_angles.items():
        if "left_" in joint:
            mirror = joint.replace("left_", "right_")
            if mirror in user_angles:
                mirror_angles[joint] = user_angles[mirror]
                mirror_angles[mirror] = angle
    if mirror_angles:
        return detect_compensation(user_angles, mirror_angles)

    return []


@mcp.tool(tags={"skill", "compare"}, annotations={"readOnlyHint": True})
def compare_full_movement(reference_name: str) -> dict:
    """Compare an entire movement sequence to the expert using DTW alignment.

    Uses Dynamic Time Warping to align the user's movement timing to the
    expert's, then scores each aligned frame. Handles different speeds.

    Requires an active coaching session with captured frames.

    Args:
        reference_name: Reference movement to compare against
    """
    _inc_tool_calls()
    if _coaching_session is None:
        return {"error": "No active coaching session."}

    ref = _reference_store.load(reference_name)
    if ref is None:
        return {"error": f"Reference '{reference_name}' not found."}

    user_frames = _coaching_session.current_rep_frames
    if len(user_frames) < 3:
        return {"error": "Need more frames. Keep performing the movement."}

    result = compare_sequences(user_frames, ref.skeletons)
    return result


# ═══════════════════════════════════════════════════════════════════════
# CATEGORY 11: SKILL COACHING — Session Management (4 tools)
# ═══════════════════════════════════════════════════════════════════════

@mcp.tool(tags={"skill", "coaching"})
def start_coaching_session(skill_name: str, reference_name: str = "",
                           primary_angle: str = "left_knee") -> dict:
    """Start a coaching session for a specific skill.

    Initializes rep counting, score tracking, and training data collection.
    Optionally loads an expert reference for comparison.

    Args:
        skill_name: Name of the skill being coached (e.g., "squat", "warrior_pose")
        reference_name: Expert reference to compare against (optional)
        primary_angle: Joint angle used for rep detection. Use left_knee for
                       squats/lunges, left_elbow for curls, left_hip for deadlifts.
    """
    _inc_tool_calls()
    global _coaching_session

    ref = None
    if reference_name:
        ref = _reference_store.load(reference_name)
        if ref is None:
            return {"error": f"Reference '{reference_name}' not found."}

    _coaching_session = CoachingSession(skill_name=skill_name, reference=ref)
    _coaching_session.set_primary_angle(primary_angle)

    return {
        "status": "session_started",
        "skill": skill_name,
        "reference_loaded": reference_name if ref else "none (zero-shot mode)",
        "primary_angle": primary_angle,
        "message": f"Coaching session for '{skill_name}' started. Begin performing the movement.",
    }


@mcp.tool(tags={"skill", "coaching"}, annotations={"readOnlyHint": True})
def get_coaching_progress() -> dict:
    """Get progress in the current coaching session.

    Returns: reps completed, average score, trend (improving/stable/declining),
    scores per rep, top corrections needed, and training data count.
    """
    _inc_tool_calls()
    if _coaching_session is None:
        return {"error": "No active coaching session."}
    return _coaching_session.get_progress()


@mcp.tool(tags={"skill", "coaching"}, annotations={"readOnlyHint": True})
def get_rep_count() -> dict:
    """Get the current rep count in the coaching session.

    Auto-detects reps from the primary joint angle's oscillation pattern.
    """
    _inc_tool_calls()
    if _coaching_session is None:
        return {"error": "No active coaching session."}

    return {
        "reps": _coaching_session.get_rep_count(),
        "skill": _coaching_session.skill_name,
        "frames_processed": _coaching_session.frame_count,
    }


@mcp.tool(tags={"skill", "coaching"})
def end_coaching_session() -> dict:
    """End the current coaching session and get the final summary.

    Returns: total reps, average score, improvement trend, duration,
    and count of training samples collected for model training.
    """
    _inc_tool_calls()
    global _coaching_session
    if _coaching_session is None:
        return {"error": "No active coaching session."}

    summary = _coaching_session.end()
    _coaching_session = None
    return summary


# ═══════════════════════════════════════════════════════════════════════
# CATEGORY 12: SKILL INTELLIGENCE — Zero-Shot & Document Parsing (2 tools)
# ═══════════════════════════════════════════════════════════════════════

@mcp.tool(tags={"skill", "training"})
def bootstrap_model(skill_name: str = "squat", n_synthetic: int = 50) -> dict:
    """Bootstrap the local scoring model with synthetic training data.

    Use this when no real coaching data exists yet. Generates
    biomechanically plausible synthetic data for the given skill
    and trains the model immediately. The model will improve
    as real coaching sessions collect data.

    Args:
        skill_name: Skill to generate data for (squat, lunge, bicep_curl, warrior_pose, etc.)
        n_synthetic: Number of synthetic samples to generate (default 50)
    """
    _inc_tool_calls()
    from aegis.hybrid_scorer import HybridScorer
    # Use the global hybrid scorer if available
    try:
        from aegis.server import _hybrid_scorer
        result = _hybrid_scorer.generate_and_train(
            skill=skill_name, n_synthetic=n_synthetic, epochs=80,
        )
    except Exception:
        hs = HybridScorer()
        result = hs.generate_and_train(
            skill=skill_name, n_synthetic=n_synthetic, epochs=80,
        )
    return result


@mcp.tool(tags={"skill", "training"})
def generate_training_data(skill_name: str = "squat", n_samples: int = 30) -> dict:
    """Generate synthetic training data for a skill without training.

    Creates biomechanically plausible angle sequences with varying
    quality levels (good, medium, poor form). Save them for later
    training or combine with real coaching data.

    Args:
        skill_name: Skill to generate data for
        n_samples: Number of samples to generate
    """
    _inc_tool_calls()
    from aegis.data_collector import DataCollector
    dc = DataCollector()
    return dc.generate_synthetic_data(skill=skill_name, n_samples=n_samples)


@mcp.tool(tags={"skill", "coaching"})
def collect_coaching_rep(similarity_score: float,
                         corrections: Annotated[str, "Comma-separated correction strings"] = "",
                         ) -> dict:
    """Manually collect the current rep data into the training pipeline.

    Call this after scoring a rep during coaching. It saves the skeleton
    data + score as a training sample for the local model.

    Args:
        similarity_score: The score (0-100) for this rep
        corrections: Comma-separated correction strings
    """
    _inc_tool_calls()
    if _coaching_session is None:
        return {"error": "No active coaching session."}
    if not _coaching_session.current_rep_frames:
        return {"error": "No frames in current rep."}

    correction_list = [c.strip() for c in corrections.split(",") if c.strip()] if corrections else []
    rep = _coaching_session.complete_rep(
        similarity_score=similarity_score,
        deviations={},
        corrections=correction_list,
    )

    # Auto-save to training pipeline
    from aegis.data_collector import DataCollector
    dc = DataCollector()
    dc.save_rep_from_session(
        skill=_coaching_session.skill_name,
        skeleton_frames=_coaching_session.reps[-1].skeleton_sequence if _coaching_session.reps else [],
        score=similarity_score,
        corrections=correction_list,
        deviations={},
    )

    return {
        "rep_number": rep.rep_number,
        "score": round(rep.similarity_score, 1),
        "saved_to_training": True,
        "total_reps": len(_coaching_session.reps),
    }


@mcp.tool(tags={"skill", "intelligence"}, annotations={"readOnlyHint": True})
def analyze_skill_from_description(skill_description: str) -> dict:
    """Analyze what joint angles and body positions matter for a described skill.

    Given a natural language skill description, returns which joints to
    monitor, what angles indicate good form, and common mistakes.
    This enables zero-shot coaching without an expert reference.

    Args:
        skill_description: Natural language description of the skill
                          (e.g., "tennis serve", "deadlift", "tree pose")
    """
    _inc_tool_calls()
    # Get current pose for context
    current_angles = {}
    if _engine:
        persons = _engine._tracked_persons if hasattr(_engine, '_tracked_persons') else []
        for p in persons:
            if p.pose and p.pose.points:
                current_angles = compute_joint_angles(p.pose.points)
                break

    return {
        "skill": skill_description,
        "available_angles": list(KEY_ANGLES.keys()),
        "angle_descriptions": ANGLE_NAMES,
        "current_user_angles": {k: round(v, 1) for k, v in current_angles.items()},
        "instruction": (
            f"You are coaching '{skill_description}'. Use your biomechanics knowledge to "
            f"determine which of the available angles matter most for this skill and what "
            f"their ideal values should be. Compare the user's current angles to your "
            f"ideal and provide specific corrections. Use speak_to_user for voice coaching."
        ),
    }


@mcp.tool(tags={"skill", "intelligence"})
def parse_skill_document(document_text: str, skill_name: str = "parsed_skill") -> dict:
    """Parse a physical skill document (PT protocol, yoga guide, etc.) into
    a structured skill definition that can be used for coaching.

    Feed this tool the text content of a PT protocol, exercise guide,
    yoga instruction, or any document describing physical movements.
    It extracts the structured information needed for coaching.

    Args:
        document_text: The text content of the document to parse
        skill_name: Name to assign to this skill
    """
    _inc_tool_calls()
    return {
        "skill_name": skill_name,
        "document_length": len(document_text),
        "document_preview": document_text[:500],
        "available_angles": list(KEY_ANGLES.keys()),
        "instruction": (
            f"You received a document describing '{skill_name}'. Extract from this document:\n"
            f"1. Target joint angles for each exercise/pose\n"
            f"2. Movement phases (start, execution, hold, recovery)\n"
            f"3. Rep counts and set counts if specified\n"
            f"4. Safety boundaries (max/min angles, contraindications)\n"
            f"5. Common mistakes to watch for\n\n"
            f"Then use start_coaching_session to begin coaching based on these targets. "
            f"Compare the user's live angles against the targets you extracted. "
            f"Use speak_to_user for real-time voice coaching."
        ),
    }


# ── Entry point (for standalone MCP server testing) ─────────────────────
if __name__ == "__main__":
    mcp.run()
