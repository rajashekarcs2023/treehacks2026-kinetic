"""
AEGIS SDK Tools — Wraps existing MCP tool logic as claude-agent-sdk custom tools.

Uses @tool decorator from claude_agent_sdk to create SdkMcpTool objects,
then bundles them into an in-process MCP server via create_sdk_mcp_server().

Tool naming: mcp__aegis__<tool_name>
"""

import json
from typing import Any

from claude_agent_sdk import tool, create_sdk_mcp_server

# Import our existing tool implementations (they access shared state via globals)
from aegis import mcp_server


# ═══════════════════════════════════════════════════════════════════════
# Helper: wrap a sync function into the SDK tool return format
# ═══════════════════════════════════════════════════════════════════════

def _text_result(data) -> dict[str, Any]:
    """Format any result as SDK tool response."""
    if isinstance(data, str):
        text = data
    else:
        text = json.dumps(data, default=str)
    return {"content": [{"type": "text", "text": text}]}


# ═══════════════════════════════════════════════════════════════════════
# CATEGORY 1: PERCEPTION (6 tools)
# ═══════════════════════════════════════════════════════════════════════

@tool("get_spatial_state",
      "Get the full current spatial state: persons, objects, poses, activities.",
      {})
async def get_spatial_state(args: dict[str, Any]) -> dict[str, Any]:
    return _text_result(mcp_server.get_spatial_state())


@tool("get_spatial_summary",
      "Get a concise human-readable summary of the current scene.",
      {})
async def get_spatial_summary(args: dict[str, Any]) -> dict[str, Any]:
    return _text_result(mcp_server.get_spatial_summary())


@tool("get_person_detail",
      "Get deep detail about a specific tracked person.",
      {"track_id": int})
async def get_person_detail(args: dict[str, Any]) -> dict[str, Any]:
    return _text_result(mcp_server.get_person_detail(args["track_id"]))


@tool("get_scene_changes",
      "Detect what changed in the scene recently.",
      {"seconds_back": int})
async def get_scene_changes(args: dict[str, Any]) -> dict[str, Any]:
    return _text_result(mcp_server.get_scene_changes(args.get("seconds_back", 10)))


@tool("get_objects_in_scene",
      "List all detected objects in the scene. Optional class_filter.",
      {"class_filter": str})
async def get_objects_in_scene(args: dict[str, Any]) -> dict[str, Any]:
    return _text_result(mcp_server.get_objects_in_scene(args.get("class_filter")))


@tool("count_objects",
      "Count how many of a specific object class are visible.",
      {"class_name": str})
async def count_objects(args: dict[str, Any]) -> dict[str, Any]:
    return _text_result(mcp_server.count_objects(args["class_name"]))


# ═══════════════════════════════════════════════════════════════════════
# CATEGORY 2: POSE ANALYSIS (3 tools)
# ═══════════════════════════════════════════════════════════════════════

@tool("analyze_posture",
      "Analyze a person's posture from skeletal landmarks.",
      {"track_id": int})
async def analyze_posture(args: dict[str, Any]) -> dict[str, Any]:
    return _text_result(mcp_server.analyze_posture(args["track_id"]))


@tool("get_pose_landmarks",
      "Get the raw 33 MediaPipe pose landmarks for a person.",
      {"track_id": int})
async def get_pose_landmarks(args: dict[str, Any]) -> dict[str, Any]:
    return _text_result(mcp_server.get_pose_landmarks(args["track_id"]))


@tool("check_body_alignment",
      "Check body alignment for an optional exercise context.",
      {"track_id": int, "exercise": str})
async def check_body_alignment(args: dict[str, Any]) -> dict[str, Any]:
    return _text_result(mcp_server.check_body_alignment(args["track_id"], args.get("exercise")))


# ═══════════════════════════════════════════════════════════════════════
# CATEGORY 3: ACTIVITY (3 tools)
# ═══════════════════════════════════════════════════════════════════════

@tool("get_activity_timeline",
      "Get activity timeline for a person over recent minutes.",
      {"track_id": int, "last_minutes": int})
async def get_activity_timeline(args: dict[str, Any]) -> dict[str, Any]:
    return _text_result(mcp_server.get_activity_timeline(args["track_id"], args.get("last_minutes", 5)))


@tool("get_time_in_activity",
      "How long a person has been in a specific activity.",
      {"track_id": int, "activity": str})
async def get_time_in_activity(args: dict[str, Any]) -> dict[str, Any]:
    return _text_result(mcp_server.get_time_in_activity(args["track_id"], args["activity"]))


@tool("get_session_stats",
      "Get aggregate statistics for the current monitoring session.",
      {})
async def get_session_stats(args: dict[str, Any]) -> dict[str, Any]:
    return _text_result(mcp_server.get_session_stats())


# ═══════════════════════════════════════════════════════════════════════
# CATEGORY 4: ZONES (3 tools)
# ═══════════════════════════════════════════════════════════════════════

@tool("set_watch_zone",
      "Define a rectangular danger/interest zone to monitor.",
      {"x1": int, "y1": int, "x2": int, "y2": int, "label": str})
async def set_watch_zone(args: dict[str, Any]) -> dict[str, Any]:
    return _text_result(mcp_server.set_watch_zone(
        args["x1"], args["y1"], args["x2"], args["y2"], args.get("label", "zone")
    ))


@tool("clear_watch_zones",
      "Remove all defined watch/danger zones.",
      {})
async def clear_watch_zones(args: dict[str, Any]) -> dict[str, Any]:
    return _text_result(mcp_server.clear_watch_zones())


@tool("check_zone_status",
      "Check who/what is in or near the watched zones.",
      {"zone_id": str})
async def check_zone_status(args: dict[str, Any]) -> dict[str, Any]:
    return _text_result(mcp_server.check_zone_status(args.get("zone_id")))


# ═══════════════════════════════════════════════════════════════════════
# CATEGORY 5: ALERTS (3 tools)
# ═══════════════════════════════════════════════════════════════════════

@tool("send_telegram_alert",
      "Send an alert message to the user via Telegram.",
      {"message": str, "include_photo": bool})
async def send_telegram_alert(args: dict[str, Any]) -> dict[str, Any]:
    return _text_result(mcp_server.send_telegram_alert(
        args["message"], args.get("include_photo", False)
    ))


@tool("speak_to_user",
      "Speak to the user via text-to-speech. Use for real-time coaching cues.",
      {"message": str, "urgency": str})
async def speak_to_user(args: dict[str, Any]) -> dict[str, Any]:
    return _text_result(mcp_server.speak_to_user(
        args["message"], args.get("urgency", "normal")
    ))


@tool("capture_photo",
      "Capture a snapshot from the camera.",
      {"annotated": bool})
async def capture_photo(args: dict[str, Any]) -> dict[str, Any]:
    return _text_result(mcp_server.capture_photo(args.get("annotated", True)))


# ═══════════════════════════════════════════════════════════════════════
# CATEGORY 6: MEMORY (2 tools)
# ═══════════════════════════════════════════════════════════════════════

@tool("save_observation",
      "Save a persistent observation for future reference.",
      {"text": str, "tags": str})
async def save_observation(args: dict[str, Any]) -> dict[str, Any]:
    tags = args.get("tags", "")
    tag_list = [t.strip() for t in tags.split(",")] if tags else []
    return _text_result(mcp_server.save_observation(args["text"], tag_list))


@tool("get_observations",
      "Retrieve saved observations, optionally filtered by tag.",
      {"tag_filter": str, "last_n": int})
async def get_observations(args: dict[str, Any]) -> dict[str, Any]:
    return _text_result(mcp_server.get_observations(
        args.get("tag_filter"), args.get("last_n", 20)
    ))


# ═══════════════════════════════════════════════════════════════════════
# CATEGORY 7: KNOWLEDGE (2 tools)
# ═══════════════════════════════════════════════════════════════════════

@tool("web_search",
      "Search the web for information about exercises, form, protocols.",
      {"query": str})
async def web_search(args: dict[str, Any]) -> dict[str, Any]:
    return _text_result(mcp_server.web_search(args["query"]))


@tool("get_current_time",
      "Get the current date, time, and timezone.",
      {})
async def get_current_time(args: dict[str, Any]) -> dict[str, Any]:
    return _text_result(mcp_server.get_current_time())


# ═══════════════════════════════════════════════════════════════════════
# CATEGORY 8: GOALS (3 tools)
# ═══════════════════════════════════════════════════════════════════════

@tool("get_current_goal",
      "Get the currently active monitoring/coaching goal.",
      {})
async def get_current_goal(args: dict[str, Any]) -> dict[str, Any]:
    return _text_result(mcp_server.get_current_goal())


@tool("update_goal",
      "Change the goal to a new natural language description.",
      {"description": str})
async def update_goal(args: dict[str, Any]) -> dict[str, Any]:
    return _text_result(mcp_server.update_goal(args["description"]))


@tool("get_goal_presets",
      "List all available preset goal shortcuts.",
      {})
async def get_goal_presets(args: dict[str, Any]) -> dict[str, Any]:
    return _text_result(mcp_server.get_goal_presets())


# ═══════════════════════════════════════════════════════════════════════
# CATEGORY 9: SKILL REFERENCE (4 tools)
# ═══════════════════════════════════════════════════════════════════════

@tool("record_reference_start",
      "Start recording an expert reference movement from live camera.",
      {"name": str})
async def record_reference_start(args: dict[str, Any]) -> dict[str, Any]:
    return _text_result(mcp_server.record_reference_start(args["name"]))


@tool("record_reference_stop",
      "Stop recording and save the expert reference.",
      {"key_angle": str})
async def record_reference_stop(args: dict[str, Any]) -> dict[str, Any]:
    return _text_result(mcp_server.record_reference_stop(args.get("key_angle", "left_knee")))


@tool("list_references",
      "List all stored expert reference movements.",
      {})
async def list_references(args: dict[str, Any]) -> dict[str, Any]:
    return _text_result(mcp_server.list_references())


@tool("load_reference_from_current",
      "Capture frames from live camera and save as a reference.",
      {"name": str, "num_frames": int, "key_angle": str})
async def load_reference_from_current(args: dict[str, Any]) -> dict[str, Any]:
    return _text_result(mcp_server.load_reference_from_current(
        args["name"], args.get("num_frames", 30), args.get("key_angle", "left_knee")
    ))


# ═══════════════════════════════════════════════════════════════════════
# CATEGORY 10: SKILL COMPARE (5 tools)
# ═══════════════════════════════════════════════════════════════════════

@tool("compare_to_reference",
      "Compare user's live pose to a stored expert reference. Returns per-joint deviations and similarity score.",
      {"reference_name": str})
async def compare_to_reference(args: dict[str, Any]) -> dict[str, Any]:
    return _text_result(mcp_server.compare_to_reference(args["reference_name"]))


@tool("get_joint_deviation",
      "Get deviation of a specific joint from expert or ideal angle.",
      {"joint_name": str, "reference_name": str})
async def get_joint_deviation(args: dict[str, Any]) -> dict[str, Any]:
    return _text_result(mcp_server.get_joint_deviation(
        args["joint_name"], args.get("reference_name", "")
    ))


@tool("get_movement_quality_analysis",
      "Analyze smoothness, symmetry, and tempo of recent movement.",
      {})
async def get_movement_quality_analysis(args: dict[str, Any]) -> dict[str, Any]:
    return _text_result(mcp_server.get_movement_quality_analysis())


@tool("detect_compensation_patterns",
      "Check if user is compensating (favoring one side). Important for injury prevention.",
      {"reference_name": str})
async def detect_compensation_patterns(args: dict[str, Any]) -> dict[str, Any]:
    return _text_result(mcp_server.detect_compensation_patterns(args.get("reference_name", "")))


@tool("compare_full_movement",
      "Compare entire movement sequence to expert using DTW alignment.",
      {"reference_name": str})
async def compare_full_movement(args: dict[str, Any]) -> dict[str, Any]:
    return _text_result(mcp_server.compare_full_movement(args["reference_name"]))


# ═══════════════════════════════════════════════════════════════════════
# CATEGORY 11: SKILL COACHING (4 tools)
# ═══════════════════════════════════════════════════════════════════════

@tool("start_coaching_session",
      "Start a coaching session for a specific skill. Initializes rep counting and score tracking.",
      {"skill_name": str, "reference_name": str, "primary_angle": str})
async def start_coaching_session(args: dict[str, Any]) -> dict[str, Any]:
    return _text_result(mcp_server.start_coaching_session(
        args["skill_name"],
        args.get("reference_name", ""),
        args.get("primary_angle", "left_knee"),
    ))


@tool("get_coaching_progress",
      "Get coaching session progress: reps, scores, trend, corrections.",
      {})
async def get_coaching_progress(args: dict[str, Any]) -> dict[str, Any]:
    return _text_result(mcp_server.get_coaching_progress())


@tool("get_rep_count",
      "Get current rep count from the active coaching session.",
      {})
async def get_rep_count(args: dict[str, Any]) -> dict[str, Any]:
    return _text_result(mcp_server.get_rep_count())


@tool("end_coaching_session",
      "End coaching session and get final summary.",
      {})
async def end_coaching_session(args: dict[str, Any]) -> dict[str, Any]:
    return _text_result(mcp_server.end_coaching_session())


# ═══════════════════════════════════════════════════════════════════════
# CATEGORY 12: SKILL INTELLIGENCE (2 tools)
# ═══════════════════════════════════════════════════════════════════════

@tool("analyze_skill_from_description",
      "Analyze what joints and angles matter for a described skill. Enables zero-shot coaching.",
      {"skill_description": str})
async def analyze_skill_from_description(args: dict[str, Any]) -> dict[str, Any]:
    return _text_result(mcp_server.analyze_skill_from_description(args["skill_description"]))


@tool("parse_skill_document",
      "Parse a PT protocol, yoga guide, or exercise doc into structured coaching data.",
      {"document_text": str, "skill_name": str})
async def parse_skill_document(args: dict[str, Any]) -> dict[str, Any]:
    return _text_result(mcp_server.parse_skill_document(
        args["document_text"], args.get("skill_name", "parsed_skill")
    ))


# ── Category 13: Training Pipeline (3 tools) ────────────────────────

@tool("bootstrap_model",
      "Bootstrap the local scoring model with synthetic training data. Use when no real data exists.",
      {"skill_name": str, "n_synthetic": int})
async def bootstrap_model_tool(args: dict[str, Any]) -> dict[str, Any]:
    return _text_result(mcp_server.bootstrap_model(
        args.get("skill_name", "squat"), args.get("n_synthetic", 50)
    ))


@tool("generate_training_data",
      "Generate synthetic training data for a skill without training the model.",
      {"skill_name": str, "n_samples": int})
async def generate_training_data_tool(args: dict[str, Any]) -> dict[str, Any]:
    return _text_result(mcp_server.generate_training_data(
        args.get("skill_name", "squat"), args.get("n_samples", 30)
    ))


@tool("collect_coaching_rep",
      "Collect the current rep data into the training pipeline during coaching.",
      {"similarity_score": float, "corrections": str})
async def collect_coaching_rep_tool(args: dict[str, Any]) -> dict[str, Any]:
    return _text_result(mcp_server.collect_coaching_rep(
        args["similarity_score"], args.get("corrections", "")
    ))


# ═══════════════════════════════════════════════════════════════════════
# Bundle all tools into an SDK MCP server
# ═══════════════════════════════════════════════════════════════════════

ALL_TOOLS = [
    # Perception
    get_spatial_state, get_spatial_summary, get_person_detail,
    get_scene_changes, get_objects_in_scene, count_objects,
    # Pose
    analyze_posture, get_pose_landmarks, check_body_alignment,
    # Activity
    get_activity_timeline, get_time_in_activity, get_session_stats,
    # Zones
    set_watch_zone, clear_watch_zones, check_zone_status,
    # Alerts
    send_telegram_alert, speak_to_user, capture_photo,
    # Memory
    save_observation, get_observations,
    # Knowledge
    web_search, get_current_time,
    # Goals
    get_current_goal, update_goal, get_goal_presets,
    # Skill Reference
    record_reference_start, record_reference_stop,
    list_references, load_reference_from_current,
    # Skill Compare
    compare_to_reference, get_joint_deviation,
    get_movement_quality_analysis, detect_compensation_patterns,
    compare_full_movement,
    # Skill Coaching
    start_coaching_session, get_coaching_progress,
    get_rep_count, end_coaching_session,
    # Skill Intelligence
    analyze_skill_from_description, parse_skill_document,
    # Training Pipeline
    bootstrap_model_tool, generate_training_data_tool, collect_coaching_rep_tool,
]


def create_aegis_mcp_server():
    """Create the AEGIS SDK MCP server with all 43 tools.

    Returns an McpSdkServerConfig ready for ClaudeAgentOptions.mcp_servers.
    """
    return create_sdk_mcp_server(
        name="aegis",
        version="1.0.0",
        tools=ALL_TOOLS,
    )


# ═══════════════════════════════════════════════════════════════════════
# Tool name lists for sub-agent tool restrictions
# ═══════════════════════════════════════════════════════════════════════

PERCEPTION_TOOL_NAMES = [
    "mcp__aegis__get_spatial_state", "mcp__aegis__get_spatial_summary",
    "mcp__aegis__get_person_detail", "mcp__aegis__get_scene_changes",
    "mcp__aegis__get_objects_in_scene", "mcp__aegis__count_objects",
    "mcp__aegis__analyze_posture", "mcp__aegis__get_pose_landmarks",
    "mcp__aegis__check_body_alignment", "mcp__aegis__get_activity_timeline",
]

COACH_TOOL_NAMES = [
    "mcp__aegis__compare_to_reference", "mcp__aegis__get_joint_deviation",
    "mcp__aegis__get_movement_quality_analysis",
    "mcp__aegis__detect_compensation_patterns",
    "mcp__aegis__compare_full_movement",
    "mcp__aegis__start_coaching_session", "mcp__aegis__get_coaching_progress",
    "mcp__aegis__get_rep_count", "mcp__aegis__end_coaching_session",
    "mcp__aegis__analyze_skill_from_description",
    "mcp__aegis__speak_to_user",
    "mcp__aegis__record_reference_start", "mcp__aegis__record_reference_stop",
    "mcp__aegis__collect_coaching_rep",
]

PROGRESS_TOOL_NAMES = [
    "mcp__aegis__list_references", "mcp__aegis__load_reference_from_current",
    "mcp__aegis__parse_skill_document",
    "mcp__aegis__get_current_goal", "mcp__aegis__update_goal",
    "mcp__aegis__get_goal_presets",
    "mcp__aegis__save_observation", "mcp__aegis__get_observations",
    "mcp__aegis__bootstrap_model", "mcp__aegis__generate_training_data",
]
