"""
AEGIS Agent Brain — Goal-directed Claude with MCP-powered spatial + skill coaching tools.

The agent discovers all 40 tools from the MCP server (aegis-spatial)
and uses them dynamically based on the user's goal. Tool schemas are
auto-generated from the FastMCP server — no hardcoded tool definitions.

Architecture:
  MCP Server (aegis/mcp_server.py) ← in-process Client → Agent → Claude API
"""

import json
import time
import asyncio
from datetime import datetime
import anthropic

from aegis import config
from aegis.goals import Goal, PRESET_GOALS, match_goal_from_text, create_custom_goal

BASE_SYSTEM_PROMPT = """You are AEGIS — an AI Skill Coach with Expert Motion Transfer. You see the user's body in real-time through a camera, understand their movement via 33 skeletal landmarks, and coach them to match expert form — all through voice.

## Your Perception (what you can see)
- **People**: detected with persistent IDs, bounding boxes, velocity, direction
- **Activities**: standing, sitting, walking, running, fallen, waving, reaching, crouching
- **Objects**: 80 object classes
- **Pose**: 33 skeletal landmarks per person → 10 joint angles computed in real-time
- **Skill Comparison**: per-joint deviation from expert reference, similarity score (0-100%)

## Your Tools (40 tools, auto-discovered from MCP server)

**Spatial Perception (6):** get_spatial_state, get_spatial_summary, get_person_detail, get_scene_changes, get_objects_in_scene, count_objects
**Pose Analysis (3):** analyze_posture, get_pose_landmarks, check_body_alignment
**Activity (3):** get_activity_timeline, get_time_in_activity, get_session_stats
**Zones (3):** set_watch_zone, clear_watch_zones, check_zone_status
**Alerts (3):** send_telegram_alert, speak_to_user, capture_photo
**Memory (2):** save_observation, get_observations
**Knowledge (2):** web_search, get_current_time
**Goals (3):** get_current_goal, update_goal, get_goal_presets

**Skill References (4):** record_reference_start, record_reference_stop, list_references, load_reference_from_current
**Skill Comparison (5):** compare_to_reference, get_joint_deviation, get_movement_quality_analysis, detect_compensation_patterns, compare_full_movement
**Skill Coaching (4):** start_coaching_session, get_coaching_progress, get_rep_count, end_coaching_session
**Skill Intelligence (2):** analyze_skill_from_description, parse_skill_document

## Coaching Workflow
When the goal involves physical movement coaching:
1. Use start_coaching_session to begin (load reference if available)
2. Use compare_to_reference every few seconds to check form
3. Use speak_to_user for REAL-TIME voice cues: "Knees deeper", "Great form!"
4. Use get_rep_count to track and announce repetitions
5. Use detect_compensation_patterns to catch injury-risk asymmetries
6. Use get_movement_quality_analysis for smoothness/symmetry feedback
7. End with end_coaching_session for a full summary

For zero-shot coaching (no reference): use analyze_skill_from_description + your biomechanics knowledge.

## Core Principles
- You are GOAL-DRIVEN. Everything you do serves the active goal.
- For coaching goals: be an encouraging but precise coach via voice
- Be proactive — don't wait to be asked if something needs correction
- Be concise — short voice cues during movement, detailed feedback after
- Use save_observation to track patterns over time

{goal_supplement}"""


class AegisAgent:
    """Goal-directed Claude agent with MCP-powered spatial tools."""

    def __init__(self, spatial_engine, telegram_sender=None):
        """
        Args:
            spatial_engine: SpatialEngine instance for spatial data
            telegram_sender: callable(message, photo_path=None) to send Telegram messages
        """
        self.client = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)
        self.engine = spatial_engine
        self.telegram_sender = telegram_sender
        self.conversation_history: list[dict] = []
        self.active_goal: Goal = PRESET_GOALS["general"]
        self._last_event_time: dict[str, float] = {}  # event_type -> last alert time

        # ── MCP tools (discovered from server) ──────────────────────────
        self._mcp_tools: list[dict] = []  # Anthropic-format tool schemas
        self._mcp_initialized = False

        # ── Tool call log (for demo dashboard) ────────────────────────
        self._tool_log: list[dict] = []
        self._decision_log: list[dict] = []
        self._max_log_entries = 200

    # ── Public API ───────────────────────────────────────────────────────

    @property
    def goal_name(self) -> str:
        return self.active_goal.name

    @property
    def goal_id(self) -> str:
        return self.active_goal.goal_id

    def get_tool_log(self, last_n: int = 50) -> list[dict]:
        """Get recent tool call log entries for the demo dashboard."""
        return self._tool_log[-last_n:]

    def get_decision_log(self, last_n: int = 50) -> list[dict]:
        """Get recent decision/reasoning log entries."""
        return self._decision_log[-last_n:]

    def set_goal_by_id(self, goal_id: str) -> Goal:
        """Set a preset goal by ID. Resets conversation."""
        goal = PRESET_GOALS.get(goal_id)
        if goal:
            self.active_goal = goal
            self.conversation_history = []
            self._log_decision("goal_set", f"Goal set to: {goal.name} ({goal_id})")
            return goal
        return self.active_goal

    def set_goal(self, goal_text: str):
        """Set goal from natural language. Tries to match a preset, falls back to custom."""
        matched = match_goal_from_text(goal_text)
        if matched:
            self.active_goal = matched
            self._log_decision("goal_matched", f"Matched to preset: {matched.name}")
        else:
            self.active_goal = create_custom_goal(goal_text)
            self._log_decision("goal_custom", f"Custom goal: {goal_text}")
        self.conversation_history = []

    def handle_user_message(self, message: str) -> str:
        """Process a user message (from Telegram or frontend). Returns agent's text response."""
        msg_lower = message.lower()

        # Detect goal-setting messages
        goal_keywords = ["watch", "monitor", "guard", "protect", "keep an eye",
                         "don't let", "alert me", "tell me if", "warn me"]
        coaching_keywords = ["coach me", "teach me", "train me", "help me learn",
                             "practice", "show me how", "correct my"]

        if any(kw in msg_lower for kw in coaching_keywords):
            self.set_goal(message)
            prompt = (
                f"The user wants coaching: \"{message}\"\n"
                "1. Use analyze_skill_from_description to understand what skill they want\n"
                "2. Check list_references for any matching expert references\n"
                "3. Use start_coaching_session to begin\n"
                "4. Use speak_to_user to greet them and explain what you'll coach\n"
                "5. Start monitoring their form with compare_to_reference or get_pose_landmarks"
            )
        elif any(kw in msg_lower for kw in goal_keywords):
            self.set_goal(message)
            prompt = (
                f"The user set a new goal: \"{message}\"\n"
                "Acknowledge it. Then use get_spatial_state to assess the current scene "
                "and confirm you're ready to monitor."
            )
        else:
            prompt = f"User asks: \"{message}\""

        return self._run_agent_loop(prompt)

    def handle_event(self, event_description: str) -> str | None:
        """
        Called by the monitoring loop when something noteworthy happens.
        Returns the agent's response, or None if event was suppressed by cooldown.
        """
        # Cooldown: don't spam alerts for the same type of event
        event_key = event_description[:50]
        now = time.time()
        last = self._last_event_time.get(event_key, 0)
        if now - last < config.EVENT_COOLDOWN:
            return None
        self._last_event_time[event_key] = now

        prompt = (
            f"PROACTIVE ALERT — Your monitoring system detected:\n"
            f"{event_description}\n\n"
            "Assess using get_spatial_state. If genuinely concerning, "
            "send_telegram_alert to the user with a clear, actionable message. "
            "Include a photo if the situation warrants it."
        )
        return self._run_agent_loop(prompt)

    def periodic_check(self) -> str | None:
        """
        Called periodically by the heartbeat loop.
        Agent checks current state and decides if anything needs attention.
        Returns response or None if all is quiet.
        """
        # If a coaching goal is active, use coaching-specific check
        if self.active_goal.category == "skill_coaching":
            return self.periodic_coaching_check()

        prompt = (
            "Periodic check. Use get_scene_changes or get_spatial_state to see what's happening. "
            "If there are active risk events or anything notable relevant to the goal, alert the user. "
            "If everything is calm, respond with just 'ALL_CLEAR' (no alert needed)."
        )
        response = self._run_agent_loop(prompt)
        if response and "ALL_CLEAR" in response:
            return None
        return response

    def periodic_coaching_check(self) -> str | None:
        """
        Called periodically during skill coaching sessions.
        Checks form, counts reps, and gives voice feedback.
        """
        prompt = (
            "Coaching check. Do the following:\n"
            "1. Use compare_to_reference (if reference loaded) OR get_pose_landmarks to see the user's current form\n"
            "2. If score < 70%, use speak_to_user with a SHORT correction (e.g., 'Bend knees deeper')\n"
            "3. If score > 90%, use speak_to_user to encourage (e.g., 'Perfect form!')\n"
            "4. Use get_rep_count to check if a new rep was completed — announce it\n"
            "5. Every 5th check, use detect_compensation_patterns for safety\n"
            "If no coaching session is active, respond 'ALL_CLEAR'."
        )
        response = self._run_agent_loop(prompt)
        if response and "ALL_CLEAR" in response:
            return None
        return response

    # ── MCP Tool Discovery ───────────────────────────────────────────────

    def _ensure_mcp_tools(self):
        """Discover tools from MCP server (once). Converts to Anthropic format."""
        if self._mcp_initialized:
            return
        try:
            self._mcp_tools = _discover_mcp_tools()
            self._mcp_initialized = True
            print(f"[AegisAgent] Discovered {len(self._mcp_tools)} MCP tools")
        except Exception as e:
            print(f"[AegisAgent] MCP discovery failed: {e}, using fallback")
            self._mcp_tools = []
            self._mcp_initialized = True

    def _get_tools(self) -> list[dict]:
        """Get tool definitions in Anthropic format (auto-discovered from MCP)."""
        self._ensure_mcp_tools()
        return self._mcp_tools

    # ── MCP Tool Execution ───────────────────────────────────────────────

    def _execute_tool(self, tool_name: str, tool_input: dict) -> str:
        """Execute a tool via the MCP server and return result as string."""
        try:
            result = _call_mcp_tool(tool_name, tool_input)
            return result
        except Exception as e:
            return f"Tool error ({tool_name}): {str(e)}"

    # ── Agent loop ───────────────────────────────────────────────────────

    def _run_agent_loop(self, prompt: str, max_turns: int = 10) -> str:
        """
        Run the agent loop: send prompt → Claude responds → execute tools → repeat
        until Claude produces a final text response.
        """
        self.conversation_history.append({"role": "user", "content": prompt})

        # Trim history to prevent context overflow (keep last 20 turns)
        if len(self.conversation_history) > 40:
            self.conversation_history = self.conversation_history[-20:]

        system = BASE_SYSTEM_PROMPT.format(goal_supplement=self.active_goal.system_supplement)

        for _ in range(max_turns):
            try:
                response = self.client.messages.create(
                    model=config.AGENT_MODEL,
                    max_tokens=1024,
                    system=system,
                    tools=self._get_tools(),
                    messages=self.conversation_history,
                )
            except Exception as e:
                error_msg = f"Agent error: {str(e)}"
                print(f"[AegisAgent] {error_msg}")
                return error_msg

            # Add assistant response to history
            self.conversation_history.append({
                "role": "assistant",
                "content": response.content,
            })

            # If Claude wants to use tools, execute them and continue
            if response.stop_reason == "tool_use":
                tool_results = []
                for block in response.content:
                    if block.type == "tool_use":
                        print(f"[AegisAgent] Tool: {block.name}({json.dumps(block.input)[:100]})")
                        result = self._execute_tool(block.name, block.input)
                        self._log_tool_call(block.name, block.input, result)
                        tool_results.append({
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": result,
                        })

                self.conversation_history.append({
                    "role": "user",
                    "content": tool_results,
                })
            else:
                # Claude produced a final response — extract text
                text_parts = [b.text for b in response.content if hasattr(b, "text")]
                final = "\n".join(text_parts) if text_parts else "(no response)"
                self._log_decision("agent_response", final[:200])
                return final

        return "(agent reached max tool turns)"

    # ── Logging helpers (for demo dashboard) ──────────────────────────────

    def _log_tool_call(self, tool_name: str, tool_input: dict, result: str):
        entry = {
            "timestamp": datetime.now().isoformat(),
            "goal": self.active_goal.name,
            "tool": tool_name,
            "input": tool_input,
            "result_preview": result[:300],
        }
        self._tool_log.append(entry)
        if len(self._tool_log) > self._max_log_entries:
            self._tool_log = self._tool_log[-self._max_log_entries:]

    def _log_decision(self, decision_type: str, detail: str):
        entry = {
            "timestamp": datetime.now().isoformat(),
            "goal": self.active_goal.name,
            "type": decision_type,
            "detail": detail,
        }
        self._decision_log.append(entry)
        if len(self._decision_log) > self._max_log_entries:
            self._decision_log = self._decision_log[-self._max_log_entries:]


# ═══════════════════════════════════════════════════════════════════════
# MCP Bridge — connects agent to MCP server in-process
# ═══════════════════════════════════════════════════════════════════════

def _get_or_create_event_loop():
    """Get the running event loop or create a new one for sync contexts."""
    try:
        loop = asyncio.get_running_loop()
        return loop, False
    except RuntimeError:
        loop = asyncio.new_event_loop()
        return loop, True


def _discover_mcp_tools() -> list[dict]:
    """Discover tools from MCP server and convert to Anthropic tool format."""
    async def _discover():
        from fastmcp import Client
        from aegis.mcp_server import mcp as mcp_server
        async with Client(mcp_server) as client:
            mcp_tools = await client.list_tools()
            anthropic_tools = []
            for t in mcp_tools:
                tool_def = {
                    "name": t.name,
                    "description": t.description or "",
                    "input_schema": t.inputSchema,
                }
                anthropic_tools.append(tool_def)
            return anthropic_tools

    loop, created = _get_or_create_event_loop()
    try:
        if created:
            return loop.run_until_complete(_discover())
        else:
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as pool:
                future = pool.submit(asyncio.run, _discover())
                return future.result(timeout=10)
    finally:
        if created:
            loop.close()


def _call_mcp_tool(tool_name: str, tool_input: dict) -> str:
    """Call an MCP tool and return the result as a string."""
    async def _call():
        from fastmcp import Client
        from aegis.mcp_server import mcp as mcp_server
        async with Client(mcp_server) as client:
            result = await client.call_tool(tool_name, tool_input)
            # Extract text content from the result
            if result.content:
                texts = []
                for block in result.content:
                    if hasattr(block, 'text'):
                        texts.append(block.text)
                return "\n".join(texts) if texts else json.dumps(result.data)
            return json.dumps(result.data) if result.data else "(empty result)"

    loop, created = _get_or_create_event_loop()
    try:
        if created:
            return loop.run_until_complete(_call())
        else:
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as pool:
                future = pool.submit(asyncio.run, _call())
                return future.result(timeout=30)
    finally:
        if created:
            loop.close()
