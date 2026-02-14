"""
AEGIS Agent — Built on the Claude Agent SDK (claude-agent-sdk).

Uses ClaudeSDKClient for persistent sessions with:
  - Custom MCP tools (40 tools via create_sdk_mcp_server)
  - Sub-agents via AgentDefinition (perception, coach, progress)
  - Hooks for safety guardrails and audit logging
  - Session continuity across multiple exchanges

Architecture (from Claude Agent SDK docs):
  ClaudeSDKClient
    ├── MCP Server: aegis (40 custom tools)
    ├── Sub-agents (invoked via Task tool):
    │   ├── perception-agent (10 tools, read-only)
    │   ├── coach-agent (13 tools, voice + comparison)
    │   └── progress-agent (8 tools, memory + goals)
    └── Hooks:
        ├── PreToolUse: safety guardrails
        ├── PostToolUse: audit logging
        └── Stop: session summary
"""

import asyncio
import json
import time
from datetime import datetime
from typing import Any

from pathlib import Path

from claude_agent_sdk import (
    ClaudeSDKClient,
    ClaudeAgentOptions,
    AgentDefinition,
    HookMatcher,
    AssistantMessage,
    TextBlock,
    ResultMessage,
    SystemMessage,
    PreToolUseHookInput,
    PostToolUseHookInput,
    StopHookInput,
    HookContext,
)

# Project root (where .claude/skills/ lives)
PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)

from aegis.sdk_tools import (
    create_aegis_mcp_server,
    PERCEPTION_TOOL_NAMES,
    COACH_TOOL_NAMES,
    PROGRESS_TOOL_NAMES,
)
from aegis.memory import MemoryStore
from aegis.goals import Goal, PRESET_GOALS


# ═══════════════════════════════════════════════════════════════════════
# HOOKS — Safety, Logging, Session Summary
# ═══════════════════════════════════════════════════════════════════════

# Audit log (in-memory, exposed via API)
_audit_log: list[dict] = []
_max_log = 500


async def safety_hook(input_data: PreToolUseHookInput, tool_use_id: str | None,
                      context: HookContext) -> dict[str, Any]:
    """PreToolUse hook: block dangerous operations, validate inputs."""
    tool_name = input_data.get("tool_name", "")
    tool_input = input_data.get("tool_input", {})

    # Block zone modifications if coaching is active
    if "zone" in tool_name and _is_coaching_active():
        return {
            "decision": "block",
            "systemMessage": "Cannot modify zones during an active coaching session.",
        }

    return {}


async def audit_hook(input_data: PostToolUseHookInput, tool_use_id: str | None,
                     context: HookContext) -> dict[str, Any]:
    """PostToolUse hook: log all tool calls for dashboard and debugging."""
    tool_name = input_data.get("tool_name", "")
    tool_input = input_data.get("tool_input", {})

    entry = {
        "timestamp": datetime.now().isoformat(),
        "tool": tool_name,
        "input_preview": json.dumps(tool_input, default=str)[:200],
        "tool_use_id": tool_use_id,
    }
    _audit_log.append(entry)
    if len(_audit_log) > _max_log:
        _audit_log[:] = _audit_log[-_max_log:]

    return {}


async def stop_hook(input_data: StopHookInput, tool_use_id: str | None,
                    context: HookContext) -> dict[str, Any]:
    """Stop hook: save session summary to memory."""
    return {}


def _is_coaching_active() -> bool:
    """Check if a coaching session is active via mcp_server state."""
    from aegis.mcp_server import _coaching_session
    return _coaching_session is not None and _coaching_session.active


# ═══════════════════════════════════════════════════════════════════════
# SUB-AGENT DEFINITIONS
# ═══════════════════════════════════════════════════════════════════════

PERCEPTION_AGENT = AgentDefinition(
    description=(
        "Spatial perception specialist. Use when the user asks about "
        "what's happening in the scene, who is present, what objects "
        "are visible, or needs posture analysis."
    ),
    prompt=(
        "You are the AEGIS Perception Agent. You analyze the physical space "
        "and people in it using computer vision data.\n\n"
        "Be precise with spatial data. Report what you see factually.\n"
        "You have read-only tools — you observe but don't modify."
    ),
    tools=PERCEPTION_TOOL_NAMES,
    model="sonnet",
)

COACH_AGENT = AgentDefinition(
    description=(
        "Real-time movement coaching specialist. Use when the user wants "
        "skill coaching, form correction, rep counting, movement comparison "
        "to an expert reference, or needs voice feedback during exercise."
    ),
    prompt=(
        "You are the AEGIS Coaching Agent. You provide real-time movement "
        "coaching through voice.\n\n"
        "COACHING STYLE:\n"
        "- Voice cues via speak_to_user must be SHORT during movement: "
        "'Knees wider', 'Chest up', 'Good!'\n"
        "- Detailed feedback only between sets or at session end\n"
        "- Track reps and announce them\n"
        "- Watch for compensation patterns (injury risk)\n\n"
        "WORKFLOW:\n"
        "1. start_coaching_session to begin\n"
        "2. compare_to_reference every few seconds\n"
        "3. speak_to_user for corrections\n"
        "4. get_rep_count to track reps\n"
        "5. detect_compensation_patterns for safety\n"
        "6. end_coaching_session for summary"
    ),
    tools=COACH_TOOL_NAMES,
    model="sonnet",
)

PROGRESS_AGENT = AgentDefinition(
    description=(
        "Progress and memory specialist. Use when the user asks about "
        "their history, goals, available references, skill progression, "
        "or when information needs to be saved for later."
    ),
    prompt=(
        "You are the AEGIS Progress Agent. You manage skill progression, "
        "references, goals, and memory.\n\n"
        "When asked about progress, provide data-driven summaries.\n"
        "When managing goals, help the user find the right coaching mode.\n"
        "Save important observations for future reference."
    ),
    tools=PROGRESS_TOOL_NAMES,
    model="haiku",
)


# ═══════════════════════════════════════════════════════════════════════
# AEGIS AGENT CLASS
# ═══════════════════════════════════════════════════════════════════════

class AegisSDKAgent:
    """Production AEGIS agent built on claude-agent-sdk.

    Uses ClaudeSDKClient for:
    - Persistent session (conversation continuity)
    - Custom MCP tools (40 AEGIS tools)
    - Sub-agents (perception, coach, progress)
    - Hooks (safety, audit, stop)
    """

    def __init__(self):
        self.memory = MemoryStore()
        self.active_goal: Goal = PRESET_GOALS["general"]
        self._client: ClaudeSDKClient | None = None
        self._session_id: str | None = None
        self._connected = False
        self._last_event_time: dict[str, float] = {}

    def _build_options(self) -> ClaudeAgentOptions:
        """Build ClaudeAgentOptions with MCP server, sub-agents, and hooks."""
        aegis_server = create_aegis_mcp_server()

        # Build system prompt with user context
        user_ctx = self.memory.get_context_for_agent("")
        goal_ctx = self.active_goal.system_supplement

        system_prompt = (
            "You are AEGIS — an AI Skill Coach with Expert Motion Transfer. "
            "You see the user's body in real-time through a camera and coach "
            "them to match expert form through voice.\n\n"
            f"Current goal: {self.active_goal.name}\n\n"
        )
        if user_ctx:
            system_prompt += f"{user_ctx}\n\n"
        if goal_ctx:
            system_prompt += f"{goal_ctx}\n\n"

        return ClaudeAgentOptions(
            system_prompt=system_prompt,
            mcp_servers={"aegis": aegis_server},
            # Main agent gets Task (for sub-agents) + all MCP tools
            allowed_tools=["Task", "mcp__aegis__*"],
            permission_mode="bypassPermissions",
            agents={
                "perception-agent": PERCEPTION_AGENT,
                "coach-agent": COACH_AGENT,
                "progress-agent": PROGRESS_AGENT,
            },
            hooks={
                "PreToolUse": [
                    HookMatcher(
                        matcher="mcp__aegis__set_watch_zone|mcp__aegis__clear_watch_zones",
                        hooks=[safety_hook],
                    ),
                ],
                "PostToolUse": [
                    HookMatcher(
                        matcher=None,  # Match all tools
                        hooks=[audit_hook],
                    ),
                ],
                "Stop": [
                    HookMatcher(hooks=[stop_hook]),
                ],
            },
            max_turns=15,
            # Enable .claude/skills/ files for domain expertise
            setting_sources=["project"],
            # Set working directory so SDK finds .claude/ config
            cwd=PROJECT_ROOT,
        )

    async def connect(self):
        """Initialize the SDK client and connect."""
        if self._connected:
            return

        options = self._build_options()
        self._client = ClaudeSDKClient(options=options)
        await self._client.connect()
        self._connected = True

    async def disconnect(self):
        """Disconnect the SDK client."""
        if self._client and self._connected:
            await self._client.disconnect()
            self._connected = False
            self._client = None

    async def send_message(self, message: str) -> str:
        """Send a message to the agent and get the response.

        The ClaudeSDKClient maintains conversation context across calls.
        """
        if not self._connected:
            await self.connect()

        await self._client.query(message)

        response_text = ""
        async for msg in self._client.receive_response():
            if isinstance(msg, AssistantMessage):
                for block in msg.content:
                    if isinstance(block, TextBlock):
                        response_text += block.text
            elif isinstance(msg, ResultMessage):
                if not self._session_id and hasattr(msg, 'session_id'):
                    self._session_id = msg.session_id

        # Save interaction to memory
        self.memory.add_observation(
            content=f"User: {message[:100]}. Response: {response_text[:100]}",
            category="interaction",
        )

        return response_text or "(no response)"

    def set_goal_by_id(self, goal_id: str) -> Goal:
        """Set a preset goal by ID. Reconnects with updated system prompt."""
        goal = PRESET_GOALS.get(goal_id)
        if goal:
            self.active_goal = goal
            # Force reconnect to update system prompt
            if self._connected:
                asyncio.create_task(self._reconnect())
            return goal
        return self.active_goal

    async def _reconnect(self):
        """Reconnect with updated options (e.g., after goal change)."""
        await self.disconnect()
        await self.connect()

    # ── API compatibility (used by server.py) ─────────────────────────

    @property
    def goal_name(self) -> str:
        return self.active_goal.name

    @property
    def goal_id(self) -> str:
        return self.active_goal.goal_id

    @property
    def conversation_history(self) -> list:
        return []  # SDK manages this internally

    def get_tool_log(self, last_n: int = 50) -> list[dict]:
        return _audit_log[-last_n:]

    def get_decision_log(self, last_n: int = 50) -> list[dict]:
        return _audit_log[-last_n:]  # Unified log

    # ── Sync backward-compat (used by monitor.py, telegram_bot.py) ────

    def _run_sync(self, coro) -> str:
        """Run an async coroutine from a sync context (background thread)."""
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            # We're inside an existing event loop — run in a new thread
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as pool:
                future = pool.submit(asyncio.run, coro)
                return future.result(timeout=60)
        else:
            return asyncio.run(coro)

    def handle_user_message(self, message: str) -> str:
        """Sync wrapper for send_message (used by telegram_bot, main.py console)."""
        return self._run_sync(self.send_message(message))

    def handle_event(self, event_description: str) -> str | None:
        """Handle a proactive event from the monitor (sync)."""
        now = time.time()
        event_key = event_description[:50]
        last = self._last_event_time.get(event_key, 0)
        if now - last < 30:  # 30s cooldown
            return None
        self._last_event_time[event_key] = now

        prompt = (
            f"PROACTIVE ALERT — Your monitoring system detected:\n"
            f"{event_description}\n\n"
            "Assess the situation. If concerning, describe what's happening."
        )
        return self._run_sync(self.send_message(prompt))

    def periodic_check(self) -> str | None:
        """Periodic check from the monitor loop (sync)."""
        if self.active_goal.category == "skill_coaching":
            prompt = (
                "Periodic coaching check. Check form, count reps, give voice feedback. "
                "If no coaching session is active, respond 'ALL_CLEAR'."
            )
        else:
            prompt = (
                "Periodic check. Assess what's happening in the scene. "
                "If anything notable, report it. Otherwise respond 'ALL_CLEAR'."
            )
        response = self._run_sync(self.send_message(prompt))
        if response and "ALL_CLEAR" in response:
            return None
        return response

    def set_goal(self, goal_text: str):
        """Set goal from natural language text (sync, used by telegram_bot)."""
        from aegis.goals import match_goal_from_text, create_custom_goal
        matched = match_goal_from_text(goal_text)
        if matched:
            self.active_goal = matched
        else:
            self.active_goal = create_custom_goal(goal_text)
        if self._connected:
            self._run_sync(self._reconnect())


# ═══════════════════════════════════════════════════════════════════════
# Module-level convenience
# ═══════════════════════════════════════════════════════════════════════

def create_agent() -> AegisSDKAgent:
    """Create a new AEGIS agent instance."""
    return AegisSDKAgent()
