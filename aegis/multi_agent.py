"""
AEGIS Multi-Agent System — Router + Specialist Sub-Agents.

Architecture:
  User Message → Router Agent (classifies intent, no tools)
                    ↓
              ┌─────┼─────────┐
              ↓     ↓         ↓
         Perception  Coach   Progress
         Agent       Agent   Agent
         (10 tools)  (12 tools) (8 tools)

Each sub-agent gets ONLY the tools it needs (10-12 max),
which is the sweet spot for Claude tool use performance.

The Router agent uses NO tools — it's a pure classifier that:
1. Reads the user's message + current goal
2. Decides which sub-agent(s) to invoke
3. Synthesizes their responses

This gives us:
- Better tool selection accuracy (fewer tools per call)
- Separation of concerns (each agent is an expert)
- Cleaner logging (know exactly which subsystem responded)
- Ability to run sub-agents in parallel for some queries
"""

import json
import time
import asyncio
from datetime import datetime
from typing import Optional
import anthropic

from aegis import config
from aegis.goals import Goal, PRESET_GOALS, create_custom_goal
from aegis.memory import MemoryStore


# ═══════════════════════════════════════════════════════════════════════
# TOOL GROUPS — each sub-agent gets a focused subset
# ═══════════════════════════════════════════════════════════════════════

PERCEPTION_TOOLS = {
    "get_spatial_state", "get_spatial_summary", "get_person_detail",
    "get_scene_changes", "get_objects_in_scene", "count_objects",
    "analyze_posture", "get_pose_landmarks", "check_body_alignment",
    "get_activity_timeline",
}

COACH_TOOLS = {
    "compare_to_reference", "get_joint_deviation",
    "get_movement_quality_analysis", "detect_compensation_patterns",
    "compare_full_movement", "start_coaching_session",
    "get_coaching_progress", "get_rep_count", "end_coaching_session",
    "analyze_skill_from_description", "speak_to_user",
    "record_reference_start", "record_reference_stop",
}

PROGRESS_TOOLS = {
    "list_references", "load_reference_from_current",
    "parse_skill_document", "get_current_goal",
    "update_goal", "get_goal_presets",
    "save_observation", "get_observations",
}

UTILITY_TOOLS = {
    "send_telegram_alert", "capture_photo",
    "set_watch_zone", "clear_watch_zones", "check_zone_status",
    "get_time_in_activity", "get_session_stats",
    "web_search", "get_current_time",
}


# ═══════════════════════════════════════════════════════════════════════
# SYSTEM PROMPTS — focused per sub-agent
# ═══════════════════════════════════════════════════════════════════════

ROUTER_PROMPT = """You are the AEGIS router. Given a user message and current context, decide which specialist agent(s) should handle it.

Reply with EXACTLY one JSON object (no markdown, no explanation):
{{
  "agents": ["perception" | "coach" | "progress"],
  "intent": "brief description of what the user wants",
  "coaching_active": true/false
}}

Rules:
- "perception": spatial awareness, what's happening in the scene, posture check, person detection
- "coach": active coaching, form correction, start/stop sessions, record references, voice feedback
- "progress": skill graphs, training data, references list, goals, session history, observations
- You can list multiple agents if the query spans concerns (e.g., ["perception", "coach"])
- If unsure, default to ["coach"] for coaching goals, ["perception"] for spatial goals

Current goal: {goal_name} (category: {goal_category})
Memory context: {memory_context}"""

PERCEPTION_PROMPT = """You are the AEGIS Perception Agent. You analyze the physical space and people in it.

You have 10 tools for spatial awareness:
- Scene: get_spatial_state, get_spatial_summary, get_scene_changes
- People: get_person_detail, count_objects, get_objects_in_scene
- Pose: analyze_posture, get_pose_landmarks, check_body_alignment
- Activity: get_activity_timeline

Be precise with spatial data. Report what you see factually.

{user_context}
{goal_supplement}"""

COACH_PROMPT = """You are the AEGIS Coaching Agent. You provide real-time movement coaching through voice.

You have 13 tools for skill coaching:
- Compare: compare_to_reference, get_joint_deviation, compare_full_movement
- Quality: get_movement_quality_analysis, detect_compensation_patterns
- Session: start_coaching_session, get_coaching_progress, get_rep_count, end_coaching_session
- Intelligence: analyze_skill_from_description
- Voice: speak_to_user
- Reference: record_reference_start, record_reference_stop

COACHING STYLE:
- Voice cues must be SHORT during movement: "Knees wider", "Chest up", "Good!"
- Detailed feedback only between sets or at session end
- Always use speak_to_user for real-time corrections
- Track reps and announce them
- Watch for compensation patterns (injury risk)

{user_context}
{goal_supplement}"""

PROGRESS_PROMPT = """You are the AEGIS Progress Agent. You manage skill progression, references, goals, and memory.

You have 8 tools:
- References: list_references, load_reference_from_current
- Intelligence: parse_skill_document
- Goals: get_current_goal, update_goal, get_goal_presets
- Memory: save_observation, get_observations

When asked about progress, provide data-driven summaries.
When managing goals, help the user find the right coaching mode.
Save important observations for future reference.

{user_context}
{goal_supplement}"""


# ═══════════════════════════════════════════════════════════════════════
# SUB-AGENT CLASS
# ═══════════════════════════════════════════════════════════════════════

class SubAgent:
    """A specialist sub-agent with a focused tool set."""

    def __init__(self, name: str, system_prompt_template: str,
                 tool_names: set[str], all_tools: list[dict],
                 client: anthropic.Anthropic):
        self.name = name
        self.system_template = system_prompt_template
        self.tool_names = tool_names
        self.client = client

        # Filter tools to only this agent's subset
        self.tools = [t for t in all_tools if t["name"] in tool_names]
        self.conversation_history: list[dict] = []

    def run(self, prompt: str, system_context: dict,
            execute_tool_fn, max_turns: int = 8) -> str:
        """Run the sub-agent loop with its focused tool set."""
        system = self.system_template.format(**system_context)

        self.conversation_history.append({"role": "user", "content": prompt})

        # Trim history
        if len(self.conversation_history) > 20:
            self.conversation_history = self.conversation_history[-10:]

        tool_calls = []

        for _ in range(max_turns):
            try:
                response = self.client.messages.create(
                    model=config.AGENT_MODEL,
                    max_tokens=1024,
                    system=system,
                    tools=self.tools,
                    messages=self.conversation_history,
                )
            except Exception as e:
                return f"[{self.name}] Error: {e}"

            self.conversation_history.append({
                "role": "assistant",
                "content": response.content,
            })

            if response.stop_reason == "tool_use":
                tool_results = []
                for block in response.content:
                    if block.type == "tool_use":
                        result = execute_tool_fn(block.name, block.input)
                        tool_calls.append({
                            "agent": self.name,
                            "tool": block.name,
                            "input": block.input,
                            "result_preview": result[:200],
                        })
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
                text_parts = [b.text for b in response.content if hasattr(b, "text")]
                return "\n".join(text_parts) if text_parts else "(no response)"

        return f"[{self.name}] reached max tool turns"

    def reset(self):
        """Clear conversation history."""
        self.conversation_history = []


# ═══════════════════════════════════════════════════════════════════════
# MULTI-AGENT ORCHESTRATOR
# ═══════════════════════════════════════════════════════════════════════

class MultiAgent:
    """Orchestrates Router + Specialist sub-agents."""

    def __init__(self, spatial_engine, telegram_sender=None):
        self.client = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)
        self.engine = spatial_engine
        self.telegram_sender = telegram_sender
        self.active_goal: Goal = PRESET_GOALS["general"]
        self.memory = MemoryStore()

        self._last_event_time: dict[str, float] = {}

        # ── MCP tools (discovered from server) ──
        self._all_mcp_tools: list[dict] = []
        self._mcp_initialized = False

        # ── Sub-agents (initialized after MCP discovery) ──
        self._perception: SubAgent | None = None
        self._coach: SubAgent | None = None
        self._progress: SubAgent | None = None

        # ── Logging ──
        self._tool_log: list[dict] = []
        self._decision_log: list[dict] = []
        self._max_log_entries = 200

    # ── Initialization ────────────────────────────────────────────────

    def _ensure_initialized(self):
        """Discover MCP tools and initialize sub-agents."""
        if self._mcp_initialized:
            return

        try:
            self._all_mcp_tools = _discover_mcp_tools()
            print(f"[MultiAgent] Discovered {len(self._all_mcp_tools)} MCP tools")
        except Exception as e:
            print(f"[MultiAgent] MCP discovery failed: {e}")
            self._all_mcp_tools = []

        # Create sub-agents with filtered tool sets
        self._perception = SubAgent(
            "perception", PERCEPTION_PROMPT,
            PERCEPTION_TOOLS, self._all_mcp_tools, self.client,
        )
        self._coach = SubAgent(
            "coach", COACH_PROMPT,
            COACH_TOOLS, self._all_mcp_tools, self.client,
        )
        self._progress = SubAgent(
            "progress", PROGRESS_PROMPT,
            PROGRESS_TOOLS, self._all_mcp_tools, self.client,
        )

        tool_counts = {
            "perception": len(self._perception.tools),
            "coach": len(self._coach.tools),
            "progress": len(self._progress.tools),
        }
        print(f"[MultiAgent] Sub-agents initialized: {tool_counts}")
        self._mcp_initialized = True

    # ── Public API ────────────────────────────────────────────────────

    @property
    def goal_name(self) -> str:
        return self.active_goal.name

    @property
    def goal_id(self) -> str:
        return self.active_goal.goal_id

    @property
    def conversation_history(self) -> list:
        """Combined conversation history from all sub-agents."""
        histories = []
        for agent in [self._perception, self._coach, self._progress]:
            if agent:
                histories.extend(agent.conversation_history)
        return histories

    def get_tool_log(self, last_n: int = 50) -> list[dict]:
        return self._tool_log[-last_n:]

    def get_decision_log(self, last_n: int = 50) -> list[dict]:
        return self._decision_log[-last_n:]

    def set_goal_by_id(self, goal_id: str) -> Goal:
        """Set a preset goal by ID."""
        goal = PRESET_GOALS.get(goal_id)
        if goal:
            self.active_goal = goal
            self._reset_sub_agents()
            self._log_decision("goal_set", f"Goal set to: {goal.name} ({goal_id})")
            return goal
        return self.active_goal

    def set_goal(self, goal_text: str):
        """Set goal using AI classification — no hardcoded keywords."""
        # Use Claude to classify the goal
        goal = self._classify_goal(goal_text)
        self.active_goal = goal
        self._reset_sub_agents()
        self._log_decision("goal_set", f"Goal: {goal.name} from '{goal_text[:80]}'")

    def handle_user_message(self, message: str) -> str:
        """Process a user message through the multi-agent pipeline."""
        self._ensure_initialized()

        # Step 1: Route to the right sub-agent(s)
        routing = self._route(message)
        agents_to_use = routing.get("agents", ["coach"])
        intent = routing.get("intent", message)

        self._log_decision("routed", f"Intent: {intent} → agents: {agents_to_use}")

        # Step 2: Build context
        context = self._build_context(message)

        # Step 3: Run sub-agent(s) and collect responses
        responses = []
        for agent_name in agents_to_use:
            agent = self._get_agent(agent_name)
            if agent:
                resp = agent.run(
                    f"User: {message}",
                    context,
                    self._execute_tool,
                )
                responses.append((agent_name, resp))

        # Step 4: Combine responses
        if len(responses) == 1:
            final = responses[0][1]
        elif len(responses) > 1:
            final = "\n\n".join(f"[{name}] {resp}" for name, resp in responses)
        else:
            final = "I'm not sure how to help with that. Could you rephrase?"

        self._log_decision("response", final[:200])

        # Step 5: Auto-save relevant observations
        self.memory.add_observation(
            content=f"User asked: {message[:100]}. Response: {final[:100]}",
            category="interaction",
        )

        return final

    def handle_event(self, event_description: str) -> str | None:
        """Handle a proactive event detection."""
        event_key = event_description[:50]
        now = time.time()
        last = self._last_event_time.get(event_key, 0)
        if now - last < config.EVENT_COOLDOWN:
            return None
        self._last_event_time[event_key] = now

        self._ensure_initialized()
        context = self._build_context(event_description)

        # Route events to perception agent
        agent = self._perception
        prompt = (
            f"PROACTIVE ALERT — detected:\n{event_description}\n\n"
            "Assess the situation. If concerning, respond with what's happening."
        )
        return agent.run(prompt, context, self._execute_tool)

    def periodic_check(self) -> str | None:
        """Periodic check — routes to appropriate sub-agent based on goal."""
        self._ensure_initialized()
        context = self._build_context("periodic check")

        if self.active_goal.category == "skill_coaching":
            agent = self._coach
            prompt = (
                "Periodic coaching check:\n"
                "1. Check the user's current form with compare_to_reference or get_pose_landmarks\n"
                "2. Give voice feedback via speak_to_user if correction needed\n"
                "3. Check rep count and announce new reps\n"
                "If no coaching session is active, respond 'ALL_CLEAR'."
            )
        else:
            agent = self._perception
            prompt = (
                "Periodic spatial check. Assess what's happening. "
                "If anything notable, report it. Otherwise respond 'ALL_CLEAR'."
            )

        response = agent.run(prompt, context, self._execute_tool)
        if response and "ALL_CLEAR" in response:
            return None
        return response

    # ── Router (AI-powered, no hardcoded keywords) ────────────────────

    def _route(self, message: str) -> dict:
        """Use Claude to classify intent and route to sub-agents.
        No hardcoded keyword lists — pure AI classification.
        """
        memory_context = self.memory.get_context_for_agent(message)

        try:
            response = self.client.messages.create(
                model="claude-sonnet-4-20250514",  # Fast model for routing
                max_tokens=200,
                system=ROUTER_PROMPT.format(
                    goal_name=self.active_goal.name,
                    goal_category=self.active_goal.category,
                    memory_context=memory_context[:500],
                ),
                messages=[{"role": "user", "content": message}],
            )
            text = response.content[0].text.strip()

            # Parse JSON from response
            # Handle potential markdown wrapping
            if "```" in text:
                text = text.split("```")[1]
                if text.startswith("json"):
                    text = text[4:]
            return json.loads(text)
        except Exception as e:
            # Fallback: use goal category to decide
            if self.active_goal.category == "skill_coaching":
                return {"agents": ["coach"], "intent": message, "coaching_active": True}
            return {"agents": ["perception"], "intent": message, "coaching_active": False}

    def _classify_goal(self, goal_text: str) -> Goal:
        """Use Claude to match a goal to a preset — no keyword matching.
        Falls back to custom goal if no preset fits.
        """
        preset_list = "\n".join(
            f"- {g.goal_id}: {g.name} — {g.description}"
            for g in PRESET_GOALS.values()
        )

        try:
            response = self.client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=100,
                system=(
                    "You classify user goals into preset categories. "
                    "Reply with ONLY the goal_id that best matches, or 'custom' if none fit.\n\n"
                    f"Available presets:\n{preset_list}"
                ),
                messages=[{"role": "user", "content": goal_text}],
            )
            goal_id = response.content[0].text.strip().lower()
            if goal_id in PRESET_GOALS:
                return PRESET_GOALS[goal_id]
        except Exception:
            pass

        return create_custom_goal(goal_text)

    # ── Context Building ──────────────────────────────────────────────

    def _build_context(self, query: str) -> dict:
        """Build the system prompt context for sub-agents."""
        memory_ctx = self.memory.get_context_for_agent(query)
        return {
            "user_context": memory_ctx if memory_ctx else "No user history yet.",
            "goal_supplement": self.active_goal.system_supplement,
        }

    # ── Agent helpers ─────────────────────────────────────────────────

    def _get_agent(self, name: str) -> SubAgent | None:
        agents = {
            "perception": self._perception,
            "coach": self._coach,
            "progress": self._progress,
        }
        return agents.get(name)

    def _reset_sub_agents(self):
        """Reset all sub-agent conversation histories on goal change."""
        for agent in [self._perception, self._coach, self._progress]:
            if agent:
                agent.reset()

    def _execute_tool(self, tool_name: str, tool_input: dict) -> str:
        """Execute a tool via MCP and log it."""
        try:
            result = _call_mcp_tool(tool_name, tool_input)
            self._log_tool_call(tool_name, tool_input, result)
            return result
        except Exception as e:
            error = f"Tool error ({tool_name}): {str(e)}"
            self._log_tool_call(tool_name, tool_input, error)
            return error

    # ── Logging ───────────────────────────────────────────────────────

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
# MCP Bridge (shared with old agent.py — same implementation)
# ═══════════════════════════════════════════════════════════════════════

def _get_or_create_event_loop():
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
            return [
                {
                    "name": t.name,
                    "description": t.description or "",
                    "input_schema": t.inputSchema,
                }
                for t in mcp_tools
            ]

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
