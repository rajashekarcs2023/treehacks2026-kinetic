"""
Gemini Live Voice Bridge — Server-side real-time voice for AEGIS.

Connects to Gemini Live API via WebSocket for:
  1. Server-proxied voice (phone sends audio via /ws/audio → server → Gemini)
  2. Standalone device voice (Pi with local mic/speaker)
  3. Proactive narration triggered by the agent (speak_to_user tool)
  4. Goal-aware system instructions that update with goal changes

Architecture:
  Audio In (mic/phone) → Gemini Live → Audio Out (speaker/phone)
  Spatial State → injected as text context → Gemini narrates scene

Uses google.genai Python SDK for Gemini Live connection.
"""

import asyncio
import json
import time
import base64
from typing import Optional

from google import genai
from google.genai import types

from aegis import config

# Audio format constants (Gemini Live requirements)
SEND_SAMPLE_RATE = 16000   # 16kHz mono 16-bit PCM input
RECV_SAMPLE_RATE = 24000   # 24kHz mono 16-bit PCM output


def _build_system_instruction(goal_name: str = "General", goal_supplement: str = "") -> str:
    """Build the Gemini Live system instruction based on active goal."""
    return f"""You are AEGIS, a spatial AI device with real-time vision and voice.
You can see physical spaces through a camera and speak naturally to the user.

## Current Goal: {goal_name}
{goal_supplement if goal_supplement else "Monitor the space and describe what you see when asked."}

## How You Work
- You receive periodic SPATIAL STATE updates as text — these are what you "see"
- Each update has: people detected (with activities, positions), objects, risk events
- You speak naturally and conversationally about what's happening
- You are proactive — alert about important events without being asked

## Voice Style
- Be natural, like a helpful friend watching the space with the user
- Use spatial language: "to your left", "near the desk", "walking toward the door"
- Keep responses concise (1-3 sentences for casual updates, more for detailed descriptions)
- Match urgency to the situation: calm for routine, urgent for falls/dangers
- Adapt your tone to the goal: coach-like for exercise, security-focused for guarding, etc.

## Key Rules
- If someone FALLS → alert IMMEDIATELY with urgency
- If asked "what do you see?" → give a full scene description
- Don't repeat the same observation unless something changed
- Remember context from earlier in the conversation"""


class GeminiBridge:
    """Server-side Gemini Live voice bridge."""

    def __init__(self):
        self._client = genai.Client(api_key=config.GEMINI_API_KEY)
        self._session = None
        self._connected = False
        self._goal_name = "General"
        self._goal_supplement = ""
        self._last_spatial_context = ""
        self._last_context_time = 0
        self._context_interval = 3.0  # seconds between spatial updates to Gemini

        # Audio queues for WebSocket-proxied audio
        self._audio_in_queue: asyncio.Queue = asyncio.Queue(maxsize=50)
        self._audio_out_queue: asyncio.Queue = asyncio.Queue(maxsize=100)

        # Callbacks
        self._on_audio_out = None  # callback(bytes) for audio output
        self._on_transcript = None  # callback(str) for transcripts

        # State
        self._receive_task = None
        self._send_task = None
        self._session_cm = None  # async context manager for session

    @property
    def is_connected(self) -> bool:
        return self._connected

    def update_goal(self, goal_name: str, goal_supplement: str = ""):
        """Update the goal context. Takes effect on next connection."""
        self._goal_name = goal_name
        self._goal_supplement = goal_supplement

    async def connect(self) -> bool:
        """Connect to Gemini Live API."""
        if not config.GEMINI_API_KEY:
            print("[GeminiBridge] No GEMINI_API_KEY configured")
            return False

        try:
            live_config = types.LiveConnectConfig(
                response_modalities=["AUDIO"],
                speech_config=types.SpeechConfig(
                    voice_config=types.VoiceConfig(
                        prebuilt_voice_config=types.PrebuiltVoiceConfig(
                            voice_name=config.GEMINI_VOICE
                        )
                    )
                ),
                system_instruction=types.Content(
                    parts=[types.Part(
                        text=_build_system_instruction(
                            self._goal_name, self._goal_supplement
                        )
                    )]
                ),
            )

            self._session_cm = self._client.aio.live.connect(
                model=config.GEMINI_MODEL,
                config=live_config,
            )
            self._session = await self._session_cm.__aenter__()
            self._connected = True
            print(f"[GeminiBridge] Connected to Gemini Live ({config.GEMINI_MODEL})")

            # Start receive loop
            self._receive_task = asyncio.create_task(self._receive_loop())
            return True

        except Exception as e:
            print(f"[GeminiBridge] Connection failed: {e}")
            self._connected = False
            return False

    async def disconnect(self):
        """Disconnect from Gemini Live."""
        self._connected = False
        if self._receive_task:
            self._receive_task.cancel()
            try:
                await self._receive_task
            except asyncio.CancelledError:
                pass
        if self._send_task:
            self._send_task.cancel()
        if self._session_cm:
            try:
                await self._session_cm.__aexit__(None, None, None)
            except Exception:
                pass
            self._session_cm = None
            self._session = None
        print("[GeminiBridge] Disconnected")

    # ── Audio Input ────────────────────────────────────────────────────

    async def send_audio(self, pcm_bytes: bytes):
        """Send raw PCM audio to Gemini (16kHz, 16-bit, mono).

        Called by the /ws/audio WebSocket handler or local mic capture.
        """
        if not self._connected or not self._session:
            return
        try:
            await self._session.send_realtime_input(
                audio=types.Blob(data=pcm_bytes, mime_type="audio/pcm;rate=16000")
            )
        except Exception as e:
            print(f"[GeminiBridge] Send audio error: {e}")

    async def send_text(self, text: str):
        """Send a text message to Gemini (e.g., spatial context, user text command).

        This is added to the conversation context alongside audio.
        """
        if not self._connected or not self._session:
            return
        try:
            await self._session.send_client_content(
                turns=types.Content(
                    role="user",
                    parts=[types.Part(text=text)]
                ),
                turn_complete=True,
            )
        except Exception as e:
            print(f"[GeminiBridge] Send text error: {e}")

    async def inject_spatial_state(self, state: dict):
        """Inject spatial state as text context into the Gemini session.

        Rate-limited to avoid flooding. Builds a concise text summary
        from the structured spatial data.
        """
        now = time.time()
        if now - self._last_context_time < self._context_interval:
            return

        context = self._build_spatial_context(state)
        if context == self._last_spatial_context:
            return

        self._last_spatial_context = context
        self._last_context_time = now

        if self._connected and self._session:
            try:
                await self._session.send_client_content(
                    turns=types.Content(
                        role="user",
                        parts=[types.Part(text=context)]
                    ),
                    turn_complete=False,  # don't trigger a response, just context
                )
            except Exception as e:
                print(f"[GeminiBridge] Spatial context inject error: {e}")

    async def speak(self, text: str):
        """Make Gemini speak a specific message (used by agent's speak_to_user tool).

        Sends the text as a user message that prompts Gemini to say it aloud.
        """
        if not self._connected:
            return False
        await self.send_text(
            f"Please say the following to the user in a natural way: \"{text}\""
        )
        return True

    # ── Audio Output ──────────────────────────────────────────────────

    async def get_audio_output(self, timeout: float = 0.1) -> Optional[bytes]:
        """Get the next chunk of audio output (24kHz PCM).

        Returns None if no audio is available within timeout.
        Used by the /ws/audio WebSocket handler to send audio back to the phone.
        """
        try:
            return await asyncio.wait_for(self._audio_out_queue.get(), timeout=timeout)
        except asyncio.TimeoutError:
            return None

    # ── Internal ──────────────────────────────────────────────────────

    async def _receive_loop(self):
        """Background task: receive responses from Gemini Live."""
        try:
            while self._connected and self._session:
                turn = self._session.receive()
                async for response in turn:
                    # Audio output
                    if (response.server_content and
                            response.server_content.model_turn):
                        for part in response.server_content.model_turn.parts:
                            if part.inline_data and isinstance(part.inline_data.data, bytes):
                                await self._audio_out_queue.put(part.inline_data.data)
                                if self._on_audio_out:
                                    self._on_audio_out(part.inline_data.data)

                    # Transcript
                    if (response.server_content and
                            response.server_content.output_transcription):
                        text = response.server_content.output_transcription.text
                        if text and self._on_transcript:
                            self._on_transcript(text)

                    # Interruption — clear playback queue
                    if (response.server_content and
                            response.server_content.interrupted):
                        while not self._audio_out_queue.empty():
                            try:
                                self._audio_out_queue.get_nowait()
                            except asyncio.QueueEmpty:
                                break

        except asyncio.CancelledError:
            pass
        except Exception as e:
            print(f"[GeminiBridge] Receive loop error: {e}")
            self._connected = False

    @staticmethod
    def _build_spatial_context(state: dict) -> str:
        """Build a concise text summary from spatial state for Gemini."""
        parts = []
        persons = state.get("persons", [])
        objects = state.get("objects", [])
        risks = state.get("risk_events", [])

        if not persons:
            parts.append("No people detected.")
        else:
            p_descs = []
            for p in persons:
                desc = (
                    f"Person {p['track_id']}: {p.get('activity', 'detected')} "
                    f"at ({p['center']['x']},{p['center']['y']}), "
                    f"speed={round(p.get('speed_px_per_sec', 0))}px/s"
                )
                p_descs.append(desc)
            parts.append(f"{len(persons)} person(s): {'; '.join(p_descs)}")

        if objects:
            obj_counts = {}
            for o in objects:
                name = o["class_name"]
                obj_counts[name] = obj_counts.get(name, 0) + 1
            obj_str = ", ".join(f"{v}x {k}" for k, v in obj_counts.items())
            parts.append(f"Objects: {obj_str}")

        if risks:
            parts.append(
                f"RISK EVENTS: {'; '.join(r['description'] for r in risks)}"
            )

        return f"[SPATIAL STATE UPDATE] {' | '.join(parts)}"
