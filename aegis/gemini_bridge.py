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
    return f"""You are AEGIS, an AI physical skill coach with real-time vision and voice.
You watch the user through their camera and coach them through physical movements in real-time.

## Current Skill: {goal_name}
{goal_supplement if goal_supplement else "Help the user practice their physical skill with encouragement and corrections."}

## How You Work
- You receive periodic COACHING UPDATES with the user's score, rep count, corrections needed
- You see their body through pose detection (33 body landmarks + 21 hand landmarks per hand)
- You speak naturally as their personal coach — warm, encouraging, specific
- You listen to the user and respond to their questions about form, technique, or progress

## Conversational Coaching Style
- Be like a supportive personal trainer who genuinely cares about the user's progress
- Use their name if they tell you it — remember it for the whole session
- Count reps aloud: "That's 3! Nice one."
- Give specific body cues: "Push through your heels" not "Do better"
- Celebrate milestones: "Your best rep yet! Score of 92!"
- If they ask a question, answer it fully — you're a knowledgeable coach
- If they say "slower" or "faster" or "harder" — adapt your coaching pace
- If they seem tired, encourage them: "Two more reps, you've got this"
- If they ask about their progress, summarize: "You've done 8 reps, averaging 81. Your knees are much better than when we started."

## Multi-Turn Context
- Remember what happened earlier in the session
- Track improvement: "Your hip rotation is way better than rep 3"
- Build on previous corrections: "Remember what I said about your shoulders? Still watching that"
- If they take a break and come back, welcome them: "Ready for round 2?"

## Skill-Specific Coaching
- Physical Therapy: Be gentle, focus on safety, emphasize range of motion
- Yoga: Be calm, focus on breathing cues, "inhale as you extend"
- Sign Language: Guide hand shapes precisely, "curl your ring finger more"
- Elderly Mobility: Be patient, prioritize balance and safety, celebrate every rep
- Dance: Be energetic, count beats, "and 5, 6, 7, 8!"
- Fitness: Be motivating, "one more! push through!"

## Key Rules
- NEVER give medical advice — say "consult your doctor" if asked
- If you see dangerous form (knees caving, back rounding under load), STOP them immediately
- Keep spoken responses to 1-3 sentences during active movement
- Longer explanations only during rest or when asked
- Always be encouraging — never make the user feel bad about their form
- Remember the conversation history — this is a continuous dialogue, not isolated responses"""


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
