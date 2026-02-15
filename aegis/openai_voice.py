"""
OpenAI Realtime Voice Bridge — Best-in-class conversational coaching voice.

Uses OpenAI's Realtime API (GPT-4o) for bidirectional voice:
  - User speaks → GPT-4o hears, reasons about coaching context, responds naturally
  - Backend injects coaching data via session.update (no conversation pollution)
  - Coaching cues delivered via response.create with per-response instructions

Architecture:
  Mic (frontend) → /ws/audio → OpenAI Realtime API → audio out → frontend speaker
  Coaching data (backend) → session.update instructions → GPT-4o sees as system context
  speak() → response.create with instructions → clean, no fake user messages
"""

import asyncio
import base64
import json
import os
import time
from typing import Optional, Callable

import websockets

OPENAI_REALTIME_URL = "wss://api.openai.com/v1/realtime?model=gpt-4o-realtime-preview-2024-12-17"


def _build_coaching_instructions(skill: str = "", session_history: str = "", coaching_data: str = "") -> str:
    base = """You are AEGIS, an AI physical skill coach. You are currently in a live coaching session, watching the user through their camera and helping them practice a physical skill.

## Your Role
- You are a warm, encouraging personal coach speaking to the user in real-time
- You speak naturally — short, punchy cues during movement, longer explanations during rest
- Keep ALL responses to 1-2 sentences max unless the user asks a question

## How to Coach
- Count reps aloud: "That's 3! Nice one."
- Give specific body cues: "Push through your heels" not "Do better"
- Celebrate milestones: "Your best rep yet! Score of 92!"
- If they ask a question, answer it fully
- If they say "slower" or "faster" or "harder" — adapt
- Track improvement: "Your hip rotation is way better than rep 3"
- If they seem tired: "Two more reps, you've got this"

## Emotional Tone Adaptation
- If scores are DROPPING (3+ reps declining): Switch to gentler, supportive tone.
  Say things like "Let's slow down and focus on form" or "Take a breath, no rush"
- If scores are IMPROVING: Match their energy! "You're on fire! That was your best one!"
- If score hits a new BEST: Big celebration — "YES! New personal best! 94 out of 100!"
- If they sound FRUSTRATED: Be empathetic — "I know it's tough. Let's break it down step by step"
- If they sound EXCITED: Be excited with them — mirror their energy

## Multi-Language Support
- If the user speaks in a different language, RESPOND IN THAT LANGUAGE
- If they say "coach me in Spanish" → switch to Spanish coaching
- Maintain the same coaching quality in any language

## Skill-Specific Coaching
- Physical Therapy: Be gentle, focus on safety, emphasize range of motion
- Yoga: Be calm, breathing cues, "inhale as you extend"
- Sign Language: Guide hand shapes precisely, "curl your ring finger more"
- Dance: Be energetic, count beats, "and 5, 6, 7, 8!"
- Fitness: Be motivating, "one more! push through!"

## Key Rules
- NEVER give medical advice — say "check with your doctor"
- If dangerous form → STOP them: "Hold on, let's fix your back position"
- Keep responses to 1-2 sentences during active movement
- Always be encouraging — never make them feel bad
- You can hear them talk — respond to their questions naturally
- If camera coverage is poor, guide them to reposition
- Do NOT repeat yourself or echo coaching data back verbatim"""

    extras = []
    if skill:
        extras.append(f"\n## Current Skill: {skill}")
    if session_history:
        extras.append(f"\n## Session History\n{session_history}")
    if coaching_data:
        extras.append(f"\n## Live Coaching Data (use to inform cues, do NOT read aloud)\n{coaching_data}")

    return base + "\n".join(extras)


class OpenAIVoiceBridge:
    """Bidirectional voice bridge using OpenAI Realtime API.

    Key design decisions for conversation quality:
    - Coaching data injected via session.update (no conversation pollution)
    - speak() uses response.create with per-response instructions (no fake user messages)
    - Speak lock prevents overlapping coaching responses
    - Audio queue uses put_nowait to never block the receive loop
    """

    def __init__(self, api_key: str):
        self._api_key = api_key
        self._ws: Optional[websockets.WebSocketClientProtocol] = None
        self._connected = False
        self._receive_task: Optional[asyncio.Task] = None

        # Audio output queue (PCM 24kHz 16-bit mono)
        self._audio_out_queue: asyncio.Queue = asyncio.Queue(maxsize=500)

        # Callbacks
        self._on_transcript: Optional[Callable[[str], None]] = None
        self._on_audio_out: Optional[Callable[[bytes], None]] = None

        # State
        self._user_speaking = False
        self._response_in_progress = False
        self._speak_lock = asyncio.Lock()
        self._current_skill = ""
        self._session_history = ""
        self._latest_coaching_data = ""

    @property
    def is_connected(self) -> bool:
        return self._connected

    async def connect(self) -> bool:
        """Connect to OpenAI Realtime API."""
        try:
            headers = {
                "Authorization": f"Bearer {self._api_key}",
                "OpenAI-Beta": "realtime=v1",
            }
            self._ws = await websockets.connect(
                OPENAI_REALTIME_URL,
                additional_headers=headers,
                ping_interval=20,
                ping_timeout=10,
                max_size=10 * 1024 * 1024,
            )
            self._connected = True
            print("[OpenAI Voice] Connected to Realtime API")

            await self._configure_session()
            self._receive_task = asyncio.create_task(self._receive_loop())

            return True
        except Exception as e:
            print(f"[OpenAI Voice] Connection failed: {e}")
            self._connected = False
            return False

    async def _configure_session(self):
        """Configure the realtime session with coaching instructions."""
        session_config = {
            "type": "session.update",
            "session": {
                "modalities": ["text", "audio"],
                "instructions": _build_coaching_instructions(
                    skill=self._current_skill,
                    session_history=self._session_history,
                    coaching_data=self._latest_coaching_data,
                ),
                "voice": "alloy",
                "input_audio_format": "pcm16",
                "output_audio_format": "pcm16",
                "input_audio_transcription": {
                    "model": "whisper-1",
                },
                "turn_detection": {
                    "type": "server_vad",
                    "threshold": 0.7,
                    "prefix_padding_ms": 300,
                    "silence_duration_ms": 800,
                },
                "temperature": 0.7,
                "max_response_output_tokens": 100,
            },
        }
        await self._ws.send(json.dumps(session_config))
        print("[OpenAI Voice] Session configured (alloy voice, coaching mode)")

    async def disconnect(self):
        """Disconnect from OpenAI Realtime API."""
        self._connected = False
        if self._receive_task:
            self._receive_task.cancel()
            try:
                await self._receive_task
            except asyncio.CancelledError:
                pass
        if self._ws:
            await self._ws.close()
            self._ws = None
        print("[OpenAI Voice] Disconnected")

    async def send_audio(self, pcm_bytes: bytes):
        """Send raw PCM audio from user's mic (24kHz 16-bit mono)."""
        if not self._connected or not self._ws:
            return
        try:
            audio_b64 = base64.b64encode(pcm_bytes).decode("utf-8")
            await self._ws.send(json.dumps({
                "type": "input_audio_buffer.append",
                "audio": audio_b64,
            }))
        except Exception as e:
            print(f"[OpenAI Voice] Send audio error: {e}")

    async def inject_coaching_context(self, context: str):
        """Update session instructions with latest coaching data.

        Uses session.update to put coaching data in the system prompt.
        This does NOT create conversation items — keeps conversation clean.
        GPT-4o sees this data as background context for its next response.
        """
        if not self._connected or not self._ws:
            return
        if self._user_speaking or self._response_in_progress:
            return

        self._latest_coaching_data = context

        try:
            await self._ws.send(json.dumps({
                "type": "session.update",
                "session": {
                    "instructions": _build_coaching_instructions(
                        skill=self._current_skill,
                        session_history=self._session_history,
                        coaching_data=context,
                    ),
                },
            }))
        except Exception as e:
            print(f"[OpenAI Voice] Context inject error: {e}")

    def set_skill(self, skill_name: str):
        """Set the current skill being coached."""
        self._current_skill = skill_name

    def set_session_history(self, history: str):
        """Set session history for personalized coaching."""
        self._session_history = history

    async def speak(self, text: str):
        """Make the AI speak a coaching message.

        Uses response.create with per-response instructions.
        Does NOT create conversation items — no conversation pollution.
        Uses a lock to prevent overlapping coaching speech.
        """
        if not self._connected or not self._ws:
            return False
        if self._user_speaking:
            return False

        async with self._speak_lock:
            # Cancel any in-progress response first
            if self._response_in_progress:
                await self._cancel_current_response()

            try:
                await self._ws.send(json.dumps({
                    "type": "response.create",
                    "response": {
                        "modalities": ["text", "audio"],
                        "instructions": (
                            f"Say this coaching cue to the user warmly and naturally. "
                            f"Do NOT add extra commentary or questions. Just deliver this cue concisely:\n\n"
                            f"{text}"
                        ),
                    },
                }))
                return True
            except Exception as e:
                print(f"[OpenAI Voice] Speak error: {e}")
                return False

    async def _cancel_current_response(self):
        """Cancel in-progress response and clear audio queue."""
        try:
            await self._ws.send(json.dumps({"type": "response.cancel"}))
        except Exception:
            pass
        self._response_in_progress = False
        self._flush_audio_queue()
        await asyncio.sleep(0.15)

    def _flush_audio_queue(self):
        """Clear all queued audio chunks."""
        while not self._audio_out_queue.empty():
            try:
                self._audio_out_queue.get_nowait()
            except asyncio.QueueEmpty:
                break

    async def get_audio_output(self, timeout: float = 0.1) -> Optional[bytes]:
        """Get next chunk of audio output (24kHz PCM 16-bit mono)."""
        try:
            return await asyncio.wait_for(self._audio_out_queue.get(), timeout=timeout)
        except asyncio.TimeoutError:
            return None

    async def _receive_loop(self):
        """Background task: receive events from OpenAI Realtime API."""
        try:
            async for raw_msg in self._ws:
                try:
                    event = json.loads(raw_msg)
                    event_type = event.get("type", "")

                    # Audio output delta — use put_nowait to never block
                    if event_type == "response.audio.delta":
                        audio_b64 = event.get("delta", "")
                        if audio_b64:
                            audio_bytes = base64.b64decode(audio_b64)
                            try:
                                self._audio_out_queue.put_nowait(audio_bytes)
                            except asyncio.QueueFull:
                                # Drop oldest chunk to make room
                                try:
                                    self._audio_out_queue.get_nowait()
                                except asyncio.QueueEmpty:
                                    pass
                                try:
                                    self._audio_out_queue.put_nowait(audio_bytes)
                                except asyncio.QueueFull:
                                    pass
                            if self._on_audio_out:
                                self._on_audio_out(audio_bytes)

                    # Audio transcript (what the AI is saying)
                    elif event_type == "response.audio_transcript.done":
                        text = event.get("transcript", "")
                        if text:
                            print(f"[OpenAI Voice] AI said: {text}")

                    # Session events
                    elif event_type == "session.created":
                        print("[OpenAI Voice] Session created successfully")
                    elif event_type == "session.updated":
                        pass  # Silent — frequent updates from context injection

                    # Errors
                    elif event_type == "error":
                        err = event.get("error", {})
                        print(f"[OpenAI Voice] Error: {err.get('message', err)}")

                    # User speaking state — critical for interruption handling
                    elif event_type == "input_audio_buffer.speech_started":
                        self._user_speaking = True
                        # Cancel current AI response immediately
                        if self._response_in_progress:
                            try:
                                await self._ws.send(json.dumps({"type": "response.cancel"}))
                            except Exception:
                                pass
                            self._response_in_progress = False
                        self._flush_audio_queue()

                    elif event_type == "input_audio_buffer.speech_stopped":
                        self._user_speaking = False

                    # User transcript
                    elif event_type == "conversation.item.input_audio_transcription.completed":
                        transcript = event.get("transcript", "")
                        if transcript:
                            print(f"[OpenAI Voice] User said: {transcript}")

                    # Response lifecycle
                    elif event_type == "response.created":
                        self._response_in_progress = True
                    elif event_type == "response.done":
                        self._response_in_progress = False

                except json.JSONDecodeError:
                    pass
                except Exception as e:
                    print(f"[OpenAI Voice] Event parse error: {e}")

        except asyncio.CancelledError:
            pass
        except websockets.ConnectionClosed:
            print("[OpenAI Voice] Connection closed")
            self._connected = False
        except Exception as e:
            print(f"[OpenAI Voice] Receive loop error: {e}")
            self._connected = False
