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
    base = """You are AEGIS, a real-time physical skill coach. You speak like a human personal trainer — warm, punchy, natural.

STYLE: Talk like a friend coaching you at the gym. Use short phrases. Sound human.
- Good: "Nice! Push those knees out more."
- Good: "That's 5! You're getting stronger."
- Good: "Whoa, slow down — let's fix that hip."
- Bad: "I can see that your form has improved significantly compared to your earlier repetitions."

RULES:
- MAX 1-2 short sentences. Under 20 words ideal.
- Use natural fillers: "okay", "alright", "nice", "there you go"
- Count reps: "That's 3!" / "Rep 7, nice form."
- Give body-specific cues: "knees out", "chest up", "squeeze at the top"
- Celebrate wins: "Yes! New best!" / "92 out of 100, that's fire!"
- If score drops 3+ reps: gentle — "Take a breath. Let's slow it down."
- If user asks a question, answer naturally and fully
- If user speaks another language, switch to that language
- NEVER give medical advice
- NEVER repeat yourself or read data aloud
- If you can't see their full body, say 'Step back a bit, I can't see your legs'"""

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

        # Interruption tracking (for conversation.item.truncate)
        self._current_response_id = None
        self._current_item_id = None
        self._audio_chunks_played = 0  # count of audio chunks sent to client

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
                    "threshold": 0.6,
                    "prefix_padding_ms": 300,
                    "silence_duration_ms": 500,
                },
                "temperature": 0.6,
                "max_response_output_tokens": 60,
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
                        "conversation": "none",  # out-of-band: doesn't pollute conversation history
                        "instructions": f"Say this exactly as a short coaching cue (under 15 words): {text}",
                        "max_response_output_tokens": 40,
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
                            self._audio_chunks_played += 1
                            try:
                                self._audio_out_queue.put_nowait(audio_bytes)
                            except asyncio.QueueFull:
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
                        # Truncate conversation to what user actually heard (WebSocket requirement)
                        if self._current_item_id:
                            audio_ms = int(self._audio_chunks_played * 60)  # ~60ms per chunk at 24kHz
                            try:
                                await self._ws.send(json.dumps({
                                    "type": "conversation.item.truncate",
                                    "item_id": self._current_item_id,
                                    "content_index": 0,
                                    "audio_end_ms": audio_ms,
                                }))
                            except Exception:
                                pass
                            self._current_item_id = None
                        self._audio_chunks_played = 0
                        self._flush_audio_queue()

                    elif event_type == "input_audio_buffer.speech_stopped":
                        self._user_speaking = False

                    # User transcript
                    elif event_type == "conversation.item.input_audio_transcription.completed":
                        transcript = event.get("transcript", "")
                        if transcript:
                            print(f"[OpenAI Voice] User said: {transcript}")

                    # Track response items for truncation
                    elif event_type == "response.output_item.added":
                        item = event.get("item", {})
                        if item.get("type") == "message" and item.get("role") == "assistant":
                            self._current_item_id = item.get("id")
                            self._audio_chunks_played = 0

                    # Response lifecycle
                    elif event_type == "response.created":
                        self._response_in_progress = True
                        self._current_response_id = event.get("response", {}).get("id")
                    elif event_type == "response.done":
                        self._response_in_progress = False
                        self._current_response_id = None

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
