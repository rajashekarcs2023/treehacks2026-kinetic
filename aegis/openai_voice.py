"""
OpenAI Realtime Voice Bridge — Best-in-class conversational coaching voice.

Uses OpenAI's Realtime API (GPT-4o) for bidirectional voice:
  - User speaks → GPT-4o hears, reasons about coaching context, responds naturally
  - Backend injects coaching data (scores, reps, corrections) as context
  - GPT-4o speaks coaching feedback in a natural, encouraging voice

Architecture:
  Mic (frontend) → /ws/audio → OpenAI Realtime API → audio out → frontend speaker
  Coaching data (backend) → injected as text context → GPT-4o incorporates into voice
"""

import asyncio
import base64
import json
import os
import time
from typing import Optional, Callable

import websockets

OPENAI_REALTIME_URL = "wss://api.openai.com/v1/realtime?model=gpt-4o-realtime-preview-2024-12-17"


def _build_coaching_instructions() -> str:
    return """You are AEGIS, an AI physical skill coach. You are currently in a live coaching session, watching the user through their camera and helping them practice a physical skill.

## Your Role
- You are a warm, encouraging personal coach speaking to the user in real-time
- You receive periodic COACHING UPDATES with their score, rep count, and form corrections
- You speak naturally — short, punchy cues during movement, longer explanations during rest

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

## Proactive Check-ins
- If score drops 3 reps in a row: "Hey, want to take a quick break? Or should we try a different approach?"
- If no movement for 15+ seconds: "You still there? Ready when you are!"
- After 10 reps: "Great set! Want to keep going or take a breather?"
- If coverage warning: "I can't see your full body — try stepping back a bit"

## Multi-Language Support
- If the user speaks in a different language, RESPOND IN THAT LANGUAGE
- If they say "coach me in Spanish" or "en español" → switch to Spanish coaching
- If they say "coach me in Hindi" or use Hindi → switch to Hindi
- Maintain the same coaching quality in any language
- You can switch back to English if they ask

## Skill-Specific Coaching
- Physical Therapy: Be gentle, focus on safety, emphasize range of motion
- Yoga: Be calm, breathing cues, "inhale as you extend"
- Sign Language: Guide hand shapes precisely, "curl your ring finger more"
- Dance: Be energetic, count beats, "and 5, 6, 7, 8!"
- Fitness: Be motivating, "one more! push through!"
- Elderly Mobility: Be patient, prioritize balance and safety, celebrate every rep
- Tai Chi: Calm and flowing, focus on weight transfer and breathing
- Ergonomics: Desk posture cues, "relax your shoulders, screen at eye level"

## Session Memory
- When you receive [SESSION HISTORY], use it to personalize coaching
- Reference past performance: "Last time you averaged 78 on squats, let's beat that"
- Remember their weak points: "We worked on knee alignment last time — let's check that"
- Track improvement across sessions: "Your form has improved so much since we started!"

## Key Rules
- NEVER give medical advice — say "check with your doctor"
- If dangerous form → STOP them: "Hold on, let's fix your back position"
- Keep responses to 1-2 sentences during active movement
- Always be encouraging — never make them feel bad
- You can hear them talk — respond to their questions naturally
- If camera coverage is poor, guide them to reposition"""


class OpenAIVoiceBridge:
    """Bidirectional voice bridge using OpenAI Realtime API."""

    def __init__(self, api_key: str):
        self._api_key = api_key
        self._ws: Optional[websockets.WebSocketClientProtocol] = None
        self._connected = False
        self._receive_task: Optional[asyncio.Task] = None

        # Audio output queue (PCM 24kHz 16-bit mono)
        self._audio_out_queue: asyncio.Queue = asyncio.Queue(maxsize=200)

        # Callbacks
        self._on_transcript: Optional[Callable[[str], None]] = None
        self._on_audio_out: Optional[Callable[[bytes], None]] = None

        # State
        self._last_context_time = 0
        self._context_interval = 5.0
        self._user_speaking = False
        self._response_in_progress = False

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
                max_size=10 * 1024 * 1024,  # 10MB max message
            )
            self._connected = True
            print("[OpenAI Voice] Connected to Realtime API")

            # Configure session
            await self._configure_session()

            # Start receive loop
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
                "instructions": _build_coaching_instructions(),
                "voice": "alloy",
                "input_audio_format": "pcm16",
                "output_audio_format": "pcm16",
                "input_audio_transcription": {
                    "model": "whisper-1",
                },
                "turn_detection": {
                    "type": "server_vad",
                    "threshold": 0.5,
                    "prefix_padding_ms": 300,
                    "silence_duration_ms": 500,
                },
                "temperature": 0.7,
                "max_response_output_tokens": 150,
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
        """Send raw PCM audio from user's mic (16kHz 16-bit mono)."""
        if not self._connected or not self._ws:
            return
        try:
            audio_b64 = base64.b64encode(pcm_bytes).decode("utf-8")
            event = {
                "type": "input_audio_buffer.append",
                "audio": audio_b64,
            }
            await self._ws.send(json.dumps(event))
        except Exception as e:
            print(f"[OpenAI Voice] Send audio error: {e}")

    async def inject_coaching_context(self, context: str):
        """Inject coaching data as a text context message.

        This tells GPT-4o about the user's current score, reps, corrections, etc.
        GPT-4o will naturally incorporate this into its voice coaching.
        Skips injection while user is speaking or AI is responding.
        """
        now = time.time()
        if now - self._last_context_time < self._context_interval:
            return
        if self._user_speaking or self._response_in_progress:
            return
        self._last_context_time = now

        if not self._connected or not self._ws:
            return

        try:
            event = {
                "type": "conversation.item.create",
                "item": {
                    "type": "message",
                    "role": "user",
                    "content": [
                        {
                            "type": "input_text",
                            "text": f"[COACHING DATA — do not read this verbatim, use it to inform your next coaching cue]\n{context}",
                        }
                    ],
                },
            }
            await self._ws.send(json.dumps(event))
            # No response.create here — context is silent background info
            # Only speak() triggers actual voice output
        except Exception as e:
            print(f"[OpenAI Voice] Context inject error: {e}")

    async def speak(self, text: str):
        """Make the AI speak a specific coaching message.
        Skips if user is currently speaking."""
        if not self._connected or not self._ws:
            return False
        if self._user_speaking:
            return False  # Don't interrupt user

        try:
            event = {
                "type": "conversation.item.create",
                "item": {
                    "type": "message",
                    "role": "user",
                    "content": [
                        {
                            "type": "input_text",
                            "text": f"Say this to the user naturally (1-2 sentences max): {text}",
                        }
                    ],
                },
            }
            await self._ws.send(json.dumps(event))

            response_event = {
                "type": "response.create",
                "response": {
                    "modalities": ["text", "audio"],
                },
            }
            await self._ws.send(json.dumps(response_event))
            return True
        except Exception as e:
            print(f"[OpenAI Voice] Speak error: {e}")
            return False

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

                    # Audio output delta
                    if event_type == "response.audio.delta":
                        audio_b64 = event.get("delta", "")
                        if audio_b64:
                            audio_bytes = base64.b64decode(audio_b64)
                            await self._audio_out_queue.put(audio_bytes)
                            if self._on_audio_out:
                                self._on_audio_out(audio_bytes)

                    # Audio transcript
                    elif event_type == "response.audio_transcript.delta":
                        text = event.get("delta", "")
                        if text and self._on_transcript:
                            self._on_transcript(text)

                    # Session created
                    elif event_type == "session.created":
                        print("[OpenAI Voice] Session created successfully")

                    # Session updated
                    elif event_type == "session.updated":
                        print("[OpenAI Voice] Session configured")

                    # Error
                    elif event_type == "error":
                        err = event.get("error", {})
                        print(f"[OpenAI Voice] Error: {err.get('message', err)}")

                    # User started speaking — pause coaching
                    elif event_type == "input_audio_buffer.speech_started":
                        self._user_speaking = True
                        self._response_in_progress = False
                        # Clear queued audio so old coaching doesn't play
                        while not self._audio_out_queue.empty():
                            try:
                                self._audio_out_queue.get_nowait()
                            except asyncio.QueueEmpty:
                                break

                    # User stopped speaking
                    elif event_type == "input_audio_buffer.speech_stopped":
                        self._user_speaking = False

                    # Input audio transcription
                    elif event_type == "conversation.item.input_audio_transcription.completed":
                        transcript = event.get("transcript", "")
                        if transcript:
                            print(f"[OpenAI Voice] User said: {transcript}")

                    # Response started
                    elif event_type == "response.created":
                        self._response_in_progress = True

                    # Response done — resume coaching
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
