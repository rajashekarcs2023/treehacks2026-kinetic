"""
AEGIS Voice Narration — Real-time spatial audio descriptions.

Converts spatial state into spoken narration using system TTS.
Designed for accessibility: narrates what's happening in the space
so someone doesn't need to see the screen.

Architecture:
  SpatialEngine → VoiceNarrator → TTS (macOS `say` or pyttsx3)
  Narration runs in a background thread with a speech queue.
  Only speaks when something CHANGES (not every frame).
"""

import subprocess
import threading
import queue
import time
import platform


class VoiceNarrator:
    """Background TTS narrator for spatial state changes."""

    def __init__(self, enabled: bool = True, voice: str = "Samantha",
                 min_interval: float = 4.0):
        """
        Args:
            enabled: Whether voice is active
            voice: macOS voice name (Samantha, Alex, Karen, Daniel, etc.)
            min_interval: Minimum seconds between utterances
        """
        self.enabled = enabled
        self.voice = voice
        self.min_interval = min_interval
        self._queue: queue.Queue = queue.Queue(maxsize=3)
        self._thread = None
        self._running = False
        self._last_speak_time = 0
        self._last_state_hash = ""

    def start(self):
        """Start the voice thread."""
        if not self.enabled:
            return
        self._running = True
        self._thread = threading.Thread(target=self._speak_loop, daemon=True)
        self._thread.start()
        self.say("AEGIS voice narration active.")
        print("[Voice] Narrator started")

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=3)

    def say(self, text: str):
        """Queue a text utterance. Non-blocking, drops if queue is full."""
        if not self.enabled or not text:
            return
        try:
            self._queue.put_nowait(text)
        except queue.Full:
            pass  # drop if backed up

    def narrate_state(self, state: dict):
        """
        Generate and speak a narration for the current spatial state.
        Only speaks if something meaningful changed.
        """
        if not self.enabled or not state:
            return

        now = time.time()
        if now - self._last_speak_time < self.min_interval:
            return

        # Build a hash of the meaningful state to detect changes
        persons = state.get("persons", [])
        objects = state.get("objects", [])
        risks = state.get("risk_events", [])

        state_hash = self._compute_hash(persons, objects, risks)
        if state_hash == self._last_state_hash:
            return  # nothing changed
        self._last_state_hash = state_hash

        # Generate narration
        narration = self._generate_narration(persons, objects, risks)
        if narration:
            self.say(narration)

    def _compute_hash(self, persons, objects, risks) -> str:
        """Simple hash to detect meaningful changes."""
        parts = []
        for p in persons:
            # Track person count, rough position (grid), and activity
            gx = p["center"]["x"] // 200
            gy = p["center"]["y"] // 200
            parts.append(f"p{p['track_id']}:{gx},{gy}:{p.get('activity', '')}")
        for r in risks:
            parts.append(f"r:{r['event_id']}")
        # Object count by class (don't need exact positions)
        obj_counts = {}
        for o in objects:
            obj_counts[o["class_name"]] = obj_counts.get(o["class_name"], 0) + 1
        for k, v in sorted(obj_counts.items()):
            parts.append(f"o:{k}:{v}")
        return "|".join(parts)

    def _generate_narration(self, persons, objects, risks) -> str:
        """Generate a concise spoken description of the scene."""
        parts = []

        # ── Critical alerts first ────────────────────────────────────────
        for p in persons:
            if p.get("activity") in ("fallen", "lying_down"):
                parts.append(f"Alert! Person {p['track_id']} appears to have fallen!")
                return " ".join(parts)  # immediate, don't bury in other info

        if risks:
            for r in risks:
                parts.append(r["description"])
            return " ".join(parts)

        # ── People ───────────────────────────────────────────────────────
        if not persons:
            parts.append("No one in view.")
        elif len(persons) == 1:
            p = persons[0]
            activity = p.get("activity", "detected")
            direction = self._direction_word(p.get("direction_deg", 0))
            speed = p.get("speed_px_per_sec", 0)

            if activity == "sitting":
                parts.append("One person, sitting.")
            elif activity in ("walking", "running"):
                parts.append(f"One person, {activity} {direction}.")
            else:
                parts.append(f"One person, {activity}.")
        else:
            activities = {}
            for p in persons:
                a = p.get("activity", "detected")
                activities[a] = activities.get(a, 0) + 1

            descs = []
            for a, count in activities.items():
                if count == 1:
                    descs.append(f"one {a}")
                else:
                    descs.append(f"{count} {a}")
            parts.append(f"{len(persons)} people: {', '.join(descs)}.")

        # ── Notable objects (keep brief) ─────────────────────────────────
        if objects:
            obj_counts = {}
            for o in objects:
                obj_counts[o["class_name"]] = obj_counts.get(o["class_name"], 0) + 1
            notable = [f"{v} {k}{'s' if v > 1 else ''}"
                       for k, v in list(obj_counts.items())[:3]]
            if notable:
                parts.append(f"I see {', '.join(notable)}.")

        return " ".join(parts)

    @staticmethod
    def _direction_word(deg: float) -> str:
        """Convert angle to simple direction word."""
        if 45 <= deg < 135:
            return "to the right"
        elif 135 <= deg < 225:
            return "away"
        elif 225 <= deg < 315:
            return "to the left"
        else:
            return "forward"

    def _speak_loop(self):
        """Background thread: pull from queue and speak."""
        while self._running:
            try:
                text = self._queue.get(timeout=1)
            except queue.Empty:
                continue

            self._last_speak_time = time.time()
            self._speak(text)

    def _speak(self, text: str):
        """Speak text using system TTS."""
        try:
            if platform.system() == "Darwin":
                # macOS: use built-in `say` command
                subprocess.run(
                    ["say", "-v", self.voice, "-r", "200", text],
                    timeout=15, capture_output=True,
                )
            else:
                # Linux/other: try espeak
                subprocess.run(
                    ["espeak", text],
                    timeout=15, capture_output=True,
                )
        except FileNotFoundError:
            print(f"[Voice] TTS not available. Would say: {text}")
        except subprocess.TimeoutExpired:
            pass
        except Exception as e:
            print(f"[Voice] TTS error: {e}")
