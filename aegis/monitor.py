"""
AEGIS Monitor — Proactive heartbeat loop.

Periodically checks the spatial state for noteworthy events
and invokes the agent to reason and act autonomously.
"""

import threading
import time

from aegis import config


class Monitor:
    """Proactive monitoring loop that watches for events and triggers the agent."""

    def __init__(self, spatial_engine, agent):
        self.engine = spatial_engine
        self.agent = agent
        self._running = False
        self._paused = False
        self._thread = None
        self._prev_risk_ids: set[str] = set()

    @property
    def is_active(self) -> bool:
        return self._running and not self._paused

    def start(self):
        """Start the monitoring loop in a background thread."""
        self._running = True
        self._paused = False
        if self._thread is None or not self._thread.is_alive():
            self._thread = threading.Thread(target=self._loop, daemon=True)
            self._thread.start()
        print(f"[Monitor] Heartbeat started (every {config.HEARTBEAT_INTERVAL}s)")

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)

    def pause(self):
        """Pause monitoring (engine keeps running, no alerts)."""
        self._paused = True
        print("[Monitor] Paused")

    def resume(self):
        """Resume monitoring."""
        self._paused = False
        if not self._running:
            self.start()
        print("[Monitor] Resumed")

    def _loop(self):
        """Main monitoring loop."""
        # Wait for engine to produce first state
        while self._running and not self.engine.get_state():
            time.sleep(1)

        print("[Monitor] Spatial engine ready. Monitoring active.")

        while self._running:
            if not self._paused:
                try:
                    self._check()
                except Exception as e:
                    print(f"[Monitor] Check error: {e}")

            time.sleep(config.HEARTBEAT_INTERVAL)

    def _check(self):
        """Single monitoring check cycle."""
        state = self.engine.get_state()
        if not state:
            return

        # ── CRITICAL: Fall detection (highest priority) ──────────────────
        for p in state.get("persons", []):
            activity = p.get("activity", "")
            if activity in ("fallen", "lying_down"):
                tid = p["track_id"]
                fall_key = f"fall_{tid}"
                if fall_key not in self._prev_risk_ids:
                    self._prev_risk_ids.add(fall_key)
                    event_text = (
                        f"CRITICAL: Person {tid} appears to have FALLEN! "
                        f"Activity={activity}, position=({p['center']['x']},{p['center']['y']}), "
                        f"speed={p['speed_px_per_sec']}px/s. "
                        "This is urgent — capture a photo and alert the user immediately."
                    )
                    print(f"[Monitor] ⚠️ FALL DETECTED: {event_text}")
                    response = self.agent.handle_event(event_text)
                    if response:
                        print(f"[Monitor] Agent response: {response[:200]}")
                    return  # prioritize fall over other events

        # ── Risk zone events ─────────────────────────────────────────────
        risk_events = state.get("risk_events", [])
        current_risk_ids = {r["event_id"] for r in risk_events}

        # Detect NEW risk events (not seen in previous check)
        new_risks = [r for r in risk_events if r["event_id"] not in self._prev_risk_ids]
        self._prev_risk_ids = (self._prev_risk_ids & current_risk_ids) | current_risk_ids

        if new_risks:
            descriptions = []
            for r in new_risks:
                descriptions.append(
                    f"- {r['description']} (risk_score={r['risk_score']}, ttc={r['ttc']}s)"
                )
            event_text = f"{len(new_risks)} new risk event(s):\n" + "\n".join(descriptions)
            print(f"[Monitor] {event_text}")

            response = self.agent.handle_event(event_text)
            if response:
                print(f"[Monitor] Agent response: {response[:200]}")
