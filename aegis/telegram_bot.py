"""
AEGIS Telegram Bot — User interface via Telegram messaging.

Users text the bot with goals and questions.
The bot forwards messages to the AegisAgent and relays responses back.
Also provides a send function for proactive agent alerts.
"""

import threading
import time
import os
import requests

from aegis import config


class TelegramBot:
    """Simple Telegram bot using raw HTTP API (no heavy dependencies)."""

    def __init__(self, agent=None, monitor=None):
        self.token = config.TELEGRAM_BOT_TOKEN
        self.chat_id = config.TELEGRAM_CHAT_ID
        self.agent = agent
        self.monitor = monitor
        self._running = False
        self._thread = None
        self._base_url = f"https://api.telegram.org/bot{self.token}"
        self._last_update_id = 0

    @property
    def is_configured(self) -> bool:
        return bool(self.token and self.chat_id)

    # ── Sending messages ─────────────────────────────────────────────────

    def send_message(self, text: str, photo_path: str = None):
        """Send a message (and optional photo) to the configured chat."""
        if not self.is_configured:
            print(f"[Telegram] (not configured) {text}")
            return

        try:
            if photo_path and os.path.exists(photo_path):
                self._send_photo(text, photo_path)
            else:
                self._send_text(text)
        except Exception as e:
            print(f"[Telegram] Send error: {e}")

    def _send_text(self, text: str):
        """Send a text message."""
        # Telegram max message length is 4096
        if len(text) > 4000:
            text = text[:4000] + "..."

        url = f"{self._base_url}/sendMessage"
        payload = {
            "chat_id": self.chat_id,
            "text": text,
            "parse_mode": "Markdown",
        }
        resp = requests.post(url, json=payload, timeout=10)
        if not resp.ok:
            # Retry without markdown if parse fails
            payload["parse_mode"] = None
            requests.post(url, json=payload, timeout=10)

    def _send_photo(self, caption: str, photo_path: str):
        """Send a photo with caption."""
        url = f"{self._base_url}/sendPhoto"
        if len(caption) > 1024:
            caption = caption[:1020] + "..."

        with open(photo_path, "rb") as f:
            files = {"photo": f}
            data = {"chat_id": self.chat_id, "caption": caption}
            requests.post(url, data=data, files=files, timeout=30)

    # ── Receiving messages (polling) ─────────────────────────────────────

    def start_polling(self):
        """Start polling for incoming messages in a background thread."""
        if not self.is_configured:
            print("[Telegram] Bot not configured (set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID)")
            return
        self._running = True
        self._thread = threading.Thread(target=self._poll_loop, daemon=True)
        self._thread.start()
        print("[Telegram] Bot polling started")

    def stop_polling(self):
        """Stop the polling loop."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)

    def _poll_loop(self):
        """Long-poll for new messages from Telegram."""
        while self._running:
            try:
                url = f"{self._base_url}/getUpdates"
                params = {
                    "offset": self._last_update_id + 1,
                    "timeout": 10,
                    "allowed_updates": ["message"],
                }
                resp = requests.get(url, params=params, timeout=15)
                if not resp.ok:
                    time.sleep(5)
                    continue

                data = resp.json()
                for update in data.get("result", []):
                    self._last_update_id = update["update_id"]
                    message = update.get("message", {})
                    text = message.get("text", "")
                    chat_id = str(message.get("chat", {}).get("id", ""))

                    if not text:
                        continue

                    # Auto-detect chat ID on first message
                    if not self.chat_id:
                        self.chat_id = chat_id
                        config.TELEGRAM_CHAT_ID = chat_id
                        print(f"[Telegram] Auto-detected chat_id: {chat_id}")

                    # Only respond to the configured chat
                    if chat_id != self.chat_id:
                        continue

                    print(f"[Telegram] User: {text}")
                    self._handle_incoming(text)

            except requests.exceptions.Timeout:
                continue
            except Exception as e:
                print(f"[Telegram] Poll error: {e}")
                time.sleep(5)

    def _handle_incoming(self, text: str):
        """Handle an incoming user message."""
        if not self.agent:
            self.send_message("AEGIS agent not connected.")
            return

        # Special commands
        cmd = text.strip().lower()

        if cmd == "/status":
            summary = self.agent.engine.get_summary()
            monitoring = "🟢 Active" if (self.monitor and self.monitor.is_active) else "🔴 Paused"
            goal = self.agent.active_goal
            self.send_message(
                f"📊 *AEGIS Status*\n"
                f"Monitoring: {monitoring}\n"
                f"Goal: {goal.icon} {goal.name}\n"
                f"_{goal.description}_\n"
                f"```\n{summary}\n```"
            )
            return

        if cmd == "/goals":
            from aegis.goals import get_all_goals
            goals = get_all_goals()
            lines = ["🎯 *Available Goals*\n"]
            for g in goals:
                active = " ← active" if g.goal_id == self.agent.goal_id else ""
                lines.append(f"{g.icon} /goal\_{g.goal_id} — {g.name}{active}")
            lines.append("\n_Or just tell me what to watch for in natural language._")
            self.send_message("\n".join(lines))
            return

        if text.strip().startswith("/goal_"):
            goal_id = text.strip().replace("/goal_", "").lower()
            goal = self.agent.set_goal_by_id(goal_id)
            # Also trigger the server monitoring loop
            try:
                import aegis.server as srv
                import asyncio
                srv._monitoring_active = True
                srv._monitoring_goal = goal_id
                srv._monitoring_alerts.clear()
                if srv._monitoring_task is None or srv._monitoring_task.done():
                    loop = asyncio.get_event_loop()
                    srv._monitoring_task = loop.create_task(srv._run_monitoring_loop())
            except Exception:
                pass
            self.send_message(
                f"{goal.icon} *Goal set: {goal.name}*\n"
                f"_{goal.description}_\n\n"
                "Monitoring started! I'll send alerts here."
            )
            return

        if cmd == "/photo":
            path = self.agent.engine.capture_snapshot()
            if path:
                self.send_message("📷 Current view:", path)
            else:
                self.send_message("Could not capture photo.")
            return

        if cmd == "/start":
            if self.monitor:
                self.monitor.resume()
                self.send_message("🟢 *AEGIS Activated*\nMonitoring started. I'm watching the space.")
            else:
                self.send_message("⚠️ Monitor not available.")
            return

        if cmd == "/stop" or cmd == "/pause":
            if self.monitor:
                self.monitor.pause()
                self.send_message("🔴 *AEGIS Paused*\nMonitoring paused. I'll stop sending alerts.\nSend /start to resume.")
            else:
                self.send_message("⚠️ Monitor not available.")
            return

        if cmd == "/resume":
            if self.monitor:
                self.monitor.resume()
                self.send_message("🟢 *AEGIS Resumed*\nMonitoring active again.")
            else:
                self.send_message("⚠️ Monitor not available.")
            return

        if cmd == "/events":
            events = self.agent.engine.get_events(10)
            if not events:
                self.send_message("No events recorded yet.")
            else:
                lines = ["📋 *Recent Events*"]
                for e in events[-10:]:
                    lines.append(f"• {e['description']} (risk: {e['risk_score']})")
                self.send_message("\n".join(lines))
            return

        if cmd == "/help":
            self.send_message(
                "🤖 *AEGIS — Goal-Directed Spatial AI*\n\n"
                "*Goals:*\n"
                "/goals — List available goals\n"
                "/goal\_desk\_watch — Watch my desk\n"
                "/goal\_posture\_coach — Coach my posture\n"
                "/goal\_driver\_monitor — Driver alertness\n"
                "/goal\_study\_focus — Study focus\n"
                "/goal\_elderly\_care — Elderly care\n\n"
                "*Control:*\n"
                "/start — Activate monitoring\n"
                "/stop — Pause monitoring\n"
                "/resume — Resume monitoring\n\n"
                "*Info:*\n"
                "/status — Current state + goal\n"
                "/photo — Capture snapshot\n"
                "/events — Recent events\n\n"
                "_Or just tell me what to watch for!_"
            )
            return

        # Forward to agent
        try:
            response = self.agent.handle_user_message(text)
            if response:
                self.send_message(response)
        except Exception as e:
            print(f"[Telegram] Agent error: {e}")
            self.send_message(f"⚠️ Agent error: {str(e)[:200]}")
