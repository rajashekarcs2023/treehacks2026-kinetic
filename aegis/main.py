"""
AEGIS Main Entry Point
======================
Starts all components:
  1. SpatialEngine (CV pipeline in background thread)
  2. TelegramBot (user interface, polling in background)
  3. Monitor (proactive heartbeat loop in background)
  4. AegisAgent (Claude brain, called by bot and monitor)

Usage:
  # Full system (needs ANTHROPIC_API_KEY, TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID):
  python -m aegis.main

  # Without Telegram (agent prints to console):
  python -m aegis.main --no-telegram

  # Test spatial engine only (no agent):
  python -m aegis.main --engine-only

Environment variables:
  ANTHROPIC_API_KEY   — Required for agent
  TELEGRAM_BOT_TOKEN  — Required for Telegram bot
  TELEGRAM_CHAT_ID    — Auto-detected on first message if not set
"""

import argparse
import asyncio
import signal
import sys
import time

from aegis import config
from aegis.spatial_engine import SpatialEngine
from aegis.sdk_agent import AegisSDKAgent
from aegis.telegram_bot import TelegramBot
from aegis.monitor import Monitor
from aegis.voice import VoiceNarrator


def parse_args():
    parser = argparse.ArgumentParser(description="AEGIS — Spatial Intelligence Agent")
    parser.add_argument("--no-telegram", action="store_true",
                        help="Run without Telegram bot (console only)")
    parser.add_argument("--engine-only", action="store_true",
                        help="Run spatial engine only (no agent, no bot)")
    parser.add_argument("--no-monitor", action="store_true",
                        help="Disable proactive monitoring loop")
    parser.add_argument("--camera", type=int, default=None,
                        help="Camera index (overrides AEGIS_CAMERA env)")
    parser.add_argument("--heartbeat", type=float, default=None,
                        help="Heartbeat interval in seconds")
    parser.add_argument("--show-camera", action="store_true",
                        help="Show live camera feed with overlays")
    parser.add_argument("--voice", action="store_true",
                        help="Enable voice narration (TTS)")
    parser.add_argument("--voice-interval", type=float, default=4.0,
                        help="Seconds between voice narrations (default 4)")
    return parser.parse_args()


def main():
    args = parse_args()

    # Apply overrides
    if args.camera is not None:
        config.CAMERA_INDEX = args.camera
    if args.heartbeat is not None:
        config.HEARTBEAT_INTERVAL = args.heartbeat

    print("=" * 60)
    print("  AEGIS — Autonomous Spatial Intelligence Agent")
    print("  OpenClaw for Physical Space")
    print("=" * 60)
    print()

    # ── 1. Start Spatial Engine ──────────────────────────────────────────
    print("[1/4] Starting Spatial Engine...")
    engine = SpatialEngine(show_camera=args.show_camera)
    engine.start()

    # Wait for first frame
    print("      Waiting for camera...")
    for _ in range(50):  # 5 second timeout
        if engine.get_state():
            break
        time.sleep(0.1)

    if not engine.get_state():
        print("ERROR: Spatial engine failed to start. Check camera.")
        engine.stop()
        sys.exit(1)
    print("      ✓ Camera active, CV pipeline running")

    if args.engine_only:
        print("\n[Engine-only mode] Printing spatial state every 2 seconds. Ctrl+C to stop.\n")
        try:
            while True:
                print(engine.get_summary())
                print()
                time.sleep(2)
        except KeyboardInterrupt:
            engine.stop()
            return

    # ── 2. Initialize Agent ──────────────────────────────────────────────
    print("[2/4] Initializing Agent...")
    if not config.ANTHROPIC_API_KEY:
        print("      ⚠ ANTHROPIC_API_KEY not set. Agent will not work.")
        print("      Set it: export ANTHROPIC_API_KEY=your-key")
        engine.stop()
        sys.exit(1)

    # Telegram bot (for sending)
    telegram = TelegramBot()

    # Create send function for the agent
    def telegram_sender(message, photo_path=None):
        telegram.send_message(message, photo_path)

    agent = AegisSDKAgent()
    telegram.agent = agent
    print("      ✓ Agent ready (claude-agent-sdk + 3 sub-agents)")

    # ── 3. Start Telegram Bot ────────────────────────────────────────────
    if not args.no_telegram:
        print("[3/4] Starting Telegram Bot...")
        if telegram.is_configured:
            telegram.start_polling()
            telegram.send_message("🤖 *AEGIS Online*\nSpatial intelligence agent is active. Send /help for commands.")
            print("      ✓ Telegram bot active")
        else:
            print("      ⚠ Telegram not configured (set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID)")
            print("      Running in console mode.")
    else:
        print("[3/4] Telegram bot disabled (--no-telegram)")

    # ── 4. Start Monitor ─────────────────────────────────────────────────
    monitor = None
    if not args.no_monitor:
        print("[4/4] Starting Monitor...")
        monitor = Monitor(engine, agent)
        monitor.start()
        print("      ✓ Proactive monitoring active")
    else:
        print("[4/4] Monitor disabled (--no-monitor)")

    # Connect monitor to telegram bot for remote control
    telegram.monitor = monitor

    # ── 5. Voice Narrator (optional) ─────────────────────────────────────
    narrator = VoiceNarrator(enabled=args.voice, min_interval=args.voice_interval)
    if args.voice:
        narrator.start()

    # ── Ready ────────────────────────────────────────────────────────────
    print()
    print("=" * 60)
    print("  AEGIS is running. Components:")
    print(f"    Spatial Engine:  ✓ (camera {config.CAMERA_INDEX})")
    print(f"    Agent:           ✓ ({config.AGENT_MODEL})")
    print(f"    Telegram:        {'✓' if telegram.is_configured and not args.no_telegram else '✗'}")
    print(f"    Monitor:         {'✓' if monitor else '✗'} (every {config.HEARTBEAT_INTERVAL}s)")
    print(f"    Voice:           {'✓' if args.voice else '✗'}")
    print("=" * 60)
    print()

    if args.no_telegram or not telegram.is_configured:
        print("Console mode active. Type messages to the agent (or 'quit' to exit):")
        print()

    # ── Main thread: console input or wait ───────────────────────────────
    def shutdown():
        print("\nShutting down AEGIS...")
        narrator.stop()
        if monitor:
            monitor.stop()
        telegram.stop_polling()
        engine.stop()
        print("AEGIS stopped.")

    # Handle Ctrl+C
    def signal_handler(sig, frame):
        shutdown()
        sys.exit(0)
    signal.signal(signal.SIGINT, signal_handler)

    # Main thread loop
    if args.show_camera:
        # Camera display must run on main thread (macOS requirement)
        import cv2
        print("Camera window active. Press 'q' in the window to quit.")
        try:
            while engine.is_running:
                display = engine.get_display_frame()
                if display is not None:
                    cv2.imshow("AEGIS", display)
                # Feed voice narrator
                if args.voice:
                    narrator.narrate_state(engine.get_state())
                key = cv2.waitKey(30) & 0xFF
                if key == ord('q'):
                    break
        except KeyboardInterrupt:
            pass
        finally:
            cv2.destroyAllWindows()
    elif args.no_telegram or not telegram.is_configured:
        # Console input mode
        try:
            while True:
                try:
                    user_input = input("You: ").strip()
                except EOFError:
                    break

                if not user_input:
                    continue
                if user_input.lower() in ("quit", "exit", "q"):
                    break
                if user_input.lower() == "status":
                    print(engine.get_summary())
                    continue
                if user_input.lower() == "photo":
                    path = engine.capture_snapshot()
                    print(f"Saved: {path}" if path else "No frame available")
                    continue
                if user_input.lower() == "state":
                    print(engine.get_state_json())
                    continue

                print("Agent thinking...")
                response = asyncio.run(agent.send_message(user_input))
                print(f"AEGIS: {response}")
                print()
        except KeyboardInterrupt:
            pass
    else:
        # Telegram mode — just wait, feed narrator
        try:
            while True:
                if args.voice:
                    narrator.narrate_state(engine.get_state())
                time.sleep(1)
        except KeyboardInterrupt:
            pass

    shutdown()


if __name__ == "__main__":
    main()
