"""AEGIS Server Runner — Start the full AEGIS backend with web dashboard.

Usage:
    python -m aegis.run_server [--camera] [--port 8000]

Modes:
    Default: External source (device/phone sends frames via WebSocket)
    --camera: Use local webcam as source (for development/testing)
"""

import argparse
import os
import uvicorn

from aegis import config
from aegis.spatial_engine import SpatialEngine
from aegis.sdk_agent import AegisSDKAgent
from aegis.gemini_bridge import GeminiBridge
from aegis.openai_voice import OpenAIVoiceBridge
from aegis.dgx_client import DGXClient
from aegis.telegram_bot import TelegramBot
import aegis.server as server_module
from aegis.server import app
import aegis.mcp_server as mcp_module


def main():
    parser = argparse.ArgumentParser(description="AEGIS Server")
    parser.add_argument("--camera", action="store_true",
                        help="Use local webcam instead of phone/device camera")
    parser.add_argument("--no-agent", action="store_true",
                        help="Run without Claude agent (CV pipeline only)")
    parser.add_argument("--port", type=int, default=config.SERVER_PORT,
                        help=f"Server port (default {config.SERVER_PORT})")
    parser.add_argument("--host", default=config.SERVER_HOST,
                        help=f"Server host (default {config.SERVER_HOST})")
    parser.add_argument("--goal", type=str, default=None,
                        help="Set initial goal by ID (e.g. desk_watch, posture_coach)")
    parser.add_argument("--dgx", type=str, default=None,
                        help="DGX Spark URL for RTMPose inference (e.g. http://gx10-eb94:8080)")
    args = parser.parse_args()

    source = "camera" if args.camera else "external"
    print("=" * 60)
    print("  AEGIS — Goal-Directed Spatial AI")
    print("  One device. Any goal. Any space.")
    print("=" * 60)
    print(f"  Source:     {'Local camera' if args.camera else 'Device (WebSocket)'}")
    print(f"  Server:     http://{args.host}:{args.port}")
    print(f"  Dashboard:  http://{args.host}:{args.port}/dashboard")
    print(f"  Agent:      {'✗ disabled' if args.no_agent else '✓ Claude Agent SDK (3 sub-agents)'}")
    has_openai = bool(config.OPENAI_API_KEY)
    print(f"  Voice:      {'✓ OpenAI Realtime (GPT-4o)' if has_openai else '✓ Gemini Live (' + config.GEMINI_VOICE + ')' if config.GEMINI_API_KEY else '✗ no voice API key'}")
    dgx_url = args.dgx or os.environ.get("DGX_URL", "")
    print(f"  DGX:        {'✓ ' + dgx_url if dgx_url else '✗ disabled (use --dgx URL)'}")
    print(f"  MCP:        ✓ aegis (45 tools via SDK MCP server)")
    print("=" * 60)
    print()

    # Create and start spatial engine
    engine = SpatialEngine(source=source)
    engine.start()
    server_module.engine = engine

    # Create agent (unless disabled)
    agent = None
    if not args.no_agent and config.ANTHROPIC_API_KEY:
        agent = AegisSDKAgent()
        if args.goal:
            agent.set_goal_by_id(args.goal)
            print(f"[Agent] Initial goal: {agent.goal_name}")
        server_module.agent = agent
        print(f"[Agent] Ready (claude-agent-sdk + 3 sub-agents + 40 MCP tools)")
    elif args.no_agent:
        print("[Agent] Disabled (--no-agent)")
    else:
        print("[Agent] No ANTHROPIC_API_KEY — agent disabled")

    # Initialize voice bridge (OpenAI Realtime preferred, Gemini fallback)
    if config.OPENAI_API_KEY:
        openai_bridge = OpenAIVoiceBridge(api_key=config.OPENAI_API_KEY)
        server_module.openai_voice = openai_bridge
        server_module.gemini_bridge = None  # Disable Gemini when OpenAI available
        print(f"[Voice] OpenAI Realtime bridge ready (GPT-4o, alloy voice)")
    elif config.GEMINI_API_KEY:
        bridge = GeminiBridge()
        if agent:
            bridge.update_goal(agent.goal_name, agent.active_goal.system_supplement)
        server_module.gemini_bridge = bridge
        server_module.openai_voice = None
        print(f"[Voice] Gemini Live bridge ready ({config.GEMINI_VOICE})")
    else:
        server_module.openai_voice = None
        server_module.gemini_bridge = None
        print("[Voice] No voice API key — voice disabled")

    # Initialize DGX Spark client (RTMPose-WholeBody on GPU)
    dgx_client = None
    if dgx_url:
        dgx_client = DGXClient(base_url=dgx_url)
        server_module.dgx_client = dgx_client
        print(f"[DGX] Client configured for {dgx_url}")
        print(f"[DGX] Will check connectivity on first request")

    # Initialize Telegram bot for alerts + incoming commands
    tg_bot = TelegramBot(agent=agent)
    if tg_bot.is_configured:
        server_module.telegram_bot = tg_bot
        tg_bot.start_polling()  # Listen for /status, /goals, /photo commands
        print(f"[Telegram] Bot configured — sending alerts + receiving commands")
    else:
        print(f"[Telegram] Not configured (set TELEGRAM_BOT_TOKEN + TELEGRAM_CHAT_ID)")

    # Initialize MCP server with shared state
    mcp_module.init(engine=engine, telegram_sender=tg_bot if tg_bot.is_configured else None, agent=agent, dgx_client=dgx_client)
    print(f"[MCP] aegis-spatial initialized (45 tools)")

    print(f"[Server] Starting on {args.host}:{args.port}")
    print()

    try:
        uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
    except KeyboardInterrupt:
        pass
    finally:
        engine.stop()
        print("\nAEGIS server stopped.")


if __name__ == "__main__":
    main()
