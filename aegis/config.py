"""
AEGIS Configuration
All settings in one place. Override via environment variables or .env file.
"""

import os
from dotenv import load_dotenv

# Load .env file from project root
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env"))

# ── Paths ────────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
SNAPSHOTS_DIR = os.path.join(DATA_DIR, "snapshots")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")

# Ensure directories exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(SNAPSHOTS_DIR, exist_ok=True)

# ── Camera ───────────────────────────────────────────────────────────────
CAMERA_INDEX = int(os.environ.get("AEGIS_CAMERA", "0"))

# ── CV Pipeline ──────────────────────────────────────────────────────────
ENABLE_POSE = True
ENABLE_DEPTH = False
PREDICTION_HORIZON = 2.0  # seconds
TTC_THRESHOLD = 2.0  # seconds

# ── Claude Agent ─────────────────────────────────────────────────────────
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
AGENT_MODEL = os.environ.get("AEGIS_MODEL", "claude-sonnet-4-20250514")

# ── Telegram ─────────────────────────────────────────────────────────────
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "")

# ── Gemini Live ─────────────────────────────────────────────────────────
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", os.environ.get("GOOGLE_API_KEY", ""))
GEMINI_MODEL = "gemini-2.5-flash-preview-native-audio-dialog"
GEMINI_VOICE = "Kore"  # Options: Kore, Charon, Fenrir, Aoede, Puck, etc.

# ── Server ──────────────────────────────────────────────────────────────
SERVER_HOST = os.environ.get("AEGIS_HOST", "0.0.0.0")
SERVER_PORT = int(os.environ.get("AEGIS_PORT", "8000"))

# ── Monitoring ───────────────────────────────────────────────────────────
HEARTBEAT_INTERVAL = 5.0  # seconds between proactive agent checks
EVENT_COOLDOWN = 15.0  # min seconds between alerts for same event type
MAX_EVENT_HISTORY = 500
