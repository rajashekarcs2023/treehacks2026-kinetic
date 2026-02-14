# Kinetic — Real-Time AI Skill Coach with Expert Motion Transfer

> "Learn any physical skill from any expert — in real-time, through voice."

**Kinetic** watches an expert's movement, extracts their skeleton, then coaches you in real-time to match it using voice guidance and visual feedback.

## Architecture

- **Eyes**: YOLO11n + MediaPipe Pose + Depth Anything V2 + ByteTrack
- **Brain**: Claude Agent (SDK) with 43 MCP tools, subagents, hooks
- **Voice**: Gemini Live bidirectional audio
- **Learning**: PyTorch skill scorer + activity classifier, self-improving from coaching data

## Quick Start

```bash
pip install -r requirements.txt
python run.py
```

## Built at TreeHacks 2026
