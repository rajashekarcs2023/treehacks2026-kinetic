"""
Video Ingestion — Download YouTube video, extract skeleton via MediaPipe, save as reference.

Usage:
    ref = await ingest_youtube("https://youtube.com/watch?v=...", "squat_expert")
    # ref is a SkeletonSequence ready for CoachingSession
"""

import asyncio
import os
import tempfile
import time
from typing import Optional

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

from aegis.pose_comparison import (
    normalize_skeleton, detect_phases, SkeletonSequence, NormalizedSkeleton,
    ReferenceStore,
)

# Resolve model path relative to project root
_MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", "pose_landmarker_lite.task")


def _extract_youtube_id(url: str) -> Optional[str]:
    """Extract video ID from various YouTube URL formats."""
    import re
    patterns = [
        r'(?:v=|/v/|youtu\.be/)([a-zA-Z0-9_-]{11})',
        r'(?:embed/)([a-zA-Z0-9_-]{11})',
        r'(?:shorts/)([a-zA-Z0-9_-]{11})',
    ]
    for p in patterns:
        m = re.search(p, url)
        if m:
            return m.group(1)
    return None


def download_youtube_video(url: str, output_dir: Optional[str] = None) -> str:
    """Download a YouTube video and return the file path."""
    import yt_dlp

    if output_dir is None:
        output_dir = tempfile.mkdtemp(prefix="kinetic_")

    ydl_opts = {
        'format': 'best[height<=720][ext=mp4]/best[height<=720]/best',
        'outtmpl': os.path.join(output_dir, '%(id)s.%(ext)s'),
        'quiet': True,
        'no_warnings': True,
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        filename = ydl.prepare_filename(info)

    if not os.path.exists(filename):
        # Try mp4 extension
        base = os.path.splitext(filename)[0]
        for ext in ['.mp4', '.webm', '.mkv']:
            if os.path.exists(base + ext):
                filename = base + ext
                break

    return filename


def extract_skeletons_from_video(
    video_path: str,
    fps_target: float = 10.0,
    max_frames: int = 300,
    min_confidence: float = 0.5,
) -> list[NormalizedSkeleton]:
    """Extract pose skeletons from a video file using MediaPipe Tasks API."""

    if not os.path.exists(_MODEL_PATH):
        raise FileNotFoundError(f"Pose model not found: {_MODEL_PATH}")

    base_options = mp_python.BaseOptions(model_asset_path=_MODEL_PATH)
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        min_pose_detection_confidence=min_confidence,
        min_tracking_confidence=min_confidence,
    )
    landmarker = vision.PoseLandmarker.create_from_options(options)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    video_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_skip = max(1, int(video_fps / fps_target))

    skeletons = []
    frame_idx = 0
    start_time = time.time()

    while cap.isOpened() and len(skeletons) < max_frames:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_skip == 0:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            timestamp_ms = int((frame_idx / video_fps) * 1000)

            result = landmarker.detect_for_video(mp_image, timestamp_ms)

            if result.pose_landmarks and len(result.pose_landmarks) > 0:
                landmarks = result.pose_landmarks[0]
                points = []
                for lm in landmarks:
                    points.append((lm.x, lm.y, lm.visibility))

                skel = normalize_skeleton(points)
                skel.timestamp = frame_idx / video_fps
                skeletons.append(skel)

        frame_idx += 1

    cap.release()
    landmarker.close()

    duration = time.time() - start_time
    print(f"[VideoIngest] Extracted {len(skeletons)} skeletons from {frame_idx} frames in {duration:.1f}s")

    return skeletons


def create_reference_from_video(
    video_path: str,
    name: str,
    key_angle: str = "left_knee",
    fps_target: float = 10.0,
    max_frames: int = 300,
) -> SkeletonSequence:
    """Extract skeletons from video and create a SkeletonSequence reference."""
    skeletons = extract_skeletons_from_video(video_path, fps_target, max_frames)

    if len(skeletons) < 3:
        raise ValueError(f"Only {len(skeletons)} poses detected — need at least 3. "
                         "Make sure the video shows a person's full body.")

    # Detect phases
    phases = detect_phases(skeletons, key_angle) if len(skeletons) > 6 else []

    seq = SkeletonSequence(
        name=name,
        skeletons=skeletons,
        phases=phases,
        fps=fps_target,
        metadata={
            "source": "video",
            "video_path": video_path,
            "frame_count": len(skeletons),
            "key_angle": key_angle,
        },
    )

    print(f"[VideoIngest] Reference '{name}': {len(skeletons)} frames, {len(phases)} phases")
    return seq


async def ingest_youtube(
    url: str,
    name: str,
    key_angle: str = "left_knee",
    reference_store: Optional[ReferenceStore] = None,
) -> SkeletonSequence:
    """Full pipeline: YouTube URL → download → extract → reference.

    Runs CPU-heavy work in a thread pool to not block the event loop.
    """
    loop = asyncio.get_event_loop()

    # Download video
    print(f"[VideoIngest] Downloading: {url}")
    video_path = await loop.run_in_executor(None, download_youtube_video, url)
    print(f"[VideoIngest] Downloaded: {video_path}")

    # Extract skeletons
    ref = await loop.run_in_executor(
        None, create_reference_from_video, video_path, name, key_angle
    )

    # Save to store if provided
    if reference_store:
        filepath = reference_store.save(ref)
        print(f"[VideoIngest] Saved reference: {filepath}")

    # Cleanup temp video
    try:
        os.remove(video_path)
    except OSError:
        pass

    return ref
