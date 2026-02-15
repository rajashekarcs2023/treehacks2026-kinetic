"""
DGX Spark Client — Sends frames to DGX for RTMPose-WholeBody inference.

If DGX is reachable: returns 133 keypoints (body + hands + face)
If DGX is unreachable: returns None (caller falls back to local MediaPipe)

Usage:
    client = DGXClient("http://gx10-eb94:8080")
    result = await client.predict(frame)  # numpy BGR image
    if result:
        print(result["body"])   # 17 body keypoints
        print(result["hands"])  # 42 hand keypoints (21 per hand)
        print(result["face"])   # 68 face keypoints
"""

import asyncio
import time
import io
from typing import Optional

import cv2
import numpy as np

try:
    import httpx
    HAS_HTTPX = True
except ImportError:
    HAS_HTTPX = False


class DGXClient:
    """Async client for DGX Spark inference server."""

    def __init__(self, base_url: str = "http://gx10-eb94:8080", timeout: float = 2.0):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._available = False
        self._last_check = 0
        self._check_interval = 30  # re-check availability every 30s
        self._client: Optional[httpx.AsyncClient] = None
        self._total_requests = 0
        self._total_ms = 0.0
        self._errors = 0

    async def _get_client(self) -> Optional[httpx.AsyncClient]:
        if not HAS_HTTPX:
            return None
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=self.timeout)
        return self._client

    async def check_health(self) -> bool:
        """Check if DGX server is reachable."""
        try:
            client = await self._get_client()
            if client is None:
                return False
            resp = await client.get(f"{self.base_url}/health")
            if resp.status_code == 200:
                data = resp.json()
                self._available = True
                self._last_check = time.time()
                print(f"[DGX] Connected — GPU: {data.get('gpu')}, "
                      f"model: {data.get('pose_model_loaded')}")
                return True
        except Exception as e:
            print(f"[DGX] Not reachable: {e}")
        self._available = False
        self._last_check = time.time()
        return False

    @property
    def is_available(self) -> bool:
        """Whether DGX was reachable at last check."""
        # Re-check if stale
        if time.time() - self._last_check > self._check_interval:
            return False  # Will trigger re-check on next predict
        return self._available

    async def predict(self, frame: np.ndarray) -> Optional[dict]:
        """Send a frame to DGX and get 133 whole-body keypoints.

        Args:
            frame: BGR numpy image from cv2

        Returns:
            dict with body, hands, face, feet keypoints — or None if unavailable
        """
        # Periodic availability check
        if not self._available and time.time() - self._last_check > self._check_interval:
            await self.check_health()

        if not self._available:
            return None

        try:
            client = await self._get_client()
            if client is None:
                return None

            t0 = time.time()

            # Encode frame as JPEG for efficient transfer
            _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            img_bytes = buf.tobytes()

            resp = await client.post(
                f"{self.base_url}/predict",
                files={"file": ("frame.jpg", img_bytes, "image/jpeg")},
            )

            if resp.status_code == 200:
                result = resp.json()
                elapsed_ms = (time.time() - t0) * 1000
                self._total_requests += 1
                self._total_ms += elapsed_ms

                result["round_trip_ms"] = round(elapsed_ms, 1)
                return result
            else:
                self._errors += 1
                return None

        except Exception:
            self._errors += 1
            # Mark unavailable so we don't spam failed requests
            if self._errors > 3:
                self._available = False
                self._errors = 0
            return None

    def get_stats(self) -> dict:
        """Get client statistics."""
        return {
            "dgx_url": self.base_url,
            "available": self._available,
            "total_requests": self._total_requests,
            "avg_round_trip_ms": round(self._total_ms / max(self._total_requests, 1), 1),
            "errors": self._errors,
        }

    async def close(self):
        """Close the HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None
