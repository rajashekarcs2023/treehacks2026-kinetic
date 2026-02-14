"""
Spatial Engine — Runs CV pipeline in a background thread,
produces structured spatial state continuously.

Architecture:
  Camera → YOLO + Pose + Tracking + Motion + Risk → Structured JSON state
  Frames are processed and discarded (privacy-first, no video storage).
  Only structured spatial data + event-triggered snapshots are kept.
"""

import threading
import time
import json
import os
import queue
import cv2
import numpy as np

from src.perception import PerceptionLayer
from src.motion import MotionModeler
from src.risk import RiskEstimator
from src.models import DangerZone, BBox

from aegis import config
from aegis.activity import classify_all


class SpatialEngine:
    """Runs CV pipeline in background, maintains live spatial state."""

    def __init__(self, show_camera: bool = False, source: str = "camera"):
        """
        Args:
            show_camera: Show live camera feed with overlays (macOS main thread)
            source: 'camera' for local webcam, 'external' for phone WebSocket frames
        """
        self.show_camera = show_camera
        self._source = source
        self._lock = threading.Lock()

        self._running = False
        self._thread = None

        # Latest state (dict, serializable to JSON)
        self._state: dict = {}
        self._frame: np.ndarray | None = None
        self._frame_count = 0

        # External frame input queue (for phone WebSocket)
        self._frame_queue: queue.Queue = queue.Queue(maxsize=2)

        # Event history
        self._events: list[dict] = []

        # Danger zones (configurable at runtime)
        self._danger_zones: list[DangerZone] = []

        # Callbacks for event notifications
        self._event_callbacks: list = []

        # Raw tracked persons (for MCP pose tools to access landmarks)
        self._tracked_persons: list = []

    def push_frame(self, frame: np.ndarray):
        """Push a frame from external source (e.g., phone WebSocket).
        Non-blocking — drops frame if queue is full (keeps latest)."""
        if self._source != "external":
            return
        try:
            # Clear old frames, keep only latest
            while not self._frame_queue.empty():
                try:
                    self._frame_queue.get_nowait()
                except queue.Empty:
                    break
            self._frame_queue.put_nowait(frame)
        except queue.Full:
            pass

    # ── Public API ───────────────────────────────────────────────────────

    def add_danger_zone(self, x1: int, y1: int, x2: int, y2: int,
                        label: str = "RESTRICTED ZONE") -> str:
        """Add a danger zone. Returns the zone_id."""
        zone_id = f"zone_{len(self._danger_zones)}"
        self._danger_zones.append(DangerZone(
            zone_id=zone_id,
            bbox=BBox(x1, y1, x2, y2),
            label=label,
        ))
        return zone_id

    def clear_danger_zones(self):
        """Remove all danger zones."""
        self._danger_zones.clear()

    def on_event(self, callback):
        """Register a callback for risk events: callback(event_dict)."""
        self._event_callbacks.append(callback)

    def get_state(self) -> dict:
        """Get current spatial state as a dict."""
        with self._lock:
            return self._state.copy()

    def get_state_json(self) -> str:
        """Get current spatial state as formatted JSON string."""
        return json.dumps(self.get_state(), indent=2)

    def get_frame(self) -> np.ndarray | None:
        """Get a copy of the latest camera frame (for snapshots)."""
        with self._lock:
            return self._frame.copy() if self._frame is not None else None

    def get_events(self, last_n: int = 50) -> list[dict]:
        """Get recent risk events."""
        with self._lock:
            return list(self._events[-last_n:])

    def capture_snapshot(self) -> str | None:
        """Save current frame to disk. Returns file path or None."""
        frame = self.get_frame()
        if frame is None:
            return None
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"snapshot_{timestamp}.jpg"
        filepath = os.path.join(config.SNAPSHOTS_DIR, filename)
        cv2.imwrite(filepath, frame)
        return filepath

    def get_summary(self) -> str:
        """Human-readable summary of current spatial state."""
        state = self.get_state()
        if not state:
            return "No spatial data available yet. Engine may still be starting."

        lines = []
        fps = state.get("fps", 0)
        lines.append(f"Frame #{state.get('frame_number', '?')} | {fps:.0f} FPS")

        persons = state.get("persons", [])
        if not persons:
            lines.append("No people detected.")
        else:
            lines.append(f"{len(persons)} person(s):")
            for p in persons:
                speed = p.get("speed_px_per_sec", 0)
                tid = p["track_id"]
                cx, cy = p["center"]["x"], p["center"]["y"]
                activity = p.get("activity", "unknown")
                line = f"  Person {tid}: pos=({cx},{cy}), speed={speed:.0f}px/s, {activity}"
                if p.get("predicted_position"):
                    pp = p["predicted_position"]
                    line += f", predicted=({pp['x']},{pp['y']})"
                lines.append(line)

        objects = state.get("objects", [])
        if objects:
            obj_summary = {}
            for o in objects:
                name = o["class_name"]
                obj_summary[name] = obj_summary.get(name, 0) + 1
            obj_str = ", ".join(f"{v}x {k}" for k, v in obj_summary.items())
            lines.append(f"Objects: {obj_str}")

        zones = state.get("danger_zones", [])
        if zones:
            lines.append(f"{len(zones)} danger zone(s) active.")

        risks = state.get("risk_events", [])
        if risks:
            lines.append(f"\u26a0\ufe0f {len(risks)} active risk(s):")
            for r in risks:
                lines.append(f"  - {r['description']}")

        return "\n".join(lines)

    def get_display_frame(self) -> np.ndarray | None:
        """Get a frame with overlays drawn (for main-thread display)."""
        with self._lock:
            if self._frame is None:
                return None
            frame = self._frame.copy()
            state = self._state.copy()

        # Draw person overlays
        for p in state.get("persons", []):
            b = p["bbox"]
            activity = p.get("activity", "")
            color = (0, 255, 0)
            if activity in ("fallen", "lying_down"):
                color = (0, 0, 255)  # red for fallen
            elif activity == "running":
                color = (0, 165, 255)  # orange for running
            cv2.rectangle(frame, (b["x1"], b["y1"]), (b["x2"], b["y2"]), color, 2)
            label = f"ID:{p['track_id']} {activity}"
            cv2.putText(frame, label, (b["x1"], b["y1"] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # Draw object overlays
        for obj in state.get("objects", []):
            b = obj["bbox"]
            cv2.rectangle(frame, (b["x1"], b["y1"]), (b["x2"], b["y2"]), (255, 200, 0), 1)
            cv2.putText(frame, obj["class_name"], (b["x1"], b["y1"] - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 200, 0), 1)

        # Draw danger zones
        for z in state.get("danger_zones", []):
            b = z["bbox"]
            cv2.rectangle(frame, (b["x1"], b["y1"]), (b["x2"], b["y2"]), (0, 0, 255), 2)
            cv2.putText(frame, z["label"], (b["x1"], b["y1"] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        # Draw risk warnings
        for i, r in enumerate(state.get("risk_events", [])):
            cv2.putText(frame, f"!! {r['description']}", (10, 70 + i * 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        # HUD
        fps = state.get("fps", 0)
        n_persons = len(state.get("persons", []))
        n_objects = len(state.get("objects", []))
        cv2.putText(frame, f"AEGIS | {fps:.0f} FPS | {n_persons} person(s) | {n_objects} object(s)",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        return frame

    @property
    def is_running(self) -> bool:
        return self._running

    # ── Lifecycle ────────────────────────────────────────────────────────

    def start(self):
        """Start the spatial engine in a background thread."""
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()

    def stop(self):
        """Stop the engine gracefully."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)

    # ── Internal processing loop ─────────────────────────────────────────

    def _run_loop(self):
        """Main processing loop (runs in background thread)."""
        # Initialize CV components
        perception = PerceptionLayer(
            enable_pose=config.ENABLE_POSE,
            enable_depth=config.ENABLE_DEPTH,
        )
        motion = MotionModeler(prediction_horizon=config.PREDICTION_HORIZON)
        risk_estimator = RiskEstimator(ttc_threshold=config.TTC_THRESHOLD)

        cap = None
        width, height = 640, 480  # default for external source

        if self._source == "camera":
            cap = cv2.VideoCapture(config.CAMERA_INDEX)
            if not cap.isOpened():
                print("[SpatialEngine] ERROR: Cannot open camera")
                self._running = False
                return
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            print(f"[SpatialEngine] Camera opened: {width}x{height}")
        else:
            print(f"[SpatialEngine] External source mode — waiting for frames")

        # Add default danger zone if none configured
        if not self._danger_zones:
            self.add_danger_zone(
                int(width * 0.65), int(height * 0.15),
                int(width * 0.95), int(height * 0.85),
                "RESTRICTED ZONE"
            )

        prev_time = time.time()
        fps = 0.0

        try:
            while self._running:
                if self._source == "camera":
                    ret, frame = cap.read()
                    if not ret:
                        time.sleep(0.01)
                        continue
                else:
                    try:
                        frame = self._frame_queue.get(timeout=0.5)
                        height, width = frame.shape[:2]
                    except queue.Empty:
                        continue

                now = time.time()
                dt = now - prev_time
                prev_time = now
                instant_fps = 1.0 / dt if dt > 0 else 0
                fps = 0.9 * fps + 0.1 * instant_fps
                self._frame_count += 1

                # ── Run CV pipeline ──────────────────────────────────────
                persons, pose_landmarks, depth_map, objects = perception.process(frame, now)
                persons = motion.update(persons)
                persons = classify_all(persons)
                risk_events = risk_estimator.assess(persons, self._danger_zones)

                # ── Build structured state ───────────────────────────────
                state = self._build_state(persons, risk_events, objects, width, height, fps, now)

                # ── Record risk events ───────────────────────────────────
                for r in risk_events:
                    event_dict = self._risk_to_dict(r)
                    with self._lock:
                        self._events.append(event_dict)
                        if len(self._events) > config.MAX_EVENT_HISTORY:
                            self._events = self._events[-config.MAX_EVENT_HISTORY:]
                    # Notify callbacks
                    for cb in self._event_callbacks:
                        try:
                            cb(event_dict)
                        except Exception as e:
                            print(f"[SpatialEngine] Event callback error: {e}")

                # ── Update shared state ──────────────────────────────────
                with self._lock:
                    self._state = state
                    self._frame = frame  # keep reference (no copy for speed)
                    self._tracked_persons = list(persons)  # raw TrackedPerson objects

        finally:
            if cap is not None:
                cap.release()
            perception.close()
            print("[SpatialEngine] Stopped")

    # ── Serialization helpers ────────────────────────────────────────────

    def _build_state(self, persons, risk_events, objects, width, height, fps, now) -> dict:
        state = {
            "timestamp": now,
            "frame_number": self._frame_count,
            "fps": round(fps, 1),
            "frame_size": {"width": width, "height": height},
            "persons": [],
            "objects": [],
            "danger_zones": [],
            "risk_events": [],
        }

        for p in persons:
            pd = {
                "track_id": p.track_id,
                "bbox": {"x1": p.bbox.x1, "y1": p.bbox.y1,
                         "x2": p.bbox.x2, "y2": p.bbox.y2},
                "center": {"x": p.bbox.cx, "y": p.bbox.cy},
                "confidence": round(p.confidence, 2),
                "velocity": {"vx": round(p.vx, 1), "vy": round(p.vy, 1)},
                "speed_px_per_sec": round(p.speed, 1),
                "direction_deg": round(p.direction_deg, 1),
                "predicted_position": None,
                "has_pose": p.pose is not None,
                "activity": p.activity,
                "activity_confidence": round(p.activity_confidence, 2),
                "depth_estimate": round(p.depth_estimate, 3) if p.depth_estimate else None,
            }
            if p.predicted_x is not None:
                pd["predicted_position"] = {"x": p.predicted_x, "y": p.predicted_y}
            state["persons"].append(pd)

        for obj in objects:
            state["objects"].append({
                "object_id": obj.object_id,
                "class_name": obj.class_name,
                "bbox": {"x1": obj.bbox.x1, "y1": obj.bbox.y1,
                         "x2": obj.bbox.x2, "y2": obj.bbox.y2},
                "center": {"x": obj.bbox.cx, "y": obj.bbox.cy},
                "confidence": round(obj.confidence, 2),
            })

        for z in self._danger_zones:
            state["danger_zones"].append({
                "zone_id": z.zone_id,
                "bbox": {"x1": z.bbox.x1, "y1": z.bbox.y1,
                         "x2": z.bbox.x2, "y2": z.bbox.y2},
                "label": z.label,
            })

        for r in risk_events:
            state["risk_events"].append(self._risk_to_dict(r))

        return state

    @staticmethod
    def _risk_to_dict(r) -> dict:
        return {
            "event_id": r.event_id,
            "person_id": r.person_id,
            "zone_id": r.zone_id,
            "ttc": round(r.ttc, 2),
            "risk_score": round(r.risk_score, 2),
            "timestamp": r.timestamp,
            "description": r.description,
        }
