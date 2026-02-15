"""
Layer 1: Multimodal Perception
================================
Converts raw webcam frames into structured object state.
- YOLO11n for person detection + ByteTrack for tracking
- MediaPipe PoseLandmarker for skeleton overlay (async)
- Depth Anything V2 for monocular depth (optional)

Input:  Raw BGR frame
Output: List of TrackedPerson objects
"""

import os
import time
import numpy as np
import cv2
import mediapipe as mp
from mediapipe.tasks.python.vision import PoseLandmarker, PoseLandmarkerOptions
from mediapipe.tasks.python.vision import RunningMode
from mediapipe.tasks.python import BaseOptions
from ultralytics import YOLO

from src.models import TrackedPerson, BBox, PoseLandmarks, DetectedObject

# COCO class names for YOLO
COCO_CLASSES = {
    0: "person", 1: "bicycle", 2: "car", 3: "motorcycle", 4: "airplane",
    5: "bus", 6: "train", 7: "truck", 8: "boat", 9: "traffic light",
    10: "fire hydrant", 11: "stop sign", 12: "parking meter", 13: "bench",
    14: "bird", 15: "cat", 16: "dog", 17: "horse", 18: "sheep", 19: "cow",
    20: "elephant", 21: "bear", 22: "zebra", 23: "giraffe", 24: "backpack",
    25: "umbrella", 26: "handbag", 27: "tie", 28: "suitcase", 29: "frisbee",
    30: "skis", 31: "snowboard", 32: "sports ball", 33: "kite", 34: "baseball bat",
    35: "baseball glove", 36: "skateboard", 37: "surfboard", 38: "tennis racket",
    39: "bottle", 40: "wine glass", 41: "cup", 42: "fork", 43: "knife",
    44: "spoon", 45: "bowl", 46: "banana", 47: "apple", 48: "sandwich",
    49: "orange", 50: "broccoli", 51: "carrot", 52: "hot dog", 53: "pizza",
    54: "donut", 55: "cake", 56: "chair", 57: "couch", 58: "potted plant",
    59: "bed", 60: "dining table", 61: "toilet", 62: "tv", 63: "laptop",
    64: "mouse", 65: "remote", 66: "keyboard", 67: "cell phone", 68: "microwave",
    69: "oven", 70: "toaster", 71: "sink", 72: "refrigerator", 73: "book",
    74: "clock", 75: "vase", 76: "scissors", 77: "teddy bear", 78: "hair drier",
    79: "toothbrush",
}


class PersonDetector:
    """YOLO11n person detection + ByteTrack tracking."""

    def __init__(self, model_path: str = "yolo11n.pt", conf_threshold: float = 0.4):
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold

    def detect_and_track(self, frame: np.ndarray, timestamp: float) -> list[TrackedPerson]:
        """Run detection + tracking, return list of TrackedPerson."""
        results = self.model.track(
            frame, persist=True, verbose=False,
            classes=[0], conf=self.conf_threshold
        )

        persons = []
        boxes = results[0].boxes
        if boxes.id is None:
            return persons

        for i in range(len(boxes)):
            track_id = int(boxes.id[i])
            x1, y1, x2, y2 = map(int, boxes.xyxy[i])
            conf = float(boxes.conf[i])

            person = TrackedPerson(
                track_id=track_id,
                bbox=BBox(x1, y1, x2, y2),
                confidence=conf,
                timestamp=timestamp,
            )
            persons.append(person)

        return persons


class ObjectDetector:
    """YOLO11n object detection for non-person objects."""

    def __init__(self, model=None, model_path: str = "yolo11n.pt", conf_threshold: float = 0.25):
        self.model = model if model is not None else YOLO(model_path)
        self.conf_threshold = conf_threshold
        self._next_id = 0

    def detect(self, frame: np.ndarray) -> list[DetectedObject]:
        """Detect non-person objects. Returns list of DetectedObject."""
        results = self.model(frame, verbose=False, conf=self.conf_threshold)
        objects = []
        boxes = results[0].boxes
        if boxes is None or len(boxes) == 0:
            return objects

        for i in range(len(boxes)):
            cls_id = int(boxes.cls[i])
            if cls_id == 0:  # skip person — handled by PersonDetector
                continue
            x1, y1, x2, y2 = map(int, boxes.xyxy[i])
            conf = float(boxes.conf[i])
            class_name = COCO_CLASSES.get(cls_id, f"class_{cls_id}")

            obj = DetectedObject(
                object_id=self._next_id,
                class_id=cls_id,
                class_name=class_name,
                bbox=BBox(x1, y1, x2, y2),
                confidence=conf,
            )
            objects.append(obj)
            self._next_id += 1

        return objects


class PoseEstimator:
    """MediaPipe PoseLandmarker in async LIVE_STREAM mode."""

    POSE_CONNECTIONS = [
        (0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8),
        (9, 10), (11, 12), (11, 13), (13, 15), (15, 17), (15, 19), (15, 21),
        (17, 19), (12, 14), (14, 16), (16, 18), (16, 20), (16, 22), (18, 20),
        (11, 23), (12, 24), (23, 24), (23, 25), (24, 26), (25, 27), (26, 28),
        (27, 29), (28, 30), (29, 31), (30, 32), (27, 31), (28, 32),
    ]

    # Key landmark indices
    NOSE = 0
    LEFT_HIP = 23
    RIGHT_HIP = 24

    def __init__(self, model_path: str = None):
        if model_path is None:
            # Prefer Full model (more accurate), fall back to Lite
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            full_path = os.path.join(base_dir, "models", "pose_landmarker_full.task")
            lite_path = os.path.join(base_dir, "models", "pose_landmarker_lite.task")
            if os.path.exists(full_path):
                model_path = full_path
                print("[Pose] Using Full model (higher accuracy)")
            else:
                model_path = lite_path
                print("[Pose] Using Lite model")

        self._latest_result = None
        self._start_time = time.time()

        # Temporal smoothing state (EMA filter per landmark)
        self._smooth_alpha = 0.4  # 0 = full smoothing, 1 = no smoothing
        self._prev_landmarks: dict[int, list[tuple[float, float, float]]] = {}  # pose_idx -> landmarks

        options = PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=RunningMode.LIVE_STREAM,
            num_poses=3,
            min_pose_detection_confidence=0.4,
            min_tracking_confidence=0.6,
            result_callback=self._on_result,
        )
        self._landmarker = PoseLandmarker.create_from_options(options)

    def _on_result(self, result, output_image, timestamp_ms):
        self._latest_result = result

    def process_async(self, frame: np.ndarray, timestamp: float):
        """Submit frame for async pose estimation."""
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        timestamp_ms = int((timestamp - self._start_time) * 1000)
        try:
            self._landmarker.detect_async(mp_image, timestamp_ms)
        except Exception:
            pass  # skip frames that arrive out of order

    def get_landmarks(self, width: int, height: int) -> list[PoseLandmarks]:
        """Get the latest pose landmarks as PoseLandmarks objects.
        Applies EMA temporal smoothing to reduce jitter."""
        if self._latest_result is None or not self._latest_result.pose_landmarks:
            return []

        results = []
        for pose_idx, pose_lms in enumerate(self._latest_result.pose_landmarks):
            raw_points = []
            for lm in pose_lms:
                px = lm.x * width
                py = lm.y * height
                vis = lm.visibility if hasattr(lm, 'visibility') else 1.0
                raw_points.append((px, py, vis))

            # Apply EMA temporal smoothing
            if pose_idx in self._prev_landmarks and len(self._prev_landmarks[pose_idx]) == len(raw_points):
                prev = self._prev_landmarks[pose_idx]
                smoothed = []
                alpha = self._smooth_alpha
                for i, (rx, ry, rv) in enumerate(raw_points):
                    px_prev, py_prev, _ = prev[i]
                    # Only smooth if visibility is decent (>0.3)
                    if rv > 0.3:
                        sx = alpha * rx + (1 - alpha) * px_prev
                        sy = alpha * ry + (1 - alpha) * py_prev
                    else:
                        sx, sy = rx, ry
                    smoothed.append((sx, sy, rv))
                raw_points = smoothed
            self._prev_landmarks[pose_idx] = raw_points

            points = [(int(x), int(y), v) for x, y, v in raw_points]

            # Compute hip midpoint
            hip_mid = None
            if (self.LEFT_HIP < len(points) and self.RIGHT_HIP < len(points)
                    and points[self.LEFT_HIP][2] > 0.5 and points[self.RIGHT_HIP][2] > 0.5):
                hip_mid = (
                    (points[self.LEFT_HIP][0] + points[self.RIGHT_HIP][0]) // 2,
                    (points[self.LEFT_HIP][1] + points[self.RIGHT_HIP][1]) // 2,
                )

            # Nose
            nose = None
            if self.NOSE < len(points) and points[self.NOSE][2] > 0.5:
                nose = (points[self.NOSE][0], points[self.NOSE][1])

            results.append(PoseLandmarks(points=points, hip_midpoint=hip_mid, nose=nose))

        return results

    def close(self):
        self._landmarker.close()


class DepthEstimator:
    """Monocular depth estimation using Depth Anything V2."""

    def __init__(self, model_name: str = "depth-anything/Depth-Anything-V2-Small-hf",
                 input_size: tuple = (320, 240), every_n_frames: int = 3):
        import torch
        from transformers import pipeline

        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        self._pipe = pipeline(
            task="depth-estimation",
            model=model_name,
            device=self.device,
        )
        self._input_size = input_size
        self._every_n = every_n_frames
        self._frame_count = 0
        self._last_depth = None

    def estimate(self, frame: np.ndarray) -> np.ndarray | None:
        """Return depth map (H x W float32, normalized 0-1) or None if skipping this frame."""
        self._frame_count += 1
        if self._frame_count % self._every_n != 0 and self._last_depth is not None:
            return self._last_depth

        from PIL import Image

        small = cv2.resize(frame, self._input_size)
        rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)

        result = self._pipe(pil_img)
        depth_map = np.array(result["depth"], dtype=np.float32)

        # Normalize to 0-1
        dmin, dmax = depth_map.min(), depth_map.max()
        if dmax - dmin > 0:
            depth_map = (depth_map - dmin) / (dmax - dmin)
        else:
            depth_map = np.zeros_like(depth_map)

        # Resize back to original frame size
        h, w = frame.shape[:2]
        self._last_depth = cv2.resize(depth_map, (w, h))
        return self._last_depth


class PerceptionLayer:
    """Orchestrates all perception components."""

    def __init__(self, enable_pose: bool = True, enable_depth: bool = False,
                 enable_objects: bool = True):
        self.detector = PersonDetector()
        self.object_detector = ObjectDetector(model=self.detector.model) if enable_objects else None
        self.pose_estimator = PoseEstimator() if enable_pose else None
        self.depth_estimator = DepthEstimator() if enable_depth else None
        self._obj_frame_count = 0
        self._obj_every_n = 5  # detect objects every 5 frames to save FPS
        self._last_objects: list[DetectedObject] = []

    def process(self, frame: np.ndarray, timestamp: float) -> tuple[list[TrackedPerson], list[PoseLandmarks], np.ndarray | None, list[DetectedObject]]:
        """
        Process a single frame through all perception components.
        Returns: (persons, pose_landmarks, depth_map, objects)
        """
        # Detection + tracking
        persons = self.detector.detect_and_track(frame, timestamp)

        # Pose estimation (async — results lag by ~1 frame, which is fine)
        pose_landmarks = []
        if self.pose_estimator:
            self.pose_estimator.process_async(frame, timestamp)
            h, w = frame.shape[:2]
            pose_landmarks = self.pose_estimator.get_landmarks(w, h)

            # Attach pose to nearest person by centroid distance
            for pose_lm in pose_landmarks:
                if pose_lm.hip_midpoint:
                    best_person = None
                    best_dist = float('inf')
                    for person in persons:
                        dx = person.bbox.cx - pose_lm.hip_midpoint[0]
                        dy = person.bbox.cy - pose_lm.hip_midpoint[1]
                        dist = (dx ** 2 + dy ** 2) ** 0.5
                        if dist < best_dist:
                            best_dist = dist
                            best_person = person
                    if best_person and best_dist < 200:
                        best_person.pose = pose_lm

        # Depth estimation (optional)
        depth_map = None
        if self.depth_estimator:
            depth_map = self.depth_estimator.estimate(frame)

            # Assign approximate depth to each person
            if depth_map is not None:
                for person in persons:
                    cx, cy = person.bbox.cx, person.bbox.cy
                    cy_clamped = max(0, min(cy, depth_map.shape[0] - 1))
                    cx_clamped = max(0, min(cx, depth_map.shape[1] - 1))
                    person.depth_estimate = float(depth_map[cy_clamped, cx_clamped])

        # Object detection (every N frames to preserve FPS)
        objects = self._last_objects
        if self.object_detector:
            self._obj_frame_count += 1
            if self._obj_frame_count % self._obj_every_n == 0:
                objects = self.object_detector.detect(frame)
                self._last_objects = objects

        return persons, pose_landmarks, depth_map, objects

    def close(self):
        if self.pose_estimator:
            self.pose_estimator.close()
