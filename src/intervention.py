"""
Layer 5: Proactive Intervention
=================================
Renders visual warnings, alert banners, and overlays on the frame.
Also handles sound alerts and event logging.

Input:  Frame + SceneState (with interventions)
Output: Annotated frame with visual warnings
"""

import math
import time
import numpy as np
import cv2
from src.models import (
    SceneState, TrackedPerson, DangerZone, RiskEvent, Intervention, PoseLandmarks
)


# Pose connections for skeleton drawing
POSE_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8),
    (9, 10), (11, 12), (11, 13), (13, 15), (15, 17), (15, 19), (15, 21),
    (17, 19), (12, 14), (14, 16), (16, 18), (16, 20), (16, 22), (18, 20),
    (11, 23), (12, 24), (23, 24), (23, 25), (24, 26), (25, 27), (26, 28),
    (27, 29), (28, 30), (29, 31), (30, 32), (27, 31), (28, 32),
]

# Color palette for tracked IDs
TRACK_COLORS = [
    (0, 255, 0), (255, 165, 0), (0, 200, 255), (255, 255, 0),
    (255, 0, 255), (0, 255, 255), (128, 255, 0), (255, 128, 0),
]


class InterventionRenderer:
    """Renders all visual overlays onto frames."""

    def __init__(self):
        self._flash_state = False

    def _get_color(self, track_id: int) -> tuple:
        return TRACK_COLORS[track_id % len(TRACK_COLORS)]

    def draw_danger_zones(self, frame: np.ndarray, zones: list[DangerZone]) -> np.ndarray:
        """Draw semi-transparent danger zones."""
        for zone in zones:
            b = zone.bbox
            overlay = frame.copy()
            cv2.rectangle(overlay, (b.x1, b.y1), (b.x2, b.y2), (0, 0, 255), -1)
            cv2.addWeighted(overlay, 0.15, frame, 0.85, 0, frame)
            cv2.rectangle(frame, (b.x1, b.y1), (b.x2, b.y2), (0, 0, 255), 2)
            cv2.putText(frame, zone.label, (b.x1 + 5, b.y1 + 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        return frame

    def draw_persons(self, frame: np.ndarray, persons: list[TrackedPerson],
                     trails: dict[int, list]) -> np.ndarray:
        """Draw bounding boxes, centroids, trails, and prediction arrows."""
        for person in persons:
            b = person.bbox
            color = self._get_color(person.track_id)

            # Bounding box
            cv2.rectangle(frame, (b.x1, b.y1), (b.x2, b.y2), color, 2)
            cv2.putText(frame, f"ID:{person.track_id} ({person.confidence:.2f})",
                        (b.x1, b.y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

            # Centroid
            cv2.circle(frame, (b.cx, b.cy), 5, (0, 0, 255), -1)

            # Trail
            trail = trails.get(person.track_id, [])
            for j in range(1, len(trail)):
                _, tx, ty = trail[j]
                _, tx_prev, ty_prev = trail[j - 1]
                alpha = j / max(len(trail), 1)
                thick = max(1, int(3 * alpha))
                cv2.line(frame, (tx_prev, ty_prev), (tx, ty), color, thick)

            # Prediction arrow
            if person.predicted_x is not None and person.speed > 10:
                cv2.arrowedLine(frame, (b.cx, b.cy),
                                (person.predicted_x, person.predicted_y),
                                color, 2, tipLength=0.3)

            # Speed info
            if person.speed > 10:
                cv2.putText(frame, f"{person.speed:.0f} px/s",
                            (b.x1, b.y2 + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

        return frame

    def draw_pose(self, frame: np.ndarray, pose_landmarks: list[PoseLandmarks]) -> np.ndarray:
        """Draw pose skeletons."""
        for pose_lm in pose_landmarks:
            points = pose_lm.points
            # Connections
            for s, e in POSE_CONNECTIONS:
                if s < len(points) and e < len(points):
                    if points[s][2] > 0.3 and points[e][2] > 0.3:
                        cv2.line(frame, (points[s][0], points[s][1]),
                                 (points[e][0], points[e][1]), (0, 255, 0), 2)
            # Points
            for px, py, vis in points:
                if vis > 0.3:
                    cv2.circle(frame, (px, py), 3, (0, 0, 255), -1)

            # Hip midpoint
            if pose_lm.hip_midpoint:
                cv2.circle(frame, pose_lm.hip_midpoint, 8, (0, 255, 255), -1)

        return frame

    def draw_risk_warnings(self, frame: np.ndarray, risk_events: list[RiskEvent],
                           persons: list[TrackedPerson]) -> np.ndarray:
        """Draw per-person risk warnings."""
        person_map = {p.track_id: p for p in persons}

        for event in risk_events:
            person = person_map.get(event.person_id)
            if not person:
                continue

            b = person.bbox

            # Color based on severity
            if event.ttc == 0.0:
                warn_color = (0, 0, 255)  # red — inside zone
                label = "!! IN ZONE !!"
            elif event.ttc < 1.0:
                warn_color = (0, 0, 255)
                label = f"COLLISION {event.ttc:.1f}s"
            else:
                ratio = event.risk_score
                warn_color = (0, int(255 * (1 - ratio)), int(255 * ratio))
                label = f"COLLISION {event.ttc:.1f}s"

            cv2.putText(frame, label, (b.x1, b.y2 + 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, warn_color, 2)
            cv2.putText(frame, f"Risk: {event.risk_score:.0%}", (b.x1, b.y2 + 55),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, warn_color, 1)

            # Draw thick red prediction arrow for high-risk
            if person.predicted_x is not None and event.risk_score > 0.5:
                cv2.arrowedLine(frame, (b.cx, b.cy),
                                (person.predicted_x, person.predicted_y),
                                (0, 0, 255), 4, tipLength=0.3)

        return frame

    def draw_alert_banner(self, frame: np.ndarray, interventions: list[Intervention]) -> np.ndarray:
        """Draw global alert banner at bottom of frame."""
        if not interventions:
            return frame

        h, w = frame.shape[:2]

        # Find most urgent intervention
        worst = min(interventions, key=lambda i: i.risk_event.ttc)
        event = worst.risk_event

        # Flashing red border for critical
        now = time.time()
        if worst.action_type in ("critical_alert", "urgent_warning"):
            if int(now * 4) % 2 == 0:
                cv2.rectangle(frame, (0, 0), (w - 1, h - 1), (0, 0, 255), 8)

        # Banner background
        banner_h = 50
        banner_y = h - banner_h
        if event.ttc == 0.0:
            banner_color = (0, 0, 200)
            text = f"CRITICAL: Person {event.person_id} IN DANGER ZONE | Risk: {event.risk_score:.0%}"
        else:
            banner_color = (0, 0, int(150 + 105 * event.risk_score))
            text = f"ALERT: Collision in {event.ttc:.1f}s | Person {event.person_id} | Risk: {event.risk_score:.0%}"

        cv2.rectangle(frame, (0, banner_y), (w, h), banner_color, -1)
        cv2.putText(frame, text, (20, h - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)

        return frame

    def draw_hud(self, frame: np.ndarray, scene: SceneState, stats: dict) -> np.ndarray:
        """Draw heads-up display with stats."""
        cv2.putText(frame, f"FPS: {scene.fps:.1f}", (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(frame, f"Persons: {len(scene.persons)}", (20, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(frame, f"Interventions: {stats.get('total', 0)}", (20, 85),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        h = frame.shape[0]
        cv2.putText(frame, "'q'=quit  'z'=add zone  'c'=clear zones  'd'=toggle depth",
                    (20, h - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)

        return frame

    def render(self, frame: np.ndarray, scene: SceneState,
               trails: dict, pose_landmarks: list, stats: dict) -> np.ndarray:
        """Full render pass — call all draw methods in order."""
        frame = self.draw_danger_zones(frame, scene.danger_zones)
        frame = self.draw_pose(frame, pose_landmarks)
        frame = self.draw_persons(frame, scene.persons, trails)
        frame = self.draw_risk_warnings(frame, scene.risk_events, scene.persons)
        frame = self.draw_alert_banner(frame, scene.interventions)
        frame = self.draw_hud(frame, scene, stats)
        return frame
