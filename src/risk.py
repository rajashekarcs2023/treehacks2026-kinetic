"""
Layer 3: Spatial Risk Estimation
=================================
Detects potential collisions and unsafe interactions
by analyzing predicted trajectories against danger zones.

Input:  List of TrackedPerson (with motion data), List of DangerZone
Output: List of RiskEvent
"""

import math
import uuid
import time
import numpy as np
from src.models import TrackedPerson, DangerZone, RiskEvent, BBox


class RiskEstimator:
    """Computes risk events based on trajectory-zone intersection."""

    def __init__(self, ttc_threshold: float = 2.0, ttc_steps: float = 0.1,
                 ttc_max: float = 5.0, min_speed: float = 5.0):
        self.ttc_threshold = ttc_threshold
        self.ttc_steps = ttc_steps
        self.ttc_max = ttc_max
        self.min_speed = min_speed

    def _point_in_bbox(self, px: float, py: float, bbox: BBox) -> bool:
        return bbox.x1 <= px <= bbox.x2 and bbox.y1 <= py <= bbox.y2

    def _compute_ttc(self, person: TrackedPerson, zone: DangerZone) -> float | None:
        """
        Estimate time-to-collision using ray marching along velocity vector.
        Returns TTC in seconds or None if no collision predicted.
        """
        if person.speed < self.min_speed:
            return None

        cx, cy = person.bbox.cx, person.bbox.cy
        vx, vy = person.vx, person.vy

        for t in np.arange(self.ttc_steps, self.ttc_max, self.ttc_steps):
            fx = cx + vx * t
            fy = cy + vy * t
            if self._point_in_bbox(fx, fy, zone.bbox):
                return float(t)

        return None

    def _compute_proximity_risk(self, person: TrackedPerson, zone: DangerZone) -> float:
        """Compute proximity-based risk score (0-1) based on distance to zone edge."""
        cx, cy = person.bbox.cx, person.bbox.cy
        zb = zone.bbox

        # Distance to nearest edge of zone
        dx = max(zb.x1 - cx, 0, cx - zb.x2)
        dy = max(zb.y1 - cy, 0, cy - zb.y2)
        dist = math.sqrt(dx ** 2 + dy ** 2)

        # Risk increases as distance decreases (within 300px)
        max_dist = 300.0
        if dist >= max_dist:
            return 0.0
        return 1.0 - (dist / max_dist)

    def assess(self, persons: list[TrackedPerson],
               danger_zones: list[DangerZone]) -> list[RiskEvent]:
        """
        Assess risk for all persons against all danger zones.
        Returns list of RiskEvent for actionable risks.
        """
        events = []
        now = time.time()

        for person in persons:
            for zone in danger_zones:
                # Check if already inside zone
                if self._point_in_bbox(person.bbox.cx, person.bbox.cy, zone.bbox):
                    event = RiskEvent(
                        event_id=str(uuid.uuid4())[:8],
                        person_id=person.track_id,
                        zone_id=zone.zone_id,
                        ttc=0.0,
                        risk_score=1.0,
                        timestamp=now,
                        description=f"Person {person.track_id} IS INSIDE {zone.label}",
                    )
                    events.append(event)
                    continue

                # Check TTC
                ttc = self._compute_ttc(person, zone)
                if ttc is not None and ttc < self.ttc_threshold:
                    risk_score = max(0.0, 1.0 - ttc / self.ttc_threshold)
                    event = RiskEvent(
                        event_id=str(uuid.uuid4())[:8],
                        person_id=person.track_id,
                        zone_id=zone.zone_id,
                        ttc=ttc,
                        risk_score=risk_score,
                        timestamp=now,
                        description=f"Person {person.track_id} → {zone.label} in {ttc:.1f}s (risk: {risk_score:.0%})",
                    )
                    events.append(event)

        return events
