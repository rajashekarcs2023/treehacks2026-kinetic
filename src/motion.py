"""
Layer 2: Motion Modeling
=========================
Computes velocity, direction, and short-horizon trajectory prediction
for each tracked person.

Input:  List of TrackedPerson (from Layer 1)
Output: Same list, enriched with velocity + predicted future position
"""

import math
import numpy as np
from collections import defaultdict
from src.models import TrackedPerson


class MotionModeler:
    """Estimates velocity and predicts future positions for tracked persons."""

    def __init__(self, history_len: int = 20, prediction_horizon: float = 2.0):
        self.history_len = history_len
        self.prediction_horizon = prediction_horizon
        # track_id -> list of (timestamp, cx, cy)
        self._trajectories: dict[int, list[tuple[float, int, int]]] = defaultdict(list)

    def update(self, persons: list[TrackedPerson]) -> list[TrackedPerson]:
        """
        Update trajectories and compute velocity + prediction for each person.
        Mutates the TrackedPerson objects in place and returns them.
        """
        for person in persons:
            tid = person.track_id
            cx, cy = person.bbox.cx, person.bbox.cy
            t = person.timestamp

            # Append to history
            self._trajectories[tid].append((t, cx, cy))
            # Trim to history length
            self._trajectories[tid] = self._trajectories[tid][-self.history_len:]

            trail = self._trajectories[tid]

            if len(trail) >= 3:
                # Weighted average velocity (more recent = higher weight)
                vx_sum, vy_sum, w_sum = 0.0, 0.0, 0.0
                for i in range(1, len(trail)):
                    t0, x0, y0 = trail[i - 1]
                    t1, x1, y1 = trail[i]
                    dt = max(1e-3, t1 - t0)
                    w = float(i)  # linear weight
                    vx_sum += w * (x1 - x0) / dt
                    vy_sum += w * (y1 - y0) / dt
                    w_sum += w

                person.vx = vx_sum / w_sum
                person.vy = vy_sum / w_sum
                person.speed = math.sqrt(person.vx ** 2 + person.vy ** 2)
                person.direction_deg = math.degrees(math.atan2(person.vy, person.vx))

                # Predicted future position
                person.predicted_x = int(cx + person.vx * self.prediction_horizon)
                person.predicted_y = int(cy + person.vy * self.prediction_horizon)

        # Clean up stale tracks (not seen in last 2 seconds)
        if persons:
            latest_t = max(p.timestamp for p in persons)
            active_ids = {p.track_id for p in persons}
            stale_ids = [
                tid for tid, trail in self._trajectories.items()
                if tid not in active_ids and (latest_t - trail[-1][0]) > 2.0
            ]
            for tid in stale_ids:
                del self._trajectories[tid]

        return persons

    def get_trail(self, track_id: int) -> list[tuple[float, int, int]]:
        """Get the position history for a given track ID."""
        return self._trajectories.get(track_id, [])

    def get_all_trails(self) -> dict[int, list[tuple[float, int, int]]]:
        """Get all active trails."""
        return dict(self._trajectories)
