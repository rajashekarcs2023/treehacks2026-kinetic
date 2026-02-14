"""
Layer 4: Agentic Decision Engine
==================================
Monitors risk events and decides when to trigger interventions.
Implements policy rules and prevents alert flooding.

Input:  List of RiskEvent (from Layer 3)
Output: List of Intervention decisions
"""

import time
import uuid
from src.models import RiskEvent, Intervention


class DecisionEngine:
    """Rule-based decision engine for triggering interventions."""

    def __init__(self, risk_threshold: float = 0.3, cooldown_seconds: float = 2.0,
                 ttc_critical: float = 1.0):
        self.risk_threshold = risk_threshold
        self.cooldown_seconds = cooldown_seconds
        self.ttc_critical = ttc_critical
        # track_id -> last intervention timestamp
        self._last_intervention: dict[int, float] = {}
        # Full intervention log
        self.intervention_log: list[Intervention] = []

    def _in_cooldown(self, person_id: int, now: float) -> bool:
        last = self._last_intervention.get(person_id, 0)
        return (now - last) < self.cooldown_seconds

    def evaluate(self, risk_events: list[RiskEvent]) -> list[Intervention]:
        """
        Evaluate risk events and decide which interventions to trigger.
        Returns list of new Intervention objects.
        """
        interventions = []
        now = time.time()

        for event in risk_events:
            # Skip low-risk events
            if event.risk_score < self.risk_threshold:
                continue

            # Skip if in cooldown for this person
            if self._in_cooldown(event.person_id, now):
                continue

            # Determine action type based on severity
            if event.ttc == 0.0:
                action_type = "critical_alert"
            elif event.ttc < self.ttc_critical:
                action_type = "urgent_warning"
            else:
                action_type = "early_warning"

            intervention = Intervention(
                intervention_id=str(uuid.uuid4())[:8],
                risk_event=event,
                action_type=action_type,
                timestamp=now,
            )
            interventions.append(intervention)

            # Update cooldown
            self._last_intervention[event.person_id] = now

            # Add to log
            self.intervention_log.append(intervention)

        return interventions

    def get_stats(self) -> dict:
        """Get intervention statistics."""
        if not self.intervention_log:
            return {"total": 0, "critical": 0, "urgent": 0, "early": 0}

        return {
            "total": len(self.intervention_log),
            "critical": sum(1 for i in self.intervention_log if i.action_type == "critical_alert"),
            "urgent": sum(1 for i in self.intervention_log if i.action_type == "urgent_warning"),
            "early": sum(1 for i in self.intervention_log if i.action_type == "early_warning"),
        }
