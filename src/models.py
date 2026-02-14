"""
Shared data models for the PREVENT pipeline.
All layers communicate through these structures.
"""

from dataclasses import dataclass, field
from typing import Optional
import time


@dataclass
class BBox:
    """Bounding box in pixel coordinates."""
    x1: int
    y1: int
    x2: int
    y2: int

    @property
    def cx(self) -> int:
        return (self.x1 + self.x2) // 2

    @property
    def cy(self) -> int:
        return (self.y1 + self.y2) // 2

    @property
    def width(self) -> int:
        return self.x2 - self.x1

    @property
    def height(self) -> int:
        return self.y2 - self.y1

    @property
    def area(self) -> int:
        return self.width * self.height


@dataclass
class PoseLandmarks:
    """Pose landmark data for a person."""
    points: list  # list of (x_px, y_px, visibility) tuples
    hip_midpoint: Optional[tuple] = None  # (x, y) in pixels
    nose: Optional[tuple] = None  # (x, y) in pixels


@dataclass
class TrackedPerson:
    """A tracked person with identity, position, and motion."""
    track_id: int
    bbox: BBox
    confidence: float
    timestamp: float
    # Motion (filled by Layer 2)
    vx: float = 0.0  # velocity x (px/s)
    vy: float = 0.0  # velocity y (px/s)
    speed: float = 0.0  # magnitude (px/s)
    direction_deg: float = 0.0  # angle in degrees
    # Prediction (filled by Layer 2)
    predicted_x: Optional[int] = None
    predicted_y: Optional[int] = None
    # Pose (optional, filled by Layer 1)
    pose: Optional[PoseLandmarks] = None
    # Depth (optional)
    depth_estimate: Optional[float] = None
    # Activity (filled by activity recognizer)
    activity: str = "unknown"
    activity_confidence: float = 0.0


@dataclass
class DetectedObject:
    """A detected non-person object in the scene."""
    object_id: int
    class_id: int
    class_name: str
    bbox: BBox
    confidence: float
    depth_estimate: Optional[float] = None


@dataclass
class DangerZone:
    """A rectangular danger zone in pixel coordinates."""
    zone_id: str
    bbox: BBox
    label: str = "DANGER ZONE"


@dataclass
class RiskEvent:
    """A detected risk/collision event."""
    event_id: str
    person_id: int
    zone_id: str
    ttc: float  # time to collision in seconds
    risk_score: float  # 0.0 to 1.0
    timestamp: float = field(default_factory=time.time)
    description: str = ""


@dataclass
class Intervention:
    """A triggered intervention action."""
    intervention_id: str
    risk_event: RiskEvent
    action_type: str  # "visual_warning", "sound_alert", "log", "banner"
    timestamp: float = field(default_factory=time.time)
    acknowledged: bool = False


@dataclass
class SceneState:
    """Complete state of the scene at a given frame."""
    frame_number: int
    timestamp: float
    persons: list  # list of TrackedPerson
    danger_zones: list  # list of DangerZone
    risk_events: list  # list of RiskEvent
    interventions: list  # list of Intervention
    fps: float = 0.0
    frame_width: int = 0
    frame_height: int = 0
