"""
AEGIS Pose Comparison Engine
=============================
Core module for the Skill Coach. Provides:

1. Skeleton normalization (position + scale invariant)
2. Joint angle computation (10 key angles)
3. Similarity scoring (per-joint deviation, weighted overall)
4. Reference store (record, save, load expert skeleton sequences)
5. DTW alignment (temporal matching for movement sequences)
6. Phase detection (preparation → execution → peak → recovery)
7. Movement quality metrics (smoothness, symmetry, tempo)
8. Rep counting (auto-detect from phase cycles)

Works with MediaPipe's 33-landmark format:
  points = [(x_px, y_px, visibility), ...] × 33
"""

import json
import math
import os
import time
import numpy as np
from dataclasses import dataclass, field
from typing import Optional


# ═══════════════════════════════════════════════════════════════════════
# LANDMARK INDICES (MediaPipe 33-point)
# ═══════════════════════════════════════════════════════════════════════

NOSE = 0
LEFT_EYE_INNER = 1
LEFT_EYE = 2
LEFT_EYE_OUTER = 3
RIGHT_EYE_INNER = 4
RIGHT_EYE = 5
RIGHT_EYE_OUTER = 6
LEFT_EAR = 7
RIGHT_EAR = 8
MOUTH_LEFT = 9
MOUTH_RIGHT = 10
LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12
LEFT_ELBOW = 13
RIGHT_ELBOW = 14
LEFT_WRIST = 15
RIGHT_WRIST = 16
LEFT_PINKY = 17
RIGHT_PINKY = 18
LEFT_INDEX = 19
RIGHT_INDEX = 20
LEFT_THUMB = 21
RIGHT_THUMB = 22
LEFT_HIP = 23
RIGHT_HIP = 24
LEFT_KNEE = 25
RIGHT_KNEE = 26
LEFT_ANKLE = 27
RIGHT_ANKLE = 28
LEFT_HEEL = 29
RIGHT_HEEL = 30
LEFT_FOOT_INDEX = 31
RIGHT_FOOT_INDEX = 32

# Key joint angle definitions: (point_a, vertex, point_c) → angle at vertex
KEY_ANGLES = {
    "left_elbow":     (LEFT_SHOULDER, LEFT_ELBOW, LEFT_WRIST),
    "right_elbow":    (RIGHT_SHOULDER, RIGHT_ELBOW, RIGHT_WRIST),
    "left_shoulder":  (LEFT_ELBOW, LEFT_SHOULDER, LEFT_HIP),
    "right_shoulder": (RIGHT_ELBOW, RIGHT_SHOULDER, RIGHT_HIP),
    "left_hip":       (LEFT_SHOULDER, LEFT_HIP, LEFT_KNEE),
    "right_hip":      (RIGHT_SHOULDER, RIGHT_HIP, RIGHT_KNEE),
    "left_knee":      (LEFT_HIP, LEFT_KNEE, LEFT_ANKLE),
    "right_knee":     (RIGHT_HIP, RIGHT_KNEE, RIGHT_ANKLE),
    "left_ankle":     (LEFT_KNEE, LEFT_ANKLE, LEFT_FOOT_INDEX),
    "right_ankle":    (RIGHT_KNEE, RIGHT_ANKLE, RIGHT_FOOT_INDEX),
    # Additional angles for more precise comparison
    "spine":          (NOSE, LEFT_SHOULDER, LEFT_HIP),       # torso lean
    "neck":           (LEFT_EAR, NOSE, LEFT_SHOULDER),       # head tilt
    "left_wrist":     (LEFT_ELBOW, LEFT_WRIST, LEFT_INDEX),  # wrist angle
    "right_wrist":    (RIGHT_ELBOW, RIGHT_WRIST, RIGHT_INDEX),
    "torso_lean":     (LEFT_SHOULDER, LEFT_HIP, LEFT_ANKLE),  # full body lean
    "hip_width":      (LEFT_KNEE, LEFT_HIP, RIGHT_HIP),       # stance width
}

# Friendly names for display
ANGLE_NAMES = {
    "left_elbow": "Left Elbow",
    "right_elbow": "Right Elbow",
    "left_shoulder": "Left Shoulder",
    "right_shoulder": "Right Shoulder",
    "left_hip": "Left Hip",
    "right_hip": "Right Hip",
    "left_knee": "Left Knee",
    "right_knee": "Right Knee",
    "left_ankle": "Left Ankle",
    "right_ankle": "Right Ankle",
    "spine": "Spine",
    "neck": "Neck",
    "left_wrist": "Left Wrist",
    "right_wrist": "Right Wrist",
    "torso_lean": "Torso Lean",
    "hip_width": "Stance Width",
}

# Default weights for similarity scoring (higher = more important)
DEFAULT_WEIGHTS = {
    "left_elbow": 1.0,
    "right_elbow": 1.0,
    "left_shoulder": 1.2,
    "right_shoulder": 1.2,
    "left_hip": 1.5,
    "right_hip": 1.5,
    "left_knee": 1.5,
    "right_knee": 1.5,
    "left_ankle": 0.8,
    "right_ankle": 0.8,
    "spine": 1.3,
    "neck": 0.6,
    "left_wrist": 0.7,
    "right_wrist": 0.7,
    "torso_lean": 1.4,
    "hip_width": 1.0,
}


# ═══════════════════════════════════════════════════════════════════════
# DATA STRUCTURES
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class NormalizedSkeleton:
    """A skeleton normalized for position and scale."""
    points: list[tuple[float, float, float]]  # (x, y, visibility) × 33
    joint_angles: dict[str, float]             # computed angles in degrees
    timestamp: float = 0.0

    def to_dict(self) -> dict:
        return {
            "points": self.points,
            "joint_angles": self.joint_angles,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "NormalizedSkeleton":
        return cls(
            points=[tuple(p) for p in d["points"]],
            joint_angles=d["joint_angles"],
            timestamp=d.get("timestamp", 0.0),
        )


@dataclass
class MovementPhase:
    """A detected phase within a movement."""
    name: str              # "preparation", "execution", "peak", "recovery"
    start_frame: int
    end_frame: int
    key_angle: str         # which angle defines this phase
    angle_at_start: float
    angle_at_end: float


@dataclass
class ComparisonResult:
    """Result of comparing user pose to expert reference."""
    similarity_score: float                   # 0-100
    per_joint_deviation: dict[str, float]     # joint → degrees off
    worst_joints: list[tuple[str, float]]     # sorted by deviation (worst first)
    best_joints: list[tuple[str, float]]      # sorted by deviation (best first)
    phase: str = ""                           # current phase if known
    phase_score: float = 0.0                  # phase-specific score

    def to_dict(self) -> dict:
        return {
            "similarity_score": round(self.similarity_score, 1),
            "per_joint_deviation": {k: round(v, 1) for k, v in self.per_joint_deviation.items()},
            "worst_joints": [(j, round(d, 1)) for j, d in self.worst_joints[:3]],
            "best_joints": [(j, round(d, 1)) for j, d in self.best_joints[:3]],
            "phase": self.phase,
            "phase_score": round(self.phase_score, 1),
        }


@dataclass
class SkeletonSequence:
    """A recorded sequence of skeletons (for an expert reference or user attempt)."""
    name: str
    skeletons: list[NormalizedSkeleton]
    phases: list[MovementPhase] = field(default_factory=list)
    fps: float = 15.0
    created_at: float = field(default_factory=time.time)
    metadata: dict = field(default_factory=dict)

    @property
    def duration(self) -> float:
        return len(self.skeletons) / max(self.fps, 1)

    @property
    def frame_count(self) -> int:
        return len(self.skeletons)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "skeletons": [s.to_dict() for s in self.skeletons],
            "phases": [
                {"name": p.name, "start_frame": p.start_frame, "end_frame": p.end_frame,
                 "key_angle": p.key_angle, "angle_at_start": p.angle_at_start,
                 "angle_at_end": p.angle_at_end}
                for p in self.phases
            ],
            "fps": self.fps,
            "created_at": self.created_at,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "SkeletonSequence":
        return cls(
            name=d["name"],
            skeletons=[NormalizedSkeleton.from_dict(s) for s in d["skeletons"]],
            phases=[
                MovementPhase(**p) for p in d.get("phases", [])
            ],
            fps=d.get("fps", 15.0),
            created_at=d.get("created_at", 0.0),
            metadata=d.get("metadata", {}),
        )


@dataclass
class RepData:
    """Data collected from a single rep."""
    rep_number: int
    similarity_score: float
    per_joint_deviation: dict[str, float]
    corrections: list[str]
    timestamp: float
    skeleton_sequence: Optional[list[dict]] = None  # raw data for training


# ═══════════════════════════════════════════════════════════════════════
# CORE FUNCTIONS: Normalization + Angles
# ═══════════════════════════════════════════════════════════════════════

def _get_point(points: list, idx: int, min_vis: float = 0.3) -> Optional[tuple[float, float]]:
    """Get a 2D point from landmarks, or None if not visible enough."""
    if idx >= len(points):
        return None
    x, y, vis = points[idx]
    if vis < min_vis:
        return None
    return (float(x), float(y))


def _compute_angle(a: tuple[float, float], b: tuple[float, float],
                   c: tuple[float, float]) -> float:
    """Compute angle at point b formed by a-b-c, in degrees."""
    ba = (a[0] - b[0], a[1] - b[1])
    bc = (c[0] - b[0], c[1] - b[1])
    dot = ba[0] * bc[0] + ba[1] * bc[1]
    mag_ba = math.sqrt(ba[0]**2 + ba[1]**2) + 1e-8
    mag_bc = math.sqrt(bc[0]**2 + bc[1]**2) + 1e-8
    cos_angle = max(-1.0, min(1.0, dot / (mag_ba * mag_bc)))
    return math.degrees(math.acos(cos_angle))


def compute_joint_angles(points: list) -> dict[str, float]:
    """Compute all key joint angles from raw landmark points.

    Returns dict of {angle_name: degrees}. Missing angles (due to
    occluded landmarks) are omitted.
    """
    angles = {}
    for name, (idx_a, idx_b, idx_c) in KEY_ANGLES.items():
        a = _get_point(points, idx_a)
        b = _get_point(points, idx_b)
        c = _get_point(points, idx_c)
        if a is not None and b is not None and c is not None:
            angles[name] = _compute_angle(a, b, c)
    return angles


def normalize_skeleton(points: list) -> NormalizedSkeleton:
    """Normalize a skeleton for position and scale invariance.

    1. Translate so hip midpoint is at origin
    2. Scale so torso length (hip to shoulder midpoint) = 1.0
    3. Compute joint angles

    Args:
        points: list of 33 (x, y, visibility) tuples (pixel coords)

    Returns:
        NormalizedSkeleton with normalized points and computed angles
    """
    if len(points) < 33:
        # Pad with zeros if needed
        points = points + [(0.0, 0.0, 0.0)] * (33 - len(points))

    # Get hip midpoint
    l_hip = _get_point(points, LEFT_HIP)
    r_hip = _get_point(points, RIGHT_HIP)

    if l_hip and r_hip:
        hip_cx = (l_hip[0] + r_hip[0]) / 2
        hip_cy = (l_hip[1] + r_hip[1]) / 2
    else:
        # Fallback: use center of all visible points
        visible = [(p[0], p[1]) for p in points if p[2] > 0.3]
        if visible:
            hip_cx = sum(p[0] for p in visible) / len(visible)
            hip_cy = sum(p[1] for p in visible) / len(visible)
        else:
            hip_cx, hip_cy = 0.0, 0.0

    # Translate to hip center
    centered = []
    for x, y, vis in points:
        centered.append((x - hip_cx, y - hip_cy, vis))

    # Compute torso length for scaling
    l_shoulder = _get_point(centered, LEFT_SHOULDER)
    r_shoulder = _get_point(centered, RIGHT_SHOULDER)
    l_hip_c = _get_point(centered, LEFT_HIP)
    r_hip_c = _get_point(centered, RIGHT_HIP)

    torso_length = 1.0  # default if can't compute
    if l_shoulder and r_shoulder and l_hip_c and r_hip_c:
        shoulder_mid = ((l_shoulder[0] + r_shoulder[0]) / 2,
                        (l_shoulder[1] + r_shoulder[1]) / 2)
        hip_mid = ((l_hip_c[0] + r_hip_c[0]) / 2,
                   (l_hip_c[1] + r_hip_c[1]) / 2)
        torso_length = math.sqrt(
            (shoulder_mid[0] - hip_mid[0])**2 +
            (shoulder_mid[1] - hip_mid[1])**2
        )
        if torso_length < 10:  # too small, probably noise
            torso_length = 1.0

    # Scale by torso length
    scaled = []
    for x, y, vis in centered:
        scaled.append((x / torso_length, y / torso_length, vis))

    # Compute joint angles from ORIGINAL points (angles are scale-invariant)
    angles = compute_joint_angles(points)

    return NormalizedSkeleton(points=scaled, joint_angles=angles)


# ═══════════════════════════════════════════════════════════════════════
# SIMILARITY SCORING
# ═══════════════════════════════════════════════════════════════════════

def _gaussian_score(deviation: float, sigma: float = 20.0) -> float:
    """Gaussian-weighted scoring: small deviations are penalized more proportionally.
    sigma=20 means 20° deviation ≈ 60% score, 10° ≈ 88%, 30° ≈ 32%, 45°+ ≈ ~0%.
    Much better than linear for detecting subtle form issues."""
    return math.exp(-(deviation ** 2) / (2 * sigma ** 2))


def _cosine_similarity_points(user_pts: list, expert_pts: list,
                              min_vis: float = 0.4) -> float:
    """Cosine similarity between normalized skeleton point vectors.
    Captures spatial positioning that angles alone miss (e.g., stance width, arm spread)."""
    # Build vectors from points where BOTH have good visibility
    u_vec, e_vec = [], []
    for i in range(min(len(user_pts), len(expert_pts))):
        u_x, u_y, u_v = user_pts[i]
        e_x, e_y, e_v = expert_pts[i]
        if u_v >= min_vis and e_v >= min_vis:
            u_vec.extend([u_x, u_y])
            e_vec.extend([e_x, e_y])
    if len(u_vec) < 6:  # need at least 3 joints
        return 0.5  # neutral
    u_arr = np.array(u_vec, dtype=np.float64)
    e_arr = np.array(e_vec, dtype=np.float64)
    dot = np.dot(u_arr, e_arr)
    norm_u = np.linalg.norm(u_arr) + 1e-8
    norm_e = np.linalg.norm(e_arr) + 1e-8
    return float(max(0.0, dot / (norm_u * norm_e)))


def compare_poses(user_angles: dict[str, float],
                  expert_angles: dict[str, float],
                  weights: dict[str, float] = None,
                  user_skeleton: Optional['NormalizedSkeleton'] = None,
                  expert_skeleton: Optional['NormalizedSkeleton'] = None) -> ComparisonResult:
    """Compare user's pose to expert's using hybrid scoring.

    Scoring dimensions:
    1. Gaussian-weighted angle deviation (70% of score) — precise joint matching
    2. Cosine similarity of normalized points (30% of score) — spatial positioning

    Returns ComparisonResult with overall score and per-joint deviations.
    """
    if weights is None:
        weights = DEFAULT_WEIGHTS

    deviations = {}
    for joint in KEY_ANGLES:
        if joint in user_angles and joint in expert_angles:
            deviations[joint] = abs(user_angles[joint] - expert_angles[joint])

    if not deviations:
        return ComparisonResult(
            similarity_score=0.0,
            per_joint_deviation={},
            worst_joints=[],
            best_joints=[],
        )

    # Dimension 1: Gaussian-weighted angle score (70%)
    total_weight = 0.0
    weighted_score = 0.0
    for joint, dev in deviations.items():
        w = weights.get(joint, 1.0)
        total_weight += w
        joint_score = _gaussian_score(dev, sigma=20.0)
        weighted_score += w * joint_score

    angle_score = 100.0 * weighted_score / max(total_weight, 1e-8)

    # Dimension 2: Cosine similarity of normalized skeleton (30%)
    spatial_score = 50.0  # neutral default if no skeletons provided
    if user_skeleton and expert_skeleton:
        cos_sim = _cosine_similarity_points(user_skeleton.points, expert_skeleton.points)
        spatial_score = 100.0 * cos_sim

    # Hybrid: 70% angles, 30% spatial
    overall = 0.7 * angle_score + 0.3 * spatial_score

    # Sort joints by deviation
    sorted_joints = sorted(deviations.items(), key=lambda x: -x[1])
    sorted_best = sorted(deviations.items(), key=lambda x: x[1])

    return ComparisonResult(
        similarity_score=overall,
        per_joint_deviation=deviations,
        worst_joints=sorted_joints[:5],
        best_joints=sorted_best[:5],
    )


def compute_skeleton_distance(skel_a: NormalizedSkeleton,
                              skel_b: NormalizedSkeleton) -> float:
    """Compute distance between two normalized skeletons.

    Uses sum of joint angle differences (more robust than point distance).
    """
    total = 0.0
    count = 0
    for joint in KEY_ANGLES:
        if joint in skel_a.joint_angles and joint in skel_b.joint_angles:
            total += abs(skel_a.joint_angles[joint] - skel_b.joint_angles[joint])
            count += 1
    return total / max(count, 1)


# ═══════════════════════════════════════════════════════════════════════
# DYNAMIC TIME WARPING
# ═══════════════════════════════════════════════════════════════════════

def dtw_align(user_seq: list[NormalizedSkeleton],
              expert_seq: list[NormalizedSkeleton]) -> tuple[list[tuple[int, int]], float]:
    """Align two skeleton sequences using Dynamic Time Warping.

    Returns:
        path: list of (user_frame, expert_frame) alignment pairs
        total_cost: accumulated DTW distance
    """
    n = len(user_seq)
    m = len(expert_seq)

    if n == 0 or m == 0:
        return [], float('inf')

    # Build cost matrix
    cost = np.zeros((n, m))
    for i in range(n):
        for j in range(m):
            cost[i, j] = compute_skeleton_distance(user_seq[i], expert_seq[j])

    # DTW accumulation matrix
    D = np.full((n + 1, m + 1), np.inf)
    D[0, 0] = 0.0

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            D[i, j] = cost[i - 1, j - 1] + min(
                D[i - 1, j],      # insertion
                D[i, j - 1],      # deletion
                D[i - 1, j - 1],  # match
            )

    total_cost = D[n, m]

    # Backtrack to get optimal path
    path = []
    i, j = n, m
    while i > 0 and j > 0:
        path.append((i - 1, j - 1))
        candidates = [D[i - 1, j - 1], D[i - 1, j], D[i, j - 1]]
        argmin = int(np.argmin(candidates))
        if argmin == 0:
            i, j = i - 1, j - 1
        elif argmin == 1:
            i = i - 1
        else:
            j = j - 1

    path.reverse()
    return path, total_cost


def compare_sequences(user_seq: list[NormalizedSkeleton],
                      expert_seq: list[NormalizedSkeleton],
                      weights: dict[str, float] = None) -> dict:
    """Compare two movement sequences using DTW alignment.

    Returns a detailed comparison with per-frame and overall scores.
    """
    if not user_seq or not expert_seq:
        return {"error": "Empty sequence", "similarity_score": 0.0}

    path, total_cost = dtw_align(user_seq, expert_seq)

    if not path:
        return {"error": "DTW alignment failed", "similarity_score": 0.0}

    # Score each aligned frame (with full skeleton for hybrid scoring)
    frame_scores = []
    frame_deviations = []
    for user_idx, expert_idx in path:
        result = compare_poses(
            user_seq[user_idx].joint_angles,
            expert_seq[expert_idx].joint_angles,
            weights,
            user_skeleton=user_seq[user_idx],
            expert_skeleton=expert_seq[expert_idx],
        )
        frame_scores.append(result.similarity_score)
        frame_deviations.append(result.per_joint_deviation)

    # Aggregate
    avg_score = sum(frame_scores) / len(frame_scores)
    min_score = min(frame_scores)
    max_score = max(frame_scores)

    # Average per-joint deviation across all frames
    avg_deviations = {}
    for joint in KEY_ANGLES:
        joint_devs = [fd.get(joint, 0) for fd in frame_deviations if joint in fd]
        if joint_devs:
            avg_deviations[joint] = sum(joint_devs) / len(joint_devs)

    worst_joints = sorted(avg_deviations.items(), key=lambda x: -x[1])

    return {
        "similarity_score": round(avg_score, 1),
        "min_frame_score": round(min_score, 1),
        "max_frame_score": round(max_score, 1),
        "frame_count": len(path),
        "dtw_cost": round(total_cost, 2),
        "avg_deviations": {k: round(v, 1) for k, v in avg_deviations.items()},
        "worst_joints": [(j, round(d, 1)) for j, d in worst_joints[:3]],
        "frame_scores": [round(s, 1) for s in frame_scores],
    }


# ═══════════════════════════════════════════════════════════════════════
# PHASE DETECTION
# ═══════════════════════════════════════════════════════════════════════

def detect_phases(sequence: list[NormalizedSkeleton],
                  key_angle: str = "left_knee",
                  min_phase_frames: int = 3) -> list[MovementPhase]:
    """Detect movement phases from a skeleton sequence.

    Looks at the key angle over time to find preparation (increasing),
    execution (decreasing to minimum), peak (at minimum), and recovery
    (increasing back).

    Works for squats (knee angle), bicep curls (elbow angle), etc.
    """
    if len(sequence) < min_phase_frames * 2:
        return []

    # Extract angle values over time
    angles = []
    for skel in sequence:
        a = skel.joint_angles.get(key_angle)
        if a is not None:
            angles.append(a)
        else:
            angles.append(angles[-1] if angles else 180.0)

    if not angles:
        return []

    angles = np.array(angles)

    # Smooth the signal
    kernel_size = min(5, len(angles))
    if kernel_size > 1:
        kernel = np.ones(kernel_size) / kernel_size
        smoothed = np.convolve(angles, kernel, mode='same')
    else:
        smoothed = angles

    # Find local minima and maxima
    phases = []
    # Simple state machine: track if angle is going down or up
    state = "idle"  # idle, descending, ascending
    phase_start = 0
    prev_angle = smoothed[0]

    for i in range(1, len(smoothed)):
        delta = smoothed[i] - prev_angle

        if state == "idle":
            if delta < -1:  # angle decreasing
                state = "descending"
                phase_start = max(0, i - 1)
                phases.append(MovementPhase(
                    name="preparation",
                    start_frame=0,
                    end_frame=phase_start,
                    key_angle=key_angle,
                    angle_at_start=float(smoothed[0]),
                    angle_at_end=float(smoothed[phase_start]),
                ))

        elif state == "descending":
            if delta > 1:  # angle started increasing = hit the bottom
                state = "ascending"
                phases.append(MovementPhase(
                    name="execution",
                    start_frame=phase_start,
                    end_frame=i,
                    key_angle=key_angle,
                    angle_at_start=float(smoothed[phase_start]),
                    angle_at_end=float(smoothed[i]),
                ))
                phase_start = i

        elif state == "ascending":
            if delta < -1:  # angle decreasing again = new rep starting
                phases.append(MovementPhase(
                    name="recovery",
                    start_frame=phase_start,
                    end_frame=i,
                    key_angle=key_angle,
                    angle_at_start=float(smoothed[phase_start]),
                    angle_at_end=float(smoothed[i]),
                ))
                state = "descending"
                phase_start = i

        prev_angle = smoothed[i]

    # Close final phase
    if state == "ascending" and phase_start < len(smoothed) - 1:
        phases.append(MovementPhase(
            name="recovery",
            start_frame=phase_start,
            end_frame=len(smoothed) - 1,
            key_angle=key_angle,
            angle_at_start=float(smoothed[phase_start]),
            angle_at_end=float(smoothed[-1]),
        ))

    return phases


def count_reps_from_angles(angles: list[float], threshold_ratio: float = 0.3) -> int:
    """Count repetitions from a series of angle values.

    A rep is defined as the angle going below a threshold and back up.
    threshold_ratio: fraction of the range used as the detection threshold.
    """
    if len(angles) < 3:
        return 0

    arr = np.array(angles)
    angle_min = arr.min()
    angle_max = arr.max()
    angle_range = angle_max - angle_min

    if angle_range < 10:  # not enough movement to count reps
        return 0

    threshold = angle_max - angle_range * threshold_ratio

    # Count crossings below threshold
    below = arr < threshold
    reps = 0
    was_below = False
    for b in below:
        if b and not was_below:
            reps += 1
        was_below = b

    return reps


# ═══════════════════════════════════════════════════════════════════════
# MOVEMENT QUALITY METRICS
# ═══════════════════════════════════════════════════════════════════════

def compute_movement_quality(sequence: list[NormalizedSkeleton]) -> dict:
    """Compute quality metrics for a movement sequence.

    Returns:
        smoothness: how smooth the movement is (0-100, higher = smoother)
        symmetry: left-right symmetry score (0-100, higher = more symmetric)
        tempo_consistency: how consistent the speed is across reps (0-100)
    """
    if len(sequence) < 3:
        return {"smoothness": 0, "symmetry": 0, "tempo_consistency": 0}

    # Smoothness: low jerk in joint angles = smooth movement
    jerk_scores = []
    for joint in KEY_ANGLES:
        angles = [s.joint_angles.get(joint) for s in sequence]
        angles = [a for a in angles if a is not None]
        if len(angles) < 3:
            continue
        # Jerk = second derivative of angle
        arr = np.array(angles)
        velocity = np.diff(arr)
        acceleration = np.diff(velocity)
        jerk = np.mean(np.abs(acceleration))
        # Map jerk to 0-100 score (lower jerk = higher smoothness)
        jerk_scores.append(max(0, 100 - jerk * 5))

    smoothness = sum(jerk_scores) / max(len(jerk_scores), 1)

    # Symmetry: compare left vs right joint angles
    symmetry_pairs = [
        ("left_elbow", "right_elbow"),
        ("left_shoulder", "right_shoulder"),
        ("left_hip", "right_hip"),
        ("left_knee", "right_knee"),
        ("left_ankle", "right_ankle"),
    ]

    sym_scores = []
    for l_joint, r_joint in symmetry_pairs:
        l_angles = [s.joint_angles.get(l_joint) for s in sequence]
        r_angles = [s.joint_angles.get(r_joint) for s in sequence]
        pairs = [(l, r) for l, r in zip(l_angles, r_angles) if l is not None and r is not None]
        if pairs:
            diffs = [abs(l - r) for l, r in pairs]
            avg_diff = sum(diffs) / len(diffs)
            sym_scores.append(max(0, 100 - avg_diff * 3))

    symmetry = sum(sym_scores) / max(len(sym_scores), 1)

    # Tempo consistency: variation in frame-to-frame angle change
    tempo_scores = []
    for joint in ["left_knee", "right_knee", "left_hip", "right_hip"]:
        angles = [s.joint_angles.get(joint) for s in sequence]
        angles = [a for a in angles if a is not None]
        if len(angles) < 5:
            continue
        velocities = np.abs(np.diff(angles))
        if velocities.mean() > 0:
            cv = velocities.std() / velocities.mean()  # coefficient of variation
            tempo_scores.append(max(0, 100 - cv * 50))

    tempo_consistency = sum(tempo_scores) / max(len(tempo_scores), 1)

    return {
        "smoothness": round(smoothness, 1),
        "symmetry": round(symmetry, 1),
        "tempo_consistency": round(tempo_consistency, 1),
    }


def detect_compensation(user_angles: dict[str, float],
                        expert_angles: dict[str, float]) -> list[dict]:
    """Detect compensatory movement patterns.

    Compensation = one side deviates significantly more than the other,
    suggesting the user is favoring one side to avoid weakness.
    """
    compensations = []

    pairs = [
        ("left_knee", "right_knee", "knee"),
        ("left_hip", "right_hip", "hip"),
        ("left_shoulder", "right_shoulder", "shoulder"),
        ("left_elbow", "right_elbow", "elbow"),
    ]

    for l_joint, r_joint, body_part in pairs:
        if all(j in user_angles and j in expert_angles for j in [l_joint, r_joint]):
            l_dev = abs(user_angles[l_joint] - expert_angles[l_joint])
            r_dev = abs(user_angles[r_joint] - expert_angles[r_joint])

            asymmetry = abs(l_dev - r_dev)
            if asymmetry > 15:  # significant asymmetry
                weak_side = "left" if l_dev > r_dev else "right"
                compensations.append({
                    "body_part": body_part,
                    "weak_side": weak_side,
                    "asymmetry_degrees": round(asymmetry, 1),
                    "description": (
                        f"Your {weak_side} {body_part} deviates {round(asymmetry, 1)}° more "
                        f"than the other side — possible compensation pattern"
                    ),
                })

    return compensations


# ═══════════════════════════════════════════════════════════════════════
# REFERENCE STORE
# ═══════════════════════════════════════════════════════════════════════

class ReferenceStore:
    """Manages stored expert skeleton sequences."""

    def __init__(self, store_dir: str = None):
        if store_dir is None:
            store_dir = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                "data", "references"
            )
        self.store_dir = store_dir
        os.makedirs(store_dir, exist_ok=True)

    def save(self, sequence: SkeletonSequence) -> str:
        """Save a skeleton sequence to disk. Returns the file path."""
        filename = f"{sequence.name.replace(' ', '_').lower()}_{int(sequence.created_at)}.json"
        filepath = os.path.join(self.store_dir, filename)
        with open(filepath, "w") as f:
            json.dump(sequence.to_dict(), f)
        return filepath

    def load(self, name: str) -> Optional[SkeletonSequence]:
        """Load a reference by name (returns most recent match)."""
        matches = []
        for fname in os.listdir(self.store_dir):
            if fname.endswith(".json") and name.replace(" ", "_").lower() in fname.lower():
                filepath = os.path.join(self.store_dir, fname)
                try:
                    with open(filepath) as f:
                        data = json.load(f)
                    matches.append(SkeletonSequence.from_dict(data))
                except Exception:
                    continue

        if not matches:
            return None
        # Return most recent
        return max(matches, key=lambda s: s.created_at)

    def list_references(self) -> list[dict]:
        """List all stored references with metadata."""
        refs = []
        for fname in os.listdir(self.store_dir):
            if not fname.endswith(".json"):
                continue
            filepath = os.path.join(self.store_dir, fname)
            try:
                with open(filepath) as f:
                    data = json.load(f)
                refs.append({
                    "name": data["name"],
                    "frame_count": len(data.get("skeletons", [])),
                    "fps": data.get("fps", 15),
                    "duration": len(data.get("skeletons", [])) / max(data.get("fps", 15), 1),
                    "phases": len(data.get("phases", [])),
                    "created_at": data.get("created_at", 0),
                    "file": fname,
                })
            except Exception:
                continue
        return sorted(refs, key=lambda r: -r["created_at"])

    def delete(self, name: str) -> bool:
        """Delete a reference by name."""
        for fname in os.listdir(self.store_dir):
            if fname.endswith(".json") and name.replace(" ", "_").lower() in fname.lower():
                os.remove(os.path.join(self.store_dir, fname))
                return True
        return False


# ═══════════════════════════════════════════════════════════════════════
# RECORDING SESSION
# ═══════════════════════════════════════════════════════════════════════

class RecordingSession:
    """Records a live skeleton sequence for use as a reference."""

    def __init__(self, name: str, fps: float = 15.0):
        self.name = name
        self.fps = fps
        self.frames: list[NormalizedSkeleton] = []
        self.start_time = time.time()
        self.active = True

    def add_frame(self, points: list) -> NormalizedSkeleton:
        """Add a frame from raw landmark points. Returns the normalized skeleton."""
        skel = normalize_skeleton(points)
        skel.timestamp = time.time() - self.start_time
        self.frames.append(skel)
        return skel

    def stop(self, key_angle: str = "left_knee") -> SkeletonSequence:
        """Stop recording and return the complete sequence with phases."""
        self.active = False
        phases = detect_phases(self.frames, key_angle) if len(self.frames) > 6 else []

        return SkeletonSequence(
            name=self.name,
            skeletons=self.frames,
            phases=phases,
            fps=self.fps,
            created_at=self.start_time,
            metadata={
                "frame_count": len(self.frames),
                "duration": time.time() - self.start_time,
            },
        )


# ═══════════════════════════════════════════════════════════════════════
# COACHING SESSION
# ═══════════════════════════════════════════════════════════════════════

class CoachingSession:
    """Manages an active coaching session — tracks reps, scores, progress."""

    def __init__(self, skill_name: str, reference: Optional[SkeletonSequence] = None):
        self.skill_name = skill_name
        self.reference = reference
        self.start_time = time.time()
        self.reps: list[RepData] = []
        self.current_rep_frames: list[NormalizedSkeleton] = []
        self.frame_count = 0
        self.active = True

        # For rep detection
        self._angle_history: dict[str, list[float]] = {}
        self._primary_angle = "left_knee"  # default, can be changed

        # Training data collection
        self.training_data: list[dict] = []

    def set_primary_angle(self, angle: str):
        """Set which angle is used for rep detection."""
        if angle in KEY_ANGLES:
            self._primary_angle = angle

    def add_frame(self, points: list) -> Optional[ComparisonResult]:
        """Process a new frame. Returns comparison result if reference exists."""
        skel = normalize_skeleton(points)
        skel.timestamp = time.time() - self.start_time
        self.current_rep_frames.append(skel)
        self.frame_count += 1

        # Track angles for rep counting
        for joint, angle in skel.joint_angles.items():
            if joint not in self._angle_history:
                self._angle_history[joint] = []
            self._angle_history[joint].append(angle)

        # Compare to reference if available
        if self.reference and self.reference.skeletons:
            # Find the best matching frame in reference (sample every 3rd for speed)
            best_idx = 0
            best_dist = float('inf')
            step = max(1, len(self.reference.skeletons) // 50)  # at most 50 comparisons
            for i in range(0, len(self.reference.skeletons), step):
                ref_skel = self.reference.skeletons[i]
                d = compute_skeleton_distance(skel, ref_skel)
                if d < best_dist:
                    best_dist = d
                    best_idx = i

            # Refine: check neighbors of best match
            for i in range(max(0, best_idx - 2), min(len(self.reference.skeletons), best_idx + 3)):
                d = compute_skeleton_distance(skel, self.reference.skeletons[i])
                if d < best_dist:
                    best_dist = d
                    best_idx = i

            ref_skel = self.reference.skeletons[best_idx]
            return compare_poses(
                skel.joint_angles, ref_skel.joint_angles,
                user_skeleton=skel, expert_skeleton=ref_skel,
            )

        return None

    def complete_rep(self, similarity_score: float,
                     deviations: dict[str, float],
                     corrections: list[str]) -> RepData:
        """Record a completed rep with its data."""
        rep = RepData(
            rep_number=len(self.reps) + 1,
            similarity_score=similarity_score,
            per_joint_deviation=deviations,
            corrections=corrections,
            timestamp=time.time(),
            skeleton_sequence=[s.to_dict() for s in self.current_rep_frames],
        )
        self.reps.append(rep)

        # Collect training data
        self.training_data.append({
            "angles_sequence": [s.joint_angles for s in self.current_rep_frames],
            "score": similarity_score,
            "corrections": corrections,
            "deviations": deviations,
        })

        # Reset for next rep
        self.current_rep_frames = []
        return rep

    def get_rep_count(self) -> int:
        """Get current rep count from angle history."""
        angles = self._angle_history.get(self._primary_angle, [])
        if len(angles) < 5:
            return len(self.reps)
        return max(len(self.reps), count_reps_from_angles(angles))

    def get_progress(self) -> dict:
        """Get coaching progress summary."""
        if not self.reps:
            return {
                "skill": self.skill_name,
                "reps_completed": 0,
                "avg_score": 0,
                "trend": "no_data",
                "duration": round(time.time() - self.start_time, 1),
            }

        scores = [r.similarity_score for r in self.reps]
        avg = sum(scores) / len(scores)
        best = max(scores)
        worst = min(scores)

        # Trend: compare first half to second half
        if len(scores) >= 4:
            mid = len(scores) // 2
            first_half = sum(scores[:mid]) / mid
            second_half = sum(scores[mid:]) / (len(scores) - mid)
            if second_half > first_half + 3:
                trend = "improving"
            elif second_half < first_half - 3:
                trend = "declining"
            else:
                trend = "stable"
        elif len(scores) >= 2 and scores[-1] > scores[0] + 3:
            trend = "improving"
        else:
            trend = "stable"

        # Most common corrections
        all_corrections = []
        for r in self.reps:
            all_corrections.extend(r.corrections)
        correction_counts = {}
        for c in all_corrections:
            correction_counts[c] = correction_counts.get(c, 0) + 1
        top_corrections = sorted(correction_counts.items(), key=lambda x: -x[1])[:3]

        return {
            "skill": self.skill_name,
            "reps_completed": len(self.reps),
            "avg_score": round(avg, 1),
            "best_score": round(best, 1),
            "worst_score": round(worst, 1),
            "latest_score": round(scores[-1], 1),
            "trend": trend,
            "scores_per_rep": [round(s, 1) for s in scores],
            "top_corrections": top_corrections,
            "duration": round(time.time() - self.start_time, 1),
            "training_samples_collected": len(self.training_data),
        }

    def end(self) -> dict:
        """End the session and return final summary."""
        self.active = False
        progress = self.get_progress()
        progress["session_ended"] = True
        progress["total_frames"] = self.frame_count
        return progress

    def get_training_data(self) -> list[dict]:
        """Get collected training data for model training."""
        return self.training_data


# ═══════════════════════════════════════════════════════════════════════
# CONVENIENCE: Quick compare from raw points
# ═══════════════════════════════════════════════════════════════════════

def quick_compare(user_points: list, expert_points: list,
                  weights: dict = None) -> ComparisonResult:
    """Quick single-frame comparison from raw landmark points.

    Uses hybrid scoring: 70% angle match + 30% spatial cosine similarity.

    Args:
        user_points: [(x, y, vis), ...] × 33 — user's current pose
        expert_points: [(x, y, vis), ...] × 33 — expert's reference pose

    Returns:
        ComparisonResult
    """
    user_skel = normalize_skeleton(user_points)
    expert_skel = normalize_skeleton(expert_points)
    return compare_poses(
        user_skel.joint_angles, expert_skel.joint_angles, weights,
        user_skeleton=user_skel, expert_skeleton=expert_skel,
    )
