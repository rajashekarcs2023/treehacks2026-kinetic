"""
Activity Recognition from Pose Landmarks
=========================================
Classifies what each person is DOING based on skeletal geometry.

Activities detected:
  - standing: upright posture
  - sitting: hip-knee angle indicates seated
  - walking: moderate speed + upright
  - running: high speed + upright
  - lying_down / fallen: torso roughly horizontal (CRITICAL for accessibility)
  - waving: hand significantly above head
  - reaching: arm extended laterally
  - crouching: knees bent significantly, torso lowered

All detection is purely geometric — no ML model needed, runs at zero cost.

Robust features:
  - Temporal smoothing: majority vote over last N frames per person
  - Multi-signal confidence: pose geometry + speed + consistency
  - Exercise detection: bent-knee + arm movements (squats, curls, etc.)
  - Hysteresis: requires sustained signal before switching activity
"""

import math
import time
from collections import deque
from src.models import TrackedPerson, PoseLandmarks


# MediaPipe landmark indices
NOSE = 0
LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12
LEFT_ELBOW = 13
RIGHT_ELBOW = 14
LEFT_WRIST = 15
RIGHT_WRIST = 16
LEFT_HIP = 23
RIGHT_HIP = 24
LEFT_KNEE = 25
RIGHT_KNEE = 26
LEFT_ANKLE = 27
RIGHT_ANKLE = 28


def _point(pose: PoseLandmarks, idx: int) -> tuple[float, float, float] | None:
    """Get a pose point (x, y, visibility) or None if not visible enough."""
    if idx >= len(pose.points):
        return None
    x, y, vis = pose.points[idx]
    if vis < 0.4:
        return None
    return (x, y, vis)


def _angle(a: tuple, b: tuple, c: tuple) -> float:
    """Compute angle at point b given three points (in degrees)."""
    ba = (a[0] - b[0], a[1] - b[1])
    bc = (c[0] - b[0], c[1] - b[1])
    dot = ba[0] * bc[0] + ba[1] * bc[1]
    mag_ba = math.sqrt(ba[0]**2 + ba[1]**2) + 1e-6
    mag_bc = math.sqrt(bc[0]**2 + bc[1]**2) + 1e-6
    cos_angle = max(-1, min(1, dot / (mag_ba * mag_bc)))
    return math.degrees(math.acos(cos_angle))


def _dist(a: tuple, b: tuple) -> float:
    """Euclidean distance between two points."""
    return math.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)


def _get_candidates(person: TrackedPerson) -> list[tuple[str, float]]:
    """Get all activity candidates with raw confidence scores (multi-signal)."""
    candidates = []
    pose = person.pose

    if pose is None:
        if person.speed > 200:
            candidates.append(("running", 0.5))
        elif person.speed > 50:
            candidates.append(("walking", 0.5))
        else:
            candidates.append(("standing", 0.3))
        return candidates

    # Get key points
    nose = _point(pose, NOSE)
    l_shoulder = _point(pose, LEFT_SHOULDER)
    r_shoulder = _point(pose, RIGHT_SHOULDER)
    l_hip = _point(pose, LEFT_HIP)
    r_hip = _point(pose, RIGHT_HIP)
    l_knee = _point(pose, LEFT_KNEE)
    r_knee = _point(pose, RIGHT_KNEE)
    l_ankle = _point(pose, LEFT_ANKLE)
    r_ankle = _point(pose, RIGHT_ANKLE)
    l_wrist = _point(pose, LEFT_WRIST)
    r_wrist = _point(pose, RIGHT_WRIST)
    l_elbow = _point(pose, LEFT_ELBOW)
    r_elbow = _point(pose, RIGHT_ELBOW)

    # ── FALLEN / LYING DOWN (critical for accessibility) ──────────────
    if l_shoulder and r_shoulder and l_hip and r_hip:
        shoulder_mid = ((l_shoulder[0] + r_shoulder[0]) / 2,
                        (l_shoulder[1] + r_shoulder[1]) / 2)
        hip_mid = ((l_hip[0] + r_hip[0]) / 2,
                   (l_hip[1] + r_hip[1]) / 2)
        torso_dx = abs(shoulder_mid[0] - hip_mid[0])
        torso_dy = abs(shoulder_mid[1] - hip_mid[1])
        if torso_dy < 1:
            torso_dy = 1
        torso_angle = math.degrees(math.atan2(torso_dx, torso_dy))

        if torso_angle > 60:
            candidates.append(("fallen", 0.85))
        elif torso_angle > 45:
            candidates.append(("lying_down", 0.7))

    # ── WAVING (hand above head) ──────────────────────────────────────
    if nose:
        for wrist in [l_wrist, r_wrist]:
            if wrist and wrist[1] < nose[1] - 50:
                if l_shoulder and wrist[1] < l_shoulder[1] - 80:
                    candidates.append(("waving", 0.75))

    # ── EXERCISING (key joint angles indicate exercise motion) ────────
    exercise_signals = 0
    if l_hip and l_knee and l_ankle:
        knee_angle = _angle(l_hip, l_knee, l_ankle)
        if 60 < knee_angle < 130:  # mid-range → active movement
            exercise_signals += 1
    if r_hip and r_knee and r_ankle:
        knee_angle = _angle(r_hip, r_knee, r_ankle)
        if 60 < knee_angle < 130:
            exercise_signals += 1
    if l_shoulder and l_elbow and l_wrist:
        elbow_angle = _angle(l_shoulder, l_elbow, l_wrist)
        if elbow_angle < 100:  # bent arm → curl-like
            exercise_signals += 1
    if r_shoulder and r_elbow and r_wrist:
        elbow_angle = _angle(r_shoulder, r_elbow, r_wrist)
        if elbow_angle < 100:
            exercise_signals += 1
    if exercise_signals >= 2 and person.speed < 100:
        candidates.append(("exercising", 0.55 + exercise_signals * 0.1))

    # ── SITTING (knee angle + low speed) ──────────────────────────────
    sitting_score = 0
    if l_hip and l_knee and l_ankle:
        if _angle(l_hip, l_knee, l_ankle) < 120:
            sitting_score += 0.5
    if r_hip and r_knee and r_ankle:
        if _angle(r_hip, r_knee, r_ankle) < 120:
            sitting_score += 0.5
    if sitting_score >= 0.5 and person.speed < 30:
        candidates.append(("sitting", 0.6 + sitting_score * 0.2))

    # ── CROUCHING ─────────────────────────────────────────────────────
    if l_hip and l_knee and r_hip and r_knee:
        hip_y = (l_hip[1] + r_hip[1]) / 2
        knee_y = (l_knee[1] + r_knee[1]) / 2
        if abs(hip_y - knee_y) < 50 and person.speed < 30:
            candidates.append(("crouching", 0.65))

    # ── REACHING (arm extended) ───────────────────────────────────────
    if l_shoulder and l_wrist:
        if _dist(l_shoulder, l_wrist) > 200 and l_wrist[1] < l_shoulder[1]:
            candidates.append(("reaching", 0.6))
    if r_shoulder and r_wrist:
        if _dist(r_shoulder, r_wrist) > 200 and r_wrist[1] < r_shoulder[1]:
            candidates.append(("reaching", 0.6))

    # ── Speed-based classification (always present as baseline) ───────
    if person.speed > 250:
        candidates.append(("running", 0.8))
    elif person.speed > 60:
        candidates.append(("walking", 0.8))
    else:
        candidates.append(("standing", 0.7))

    return candidates


# Safety-critical activities get priority boost so they're never masked
_PRIORITY_BOOST = {"fallen": 0.2, "lying_down": 0.15}


def classify_activity(person: TrackedPerson) -> tuple[str, float]:
    """Classify the activity of a tracked person based on pose + motion.

    Returns (activity_name, confidence). Uses multi-signal scoring
    with priority boost for safety-critical activities (fallen, lying_down).
    """
    candidates = _get_candidates(person)
    if not candidates:
        return ("unknown", 0.0)
    # Apply priority boost for safety-critical activities
    boosted = [(act, conf + _PRIORITY_BOOST.get(act, 0.0)) for act, conf in candidates]
    best_act, best_conf = max(boosted, key=lambda x: x[1])
    # Return original confidence (without boost) for the winning activity
    orig_conf = next(c for a, c in candidates if a == best_act)
    return (best_act, orig_conf)


# ═══════════════════════════════════════════════════════════════════════
# Temporal Smoothing — majority vote over recent frames per person
# ═══════════════════════════════════════════════════════════════════════

class ActivityClassifier:
    """Hybrid activity classifier: ML model + heuristic fallback + temporal smoothing.

    Three-tier approach:
      1. ML model (if trained): temporal 1D CNN on joint angle sequences
      2. Heuristic fallback: geometric rules for when ML model isn't available
      3. Temporal smoothing: majority vote + hysteresis on top of either

    The ML model is preferred when available and confident (>0.5).
    Falls back to heuristics otherwise. Both go through temporal smoothing.
    """

    def __init__(self, window_size: int = 10, switch_threshold: int = 4,
                 ml_confidence_threshold: float = 0.5):
        self.window_size = window_size
        self.switch_threshold = switch_threshold
        self.ml_confidence_threshold = ml_confidence_threshold
        self._history: dict[int, deque] = {}
        self._current: dict[int, tuple[str, float]] = {}
        self._last_seen: dict[int, float] = {}

        # ML model (lazy-loaded)
        self._ml_model = None
        self._ml_buffer = None
        self._ml_loaded = False
        self._ml_predictions = 0
        self._heuristic_predictions = 0

    def _ensure_ml(self):
        """Lazy-load the ML model and frame buffer."""
        if self._ml_loaded:
            return
        self._ml_loaded = True
        try:
            from aegis.activity_model import ActivityModelTrainer, ActivityFrameBuffer, extract_features
            self._extract_features = extract_features
            trainer = ActivityModelTrainer()
            if trainer.load():
                self._ml_model = trainer
                self._ml_buffer = ActivityFrameBuffer()
                print(f"[ActivityClassifier] ML model loaded (acc={trainer.metadata.get('best_val_accuracy', '?')})")
            else:
                print("[ActivityClassifier] No ML model found, using heuristics")
        except Exception as e:
            print(f"[ActivityClassifier] ML model load failed: {e}, using heuristics")

    def _classify_one(self, person: TrackedPerson) -> tuple[str, float, str]:
        """Classify a single person. Returns (activity, confidence, source)."""
        self._ensure_ml()

        # Try ML model first
        if self._ml_model is not None and self._ml_buffer is not None:
            pid = person.track_id
            features = self._extract_features(person)
            self._ml_buffer.add_frame(pid, features)

            if self._ml_buffer.has_enough(pid, min_frames=3):
                window = self._ml_buffer.get_window(pid)
                activity, confidence, _ = self._ml_model.predict(window)

                if confidence >= self.ml_confidence_threshold:
                    self._ml_predictions += 1
                    return (activity, confidence, "ml")

        # Fallback to heuristic
        activity, confidence = classify_activity(person)
        self._heuristic_predictions += 1
        return (activity, confidence, "heuristic")

    def classify(self, person: TrackedPerson) -> tuple[str, float]:
        """Classify with ML + heuristic fallback + temporal smoothing."""
        pid = person.track_id
        raw_activity, raw_conf, source = self._classify_one(person)

        if pid not in self._history:
            self._history[pid] = deque(maxlen=self.window_size)
            self._current[pid] = (raw_activity, raw_conf)

        self._history[pid].append((raw_activity, raw_conf))
        self._last_seen[pid] = time.time()

        # Majority vote
        counts: dict[str, list[float]] = {}
        for act, conf in self._history[pid]:
            if act not in counts:
                counts[act] = []
            counts[act].append(conf)

        best_act = raw_activity
        best_score = 0.0
        for act, confs in counts.items():
            score = len(confs) * (sum(confs) / len(confs))
            if score > best_score:
                best_score = score
                best_act = act

        # Hysteresis
        current_act = self._current[pid][0]
        if best_act != current_act:
            new_count = len(counts.get(best_act, []))
            if new_count >= self.switch_threshold:
                avg_conf = sum(counts[best_act]) / len(counts[best_act])
                self._current[pid] = (best_act, avg_conf)
        else:
            avg_conf = sum(counts[best_act]) / len(counts[best_act])
            self._current[pid] = (best_act, avg_conf)

        return self._current[pid]

    def classify_all(self, persons: list[TrackedPerson]) -> list[TrackedPerson]:
        """Classify all persons with ML + heuristic + temporal smoothing."""
        for person in persons:
            activity, confidence = self.classify(person)
            person.activity = activity
            person.activity_confidence = confidence

        # Prune stale
        now = time.time()
        stale = [pid for pid, t in self._last_seen.items() if now - t > 30]
        for pid in stale:
            self._history.pop(pid, None)
            self._current.pop(pid, None)
            self._last_seen.pop(pid, None)
        if self._ml_buffer:
            self._ml_buffer.prune_stale()

        return persons

    def train_ml_model(self, n_per_class: int = 150, epochs: int = 60) -> dict:
        """Train (or retrain) the ML activity model from synthetic data."""
        from aegis.activity_model import ActivityModelTrainer, ActivityFrameBuffer, extract_features
        trainer = ActivityModelTrainer()
        result = trainer.generate_and_train(n_per_class=n_per_class, epochs=epochs)
        if trainer.trained:
            trainer.save()
            self._ml_model = trainer
            self._ml_buffer = ActivityFrameBuffer()
            self._extract_features = extract_features
            self._ml_loaded = True
        return result

    def get_stats(self) -> dict:
        return {
            "ml_model_loaded": self._ml_model is not None,
            "ml_predictions": self._ml_predictions,
            "heuristic_predictions": self._heuristic_predictions,
            "ml_ratio": round(self._ml_predictions / max(self._ml_predictions + self._heuristic_predictions, 1), 3),
            "tracked_persons": len(self._current),
            "ml_metadata": self._ml_model.metadata if self._ml_model else None,
        }


# Global instance for use in spatial_engine
_classifier = ActivityClassifier()


def classify_all(persons: list[TrackedPerson]) -> list[TrackedPerson]:
    """Classify activity for all persons with ML + heuristic + temporal smoothing."""
    return _classifier.classify_all(persons)
