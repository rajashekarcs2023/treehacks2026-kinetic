"""
AEGIS Data Collection Pipeline — Every rep → labeled training data.

Automatically collects training data from coaching sessions:
- Angle sequences per rep (input features)
- Similarity scores (labels)
- Corrections (metadata)
- Skill name and reference used

Data is stored as JSONL files (one line per sample) for easy streaming
into the 1D CNN SkillScorer model.

Data format per sample:
{
    "skill": "squat",
    "angles_sequence": [[angle1, angle2, ...], ...],  # T × N_angles
    "score": 85.2,
    "corrections": ["left_knee_too_open"],
    "deviations": {"left_knee": 12.3, ...},
    "duration_frames": 45,
    "reference_name": "perfect_squat",
    "timestamp": 1707900000.0
}
"""

import json
import math
import os
import random
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np


# Ordered list of angle keys for consistent feature vectors
ANGLE_KEYS = [
    "left_elbow", "right_elbow",
    "left_shoulder", "right_shoulder",
    "left_hip", "right_hip",
    "left_knee", "right_knee",
    "left_ankle", "right_ankle",
]

# Biomechanically plausible angle ranges (degrees) per joint
ANGLE_RANGES = {
    "left_elbow": (30, 180), "right_elbow": (30, 180),
    "left_shoulder": (10, 180), "right_shoulder": (10, 180),
    "left_hip": (30, 180), "right_hip": (30, 180),
    "left_knee": (30, 180), "right_knee": (30, 180),
    "left_ankle": (60, 140), "right_ankle": (60, 140),
}


@dataclass
class TrainingSample:
    """A single training sample from one rep."""
    skill: str
    angles_sequence: list[list[float]]  # T × 10 (one row per frame)
    score: float
    corrections: list[str] = field(default_factory=list)
    deviations: dict[str, float] = field(default_factory=dict)
    duration_frames: int = 0
    reference_name: str = ""
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return {
            "skill": self.skill,
            "angles_sequence": self.angles_sequence,
            "score": round(self.score, 2),
            "corrections": self.corrections,
            "deviations": {k: round(v, 2) for k, v in self.deviations.items()},
            "duration_frames": self.duration_frames,
            "reference_name": self.reference_name,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "TrainingSample":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    def validate(self) -> list[str]:
        """Validate this sample. Returns list of issues (empty = valid)."""
        issues = []
        if not self.skill:
            issues.append("missing skill name")
        if not self.angles_sequence:
            issues.append("empty angles_sequence")
        if self.score < 0 or self.score > 100:
            issues.append(f"score out of range: {self.score}")
        if self.duration_frames < 1:
            issues.append(f"invalid duration_frames: {self.duration_frames}")
        # Check for NaN/Inf in angles
        for i, frame in enumerate(self.angles_sequence):
            if len(frame) != len(ANGLE_KEYS):
                issues.append(f"frame {i} has {len(frame)} angles, expected {len(ANGLE_KEYS)}")
                break
            for j, val in enumerate(frame):
                if math.isnan(val) or math.isinf(val):
                    issues.append(f"frame {i}, angle {j} ({ANGLE_KEYS[j]}): NaN/Inf")
                    break
                lo, hi = ANGLE_RANGES[ANGLE_KEYS[j]]
                if val < lo - 20 or val > hi + 20:  # allow some slack
                    issues.append(f"frame {i}, {ANGLE_KEYS[j]}={val:.1f} outside plausible range [{lo},{hi}]")
                    break
        return issues


class DataCollector:
    """Collects and stores training data from coaching sessions."""

    def __init__(self, data_dir: str = None):
        if data_dir is None:
            data_dir = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                "data", "training"
            )
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)

    def _get_filepath(self, skill: str) -> str:
        """Get the JSONL file path for a skill."""
        safe_name = skill.replace(" ", "_").lower()
        return os.path.join(self.data_dir, f"{safe_name}.jsonl")

    def save_sample(self, sample: TrainingSample, validate: bool = True) -> dict:
        """Append a training sample to the skill's JSONL file.

        Returns: {"saved": True/False, "issues": [...], "path": str}
        """
        if validate:
            issues = sample.validate()
            if issues:
                return {"saved": False, "issues": issues, "path": ""}

        filepath = self._get_filepath(sample.skill)
        with open(filepath, "a") as f:
            f.write(json.dumps(sample.to_dict()) + "\n")
        return {"saved": True, "issues": [], "path": filepath}

    def save_rep_from_session(self, skill: str, skeleton_frames: list,
                              score: float, corrections: list[str],
                              deviations: dict[str, float],
                              reference_name: str = ""):
        """Convert coaching session rep data into a training sample and save.

        Args:
            skill: Skill name
            skeleton_frames: List of NormalizedSkeleton objects from the rep
            score: Similarity score for this rep
            corrections: List of correction strings
            deviations: Per-joint deviation dict
            reference_name: Name of the reference used
        """
        # Convert skeletons to angle matrix (T × 10)
        angles_sequence = []
        for skel in skeleton_frames:
            angles = skel.joint_angles if hasattr(skel, 'joint_angles') else {}
            row = [angles.get(key, 0.0) for key in ANGLE_KEYS]
            angles_sequence.append(row)

        sample = TrainingSample(
            skill=skill,
            angles_sequence=angles_sequence,
            score=score,
            corrections=corrections,
            deviations=deviations,
            duration_frames=len(skeleton_frames),
            reference_name=reference_name,
        )
        self.save_sample(sample)
        return sample

    def load_samples(self, skill: str) -> list[TrainingSample]:
        """Load all training samples for a skill."""
        filepath = self._get_filepath(skill)
        if not os.path.exists(filepath):
            return []

        samples = []
        with open(filepath) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        samples.append(TrainingSample.from_dict(json.loads(line)))
                    except Exception:
                        continue
        return samples

    def load_all_samples(self) -> list[TrainingSample]:
        """Load all training samples across all skills."""
        all_samples = []
        for fname in os.listdir(self.data_dir):
            if fname.endswith(".jsonl"):
                skill = fname[:-6]  # remove .jsonl
                all_samples.extend(self.load_samples(skill))
        return all_samples

    def get_last_session_summary(self, skill: str, max_samples: int = 20) -> dict | None:
        """Get a summary of the last coaching session for a skill.

        Returns avg score, total reps, top corrections, best/worst scores.
        Used to inject session memory into the voice coach.
        """
        filepath = self._get_filepath(skill)
        if not os.path.exists(filepath):
            return None

        # Read last N lines efficiently
        lines = []
        try:
            with open(filepath) as f:
                all_lines = f.readlines()
                lines = all_lines[-max_samples:] if len(all_lines) > max_samples else all_lines
        except Exception:
            return None

        if not lines:
            return None

        scores = []
        corrections_count: dict[str, int] = {}
        for line in lines:
            try:
                d = json.loads(line.strip())
                scores.append(d.get("score", 0))
                for c in d.get("corrections", []):
                    corrections_count[c] = corrections_count.get(c, 0) + 1
            except Exception:
                continue

        if not scores:
            return None

        top_corrections = sorted(corrections_count.items(), key=lambda x: -x[1])[:3]
        return {
            "skill": skill,
            "total_reps": len(scores),
            "avg_score": round(sum(scores) / len(scores), 1),
            "best_score": round(max(scores), 1),
            "worst_score": round(min(scores), 1),
            "top_corrections": [c[0] for c in top_corrections],
            "improving": len(scores) >= 3 and scores[-1] > scores[0],
        }

    def get_dataset_stats(self) -> dict:
        """Get statistics about the collected training data."""
        stats = {
            "total_samples": 0,
            "skills": {},
            "total_files": 0,
        }

        for fname in os.listdir(self.data_dir):
            if not fname.endswith(".jsonl"):
                continue
            stats["total_files"] += 1
            skill = fname[:-6]
            filepath = os.path.join(self.data_dir, fname)

            count = 0
            scores = []
            with open(filepath) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            d = json.loads(line)
                            count += 1
                            scores.append(d.get("score", 0))
                        except Exception:
                            continue

            stats["total_samples"] += count
            if count > 0:
                stats["skills"][skill] = {
                    "samples": count,
                    "avg_score": round(sum(scores) / len(scores), 1),
                    "min_score": round(min(scores), 1),
                    "max_score": round(max(scores), 1),
                }

        return stats

    def export_for_training(self, skill: str = None,
                            min_frames: int = 5,
                            pad_to: int = 60) -> dict:
        """Export data in a format ready for model training.

        Returns:
            {
                "X": list of angle sequences (N × pad_to × 10), padded/truncated
                "y": list of scores (N,)
                "skills": list of skill names (N,)
                "n_samples": int
                "n_features": 10
                "seq_length": pad_to
            }
        """
        if skill:
            samples = self.load_samples(skill)
        else:
            samples = self.load_all_samples()

        X = []
        y = []
        skills = []

        for sample in samples:
            seq = sample.angles_sequence
            if len(seq) < min_frames:
                continue

            # Pad or truncate to fixed length
            if len(seq) > pad_to:
                seq = seq[:pad_to]
            elif len(seq) < pad_to:
                # Pad with last frame repeated
                last = seq[-1] if seq else [0.0] * len(ANGLE_KEYS)
                seq = seq + [last] * (pad_to - len(seq))

            X.append(seq)
            y.append(sample.score)
            skills.append(sample.skill)

        return {
            "X": X,
            "y": y,
            "skills": skills,
            "n_samples": len(X),
            "n_features": len(ANGLE_KEYS),
            "seq_length": pad_to,
            "angle_keys": ANGLE_KEYS,
        }

    def clear_skill_data(self, skill: str) -> bool:
        """Clear all training data for a skill."""
        filepath = self._get_filepath(skill)
        if os.path.exists(filepath):
            os.remove(filepath)
            return True
        return False

    # ── Data Augmentation ─────────────────────────────────────────────

    def augment_sample(self, sample: TrainingSample, n_augments: int = 3) -> list[TrainingSample]:
        """Generate augmented copies of a training sample.

        Augmentations:
          1. Time warp: stretch/compress the sequence
          2. Noise injection: add Gaussian noise to angles
          3. Score jitter: small perturbation to score label
          4. Mirror: swap left/right joints
        """
        augmented = []
        seq = np.array(sample.angles_sequence, dtype=np.float64)
        T, C = seq.shape

        for _ in range(n_augments):
            aug_seq = seq.copy()
            aug_score = sample.score
            aug_type = random.choice(["noise", "time_warp", "mirror", "combo"])

            if aug_type in ("noise", "combo"):
                # Add Gaussian noise (std=2-5 degrees)
                noise_std = random.uniform(2.0, 5.0)
                aug_seq += np.random.randn(T, C) * noise_std
                # Clamp to valid ranges
                for j, key in enumerate(ANGLE_KEYS):
                    lo, hi = ANGLE_RANGES[key]
                    aug_seq[:, j] = np.clip(aug_seq[:, j], lo, hi)
                # Worse form = noise reduces score slightly
                aug_score = max(0, min(100, aug_score - noise_std * 0.5))

            if aug_type in ("time_warp", "combo"):
                # Stretch or compress by 0.8-1.2x
                factor = random.uniform(0.8, 1.2)
                new_T = max(5, int(T * factor))
                indices = np.linspace(0, T - 1, new_T).astype(int)
                aug_seq = aug_seq[indices]

            if aug_type == "mirror":
                # Swap left and right joint columns
                mirrored = aug_seq.copy()
                for j in range(0, C, 2):  # ANGLE_KEYS alternates L/R
                    if j + 1 < C:
                        mirrored[:, j], mirrored[:, j + 1] = aug_seq[:, j + 1].copy(), aug_seq[:, j].copy()
                aug_seq = mirrored

            # Small score jitter
            aug_score += random.gauss(0, 1.5)
            aug_score = max(0, min(100, aug_score))

            new_sample = TrainingSample(
                skill=sample.skill,
                angles_sequence=aug_seq.tolist(),
                score=round(aug_score, 2),
                corrections=sample.corrections,
                deviations=sample.deviations,
                duration_frames=len(aug_seq),
                reference_name=sample.reference_name,
            )
            augmented.append(new_sample)

        return augmented

    def export_for_training_augmented(self, skill: str = None,
                                      min_frames: int = 5,
                                      pad_to: int = 60,
                                      augment_factor: int = 5) -> dict:
        """Export data with augmentation for robust training.

        With 10 real samples and augment_factor=5, you get 60 total.
        """
        if skill:
            samples = self.load_samples(skill)
        else:
            samples = self.load_all_samples()

        # Filter valid samples
        valid = [s for s in samples if len(s.angles_sequence) >= min_frames and not s.validate()]

        # Augment
        all_samples = list(valid)
        for s in valid:
            all_samples.extend(self.augment_sample(s, n_augments=augment_factor))

        # Shuffle
        random.shuffle(all_samples)

        X, y, skills_list = [], [], []
        for sample in all_samples:
            seq = sample.angles_sequence
            if len(seq) > pad_to:
                seq = seq[:pad_to]
            elif len(seq) < pad_to:
                last = seq[-1] if seq else [0.0] * len(ANGLE_KEYS)
                seq = seq + [last] * (pad_to - len(seq))
            X.append(seq)
            y.append(sample.score)
            skills_list.append(sample.skill)

        return {
            "X": X, "y": y, "skills": skills_list,
            "n_samples": len(X),
            "n_real": len(valid),
            "n_augmented": len(X) - len(valid),
            "n_features": len(ANGLE_KEYS),
            "seq_length": pad_to,
            "angle_keys": ANGLE_KEYS,
        }

    # ── Synthetic Data Generation ─────────────────────────────────────

    def generate_synthetic_data(self, skill: str, n_samples: int = 50,
                                 ideal_angles: dict[str, float] = None) -> dict:
        """Generate synthetic training data for bootstrapping when no real data exists.

        Creates samples with varying quality levels (good, medium, poor form)
        based on ideal angles for a given skill.

        Args:
            skill: Skill name
            n_samples: Number of synthetic samples to generate
            ideal_angles: Dict of ideal joint angles for perfect form.
                          If None, uses defaults for common skills.
        """
        if ideal_angles is None:
            ideal_angles = _get_default_ideal_angles(skill)

        generated = 0
        for _ in range(n_samples):
            # Random quality level: score between 30-100
            target_score = random.uniform(30, 100)
            # Deviation proportional to how bad the form is
            max_deviation = (100 - target_score) * 0.5  # 0° for perfect, 35° for score=30

            # Generate a sequence of 20-80 frames
            n_frames = random.randint(20, 80)
            seq = []
            for t in range(n_frames):
                frame = []
                for key in ANGLE_KEYS:
                    ideal = ideal_angles.get(key, 120.0)
                    lo, hi = ANGLE_RANGES[key]
                    # Add per-joint deviation
                    dev = random.gauss(0, max_deviation * 0.5)
                    # Add temporal variation (smooth movement)
                    phase = math.sin(2 * math.pi * t / max(n_frames, 1) * random.uniform(0.5, 2))
                    angle = ideal + dev + phase * 10
                    angle = max(lo, min(hi, angle))
                    frame.append(round(angle, 1))
                seq.append(frame)

            # Compute actual score from deviation
            avg_devs = []
            for j, key in enumerate(ANGLE_KEYS):
                ideal = ideal_angles.get(key, 120.0)
                col = [seq[t][j] for t in range(n_frames)]
                avg_devs.append(abs(np.mean(col) - ideal))
            actual_score = max(0, min(100, 100 - np.mean(avg_devs) * 2.5))

            sample = TrainingSample(
                skill=skill,
                angles_sequence=seq,
                score=round(actual_score, 2),
                corrections=[],
                deviations={},
                duration_frames=n_frames,
                reference_name="synthetic",
            )
            result = self.save_sample(sample, validate=True)
            if result["saved"]:
                generated += 1

        return {
            "skill": skill,
            "generated": generated,
            "requested": n_samples,
            "ideal_angles": ideal_angles,
        }

    # ── Train/Val/Test Split ──────────────────────────────────────────

    def export_split(self, skill: str = None, pad_to: int = 60,
                     min_frames: int = 5,
                     train_ratio: float = 0.7,
                     val_ratio: float = 0.15,
                     augment_train: int = 3) -> dict:
        """Export data with proper train/val/test split.

        Only training set gets augmented. Val and test are real data only.
        """
        if skill:
            samples = self.load_samples(skill)
        else:
            samples = self.load_all_samples()

        valid = [s for s in samples if len(s.angles_sequence) >= min_frames and not s.validate()]
        random.shuffle(valid)

        n = len(valid)
        if n < 3:
            return {"error": f"Need at least 3 valid samples, have {n}"}

        n_train = max(1, int(n * train_ratio))
        n_val = max(1, int(n * val_ratio))
        n_test = max(1, n - n_train - n_val)

        train_samples = valid[:n_train]
        val_samples = valid[n_train:n_train + n_val]
        test_samples = valid[n_train + n_val:]

        # Augment training set only
        aug_train = list(train_samples)
        if augment_train > 0:
            for s in train_samples:
                aug_train.extend(self.augment_sample(s, n_augments=augment_train))
        random.shuffle(aug_train)

        def to_arrays(sample_list):
            X, y = [], []
            for s in sample_list:
                seq = s.angles_sequence
                if len(seq) > pad_to:
                    seq = seq[:pad_to]
                elif len(seq) < pad_to:
                    last = seq[-1] if seq else [0.0] * len(ANGLE_KEYS)
                    seq = seq + [last] * (pad_to - len(seq))
                X.append(seq)
                y.append(s.score)
            return X, y

        train_X, train_y = to_arrays(aug_train)
        val_X, val_y = to_arrays(val_samples)
        test_X, test_y = to_arrays(test_samples)

        return {
            "train": {"X": train_X, "y": train_y, "n": len(train_X)},
            "val": {"X": val_X, "y": val_y, "n": len(val_X)},
            "test": {"X": test_X, "y": test_y, "n": len(test_X)},
            "n_real": n,
            "n_train_augmented": len(train_X),
            "seq_length": pad_to,
            "n_features": len(ANGLE_KEYS),
        }


# ═══════════════════════════════════════════════════════════════════════
# Ideal angles for common skills (for synthetic data generation)
# ═══════════════════════════════════════════════════════════════════════

def _get_default_ideal_angles(skill: str) -> dict[str, float]:
    """Return ideal joint angles for common skills."""
    defaults = {
        "squat": {
            "left_knee": 90, "right_knee": 90,
            "left_hip": 90, "right_hip": 90,
            "left_ankle": 80, "right_ankle": 80,
            "left_shoulder": 90, "right_shoulder": 90,
            "left_elbow": 170, "right_elbow": 170,
        },
        "lunge": {
            "left_knee": 90, "right_knee": 90,
            "left_hip": 100, "right_hip": 120,
            "left_ankle": 80, "right_ankle": 90,
            "left_shoulder": 90, "right_shoulder": 90,
            "left_elbow": 170, "right_elbow": 170,
        },
        "bicep_curl": {
            "left_elbow": 40, "right_elbow": 40,
            "left_shoulder": 30, "right_shoulder": 30,
            "left_hip": 170, "right_hip": 170,
            "left_knee": 175, "right_knee": 175,
            "left_ankle": 90, "right_ankle": 90,
        },
        "warrior_pose": {
            "left_knee": 90, "right_knee": 170,
            "left_hip": 100, "right_hip": 140,
            "left_ankle": 80, "right_ankle": 90,
            "left_shoulder": 170, "right_shoulder": 170,
            "left_elbow": 170, "right_elbow": 170,
        },
    }
    # Fuzzy match
    skill_lower = skill.lower().replace(" ", "_")
    for key, angles in defaults.items():
        if key in skill_lower or skill_lower in key:
            return angles
    # Generic fallback: standing neutral
    return {k: 160.0 for k in ANGLE_KEYS}
