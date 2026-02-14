"""
AEGIS Hybrid Scorer — Local model (instant) + Claude (deep).

Two-tier scoring system:
  Tier 1 (instant, <1ms): Local SkillScorer CNN
    - Runs on every frame
    - Provides real-time score bar updates
    - No network latency

  Tier 2 (deep, ~2s): Claude agent reasoning
    - Runs every N seconds or on significant events
    - Provides detailed natural language corrections
    - Uses biomechanics knowledge for context-aware coaching
    - Powered via MCP tools (compare_to_reference, get_joint_deviation, etc.)

The hybrid approach gives:
  - Instant visual feedback (score bar, joint colors)
  - Deep coaching wisdom (voice corrections, technique insights)
  - Graceful degradation (works offline with just local model)
"""

import os
import time
from typing import Optional
from dataclasses import dataclass, field

import numpy as np

from aegis.skill_scorer import SkillScorer
from aegis.data_collector import DataCollector, ANGLE_KEYS

# Try PyTorch scorer (GPU-accelerated), fall back to NumPy
try:
    from aegis.skill_scorer_torch import SkillScorerTorch, TORCH_AVAILABLE
except ImportError:
    TORCH_AVAILABLE = False


@dataclass
class ScoringResult:
    """Combined result from both scoring tiers."""
    # Tier 1: Local model (always available if trained)
    local_score: Optional[float] = None
    local_latency_ms: float = 0.0

    # Tier 2: Claude agent (available when triggered)
    claude_score: Optional[float] = None
    claude_corrections: list[str] = field(default_factory=list)
    claude_latency_ms: float = 0.0

    # Combined
    final_score: float = 0.0
    source: str = "none"  # "local", "claude", "hybrid"
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return {
            "final_score": round(self.final_score, 1),
            "source": self.source,
            "local_score": round(self.local_score, 1) if self.local_score is not None else None,
            "local_latency_ms": round(self.local_latency_ms, 2),
            "claude_score": round(self.claude_score, 1) if self.claude_score is not None else None,
            "claude_corrections": self.claude_corrections,
            "claude_latency_ms": round(self.claude_latency_ms, 2),
            "timestamp": self.timestamp,
        }


class HybridScorer:
    """Two-tier scoring: instant local model + deep Claude analysis."""

    def __init__(self):
        self.local_model: Optional[SkillScorer] = None
        self.data_collector = DataCollector()

        # Configuration
        self.claude_interval_sec = 5.0  # How often to trigger Claude analysis
        self.score_threshold_for_claude = 15.0  # Trigger Claude if score drops by this much
        self.blend_weight_local = 0.6  # Weight for local model in hybrid score
        self.blend_weight_claude = 0.4  # Weight for Claude in hybrid score

        # State
        self._last_claude_time = 0.0
        self._last_local_score = None
        self._scores_buffer: list[float] = []
        self._samples_since_retrain = 0
        self._train_count = 0
        self.retrain_threshold = 20  # auto-retrain after this many new reps

        # Try to load existing model
        self._try_load_model()

    def _try_load_model(self):
        """Try to load a pre-trained model. Prefers PyTorch (GPU), falls back to NumPy (CPU)."""
        # Try PyTorch first (better quality, GPU-accelerated)
        if TORCH_AVAILABLE:
            try:
                torch_model = SkillScorerTorch.load()
                if torch_model and torch_model.trained:
                    self._torch_model = torch_model
                    self.local_model = None  # Use torch instead
                    return
            except Exception:
                pass

        # Fall back to NumPy scorer
        model = SkillScorer.load()
        if model and model.trained:
            self.local_model = model

    @property
    def local_model_available(self) -> bool:
        has_numpy = self.local_model is not None and self.local_model.trained
        has_torch = hasattr(self, '_torch_model') and self._torch_model is not None
        return has_numpy or has_torch

    def score_local(self, angles_sequence: list[list[float]]) -> Optional[float]:
        """Tier 1: Get instant score from local model.

        Prefers PyTorch model (GPU) if available, falls back to NumPy (CPU).

        Args:
            angles_sequence: List of frames, each frame is 10 joint angles.

        Returns:
            Score (0-100) or None if model not trained.
        """
        if not self.local_model_available:
            return None

        start = time.time()

        # Prefer PyTorch model
        if hasattr(self, '_torch_model') and self._torch_model is not None:
            score = self._torch_model.predict(angles_sequence)
        elif self.local_model is not None:
            score = self.local_model.predict(angles_sequence)
        else:
            return None

        latency = (time.time() - start) * 1000

        self._last_local_score = score
        self._scores_buffer.append(score)
        if len(self._scores_buffer) > 30:
            self._scores_buffer = self._scores_buffer[-30:]

        return score

    def should_trigger_claude(self) -> bool:
        """Decide if Claude analysis should be triggered.

        Triggers when:
        1. Enough time has passed since last Claude call
        2. Score dropped significantly (potential form breakdown)
        3. Score is consistently low (needs coaching intervention)
        """
        now = time.time()
        elapsed = now - self._last_claude_time

        # Time-based trigger
        if elapsed >= self.claude_interval_sec:
            return True

        # Score-drop trigger
        if len(self._scores_buffer) >= 5:
            recent = self._scores_buffer[-3:]
            earlier = self._scores_buffer[-6:-3]
            if earlier:
                recent_avg = sum(recent) / len(recent)
                earlier_avg = sum(earlier) / len(earlier)
                if earlier_avg - recent_avg > self.score_threshold_for_claude:
                    return True

        return False

    def record_claude_result(self, score: float, corrections: list[str],
                             latency_ms: float):
        """Record the result of a Claude analysis (called by the agent)."""
        self._last_claude_time = time.time()
        self._last_claude_score = score
        self._last_claude_corrections = corrections
        self._last_claude_latency = latency_ms

    def get_hybrid_score(self, angles_sequence: list[list[float]],
                         claude_score: float = None,
                         claude_corrections: list[str] = None) -> ScoringResult:
        """Get combined score from both tiers.

        Args:
            angles_sequence: Current angle sequence for local scoring
            claude_score: Optional score from Claude (if recently computed)
            claude_corrections: Optional corrections from Claude

        Returns:
            ScoringResult with blended score and metadata.
        """
        result = ScoringResult()

        # Tier 1: Local model
        start = time.time()
        local_score = self.score_local(angles_sequence)
        result.local_latency_ms = (time.time() - start) * 1000
        result.local_score = local_score

        # Tier 2: Claude (if provided)
        result.claude_score = claude_score
        result.claude_corrections = claude_corrections or []

        # Blend scores
        if local_score is not None and claude_score is not None:
            result.final_score = (
                self.blend_weight_local * local_score +
                self.blend_weight_claude * claude_score
            )
            result.source = "hybrid"
        elif local_score is not None:
            result.final_score = local_score
            result.source = "local"
        elif claude_score is not None:
            result.final_score = claude_score
            result.source = "claude"
        else:
            result.final_score = 0
            result.source = "none"

        return result

    def collect_rep(self, skill: str, skeleton_frames: list,
                     score: float, corrections: list[str],
                     deviations: dict[str, float],
                     reference_name: str = "") -> dict:
        """Auto-collect a rep from a coaching session into the training pipeline.

        This is the bridge between CoachingSession and DataCollector.
        Call this after every completed rep during coaching.
        """
        result = self.data_collector.save_rep_from_session(
            skill=skill,
            skeleton_frames=skeleton_frames,
            score=score,
            corrections=corrections,
            deviations=deviations,
            reference_name=reference_name,
        )
        self._samples_since_retrain += 1

        # Auto-retrain trigger: every N new samples
        should_retrain = (
            self._samples_since_retrain >= self.retrain_threshold
            and self.data_collector.get_dataset_stats()["total_samples"] >= 10
        )
        return {
            "saved": True,
            "samples_since_retrain": self._samples_since_retrain,
            "should_retrain": should_retrain,
        }

    def train_from_collected_data(self, skill: str = None,
                                  epochs: int = 50,
                                  use_gpu: bool = True,
                                  augment: bool = True) -> dict:
        """Train (or retrain) the local model from collected coaching data.

        Uses augmented data by default for robustness.
        Uses PyTorch (GPU/MPS) when available for better quality.
        Falls back to NumPy evolutionary strategy on CPU.

        Args:
            skill: Train on specific skill data, or all if None
            epochs: Training epochs
            use_gpu: If True and PyTorch available, use GPU training
            augment: If True, apply data augmentation (5x multiplier)

        Returns:
            Training summary dict.
        """
        if augment:
            export = self.data_collector.export_for_training_augmented(
                skill=skill, pad_to=60, augment_factor=5,
            )
        else:
            export = self.data_collector.export_for_training(skill=skill, pad_to=60)

        if export["n_samples"] < 5:
            return {
                "error": f"Need at least 5 samples, have {export['n_samples']}. "
                         "Complete more coaching sessions to collect data.",
                "hint": "Use generate_and_train() to bootstrap with synthetic data.",
            }

        result = self._train_model(export, epochs, use_gpu)
        self._samples_since_retrain = 0
        self._train_count += 1
        result["train_version"] = self._train_count
        result["n_real"] = export.get("n_real", export["n_samples"])
        result["n_augmented"] = export.get("n_augmented", 0)
        return result

    def generate_and_train(self, skill: str, n_synthetic: int = 50,
                           epochs: int = 80, use_gpu: bool = True) -> dict:
        """Bootstrap: generate synthetic data + train model in one call.

        Use this when you have zero real training data to get a working
        model immediately. The model will improve as real data comes in.
        """
        gen_result = self.data_collector.generate_synthetic_data(
            skill=skill, n_samples=n_synthetic,
        )

        # Train on synthetic + any existing real data (with augmentation)
        export = self.data_collector.export_for_training_augmented(
            skill=skill, pad_to=60, augment_factor=3,
        )

        if export["n_samples"] < 5:
            return {"error": "Synthetic generation failed", "gen_result": gen_result}

        train_result = self._train_model(export, epochs, use_gpu)
        self._train_count += 1

        return {
            "status": "bootstrapped",
            "synthetic_generated": gen_result["generated"],
            "total_training_samples": export["n_samples"],
            "train_version": self._train_count,
            **train_result,
        }

    def train_with_split(self, skill: str = None, epochs: int = 80,
                         use_gpu: bool = True) -> dict:
        """Train with proper train/val/test split for evaluation.

        Returns train, val, and test metrics separately.
        """
        split = self.data_collector.export_split(
            skill=skill, pad_to=60, augment_train=5,
        )
        if "error" in split:
            return split

        # Train on training set
        train_export = {
            "X": split["train"]["X"],
            "y": split["train"]["y"],
            "n_samples": split["train"]["n"],
            "n_features": split["n_features"],
            "seq_length": split["seq_length"],
        }
        train_result = self._train_model(train_export, epochs, use_gpu)

        # Evaluate on val and test sets
        val_metrics = self._evaluate(split["val"]["X"], split["val"]["y"])
        test_metrics = self._evaluate(split["test"]["X"], split["test"]["y"])

        self._train_count += 1
        return {
            "status": "trained_with_split",
            "train_version": self._train_count,
            "train": {"n": split["train"]["n"], **train_result},
            "val": {"n": split["val"]["n"], **val_metrics},
            "test": {"n": split["test"]["n"], **test_metrics},
            "n_real_samples": split["n_real"],
        }

    def _train_model(self, export: dict, epochs: int, use_gpu: bool) -> dict:
        """Internal: train model on prepared data."""
        # Prefer PyTorch training (GPU-accelerated, proper backprop)
        if use_gpu and TORCH_AVAILABLE:
            try:
                torch_model = SkillScorerTorch(
                    seq_len=export["seq_length"],
                    n_features=export["n_features"],
                )
                result = torch_model.train(
                    export["X"], export["y"],
                    epochs=epochs, verbose=False,
                )
                filepath = torch_model.save()
                self._torch_model = torch_model
                self.local_model = None
                result["model_saved"] = filepath
                result["backend"] = "pytorch"
                result["status"] = "trained"
                return result
            except Exception as e:
                print(f"[HybridScorer] PyTorch training failed: {e}, falling back to NumPy")

        # Fallback: NumPy evolutionary strategy (CPU, no dependencies)
        model = SkillScorer(seq_len=export["seq_length"], n_features=export["n_features"])
        result = model.train(export["X"], export["y"], epochs=epochs, verbose=False)
        filepath = model.save()
        self.local_model = model

        result["model_saved"] = filepath
        result["backend"] = "numpy"
        result["status"] = "trained"
        return result

    def _evaluate(self, X: list, y: list) -> dict:
        """Evaluate the current model on a dataset."""
        if not X:
            return {"rmse": 0, "mae": 0, "correlation": 0}

        preds = []
        for seq in X:
            preds.append(self.score_local(seq) or 0.0)

        preds_arr = np.array(preds)
        y_arr = np.array(y)
        rmse = float(np.sqrt(np.mean((preds_arr - y_arr) ** 2)))
        mae = float(np.mean(np.abs(preds_arr - y_arr)))
        corr = float(np.corrcoef(preds_arr, y_arr)[0, 1]) if len(y) > 1 else 0.0

        return {
            "rmse": round(rmse, 2),
            "mae": round(mae, 2),
            "correlation": round(corr, 3),
        }

    def get_status(self) -> dict:
        """Get hybrid scorer status for API."""
        torch_status = None
        if hasattr(self, '_torch_model') and self._torch_model is not None:
            torch_status = self._torch_model.get_status()
        return {
            "local_model_available": self.local_model_available,
            "numpy_model": self.local_model.get_status() if self.local_model else None,
            "torch_model": torch_status,
            "torch_available": TORCH_AVAILABLE,
            "claude_interval_sec": self.claude_interval_sec,
            "blend_weights": {
                "local": self.blend_weight_local,
                "claude": self.blend_weight_claude,
            },
            "training_data": self.data_collector.get_dataset_stats(),
        }
