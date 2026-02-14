"""
AEGIS ML Activity Classifier — Temporal 1D CNN on Joint Angles.

A proper learned model that classifies activities from pose keypoint
sequences, replacing fragile hand-crafted heuristics.

Architecture:
    Input: (batch, window_size, n_features)
        - window_size = 15 frames (~1 second at 15 FPS)
        - n_features = 13 (10 joint angles + speed + torso_angle + body_ratio)

    Conv1D(13→64, k=3) → BatchNorm → ReLU → MaxPool
    Conv1D(64→128, k=3) → BatchNorm → ReLU → GlobalAvgPool
    Linear(128→64) → ReLU → Dropout(0.3) → Linear(64→n_classes)

Activities:
    0: standing, 1: sitting, 2: walking, 3: running,
    4: exercising, 5: fallen, 6: lying_down, 7: waving,
    8: reaching, 9: crouching, 10: unknown

Self-improving loop:
    1. Heuristic classifier labels frames during initial use
    2. Labels + angles stored as training data
    3. Model trained → replaces heuristics for smoother predictions
    4. Model predictions + user corrections → better data → retrain

Integration:
    - Plugs into ActivityClassifier as a "ml_mode" alongside heuristic fallback
    - Uses same temporal smoothing and hysteresis on top of model predictions
    - Falls back to heuristics if model not trained or confidence too low
"""

import json
import math
import os
import random
import time
from typing import Optional

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


# ═══════════════════════════════════════════════════════════════════════
# Activity labels
# ═══════════════════════════════════════════════════════════════════════

ACTIVITY_LABELS = [
    "standing", "sitting", "walking", "running", "exercising",
    "fallen", "lying_down", "waving", "reaching", "crouching", "unknown",
]
LABEL_TO_IDX = {label: i for i, label in enumerate(ACTIVITY_LABELS)}
N_CLASSES = len(ACTIVITY_LABELS)

# Features extracted per frame
FEATURE_NAMES = [
    "left_elbow", "right_elbow",
    "left_shoulder", "right_shoulder",
    "left_hip", "right_hip",
    "left_knee", "right_knee",
    "left_ankle", "right_ankle",
    "speed",          # px/s, normalized
    "torso_angle",    # degrees from vertical
    "body_ratio",     # height/width of bounding box
]
N_FEATURES = len(FEATURE_NAMES)
WINDOW_SIZE = 15  # ~1 second at 15 FPS


# ═══════════════════════════════════════════════════════════════════════
# Feature extraction from TrackedPerson
# ═══════════════════════════════════════════════════════════════════════

def extract_features(person) -> list[float]:
    """Extract a feature vector from a TrackedPerson for ML classification.

    Returns a list of N_FEATURES floats. Handles missing pose gracefully.
    """
    from aegis.pose_comparison import compute_joint_angles

    features = [0.0] * N_FEATURES
    pose = person.pose

    # Joint angles (0-9)
    if pose and hasattr(pose, 'points') and len(pose.points) >= 29:
        try:
            angles = compute_joint_angles(pose.points)
            angle_keys = [
                "left_elbow", "right_elbow",
                "left_shoulder", "right_shoulder",
                "left_hip", "right_hip",
                "left_knee", "right_knee",
                "left_ankle", "right_ankle",
            ]
            for i, key in enumerate(angle_keys):
                features[i] = angles.get(key, 0.0) / 180.0  # normalize to [0, 1]
        except Exception:
            pass

    # Speed (10) — normalize to [0, 1] with soft cap at 500 px/s
    features[10] = min(person.speed / 500.0, 1.0) if person.speed else 0.0

    # Torso angle (11) — angle from vertical in [0, 1] where 0=upright, 1=horizontal
    if pose and hasattr(pose, 'points') and len(pose.points) >= 25:
        pts = pose.points
        l_sh = pts[11] if len(pts) > 11 else (0, 0, 0)
        r_sh = pts[12] if len(pts) > 12 else (0, 0, 0)
        l_hip = pts[23] if len(pts) > 23 else (0, 0, 0)
        r_hip = pts[24] if len(pts) > 24 else (0, 0, 0)
        if l_sh[2] > 0.3 and r_sh[2] > 0.3 and l_hip[2] > 0.3 and r_hip[2] > 0.3:
            sh_mid = ((l_sh[0] + r_sh[0]) / 2, (l_sh[1] + r_sh[1]) / 2)
            hip_mid = ((l_hip[0] + r_hip[0]) / 2, (l_hip[1] + r_hip[1]) / 2)
            dx = abs(sh_mid[0] - hip_mid[0])
            dy = abs(sh_mid[1] - hip_mid[1]) + 1e-6
            torso_angle = math.degrees(math.atan2(dx, dy)) / 90.0  # normalize
            features[11] = min(torso_angle, 1.0)

    # Body ratio (12) — bbox height/width, tall=standing, wide=lying
    bbox = person.bbox
    if hasattr(bbox, 'height') and hasattr(bbox, 'width') and bbox.width > 0:
        ratio = bbox.height / max(bbox.width, 1)
        features[12] = min(ratio / 3.0, 1.0)  # normalize, typical standing ~2.0
    elif hasattr(bbox, '__len__') and len(bbox) == 4:
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        features[12] = min(h / max(w, 1) / 3.0, 1.0)

    return features


# ═══════════════════════════════════════════════════════════════════════
# PyTorch Model
# ═══════════════════════════════════════════════════════════════════════

if TORCH_AVAILABLE:
    class ActivityNet(nn.Module):
        """Temporal 1D CNN for activity classification from joint angle sequences."""

        def __init__(self, n_features: int = N_FEATURES, n_classes: int = N_CLASSES,
                     window_size: int = WINDOW_SIZE):
            super().__init__()
            self.window_size = window_size

            self.conv1 = nn.Sequential(
                nn.Conv1d(n_features, 64, kernel_size=3, padding=1),
                nn.BatchNorm1d(64),
                nn.ReLU(),
                nn.MaxPool1d(2),
            )
            self.conv2 = nn.Sequential(
                nn.Conv1d(64, 128, kernel_size=3, padding=1),
                nn.BatchNorm1d(128),
                nn.ReLU(),
            )
            # Global average pooling → classifier
            self.classifier = nn.Sequential(
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(64, n_classes),
            )

        def forward(self, x):
            """x: (batch, window_size, n_features) → (batch, n_classes) logits"""
            # Permute to (batch, n_features, window_size) for Conv1d
            x = x.permute(0, 2, 1)
            x = self.conv1(x)
            x = self.conv2(x)
            x = x.mean(dim=2)  # global avg pool → (batch, 128)
            return self.classifier(x)

    class ActivityDataset(Dataset):
        """Dataset for activity classification training."""

        def __init__(self, X, y, window_size=WINDOW_SIZE):
            self.X = []
            self.y = []
            for seq, label in zip(X, y):
                if len(seq) > window_size:
                    seq = seq[:window_size]
                elif len(seq) < window_size:
                    last = seq[-1] if seq else [0.0] * N_FEATURES
                    seq = seq + [last] * (window_size - len(seq))
                self.X.append(seq)
                self.y.append(label)

        def __len__(self):
            return len(self.X)

        def __getitem__(self, idx):
            return (
                torch.tensor(self.X[idx], dtype=torch.float32),
                torch.tensor(self.y[idx], dtype=torch.long),
            )


# ═══════════════════════════════════════════════════════════════════════
# Synthetic Activity Data Generator
# ═══════════════════════════════════════════════════════════════════════

# Biomechanical profiles for each activity
_ACTIVITY_PROFILES = {
    "standing": {
        "angles": {"left_knee": 170, "right_knee": 170, "left_hip": 170, "right_hip": 170,
                    "left_elbow": 160, "right_elbow": 160, "left_shoulder": 30, "right_shoulder": 30,
                    "left_ankle": 90, "right_ankle": 90},
        "speed_range": (0, 30), "torso_angle": 0.05, "body_ratio": 0.65,
        "angle_noise": 8, "speed_noise": 10,
    },
    "sitting": {
        "angles": {"left_knee": 90, "right_knee": 90, "left_hip": 90, "right_hip": 90,
                    "left_elbow": 100, "right_elbow": 100, "left_shoulder": 30, "right_shoulder": 30,
                    "left_ankle": 90, "right_ankle": 90},
        "speed_range": (0, 10), "torso_angle": 0.1, "body_ratio": 0.4,
        "angle_noise": 12, "speed_noise": 5,
    },
    "walking": {
        "angles": {"left_knee": 155, "right_knee": 155, "left_hip": 160, "right_hip": 160,
                    "left_elbow": 140, "right_elbow": 140, "left_shoulder": 40, "right_shoulder": 40,
                    "left_ankle": 85, "right_ankle": 85},
        "speed_range": (60, 180), "torso_angle": 0.08, "body_ratio": 0.6,
        "angle_noise": 15, "speed_noise": 40,
    },
    "running": {
        "angles": {"left_knee": 140, "right_knee": 140, "left_hip": 150, "right_hip": 150,
                    "left_elbow": 90, "right_elbow": 90, "left_shoulder": 60, "right_shoulder": 60,
                    "left_ankle": 80, "right_ankle": 80},
        "speed_range": (200, 500), "torso_angle": 0.15, "body_ratio": 0.55,
        "angle_noise": 20, "speed_noise": 80,
    },
    "exercising": {
        "angles": {"left_knee": 100, "right_knee": 100, "left_hip": 110, "right_hip": 110,
                    "left_elbow": 80, "right_elbow": 80, "left_shoulder": 50, "right_shoulder": 50,
                    "left_ankle": 80, "right_ankle": 80},
        "speed_range": (10, 80), "torso_angle": 0.12, "body_ratio": 0.5,
        "angle_noise": 25, "speed_noise": 20,  # high angle variation = movement
    },
    "fallen": {
        "angles": {"left_knee": 150, "right_knee": 150, "left_hip": 160, "right_hip": 160,
                    "left_elbow": 130, "right_elbow": 130, "left_shoulder": 40, "right_shoulder": 40,
                    "left_ankle": 90, "right_ankle": 90},
        "speed_range": (0, 15), "torso_angle": 0.85, "body_ratio": 0.2,
        "angle_noise": 10, "speed_noise": 5,
    },
    "lying_down": {
        "angles": {"left_knee": 165, "right_knee": 165, "left_hip": 170, "right_hip": 170,
                    "left_elbow": 160, "right_elbow": 160, "left_shoulder": 20, "right_shoulder": 20,
                    "left_ankle": 90, "right_ankle": 90},
        "speed_range": (0, 5), "torso_angle": 0.9, "body_ratio": 0.15,
        "angle_noise": 5, "speed_noise": 2,
    },
    "waving": {
        "angles": {"left_knee": 170, "right_knee": 170, "left_hip": 170, "right_hip": 170,
                    "left_elbow": 160, "right_elbow": 60, "left_shoulder": 30, "right_shoulder": 150,
                    "left_ankle": 90, "right_ankle": 90},
        "speed_range": (0, 30), "torso_angle": 0.05, "body_ratio": 0.65,
        "angle_noise": 15, "speed_noise": 10,
    },
    "reaching": {
        "angles": {"left_knee": 170, "right_knee": 170, "left_hip": 170, "right_hip": 170,
                    "left_elbow": 170, "right_elbow": 170, "left_shoulder": 120, "right_shoulder": 30,
                    "left_ankle": 90, "right_ankle": 90},
        "speed_range": (0, 40), "torso_angle": 0.1, "body_ratio": 0.6,
        "angle_noise": 10, "speed_noise": 15,
    },
    "crouching": {
        "angles": {"left_knee": 70, "right_knee": 70, "left_hip": 80, "right_hip": 80,
                    "left_elbow": 120, "right_elbow": 120, "left_shoulder": 30, "right_shoulder": 30,
                    "left_ankle": 70, "right_ankle": 70},
        "speed_range": (0, 15), "torso_angle": 0.2, "body_ratio": 0.35,
        "angle_noise": 10, "speed_noise": 5,
    },
}

ANGLE_KEYS = [
    "left_elbow", "right_elbow", "left_shoulder", "right_shoulder",
    "left_hip", "right_hip", "left_knee", "right_knee",
    "left_ankle", "right_ankle",
]


def generate_activity_samples(n_per_class: int = 100,
                               window_size: int = WINDOW_SIZE) -> tuple[list, list]:
    """Generate synthetic training data for all activity classes.

    Returns (X, y) where:
        X: list of sequences, each (window_size, N_FEATURES)
        y: list of class indices

    Each sample has realistic temporal dynamics:
        - Smooth angle transitions (not random noise per frame)
        - Speed varies smoothly over the window
        - Augmentation: random offset, scale, noise level
    """
    X, y = [], []

    for activity, profile in _ACTIVITY_PROFILES.items():
        label_idx = LABEL_TO_IDX[activity]

        for _ in range(n_per_class):
            seq = []
            # Random variation for this sample
            angle_offset = {k: random.gauss(0, profile["angle_noise"] * 0.5) for k in ANGLE_KEYS}
            speed_base = random.uniform(*profile["speed_range"])
            torso_base = profile["torso_angle"] + random.gauss(0, 0.05)
            ratio_base = profile["body_ratio"] + random.gauss(0, 0.05)

            for t in range(window_size):
                frame = []
                # Temporal dynamics: smooth sinusoidal variation
                phase = math.sin(2 * math.pi * t / window_size * random.uniform(0.5, 2.0))

                # Joint angles (features 0-9)
                for key in ANGLE_KEYS:
                    base = profile["angles"][key]
                    val = base + angle_offset[key] + phase * profile["angle_noise"]
                    val = max(10, min(180, val))
                    frame.append(val / 180.0)  # normalized

                # Speed (feature 10)
                speed = speed_base + random.gauss(0, profile["speed_noise"]) + phase * profile["speed_noise"] * 0.5
                frame.append(min(max(speed, 0) / 500.0, 1.0))

                # Torso angle (feature 11)
                torso = torso_base + random.gauss(0, 0.03) + phase * 0.02
                frame.append(min(max(torso, 0), 1.0))

                # Body ratio (feature 12)
                ratio = ratio_base + random.gauss(0, 0.02)
                frame.append(min(max(ratio, 0.05), 1.0))

                seq.append(frame)

            X.append(seq)
            y.append(label_idx)

    # Shuffle
    combined = list(zip(X, y))
    random.shuffle(combined)
    X, y = zip(*combined)
    return list(X), list(y)


# ═══════════════════════════════════════════════════════════════════════
# Activity Model Trainer
# ═══════════════════════════════════════════════════════════════════════

class ActivityModelTrainer:
    """Trains and manages the ML activity classifier."""

    def __init__(self, data_dir: str = None):
        if data_dir is None:
            data_dir = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                "data", "activity"
            )
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)
        self.model: Optional['ActivityNet'] = None
        self.device = "cpu"
        self.trained = False
        self.metadata: dict = {}

        if TORCH_AVAILABLE:
            if torch.cuda.is_available():
                self.device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self.device = "mps"

    def generate_and_train(self, n_per_class: int = 150, epochs: int = 60,
                           lr: float = 0.001, batch_size: int = 32) -> dict:
        """Generate synthetic data and train the model end-to-end."""
        if not TORCH_AVAILABLE:
            return {"error": "PyTorch not available"}

        print(f"[ActivityModel] Generating {n_per_class} samples per class...")
        X, y = generate_activity_samples(n_per_class=n_per_class)

        return self.train(X, y, epochs=epochs, lr=lr, batch_size=batch_size)

    def train(self, X: list, y: list, epochs: int = 60,
              lr: float = 0.001, batch_size: int = 32,
              val_split: float = 0.15) -> dict:
        """Train the activity classifier.

        Args:
            X: List of sequences (N, window_size, N_FEATURES)
            y: List of class indices (N,)
        """
        if not TORCH_AVAILABLE:
            return {"error": "PyTorch not available"}

        n = len(X)
        if n < 20:
            return {"error": f"Need at least 20 samples, have {n}"}

        # Train/val split
        indices = list(range(n))
        random.shuffle(indices)
        val_n = max(1, int(n * val_split))
        val_idx = indices[:val_n]
        train_idx = indices[val_n:]

        train_ds = ActivityDataset([X[i] for i in train_idx], [y[i] for i in train_idx])
        val_ds = ActivityDataset([X[i] for i in val_idx], [y[i] for i in val_idx])
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=batch_size)

        # Create model
        self.model = ActivityNet().to(self.device)
        optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=8, factor=0.5)
        criterion = nn.CrossEntropyLoss()

        best_val_acc = 0.0
        best_state = None
        patience_counter = 0
        early_stop_patience = 15

        start_time = time.time()

        for epoch in range(epochs):
            # Train
            self.model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            for batch_X, batch_y in train_loader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)
                optimizer.zero_grad()
                logits = self.model(batch_X)
                loss = criterion(logits, batch_y)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
                train_correct += (logits.argmax(1) == batch_y).sum().item()
                train_total += len(batch_y)

            # Validate
            self.model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    batch_X = batch_X.to(self.device)
                    batch_y = batch_y.to(self.device)
                    logits = self.model(batch_X)
                    val_loss += criterion(logits, batch_y).item()
                    val_correct += (logits.argmax(1) == batch_y).sum().item()
                    val_total += len(batch_y)

            val_acc = val_correct / max(val_total, 1)
            scheduler.step(val_loss / max(len(val_loader), 1))

            # Early stopping
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= early_stop_patience:
                    print(f"  Early stop at epoch {epoch + 1}")
                    break

            if (epoch + 1) % 10 == 0:
                train_acc = train_correct / max(train_total, 1)
                print(f"  Epoch {epoch+1}/{epochs}: "
                      f"train_acc={train_acc:.3f}, val_acc={val_acc:.3f}, "
                      f"val_loss={val_loss/max(len(val_loader),1):.4f}")

        # Restore best
        if best_state:
            self.model.load_state_dict(best_state)
            self.model.to(self.device)

        self.trained = True
        train_time = time.time() - start_time

        # Compute per-class accuracy
        self.model.eval()
        class_correct = [0] * N_CLASSES
        class_total = [0] * N_CLASSES
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)
                preds = self.model(batch_X).argmax(1)
                for pred, label in zip(preds, batch_y):
                    class_total[label.item()] += 1
                    if pred.item() == label.item():
                        class_correct[label.item()] += 1

        per_class = {}
        for i, name in enumerate(ACTIVITY_LABELS):
            if class_total[i] > 0:
                per_class[name] = {
                    "accuracy": round(class_correct[i] / class_total[i], 3),
                    "n_samples": class_total[i],
                }

        self.metadata = {
            "trained_at": time.time(),
            "device": self.device,
            "n_samples": n,
            "n_train": len(train_idx),
            "n_val": val_n,
            "epochs_run": epoch + 1,
            "best_val_accuracy": round(best_val_acc, 4),
            "train_time_sec": round(train_time, 1),
            "per_class_accuracy": per_class,
            "n_classes": N_CLASSES,
            "param_count": sum(p.numel() for p in self.model.parameters()),
        }
        return self.metadata

    def predict(self, feature_window: list[list[float]]) -> tuple[str, float, dict]:
        """Predict activity from a window of feature vectors.

        Args:
            feature_window: List of N_FEATURES-length vectors (up to WINDOW_SIZE frames)

        Returns:
            (activity_name, confidence, all_probs_dict)
        """
        if not self.trained or self.model is None:
            return ("unknown", 0.0, {})

        # Pad/truncate
        seq = list(feature_window)
        if len(seq) > WINDOW_SIZE:
            seq = seq[-WINDOW_SIZE:]
        elif len(seq) < WINDOW_SIZE:
            last = seq[-1] if seq else [0.0] * N_FEATURES
            seq = [last] * (WINDOW_SIZE - len(seq)) + seq  # left-pad

        self.model.eval()
        x = torch.tensor([seq], dtype=torch.float32).to(self.device)
        with torch.no_grad():
            logits = self.model(x)
            probs = torch.softmax(logits, dim=1)[0]

        probs_np = probs.cpu().numpy()
        best_idx = int(probs_np.argmax())
        best_conf = float(probs_np[best_idx])
        activity = ACTIVITY_LABELS[best_idx]

        all_probs = {ACTIVITY_LABELS[i]: round(float(probs_np[i]), 3)
                     for i in range(N_CLASSES) if float(probs_np[i]) > 0.01}

        return (activity, best_conf, all_probs)

    def save(self, filepath: str = None) -> str:
        if not self.trained or self.model is None:
            return ""
        if filepath is None:
            filepath = os.path.join(self.data_dir, "activity_model.pt")
        torch.save({
            "model_state": self.model.state_dict(),
            "metadata": self.metadata,
        }, filepath)
        return filepath

    def load(self, filepath: str = None) -> bool:
        if not TORCH_AVAILABLE:
            return False
        if filepath is None:
            filepath = os.path.join(self.data_dir, "activity_model.pt")
        if not os.path.exists(filepath):
            return False
        data = torch.load(filepath, map_location="cpu", weights_only=False)
        self.model = ActivityNet().to(self.device)
        self.model.load_state_dict(data["model_state"])
        self.model.to(self.device)
        self.metadata = data.get("metadata", {})
        self.trained = True
        return True

    def get_status(self) -> dict:
        return {
            "trained": self.trained,
            "torch_available": TORCH_AVAILABLE,
            "device": self.device,
            "metadata": self.metadata,
            "n_classes": N_CLASSES,
            "activity_labels": ACTIVITY_LABELS,
        }


# ═══════════════════════════════════════════════════════════════════════
# Frame Buffer — accumulates features for windowed prediction
# ═══════════════════════════════════════════════════════════════════════

class ActivityFrameBuffer:
    """Per-person frame buffer for windowed ML prediction.

    Accumulates feature vectors and provides windowed predictions.
    """

    def __init__(self, window_size: int = WINDOW_SIZE):
        self.window_size = window_size
        self._buffers: dict[int, list[list[float]]] = {}  # track_id → frames
        self._last_seen: dict[int, float] = {}

    def add_frame(self, track_id: int, features: list[float]):
        """Add a feature vector for a person."""
        if track_id not in self._buffers:
            self._buffers[track_id] = []
        self._buffers[track_id].append(features)
        # Keep only last window_size * 2 frames (rolling buffer)
        if len(self._buffers[track_id]) > self.window_size * 2:
            self._buffers[track_id] = self._buffers[track_id][-self.window_size:]
        self._last_seen[track_id] = time.time()

    def get_window(self, track_id: int) -> list[list[float]]:
        """Get the most recent window of features for a person."""
        buf = self._buffers.get(track_id, [])
        return buf[-self.window_size:]

    def has_enough(self, track_id: int, min_frames: int = 5) -> bool:
        """Check if we have enough frames for a prediction."""
        return len(self._buffers.get(track_id, [])) >= min_frames

    def prune_stale(self, max_age_sec: float = 30.0):
        """Remove buffers for persons not seen recently."""
        now = time.time()
        stale = [tid for tid, t in self._last_seen.items() if now - t > max_age_sec]
        for tid in stale:
            self._buffers.pop(tid, None)
            self._last_seen.pop(tid, None)
