"""
AEGIS SkillScorer (PyTorch) — GPU-accelerated model for DGX/NVIDIA training.

This is the production model that replaces the NumPy prototype.
Designed to run on:
  - NVIDIA DGX Spark (training)
  - Apple MPS (development)
  - CPU (fallback)

Architecture:
    Input: (batch, seq_len, 10) — 10 joint angles over time
    Conv1D(10→64, k=5) → BatchNorm → ReLU → MaxPool
    Conv1D(64→128, k=3) → BatchNorm → ReLU → MaxPool
    Conv1D(128→64, k=3) → BatchNorm → ReLU → GlobalAvgPool
    Linear(64→32) → ReLU → Dropout(0.3) → Linear(32→1) → Sigmoid × 100

Self-improving loop:
    1. Coaching sessions collect data (angle sequences + Claude scores)
    2. Data exported via DataCollector.export_for_training()
    3. Model trained on DGX (or MPS/CPU)
    4. Trained model provides instant scoring (< 1ms)
    5. Better scoring → better coaching → more data → repeat
"""

import json
import os
import time
from typing import Optional

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


def get_device() -> str:
    """Get the best available device."""
    if not TORCH_AVAILABLE:
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class SkillScorerNet(nn.Module):
    """1D CNN for movement quality scoring."""

    def __init__(self, n_features: int = 10, seq_len: int = 60):
        super().__init__()
        self.seq_len = seq_len

        # Conv blocks (input: batch × seq_len × n_features → permuted to batch × n_features × seq_len)
        self.conv1 = nn.Sequential(
            nn.Conv1d(n_features, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2),
        )
        self.conv3 = nn.Sequential(
            nn.Conv1d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
        )

        # Global average pooling → dense
        self.classifier = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        """x: (batch, seq_len, n_features) → (batch,) scores in [0, 100]"""
        # Normalize input: angles are 0-180°, scale to 0-1
        x = x / 180.0

        # Permute to (batch, n_features, seq_len) for Conv1d
        x = x.permute(0, 2, 1)

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)

        # Global average pooling
        x = x.mean(dim=2)  # (batch, 64)

        # Classify
        x = self.classifier(x)  # (batch, 1)
        return x.squeeze(1) * 100.0  # Scale to 0-100


class SkillDataset(Dataset):
    """PyTorch dataset for skill scoring training data."""

    def __init__(self, X, y, seq_len: int = 60):
        self.seq_len = seq_len
        self.X = []
        self.y = []

        for seq, score in zip(X, y):
            # Pad or truncate
            if len(seq) > seq_len:
                seq = seq[:seq_len]
            elif len(seq) < seq_len:
                last = seq[-1] if seq else [0.0] * 10
                seq = seq + [last] * (seq_len - len(seq))
            self.X.append(seq)
            self.y.append(score)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.X[idx], dtype=torch.float32),
            torch.tensor(self.y[idx], dtype=torch.float32),
        )


class SkillScorerTorch:
    """GPU-accelerated SkillScorer with proper training."""

    def __init__(self, seq_len: int = 60, n_features: int = 10):
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch not installed. Install with: pip install torch")

        self.seq_len = seq_len
        self.n_features = n_features
        self.device = get_device()
        self.model = SkillScorerNet(n_features, seq_len).to(self.device)
        self.trained = False
        self.metadata: dict = {}
        self.train_history: list[dict] = []

    def train(self, X: list, y: list, epochs: int = 100,
              lr: float = 0.001, batch_size: int = 16,
              val_split: float = 0.2, verbose: bool = True) -> dict:
        """Train the model with proper backprop + validation.

        Args:
            X: List of angle sequences (N, T, 10)
            y: List of target scores (N,)
            epochs: Training epochs
            lr: Learning rate
            batch_size: Batch size
            val_split: Fraction for validation
            verbose: Print progress

        Returns:
            Training summary dict.
        """
        import numpy as np

        n = len(X)
        if n < 5:
            return {"error": f"Need at least 5 samples, have {n}"}

        # Split train/val
        indices = np.random.permutation(n)
        val_n = max(1, int(n * val_split))
        val_idx = indices[:val_n]
        train_idx = indices[val_n:]

        train_X = [X[i] for i in train_idx]
        train_y = [y[i] for i in train_idx]
        val_X = [X[i] for i in val_idx]
        val_y = [y[i] for i in val_idx]

        train_ds = SkillDataset(train_X, train_y, self.seq_len)
        val_ds = SkillDataset(val_X, val_y, self.seq_len)
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=batch_size)

        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
        criterion = nn.MSELoss()

        best_val_loss = float('inf')
        best_state = None
        self.train_history = []

        start_time = time.time()

        for epoch in range(epochs):
            # Training
            self.model.train()
            train_loss = 0.0
            for batch_X, batch_y in train_loader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)

                optimizer.zero_grad()
                preds = self.model(batch_X)
                loss = criterion(preds, batch_y)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            train_loss /= len(train_loader)

            # Validation
            self.model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    batch_X = batch_X.to(self.device)
                    batch_y = batch_y.to(self.device)
                    preds = self.model(batch_X)
                    val_loss += criterion(preds, batch_y).item()
            val_loss /= max(len(val_loader), 1)

            scheduler.step(val_loss)

            self.train_history.append({
                "epoch": epoch + 1,
                "train_loss": round(train_loss, 4),
                "val_loss": round(val_loss, 4),
            })

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}

            if verbose and (epoch + 1) % 10 == 0:
                print(f"  Epoch {epoch+1}/{epochs}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")

        # Restore best model
        if best_state:
            self.model.load_state_dict(best_state)
            self.model.to(self.device)

        self.trained = True
        train_time = time.time() - start_time

        # Compute final metrics
        import numpy as np
        all_preds = self.predict_batch(X)
        y_arr = np.array(y)
        rmse = float(np.sqrt(np.mean((all_preds - y_arr) ** 2)))
        corr = float(np.corrcoef(all_preds, y_arr)[0, 1]) if len(y) > 1 else 0.0

        self.metadata = {
            "trained_at": time.time(),
            "device": self.device,
            "n_samples": n,
            "n_train": len(train_idx),
            "n_val": val_n,
            "epochs": epochs,
            "best_val_loss": round(best_val_loss, 4),
            "final_rmse": round(rmse, 2),
            "correlation": round(corr, 3),
            "train_time_sec": round(train_time, 1),
            "param_count": sum(p.numel() for p in self.model.parameters()),
        }
        return self.metadata

    def predict(self, angles_sequence: list[list[float]]) -> float:
        """Predict score for a single sequence."""
        self.model.eval()
        # Pad/truncate
        seq = list(angles_sequence)
        if len(seq) > self.seq_len:
            seq = seq[:self.seq_len]
        elif len(seq) < self.seq_len:
            last = seq[-1] if seq else [0.0] * self.n_features
            seq = seq + [last] * (self.seq_len - len(seq))

        x = torch.tensor([seq], dtype=torch.float32).to(self.device)
        with torch.no_grad():
            score = self.model(x)
        return float(score[0].cpu())

    def predict_batch(self, sequences: list) -> 'numpy.ndarray':
        """Predict scores for a batch of sequences."""
        import numpy as np
        ds = SkillDataset(sequences, [0.0] * len(sequences), self.seq_len)
        loader = DataLoader(ds, batch_size=32)
        self.model.eval()
        all_preds = []
        with torch.no_grad():
            for batch_X, _ in loader:
                batch_X = batch_X.to(self.device)
                preds = self.model(batch_X)
                all_preds.extend(preds.cpu().numpy().tolist())
        return np.array(all_preds)

    def save(self, filepath: str = None) -> str:
        """Save model to disk."""
        if filepath is None:
            model_dir = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                "data", "models"
            )
            os.makedirs(model_dir, exist_ok=True)
            filepath = os.path.join(model_dir, "skill_scorer_torch.pt")

        torch.save({
            "model_state": self.model.state_dict(),
            "metadata": self.metadata,
            "train_history": self.train_history,
            "seq_len": self.seq_len,
            "n_features": self.n_features,
        }, filepath)
        return filepath

    @classmethod
    def load(cls, filepath: str = None) -> Optional["SkillScorerTorch"]:
        """Load model from disk."""
        if not TORCH_AVAILABLE:
            return None

        if filepath is None:
            filepath = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                "data", "models", "skill_scorer_torch.pt"
            )

        if not os.path.exists(filepath):
            return None

        data = torch.load(filepath, map_location="cpu", weights_only=False)
        scorer = cls(seq_len=data["seq_len"], n_features=data["n_features"])
        scorer.model.load_state_dict(data["model_state"])
        scorer.model.to(scorer.device)
        scorer.metadata = data.get("metadata", {})
        scorer.train_history = data.get("train_history", [])
        scorer.trained = True
        return scorer

    def get_status(self) -> dict:
        return {
            "torch_available": TORCH_AVAILABLE,
            "device": self.device,
            "trained": self.trained,
            "metadata": self.metadata,
            "param_count": sum(p.numel() for p in self.model.parameters()),
        }
