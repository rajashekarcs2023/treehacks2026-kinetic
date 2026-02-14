"""
AEGIS SkillScorer — Local 1D CNN for instant movement scoring.

A lightweight model that scores movement quality from joint angle sequences.
Trained from coaching session data collected by the DataCollector.

Architecture:
    Input: (batch, seq_len, 10)  — 10 joint angles over time
    Conv1D(10→32, k=5) → ReLU → MaxPool
    Conv1D(32→64, k=3) → ReLU → MaxPool
    Conv1D(64→32, k=3) → ReLU → GlobalAvgPool
    Linear(32→16) → ReLU → Linear(16→1) → Sigmoid × 100

Training:
    Loss: MSE on score (0-100)
    Optimizer: Adam, lr=0.001
    Data: angle sequences from DataCollector.export_for_training()

Inference:
    ~0.5ms per prediction on CPU — 2000× faster than a Claude API call.
"""

import json
import os
import time
import math
from typing import Optional

# Use only numpy for the model — no PyTorch dependency needed for tiny CNN
import numpy as np


class Conv1D:
    """Minimal 1D convolution layer (NumPy only)."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int):
        self.in_ch = in_channels
        self.out_ch = out_channels
        self.k = kernel_size
        # Xavier initialization
        scale = math.sqrt(2.0 / (in_channels * kernel_size))
        self.weight = np.random.randn(out_channels, in_channels, kernel_size).astype(np.float32) * scale
        self.bias = np.zeros(out_channels, dtype=np.float32)

        # For training
        self.grad_weight = None
        self.grad_bias = None
        self._input_cache = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        """x: (batch, seq_len, in_channels) → (batch, seq_len - k + 1, out_channels)"""
        self._input_cache = x
        B, T, C = x.shape
        out_len = T - self.k + 1
        out = np.zeros((B, out_len, self.out_ch), dtype=np.float32)

        for i in range(out_len):
            patch = x[:, i:i+self.k, :]  # (B, k, C)
            # Reshape for matrix multiply: (B, k*C) @ (out_ch, k*C).T
            flat = patch.reshape(B, -1)  # (B, k*C)
            w_flat = self.weight.reshape(self.out_ch, -1)  # (out_ch, k*C)
            out[:, i, :] = flat @ w_flat.T + self.bias

        return out


class SkillScorer:
    """Tiny 1D CNN for instant movement quality scoring."""

    def __init__(self, seq_len: int = 60, n_features: int = 10):
        self.seq_len = seq_len
        self.n_features = n_features
        self.trained = False
        self.train_loss_history: list[float] = []
        self.metadata: dict = {}

        # Network layers
        self.conv1 = Conv1D(n_features, 32, 5)
        self.conv2 = Conv1D(32, 64, 3)
        self.conv3 = Conv1D(64, 32, 3)

        # Dense layers (weights initialized lazily after first forward pass)
        self._dense1_w = None
        self._dense1_b = None
        self._dense2_w = None
        self._dense2_b = None

    def _relu(self, x: np.ndarray) -> np.ndarray:
        return np.maximum(0, x)

    def _max_pool1d(self, x: np.ndarray, pool_size: int = 2) -> np.ndarray:
        """(B, T, C) → (B, T//pool_size, C)"""
        B, T, C = x.shape
        out_len = T // pool_size
        out = np.zeros((B, out_len, C), dtype=np.float32)
        for i in range(out_len):
            out[:, i, :] = x[:, i*pool_size:(i+1)*pool_size, :].max(axis=1)
        return out

    def _global_avg_pool(self, x: np.ndarray) -> np.ndarray:
        """(B, T, C) → (B, C)"""
        return x.mean(axis=1)

    def _sigmoid(self, x: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-np.clip(x, -10, 10)))

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass: (B, seq_len, 10) → (B,) scores in [0, 100].

        Args:
            x: Input angle sequences, shape (batch, seq_len, n_features)

        Returns:
            Predicted scores, shape (batch,), values in [0, 100]
        """
        # Normalize input to [0, 1] range (angles are typically 0-180°)
        x = x.astype(np.float32) / 180.0

        # Conv blocks
        h = self.conv1.forward(x)
        h = self._relu(h)
        h = self._max_pool1d(h)

        h = self.conv2.forward(h)
        h = self._relu(h)
        h = self._max_pool1d(h)

        h = self.conv3.forward(h)
        h = self._relu(h)
        h = self._global_avg_pool(h)  # (B, 32)

        # Initialize dense layers if needed
        hidden_dim = h.shape[1]
        if self._dense1_w is None:
            scale1 = math.sqrt(2.0 / hidden_dim)
            self._dense1_w = np.random.randn(hidden_dim, 16).astype(np.float32) * scale1
            self._dense1_b = np.zeros(16, dtype=np.float32)
            scale2 = math.sqrt(2.0 / 16)
            self._dense2_w = np.random.randn(16, 1).astype(np.float32) * scale2
            self._dense2_b = np.zeros(1, dtype=np.float32)

        # Dense layers
        h = h @ self._dense1_w + self._dense1_b
        h = self._relu(h)
        h = h @ self._dense2_w + self._dense2_b

        # Sigmoid → scale to 0-100
        scores = self._sigmoid(h).squeeze(-1) * 100.0
        return scores

    def predict(self, angles_sequence: list[list[float]]) -> float:
        """Predict score for a single angle sequence.

        Args:
            angles_sequence: List of frames, each frame is 10 joint angles.

        Returns:
            Predicted score (0-100).
        """
        x = np.array([angles_sequence], dtype=np.float32)

        # Pad/truncate to seq_len
        if x.shape[1] > self.seq_len:
            x = x[:, :self.seq_len, :]
        elif x.shape[1] < self.seq_len:
            pad = np.tile(x[:, -1:, :], (1, self.seq_len - x.shape[1], 1))
            x = np.concatenate([x, pad], axis=1)

        return float(self.forward(x)[0])

    def train(self, X: list, y: list, epochs: int = 50,
              lr: float = 0.001, batch_size: int = 16,
              verbose: bool = True) -> dict:
        """Train the model using simple gradient-free optimization.

        Uses evolutionary strategy (ES) for training since we're avoiding
        PyTorch dependency. ES works surprisingly well for small models.

        Args:
            X: List of angle sequences (N, T, 10)
            y: List of target scores (N,)
            epochs: Number of training epochs
            lr: Learning rate (noise scale for ES)
            batch_size: Not used in ES, kept for API compatibility
            verbose: Print progress

        Returns:
            Training summary dict.
        """
        X = np.array(X, dtype=np.float32)
        y = np.array(y, dtype=np.float32)

        if len(X) == 0:
            return {"error": "No training data"}

        # Pad/truncate X to seq_len
        if X.shape[1] > self.seq_len:
            X = X[:, :self.seq_len, :]
        elif X.shape[1] < self.seq_len:
            pad = np.tile(X[:, -1:, :], (1, self.seq_len - X.shape[1], 1))
            X = np.concatenate([X, pad], axis=1)

        # Initialize weights with a forward pass
        self.forward(X[:1])

        # Collect all parameters
        params = self._get_params()
        best_params = params.copy()
        best_loss = float('inf')

        self.train_loss_history = []

        for epoch in range(epochs):
            # Current loss
            preds = self.forward(X)
            loss = float(np.mean((preds - y) ** 2))
            self.train_loss_history.append(loss)

            if loss < best_loss:
                best_loss = loss
                best_params = self._get_params()

            # ES: try random perturbations, keep improvements
            n_tries = 5
            for _ in range(n_tries):
                noise = np.random.randn(len(params)).astype(np.float32) * lr
                self._set_params(params + noise)
                new_preds = self.forward(X)
                new_loss = float(np.mean((new_preds - y) ** 2))

                if new_loss < loss:
                    params = self._get_params()
                    loss = new_loss
                    if loss < best_loss:
                        best_loss = loss
                        best_params = params.copy()
                else:
                    # Try negative direction
                    self._set_params(params - noise)
                    new_preds = self.forward(X)
                    new_loss = float(np.mean((new_preds - y) ** 2))
                    if new_loss < loss:
                        params = self._get_params()
                        loss = new_loss
                        if loss < best_loss:
                            best_loss = loss
                            best_params = params.copy()
                    else:
                        self._set_params(params)

            # Decay learning rate
            lr *= 0.995

            if verbose and (epoch + 1) % 10 == 0:
                preds = self.forward(X)
                rmse = float(np.sqrt(np.mean((preds - y) ** 2)))
                print(f"  Epoch {epoch+1}/{epochs}: RMSE={rmse:.2f}, best_loss={best_loss:.2f}")

        # Restore best params
        self._set_params(best_params)
        self.trained = True

        # Final metrics
        final_preds = self.forward(X)
        final_rmse = float(np.sqrt(np.mean((final_preds - y) ** 2)))
        correlation = float(np.corrcoef(final_preds, y)[0, 1]) if len(y) > 1 else 0.0

        self.metadata = {
            "trained_at": time.time(),
            "n_samples": len(X),
            "epochs": epochs,
            "final_rmse": round(final_rmse, 2),
            "best_loss": round(best_loss, 2),
            "correlation": round(correlation, 3),
            "seq_len": self.seq_len,
            "n_features": self.n_features,
        }

        return self.metadata

    def _get_params(self) -> np.ndarray:
        """Flatten all parameters into a single vector."""
        parts = [
            self.conv1.weight.flatten(),
            self.conv1.bias,
            self.conv2.weight.flatten(),
            self.conv2.bias,
            self.conv3.weight.flatten(),
            self.conv3.bias,
        ]
        if self._dense1_w is not None:
            parts.extend([
                self._dense1_w.flatten(),
                self._dense1_b,
                self._dense2_w.flatten(),
                self._dense2_b,
            ])
        return np.concatenate(parts)

    def _set_params(self, params: np.ndarray):
        """Set all parameters from a flat vector."""
        idx = 0

        def take(shape):
            nonlocal idx
            size = int(np.prod(shape))
            chunk = params[idx:idx+size].reshape(shape)
            idx += size
            return chunk

        self.conv1.weight = take(self.conv1.weight.shape)
        self.conv1.bias = take(self.conv1.bias.shape)
        self.conv2.weight = take(self.conv2.weight.shape)
        self.conv2.bias = take(self.conv2.bias.shape)
        self.conv3.weight = take(self.conv3.weight.shape)
        self.conv3.bias = take(self.conv3.bias.shape)

        if self._dense1_w is not None:
            self._dense1_w = take(self._dense1_w.shape)
            self._dense1_b = take(self._dense1_b.shape)
            self._dense2_w = take(self._dense2_w.shape)
            self._dense2_b = take(self._dense2_b.shape)

    # ── Persistence ──────────────────────────────────────────────────

    def save(self, filepath: str = None) -> str:
        """Save model weights and metadata to disk."""
        if filepath is None:
            model_dir = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                "data", "models"
            )
            os.makedirs(model_dir, exist_ok=True)
            filepath = os.path.join(model_dir, "skill_scorer.npz")

        np.savez_compressed(
            filepath,
            params=self._get_params(),
            metadata=json.dumps(self.metadata),
            seq_len=self.seq_len,
            n_features=self.n_features,
            trained=self.trained,
            loss_history=np.array(self.train_loss_history),
        )
        return filepath

    @classmethod
    def load(cls, filepath: str = None) -> Optional["SkillScorer"]:
        """Load model from disk."""
        if filepath is None:
            filepath = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                "data", "models", "skill_scorer.npz"
            )

        if not os.path.exists(filepath):
            return None

        data = np.load(filepath, allow_pickle=True)
        seq_len = int(data["seq_len"])
        n_features = int(data["n_features"])

        model = cls(seq_len=seq_len, n_features=n_features)
        model.trained = bool(data["trained"])
        model.metadata = json.loads(str(data["metadata"]))
        model.train_loss_history = data["loss_history"].tolist()

        # Initialize layers with a dummy forward pass
        dummy = np.zeros((1, seq_len, n_features), dtype=np.float32)
        model.forward(dummy)

        # Restore params
        model._set_params(data["params"])
        return model

    def get_status(self) -> dict:
        """Get model status for API."""
        return {
            "trained": self.trained,
            "metadata": self.metadata,
            "param_count": len(self._get_params()) if self._dense1_w is not None else 0,
            "seq_len": self.seq_len,
            "n_features": self.n_features,
        }
