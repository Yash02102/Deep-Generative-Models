from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class AffineCoupling1D:
    """Minimal 2D affine coupling layer for normalizing-flow intuition."""

    scale: float = 0.75
    shift: float = 0.25

    def forward(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        x = np.atleast_2d(x).astype(float)
        y = x.copy()
        log_scale = self.scale * np.tanh(x[:, 0])
        y[:, 1] = x[:, 1] * np.exp(log_scale) + self.shift * x[:, 0]
        return y, log_scale

    def inverse(self, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        y = np.atleast_2d(y).astype(float)
        x = y.copy()
        log_scale = self.scale * np.tanh(y[:, 0])
        x[:, 1] = (y[:, 1] - self.shift * y[:, 0]) * np.exp(-log_scale)
        return x, -log_scale
