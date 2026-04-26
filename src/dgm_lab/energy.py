from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class GaussianMixtureEnergy:
    """Analytic 2D Gaussian-mixture energy used for sampling experiments."""

    centers: np.ndarray
    scale: float = 0.35

    @classmethod
    def four_modes(cls, radius: float = 2.0, scale: float = 0.35) -> "GaussianMixtureEnergy":
        centers = np.array(
            [[radius, radius], [radius, -radius], [-radius, radius], [-radius, -radius]],
            dtype=float,
        )
        return cls(centers=centers, scale=scale)

    def energy(self, x: np.ndarray) -> np.ndarray:
        x = np.atleast_2d(x).astype(float)
        sq_dist = np.sum((x[:, None, :] - self.centers[None, :, :]) ** 2, axis=-1)
        logits = -0.5 * sq_dist / (self.scale**2)
        max_logit = np.max(logits, axis=1, keepdims=True)
        log_prob = max_logit[:, 0] + np.log(np.mean(np.exp(logits - max_logit), axis=1))
        return -log_prob

    def score(self, x: np.ndarray) -> np.ndarray:
        """Return gradient of log density with respect to x."""

        x = np.atleast_2d(x).astype(float)
        diff = self.centers[None, :, :] - x[:, None, :]
        logits = -0.5 * np.sum(diff**2, axis=-1) / (self.scale**2)
        weights = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        weights = weights / np.sum(weights, axis=1, keepdims=True)
        return np.sum(weights[:, :, None] * diff, axis=1) / (self.scale**2)


def langevin_sample(
    energy_model: GaussianMixtureEnergy,
    num_samples: int,
    steps: int,
    step_size: float,
    seed: int = 0,
) -> np.ndarray:
    """Sample from an energy model using unadjusted Langevin dynamics."""

    rng = np.random.default_rng(seed)
    x = rng.normal(size=(num_samples, 2))
    noise_scale = np.sqrt(2.0 * step_size)

    for _ in range(steps):
        x = x + step_size * energy_model.score(x)
        x = x + noise_scale * rng.normal(size=x.shape)

    return x
