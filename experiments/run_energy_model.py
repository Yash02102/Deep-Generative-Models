from __future__ import annotations

import argparse

import numpy as np

from dgm_lab import GaussianMixtureEnergy, langevin_sample


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a toy energy-based sampling experiment.")
    parser.add_argument("--samples", type=int, default=512)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--step-size", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    model = GaussianMixtureEnergy.four_modes()
    samples = langevin_sample(model, args.samples, args.steps, args.step_size, args.seed)
    energies = model.energy(samples)

    print("Toy EBM sampling summary")
    print(f"samples: {len(samples)}")
    print(f"mean: {np.round(samples.mean(axis=0), 3).tolist()}")
    print(f"std: {np.round(samples.std(axis=0), 3).tolist()}")
    print(f"mean_energy: {float(np.mean(energies)):.3f}")


if __name__ == "__main__":
    main()
