# Deep Generative Models Lab

This repository is a compact learning lab for deep generative modeling. It focuses on small, inspectable experiments that make the core ideas visible before scaling them to image models.

## What This Covers

- Energy-based modeling on 2D toy data.
- Langevin dynamics as a sampler for learned or hand-built energies.
- Normalizing-flow intuition through invertible affine coupling layers.
- Experiment hygiene: deterministic seeds, metrics, and small tests.

## Repository Layout

- `src/dgm_lab/` reusable experiment code.
- `experiments/` runnable scripts for toy studies.
- `docs/` short learning notes and experiment guides.
- `tests/` smoke tests for the core math and sampling utilities.
- `reports/` retained written analysis from earlier explorations.

## Quick Start

```bash
py -m pip install -e .[dev]
py -m pytest
py experiments/run_energy_model.py --steps 200 --samples 256
```

The examples intentionally use NumPy-first implementations so the behavior is easy to read and modify. They are not optimized training pipelines.

## Learning Path

1. Read `docs/energy_based_models.md`.
2. Run the energy-model experiment and inspect the printed sample statistics.
3. Open `src/dgm_lab/flows.py` to trace how the affine coupling transform stays invertible.
4. Extend the toy data distribution or add a plotting script under `experiments/`.


