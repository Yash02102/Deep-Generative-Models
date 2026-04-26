import numpy as np

from dgm_lab import GaussianMixtureEnergy, langevin_sample


def test_energy_is_lower_near_modes_than_origin():
    model = GaussianMixtureEnergy.four_modes()

    mode_energy = model.energy(np.array([[2.0, 2.0]]))[0]
    origin_energy = model.energy(np.array([[0.0, 0.0]]))[0]

    assert mode_energy < origin_energy


def test_langevin_sampler_shape_is_stable():
    model = GaussianMixtureEnergy.four_modes()
    samples = langevin_sample(model, num_samples=16, steps=5, step_size=0.01, seed=1)

    assert samples.shape == (16, 2)
    assert np.isfinite(samples).all()
