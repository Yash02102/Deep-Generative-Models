import numpy as np

from dgm_lab import AffineCoupling1D


def test_affine_coupling_inverse_recovers_input():
    layer = AffineCoupling1D()
    x = np.array([[0.5, -1.2], [1.0, 0.3]])

    y, log_det = layer.forward(x)
    recovered, inv_log_det = layer.inverse(y)

    assert np.allclose(recovered, x)
    assert np.allclose(log_det + inv_log_det, 0.0)
