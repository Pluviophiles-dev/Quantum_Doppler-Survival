import numpy as np

from qdboundary.formulas import (
    gamma_max,
    gq_pure_loss,
    tmsv_pure_loss_qfi,
    wrapping_probability_gaussian,
)
from qdboundary.rayleigh import return_photons


def test_lossless_tmsv_qfi():
    Ns = 3.0
    assert np.isclose(tmsv_pure_loss_qfi(1.0, Ns), 4 * Ns * (Ns + 1))


def test_sql_boundary_equal_signal_energy():
    for Ns in [0.5, 1.5, 10.0, 100.0]:
        assert np.isclose(gq_pure_loss(0.5, Ns), 1.0)


def test_high_loss_collapse():
    assert gq_pure_loss(0.3, 100.0) < 1.0


def test_gamma_boundary_anchor():
    assert np.isclose(gamma_max(np.array([0.9]), 100.0, a=1.0)[0], 1.5706, rtol=5e-4)


def test_wrapping_probability_monotone():
    vals = wrapping_probability_gaussian(np.array([0.01, 0.1, 1.0]))
    assert np.all(np.diff(vals) > 0)


def test_rayleigh_budget_wavelength_monotone():
    common = dict(
        pressure_MPa=30,
        temperature_K=298.15,
        pulse_energy_J=1e-6,
        probe_length_m=0.01,
        collection_fraction=1e-7,
        eta_sys=0.05,
    )
    n532 = return_photons(lambda_nm=532, **common)
    n1064 = return_photons(lambda_nm=1064, **common)
    n1550 = return_photons(lambda_nm=1550, **common)
    assert n532 > n1064 > n1550
