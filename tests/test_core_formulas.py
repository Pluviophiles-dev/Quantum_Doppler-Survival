import numpy as np

from qdboundary.formulas import (
    gamma_max,
    gq_equal_total_energy,
    gq_pure_loss,
    gq_pure_loss_legacy_factor2,
    eta_threshold_equal_total,
    tmsv_pure_loss_qfi,
    wrapping_probability_gaussian,
)
from qdboundary.rayleigh import return_photons


def test_lossless_tmsv_qfi():
    Ns = 3.0
    assert np.isclose(tmsv_pure_loss_qfi(1.0, Ns), 4 * Ns * (Ns + 1))


def test_fock_consistent_formula_has_no_eta_half_equal_signal_boundary():
    for Ns in [0.5, 1.5, 10.0, 100.0]:
        assert gq_pure_loss(0.5, Ns) > 1.0


def test_legacy_factor2_formula_is_not_default():
    Ns = 3.0
    eta = 0.7
    assert gq_pure_loss(eta, Ns) > gq_pure_loss_legacy_factor2(eta, Ns)


def test_equal_total_threshold_corrected():
    Ns = 10.0
    thr = eta_threshold_equal_total(Ns)
    assert np.isclose(thr, 0.55)
    assert gq_equal_total_energy(thr + 1e-3, Ns) > 1.0
    assert gq_equal_total_energy(thr - 1e-3, Ns) < 1.0


def test_high_loss_equal_signal_collapses_to_near_unity_not_false_advantage():
    assert np.isclose(gq_pure_loss(0.0, 100.0), 1.0)
    assert gq_pure_loss(0.005, 100.0) < 1.01


def test_gamma_boundary_anchor_corrected_formula():
    expected = np.log((100.0 + 1.0) / (1.0 + (1.0 - 0.9) * 100.0))
    assert np.isclose(gamma_max(np.array([0.9]), 100.0, a=1.0)[0], expected, rtol=1e-10)


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
