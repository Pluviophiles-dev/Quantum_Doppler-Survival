from qdboundary_enhanced.gaussian_fidelity import (
    tmsv_cov, gaussian_fidelity_amplitude, signal_phase_rotate,
    build_cov, is_physical_cov, GaussianPoint, summarize_point
)
from qdboundary_enhanced.transduction import GasState, OpticalBudget, rayleigh_return_photons, effective_signal_channel_eta, gaussian_spatial_mode_overlap, idler_time_gate_efficiency
from qdboundary_enhanced.detector_fi import DetectorModel, max_fi_over_phase
from qdboundary.fock import prepare_noisy_tmsv_density
from qdboundary.qzzb import qzzb_phase_bound
import numpy as np


def test_enhanced_package_smoke():
    V = tmsv_cov(1.5)
    assert abs(gaussian_fidelity_amplitude(V, V) - 1.0) < 1e-9
    Vvac = np.eye(4)
    assert abs(gaussian_fidelity_amplitude(Vvac, signal_phase_rotate(Vvac, 0.8)) - 1.0) < 1e-9
    V2 = build_cov(10.0, 0.8, 0.9, 0.5)
    ok, mineig = is_physical_cov(V2)
    assert ok, mineig
    gs = summarize_point(GaussianPoint(ns=3.0, eta_s=0.8, eta_i=0.9, gamma=0.2, tau_points=51))
    assert gs["diagnostic_gaussian_qzzb_squared_fidelity"] >= 0
    ret = rayleigh_return_photons(GasState(), OpticalBudget())
    assert ret["Nret"] >= 0
    eta = effective_signal_channel_eta(GasState(), OpticalBudget())
    assert 0 <= eta["eta_conditional_after_collection"] <= 1
    assert 0 <= eta["spatial_mode_overlap_used"] <= 1
    assert 0 <= eta["temporal_mode_overlap"] <= 1
    assert 0 <= gaussian_spatial_mode_overlap(1064, 1e-3, 1e-3, 0, 0, 1.0) <= 1
    assert 0 <= idler_time_gate_efficiency(5e-9, 5e-9, 0, 0, 0.9) <= 1
    det = max_fi_over_phase(DetectorModel())
    assert det["max_classical_fi"] >= 0


def test_qzzb_uses_single_squared_fidelity_convention():
    rho = prepare_noisy_tmsv_density(Ns=0.5, cutoff=4, eta_s=0.8, eta_i=0.9, gamma=0.1)
    zz = qzzb_phase_bound(rho, cutoff=4, prior_width=np.pi / 2, tau_points=5)
    assert zz >= 0
