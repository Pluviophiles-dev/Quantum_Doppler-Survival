from __future__ import annotations

import numpy as np
from scipy.special import erfc


def coherent_qfi(eta_s: np.ndarray | float, Ns: float) -> np.ndarray | float:
    """Coherent-state phase QFI under signal loss, equal signal-mode energy."""
    return 4.0 * np.asarray(eta_s) * Ns


def tmsv_pure_loss_qfi(eta_s: np.ndarray | float, Ns: float) -> np.ndarray | float:
    """Analytic TMSV QFI under one-sided pure loss with ideal idler preservation."""
    eta_s = np.asarray(eta_s)
    return 4.0 * eta_s * Ns * (Ns + 1.0) / (1.0 + 2.0 * (1.0 - eta_s) * Ns)


def gq_pure_loss(eta_s: np.ndarray | float, Ns: float) -> np.ndarray | float:
    """Analytic advantage ratio F_Q(TMSV, loss) / F_Q(coherent)."""
    eta_s = np.asarray(eta_s)
    return (Ns + 1.0) / (1.0 + 2.0 * (1.0 - eta_s) * Ns)


def geff(eta_s: np.ndarray | float, Ns: float, gamma: np.ndarray | float, a: float = 1.0) -> np.ndarray | float:
    """First-order effective advantage envelope G_Q exp(-a Gamma)."""
    return gq_pure_loss(eta_s, Ns) * np.exp(-a * np.asarray(gamma))


def gamma_max(eta_s: np.ndarray | float, Ns: float, a: float = 1.0) -> np.ndarray:
    """Survival boundary Gamma_max = ln(G_Q)/a for G_Q>1; NaN otherwise."""
    g = gq_pure_loss(eta_s, Ns)
    out = np.full_like(np.asarray(g, dtype=float), np.nan)
    mask = g > 1.0
    out[mask] = np.log(g[mask]) / a
    return out


def local_phase_variance_from_qfi(qfi: np.ndarray | float, M: int) -> np.ndarray | float:
    qfi = np.asarray(qfi, dtype=float)
    return 1.0 / np.maximum(M * qfi, np.finfo(float).tiny)


def wrapping_probability_gaussian(var_phi: np.ndarray | float, period: float = 2.0 * np.pi) -> np.ndarray | float:
    """Two-sided probability that a zero-mean Gaussian phase error exceeds half a period."""
    var_phi = np.asarray(var_phi, dtype=float)
    sigma = np.sqrt(np.maximum(var_phi, np.finfo(float).tiny))
    threshold = 0.5 * period
    return erfc(threshold / (np.sqrt(2.0) * sigma))


def doppler_k(lambda0_m: float, ng: float = 1.000444, backscatter: bool = True) -> float:
    if backscatter:
        return 4.0 * np.pi * ng / lambda0_m
    return 2.0 * np.pi * ng / lambda0_m


def velocity_rmse_from_phase_variance(var_phi: np.ndarray | float, lambda0_m: float, tau_int: float, ng: float = 1.000444) -> np.ndarray | float:
    kd = doppler_k(lambda0_m, ng=ng, backscatter=True)
    return np.sqrt(np.asarray(var_phi)) / (kd * tau_int)
