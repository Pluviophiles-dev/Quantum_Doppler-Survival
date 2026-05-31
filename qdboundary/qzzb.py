from __future__ import annotations

from typing import Literal

import numpy as np

from .fock import apply_signal_phase_shift, hermitize


def _sqrt_psd(rho: np.ndarray, tol: float = 1e-13) -> np.ndarray:
    """Hermitian positive-semidefinite square root by eigendecomposition."""
    vals, vecs = np.linalg.eigh(hermitize(rho))
    vals = np.where(vals > tol, vals, 0.0)
    return (vecs * np.sqrt(vals)[None, :]) @ vecs.conj().T


def fidelity_unsquared(rho: np.ndarray, sigma: np.ndarray) -> float:
    """Uhlmann fidelity amplitude Tr sqrt(sqrt(rho) sigma sqrt(rho)).

    Density matrices are Hermitized and small negative eigenvalues from roundoff
    are projected to zero inside the PSD square-root routine.
    """
    rho = hermitize(rho)
    sigma = hermitize(sigma)
    sr = _sqrt_psd(rho)
    mid = hermitize(sr @ sigma @ sr)
    vals = np.linalg.eigvalsh(mid)
    vals = np.maximum(vals.real, 0.0)
    val = float(np.sum(np.sqrt(vals)))
    return float(np.clip(val, 0.0, 1.0))


def qzzb_phase_bound(
    rho0: np.ndarray,
    cutoff: int,
    prior_width: float,
    tau_points: int = 41,
    convention: Literal["paper_unsquared", "squared"] = "paper_unsquared",
) -> float:
    """Numerically integrate a QZZB phase bound over tau in [0,W].

    default convention follows the manuscript's written form:
        integrand = 0.5 * tau * (1 - tau/W) * [1 - sqrt(1 - F_amp)]
    where F_amp = Tr sqrt(sqrt(rho) sigma sqrt(rho)).

    For the common squared-fidelity convention, set convention='squared'.
    """
    W = float(prior_width)
    taus = np.linspace(0.0, W, int(tau_points))
    vals = []
    for tau in taus:
        sig = apply_signal_phase_shift(rho0, cutoff, tau)
        F_amp = fidelity_unsquared(rho0, sig)
        if convention == "paper_unsquared":
            distinguish = 1.0 - np.sqrt(max(0.0, 1.0 - F_amp))
        elif convention == "squared":
            F_sq = F_amp**2
            distinguish = 1.0 - np.sqrt(max(0.0, 1.0 - F_sq))
        else:
            raise ValueError(f"Unknown convention: {convention}")
        vals.append(0.5 * tau * (1.0 - tau / W) * distinguish)
    return float(np.trapz(vals, taus))
