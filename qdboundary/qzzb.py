from __future__ import annotations

import numpy as np

from .fock import apply_signal_phase_shift, hermitize


def _sqrt_psd(rho: np.ndarray, tol: float = 1e-13) -> np.ndarray:
    """Hermitian positive-semidefinite square root by eigendecomposition."""
    vals, vecs = np.linalg.eigh(hermitize(rho))
    vals = np.where(vals > tol, vals, 0.0)
    return (vecs * np.sqrt(vals)[None, :]) @ vecs.conj().T


def fidelity_amplitude(rho: np.ndarray, sigma: np.ndarray) -> float:
    """Return the Uhlmann fidelity amplitude.

    This function returns
        F_amp(rho, sigma) = Tr sqrt(sqrt(rho) sigma sqrt(rho)).

    The squared Uhlmann fidelity used in the QZZB integrand is
        F = F_amp**2.

    Keeping the amplitude and squared fidelity explicitly separated prevents the
    ambiguity that can arise when different papers use different symbols for
    fidelity.
    """
    rho = hermitize(rho)
    sigma = hermitize(sigma)
    sr = _sqrt_psd(rho)
    mid = hermitize(sr @ sigma @ sr)
    vals = np.linalg.eigvalsh(mid)
    vals = np.maximum(vals.real, 0.0)
    val = float(np.sum(np.sqrt(vals)))
    return float(np.clip(val, 0.0, 1.0))


# Backward-compatible alias for older scripts. The old name was potentially
# ambiguous, because it returned an amplitude, not the squared Uhlmann fidelity.
fidelity_unsquared = fidelity_amplitude


def uhlmann_fidelity_squared(rho: np.ndarray, sigma: np.ndarray) -> float:
    """Return the squared Uhlmann fidelity F in the standard QZZB convention."""
    amp = fidelity_amplitude(rho, sigma)
    return float(np.clip(amp * amp, 0.0, 1.0))


def qzzb_distinguishability_from_fidelity_squared(F_sq: float | np.ndarray) -> float | np.ndarray:
    """QZZB binary-testing bracket using squared Uhlmann fidelity.

    For F(rho, sigma) = [Tr sqrt(sqrt(rho) sigma sqrt(rho))]^2, the bracket is
        1 - sqrt(1 - F).

    No amplitude-as-written switch is provided. Using the amplitude directly in
    this expression would change the bound and is therefore disallowed.
    """
    F_sq_arr = np.clip(F_sq, 0.0, 1.0)
    return 1.0 - np.sqrt(np.maximum(0.0, 1.0 - F_sq_arr))


def qzzb_integrand_from_fidelity_squared(
    tau: float | np.ndarray,
    F_sq: float | np.ndarray,
    prior_width: float,
) -> float | np.ndarray:
    """Return the scalar QZZB integrand for a phase prior of width W."""
    W = float(prior_width)
    if W <= 0:
        raise ValueError("prior_width must be positive.")
    return 0.5 * tau * (1.0 - tau / W) * qzzb_distinguishability_from_fidelity_squared(F_sq)


def qzzb_phase_bound(
    rho0: np.ndarray,
    cutoff: int,
    prior_width: float,
    tau_points: int = 41,
) -> float:
    """Numerically integrate a phase QZZB over tau in [0, W].

    Convention used here, and only here:
        F(rho, sigma) = [Tr sqrt(sqrt(rho) sigma sqrt(rho))]^2

        Sigma_ZZ >= 1/2 int_0^W d tau tau (1 - tau/W)
                    [1 - sqrt(1 - F(rho_phi, rho_{phi+tau}))].

    The older paper_unsquared/amplitude_as_written branch has intentionally been
    removed, because a string-selectable fidelity convention makes the numerical
    bound physically ambiguous.
    """
    W = float(prior_width)
    if W <= 0:
        raise ValueError("prior_width must be positive.")
    n_tau = int(tau_points)
    if n_tau < 3:
        raise ValueError("tau_points must be at least 3.")

    taus = np.linspace(0.0, W, n_tau)
    vals = []
    for tau in taus:
        sig = apply_signal_phase_shift(rho0, cutoff, float(tau))
        F_sq = uhlmann_fidelity_squared(rho0, sig)
        vals.append(qzzb_integrand_from_fidelity_squared(float(tau), F_sq, W))
    try:
        integral = np.trapezoid(vals, taus)
    except AttributeError:  # NumPy < 2.0
        integral = np.trapz(vals, taus)
    return float(integral)
