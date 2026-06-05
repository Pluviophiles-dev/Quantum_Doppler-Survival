#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Covariance-matrix Gaussian fidelity diagnostics for TMSV-assisted phase estimation.

Convention
----------
Internal public functions use hbar=2 covariance matrices unless stated otherwise:
vacuum covariance V_vac = I.

Scope and integrity constraints
-------------------------------
Exact number-dephasing phase diffusion is a non-Gaussian channel. This module does
not claim to solve that channel in covariance form. The only nonzero-gamma option
implemented here is an explicitly labelled cross-covariance damping surrogate,
intended for stress testing and sanity checks only.

The QZZB-like diagnostic in this file uses the squared Uhlmann fidelity convention
exclusively:
    F = [Tr sqrt(sqrt(rho) sigma sqrt(rho))]^2.
The older amplitude-as-written branch has been removed to avoid convention-tuning.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal
import math
import numpy as np
from numpy.typing import NDArray
from scipy.linalg import block_diag
from scipy.integrate import trapezoid

Array = NDArray[np.float64]
DiffusionModel = Literal["none", "cross_covariance_surrogate"]


@dataclass(frozen=True)
class GaussianPoint:
    ns: float = 10.0
    eta_s: float = 0.9
    eta_i: float = 1.0
    gamma: float = 0.3
    prior_width: float = math.pi
    tau_points: int = 401
    fd_delta: float = 2e-4
    diffusion_model: DiffusionModel = "cross_covariance_surrogate"


def omega(n_modes: int) -> Array:
    j = np.array([[0.0, 1.0], [-1.0, 0.0]], dtype=float)
    return block_diag(*([j] * n_modes)).astype(float)


def rot(phi: float) -> Array:
    c, s = math.cos(phi), math.sin(phi)
    return np.array([[c, -s], [s, c]], dtype=float)


def tmsv_cov(ns: float) -> Array:
    if ns < 0:
        raise ValueError("ns must be non-negative.")
    a = 2.0 * ns + 1.0
    c = 2.0 * math.sqrt(ns * (ns + 1.0))
    z = np.diag([1.0, -1.0])
    return np.block([[a * np.eye(2), c * z],
                     [c * z,        a * np.eye(2)]]).astype(float)


def apply_two_mode_loss(V: Array, eta_s: float, eta_i: float, hbar: float = 2.0) -> Array:
    if not (0.0 <= eta_s <= 1.0 and 0.0 <= eta_i <= 1.0):
        raise ValueError("eta_s and eta_i must be in [0,1].")
    k = block_diag(math.sqrt(eta_s) * np.eye(2), math.sqrt(eta_i) * np.eye(2))
    vac = hbar / 2.0
    noise = block_diag((1.0 - eta_s) * vac * np.eye(2),
                       (1.0 - eta_i) * vac * np.eye(2))
    return k @ V @ k.T + noise


def apply_cross_covariance_surrogate_diffusion(V: Array, gamma: float) -> Array:
    """Apply an explicitly non-exact covariance-level dephasing surrogate.

    This operation is not the exact solution of number dephasing. It only damps
    the signal-idler covariance blocks by exp(-gamma/2), leaving local blocks
    unchanged. Use it for labelled stress tests only, not as a proof of a
    non-Gaussian dephasing QFI or QZZB.
    """
    if gamma < 0:
        raise ValueError("gamma must be non-negative.")
    out = V.copy()
    damp = math.exp(-0.5 * gamma)
    out[:2, 2:4] *= damp
    out[2:4, :2] *= damp
    return out


# Backward-compatible alias, intentionally verbose in the docstring above.
apply_gaussianized_phase_diffusion = apply_cross_covariance_surrogate_diffusion


def signal_phase_rotate(V: Array, phi: float) -> Array:
    s = block_diag(rot(phi), np.eye(2))
    return s @ V @ s.T


def is_physical_cov(V_hbar2: Array, tol: float = 1e-8) -> tuple[bool, float]:
    n = V_hbar2.shape[0] // 2
    eig = np.linalg.eigvalsh(V_hbar2 + 1j * omega(n))
    min_eig = float(np.min(np.real(eig)))
    return min_eig >= -tol, min_eig


def _positive_w_eigs_banchi(V1_std: Array, V2_std: Array) -> Array:
    n = V1_std.shape[0] // 2
    Om = omega(n)
    Vsum = V1_std + V2_std
    inv_sum = np.linalg.inv(Vsum)
    # Banchi/Braunstein/Pirandola auxiliary expression in hbar=1 convention.
    Vaux = Om.T @ inv_sum @ (Om / 4.0 + V2_std @ Om @ V1_std)
    Waux = -2.0 * Vaux @ (1j * Om)
    vals = np.linalg.eigvals(Waux)
    vals = np.sort(np.abs(np.real_if_close(vals, tol=1000).real))
    if len(vals) != 2 * n:
        raise RuntimeError("Unexpected auxiliary spectrum size.")
    return np.maximum(vals[::2], 1.0)


def gaussian_fidelity_amplitude(V1_hbar2: Array, V2_hbar2: Array) -> float:
    """Uhlmann fidelity amplitude for zero-mean Gaussian states."""
    if V1_hbar2.shape != V2_hbar2.shape:
        raise ValueError("Covariance matrices must have the same shape.")
    ok1, min1 = is_physical_cov(V1_hbar2)
    ok2, min2 = is_physical_cov(V2_hbar2)
    if not ok1 or not ok2:
        raise ValueError(f"Unphysical covariance. min eigs=({min1:.3e},{min2:.3e})")
    V1 = V1_hbar2 / 2.0
    V2 = V2_hbar2 / 2.0
    det_sum = float(np.linalg.det(V1 + V2))
    if det_sum <= 0:
        raise ValueError("V1+V2 determinant must be positive.")
    w = _positive_w_eigs_banchi(V1, V2)
    prefactor = 1.0
    for wk in w:
        prefactor *= math.sqrt(wk + math.sqrt(max(wk * wk - 1.0, 0.0)))
    F_amp = prefactor / (det_sum ** 0.25)
    return float(min(1.0, max(0.0, np.real_if_close(F_amp).real)))


def gaussian_fidelity_squared(V1_hbar2: Array, V2_hbar2: Array) -> float:
    """Squared Uhlmann fidelity F = F_amp**2."""
    amp = gaussian_fidelity_amplitude(V1_hbar2, V2_hbar2)
    return float(np.clip(amp * amp, 0.0, 1.0))


def build_cov(ns: float, eta_s: float, eta_i: float, gamma: float,
              diffusion_model: DiffusionModel = "cross_covariance_surrogate") -> Array:
    V = tmsv_cov(ns)
    V = apply_two_mode_loss(V, eta_s, eta_i)
    if diffusion_model == "none":
        if gamma != 0:
            # Gamma is ignored under the no-diffusion model by design.
            pass
        return V
    if diffusion_model == "cross_covariance_surrogate":
        return apply_cross_covariance_surrogate_diffusion(V, gamma)
    raise ValueError(f"Unknown diffusion_model: {diffusion_model}")


def fidelity_curve(point: GaussianPoint) -> dict[str, Array]:
    base = build_cov(point.ns, point.eta_s, point.eta_i, point.gamma, point.diffusion_model)
    taus = np.linspace(0.0, point.prior_width, point.tau_points)
    F_amp = np.array([gaussian_fidelity_amplitude(base, signal_phase_rotate(base, float(t))) for t in taus])
    F_sq = F_amp ** 2
    return {
        "tau_rad": taus,
        "fidelity_amplitude": F_amp,
        "fidelity_squared": F_sq,
        "bures_distance_sq": 2.0 * (1.0 - F_amp),
    }


def qzzb_integrand(tau: Array, F_sq: Array, W: float) -> Array:
    """QZZB integrand using squared Uhlmann fidelity only."""
    if W <= 0:
        raise ValueError("W must be positive.")
    bracket = 1.0 - np.sqrt(np.maximum(0.0, 1.0 - np.clip(F_sq, 0.0, 1.0)))
    return 0.5 * tau * (1.0 - tau / W) * bracket


def diagnostic_qzzb(point: GaussianPoint) -> float:
    c = fidelity_curve(point)
    y = qzzb_integrand(c["tau_rad"], c["fidelity_squared"], point.prior_width)
    return float(trapezoid(y, c["tau_rad"]))


def local_bures_qfi_fd(point: GaussianPoint, fd_delta: float | None = None,
                       n_steps: int = 4) -> float:
    """Finite-difference Bures QFI diagnostic from local fidelity curvature.

    This remains a numerical diagnostic rather than an analytic Gaussian-QFI
    formula. The companion local_bures_qfi_convergence function should be used
    to audit fd_delta sensitivity whenever this value is reported.

    The fit is constrained through the physically known point F_amp(0)=1:
    1 - F_amp(delta) = slope * delta**2 + O(delta**4).  Earlier versions used
    an unconstrained intercept and raised on tiny negative intercepts, which
    made the diagnostic brittle across NumPy/SciPy versions.
    """
    d0 = float(point.fd_delta if fd_delta is None else fd_delta)
    if d0 <= 0:
        raise ValueError("fd_delta must be positive.")
    if n_steps < 3:
        raise ValueError("n_steps must be at least 3.")
    base = build_cov(point.ns, point.eta_s, point.eta_i, point.gamma, point.diffusion_model)
    deltas = d0 * np.arange(1, n_steps + 1, dtype=float)
    x, y = [], []
    for d in deltas:
        F_amp = gaussian_fidelity_amplitude(base, signal_phase_rotate(base, float(d)))
        x.append(d * d)
        y.append(1.0 - F_amp)
    x_arr = np.array(x)
    y_arr = np.array(y)
    denom = float(np.dot(x_arr, x_arr))
    if denom <= 0:
        raise RuntimeError("Degenerate finite-difference grid.")
    slope_origin = float(np.dot(x_arr, y_arr) / denom)
    return float(max(0.0, 8.0 * slope_origin))


# Backward-compatible name. Kept only because existing scripts may call it.
local_bures_qfi = local_bures_qfi_fd


def local_bures_qfi_convergence(point: GaussianPoint,
                                deltas: tuple[float, ...] = (1e-5, 2e-5, 5e-5, 1e-4, 2e-4, 5e-4),
                                n_steps: int = 4) -> list[dict[str, float]]:
    """Return a finite-difference step-size audit table for the Bures QFI."""
    rows: list[dict[str, float]] = []
    for d in deltas:
        qfi = local_bures_qfi_fd(point, fd_delta=float(d), n_steps=n_steps)
        rows.append({"fd_delta": float(d), "qfi_fd": qfi})
    vals = np.array([r["qfi_fd"] for r in rows], dtype=float)
    median = float(np.median(vals))
    for r in rows:
        r["relative_deviation_from_median"] = abs(r["qfi_fd"] - median) / abs(median) if median else float("nan")
    return rows


def pure_loss_ratio_equal_signal(ns: float, eta_s: float) -> float:
    if ns <= 0 or eta_s <= 0:
        return float("nan")
    return (ns + 1.0) / (1.0 + 2.0 * (1.0 - eta_s) * ns)


def summarize_point(point: GaussianPoint) -> dict[str, float | str | bool]:
    V = build_cov(point.ns, point.eta_s, point.eta_i, point.gamma, point.diffusion_model)
    ok, min_eig = is_physical_cov(V)
    qfi = local_bures_qfi_fd(point)
    coherent_qfi = 4.0 * point.eta_s * point.ns
    return {
        "ns": point.ns,
        "eta_s": point.eta_s,
        "eta_i": point.eta_i,
        "gamma": point.gamma,
        "prior_width": point.prior_width,
        "physical": bool(ok),
        "physical_min_eig": min_eig,
        "diffusion_model": point.diffusion_model,
        "diffusion_model_scope": (
            "exact Gaussian covariance calculation without dephasing" if point.diffusion_model == "none"
            else "cross-covariance damping surrogate; not an exact non-Gaussian number-dephasing solution"
        ),
        "surrogate_gaussian_local_bures_qfi_fd": qfi,
        "coherent_sql_qfi_equal_signal": coherent_qfi,
        "surrogate_gaussian_qfi_ratio_vs_coherent": qfi / coherent_qfi if coherent_qfi > 0 else float("nan"),
        "diagnostic_gaussian_qzzb_squared_fidelity": diagnostic_qzzb(point),
        "pure_loss_ratio_equal_signal_no_diffusion": pure_loss_ratio_equal_signal(point.ns, point.eta_s),
        "qzzb_fidelity_convention": "squared_Uhlmann_fidelity_only",
    }


def scan_eta_gamma(ns: float, eta_i: float, eta_s_grid: Array, gamma_grid: Array,
                   prior_width: float = math.pi, tau_points: int = 301,
                   diffusion_model: DiffusionModel = "cross_covariance_surrogate") -> list[dict[str, float | str | bool]]:
    rows = []
    for gamma in gamma_grid:
        for eta_s in eta_s_grid:
            pt = GaussianPoint(ns=ns, eta_s=float(eta_s), eta_i=eta_i, gamma=float(gamma),
                               prior_width=prior_width, tau_points=tau_points,
                               diffusion_model=diffusion_model)
            rows.append(summarize_point(pt))
    return rows
