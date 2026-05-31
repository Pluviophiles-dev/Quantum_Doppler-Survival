#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Quantum Doppler survival-boundary numerical add-on package
"""

from __future__ import annotations

import argparse
import csv
import math
import os
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import sqrtm
from scipy.special import comb
from scipy.stats import norm

try:
    import qutip as qt  # type: ignore
    _HAS_QUTIP = True
except Exception:
    qt = None
    _HAS_QUTIP = False


# -----------------------------------------------------------------------------
# Basic manuscript-model functions
# -----------------------------------------------------------------------------


def doppler_vector_magnitude_backscatter(lambda_0: float = 532e-9, n_index: float = 1.000444) -> float:
    """Backscatter Doppler vector magnitude |K_D| = 4*pi*n/lambda_0 in rad/m."""
    return 4.0 * np.pi * n_index / lambda_0


def calculate_GQ_advantage(eta: np.ndarray | float, N_S: float) -> np.ndarray | float:
    """Pure-loss TMSV advantage ratio G_Q under equal signal-mode energy."""
    eta_arr = np.asarray(eta, dtype=float)
    if np.any((eta_arr <= 0.0) | (eta_arr > 1.0)):
        raise ValueError("eta must satisfy 0 < eta <= 1.")
    if N_S <= 0:
        raise ValueError("N_S must be positive.")
    out = (N_S + 1.0) / (1.0 + 2.0 * (1.0 - eta_arr) * N_S)
    return float(out) if np.isscalar(eta) else out


def calculate_Geff(eta: np.ndarray | float, N_S: float, Gamma: np.ndarray | float, a: float = 1.0) -> np.ndarray | float:
    """First-order effective advantage G_eff = G_Q exp(-a Gamma)."""
    return calculate_GQ_advantage(eta, N_S) * np.exp(-a * np.asarray(Gamma, dtype=float))


def eta_boundary_for_Geff_one(N_S: float, Gamma: float, a: float = 1.0) -> float:
    """Solve G_eff(eta, N_S, Gamma) = 1 for eta. Return nan if not valid."""
    target = np.exp(a * Gamma)
    denom_needed = (N_S + 1.0) / target
    eta = 1.0 - (denom_needed - 1.0) / (2.0 * N_S)
    if eta <= 0 or eta > 1:
        return float("nan")
    return float(eta)


def gamma_boundary_for_Geff_one(eta: float, N_S: float, a: float = 1.0) -> float:
    """Solve G_eff(eta, N_S, Gamma) = 1 for Gamma."""
    gq = calculate_GQ_advantage(eta, N_S)
    return float(np.log(gq) / a) if gq > 1 else float("nan")


def local_phase_variances(eta: np.ndarray | float, N_S: float, Gamma: np.ndarray | float, M: int = 1, a: float = 1.0):
    """Coherent-state and first-order TMSV phase variances."""
    eta_arr = np.asarray(eta, dtype=float)
    gamma_arr = np.asarray(Gamma, dtype=float)
    var_cs = 1.0 / (4.0 * M * eta_arr * N_S)
    g_eff = calculate_Geff(eta_arr, N_S, gamma_arr, a=a)
    var_tmsv = var_cs / g_eff
    return var_cs, var_tmsv, g_eff


def phase_wrapping_probability(sigma_phi: np.ndarray | float, threshold: float = np.pi) -> np.ndarray | float:
    """Gaussian phase-wrapping risk Pr(|error| > threshold)."""
    sigma = np.asarray(sigma_phi, dtype=float)
    sigma = np.maximum(sigma, 1e-300)
    p = 2.0 * norm.sf(threshold / sigma)
    return float(p) if np.isscalar(sigma_phi) else p


# -----------------------------------------------------------------------------
# Finite-dimensional TMSV, channels, fidelity, QFI
# -----------------------------------------------------------------------------


def tmsv_density(N_S: float, ncut: int) -> np.ndarray:
    """Truncated two-mode squeezed vacuum density matrix in |n_signal,n_idler> basis."""
    if N_S <= 0:
        raise ValueError("N_S must be positive.")
    if ncut < 3:
        raise ValueError("ncut should be at least 3.")
    lam = math.sqrt(N_S / (N_S + 1.0))
    amps = np.array([math.sqrt(1.0 - lam * lam) * (lam ** n) for n in range(ncut)], dtype=complex)
    # Renormalize because of finite Fock cutoff.
    amps = amps / np.linalg.norm(amps)
    psi = np.zeros(ncut * ncut, dtype=complex)
    for n, a_n in enumerate(amps):
        psi[n * ncut + n] = a_n
    return np.outer(psi, psi.conjugate())


def truncated_signal_photon_number(N_S: float, ncut: int) -> float:
    """Mean signal photon number after TMSV cutoff renormalization."""
    lam = math.sqrt(N_S / (N_S + 1.0))
    probs = np.array([(1.0 - lam * lam) * (lam ** (2 * n)) for n in range(ncut)], dtype=float)
    probs = probs / probs.sum()
    return float(np.dot(np.arange(ncut), probs))


def single_mode_number_operator(ncut: int) -> np.ndarray:
    return np.diag(np.arange(ncut).astype(float))


def two_mode_signal_number_operator(ncut: int) -> np.ndarray:
    return np.kron(single_mode_number_operator(ncut), np.eye(ncut))


def two_mode_idler_number_operator(ncut: int) -> np.ndarray:
    return np.kron(np.eye(ncut), single_mode_number_operator(ncut))


def phase_rotate_signal(rho: np.ndarray, phi: float, ncut: int) -> np.ndarray:
    """Apply exp(-i phi n_signal) to a two-mode density matrix."""
    phases = np.exp(-1j * phi * np.repeat(np.arange(ncut), ncut))
    return (phases[:, None] * rho) * phases.conjugate()[None, :]


def loss_kraus_single_mode(eta: float, ncut: int, tol: float = 1e-14) -> List[np.ndarray]:
    """Single-mode pure-loss Kraus operators truncated to ncut Fock states."""
    if not (0.0 <= eta <= 1.0):
        raise ValueError("eta must satisfy 0 <= eta <= 1.")
    kraus = []
    for ell in range(ncut):
        K = np.zeros((ncut, ncut), dtype=complex)
        for n in range(ell, ncut):
            amp = math.sqrt(comb(n, ell, exact=False) * ((1.0 - eta) ** ell) * (eta ** (n - ell)))
            K[n - ell, n] = amp
        if np.linalg.norm(K) > tol:
            kraus.append(K)
    return kraus


def apply_loss_mode(rho: np.ndarray, eta: float, ncut: int, mode: str = "signal") -> np.ndarray:
    """Apply pure loss to signal or idler mode with truncated Kraus operators."""
    eye = np.eye(ncut, dtype=complex)
    out = np.zeros_like(rho, dtype=complex)
    for K in loss_kraus_single_mode(eta, ncut):
        K2 = np.kron(K, eye) if mode == "signal" else np.kron(eye, K)
        out += K2 @ rho @ K2.conjugate().T
    # Numerical trace correction: the truncated channel is trace-preserving for states within cutoff,
    # up to roundoff.
    tr = np.trace(out).real
    if tr > 0:
        out = out / tr
    return _hermitize(out)


def apply_signal_phase_diffusion_exact(rho: np.ndarray, Gamma: float, ncut: int) -> np.ndarray:
    """
    Exact signal-mode number dephasing for the GKLS jump L=sqrt(gamma_phi) n.

    In the number basis, off-diagonal blocks decay as
        rho_{n_s,n_i; m_s,m_i} -> exp[-Gamma/2 * (n_s-m_s)^2] rho_{...}.
    """
    if Gamma < 0:
        raise ValueError("Gamma must be nonnegative.")
    ns = np.repeat(np.arange(ncut), ncut)
    decay = np.exp(-0.5 * Gamma * (ns[:, None] - ns[None, :]) ** 2)
    return _hermitize(rho * decay)


def build_noisy_tmsv_density(
    N_S: float,
    ncut: int,
    eta_signal: float = 1.0,
    eta_idler: float = 1.0,
    Gamma: float = 0.0,
) -> np.ndarray:
    """Build truncated TMSV density after signal loss, idler loss, and signal phase diffusion."""
    rho = tmsv_density(N_S, ncut)
    if eta_signal < 1.0:
        rho = apply_loss_mode(rho, eta_signal, ncut, mode="signal")
    if eta_idler < 1.0:
        rho = apply_loss_mode(rho, eta_idler, ncut, mode="idler")
    if Gamma > 0.0:
        rho = apply_signal_phase_diffusion_exact(rho, Gamma, ncut)
    return _hermitize(rho / np.trace(rho).real)


def _hermitize(a: np.ndarray) -> np.ndarray:
    return 0.5 * (a + a.conjugate().T)


def _sqrt_psd(mat: np.ndarray, tol: float = 1e-14) -> np.ndarray:
    """Fast Hermitian positive-semidefinite matrix square root by eigendecomposition."""
    mat = _hermitize(mat)
    vals, vecs = np.linalg.eigh(mat)
    vals = np.where(vals > tol, vals, 0.0)
    return (vecs * np.sqrt(vals)[None, :]) @ vecs.conjugate().T


def _fidelity_from_sqrt_rho(sqrt_rho: np.ndarray, sigma: np.ndarray) -> float:
    """Uhlmann fidelity using a precomputed sqrt(rho)."""
    middle = _hermitize(sqrt_rho @ _hermitize(sigma) @ sqrt_rho)
    vals = np.linalg.eigvalsh(middle)
    vals = np.maximum(vals.real, 0.0)
    f = float(np.sum(np.sqrt(vals)))
    return float(np.clip(f, 0.0, 1.0))


def density_fidelity(rho: np.ndarray, sigma: np.ndarray) -> float:
    """
    Uhlmann fidelity F(rho,sigma) in [0,1].

    The returned value follows QuTiP/scientific convention:
        F = Tr sqrt(sqrt(rho) sigma sqrt(rho)).
    If a squared-fidelity convention is desired, square this output.
    """
    rho = _hermitize(rho)
    sigma = _hermitize(sigma)
    if _HAS_QUTIP:
        try:
            return float(qt.metrics.fidelity(qt.Qobj(rho), qt.Qobj(sigma)))
        except Exception:
            pass
    return _fidelity_from_sqrt_rho(_sqrt_psd(rho), sigma)


def qfi_unitary_phase(rho: np.ndarray, generator: np.ndarray, eig_tol: float = 1e-12) -> float:
    """
    Quantum Fisher information for rho_phi = exp(-i phi G) rho exp(i phi G).

    Formula: F_Q = 2 sum_{i,j} (lambda_i-lambda_j)^2/(lambda_i+lambda_j) |G_ij|^2.
    """
    rho = _hermitize(rho)
    vals, vecs = np.linalg.eigh(rho)
    vals = np.maximum(vals.real, 0.0)
    G_eig = vecs.conjugate().T @ generator @ vecs
    fq = 0.0
    for i in range(len(vals)):
        for j in range(len(vals)):
            denom = vals[i] + vals[j]
            if denom > eig_tol:
                fq += 2.0 * ((vals[i] - vals[j]) ** 2 / denom) * (abs(G_eig[i, j]) ** 2)
    return float(np.real_if_close(fq).real)


def qzzb_bound_for_state(
    rho_no_phase: np.ndarray,
    ncut: int,
    W: float = np.pi,
    tau_points: int = 41,
    use_squared_fidelity: bool = False,
) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute toy QZZB lower bound for a phase prior of width W.

    Returns:
        bound, tau_grid, fidelity_grid, integrand_grid
    """
    if W <= 0:
        raise ValueError("W must be positive.")
    taus = np.linspace(0.0, W, tau_points)
    rho0 = phase_rotate_signal(rho_no_phase, 0.0, ncut)
    sqrt_rho0 = _sqrt_psd(rho0)
    fidelities = []
    for tau in taus:
        rhot = phase_rotate_signal(rho_no_phase, float(tau), ncut)
        f = _fidelity_from_sqrt_rho(sqrt_rho0, rhot)
        if use_squared_fidelity:
            f = f * f
        fidelities.append(float(np.clip(f, 0.0, 1.0)))
    F = np.asarray(fidelities)
    integrand = 0.5 * taus * (1.0 - taus / W) * (1.0 - np.sqrt(np.maximum(0.0, 1.0 - F)))
    bound = float(np.trapezoid(integrand, taus))
    return bound, taus, F, integrand


# -----------------------------------------------------------------------------
# Figure-generation functions
# -----------------------------------------------------------------------------


@dataclass
class ToyConfig:
    N_S_toy: float = 3.0
    ncut: int = 12
    M: int = 1
    W: float = np.pi
    tau_points: int = 41
    lambda_0: float = 532e-9
    tau_int: float = 1e-6
    n_index: float = 1.000444
    a_surrogate: float = 1.0


def run_qzzb_toy_guardcheck(output_dir: str, cfg: ToyConfig) -> None:
    """Three-point QZZB toy guard-check near local-valid/transition/strong-diffusion regimes."""
    os.makedirs(output_dir, exist_ok=True)
    cases = [
        ("A_local_valid", 0.90, 0.50),
        ("B_transition", 0.90, 1.50),
        ("C_stop_extrapolation", 0.90, 2.00),
    ]
    rows = []
    fig, ax = plt.subplots(figsize=(7.2, 5.2))

    for label, eta, Gamma in cases:
        rho = build_noisy_tmsv_density(cfg.N_S_toy, cfg.ncut, eta_signal=eta, eta_idler=1.0, Gamma=Gamma)
        qzzb, taus, F, integrand = qzzb_bound_for_state(rho, cfg.ncut, W=cfg.W, tau_points=cfg.tau_points)
        ns_eff = truncated_signal_photon_number(cfg.N_S_toy, cfg.ncut)
        var_cs, var_sur, geff = local_phase_variances(eta, ns_eff, Gamma, M=cfg.M, a=cfg.a_surrogate)
        guarded = max(float(var_sur), qzzb)
        fq_num = qfi_unitary_phase(rho, two_mode_signal_number_operator(cfg.ncut))
        local_qcrb_num = 1.0 / max(fq_num, 1e-300)
        rows.append({
            "case": label,
            "N_S_input": cfg.N_S_toy,
            "N_S_cutoff_effective": ns_eff,
            "ncut": cfg.ncut,
            "eta_signal": eta,
            "eta_idler": 1.0,
            "Gamma": Gamma,
            "G_eff_surrogate": float(geff),
            "Var_CS_local": float(var_cs),
            "Var_TMSV_surrogate": float(var_sur),
            "QZZB_phase_variance_lower_bound": qzzb,
            "Guarded_phase_variance": guarded,
            "Numerical_QFI_finite_dim": fq_num,
            "Numerical_QCRB_finite_dim": local_qcrb_num,
            "backend_fidelity": "qutip" if _HAS_QUTIP else "scipy_dense",
        })
        ax.plot(taus, F, lw=2.0, label=f"{label}: Γ={Gamma}, QZZB={qzzb:.3g}")

    ax.set_xlabel(r"Phase separation $	au$ (rad)")
    ax.set_ylabel(r"Fidelity $F(\rho_\phi,\rho_{\phi+\tau})$")
    ax.set_title("Toy QZZB guard-check: finite-dimensional TMSV", fontweight="bold")
    ax.grid(True, ls="--", alpha=0.35)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig_path = os.path.join(output_dir, "qzzb_toy_guardcheck_fidelity.png")
    fig.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    csv_path = os.path.join(output_dir, "qzzb_toy_guardcheck_summary.csv")
    _write_csv(csv_path, rows)
    print(f"[saved] {fig_path}")
    print(f"[saved] {csv_path}")


def run_fig5_qzzb_guarded_rmse(output_dir: str, cfg: ToyConfig) -> None:
    """Replacement Fig. 5: analytic local RMSE plus QZZB-guarded toy RMSE."""
    os.makedirs(output_dir, exist_ok=True)
    K_D = doppler_vector_magnitude_backscatter(cfg.lambda_0, cfg.n_index)
    ns_eff = truncated_signal_photon_number(cfg.N_S_toy, cfg.ncut)

    # Panel A: scan eta at fixed Gamma.
    Gamma_fixed = 0.50
    eta_scan = np.linspace(0.20, 1.00, 31)
    rmse_cs_eta = []
    rmse_sur_eta = []
    rmse_guard_eta = []
    qzzb_eta = []
    geff_eta = []
    for eta in eta_scan:
        var_cs, var_sur, geff = local_phase_variances(eta, ns_eff, Gamma_fixed, M=cfg.M, a=cfg.a_surrogate)
        rho = build_noisy_tmsv_density(cfg.N_S_toy, cfg.ncut, eta_signal=float(eta), eta_idler=1.0, Gamma=Gamma_fixed)
        qzzb, *_ = qzzb_bound_for_state(rho, cfg.ncut, W=cfg.W, tau_points=cfg.tau_points)
        var_guard = max(float(var_sur), qzzb)
        rmse_cs_eta.append(math.sqrt(float(var_cs)) / (K_D * cfg.tau_int))
        rmse_sur_eta.append(math.sqrt(float(var_sur)) / (K_D * cfg.tau_int))
        rmse_guard_eta.append(math.sqrt(var_guard) / (K_D * cfg.tau_int))
        qzzb_eta.append(qzzb)
        geff_eta.append(float(geff))

    # Panel B: scan Gamma at fixed eta.
    eta_fixed = 0.90
    Gamma_scan = np.linspace(0.0, 2.5, 31)
    rmse_cs_gamma = []
    rmse_sur_gamma = []
    rmse_guard_gamma = []
    qzzb_gamma = []
    geff_gamma = []
    for Gamma in Gamma_scan:
        var_cs, var_sur, geff = local_phase_variances(eta_fixed, ns_eff, float(Gamma), M=cfg.M, a=cfg.a_surrogate)
        rho = build_noisy_tmsv_density(cfg.N_S_toy, cfg.ncut, eta_signal=eta_fixed, eta_idler=1.0, Gamma=float(Gamma))
        qzzb, *_ = qzzb_bound_for_state(rho, cfg.ncut, W=cfg.W, tau_points=cfg.tau_points)
        var_guard = max(float(var_sur), qzzb)
        rmse_cs_gamma.append(math.sqrt(float(var_cs)) / (K_D * cfg.tau_int))
        rmse_sur_gamma.append(math.sqrt(float(var_sur)) / (K_D * cfg.tau_int))
        rmse_guard_gamma.append(math.sqrt(var_guard) / (K_D * cfg.tau_int))
        qzzb_gamma.append(qzzb)
        geff_gamma.append(float(geff))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14.2, 5.7))
    ax1.plot(eta_scan, rmse_cs_eta, ls="--", lw=2.0, label="coherent local SQL")
    ax1.plot(eta_scan, rmse_sur_eta, lw=2.0, label="TMSV local surrogate")
    ax1.plot(eta_scan, rmse_guard_eta, lw=2.4, label="QZZB-guarded TMSV")
    eta_b = eta_boundary_for_Geff_one(ns_eff, Gamma_fixed, a=cfg.a_surrogate)
    if np.isfinite(eta_b):
        ax1.axvline(eta_b, ls=":", lw=1.8, label=fr"$G_{{eff}}=1$: $eta\approx{eta_b:.2f}$".replace("\x7feta", "\\eta"))
    ax1.set_xlabel(r"Signal transmittance $\eta_s$")
    ax1.set_ylabel("Velocity RMSE lower scale (m/s)")
    ax1.set_title(fr"Guarded RMSE vs $\eta_s$ ($N_S^{{toy}}={cfg.N_S_toy}$, $\Gamma={Gamma_fixed}$)", fontweight="bold")
    ax1.grid(True, ls="--", alpha=0.35)
    ax1.legend(fontsize=8)

    ax2.plot(Gamma_scan, rmse_cs_gamma, ls="--", lw=2.0, label="coherent local SQL")
    ax2.plot(Gamma_scan, rmse_sur_gamma, lw=2.0, label="TMSV local surrogate")
    ax2.plot(Gamma_scan, rmse_guard_gamma, lw=2.4, label="QZZB-guarded TMSV")
    gamma_b = gamma_boundary_for_Geff_one(eta_fixed, ns_eff, a=cfg.a_surrogate)
    if np.isfinite(gamma_b):
        ax2.axvline(gamma_b, ls=":", lw=1.8, label=fr"$G_{{eff}}=1$: $\Gamma\approx{gamma_b:.2f}$")
    ax2.set_xlabel(r"Accumulated phase diffusion $\Gamma$")
    ax2.set_ylabel("Velocity RMSE lower scale (m/s)")
    ax2.set_title(fr"Guarded RMSE vs $\Gamma$ ($N_S^{{toy}}={cfg.N_S_toy}$, $\eta_s={eta_fixed}$)", fontweight="bold")
    ax2.grid(True, ls="--", alpha=0.35)
    ax2.legend(fontsize=8)

    fig.suptitle("Fig. 5 replacement: QZZB-guarded velocity RMSE propagation", fontsize=15, fontweight="bold")
    fig.text(
        0.5,
        -0.02,
        "The finite-dimensional QZZB curve is a toy global guardrail. It is used to indicate where local QFI/surrogate RMSE should stop being extrapolated.",
        ha="center",
        fontsize=9,
    )
    fig.tight_layout()
    fig_path = os.path.join(output_dir, "fig5_qzzb_guarded_rmse.png")
    fig.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    csv_path = os.path.join(output_dir, "fig5_qzzb_guarded_rmse_data.csv")
    rows = []
    for eta, a, b, c, q, g in zip(eta_scan, rmse_cs_eta, rmse_sur_eta, rmse_guard_eta, qzzb_eta, geff_eta):
        rows.append({"scan": "eta", "eta_signal": float(eta), "Gamma": Gamma_fixed, "rmse_cs": a, "rmse_surrogate": b, "rmse_guarded": c, "qzzb_phase_variance": q, "G_eff": g})
    for Gamma, a, b, c, q, g in zip(Gamma_scan, rmse_cs_gamma, rmse_sur_gamma, rmse_guard_gamma, qzzb_gamma, geff_gamma):
        rows.append({"scan": "Gamma", "eta_signal": eta_fixed, "Gamma": float(Gamma), "rmse_cs": a, "rmse_surrogate": b, "rmse_guarded": c, "qzzb_phase_variance": q, "G_eff": g})
    _write_csv(csv_path, rows)
    print(f"[saved] {fig_path}")
    print(f"[saved] {csv_path}")


def run_idler_loss_sensitivity(output_dir: str, cfg: ToyConfig) -> None:
    """Finite-dimensional numerical-QFI survival map over signal and idler loss."""
    os.makedirs(output_dir, exist_ok=True)
    ngrid = 25
    eta_s_grid = np.linspace(0.20, 1.00, ngrid)
    eta_i_grid = np.linspace(0.20, 1.00, ngrid)
    G = two_mode_signal_number_operator(cfg.ncut)
    ns_eff = truncated_signal_photon_number(cfg.N_S_toy, cfg.ncut)
    ratio = np.zeros((ngrid, ngrid))

    for iy, eta_i in enumerate(eta_i_grid):
        for ix, eta_s in enumerate(eta_s_grid):
            rho = build_noisy_tmsv_density(cfg.N_S_toy, cfg.ncut, eta_signal=float(eta_s), eta_idler=float(eta_i), Gamma=0.0)
            fq = qfi_unitary_phase(rho, G)
            fcs = 4.0 * eta_s * ns_eff
            ratio[iy, ix] = fq / max(fcs, 1e-300)

    fig, ax = plt.subplots(figsize=(7.2, 5.8))
    im = ax.imshow(
        ratio,
        origin="lower",
        extent=[eta_s_grid[0], eta_s_grid[-1], eta_i_grid[0], eta_i_grid[-1]],
        aspect="auto",
        vmin=0.0,
        vmax=min(4.0, max(1.1, np.nanmax(ratio))),
    )
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(r"Finite-dimensional advantage $F_Q^{TMSV}/F_Q^{CS}$")
    cs = ax.contour(eta_s_grid, eta_i_grid, ratio, levels=[1.0], linewidths=2.0)
    ax.clabel(cs, fmt={1.0: "G=1"}, fontsize=9)
    ax.set_xlabel(r"Signal transmittance $\eta_s$")
    ax.set_ylabel(r"Idler storage efficiency $\eta_i$")
    ax.set_title(fr"Idler-loss sensitivity ($N_S^{{toy}}={cfg.N_S_toy}$, ncut={cfg.ncut})", fontweight="bold")
    ax.grid(False)
    fig.tight_layout()
    fig_path = os.path.join(output_dir, "idler_loss_sensitivity_qfi_map.png")
    fig.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    csv_path = os.path.join(output_dir, "idler_loss_sensitivity_qfi_map.csv")
    rows = []
    for iy, eta_i in enumerate(eta_i_grid):
        for ix, eta_s in enumerate(eta_s_grid):
            rows.append({"eta_signal": float(eta_s), "eta_idler": float(eta_i), "advantage_ratio_numerical_QFI": float(ratio[iy, ix])})
    _write_csv(csv_path, rows)
    print(f"[saved] {fig_path}")
    print(f"[saved] {csv_path}")


def run_phase_wrapping_risk_map(output_dir: str, cfg: ToyConfig) -> None:
    """Analytic surrogate phase-wrapping risk map over eta and Gamma."""
    os.makedirs(output_dir, exist_ok=True)
    eta_grid = np.linspace(0.10, 1.00, 180)
    gamma_grid = np.linspace(0.0, 5.0, 180)
    EE, GG = np.meshgrid(eta_grid, gamma_grid)
    _, var_tmsv, geff = local_phase_variances(EE, cfg.N_S_toy, GG, M=cfg.M, a=cfg.a_surrogate)
    pwrap = phase_wrapping_probability(np.sqrt(var_tmsv), threshold=np.pi)

    fig, ax = plt.subplots(figsize=(7.5, 5.7))
    im = ax.imshow(
        np.log10(np.maximum(pwrap, 1e-12)),
        origin="lower",
        extent=[eta_grid[0], eta_grid[-1], gamma_grid[0], gamma_grid[-1]],
        aspect="auto",
        vmin=-12,
        vmax=0,
    )
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(r"$\log_{10} P_{wrap}$")
    cs1 = ax.contour(eta_grid, gamma_grid, geff, levels=[1.0], colors="white", linewidths=2.0)
    ax.clabel(cs1, fmt={1.0: r"$G_{eff}=1$"}, fontsize=9)
    # Practical phase-wrapping contours.
    levels = [1e-6, 1e-3, 1e-2]
    cs2 = ax.contour(eta_grid, gamma_grid, pwrap, levels=levels, colors="black", linestyles="--", linewidths=1.2)
    ax.clabel(cs2, fmt={1e-6: "1e-6", 1e-3: "1e-3", 1e-2: "1e-2"}, fontsize=8)
    ax.set_xlabel(r"System transmittance $\eta$")
    ax.set_ylabel(r"Accumulated phase diffusion $\Gamma$")
    ax.set_title(fr"Phase-wrapping risk map ($N_S={cfg.N_S_toy}$, M={cfg.M})", fontweight="bold")
    fig.tight_layout()
    fig_path = os.path.join(output_dir, "phase_wrapping_risk_map.png")
    fig.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    csv_path = os.path.join(output_dir, "phase_wrapping_risk_map.csv")
    # Store a downsampled version for compactness.
    rows = []
    for iy in range(0, len(gamma_grid), 6):
        for ix in range(0, len(eta_grid), 6):
            rows.append({"eta": float(EE[iy, ix]), "Gamma": float(GG[iy, ix]), "G_eff": float(geff[iy, ix]), "P_wrap": float(pwrap[iy, ix])})
    _write_csv(csv_path, rows)
    print(f"[saved] {fig_path}")
    print(f"[saved] {csv_path}")


def run_cutoff_convergence(output_dir: str, cfg: ToyConfig) -> None:
    """Small cutoff-convergence diagnostic for the toy QZZB guard-check."""
    os.makedirs(output_dir, exist_ok=True)
    # Keep this diagnostic lightweight. Increase the list manually if higher cutoff convergence is needed.
    ncuts = sorted(set([6, 8, 10, 12, min(14, max(6, cfg.ncut + 2))]))
    rows = []
    for nc in ncuts:
        local_cfg = ToyConfig(**{**cfg.__dict__, "ncut": nc})
        rho = build_noisy_tmsv_density(local_cfg.N_S_toy, nc, eta_signal=0.90, eta_idler=1.0, Gamma=0.50)
        qzzb, *_ = qzzb_bound_for_state(rho, nc, W=local_cfg.W, tau_points=local_cfg.tau_points)
        ns_eff = truncated_signal_photon_number(local_cfg.N_S_toy, nc)
        fq = qfi_unitary_phase(rho, two_mode_signal_number_operator(nc))
        rows.append({"ncut": nc, "N_S_cutoff_effective": ns_eff, "QZZB_phase_variance": qzzb, "numerical_QFI": fq})

    fig, ax1 = plt.subplots(figsize=(7.0, 5.0))
    ax1.plot([r["ncut"] for r in rows], [r["QZZB_phase_variance"] for r in rows], marker="o", lw=2.0)
    ax1.set_xlabel("Fock cutoff ncut")
    ax1.set_ylabel("Toy QZZB phase-variance lower bound")
    ax1.set_title(fr"Cutoff convergence diagnostic ($N_S^{{toy}}={cfg.N_S_toy}$)", fontweight="bold")
    ax1.grid(True, ls="--", alpha=0.35)
    fig.tight_layout()
    fig_path = os.path.join(output_dir, "cutoff_convergence_qzzb.png")
    fig.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    csv_path = os.path.join(output_dir, "cutoff_convergence_qzzb.csv")
    _write_csv(csv_path, rows)
    print(f"[saved] {fig_path}")
    print(f"[saved] {csv_path}")


# -----------------------------------------------------------------------------
# Utilities and CLI
# -----------------------------------------------------------------------------


def _write_csv(path: str, rows: List[Dict[str, object]]) -> None:
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_readme(output_dir: str, cfg: ToyConfig) -> None:
    text = f"""# Quantum Doppler QZZB numerical outputs

Generated by `quantum_doppler_qzzb_package.py`.

Backend fidelity: {'qutip' if _HAS_QUTIP else 'scipy_dense'}

Toy configuration:
- N_S_toy = {cfg.N_S_toy}
- ncut = {cfg.ncut}
- M = {cfg.M}
- phase prior width W = {cfg.W}
- QZZB tau grid points = {cfg.tau_points}
- lambda_0 = {cfg.lambda_0} m
- tau_int = {cfg.tau_int} s
- refractive index = {cfg.n_index}

Interpretation:
- The QZZB module is a finite-dimensional toy guard-check, not a full global optimality proof.
- The QZZB-guarded RMSE is computed by max(local surrogate variance, QZZB lower bound), then mapped to velocity through sigma_v = sigma_phi/(|K_D| tau_int).
- The idler-loss map uses a finite-dimensional numerical QFI formula for unitary signal-mode phase encoding.
- The phase-wrapping map uses the local Gaussian error estimate P(|error|>pi).
"""
    with open(os.path.join(output_dir, "README_outputs.md"), "w", encoding="utf-8") as f:
        f.write(text)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="QuTiP-style toy QZZB and guarded RMSE package for a quantum Doppler manuscript.")
    parser.add_argument("--mode", choices=["all", "qzzb", "fig5", "idler", "wrap", "cutoff"], default="all")
    parser.add_argument("--output-dir", default="qzzb_outputs")
    parser.add_argument("--ns-toy", type=float, default=3.0, help="Toy signal-mode photon number for finite-dimensional calculations.")
    parser.add_argument("--ncut", type=int, default=12, help="Fock cutoff per mode.")
    parser.add_argument("--M", type=int, default=1, help="Number of independent samples for local variance models.")
    parser.add_argument("--W", type=float, default=float(np.pi), help="Phase prior width for toy QZZB.")
    parser.add_argument("--tau-points", type=int, default=41, help="QZZB integration grid points.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = ToyConfig(N_S_toy=args.ns_toy, ncut=args.ncut, M=args.M, W=args.W, tau_points=args.tau_points)
    os.makedirs(args.output_dir, exist_ok=True)
    print("QuTiP available:", _HAS_QUTIP)
    print("Output directory:", args.output_dir)
    print("Effective cutoff-renormalized N_S:", truncated_signal_photon_number(cfg.N_S_toy, cfg.ncut))

    if args.mode in ("all", "qzzb"):
        run_qzzb_toy_guardcheck(args.output_dir, cfg)
    if args.mode in ("all", "fig5"):
        run_fig5_qzzb_guarded_rmse(args.output_dir, cfg)
    if args.mode in ("all", "idler"):
        run_idler_loss_sensitivity(args.output_dir, cfg)
    if args.mode in ("all", "wrap"):
        run_phase_wrapping_risk_map(args.output_dir, cfg)
    if args.mode in ("all", "cutoff"):
        run_cutoff_convergence(args.output_dir, cfg)
    write_readme(args.output_dir, cfg)
    print(f"[saved] {os.path.join(args.output_dir, 'README_outputs.md')}")


if __name__ == "__main__":
    main()
