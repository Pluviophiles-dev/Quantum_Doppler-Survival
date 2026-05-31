#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
multilayer_survival_map.py

Purpose
-------
Generate a "multilayer survival map" for photon-starved high-pressure-gas
quantum-enhanced Rayleigh-Doppler velocimetry.

This script is designed for the manuscript narrative:

    Rayleigh photon budget
      -> detector admissibility
      -> pure-loss TMSV QFI advantage
      -> loss-diffusion survival
      -> phase-wrapping / guard-band screening
      -> idler-preservation screening
      -> usable quantum-advantage island

The script deliberately separates:
    N_in / N_ret / N_S
and
    eta_sys / eta_s / eta_i

It produces:
    outputs/multilayer_survival_map.png
    outputs/multilayer_survival_map.pdf
    outputs/scenario_boundary_table.csv

Dependencies
------------
numpy, matplotlib

Example
-------
python multilayer_survival_map.py

Author note
-----------
The detector, idler, and phase-wrapping layers are diagnostic screening layers.
The "guarded" class here is a local-boundary guard-band / wrapping-risk screen.
If you have a finite-dimensional QZZB module, replace the guard-band logic by
your computed Sigma_ZZ / Var_local ratio.
"""

from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Patch


# -----------------------------
# Physical constants
# -----------------------------
H = 6.62607015e-34
C = 299792458.0
KB = 1.380649e-23
R = 8.314462618


# -----------------------------
# Methane / Rayleigh defaults
# -----------------------------
@dataclass
class PhotonBudgetConfig:
    temperature_K: float = 298.15
    probe_length_m: float = 0.01
    pulse_energy_J: float = 1.0e-6
    collection_fraction: float = 1.0e-7       # Omega / 4pi
    eta_sys: float = 0.05
    king_factor: float = 1.04
    reference_density_m3: float = 2.68678e25
    reference_refractive_index: float = 1.000444

    # Peng-Robinson methane proxy
    methane_Tc_K: float = 190.564
    methane_Pc_Pa: float = 4.5992e6
    methane_omega: float = 0.011


@dataclass
class QuantumBoundaryConfig:
    N_S: float = 100.0
    M: float = 1.0
    diffusion_exponent_a: float = 1.0

    # Idler preservation screen.
    # This is a transparent engineering/diagnostic threshold, not a theorem.
    eta_i: float = 0.90
    eta_i_min: float = 0.75

    # Phase wrapping diagnostic threshold.
    p_wrap_threshold: float = 1.0e-3

    # Guard band near the local boundary.
    # Points with 1 < G_eff < guard_Geff_min are labeled "guarded".
    guard_Geff_min: float = 1.25

    # Detector screen.
    dark_count_rate_Hz: float = 100.0
    background_count_rate_Hz: float = 0.0
    gate_time_s: float = 1.0e-6
    SBR_min: float = 5.0


# -----------------------------
# Rayleigh photon budget
# -----------------------------
def peng_robinson_Z_methane(P_Pa: float, T_K: float, cfg: PhotonBudgetConfig) -> float:
    """Gas-phase compressibility factor Z for methane using Peng-Robinson EOS."""
    Tr = T_K / cfg.methane_Tc_K
    kappa = 0.37464 + 1.54226 * cfg.methane_omega - 0.26992 * cfg.methane_omega**2
    alpha = (1.0 + kappa * (1.0 - math.sqrt(Tr))) ** 2

    a = 0.45724 * R**2 * cfg.methane_Tc_K**2 / cfg.methane_Pc_Pa * alpha
    b = 0.07780 * R * cfg.methane_Tc_K / cfg.methane_Pc_Pa

    A = a * P_Pa / (R**2 * T_K**2)
    B = b * P_Pa / (R * T_K)

    # PR cubic:
    # Z^3 - (1-B)Z^2 + (A - 3B^2 - 2B)Z - (AB - B^2 - B^3) = 0
    coeff = [
        1.0,
        -(1.0 - B),
        A - 3.0 * B**2 - 2.0 * B,
        -(A * B - B**2 - B**3),
    ]
    roots = np.roots(coeff)
    real_roots = sorted([r.real for r in roots if abs(r.imag) < 1e-8])
    if not real_roots:
        return float(max(roots, key=lambda z: z.real).real)

    # Gas-phase root is the largest real root.
    return float(real_roots[-1])


def methane_number_density(P_MPa: float, cfg: PhotonBudgetConfig) -> float:
    """Molecular number density n(P,T) [m^-3] from PR methane proxy."""
    P_Pa = P_MPa * 1e6
    Z = peng_robinson_Z_methane(P_Pa, cfg.temperature_K, cfg)
    return P_Pa / (Z * KB * cfg.temperature_K)


def rayleigh_cross_section(lambda_nm: np.ndarray | float, cfg: PhotonBudgetConfig) -> np.ndarray:
    """
    Effective single-molecule Rayleigh cross section [m^2].

    sigma_R = 24*pi^3/(n0^2 * lambda^4) *
              ((n_ref^2 - 1)/(n_ref^2 + 2))^2 * F_K
    """
    lam = np.asarray(lambda_nm, dtype=float) * 1e-9
    refr = (cfg.reference_refractive_index**2 - 1.0) / (
        cfg.reference_refractive_index**2 + 2.0
    )
    sigma = (
        24.0
        * np.pi**3
        / (cfg.reference_density_m3**2 * lam**4)
        * refr**2
        * cfg.king_factor
    )
    return sigma


def rayleigh_return_photons(
    P_MPa: np.ndarray | float,
    lambda_nm: np.ndarray | float,
    cfg: PhotonBudgetConfig,
) -> np.ndarray:
    """Mean collected Rayleigh-return photons per pulse."""
    P_arr = np.asarray(P_MPa, dtype=float)
    lam_nm_arr = np.asarray(lambda_nm, dtype=float)
    lam_m = lam_nm_arr * 1e-9

    Nin = cfg.pulse_energy_J / (H * C / lam_m)
    sigma = rayleigh_cross_section(lam_nm_arr, cfg)

    # Density depends only on pressure.  When P_arr is a meshgrid, avoid
    # solving the Peng-Robinson cubic at every wavelength cell.
    p_flat = P_arr.ravel()
    p_unique, inv = np.unique(p_flat, return_inverse=True)
    n_unique = np.array([methane_number_density(float(p), cfg) for p in p_unique])
    n = n_unique[inv].reshape(P_arr.shape)

    return (
        Nin
        * n
        * cfg.probe_length_m
        * sigma
        * cfg.collection_fraction
        * cfg.eta_sys
    )


def zero_count_probability(N_ret: np.ndarray | float) -> np.ndarray:
    return np.exp(-np.asarray(N_ret, dtype=float))


def photon_regime(N_ret: float) -> str:
    p0 = float(math.exp(-N_ret))
    if p0 < 0.01:
        return "photon-rich"
    if p0 < 0.37:
        return "low-return"
    if p0 < 0.90:
        return "photon-starved"
    return "extreme photon-starved"


# -----------------------------
# Quantum boundary functions
# -----------------------------
def coherent_qfi(eta_s: np.ndarray, N_S: float) -> np.ndarray:
    return 4.0 * eta_s * N_S


def tmsv_pure_loss_qfi(eta_s: np.ndarray, N_S: float) -> np.ndarray:
    return 4.0 * eta_s * N_S * (N_S + 1.0) / (1.0 + 2.0 * (1.0 - eta_s) * N_S)


def G_Q(eta_s: np.ndarray, N_S: float) -> np.ndarray:
    """Equal-signal-energy TMSV/coherent pure-loss QFI advantage ratio."""
    return (N_S + 1.0) / (1.0 + 2.0 * (1.0 - eta_s) * N_S)


def G_total(eta_s: np.ndarray, N_S: float) -> np.ndarray:
    """Conservative equal-total-source-energy advantage ratio."""
    return (N_S + 1.0) / (2.0 * (1.0 + 2.0 * (1.0 - eta_s) * N_S))


def G_eff(eta_s: np.ndarray, gamma: np.ndarray, qcfg: QuantumBoundaryConfig) -> np.ndarray:
    return G_Q(eta_s, qcfg.N_S) * np.exp(-qcfg.diffusion_exponent_a * gamma)


def local_phase_variance_surrogate(
    eta_s: np.ndarray,
    gamma: np.ndarray,
    qcfg: QuantumBoundaryConfig,
) -> np.ndarray:
    """
    Local surrogate TMSV phase variance.

    Var_CS = 1 / (4 M eta_s N_S)
    Var_TMSV_sur = Var_CS / G_eff

    This is a local diagnostic, not a global attainable bound.
    """
    var_cs = 1.0 / (4.0 * qcfg.M * np.maximum(eta_s, 1e-15) * qcfg.N_S)
    return var_cs / np.maximum(G_eff(eta_s, gamma, qcfg), 1e-15)


def phase_wrapping_probability(
    eta_s: np.ndarray,
    gamma: np.ndarray,
    qcfg: QuantumBoundaryConfig,
) -> np.ndarray:
    """Gaussian local phase-wrapping risk: Pr(|Delta phi| > pi)."""
    var_phi = local_phase_variance_surrogate(eta_s, gamma, qcfg)
    sigma_phi = np.sqrt(np.maximum(var_phi, 1e-30))

    # erfc is scalar in math; vectorize it.
    erfc_vec = np.vectorize(math.erfc)
    return erfc_vec(np.pi / (np.sqrt(2.0) * sigma_phi))


def detector_admissible(N_ret: np.ndarray | float, qcfg: QuantumBoundaryConfig) -> np.ndarray:
    N_noise = (qcfg.dark_count_rate_Hz + qcfg.background_count_rate_Hz) * qcfg.gate_time_s
    N_ret_arr = np.asarray(N_ret, dtype=float)
    sbr = N_ret_arr / max(N_noise, 1e-30)
    return (sbr >= qcfg.SBR_min) & (N_ret_arr >= N_noise)


def detector_margin_log10(N_ret: np.ndarray | float, qcfg: QuantumBoundaryConfig) -> np.ndarray:
    """log10(SBR / SBR_min). Positive values pass the SBR screen."""
    N_noise = (qcfg.dark_count_rate_Hz + qcfg.background_count_rate_Hz) * qcfg.gate_time_s
    sbr = np.asarray(N_ret, dtype=float) / max(N_noise, 1e-30)
    return np.log10(np.maximum(sbr / qcfg.SBR_min, 1e-30))


# Classification codes.
# Lower codes are harder vetoes.
CLASS_LABELS = {
    0: "detector-limited",
    1: "idler-limited",
    2: "no local QFI advantage",
    3: "phase/wrapping stop",
    4: "guarded",
    5: "usable island",
}


def classify_grid(
    eta_grid: np.ndarray,
    gamma_grid: np.ndarray,
    detector_pass: bool,
    qcfg: QuantumBoundaryConfig,
) -> np.ndarray:
    """
    Multilayer diagnostic classification.

    This deliberately uses transparent screens:
      detector pass/fail
      idler preservation threshold
      G_eff > 1 local advantage
      phase-wrapping risk threshold
      guard band near Geff = 1
      otherwise usable

    Replace the guard-band logic with a real QZZB ratio if available.
    """
    Ge = G_eff(eta_grid, gamma_grid, qcfg)
    Pwrap = phase_wrapping_probability(eta_grid, gamma_grid, qcfg)

    cls = np.full_like(eta_grid, fill_value=5, dtype=int)

    # Layer 1: detector veto. For a fixed scenario this is all-or-nothing.
    if not detector_pass:
        cls[:, :] = 0
        return cls

    # Layer 2: idler veto.
    if qcfg.eta_i < qcfg.eta_i_min:
        cls[:, :] = 1
        return cls

    # Layer 3: local QFI advantage.
    cls[Ge <= 1.0] = 2

    # Layer 4: phase wrapping stop.
    cls[(Ge > 1.0) & (Pwrap >= qcfg.p_wrap_threshold)] = 3

    # Layer 5: guard band near boundary.
    cls[(Ge > 1.0) & (Ge < qcfg.guard_Geff_min) & (Pwrap < qcfg.p_wrap_threshold)] = 4

    # Remaining points are usable island.
    return cls


def layer_fractions(
    eta_grid: np.ndarray,
    gamma_grid: np.ndarray,
    scenario_Nret: float,
    qcfg: QuantumBoundaryConfig,
) -> Dict[str, float]:
    """
    Fraction of eta-Gamma grid surviving each sequential layer.

    This is useful for the "survival funnel" panel.
    """
    total = eta_grid.size
    fractions: Dict[str, float] = {}

    det = detector_admissible(scenario_Nret, qcfg)
    fractions["photon budget anchor"] = 1.0
    fractions["detector admissible"] = 1.0 if bool(det) else 0.0

    if not bool(det):
        fractions["idler preserved"] = 0.0
        fractions["G_eff > 1"] = 0.0
        fractions["phase-wrap safe"] = 0.0
        fractions["usable island"] = 0.0
        return fractions

    idler = qcfg.eta_i >= qcfg.eta_i_min
    fractions["idler preserved"] = 1.0 if idler else 0.0
    if not idler:
        fractions["G_eff > 1"] = 0.0
        fractions["phase-wrap safe"] = 0.0
        fractions["usable island"] = 0.0
        return fractions

    Ge = G_eff(eta_grid, gamma_grid, qcfg)
    Pwrap = phase_wrapping_probability(eta_grid, gamma_grid, qcfg)

    mask_adv = Ge > 1.0
    mask_wrap = mask_adv & (Pwrap < qcfg.p_wrap_threshold)
    mask_usable = mask_wrap & (Ge >= qcfg.guard_Geff_min)

    fractions["G_eff > 1"] = float(mask_adv.sum() / total)
    fractions["phase-wrap safe"] = float(mask_wrap.sum() / total)
    fractions["usable island"] = float(mask_usable.sum() / total)
    return fractions


# -----------------------------
# Scenario table
# -----------------------------
def scenario_verdict(
    Nret: float,
    eta_s: float,
    eta_i: float,
    gamma: float,
    pcfg: PhotonBudgetConfig,
    qcfg_base: QuantumBoundaryConfig,
) -> Tuple[str, str, float, float, float]:
    """Return verdict, reason, GQ, Geff, Pwrap for a single scenario row."""
    qcfg = QuantumBoundaryConfig(**vars(qcfg_base))
    qcfg.eta_i = eta_i

    det_pass = bool(detector_admissible(Nret, qcfg))
    gq = float(G_Q(np.array([eta_s]), qcfg.N_S)[0])
    ge = float(G_eff(np.array([eta_s]), np.array([gamma]), qcfg)[0])
    pw = float(phase_wrapping_probability(np.array([eta_s]), np.array([gamma]), qcfg)[0])

    if not det_pass:
        return "detector-limited", "SBR or Nret below detector screen", gq, ge, pw
    if eta_i < qcfg.eta_i_min:
        return "idler-limited", "idler efficiency below configured survival threshold", gq, ge, pw
    if ge <= 1.0:
        return "stop-extrapolation", "G_eff <= 1", gq, ge, pw
    if pw >= qcfg.p_wrap_threshold:
        return "stop-extrapolation", "phase-wrapping risk exceeds threshold", gq, ge, pw
    if ge < qcfg.guard_Geff_min:
        return "guarded", "near local boundary; QZZB/global audit recommended", gq, ge, pw
    return "local-valid", "passes configured multilayer screens", gq, ge, pw


def write_scenario_table(
    out_csv: Path,
    pcfg: PhotonBudgetConfig,
    qcfg: QuantumBoundaryConfig,
) -> None:
    """
    Write a scenario table with pass / guarded / fail examples.

    These examples are intentionally selected to make the boundary logic visible.
    """
    rows = [
        # label, P, lambda, eta_s, eta_i, Gamma, detector note
        ("A_pass_green_532", 30.0, 532.0, 0.90, 0.90, 0.50),
        ("B_photon_starved_pass_1064", 30.0, 1064.0, 0.90, 0.90, 0.50),
        ("C_detector_limited_1550_high_dark", 30.0, 1550.0, 0.90, 0.90, 0.50),
        ("D_signal_loss_fail", 30.0, 532.0, 0.45, 0.90, 0.50),
        ("E_diffusion_fail", 30.0, 532.0, 0.90, 0.90, 2.00),
        ("F_idler_limited", 30.0, 532.0, 0.90, 0.50, 0.50),
        ("G_guarded_near_boundary", 30.0, 532.0, 0.70, 0.90, 0.55),
    ]

    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "case",
            "P_MPa",
            "lambda_nm",
            "N_ret",
            "P0",
            "photon_regime",
            "eta_s",
            "eta_i",
            "Gamma",
            "G_Q",
            "G_eff",
            "P_wrap",
            "detector_margin_log10_SBR_over_SBRmin",
            "verdict",
            "reason",
        ])

        for label, P, lam, eta_s, eta_i, gamma in rows:
            # For the detector-limited row, intentionally use a harsher detector setting
            # so the table contains a real detector-fail example.
            local_qcfg = QuantumBoundaryConfig(**vars(qcfg))
            if "detector_limited" in label:
                local_qcfg.dark_count_rate_Hz = 1.0e4
                local_qcfg.SBR_min = 5.0

            nret = float(rayleigh_return_photons(P, lam, pcfg))
            p0 = float(zero_count_probability(nret))
            verdict, reason, gq, ge, pw = scenario_verdict(
                nret, eta_s, eta_i, gamma, pcfg, local_qcfg
            )
            margin = float(detector_margin_log10(nret, local_qcfg))
            writer.writerow([
                label,
                f"{P:.3g}",
                f"{lam:.3g}",
                f"{nret:.6g}",
                f"{p0:.6g}",
                photon_regime(nret),
                f"{eta_s:.3g}",
                f"{eta_i:.3g}",
                f"{gamma:.3g}",
                f"{gq:.6g}",
                f"{ge:.6g}",
                f"{pw:.6g}",
                f"{margin:.6g}",
                verdict,
                reason,
            ])


# -----------------------------
# Plotting
# -----------------------------
def make_multilayer_figure(
    out_png: Path,
    out_pdf: Path,
    pcfg: PhotonBudgetConfig,
    qcfg: QuantumBoundaryConfig,
    scenario_pressure_MPa: float = 30.0,
    scenario_lambda_nm: float = 1064.0,
) -> None:
    out_png.parent.mkdir(parents=True, exist_ok=True)

    # Grids
    eta_vals = np.linspace(0.2, 1.0, 161)
    gamma_vals = np.linspace(0.0, 3.0, 151)
    ETA, GAMMA = np.meshgrid(eta_vals, gamma_vals)

    scenario_Nret = float(rayleigh_return_photons(scenario_pressure_MPa, scenario_lambda_nm, pcfg))
    det_pass = bool(detector_admissible(scenario_Nret, qcfg))
    cls = classify_grid(ETA, GAMMA, det_pass, qcfg)

    # Photon-budget map grid
    P_vals = np.linspace(1.0, 35.0, 80)
    L_vals = np.linspace(532.0, 1550.0, 80)
    PP, LL = np.meshgrid(P_vals, L_vals)
    NRET = rayleigh_return_photons(PP, LL, pcfg)
    P0 = zero_count_probability(NRET)
    det_margin = detector_margin_log10(NRET, qcfg)

    # Survival funnel
    fracs = layer_fractions(ETA, GAMMA, scenario_Nret, qcfg)
    stages = list(fracs.keys())
    vals = np.array([fracs[s] for s in stages])

    # Classification colors
    colors = [
        "#4d4d4d",  # detector-limited
        "#6a51a3",  # idler-limited
        "#d73027",  # no local QFI advantage
        "#fc8d59",  # phase/wrapping stop
        "#fee08b",  # guarded
        "#1a9850",  # usable island
    ]
    cmap = ListedColormap(colors)
    norm = BoundaryNorm(np.arange(-0.5, 6.5, 1.0), cmap.N)

    fig = plt.figure(figsize=(15.5, 10.5))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.0], width_ratios=[1.05, 1.0])

    # Panel A: classification map
    ax0 = fig.add_subplot(gs[0, 0])
    im0 = ax0.pcolormesh(ETA, GAMMA, cls, cmap=cmap, norm=norm, shading="auto")
    Ge = G_eff(ETA, GAMMA, qcfg)
    ax0.contour(ETA, GAMMA, Ge, levels=[1.0], colors="white", linewidths=2.0)
    ax0.contour(ETA, GAMMA, Ge, levels=[qcfg.guard_Geff_min], colors="black", linewidths=1.2, linestyles="--")
    ax0.axvline(0.5, color="white", linestyle=":", linewidth=1.5)
    ax0.axvline(0.75, color="black", linestyle=":", linewidth=1.5)
    ax0.set_xlabel(r"Signal transmittance $\eta_s$")
    ax0.set_ylabel(r"Accumulated phase diffusion $\Gamma$")
    ax0.set_title(
        "A. Multilayer survival classification\n"
        f"scenario: P={scenario_pressure_MPa:g} MPa, λ={scenario_lambda_nm:g} nm, "
        f"Nret={scenario_Nret:.3g}, detector={'pass' if det_pass else 'fail'}"
    )
    legend_handles = [Patch(facecolor=colors[k], edgecolor="none", label=CLASS_LABELS[k]) for k in range(6)]
    ax0.legend(handles=legend_handles, loc="upper right", fontsize=8, frameon=True)

    # Panel B: survival funnel
    ax1 = fig.add_subplot(gs[0, 1])
    y = np.arange(len(stages))
    ax1.barh(y, vals)
    ax1.set_yticks(y, stages)
    ax1.invert_yaxis()
    ax1.set_xlim(0.0, 1.05)
    ax1.set_xlabel("Fraction of ηs–Γ grid surviving")
    ax1.set_title("B. Sequential survival funnel")
    for yy, v in zip(y, vals):
        ax1.text(v + 0.02, yy, f"{100*v:.1f}%", va="center", fontsize=9)
    ax1.grid(axis="x", alpha=0.25)

    # Panel C: Rayleigh photon budget / zero-count map
    ax2 = fig.add_subplot(gs[1, 0])
    im2 = ax2.pcolormesh(PP, LL, np.log10(np.maximum(NRET, 1e-12)), shading="auto")
    cb2 = fig.colorbar(im2, ax=ax2)
    cb2.set_label(r"$\log_{10} N_{\rm ret}$")
    # Regime contours in P0: 0.01, 0.37, 0.90
    try:
        cs = ax2.contour(PP, LL, P0, levels=[0.01, 0.37, 0.90], colors="white", linewidths=1.2)
        ax2.clabel(cs, inline=True, fontsize=8, fmt={0.01: "P0=0.01", 0.37: "P0=0.37", 0.90: "P0=0.90"})
    except Exception:
        pass
    ax2.scatter([scenario_pressure_MPa], [scenario_lambda_nm], s=70, marker="*", color="red", edgecolor="black", zorder=10)
    ax2.set_xlabel("Pressure P (MPa)")
    ax2.set_ylabel(r"Wavelength $\lambda_0$ (nm)")
    ax2.set_title("C. Rayleigh photon-budget layer")

    # Panel D: detector admissibility margin
    ax3 = fig.add_subplot(gs[1, 1])
    im3 = ax3.pcolormesh(PP, LL, det_margin, shading="auto")
    cb3 = fig.colorbar(im3, ax=ax3)
    cb3.set_label(r"$\log_{10}[(SBR)/(SBR_{\min})]$")
    ax3.contour(PP, LL, det_margin, levels=[0.0], colors="white", linewidths=2.0)
    ax3.scatter([scenario_pressure_MPa], [scenario_lambda_nm], s=70, marker="*", color="red", edgecolor="black", zorder=10)
    ax3.set_xlabel("Pressure P (MPa)")
    ax3.set_ylabel(r"Wavelength $\lambda_0$ (nm)")
    ax3.set_title("D. Detector-admissibility layer")

    fig.suptitle(
        "Multilayer survival map for photon-starved quantum-enhanced Rayleigh-Doppler velocimetry",
        fontsize=14,
        y=0.99,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.965])

    fig.savefig(out_png, dpi=300)
    fig.savefig(out_pdf)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a multilayer survival map for the quantum Doppler boundary framework."
    )
    parser.add_argument("--outdir", type=str, default="outputs", help="Output directory.")
    parser.add_argument("--pressure", type=float, default=30.0, help="Scenario pressure in MPa.")
    parser.add_argument("--wavelength", type=float, default=1064.0, help="Scenario wavelength in nm.")

    # Photon budget settings.
    parser.add_argument("--pulse-energy", type=float, default=1e-6, help="Pulse energy [J].")
    parser.add_argument("--collection-fraction", type=float, default=1e-7, help="Omega / 4pi.")
    parser.add_argument("--eta-sys", type=float, default=0.05, help="Macroscopic system efficiency.")

    # Quantum settings.
    parser.add_argument("--N-S", type=float, default=100.0, help="Signal-mode photon number N_S.")
    parser.add_argument("--M", type=float, default=1.0, help="Independent samples M.")
    parser.add_argument("--a", type=float, default=1.0, help="Diffusion exponent coefficient a in exp(-a Gamma).")
    parser.add_argument("--eta-i", type=float, default=0.90, help="Idler efficiency eta_i.")
    parser.add_argument("--eta-i-min", type=float, default=0.75, help="Idler survival threshold.")
    parser.add_argument("--pwrap-threshold", type=float, default=1e-3, help="Phase-wrapping probability threshold.")
    parser.add_argument("--guard-Geff-min", type=float, default=1.25, help="Guard band threshold above Geff=1.")

    # Detector settings.
    parser.add_argument("--dark-rate", type=float, default=100.0, help="Dark count rate [Hz].")
    parser.add_argument("--background-rate", type=float, default=0.0, help="Background count rate [Hz].")
    parser.add_argument("--gate-time", type=float, default=1e-6, help="Gate time [s].")
    parser.add_argument("--SBR-min", type=float, default=5.0, help="Minimum signal-to-background ratio.")

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    pcfg = PhotonBudgetConfig(
        pulse_energy_J=args.pulse_energy,
        collection_fraction=args.collection_fraction,
        eta_sys=args.eta_sys,
    )
    qcfg = QuantumBoundaryConfig(
        N_S=args.N_S,
        M=args.M,
        diffusion_exponent_a=args.a,
        eta_i=args.eta_i,
        eta_i_min=args.eta_i_min,
        p_wrap_threshold=args.pwrap_threshold,
        guard_Geff_min=args.guard_Geff_min,
        dark_count_rate_Hz=args.dark_rate,
        background_count_rate_Hz=args.background_rate,
        gate_time_s=args.gate_time,
        SBR_min=args.SBR_min,
    )

    fig_png = outdir / "multilayer_survival_map.png"
    fig_pdf = outdir / "multilayer_survival_map.pdf"
    table_csv = outdir / "scenario_boundary_table.csv"

    make_multilayer_figure(
        fig_png,
        fig_pdf,
        pcfg,
        qcfg,
        scenario_pressure_MPa=args.pressure,
        scenario_lambda_nm=args.wavelength,
    )
    write_scenario_table(table_csv, pcfg, qcfg)

    # Print manuscript-useful anchors.
    nret_anchor = [
        (lam, float(rayleigh_return_photons(30.0, lam, pcfg)))
        for lam in [532.0, 633.0, 1064.0, 1550.0]
    ]
    print("Generated:")
    print(f"  {fig_png}")
    print(f"  {fig_pdf}")
    print(f"  {table_csv}")
    print("")
    print("30 MPa constrained-case Rayleigh anchors:")
    for lam, nret in nret_anchor:
        print(f"  lambda={lam:6.1f} nm: N_ret={nret:.6g}, P0={math.exp(-nret):.6g}, regime={photon_regime(nret)}")
    print("")
    print("Analytic thresholds:")
    print("  Equal-signal-energy pure-loss SQL boundary: eta_s = 0.5")
    print(f"  Equal-total-source-energy threshold: eta_c = {0.75 + 1.0/(4.0*qcfg.N_S):.6g} for N_S={qcfg.N_S:g}")
    print("  Loss-diffusion boundary: Gamma_max = ln G_Q(eta_s, N_S) / a")


if __name__ == "__main__":
    main()
