#!/usr/bin/env python3
"""Compact v8.1 audit package for conditional Rayleigh-Doppler/TMSV boundaries.

This single-file package intentionally keeps the reproducibility interface small.
It produces the core v8.1 stress-test outputs used by the manuscript:
  1) dense-gas photon-budget sensitivity with an explicit correction factor;
  2) low-photon microscopic-mode maps linked to the Rayleigh return budget;
  3) exact finite-Fock number-dephasing QFI with optional idler phase diffusion;
  4) pure-loss formula audit columns and convergence/tail diagnostics.

The finite-Fock calculation is a diagnostic audit, not a high-NS certificate.
"""
from __future__ import annotations

import argparse
import json
import math
from functools import lru_cache
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.linalg import eigh

# -----------------------------------------------------------------------------
# Analytic budget and benchmark formulas
# -----------------------------------------------------------------------------

H = 6.62607015e-34
C = 299792458.0
KB = 1.380649e-23
NA = 6.02214076e23
R = 8.31446261815324


def emitted_photons(pulse_energy_j: float, wavelength_m: float) -> float:
    return pulse_energy_j / (H * C / wavelength_m)


def ideal_gas_number_density(P_pa: float, T_k: float = 298.15, compressibility_Z: float = 1.0) -> float:
    """Number density in m^-3.  A real-gas Z may be supplied externally."""
    return P_pa / (compressibility_Z * KB * T_k)


def methane_rayleigh_cross_section(lambda0_m: float, n_ref: float = 1.000444, king_factor: float = 1.04) -> float:
    """First-order molecular Rayleigh cross-section proxy.

    This is retained as a baseline proxy.  Dense-gas correlations and Cabannes/
    composition corrections must be represented by the dense_factor in the budget.
    """
    return (24.0 * math.pi**3 / lambda0_m**4) * ((n_ref**2 - 1.0) / (n_ref**2 + 2.0))**2 * king_factor


def rayleigh_return_photons(
    pressure_mpa: float,
    wavelength_nm: float,
    pulse_energy_j: float = 1e-6,
    collection_fraction: float = 1e-7,
    eta_sys: float = 0.05,
    path_length_m: float = 0.01,
    temperature_k: float = 298.15,
    dense_factor: float = 1.0,
    compressibility_Z: float = 1.0,
) -> float:
    """First-order Rayleigh return photon budget with explicit dense-gas factor.

    The compact package uses the manuscript's constrained 30 MPa anchors as
    reference values and scales them linearly with pressure and C_dense.  This
    avoids presenting a calibrated dense-gas scattering calculation while still
    auditing sensitivity to multiplicative dense-gas corrections.
    """
    anchors_30mpa = {532: 1.30, 633: 0.772, 1064: 0.163, 1550: 0.0526}
    lam = min(anchors_30mpa, key=lambda x: abs(x - int(round(wavelength_nm))))
    return anchors_30mpa[lam] * (pressure_mpa / 30.0) * dense_factor


def gq_pure_loss(eta_s: float | np.ndarray, Ns: float) -> float | np.ndarray:
    """Fock-consistent equal-signal-energy pure-loss TMSV/coherent QFI ratio."""
    eta_s = np.asarray(eta_s)
    return (Ns + 1.0) / (1.0 + (1.0 - eta_s) * Ns)


def gq_equal_total(eta_s: float | np.ndarray, Ns: float) -> float | np.ndarray:
    return 0.5 * gq_pure_loss(eta_s, Ns)


def geff_candidate(eta_s: float | np.ndarray, Ns: float, Gamma: float | np.ndarray) -> float | np.ndarray:
    """Optimistic candidate envelope only; not an exact non-Gaussian QFI."""
    return gq_pure_loss(eta_s, Ns) * np.exp(-np.asarray(Gamma))


def coherent_qfi(eta_s: float, Ns: float) -> float:
    return 4.0 * eta_s * Ns


def tmsv_tail_probability(Ns: float, cutoff: int) -> float:
    lam = Ns / (Ns + 1.0)
    return float(lam**cutoff)


def truncated_tmsv_mean_ns(Ns: float, cutoff: int) -> float:
    lam = Ns / (Ns + 1.0)
    n = np.arange(cutoff, dtype=float)
    p = (1.0 - lam) * lam**n
    p /= p.sum()
    return float(np.sum(n * p))

# -----------------------------------------------------------------------------
# Finite-Fock exact QFI utilities
# -----------------------------------------------------------------------------


def hermitize(A: np.ndarray) -> np.ndarray:
    return 0.5 * (A + A.conj().T)


def tmsv_state_vector(Ns: float, cutoff: int) -> np.ndarray:
    lam = Ns / (Ns + 1.0)
    coeff = np.array([math.sqrt((1.0 - lam) * lam**n) for n in range(cutoff)], dtype=complex)
    coeff /= np.linalg.norm(coeff)
    vec = np.zeros(cutoff * cutoff, dtype=complex)
    for n, c in enumerate(coeff):
        vec[n * cutoff + n] = c
    return vec


@lru_cache(maxsize=256)
def loss_kraus_single(cutoff: int, eta: float) -> tuple[np.ndarray, ...]:
    eta = float(eta)
    ops = []
    for lost in range(cutoff):
        K = np.zeros((cutoff, cutoff), dtype=complex)
        for n in range(lost, cutoff):
            K[n - lost, n] = math.sqrt(math.comb(n, lost)) * ((1.0 - eta) ** (lost / 2.0)) * (eta ** ((n - lost) / 2.0))
        ops.append(K)
    return tuple(ops)


def apply_two_mode_loss(rho: np.ndarray, cutoff: int, eta_s: float, eta_i: float) -> np.ndarray:
    Ks = loss_kraus_single(cutoff, float(eta_s))
    Ki = loss_kraus_single(cutoff, float(eta_i))
    R4 = rho.reshape(cutoff, cutoff, cutoff, cutoff)
    Rs = np.zeros_like(R4, dtype=complex)
    for A in Ks:
        Rs += np.einsum("sa,aibj,tb->sitj", A, R4, A.conj(), optimize=True)
    Ri = np.zeros_like(Rs, dtype=complex)
    for B in Ki:
        Ri += np.einsum("ia,satb,jb->sitj", B, Rs, B.conj(), optimize=True)
    out = Ri.reshape(cutoff * cutoff, cutoff * cutoff)
    tr = out.trace().real
    if tr > 0:
        out /= tr
    return hermitize(out)


def apply_two_mode_phase_diffusion(rho: np.ndarray, cutoff: int, gamma_s: float, gamma_i: float = 0.0) -> np.ndarray:
    """Exact number dephasing on signal and idler modes in finite Fock basis."""
    R4 = rho.reshape(cutoff, cutoff, cutoff, cutoff).copy()
    ns = np.arange(cutoff)[:, None]
    ms = np.arange(cutoff)[None, :]
    fs = np.exp(-0.5 * float(gamma_s) * (ns - ms) ** 2)
    ni = np.arange(cutoff)[:, None]
    mi = np.arange(cutoff)[None, :]
    fi = np.exp(-0.5 * float(gamma_i) * (ni - mi) ** 2)
    R4 *= fs[:, None, :, None] * fi[None, :, None, :]
    return hermitize(R4.reshape(cutoff * cutoff, cutoff * cutoff))


def number_operator_signal(cutoff: int) -> np.ndarray:
    return np.kron(np.diag(np.arange(cutoff, dtype=float)), np.eye(cutoff))


def qfi_unitary_generator(rho: np.ndarray, generator: np.ndarray, eig_tol: float = 1e-12) -> float:
    vals, vecs = eigh(hermitize(rho))
    vals = np.maximum(vals.real, 0.0)
    G = vecs.conj().T @ generator @ vecs
    li, lj = vals[:, None], vals[None, :]
    denom = li + lj
    mask = denom > eig_tol
    weights = np.zeros_like(denom, dtype=float)
    weights[mask] = ((li - lj)[mask] ** 2) / denom[mask]
    F = 2.0 * np.sum(weights * np.abs(G) ** 2)
    return float(max(np.real_if_close(F).real, 0.0))


def exact_qfi_ratio(Ns: float, cutoff: int, eta_s: float, eta_i: float, gamma_s: float, gamma_i: float = 0.0) -> tuple[float, float]:
    psi = tmsv_state_vector(Ns, cutoff)
    rho = np.outer(psi, psi.conj())
    rho = apply_two_mode_loss(rho, cutoff, eta_s, eta_i)
    if gamma_s != 0.0 or gamma_i != 0.0:
        rho = apply_two_mode_phase_diffusion(rho, cutoff, gamma_s, gamma_i)
    F = qfi_unitary_generator(rho, number_operator_signal(cutoff))
    Fc = coherent_qfi(eta_s, Ns)
    return F, F / Fc if Fc > 0 else float("nan")

# -----------------------------------------------------------------------------
# Reproducibility workflows
# -----------------------------------------------------------------------------


def run_photon_budget_dense(outdir: Path) -> pd.DataFrame:
    rows = []
    for pressure in [1, 5, 10, 20, 30, 35]:
        for lam in [532, 633, 1064, 1550]:
            for dense in [0.1, 0.3, 1.0, 3.0, 10.0]:
                Nret = rayleigh_return_photons(pressure, lam, dense_factor=dense)
                for chi in [1.0, 0.3, 0.1]:
                    rows.append({
                        "pressure_MPa": pressure,
                        "wavelength_nm": lam,
                        "dense_factor": dense,
                        "Nret": Nret,
                        "zero_count_probability": math.exp(-Nret),
                        "mode_fraction_chi": chi,
                        "N_mode_max": chi * Nret,
                        "scope_note": "N_mode_max = chi*Nret is an upper occupancy anchor for a collected coherent signal mode; high-NS maps are target-channel topology, not predicted pipeline points.",
                    })
    df = pd.DataFrame(rows)
    df.to_csv(outdir / "photon_budget_dense_mode_bridge.csv", index=False)

    figdir = outdir / "figures"
    figdir.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(7.2, 4.8))
    for dense in [0.1, 1.0, 10.0]:
        sub = df[(df.wavelength_nm == 1064) & (df.mode_fraction_chi == 1.0) & (df.dense_factor == dense)]
        plt.plot(sub.pressure_MPa, sub.Nret, marker="o", label=f"C_dense={dense:g}")
    plt.yscale("log")
    plt.xlabel("Pressure (MPa)")
    plt.ylabel("Expected Rayleigh return photons per pulse")
    plt.title("Dense-gas correction sensitivity in the Rayleigh budget")
    plt.legend()
    plt.tight_layout()
    plt.savefig(figdir / "fig_dense_factor_photon_budget.png", dpi=220)
    plt.close()
    return df


def run_exact_low_ns_idler_phase(outdir: Path, cutoff: int = 6) -> pd.DataFrame:
    Ns_values = [0.1, 0.5, 1.0]
    eta_values = [0.5, 0.9]
    gamma_values = [0.0, 0.5, 1.0]
    eta_i_values = [1.0, 0.3]
    idler_phase_factors = [0.0, 1.0]
    rows = []
    for Ns in Ns_values:
        NS_cut = truncated_tmsv_mean_ns(Ns, cutoff)
        tail = tmsv_tail_probability(Ns, cutoff)
        for eta_s in eta_values:
            for eta_i in eta_i_values:
                for gamma_s in gamma_values:
                    for alpha_i in idler_phase_factors:
                        gamma_i = alpha_i * gamma_s
                        F, G = exact_qfi_ratio(Ns, cutoff, eta_s, eta_i, gamma_s, gamma_i)
                        Genv = geff_candidate(eta_s, Ns, gamma_s)
                        rows.append({
                            "NS_target": Ns,
                            "cutoff": cutoff,
                            "NS_cut": NS_cut,
                            "NS_cut_over_NS_target": NS_cut / Ns,
                            "tail_probability": tail,
                            "eta_s": eta_s,
                            "eta_i": eta_i,
                            "Gamma_s": gamma_s,
                            "Gamma_i": gamma_i,
                            "idler_phase_factor": alpha_i,
                            "F_Q_exact_TMSV": F,
                            "F_Q_coherent": coherent_qfi(eta_s, Ns),
                            "G_exact": G,
                            "G_eff_candidate_envelope": Genv,
                            "G_exact_gt_1": bool(G > 1.0),
                            "G_eff_gt_1": bool(Genv > 1.0),
                            "scope_note": "low-photon finite-Fock exact QFI; Gamma_i represents idler phase-reference drift stress test",
                        })
    df = pd.DataFrame(rows)
    df.to_csv(outdir / "exact_low_ns_idler_phase_qfi.csv", index=False)

    figdir = outdir / "figures"
    figdir.mkdir(parents=True, exist_ok=True)
    # Low-NS map: max over tested eta_i? Use eta_i=1, Gamma_i=0, eta_s=0.9 vs Gamma.
    plt.figure(figsize=(7.2, 4.8))
    for Ns in Ns_values:
        sub = df[(df.NS_target == Ns) & (df.eta_s == 0.9) & (df.eta_i == 1.0) & (df.idler_phase_factor == 0.0)].sort_values("Gamma_s")
        plt.plot(sub.Gamma_s, sub.G_exact, marker="o", label=f"Ns={Ns:g}")
    plt.axhline(1.0, color="k", linewidth=1.0)
    plt.xlabel("Signal phase diffusion Gamma_s")
    plt.ylabel("Exact QFI advantage ratio")
    plt.title("Low-photon exact dephasing QFI: physically linked Ns regime")
    plt.legend()
    plt.tight_layout()
    plt.savefig(figdir / "fig_low_ns_exact_qfi.png", dpi=220)
    plt.close()

    # Idler phase drift veto for Ns=1.0, eta_s=0.9
    plt.figure(figsize=(7.2, 4.8))
    for eta_i in eta_i_values:
        for alpha in idler_phase_factors:
            sub = df[(df.NS_target == 1.0) & (df.eta_s == 0.9) & (df.eta_i == eta_i) & (df.idler_phase_factor == alpha)].sort_values("Gamma_s")
            plt.plot(sub.Gamma_s, sub.G_exact, marker="o", linestyle="-" if alpha == 0 else "--", label=f"eta_i={eta_i:g}, Gamma_i={alpha:g} Gamma_s")
    plt.axhline(1.0, color="k", linewidth=1.0)
    plt.xlabel("Signal phase diffusion Gamma_s")
    plt.ylabel("Exact QFI advantage ratio")
    plt.title("Idler phase-reference drift stress test")
    plt.legend(fontsize=7, ncol=2)
    plt.tight_layout()
    plt.savefig(figdir / "fig_idler_phase_noise_veto.png", dpi=220)
    plt.close()
    return df


def run_thermal_excess_placeholder(outdir: Path) -> pd.DataFrame:
    """Small excess-noise stress table.

    The optical room-temperature thermal occupancy is not assumed to be 1e-3.
    This table treats nth as a generic excess-noise stress coordinate and records
    whether it should be interpreted at detector/background or microscopic-channel
    level in later instrument models.
    """
    rows = []
    for nth in [0.0, 1e-6, 1e-5, 1e-4, 1e-3]:
        for Ns in [0.1, 0.5, 1.0, 1.5]:
            rows.append({
                "n_th_or_excess_mode_noise": nth,
                "NS_target": Ns,
                "noise_to_signal_ratio": nth / Ns if Ns > 0 else float("nan"),
                "interpretation": "excess-noise stress coordinate, not a room-temperature optical blackbody claim",
                "recommendation": "model as detector/background unless a same-spatiotemporal-mode thermal-loss mechanism is specified",
            })
    df = pd.DataFrame(rows)
    df.to_csv(outdir / "thermal_excess_noise_stress_scope.csv", index=False)
    return df


def write_summary(outdir: Path, budget: pd.DataFrame, qfi: pd.DataFrame) -> None:
    highest = qfi.copy()
    summary = {
        "version": "v8.1 compact audit",
        "photon_budget": {
            "dense_factor_range": [float(budget.dense_factor.min()), float(budget.dense_factor.max())],
            "min_Nret": float(budget.Nret.min()),
            "max_Nret": float(budget.Nret.max()),
            "mode_bridge": "N_mode = chi_mode*Nret; high-NS maps are target-channel topology, not predicted pipeline occupancy",
        },
        "exact_qfi": {
            "points": int(len(highest)),
            "fraction_G_exact_gt_1": float(highest.G_exact_gt_1.mean()),
            "fraction_G_exact_gt_1_when_Gamma_s_gt_0": float(highest[highest.Gamma_s > 0].G_exact_gt_1.mean()),
            "max_tail_probability": float(highest.tail_probability.max()),
            "cutoff": int(highest.cutoff.max()),
            "scope": "low-photon finite-Fock diagnostic with idler phase-noise stress test",
        },
        "G_eff": "retained only as optimistic candidate envelope, not exact non-Gaussian QFI",
        "macro_uncertainty_appendix": "removed/downgraded from manuscript; no single-seed toy curve is used for claims",
    }
    (outdir / "SUMMARY_v8_1.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    lines = ["# v8.1 compact audit summary", "", json.dumps(summary, indent=2)]
    (outdir / "SUMMARY_v8_1.md").write_text("\n".join(lines), encoding="utf-8")


def run_all(outdir: Path) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    budget = run_photon_budget_dense(outdir)
    qfi = run_exact_low_ns_idler_phase(outdir)
    run_thermal_excess_placeholder(outdir)
    write_summary(outdir, budget, qfi)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", default="outputs_v8_1_compact")
    args = parser.parse_args()
    run_all(Path(args.outdir))
    print(f"v8.1 outputs written to {Path(args.outdir).resolve()}")


if __name__ == "__main__":
    main()
