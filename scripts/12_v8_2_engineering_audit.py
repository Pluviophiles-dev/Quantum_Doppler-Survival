#!/usr/bin/env python3
"""v8.2 engineering-audit add-on for the unified quantum Doppler package.

This script keeps all earlier v8/v8.1 scripts intact and adds engineering-facing
checks requested by a conservative review:

1. multimode Rayleigh return -> single-mode signal occupancy penalty;
2. idler delay-line transmission/phase-stability requirement;
3. gate-based detector dark/background probability and classical receiver FI;
4. phase-structure-function bridge for the diffusion coordinate Gamma;
5. PR/EOS multiplicative sensitivity for photon-regime labels;
6. finite-Fock QZZB cutoff convergence visibility.

The outputs are audit tables and figures. They are not end-to-end validation.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from qdboundary.fock import prepare_noisy_tmsv_density, tmsv_tail_probability
from qdboundary.qzzb import qzzb_phase_bound
from qdboundary_enhanced.detector_fi import DetectorModel, max_fi_over_phase


def ensure_figdir(outdir: Path) -> Path:
    figdir = outdir / "figures"
    figdir.mkdir(parents=True, exist_ok=True)
    return figdir


def mode_bridge_scan(outdir: Path) -> pd.DataFrame:
    """Audit how multimode Rayleigh photons shrink under single-mode projection."""
    nret_anchors = {
        "532nm_30MPa_constrained": 1.30,
        "1064nm_30MPa_constrained": 0.163,
        "1550nm_30MPa_constrained": 0.0526,
    }
    dense_factors = [0.1, 0.3, 1.0, 3.0, 10.0]
    eta_spatial_values = [1.0, 1e-1, 1e-2, 1e-3, 1e-4]
    eta_spectral_values = [1.0, 0.3, 0.1]
    eta_temporal_values = [1.0, 0.3, 0.1]
    eta_pol = 0.5
    eta_overlap = 0.8

    rows: list[dict[str, float | str | bool]] = []
    for anchor, nret0 in nret_anchors.items():
        for cdense in dense_factors:
            nret = nret0 * cdense
            for eta_sp in eta_spatial_values:
                for eta_spec in eta_spectral_values:
                    for eta_temp in eta_temporal_values:
                        chi = eta_sp * eta_spec * eta_temp * eta_pol * eta_overlap
                        nmode = nret * chi
                        rows.append({
                            "anchor": anchor,
                            "Nret_baseline": nret0,
                            "dense_factor": cdense,
                            "Nret_after_dense_factor": nret,
                            "eta_spatial": eta_sp,
                            "eta_spectral": eta_spec,
                            "eta_temporal": eta_temp,
                            "eta_polarization": eta_pol,
                            "eta_overlap": eta_overlap,
                            "chi_mode_total": chi,
                            "N_mode": nmode,
                            "sub_unity_mode": bool(nmode < 1.0),
                            "note": "N_mode is the single-mode occupancy upper anchor; high-NS plots are target-channel topology unless an added accumulation/cavity model is specified.",
                        })
    df = pd.DataFrame(rows)
    df.to_csv(outdir / "mode_bridge_multimode_to_singlemode.csv", index=False)

    figdir = ensure_figdir(outdir)
    plt.figure(figsize=(7.4, 4.9))
    for anchor, nret0 in nret_anchors.items():
        xs = np.array(eta_spatial_values)
        ys = []
        for eta_sp in xs:
            chi = eta_sp * 0.3 * 0.3 * eta_pol * eta_overlap
            ys.append(nret0 * chi)
        plt.plot(xs, ys, marker="o", label=anchor.replace("_", " "))
    plt.xscale("log")
    plt.yscale("log")
    plt.axhline(1.0, color="k", linewidth=1.0)
    plt.xlabel("Spatial single-mode coupling efficiency")
    plt.ylabel("Mode occupancy N_mode (with eta_spec=eta_temp=0.3)")
    plt.title("Multimode Rayleigh return to single-mode occupancy penalty")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(figdir / "fig_mode_bridge_singlemode_penalty.png", dpi=220)
    plt.close()
    return df


def idler_delay_scan(outdir: Path) -> pd.DataFrame:
    """Translate static idler efficiency into length/time engineering requirements."""
    rows = []
    fiber_loss_db_per_km = 0.2
    c_fiber_m_s = 2.0e8
    lengths_km = [0, 0.01, 0.1, 1, 5, 10, 25, 50]
    memory_times_s = [0, 1e-9, 10e-9, 100e-9, 1e-6, 10e-6, 100e-6, 1e-3]
    coherence_times_s = [1e-6, 10e-6, 100e-6, 1e-3]
    eta_i0 = 0.95

    for Lkm in lengths_km:
        eta_fiber = eta_i0 * 10 ** (-fiber_loss_db_per_km * Lkm / 10.0)
        tau = (Lkm * 1000.0) / c_fiber_m_s
        for tau_c in coherence_times_s:
            gamma_i = 2.0 * tau / tau_c if tau_c > 0 else float("inf")
            rows.append({
                "model": "fiber_delay",
                "length_km": Lkm,
                "delay_s": tau,
                "fiber_loss_db_per_km": fiber_loss_db_per_km,
                "eta_i0_coupling": eta_i0,
                "eta_i_transmission": eta_fiber,
                "idler_phase_coherence_time_s": tau_c,
                "Gamma_i_phase_drift_proxy": gamma_i,
                "note": "eta_i(L)=eta_i0*10^(-alpha L/10); Gamma_i is a phase-reference drift stress coordinate, not a calibrated fiber-noise model.",
            })
    for tau in memory_times_s:
        for lifetime in [1e-6, 10e-6, 100e-6, 1e-3, 1e-2]:
            eta_mem = eta_i0 * math.exp(-tau / lifetime) if lifetime > 0 else 0.0
            rows.append({
                "model": "memory_delay",
                "length_km": float("nan"),
                "delay_s": tau,
                "fiber_loss_db_per_km": float("nan"),
                "eta_i0_coupling": eta_i0,
                "eta_i_transmission": eta_mem,
                "idler_phase_coherence_time_s": lifetime,
                "Gamma_i_phase_drift_proxy": 2.0 * tau / lifetime if lifetime > 0 else float("inf"),
                "note": "eta_i(t)=eta_i0*exp(-t/tau_life); static eta_i scans define requirements rather than field-ready assumptions.",
            })
    df = pd.DataFrame(rows)
    df.to_csv(outdir / "idler_delay_line_engineering_scan.csv", index=False)

    figdir = ensure_figdir(outdir)
    plt.figure(figsize=(7.4, 4.9))
    sub = df[df.model == "fiber_delay"].drop_duplicates("length_km")
    plt.plot(sub.length_km, sub.eta_i_transmission, marker="o")
    plt.xlabel("Fiber delay length (km)")
    plt.ylabel("Idler transmission eta_i")
    plt.title("Idler delay-line transmission requirement")
    plt.tight_layout()
    plt.savefig(figdir / "fig_idler_delay_transmission.png", dpi=220)
    plt.close()

    plt.figure(figsize=(7.4, 4.9))
    for tau_c in coherence_times_s:
        sub = df[(df.model == "fiber_delay") & (df.idler_phase_coherence_time_s == tau_c)]
        plt.plot(sub.length_km, sub.Gamma_i_phase_drift_proxy, marker="o", label=f"tau_c={tau_c:g}s")
    plt.yscale("log")
    plt.xlabel("Fiber delay length (km)")
    plt.ylabel("Gamma_i phase-drift proxy")
    plt.title("Idler phase-reference stability requirement")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(figdir / "fig_idler_phase_stability_requirement.png", dpi=220)
    plt.close()
    return df


def detector_gate_classical_baseline(outdir: Path) -> pd.DataFrame:
    """Convert dark/background rates to per-gate probabilities and add classical FI."""
    rows = []
    nrets = [0.0526, 0.163, 0.772, 1.30]
    gate_times = [1e-9, 10e-9, 100e-9, 1e-6, 10e-6]
    dark_rates = [1, 10, 100, 1000]
    bg_rate = 100.0
    visibility = 0.5
    for nret in nrets:
        for gate in gate_times:
            for rd in dark_rates:
                m = DetectorModel(nret=nret, dark_rate_hz=rd, background_rate_hz=bg_rate, gate_time_s=gate, visibility=visibility, samples=1)
                res = max_fi_over_phase(m)
                mu_noise = (rd + bg_rate) * gate
                rows.append({
                    "Nret": nret,
                    "gate_time_s": gate,
                    "dark_rate_hz": rd,
                    "background_rate_hz": bg_rate,
                    "dark_probability_per_gate": 1.0 - math.exp(-rd * gate),
                    "background_probability_per_gate": 1.0 - math.exp(-bg_rate * gate),
                    "noise_counts_per_gate": mu_noise,
                    "SBR_gate": nret / mu_noise if mu_noise > 0 else float("inf"),
                    "visibility": visibility,
                    "classical_heterodyne_like_FI_max": res["max_classical_fi"],
                    "classical_crlb_phase_variance": res["crlb_phase_variance"],
                    "note": "classical FI is receiver-level Poisson sinusoidal-counting benchmark, not detector optimality.",
                })
    df = pd.DataFrame(rows)
    df.to_csv(outdir / "detector_gate_classical_baseline.csv", index=False)

    figdir = ensure_figdir(outdir)
    plt.figure(figsize=(7.4, 4.9))
    for nret in nrets:
        sub = df[(df.Nret == nret) & (df.dark_rate_hz == 100)].sort_values("gate_time_s")
        plt.plot(sub.gate_time_s, sub.SBR_gate, marker="o", label=f"Nret={nret:g}")
    plt.xscale("log")
    plt.yscale("log")
    plt.axhline(3.0, color="k", linewidth=1.0)
    plt.xlabel("Detector gate time (s)")
    plt.ylabel("Gate-based signal-to-background ratio")
    plt.title("Detector admissibility as per-gate probability")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(figdir / "fig_gate_based_detector_sbr.png", dpi=220)
    plt.close()

    plt.figure(figsize=(7.4, 4.9))
    for nret in nrets:
        sub = df[(df.Nret == nret) & (df.dark_rate_hz == 100)].sort_values("gate_time_s")
        plt.plot(sub.gate_time_s, sub.classical_heterodyne_like_FI_max, marker="o", label=f"Nret={nret:g}")
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Detector gate time (s)")
    plt.ylabel("Max classical FI per sample")
    plt.title("Classical receiver baseline under the same photon budget")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(figdir / "fig_classical_receiver_fi_baseline.png", dpi=220)
    plt.close()
    return df


def gamma_structure_bridge(outdir: Path) -> pd.DataFrame:
    """Relate Gamma to a phase-structure-function/refractivity variance coordinate."""
    rows = []
    lambda_nm = [532, 1064, 1550]
    L_values = [0.001, 0.01, 0.05]
    Lc_values = [1e-4, 1e-3, 1e-2]
    sigma_n_values = [1e-7, 1e-6, 1e-5, 1e-4]
    Cc = 1.0
    for lam_nm in lambda_nm:
        k0 = 2 * math.pi / (lam_nm * 1e-9)
        for L in L_values:
            for Lc in Lc_values:
                for sig in sigma_n_values:
                    phase_variance = Cc * k0 * k0 * L * Lc * sig * sig
                    rows.append({
                        "wavelength_nm": lam_nm,
                        "path_length_m": L,
                        "correlation_length_m": Lc,
                        "sigma_n_rms": sig,
                        "phase_structure_function_proxy": 2.0 * phase_variance,
                        "Gamma_proxy": phase_variance,
                        "note": "Gamma_proxy follows from coarse-grained phase variance k0^2*L*Lc*sigma_n^2; it is a random-medium bridge, not a non-Gaussian QFI formula.",
                    })
    df = pd.DataFrame(rows)
    df.to_csv(outdir / "gamma_phase_structure_bridge.csv", index=False)

    figdir = ensure_figdir(outdir)
    plt.figure(figsize=(7.4, 4.9))
    for Lc in Lc_values:
        sub = df[(df.wavelength_nm == 1064) & (df.path_length_m == 0.01) & (df.correlation_length_m == Lc)].sort_values("sigma_n_rms")
        plt.plot(sub.sigma_n_rms, sub.Gamma_proxy, marker="o", label=f"Lc={Lc:g} m")
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("RMS refractive-index fluctuation sigma_n")
    plt.ylabel("Gamma proxy")
    plt.title("Random-medium phase-variance bridge for Gamma")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(figdir / "fig_gamma_structure_function_bridge.png", dpi=220)
    plt.close()
    return df


def eos_sensitivity(outdir: Path) -> pd.DataFrame:
    rows = []
    anchors = {532: 1.30, 1064: 0.163, 1550: 0.0526}
    for lam, n0 in anchors.items():
        for delta in [-0.2, -0.1, -0.05, 0, 0.05, 0.1, 0.2]:
            nret = n0 * (1.0 + delta)
            p0 = math.exp(-nret)
            if p0 < 0.01:
                regime = "photon-rich"
            elif p0 < 0.37:
                regime = "low-return"
            elif p0 < 0.90:
                regime = "photon-starved"
            else:
                regime = "extreme-photon-starved"
            rows.append({
                "wavelength_nm": lam,
                "baseline_Nret": n0,
                "EOS_density_multiplier_delta": delta,
                "Nret_after_EOS_sensitivity": nret,
                "zero_count_probability": p0,
                "photon_regime": regime,
                "note": "PR methane density is a proxy; real natural-gas/H2 mixtures require GERG/AGA8/REFPROP-level EOS for calibrated predictions.",
            })
    df = pd.DataFrame(rows)
    df.to_csv(outdir / "eos_density_sensitivity.csv", index=False)
    return df


def qzzb_cutoff_convergence(outdir: Path) -> pd.DataFrame:
    rows = []
    Ns_values = [0.5, 1.0, 1.5]
    cutoffs = [4, 6, 8, 10]
    eta_s = 0.9
    eta_i = 1.0
    gamma = 0.5
    prior_width = math.pi / 2
    for Ns in Ns_values:
        prev = None
        for cutoff in cutoffs:
            rho = prepare_noisy_tmsv_density(Ns=Ns, cutoff=cutoff, eta_s=eta_s, eta_i=eta_i, gamma=gamma)
            zz = qzzb_phase_bound(rho, cutoff=cutoff, prior_width=prior_width, tau_points=21)
            tail = tmsv_tail_probability(Ns, cutoff)
            rel_change = abs(zz - prev) / max(abs(zz), 1e-300) if prev is not None else float("nan")
            rows.append({
                "NS_target": Ns,
                "cutoff": cutoff,
                "eta_s": eta_s,
                "eta_i": eta_i,
                "Gamma": gamma,
                "prior_width": prior_width,
                "Sigma_ZZ": zz,
                "tail_probability": tail,
                "relative_change_from_previous_cutoff": rel_change,
                "converged_by_10pct_step_rule": bool(rel_change < 0.10) if prev is not None else False,
                "note": "finite-Fock QZZB audit; non-converged high-NS points are stress tests, not evidence of advantage.",
            })
            prev = zz
    df = pd.DataFrame(rows)
    df.to_csv(outdir / "qzzb_cutoff_convergence_audit_v8_2.csv", index=False)

    figdir = ensure_figdir(outdir)
    plt.figure(figsize=(7.4, 4.9))
    for Ns in Ns_values:
        sub = df[df.NS_target == Ns]
        plt.plot(sub.cutoff, sub.Sigma_ZZ, marker="o", label=f"Ns={Ns:g}")
    plt.xlabel("Fock cutoff")
    plt.ylabel("Diagnostic QZZB phase bound")
    plt.title("Finite-Fock QZZB cutoff convergence audit")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(figdir / "fig_qzzb_cutoff_convergence_v8_2.png", dpi=220)
    plt.close()
    return df


def write_summary(outdir: Path, tables: dict[str, pd.DataFrame]) -> None:
    summary = {
        "version": "v8.2 engineering audit",
        "outputs": {name: int(len(df)) for name, df in tables.items()},
        "major_changes": [
            "explicit multimode-to-single-mode mode-coupling penalty chi_mode",
            "idler transmission and phase-stability requirements as functions of delay length/time",
            "gate-based dark/background probabilities plus classical receiver FI baseline",
            "Gamma linked to a random-medium phase-structure-function proxy",
            "PR/EOS density handled as multiplicative sensitivity, not calibrated gas-mixture prediction",
            "finite-Fock QZZB cutoff convergence made visible",
        ],
        "claim_scope": "engineering-audit outputs; no end-to-end quantum advantage is certified",
    }
    (outdir / "SUMMARY_v8_2.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (outdir / "SUMMARY_v8_2.md").write_text("# v8.2 engineering audit summary\n\n" + json.dumps(summary, indent=2), encoding="utf-8")


def run_all(outdir: Path) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    tables = {
        "mode_bridge": mode_bridge_scan(outdir),
        "idler_delay": idler_delay_scan(outdir),
        "detector_gate_classical": detector_gate_classical_baseline(outdir),
        "gamma_structure_bridge": gamma_structure_bridge(outdir),
        "eos_sensitivity": eos_sensitivity(outdir),
        "qzzb_cutoff_convergence": qzzb_cutoff_convergence(outdir),
    }
    write_summary(outdir, tables)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", default="outputs_v8_2_engineering_audit")
    args = parser.parse_args()
    run_all(Path(args.outdir))
    print(f"v8.2 engineering audit outputs written to {Path(args.outdir).resolve()}")


if __name__ == "__main__":
    main()
