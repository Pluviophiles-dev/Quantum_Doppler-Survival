#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Run enhanced analysis package and generate outputs.

Usage:
    python scripts/run_enhanced_analysis.py --outdir outputs --quick
    python scripts/run_enhanced_analysis.py --outdir outputs
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import json
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from qdboundary_enhanced.gaussian_fidelity import GaussianPoint, fidelity_curve, summarize_point, scan_eta_gamma
from qdboundary_enhanced.transduction import GasState, OpticalBudget, scan_pressure_wavelength
from qdboundary_enhanced.detector_fi import DetectorModel, fi_curve, scan_detector
from qdboundary_enhanced.multilayer import Scenario, classify_scenario
from qdboundary_enhanced.plotting import heatmap_from_grid
from qdboundary_enhanced.audit import write_integrity_audit
from qdboundary_enhanced.stressors import (
    InstrumentStress,
    MultipleScatteringStress,
    RefractiveIndexStress,
    gamma_from_refractive_stress,
    instrument_readiness_verdict,
    mode_purity_from_stress,
    multiple_scattering_verdict,
)


def save_json(obj, path: Path):
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", type=str, default="outputs_enhanced")
    ap.add_argument("--quick", action="store_true", help="Faster low-resolution scan.")
    args = ap.parse_args()

    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)
    write_integrity_audit(out)

    # 1. Gaussian fidelity single point + high-NS grid
    tau_points = 31 if args.quick else 401
    eta_num = 7 if args.quick else 41
    gamma_num = 5 if args.quick else 31

    pt = GaussianPoint(ns=30.0, eta_s=0.9, eta_i=0.95, gamma=0.3, tau_points=tau_points)
    summary = summarize_point(pt)
    save_json(summary, out / "gaussian_single_summary.json")
    pd.DataFrame([summary]).to_csv(out / "gaussian_single_summary.csv", index=False)

    curve = fidelity_curve(pt)
    pd.DataFrame(curve).to_csv(out / "gaussian_fidelity_curve.csv", index=False)

    fig = plt.figure(figsize=(7, 4.5))
    plt.plot(curve["tau_rad"], curve["fidelity_amplitude"], label="fidelity amplitude")
    plt.plot(curve["tau_rad"], curve["fidelity_squared"], label="squared fidelity")
    plt.xlabel("phase separation tau (rad)")
    plt.ylabel("fidelity")
    plt.title("High-NS Gaussian covariance fidelity diagnostic")
    plt.legend()
    plt.tight_layout()
    fig.savefig(out / "fig_gaussian_fidelity_curve.png", dpi=240)
    plt.close(fig)

    rows = scan_eta_gamma(
        ns=30.0,
        eta_i=0.95,
        eta_s_grid=np.linspace(0.2, 1.0, eta_num),
        gamma_grid=np.linspace(0.0, 3.0, gamma_num),
        tau_points=tau_points,
    )
    df_g = pd.DataFrame(rows)
    df_g.to_csv(out / "gaussian_eta_gamma_scan.csv", index=False)
    heatmap_from_grid(df_g, "eta_s", "gamma", "surrogate_gaussian_qfi_ratio_vs_coherent",
                      out / "fig_gaussian_qfi_ratio_map.png",
                      "Gaussian Bures-QFI ratio vs coherent benchmark")
    heatmap_from_grid(df_g, "eta_s", "gamma", "diagnostic_gaussian_qzzb_squared_fidelity",
                      out / "fig_gaussian_qzzb_map.png",
                      "Gaussian-fidelity diagnostic QZZB map")

    # 2. Rayleigh transduction bridge
    gas = GasState(pressure_mpa=30.0, compressibility_z=0.86)
    # Match the constrained photon-budget scale used in the manuscript: the
    # product window * optical train * detector efficiency is approximately 0.05.
    opt = OpticalBudget(wavelength_nm=1064.0, pulse_energy_j=1e-6,
                        window_transmission=0.98, optical_train_transmission=0.10,
                        detector_efficiency=0.50)
    rows_t = scan_pressure_wavelength(
        pressures_mpa=np.linspace(1, 35, 9 if args.quick else 69),
        wavelengths_nm=np.array([532, 633, 1064, 1550], dtype=float),
        gas_template=gas,
        opt_template=opt,
    )
    df_t = pd.DataFrame(rows_t)
    df_t.to_csv(out / "rayleigh_transduction_scan.csv", index=False)

    stress_rows = []
    for _, row in df_t.iterrows():
        refr = RefractiveIndexStress(
            wavelength_nm=float(row["wavelength_nm"]),
            path_length_m=opt.probe_length_m,
            correlation_length_m=1.0e-3,
            sigma_n=1.0e-7,
        )
        ms = MultipleScatteringStress(
            number_density_m3=float(row["number_density_m3"]),
            cross_section_m2=float(row["sigma_rayleigh_m2"]),
            path_length_m=opt.probe_length_m,
        )
        ms_row = multiple_scattering_verdict(ms)
        inst = InstrumentStress(
            calibration_relative_uncertainty=0.05,
            timing_jitter_to_gate_ratio=opt.detector_jitter_rms_s / opt.receiver_gate_rms_s,
            background_to_signal_ratio=1.0 / max(float(row["Nret"]), 1e-300),
        )
        inst_row = instrument_readiness_verdict(inst)
        stress_rows.append({
            "pressure_mpa": float(row["pressure_mpa"]),
            "wavelength_nm": float(row["wavelength_nm"]),
            "gamma_refractive_proxy": gamma_from_refractive_stress(refr),
            "optical_depth": ms_row["optical_depth"],
            "multiple_scattering_screen": ms_row["multiple_scattering_screen"],
            "stress_mode_purity": mode_purity_from_stress(opt.turbulence_mode_purity, float(ms_row["optical_depth"])),
            "instrument_screen": inst_row["instrument_screen"],
            "instrument_failed_terms": inst_row["failed_terms"],
            "scope_note": "stress screens only; not full turbulence, multiple-scattering, or instrument validation",
        })
    pd.DataFrame(stress_rows).to_csv(out / "propagation_instrument_stress_screens.csv", index=False)

    fig = plt.figure(figsize=(7, 4.8))
    for lam in [532, 633, 1064, 1550]:
        sub = df_t[df_t["wavelength_nm"] == lam]
        plt.plot(sub["pressure_mpa"], sub["Nret"], label=f"{lam} nm")
    plt.yscale("log")
    plt.xlabel("pressure (MPa)")
    plt.ylabel("returned Rayleigh photons per pulse")
    plt.title("Parameterized Rayleigh photon-budget bridge")
    plt.legend()
    plt.tight_layout()
    fig.savefig(out / "fig_rayleigh_photon_budget_bridge.png", dpi=240)
    plt.close(fig)

    # 3. Detector likelihood/FI
    det = DetectorModel(nret=0.1625, dark_rate_hz=25, background_rate_hz=100, gate_time_s=1e-6, visibility=0.5)
    c = fi_curve(det)
    pd.DataFrame(c).to_csv(out / "detector_fi_curve.csv", index=False)

    fig = plt.figure(figsize=(7, 4.5))
    plt.plot(c["phi_rad"], c["classical_fi"])
    plt.xlabel("phase phi (rad)")
    plt.ylabel("classical Fisher information")
    plt.title("Poisson detector likelihood FI benchmark")
    plt.tight_layout()
    fig.savefig(out / "fig_detector_fi_curve.png", dpi=240)
    plt.close(fig)

    df_d = pd.DataFrame(scan_detector(
        nret_values=np.logspace(-3, 1, 10 if args.quick else 60),
        dark_rates=[1, 10, 25, 100, 1000],
        background_rates=[0, 10, 100, 1000],
        gate_time_s=1e-6,
        visibility=0.5,
    ))
    df_d.to_csv(out / "detector_fi_scan.csv", index=False)

    # 4. Integrated scenarios
    scenarios = [
        Scenario(gas=GasState(pressure_mpa=30.0, compressibility_z=0.86),
                 optics=OpticalBudget(wavelength_nm=532, pulse_energy_j=1e-6, window_transmission=0.98, optical_train_transmission=0.10, detector_efficiency=0.50),
                 ns=30, eta_i=0.95, gamma=0.3),
        Scenario(gas=GasState(pressure_mpa=30.0, compressibility_z=0.86),
                 optics=OpticalBudget(wavelength_nm=1064, pulse_energy_j=1e-6, window_transmission=0.98, optical_train_transmission=0.10, detector_efficiency=0.50),
                 ns=30, eta_i=0.95, gamma=0.3),
        Scenario(gas=GasState(pressure_mpa=30.0, compressibility_z=0.86),
                 optics=OpticalBudget(wavelength_nm=1550, pulse_energy_j=1e-6, window_transmission=0.98, optical_train_transmission=0.10, detector_efficiency=0.50),
                 ns=30, eta_i=0.95, gamma=0.3, background_rate_hz=10000),
        Scenario(gas=GasState(pressure_mpa=30.0, compressibility_z=0.86),
                 optics=OpticalBudget(wavelength_nm=532, pulse_energy_j=1e-6, mode_overlap=0.03, window_transmission=0.98, optical_train_transmission=0.10, detector_efficiency=0.50),
                 ns=30, eta_i=0.95, gamma=0.3),
        Scenario(gas=GasState(pressure_mpa=30.0, compressibility_z=0.86),
                 optics=OpticalBudget(wavelength_nm=532, pulse_energy_j=1e-6, window_transmission=0.98, optical_train_transmission=0.10, detector_efficiency=0.50),
                 ns=30, eta_i=0.40, gamma=0.3),
        Scenario(gas=GasState(pressure_mpa=30.0, compressibility_z=0.86),
                 optics=OpticalBudget(wavelength_nm=532, pulse_energy_j=1e-6, window_transmission=0.98, optical_train_transmission=0.10, detector_efficiency=0.50),
                 ns=30, eta_i=0.95, gamma=2.0),
    ]
    df_s = pd.DataFrame([classify_scenario(s) for s in scenarios])
    df_s.to_csv(out / "integrated_scenario_verdicts.csv", index=False)

    print(f"Enhanced analysis complete. Outputs written to {out.resolve()}")


if __name__ == "__main__":
    main()
