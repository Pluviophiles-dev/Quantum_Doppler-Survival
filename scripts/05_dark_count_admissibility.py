#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
05_dark_count_admissibility.py

Detector-level dark-count / background-count admissibility scan for the
photon-starved quantum Doppler manuscript.

Key idea
--------
The quantum-channel loss parameter eta_s should not be conflated with detector
dark counts. This script adds a detector-layer constraint on top of the Rayleigh
return-photon budget:

    N_noise = (R_dark + R_bg) * tau_gate
    SBR     = N_ret / N_noise
    admissible if SBR >= SBR_min and N_ret >= N_noise

This is not a replacement for the TMSV QFI model. It is a photon-budget
admissibility check for the "last mile" of experimental realism.

Usage
-----
    PYTHONPATH=. python scripts/05_dark_count_admissibility.py --config configs/default.json

Outputs
-------
    figures/fig_dark_count_admissibility.png
    figures/fig_dark_count_margin.png
    data/dark_count_admissibility.csv

Dependencies
------------
    numpy, pandas, matplotlib
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    from qdboundary.config import load_config as _load_config
    from qdboundary.rayleigh import return_photons as _return_photons
except Exception:
    _load_config = None
    _return_photons = None

AVOGADRO = 6.02214076e23
R = 8.31446261815324
C = 299792458.0
H = 6.62607015e-34
N0 = 2.68678e25


def load_config(path: str | Path) -> dict:
    path = Path(path)
    if _load_config is not None and path.exists():
        cfg = _load_config(path)
    elif path.exists():
        with path.open("r", encoding="utf-8") as f:
            cfg = json.load(f)
        for p in cfg.get("paths", {}).values():
            Path(p).mkdir(parents=True, exist_ok=True)
    else:
        cfg = {}

    cfg.setdefault("paths", {"figures": "figures", "data": "data"})
    cfg.setdefault("rayleigh", {})
    cfg.setdefault("detector", {})

    r = cfg["rayleigh"]
    r.setdefault("pressure_MPa_values", [10, 30, 35])
    r.setdefault("wavelength_nm_values", [532, 633, 1064, 1550])
    r.setdefault("temperature_K", 298.15)
    r.setdefault("pulse_energy_J", 1e-6)
    r.setdefault("probe_length_m", 0.01)
    r.setdefault("collection_fraction", 1e-7)
    r.setdefault("eta_sys", 0.05)
    r.setdefault("n_ref", 1.000444)
    r.setdefault("king_factor", 1.04)

    d = cfg["detector"]
    d.setdefault("dark_count_rate_Hz_values", [0.1, 1.0, 10.0, 100.0])
    d.setdefault("background_rate_Hz", 1.0)
    d.setdefault("gate_time_s", 1e-6)
    d.setdefault("sbr_min", 3.0)
    d.setdefault("min_signal_to_noise_photons", 1.0)

    Path(cfg["paths"]["figures"]).mkdir(parents=True, exist_ok=True)
    Path(cfg["paths"]["data"]).mkdir(parents=True, exist_ok=True)
    return cfg


def methane_number_density_peng_robinson(P_Pa: float, T_K: float) -> float:
    Tc = 190.564
    Pc = 4.5992e6
    omega = 0.01142
    kappa = 0.37464 + 1.54226 * omega - 0.26992 * omega**2
    alpha = (1.0 + kappa * (1.0 - np.sqrt(T_K / Tc))) ** 2
    a = 0.45724 * R**2 * Tc**2 / Pc * alpha
    b = 0.07780 * R * Tc / Pc
    A = a * P_Pa / (R**2 * T_K**2)
    B = b * P_Pa / (R * T_K)
    coeff = [1.0, -(1.0 - B), A - 3.0 * B**2 - 2.0 * B, -(A * B - B**2 - B**3)]
    roots = np.roots(coeff)
    real_roots = sorted([r.real for r in roots if abs(r.imag) < 1e-8 and r.real > 0])
    Z = max(real_roots) if real_roots else 1.0
    mol_density = P_Pa / (Z * R * T_K)
    return float(mol_density * AVOGADRO)


def rayleigh_cross_section(lambda0_m: float, n_ref: float = 1.000444, king_factor: float = 1.04, n0: float = N0) -> float:
    ratio = ((n_ref**2 - 1.0) / (n_ref**2 + 2.0)) ** 2
    return float((24.0 * np.pi**3 / (n0**2 * lambda0_m**4)) * ratio * king_factor)


def return_photons_fallback(
    pressure_MPa: float,
    temperature_K: float,
    lambda_nm: float,
    pulse_energy_J: float,
    probe_length_m: float,
    collection_fraction: float,
    eta_sys: float,
    n_ref: float = 1.000444,
    king_factor: float = 1.04,
) -> float:
    lambda0_m = lambda_nm * 1e-9
    n = methane_number_density_peng_robinson(pressure_MPa * 1e6, temperature_K)
    sigma = rayleigh_cross_section(lambda0_m, n_ref=n_ref, king_factor=king_factor)
    nin = pulse_energy_J / (H * C / lambda0_m)
    return float(nin * n * probe_length_m * sigma * collection_fraction * eta_sys)


def return_photons(**kwargs) -> float:
    if _return_photons is not None:
        return float(_return_photons(**kwargs))
    return return_photons_fallback(**kwargs)


def photon_regime_from_Nret(Nret: float) -> str:
    P0 = float(np.exp(-max(Nret, 0.0)))
    if P0 < 0.01:
        return "photon-rich"
    if P0 < 0.37:
        return "low-return"
    if P0 < 0.90:
        return "photon-starved"
    return "extreme photon-starved"


def compute_rows(cfg: dict) -> pd.DataFrame:
    r = cfg["rayleigh"]
    d = cfg["detector"]
    pressures = r["pressure_MPa_values"]
    wavelengths = r["wavelength_nm_values"]
    dark_rates = d["dark_count_rate_Hz_values"]
    bg_rate = float(d["background_rate_Hz"])
    gate = float(d["gate_time_s"])
    sbr_min = float(d["sbr_min"])
    min_signal_to_noise = float(d.get("min_signal_to_noise_photons", 1.0))

    rows = []
    for P in pressures:
        for lam in wavelengths:
            Nret = return_photons(
                pressure_MPa=float(P),
                temperature_K=float(r["temperature_K"]),
                lambda_nm=float(lam),
                pulse_energy_J=float(r["pulse_energy_J"]),
                probe_length_m=float(r["probe_length_m"]),
                collection_fraction=float(r["collection_fraction"]),
                eta_sys=float(r["eta_sys"]),
                n_ref=float(r.get("n_ref", 1.000444)),
                king_factor=float(r.get("king_factor", 1.04)),
            )
            for rdark in dark_rates:
                Ndark = float(rdark) * gate
                Nbg = bg_rate * gate
                Nnoise = Ndark + Nbg
                sbr = Nret / max(Nnoise, np.finfo(float).tiny)
                margin = np.log10(max(sbr, np.finfo(float).tiny) / sbr_min)
                admissible = (sbr >= sbr_min) and (Nret >= min_signal_to_noise * Nnoise)
                rows.append({
                    "pressure_MPa": float(P),
                    "lambda_nm": float(lam),
                    "Nret_rayleigh_budget": Nret,
                    "P0_zero_count_signal_only": np.exp(-Nret),
                    "photon_regime_signal_only": photon_regime_from_Nret(Nret),
                    "dark_count_rate_Hz": float(rdark),
                    "background_rate_Hz": bg_rate,
                    "gate_time_s": gate,
                    "N_dark_per_gate": Ndark,
                    "N_background_per_gate": Nbg,
                    "N_noise_per_gate": Nnoise,
                    "SBR_Nret_over_noise": sbr,
                    "log10_SBR": np.log10(max(sbr, np.finfo(float).tiny)),
                    "SBR_min": sbr_min,
                    "log10_margin_over_SBR_min": margin,
                    "detector_admissible": bool(admissible),
                })
    return pd.DataFrame(rows)


def plot_admissibility(df: pd.DataFrame, cfg: dict) -> None:
    figdir = Path(cfg["paths"]["figures"])
    datadir = Path(cfg["paths"]["data"])
    datadir.mkdir(parents=True, exist_ok=True)
    figdir.mkdir(parents=True, exist_ok=True)
    df.to_csv(datadir / "dark_count_admissibility.csv", index=False)

    dark_rates = sorted(df["dark_count_rate_Hz"].unique())
    wavelengths = sorted(df["lambda_nm"].unique())
    pressures = sorted(df["pressure_MPa"].unique())

    ncols = min(2, len(dark_rates))
    nrows = int(np.ceil(len(dark_rates) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(6.8 * ncols, 4.9 * nrows), squeeze=False)

    for ax, rdark in zip(axes.ravel(), dark_rates):
        sub = df[df["dark_count_rate_Hz"] == rdark]
        pivot = sub.pivot(index="lambda_nm", columns="pressure_MPa", values="log10_SBR")
        Z = pivot.loc[wavelengths, pressures].to_numpy()
        im = ax.imshow(Z, origin="lower", aspect="auto", vmin=0, vmax=max(6, np.nanmax(Z)))
        ax.set_xticks(range(len(pressures)))
        ax.set_xticklabels([f"{p:g}" for p in pressures])
        ax.set_yticks(range(len(wavelengths)))
        ax.set_yticklabels([f"{w:g}" for w in wavelengths])
        ax.set_xlabel("Pressure P (MPa)")
        ax.set_ylabel(r"Wavelength $\lambda_0$ (nm)")
        ax.set_title(fr"$R_{{dark}}={rdark:g}$ Hz, gate={cfg['detector']['gate_time_s']:.1e} s")
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label(r"$\log_{10}(N_{ret}/N_{noise})$")

        adm = sub.pivot(index="lambda_nm", columns="pressure_MPa", values="detector_admissible").loc[wavelengths, pressures].to_numpy()
        for i in range(len(wavelengths)):
            for j in range(len(pressures)):
                if not bool(adm[i, j]):
                    ax.text(j, i, "×", ha="center", va="center", fontsize=18, fontweight="bold", color="white")

    for ax in axes.ravel()[len(dark_rates):]:
        ax.axis("off")

    fig.suptitle("Detector-level dark-count admissibility on top of the Rayleigh photon budget", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(figdir / "fig_dark_count_admissibility.png", dpi=300, bbox_inches="tight")
    fig.savefig(figdir / "fig_dark_count_admissibility.pdf", bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.3, 5.4))
    for rdark in dark_rates:
        for lam in wavelengths:
            sub = df[(df["dark_count_rate_Hz"] == rdark) & (df["lambda_nm"] == lam)].sort_values("pressure_MPa")
            label = fr"$\lambda={lam:g}$ nm, $R_d={rdark:g}$ Hz" if rdark in [dark_rates[0], dark_rates[-1]] else None
            alpha = 0.95 if rdark in [dark_rates[0], dark_rates[-1]] else 0.35
            ax.plot(sub["pressure_MPa"], sub["log10_margin_over_SBR_min"], marker="o", lw=1.6, alpha=alpha, label=label)
    ax.axhline(0, color="black", ls="--", lw=1.4, label="admissibility boundary")
    ax.set_xlabel("Pressure P (MPa)")
    ax.set_ylabel(r"$\log_{10}[(N_{ret}/N_{noise})/\mathrm{SBR}_{min}]$")
    ax.set_title("Detector admissibility margin: positive values pass the SBR criterion", fontweight="bold")
    ax.grid(True, ls="--", alpha=0.35)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(figdir / "fig_dark_count_margin.png", dpi=300, bbox_inches="tight")
    fig.savefig(figdir / "fig_dark_count_margin.pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.json")
    args = parser.parse_args()
    cfg = load_config(args.config)
    df = compute_rows(cfg)
    plot_admissibility(df, cfg)
    n_pass = int(df["detector_admissible"].sum())
    print(f"Saved dark-count admissibility outputs. Passing points: {n_pass}/{len(df)}")


if __name__ == "__main__":
    main()
