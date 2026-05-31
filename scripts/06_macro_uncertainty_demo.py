#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
06_macro_uncertainty_demo.py

Illustrative 1D macro-scale uncertainty-transfer demo.

Purpose
-------
This script does NOT claim a completed industrial EnKF or pipeline-flow
reconstruction. It only demonstrates a minimal uncertainty-transfer interface:

    microscopic phase RMSE  -> velocity-observation noise
                            -> 1D velocity-profile uncertainty band

The quantum-enhanced case uses the manuscript's local effective relation

    sigma_v,TMSV = sigma_v,CS / sqrt(G_eff)

when G_eff > 0, without claiming global optimality outside the QZZB-valid region.

Usage
-----
    PYTHONPATH=. python scripts/06_macro_uncertainty_demo.py --config configs/default.json

Outputs
-------
    figures/fig_macro_uncertainty_transfer.png
    figures/fig_macro_uncertainty_transfer.pdf
    data/macro_uncertainty_transfer.csv
    data/macro_uncertainty_summary.csv

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
    from qdboundary.formulas import geff as _geff
    from qdboundary.formulas import doppler_k as _doppler_k
except Exception:
    _load_config = None
    _geff = None
    _doppler_k = None


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
    cfg.setdefault("model", {})
    cfg["model"].setdefault("Ns_main", 100.0)
    cfg["model"].setdefault("M", 1)

    cfg.setdefault("rayleigh", {})
    cfg["rayleigh"].setdefault("eta_s_assumed", 0.90)
    cfg["rayleigh"].setdefault("gamma_assumed", 0.50)
    cfg["rayleigh"].setdefault("n_ref", 1.000444)

    cfg.setdefault("macro_demo", {})
    m = cfg["macro_demo"]
    m.setdefault("pipe_length_m", 100.0)
    m.setdefault("v0_m_per_s", 15.0)
    m.setdefault("delta_v_m_per_s", 1.0)
    m.setdefault("num_grid", 240)
    m.setdefault("sensor_positions_m", [10, 30, 50, 70, 90])
    m.setdefault("correlation_length_m", 25.0)
    m.setdefault("lambda0_m", 532e-9)
    m.setdefault("tau_int_s", 1e-6)
    m.setdefault("random_seed", 7)
    m.setdefault("use_single_noise_realization", True)

    Path(cfg["paths"]["figures"]).mkdir(parents=True, exist_ok=True)
    Path(cfg["paths"]["data"]).mkdir(parents=True, exist_ok=True)
    return cfg


def geff(eta_s: float, Ns: float, Gamma: float) -> float:
    if _geff is not None:
        return float(_geff(eta_s, Ns, Gamma, a=1.0))
    return float(((Ns + 1.0) / (1.0 + 2.0 * (1.0 - eta_s) * Ns)) * np.exp(-Gamma))


def doppler_k(lambda0_m: float, ng: float = 1.000444) -> float:
    if _doppler_k is not None:
        try:
            return float(_doppler_k(lambda0_m, ng))
        except TypeError:
            return float(_doppler_k(lambda0_m=lambda0_m, ng=ng))
    return float(4.0 * np.pi * ng / lambda0_m)


def classical_velocity_sigma(lambda0_m: float, tau_int: float, ng: float, eta_s: float, Ns: float, M: int) -> float:
    sigma_phi = np.sqrt(1.0 / (4.0 * M * eta_s * Ns))
    return float(sigma_phi / (tau_int * doppler_k(lambda0_m, ng)))


def true_velocity_profile(x: np.ndarray, L: float, v0: float, dv: float) -> np.ndarray:
    return v0 + dv * np.sin(np.pi * x / L) + 0.25 * dv * np.sin(2.0 * np.pi * x / L + 0.4)


def rbf_kernel(xa: np.ndarray, xb: np.ndarray, length_scale: float, amplitude: float) -> np.ndarray:
    xa = np.asarray(xa, dtype=float)[:, None]
    xb = np.asarray(xb, dtype=float)[None, :]
    return amplitude**2 * np.exp(-0.5 * ((xa - xb) / length_scale) ** 2)


def gp_condition(x_grid: np.ndarray, x_obs: np.ndarray, y_obs: np.ndarray, noise_sigma: float, length_scale: float, amplitude: float, mean_value: float):
    Koo = rbf_kernel(x_obs, x_obs, length_scale, amplitude) + (noise_sigma**2 + 1e-12) * np.eye(len(x_obs))
    Kgo = rbf_kernel(x_grid, x_obs, length_scale, amplitude)
    Kgg_diag = np.diag(rbf_kernel(x_grid, x_grid, length_scale, amplitude))
    y_centered = y_obs - mean_value
    alpha = np.linalg.solve(Koo, y_centered)
    mean = mean_value + Kgo @ alpha
    v = np.linalg.solve(Koo, Kgo.T)
    var = np.maximum(Kgg_diag - np.sum(Kgo * v.T, axis=1), 0.0)
    return mean, var


def run_demo(cfg: dict) -> None:
    figdir = Path(cfg["paths"]["figures"])
    datadir = Path(cfg["paths"]["data"])

    m = cfg["macro_demo"]
    L = float(m["pipe_length_m"])
    v0 = float(m["v0_m_per_s"])
    dv = float(m["delta_v_m_per_s"])
    n_grid = int(m["num_grid"])
    x_grid = np.linspace(0.0, L, n_grid)
    x_obs = np.asarray(m["sensor_positions_m"], dtype=float)
    length_scale = float(m["correlation_length_m"])
    lambda0_m = float(m["lambda0_m"])
    tau_int = float(m["tau_int_s"])
    rng = np.random.default_rng(int(m["random_seed"]))

    Ns = float(cfg["model"]["Ns_main"])
    M = int(cfg["model"]["M"])
    eta_s = float(cfg["rayleigh"]["eta_s_assumed"])
    Gamma = float(cfg["rayleigh"]["gamma_assumed"])
    ng = float(cfg["rayleigh"].get("n_ref", 1.000444))

    g_eff = max(geff(eta_s, Ns, Gamma), np.finfo(float).tiny)
    sigma_cs = classical_velocity_sigma(lambda0_m, tau_int, ng, eta_s, Ns, M)
    sigma_tmsv = sigma_cs / np.sqrt(g_eff)

    v_true_grid = true_velocity_profile(x_grid, L, v0, dv)
    v_true_obs = true_velocity_profile(x_obs, L, v0, dv)

    z = rng.normal(size=len(x_obs))
    if bool(m["use_single_noise_realization"]):
        y_cs = v_true_obs + sigma_cs * z
        y_tmsv = v_true_obs + sigma_tmsv * z
    else:
        y_cs = v_true_obs + rng.normal(scale=sigma_cs, size=len(x_obs))
        y_tmsv = v_true_obs + rng.normal(scale=sigma_tmsv, size=len(x_obs))

    amplitude = max(0.5 * dv, 0.2)
    mean_cs, var_cs = gp_condition(x_grid, x_obs, y_cs, sigma_cs, length_scale, amplitude, mean_value=v0)
    mean_t, var_t = gp_condition(x_grid, x_obs, y_tmsv, sigma_tmsv, length_scale, amplitude, mean_value=v0)

    rmse_cs = float(np.sqrt(np.mean((mean_cs - v_true_grid) ** 2)))
    rmse_t = float(np.sqrt(np.mean((mean_t - v_true_grid) ** 2)))
    avg_unc_cs = float(np.mean(np.sqrt(var_cs)))
    avg_unc_t = float(np.mean(np.sqrt(var_t)))

    out = pd.DataFrame({
        "x_m": x_grid,
        "v_true_m_per_s": v_true_grid,
        "v_recon_classical_m_per_s": mean_cs,
        "u_classical_m_per_s": np.sqrt(var_cs),
        "v_recon_tmsv_m_per_s": mean_t,
        "u_tmsv_m_per_s": np.sqrt(var_t),
    })
    out.to_csv(datadir / "macro_uncertainty_transfer.csv", index=False)

    summary = pd.DataFrame([{
        "Ns": Ns,
        "M": M,
        "eta_s": eta_s,
        "Gamma": Gamma,
        "G_eff": g_eff,
        "sigma_v_classical_m_per_s": sigma_cs,
        "sigma_v_tmsv_m_per_s": sigma_tmsv,
        "rmse_reconstruction_classical_m_per_s": rmse_cs,
        "rmse_reconstruction_tmsv_m_per_s": rmse_t,
        "mean_posterior_uncertainty_classical_m_per_s": avg_unc_cs,
        "mean_posterior_uncertainty_tmsv_m_per_s": avg_unc_t,
        "interpretation": "illustrative 1D uncertainty-transfer demo; not a completed industrial EnKF"
    }])
    summary.to_csv(datadir / "macro_uncertainty_summary.csv", index=False)

    fig, ax = plt.subplots(figsize=(10.4, 5.9))
    ax.plot(x_grid, v_true_grid, "k-", lw=2.2, label="true 1D velocity profile")

    ax.plot(x_grid, mean_cs, "--", lw=2.0, label="classical reconstruction")
    ax.fill_between(x_grid, mean_cs - 1.96 * np.sqrt(var_cs), mean_cs + 1.96 * np.sqrt(var_cs), alpha=0.18, label="classical 95% band")

    ax.plot(x_grid, mean_t, "-", lw=2.2, label="TMSV-effective reconstruction")
    ax.fill_between(x_grid, mean_t - 1.96 * np.sqrt(var_t), mean_t + 1.96 * np.sqrt(var_t), alpha=0.20, label="TMSV 95% band")

    ax.scatter(x_obs, y_cs, marker="x", s=55, label="classical noisy observations")
    ax.scatter(x_obs, y_tmsv, marker="o", s=38, facecolors="none", label="TMSV noisy observations")

    ax.set_xlabel("Position along pipe x (m)")
    ax.set_ylabel("Velocity v(x) (m/s)")
    ax.set_title("Illustrative 1D macro uncertainty transfer from phase RMSE to velocity-field band", fontweight="bold")
    ax.grid(True, ls="--", alpha=0.35)

    text = (fr"$G_{{eff}}={g_eff:.2f}$, "
            fr"$\sigma_v^{{CS}}={sigma_cs:.3g}$ m/s, "
            fr"$\sigma_v^{{TMSV}}={sigma_tmsv:.3g}$ m/s" "\n"
            "This is an illustrative uncertainty-transfer interface, not an industrial EnKF validation.")
    ax.text(0.02, 0.02, text, transform=ax.transAxes, fontsize=9, va="bottom",
            bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="0.5", alpha=0.86))

    ax.legend(fontsize=8.5, ncol=2, loc="upper right")
    fig.tight_layout()
    fig.savefig(figdir / "fig_macro_uncertainty_transfer.png", dpi=300, bbox_inches="tight")
    fig.savefig(figdir / "fig_macro_uncertainty_transfer.pdf", bbox_inches="tight")
    plt.close(fig)

    print("Saved macro uncertainty-transfer outputs.")
    print(summary.to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.json")
    args = parser.parse_args()
    cfg = load_config(args.config)
    run_demo(cfg)


if __name__ == "__main__":
    main()
