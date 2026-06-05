from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from qdboundary.classification import CLASS_TO_INT, INT_TO_CLASS, classify_boundary_point
from qdboundary.config import load_config
from qdboundary.fock import number_operator_signal, prepare_noisy_tmsv_density, qfi_unitary_generator, tmsv_tail_probability
from qdboundary.formulas import geff, local_phase_variance_from_qfi, wrapping_probability_gaussian
from qdboundary.plotting import savefig
from qdboundary.qzzb import qzzb_phase_bound


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.json")
    parser.add_argument("--fast", action="store_true", help="Use a smaller grid for quick testing.")
    args = parser.parse_args()
    cfg = load_config(args.config)

    Ns = float(cfg["model"]["Ns_diag"])
    cutoff = int(cfg["fock"]["cutoff"])
    M = int(cfg["model"]["M"])
    eta_n = 9 if args.fast else int(cfg["grids"]["eta_points_qzzb"])
    gamma_n = 9 if args.fast else int(cfg["grids"]["gamma_points_qzzb"])
    tau_points = 21 if args.fast else int(cfg["grids"]["tau_points_qzzb"])
    etas = np.linspace(cfg["grids"]["eta_min"], cfg["grids"]["eta_max"], eta_n)
    gammas = np.linspace(cfg["grids"]["gamma_min"], cfg["grids"]["gamma_max"], gamma_n)
    W = float(cfg["model"]["qzzb_prior_width"])
    eta_i = float(cfg["rayleigh"]["eta_i_assumed"])
    Gop = number_operator_signal(cutoff)

    max_points = int(cfg["fock"].get("max_qzzb_grid_points", 441))
    if len(etas) * len(gammas) > max_points:
        raise RuntimeError(f"QZZB grid has {len(etas)*len(gammas)} points; reduce grid or raise max_qzzb_grid_points.")

    rows = []
    cls_grid = np.zeros((len(gammas), len(etas)), dtype=int)
    guard_grid = np.zeros_like(cls_grid, dtype=float)
    wrap_grid = np.zeros_like(cls_grid, dtype=float)
    geff_grid = np.zeros_like(cls_grid, dtype=float)

    for gi, gamma in enumerate(gammas):
        for ei, eta_s in enumerate(etas):
            rho = prepare_noisy_tmsv_density(Ns, cutoff, eta_s, eta_i, gamma)
            Fq = qfi_unitary_generator(rho, Gop)
            local_var = local_phase_variance_from_qfi(Fq, M)
            zz = qzzb_phase_bound(rho, cutoff, W, tau_points=tau_points)
            guard_ratio = zz / local_var if local_var > 0 else np.inf
            pwrap = wrapping_probability_gaussian(local_var)
            g_eff = geff(eta_s, Ns, gamma, a=1.0)
            label = classify_boundary_point(
                g_eff, guard_ratio, pwrap,
                guard_ratio_threshold=cfg["classification"]["guard_ratio_threshold"],
                wrap_probability_threshold=cfg["classification"]["wrap_probability_threshold"],
                local_ratio_tolerance=cfg["classification"]["local_ratio_tolerance"],
            )
            cls_grid[gi, ei] = CLASS_TO_INT[label]
            guard_grid[gi, ei] = guard_ratio
            wrap_grid[gi, ei] = pwrap
            geff_grid[gi, ei] = g_eff
            rows.append({
                "Ns_diag": Ns, "cutoff": cutoff, "tail_probability": tmsv_tail_probability(Ns, cutoff),
                "eta_s": eta_s, "eta_i": eta_i, "Gamma": gamma,
                "Fq_fock": Fq, "local_variance": local_var, "qzzb_bound": zz,
                "qzzb_fidelity_convention": "squared_Uhlmann_fidelity_only",
                "guard_ratio": guard_ratio, "phase_wrapping_probability": pwrap,
                "Geff_analytic_envelope": g_eff, "classification": label
            })

    pd.DataFrame(rows).to_csv(Path(cfg["paths"]["data"]) / "qzzb_diagnostic_phase_diagram.csv", index=False)
    E, Gm = np.meshgrid(etas, gammas)

    plt.figure(figsize=(6.4, 5.0))
    im = plt.pcolormesh(E, Gm, np.log10(np.maximum(guard_grid, 1e-12)), shading="auto")
    plt.contour(E, Gm, geff_grid, levels=[1.0], colors="w", linewidths=1.2)
    plt.xlabel("Signal transmittance eta_s")
    plt.ylabel("Accumulated phase diffusion Gamma")
    plt.title("QZZB guard ratio map: log10(Sigma_ZZ / Var_local)")
    plt.colorbar(im, label="log10 guard ratio")
    savefig(Path(cfg["paths"]["figures"]) / "fig_qzzb_guard_ratio_map.png")

    plt.figure(figsize=(6.4, 5.0))
    cmap = plt.get_cmap("viridis", 3)
    im = plt.pcolormesh(E, Gm, cls_grid, shading="auto", cmap=cmap, vmin=-0.5, vmax=2.5)
    cbar = plt.colorbar(im, ticks=[0, 1, 2])
    cbar.ax.set_yticklabels([INT_TO_CLASS[i] for i in [0, 1, 2]])
    plt.contour(E, Gm, geff_grid, levels=[1.0], colors="w", linewidths=1.2)
    plt.xlabel("Signal transmittance eta_s")
    plt.ylabel("Accumulated phase diffusion Gamma")
    plt.title("QZZB-audited diagnostic classification")
    savefig(Path(cfg["paths"]["figures"]) / "fig_qzzb_diagnostic_classification.png")


if __name__ == "__main__":
    main()
