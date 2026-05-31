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

from qdboundary.config import load_config
from qdboundary.covariance import covariance_purity, epr_correlation_strength, tmsv_covariance
from qdboundary.fock import number_operator_signal, prepare_noisy_tmsv_density, qfi_unitary_generator, tmsv_tail_probability
from qdboundary.formulas import coherent_qfi, gq_pure_loss
from qdboundary.plotting import savefig


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.json")
    args = parser.parse_args()
    cfg = load_config(args.config)

    Ns = float(cfg["model"]["Ns_diag"])
    cutoff = int(cfg["fock"]["cutoff"])
    gamma = 0.0
    etas = np.linspace(cfg["grids"]["eta_min"], cfg["grids"]["eta_max"], cfg["grids"]["eta_points_idler"])
    etai = np.linspace(cfg["grids"]["eta_min"], cfg["grids"]["eta_max"], cfg["grids"]["eta_points_idler"])
    Gop = number_operator_signal(cutoff)

    rows = []
    ratio_grid = np.zeros((len(etai), len(etas)))
    purity_grid = np.zeros_like(ratio_grid)
    corr_grid = np.zeros_like(ratio_grid)

    for i, ei in enumerate(etai):
        for j, es in enumerate(etas):
            rho = prepare_noisy_tmsv_density(Ns, cutoff, es, ei, gamma)
            F = qfi_unitary_generator(rho, Gop)
            Fcs = coherent_qfi(es, Ns)
            ratio = F / Fcs if Fcs > 0 else np.nan
            V = tmsv_covariance(Ns, es, ei)
            purity = covariance_purity(V)
            corr = epr_correlation_strength(V)
            ratio_grid[i, j] = ratio
            purity_grid[i, j] = purity
            corr_grid[i, j] = corr
            rows.append({
                "Ns_diag": Ns, "cutoff": cutoff, "tail_probability": tmsv_tail_probability(Ns, cutoff),
                "eta_s": es, "eta_i": ei, "gamma": gamma,
                "Fq_fock": F, "Fq_coherent": Fcs, "advantage_ratio_fock": ratio,
                "covariance_purity": purity, "epr_correlation_strength": corr,
                "ideal_idler_analytic_ratio": gq_pure_loss(es, Ns)
            })

    df = pd.DataFrame(rows)
    df.to_csv(Path(cfg["paths"]["data"]) / "idler_loss_sensitivity.csv", index=False)

    E, I = np.meshgrid(etas, etai)
    plt.figure(figsize=(6.0, 5.0))
    im = plt.pcolormesh(E, I, ratio_grid, shading="auto")
    plt.contour(E, I, ratio_grid, levels=[1.0], colors="k", linewidths=1.2)
    plt.xlabel("Signal transmittance eta_s")
    plt.ylabel("Idler transmittance eta_i")
    plt.title(f"Idler-loss-aware Fock QFI ratio, Ns={Ns}, cutoff={cutoff}")
    plt.colorbar(im, label="F_Q(TMSV noisy) / F_Q(coherent)")
    savefig(Path(cfg["paths"]["figures"]) / "fig_idler_loss_fock_qfi_ratio.png")

    plt.figure(figsize=(6.0, 5.0))
    im = plt.pcolormesh(E, I, corr_grid, shading="auto")
    plt.xlabel("Signal transmittance eta_s")
    plt.ylabel("Idler transmittance eta_i")
    plt.title("Gaussian covariance diagnostic: EPR block strength")
    plt.colorbar(im, label="||C_si||_F")
    savefig(Path(cfg["paths"]["figures"]) / "fig_idler_loss_covariance_correlation.png")

    # Cutoff convergence at a few boundary-relevant points.
    conv_rows = []
    points = [(0.9, 0.9), (0.9, 0.7), (0.65, 0.9), (0.5, 0.9)]
    for c in cfg["fock"]["cutoff_convergence_list"]:
        Gc = number_operator_signal(int(c))
        for es, ei in points:
            rho = prepare_noisy_tmsv_density(Ns, int(c), es, ei, 0.0)
            F = qfi_unitary_generator(rho, Gc)
            conv_rows.append({"cutoff": c, "Ns_diag": Ns, "eta_s": es, "eta_i": ei, "Fq": F, "ratio": F / coherent_qfi(es, Ns), "tail_probability": tmsv_tail_probability(Ns, int(c))})
    pd.DataFrame(conv_rows).to_csv(Path(cfg["paths"]["data"]) / "idler_loss_cutoff_convergence.csv", index=False)


if __name__ == "__main__":
    main()
