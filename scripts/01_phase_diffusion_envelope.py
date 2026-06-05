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
from qdboundary.formulas import gamma_max, geff, gq_pure_loss, eta_threshold_equal_total
from qdboundary.plotting import savefig


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.json")
    args = parser.parse_args()
    cfg = load_config(args.config)

    Ns_values = [10.0, 30.0, float(cfg["model"]["Ns_main"])]
    a_values = cfg["model"]["phase_diffusion_a_values"]
    eta = np.linspace(cfg["grids"]["eta_min"], cfg["grids"]["eta_max"], cfg["grids"]["eta_points_envelope"])

    rows = []
    plt.figure(figsize=(7.2, 4.8))
    for Ns in Ns_values:
        for a in a_values:
            gm = gamma_max(eta, Ns, a=a)
            for e, g in zip(eta, gm):
                rows.append({"Ns": Ns, "a": a, "eta_s": e, "gamma_max": g, "GQ": gq_pure_loss(e, Ns)})
            plt.plot(eta, gm, label=f"Ns={Ns:g}, a={a:g}")
    thr = eta_threshold_equal_total(float(cfg["model"]["Ns_main"]))
    if np.isfinite(thr) and 0 <= thr <= 1:
        plt.axvline(thr, linestyle="--", linewidth=1.0, label=f"equal-total threshold eta={thr:.3f}")
    plt.xlabel("Signal transmittance eta_s")
    plt.ylabel("Survival boundary Gamma_max")
    plt.title("Phase-diffusion uncertainty envelope")
    plt.ylim(0, cfg["grids"]["gamma_max"])
    plt.legend(fontsize=8, ncol=2)
    savefig(Path(cfg["paths"]["figures"]) / "fig_phase_diffusion_envelope.png")

    df = pd.DataFrame(rows)
    df.to_csv(Path(cfg["paths"]["data"]) / "phase_diffusion_envelope.csv", index=False)

    # Compact diagnostic grid for the baseline Ns_main, a=1.
    gamma = np.linspace(cfg["grids"]["gamma_min"], cfg["grids"]["gamma_max"], cfg["grids"]["gamma_points_envelope"])
    E, G = np.meshgrid(eta, gamma)
    Z = geff(E, cfg["model"]["Ns_main"], G, a=1.0)
    plt.figure(figsize=(6.2, 4.8))
    im = plt.pcolormesh(E, G, Z, shading="auto")
    plt.contour(E, G, Z, levels=[1.0], colors="k", linewidths=1.2)
    plt.colorbar(im, label="G_eff")
    plt.xlabel("Signal transmittance eta_s")
    plt.ylabel("Accumulated phase diffusion Gamma")
    plt.title("Baseline local screening map")
    savefig(Path(cfg["paths"]["figures"]) / "fig_baseline_geff_map.png")


if __name__ == "__main__":
    main()
