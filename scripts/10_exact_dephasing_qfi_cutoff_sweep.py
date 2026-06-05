#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Exact finite-Fock non-Gaussian phase-diffusion QFI cutoff sweep.

This script evaluates the signal-phase QFI after independent signal/idler loss
and exact signal-mode number dephasing in a truncated two-mode Fock basis.  It
is a diagnostic finite-cutoff calculation.  Rows with large TMSV tail
probability or missing cutoff convergence must not be called converged exact
high-NS results.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from qdboundary.fock import (
    number_operator_signal,
    prepare_noisy_tmsv_density,
    qfi_unitary_generator,
    tmsv_tail_probability,
)
from qdboundary.formulas import coherent_qfi, geff, gq_pure_loss, gq_pure_loss_legacy_factor2
from qdboundary.plotting import savefig


PRESETS = {
    "smoke": {
        "Ns": [0.5, 1.5, 3.0],
        "cutoffs": [8, 12],
        "eta_s": [0.5, 0.7, 0.9],
        "Gamma": [0.0, 0.5, 1.0],
        "eta_i": [1.0, 0.9],
    },
    "coarse": {
        "Ns": [0.5, 1.5, 3.0, 5.0],
        "cutoffs": [8, 12, 16, 24],
        "eta_s": [0.5, 0.7, 0.9, 1.0],
        "Gamma": [0.0, 0.5, 1.0, 2.0, 3.0],
        "eta_i": [1.0, 0.9, 0.7],
    },
    "ns10_stress": {
        "Ns": [10.0],
        "cutoffs": [24, 32],
        "eta_s": [0.7, 0.9],
        "Gamma": [0.0, 0.5, 1.0, 2.0, 3.0],
        "eta_i": [1.0, 0.9, 0.7],
    },
    "full": {
        "Ns": [0.5, 1.5, 3.0, 5.0, 10.0],
        "cutoffs": [8, 12, 16, 24, 32],
        "eta_s": [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        "Gamma": [0.0, 0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0],
        "eta_i": [1.0, 0.9, 0.7],
    },
}


def truncated_tmsv_mean_ns(Ns: float, cutoff: int) -> float:
    """Mean photons per mode of the normalized truncated TMSV source."""
    if Ns < 0:
        raise ValueError("Ns must be non-negative.")
    if cutoff < 1:
        raise ValueError("cutoff must be positive.")
    if Ns == 0:
        return 0.0
    lam = Ns / (Ns + 1.0)
    n = np.arange(cutoff, dtype=float)
    p = (1.0 - lam) * lam ** n
    p /= np.sum(p)
    return float(np.sum(n * p))


def fock_consistent_pure_loss_ratio(eta_s: float, Ns: float) -> float:
    """Pure-loss TMSV/coherent ratio matched by finite-Fock SLD QFI at Gamma=0."""
    if Ns <= 0 or eta_s <= 0:
        return float("nan")
    return float(gq_pure_loss(float(eta_s), float(Ns)))


def row_status(row: pd.Series, tail_warn: float) -> str:
    if bool(row.get("converged_flag", False)):
        if row["tail_probability"] <= tail_warn:
            return "converged_diagnostic"
        return "cutoff_converged_but_tail_large"
    if math.isnan(float(row.get("convergence_relative_change_vs_previous_cutoff", math.nan))):
        return "no_previous_cutoff"
    return "not_converged_by_5pct_rule"


def add_convergence_flags(df: pd.DataFrame, rel_tol: float, tail_warn: float) -> pd.DataFrame:
    df = df.sort_values(["NS_target", "eta_s", "eta_i", "Gamma", "cutoff"]).copy()
    df["convergence_relative_change_vs_previous_cutoff"] = np.nan
    df["converged_flag"] = False
    group_cols = ["NS_target", "eta_s", "eta_i", "Gamma"]
    for _, idx in df.groupby(group_cols).groups.items():
        sub = df.loc[list(idx)].sort_values("cutoff")
        prev_qfi = None
        for row_idx, row in sub.iterrows():
            qfi = float(row["F_Q_exact_TMSV"])
            if prev_qfi is not None and qfi > 0:
                rel = abs(qfi - prev_qfi) / qfi
                df.loc[row_idx, "convergence_relative_change_vs_previous_cutoff"] = rel
                df.loc[row_idx, "converged_flag"] = bool(rel < rel_tol)
            prev_qfi = qfi
    df["status"] = [row_status(row, tail_warn) for _, row in df.iterrows()]
    return df


def compute_rows(params: dict[str, list[float]], max_points: int | None = None) -> list[dict[str, float | str | bool]]:
    rows: list[dict[str, float | str | bool]] = []
    total = (
        len(params["Ns"]) * len(params["cutoffs"]) * len(params["eta_s"]) *
        len(params["Gamma"]) * len(params["eta_i"])
    )
    if max_points is not None and total > max_points:
        raise RuntimeError(f"Requested {total} points, above --max-points={max_points}.")
    done = 0
    t0 = time.time()
    for Ns in params["Ns"]:
        for cutoff in params["cutoffs"]:
            Gop = number_operator_signal(int(cutoff))
            NS_cut = truncated_tmsv_mean_ns(float(Ns), int(cutoff))
            tail = tmsv_tail_probability(float(Ns), int(cutoff))
            for eta_s in params["eta_s"]:
                F_coh = float(coherent_qfi(float(eta_s), float(Ns)))
                G_pure = float(gq_pure_loss(float(eta_s), float(Ns)))
                G_pure_legacy = float(gq_pure_loss_legacy_factor2(float(eta_s), float(Ns)))
                G_pure_fock = fock_consistent_pure_loss_ratio(float(eta_s), float(Ns))
                for eta_i in params["eta_i"]:
                    for Gamma in params["Gamma"]:
                        rho = prepare_noisy_tmsv_density(float(Ns), int(cutoff), float(eta_s), float(eta_i), float(Gamma))
                        F_exact = qfi_unitary_generator(rho, Gop)
                        G_exact = F_exact / F_coh if F_coh > 0 else float("nan")
                        G_env = float(geff(float(eta_s), float(Ns), float(Gamma), a=1.0))
                        G_env_fock = float(G_pure_fock * math.exp(-float(Gamma)))
                        rows.append({
                            "NS_target": float(Ns),
                            "cutoff": int(cutoff),
                            "NS_cut": NS_cut,
                            "NS_cut_over_NS_target": NS_cut / float(Ns) if Ns > 0 else float("nan"),
                            "tail_probability": tail,
                            "eta_s": float(eta_s),
                            "eta_i": float(eta_i),
                            "Gamma": float(Gamma),
                            "F_Q_exact_TMSV": F_exact,
                            "F_Q_coherent": F_coh,
                            "G_exact": G_exact,
                            "G_Q_pure_loss": G_pure,
                            "G_eff_envelope": G_env,
                            "G_exact_minus_G_eff": G_exact - G_env,
                            "G_exact_over_G_eff": G_exact / G_env if G_env > 0 else float("nan"),
                            "exact_le_envelope": bool(G_exact <= G_env),
                            "G_Q_pure_loss_legacy_factor2": G_pure_legacy,
                            "G_eff_legacy_factor2_envelope": G_pure_legacy * math.exp(-float(Gamma)),
                            "G_Q_pure_loss_fock_consistent": G_pure_fock,
                            "G_eff_fock_consistent_envelope": G_env_fock,
                            "G_exact_minus_G_eff_fock_consistent": G_exact - G_env_fock,
                            "G_exact_over_G_eff_fock_consistent": G_exact / G_env_fock if G_env_fock > 0 else float("nan"),
                            "exact_le_fock_consistent_envelope": bool(G_exact <= G_env_fock),
                            "pure_loss_formula_audit_note": "current package formula is Fock-consistent; legacy factor-2 expression is retained only as an audit column",
                            "qfi_method": "SLD_spectral_formula_finite_Fock",
                            "scope_note": "finite-cutoff exact number-dephasing diagnostic; not a high-NS convergence certificate unless convergence flag and tail are acceptable",
                        })
                        done += 1
                        if done % 25 == 0 or done == total:
                            elapsed = time.time() - t0
                            print(f"Computed {done}/{total} points in {elapsed:.1f}s", flush=True)
    return rows


def make_figures(df: pd.DataFrame, outdir: Path) -> None:
    figures = outdir / "figures"
    figures.mkdir(parents=True, exist_ok=True)

    conv = (
        df[df["eta_s"].isin([0.7, 0.9]) & df["eta_i"].eq(1.0) & df["Gamma"].isin([0.0, 1.0])]
        .sort_values(["NS_target", "Gamma", "eta_s", "cutoff"])
    )
    if not conv.empty:
        plt.figure(figsize=(7.2, 4.8))
        for (Ns, eta_s, Gamma), sub in conv.groupby(["NS_target", "eta_s", "Gamma"]):
            plt.plot(sub["cutoff"], sub["G_exact"], marker="o", label=f"Ns={Ns:g}, eta={eta_s:g}, Gm={Gamma:g}")
        plt.xlabel("Fock cutoff")
        plt.ylabel("G_exact = F_Q(TMSV)/F_Q(coherent)")
        plt.title("Exact dephasing-QFI cutoff convergence diagnostic")
        plt.legend(fontsize=7, ncol=2)
        savefig(figures / "fig_exact_dephasing_qfi_cutoff.png")

    highest = df.loc[df.groupby(["NS_target", "eta_s", "eta_i", "Gamma"])["cutoff"].idxmax()].copy()
    plt.figure(figsize=(5.8, 5.2))
    colors = np.where(highest["converged_flag"], "#1b9e77", "#d95f02")
    plt.scatter(highest["G_eff_envelope"], highest["G_exact"], c=colors, s=28, alpha=0.8)
    lim = max(float(highest["G_eff_envelope"].max()), float(highest["G_exact"].max()), 1.0)
    plt.plot([0, lim], [0, lim], "k--", linewidth=1.0, label="G_exact = G_eff")
    plt.xlabel("G_eff envelope")
    plt.ylabel("G_exact finite-Fock QFI ratio")
    plt.title("Exact dephasing QFI vs heuristic envelope")
    plt.legend()
    savefig(figures / "fig_exact_dephasing_qfi_vs_geff.png")

    boundary = highest[highest["eta_i"].eq(1.0)].copy()
    if not boundary.empty:
        plt.figure(figsize=(7.0, 4.8))
        for Ns, sub in boundary.groupby("NS_target"):
            sub2 = sub[sub["eta_s"].eq(0.9)].sort_values("Gamma")
            if not sub2.empty:
                plt.plot(sub2["Gamma"], sub2["G_exact"], marker="o", label=f"exact Ns={Ns:g}")
                plt.plot(sub2["Gamma"], sub2["G_eff_envelope"], linestyle="--", label=f"env Ns={Ns:g}")
        plt.axhline(1.0, color="k", linewidth=1.0)
        plt.xlabel("Gamma")
        plt.ylabel("Advantage ratio")
        plt.title("QFI ratio vs phase diffusion at eta_s=0.9, eta_i=1.0")
        plt.legend(fontsize=7, ncol=2)
        savefig(figures / "fig_exact_dephasing_qfi_vs_gamma.png")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preset", choices=sorted(PRESETS), default="coarse")
    parser.add_argument("--outdir", default="outputs_exact_dephasing_qfi")
    parser.add_argument("--max-points", type=int, default=None)
    parser.add_argument("--rel-tol", type=float, default=0.05)
    parser.add_argument("--tail-warn", type=float, default=0.05)
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    params = PRESETS[args.preset]
    (outdir / "sweep_parameters.json").write_text(json.dumps(params, indent=2), encoding="utf-8")

    rows = compute_rows(params, max_points=args.max_points)
    df = pd.DataFrame(rows)
    df = add_convergence_flags(df, args.rel_tol, args.tail_warn)

    sweep_path = outdir / "exact_dephasing_qfi_cutoff_sweep.csv"
    df.to_csv(sweep_path, index=False)

    highest = df.loc[df.groupby(["NS_target", "eta_s", "eta_i", "Gamma"])["cutoff"].idxmax()].copy()
    highest.to_csv(outdir / "exact_dephasing_qfi_vs_envelope.csv", index=False)
    summary = {
        "preset": args.preset,
        "points": int(len(df)),
        "highest_cutoff_points": int(len(highest)),
        "fraction_highest_exact_le_envelope": float(highest["exact_le_envelope"].mean()) if len(highest) else float("nan"),
        "fraction_highest_exact_le_fock_consistent_envelope": float(highest["exact_le_fock_consistent_envelope"].mean()) if len(highest) else float("nan"),
        "fraction_highest_converged_by_5pct_rule": float(highest["converged_flag"].mean()) if len(highest) else float("nan"),
        "max_tail_probability_highest": float(highest["tail_probability"].max()) if len(highest) else float("nan"),
        "scope_note": "Objective finite-cutoff diagnostic; inspect convergence and tail probability before using in manuscript. G_eff is an optimistic candidate envelope, not a rigorous bound.",
    }
    (outdir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    make_figures(df, outdir)
    print(json.dumps(summary, indent=2), flush=True)
    print(f"Outputs written to {outdir.resolve()}", flush=True)


if __name__ == "__main__":
    main()
