#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Audit the pure-loss TMSV QFI formula against finite-Fock SLD QFI.

The audit compares the current Fock-consistent analytic expression and the
legacy factor-2 expression against an exact finite-Fock calculation at Gamma=0.
It is intended to catch convention drift before manuscript figures are made.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from qdboundary.fock import number_operator_signal, prepare_noisy_tmsv_density, qfi_unitary_generator, tmsv_tail_probability
from qdboundary.formulas import coherent_qfi, gq_pure_loss, gq_pure_loss_legacy_factor2
from qdboundary.plotting import savefig


def truncated_tmsv_mean_ns(Ns: float, cutoff: int) -> float:
    if Ns == 0:
        return 0.0
    lam = Ns / (Ns + 1.0)
    n = np.arange(cutoff, dtype=float)
    p = (1.0 - lam) * lam ** n
    p /= np.sum(p)
    return float(np.sum(n * p))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", default="outputs_formula_audit")
    parser.add_argument("--rel-tol", type=float, default=5e-2)
    args = parser.parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # The default is intentionally lightweight so the audit runs quickly in CI.
    # Use the exact-dephasing sweep script for heavier cutoff-24/32 diagnostics.
    Ns_values = [0.5, 1.5, 3.0]
    cutoffs = [8, 12]
    etas = [0.5, 0.7, 0.9, 1.0]
    rows = []
    for Ns in Ns_values:
        for cutoff in cutoffs:
            Gop = number_operator_signal(cutoff)
            tail = tmsv_tail_probability(Ns, cutoff)
            ns_cut = truncated_tmsv_mean_ns(Ns, cutoff)
            for eta in etas:
                rho = prepare_noisy_tmsv_density(Ns, cutoff, eta, 1.0, 0.0)
                F = qfi_unitary_generator(rho, Gop)
                Fcs = coherent_qfi(eta, Ns)
                G_exact = F / Fcs if Fcs > 0 else np.nan
                G_current = float(gq_pure_loss(eta, Ns))
                G_legacy = float(gq_pure_loss_legacy_factor2(eta, Ns))
                rows.append({
                    "NS_target": Ns,
                    "cutoff": cutoff,
                    "NS_cut": ns_cut,
                    "NS_cut_over_NS_target": ns_cut / Ns,
                    "tail_probability": tail,
                    "eta_s": eta,
                    "Gamma": 0.0,
                    "F_Q_exact_TMSV": F,
                    "F_Q_coherent": Fcs,
                    "G_exact_Gamma0": G_exact,
                    "G_formula_current_fock_consistent": G_current,
                    "G_formula_legacy_factor2": G_legacy,
                    "relative_error_current": abs(G_exact - G_current) / max(abs(G_current), 1e-15),
                    "relative_error_legacy": abs(G_exact - G_legacy) / max(abs(G_legacy), 1e-15),
                    "current_formula_matches_within_tol": abs(G_exact - G_current) / max(abs(G_current), 1e-15) < args.rel_tol,
                    "legacy_formula_matches_within_tol": abs(G_exact - G_legacy) / max(abs(G_legacy), 1e-15) < args.rel_tol,
                })
    df = pd.DataFrame(rows)
    df.to_csv(outdir / "pure_loss_formula_audit.csv", index=False)
    highest = df.loc[df.groupby(["NS_target", "eta_s"])["cutoff"].idxmax()].copy()
    highest.to_csv(outdir / "pure_loss_formula_audit_highest_cutoff.csv", index=False)

    summary = {
        "points": int(len(df)),
        "highest_cutoff_points": int(len(highest)),
        "fraction_current_matches_5pct_highest": float(highest["current_formula_matches_within_tol"].mean()),
        "fraction_legacy_matches_5pct_highest": float(highest["legacy_formula_matches_within_tol"].mean()),
        "max_tail_probability_highest": float(highest["tail_probability"].max()),
        "interpretation": "The current analytic formula is the Fock-consistent convention used by the exact finite-Fock SLD-QFI implementation; the legacy factor-2 formula is retained only for audit/reproducibility.",
    }
    (outdir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    plt.figure(figsize=(6.0, 5.0))
    plt.scatter(highest["G_formula_current_fock_consistent"], highest["G_exact_Gamma0"], label="current fock-consistent", marker="o")
    plt.scatter(highest["G_formula_legacy_factor2"], highest["G_exact_Gamma0"], label="legacy factor-2", marker="x")
    lim = max(highest["G_exact_Gamma0"].max(), highest["G_formula_current_fock_consistent"].max(), highest["G_formula_legacy_factor2"].max()) * 1.05
    plt.plot([0, lim], [0, lim], "k--", linewidth=1.0)
    plt.xlabel("Analytic ratio")
    plt.ylabel("Finite-Fock exact ratio at Gamma=0")
    plt.title("Pure-loss TMSV QFI formula audit")
    plt.legend()
    savefig(outdir / "fig_pure_loss_formula_audit.png")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
