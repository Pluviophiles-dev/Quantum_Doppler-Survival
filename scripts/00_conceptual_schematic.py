#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
00_conceptual_schematic.py

System-level overview schematic for the photon-starved quantum Doppler manuscript.

Purpose
-------
This script generates a clean conceptual figure showing the full physical chain:

    TMSV source -> signal mode -> high-pressure gas pipe / Rayleigh Doppler phase encoding
                -> loss + phase diffusion + dark-count-limited receiver
                -> SQL / TMSV-QFI / QZZB guardrail decision layer

The figure is intentionally schematic, not a quantitative model.

Usage
-----
    python scripts/00_conceptual_schematic.py
    python scripts/00_conceptual_schematic.py --output figures/fig0_conceptual_schematic.pdf

Dependencies
------------
    numpy, matplotlib
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
import matplotlib as mpl


def _box(ax, xy, w, h, text, fc="#F7F7F7", ec="#222222", lw=1.5, fontsize=10, radius=0.08):
    x, y = xy
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0.025,rounding_size={radius}",
        fc=fc, ec=ec, lw=lw
    )
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fontsize)
    return patch


def _arrow(ax, start, end, text=None, rad=0.0, lw=1.8, color="#222222", fontsize=9, text_offset=(0, 0)):
    arrow = FancyArrowPatch(
        start, end,
        arrowstyle="-|>",
        mutation_scale=15,
        lw=lw,
        color=color,
        connectionstyle=f"arc3,rad={rad}"
    )
    ax.add_patch(arrow)
    if text:
        mx = 0.5 * (start[0] + end[0]) + text_offset[0]
        my = 0.5 * (start[1] + end[1]) + text_offset[1]
        ax.text(mx, my, text, ha="center", va="center", fontsize=fontsize, color=color)
    return arrow


def draw_conceptual_schematic(output: Path, dpi: int = 300) -> None:
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)

    mpl.rcParams.update({"font.size": 10, "axes.linewidth": 1.0, "mathtext.default": "regular"})

    fig, ax = plt.subplots(figsize=(13.8, 7.4))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 8)
    ax.axis("off")

    ax.text(7, 7.65, "System overview: photon-starved quantum-enhanced Doppler velocimetry",
            ha="center", va="center", fontsize=16, fontweight="bold")

    _box(ax, (0.45, 4.95), 2.25, 1.05, "TMSV source\nsignal + idler", fc="#EAF2FF", ec="#1F4E79", fontsize=11)
    ax.text(1.58, 4.73, r"$N_S$ resource accounting", ha="center", fontsize=9, color="#1F4E79")

    _box(ax, (0.55, 1.55), 2.05, 0.85, "idler storage /\nreference arm", fc="#F2EAFE", ec="#5E3C99", fontsize=10)
    ax.text(1.57, 1.33, r"idler preservation $\eta_i$", ha="center", fontsize=9, color="#5E3C99")

    pipe_x, pipe_y = 4.15, 3.12
    pipe_w, pipe_h = 4.15, 1.55
    pipe = FancyBboxPatch((pipe_x, pipe_y), pipe_w, pipe_h,
                          boxstyle="round,pad=0.02,rounding_size=0.25",
                          fc="#F4F4F4", ec="#3A3A3A", lw=1.8)
    ax.add_patch(pipe)
    ax.text(pipe_x + pipe_w / 2, pipe_y + pipe_h + 0.28, "high-pressure clean gas pipe", ha="center", fontsize=11, fontweight="bold")

    for yy in [3.45, 3.90, 4.35]:
        _arrow(ax, (4.35, yy), (7.95, yy), color="#777777", lw=1.2)
    ax.text(6.15, 3.02, r"gas velocity projection $v_q$", ha="center", fontsize=9, color="#555555")

    vol = Rectangle((5.78, 3.22), 0.70, 1.35, fc="#FFF3C4", ec="#9A7D0A", lw=1.2, alpha=0.95)
    ax.add_patch(vol)
    ax.text(6.13, 4.80, "Rayleigh\nvolume", ha="center", fontsize=9, color="#8A6D00")
    for dx, dy in [(0.82, 0.7), (0.95, 0.25), (0.83, -0.22)]:
        _arrow(ax, (6.42, 3.92), (7.65 + dx, 4.15 + dy), color="#CC7A00", lw=1.1)
    ax.text(8.25, 5.05, r"weak return $N_{ret}$", ha="center", fontsize=9, color="#A05A00")

    _box(ax, (4.35, 5.55), 3.70, 0.85, r"signal channel constraints: loss $\eta_s$  +  phase diffusion $\Gamma$",
         fc="#FFF0F0", ec="#B22222", fontsize=10)

    _box(ax, (9.25, 3.35), 2.05, 1.05, "receiver /\nphase estimator", fc="#EFFFF1", ec="#2E7D32", fontsize=11)
    ax.text(10.28, 3.12, r"dark/background counts: $N_{dark}+N_{bg}$", ha="center", fontsize=8.6, color="#2E7D32")

    _box(ax, (11.9, 3.05), 1.65, 1.75, "decision layer\n\nSQL baseline\nTMSV QFI\nQZZB guard", fc="#FFF8E7", ec="#9C6500", fontsize=9.3)
    _box(ax, (11.75, 5.45), 1.95, 0.62, "local-valid", fc="#E8F5E9", ec="#2E7D32", fontsize=9)
    _box(ax, (11.75, 4.78), 1.95, 0.62, "guarded", fc="#FFF8E1", ec="#C48A00", fontsize=9)
    _box(ax, (11.75, 2.25), 1.95, 0.62, "stop-extrapolation", fc="#FFEBEE", ec="#B71C1C", fontsize=9)

    _arrow(ax, (2.70, 5.48), (4.15, 4.45), text="signal mode", color="#1F4E79", text_offset=(0.00, 0.24))
    _arrow(ax, (2.70, 5.15), (0.95, 2.42), text="idler mode", color="#5E3C99", rad=0.25, text_offset=(-0.55, 0.05))
    _arrow(ax, (2.62, 1.98), (9.25, 3.42), text=r"correlation retained only if $\eta_i$ is high", color="#5E3C99", rad=-0.12, text_offset=(0.00, -0.35))
    _arrow(ax, (8.65, 4.05), (9.25, 3.88), text="collected return", color="#CC7A00", text_offset=(0.10, 0.32))
    _arrow(ax, (11.30, 3.88), (11.90, 3.88), color="#2E7D32")

    formula = (r"$N_{ret}=N_{in}nL\sigma_R(\lambda_0)(\Omega/4\pi)\eta_{sys}$"
               "\n" r"$G_{eff}=G_Q(\eta_s,N_S)\exp(-a\Gamma)$"
               "\n" r"$\hat v_q=\hat\phi/(|q|\tau_{int})$")
    _box(ax, (3.72, 0.55), 6.50, 1.05, formula, fc="#FFFFFF", ec="#666666", fontsize=10, radius=0.05)

    note = ("The schematic separates the macro photon budget, microscopic quantum-channel boundary, "
            "detector admissibility, and global QZZB guardrail.")
    ax.text(7, 0.22, note, ha="center", va="center", fontsize=9, color="#444444")

    fig.tight_layout()
    fig.savefig(output, dpi=dpi, bbox_inches="tight")
    if output.suffix.lower() == ".png":
        fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    elif output.suffix.lower() == ".pdf":
        fig.savefig(output.with_suffix(".png"), dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="figures/fig0_conceptual_schematic.png")
    parser.add_argument("--config", default=None, help="Accepted for run_all compatibility; not used by the schematic.")
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args()
    draw_conceptual_schematic(Path(args.output), dpi=args.dpi)
    print(f"Saved conceptual schematic to {args.output}")


if __name__ == "__main__":
    main()
