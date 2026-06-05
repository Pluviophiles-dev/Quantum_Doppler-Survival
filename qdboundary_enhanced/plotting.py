#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Plot helpers for generated CSV tables."""
from __future__ import annotations

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


def heatmap_from_grid(df: pd.DataFrame, x: str, y: str, z: str, out: str | Path, title: str) -> None:
    pivot = df.pivot(index=y, columns=x, values=z)
    fig = plt.figure(figsize=(7, 5))
    plt.imshow(
        pivot.to_numpy(),
        origin="lower",
        aspect="auto",
        extent=[
            float(pivot.columns.min()), float(pivot.columns.max()),
            float(pivot.index.min()), float(pivot.index.max())
        ],
    )
    plt.xlabel(x)
    plt.ylabel(y)
    plt.title(title)
    plt.colorbar(label=z)
    plt.tight_layout()
    fig.savefig(out, dpi=240)
    plt.close(fig)
