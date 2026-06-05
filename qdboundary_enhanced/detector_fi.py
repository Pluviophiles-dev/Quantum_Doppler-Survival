#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Detector likelihood and classical Fisher-information benchmark.

The model is intentionally simple but auditable:
    mu(phi) = mu0 * [1 + visibility * cos(phi)]
    k ~ Poisson(mu(phi) + dark/background counts)

It gives a receiver-level classical FI benchmark that can be compared with local
QFI-derived phase variance. This does not claim detector optimality.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class DetectorModel:
    nret: float = 0.16
    dark_rate_hz: float = 25.0
    background_rate_hz: float = 100.0
    gate_time_s: float = 1e-6
    visibility: float = 0.5
    samples: int = 1


def noise_counts(model: DetectorModel) -> float:
    return (model.dark_rate_hz + model.background_rate_hz) * model.gate_time_s


def mean_counts(phi: float, model: DetectorModel) -> float:
    # phase-dependent part, shifted to remain non-negative
    signal = model.nret * (1.0 + model.visibility * math.cos(phi))
    return max(signal + noise_counts(model), 1e-300)


def dmean_dphi(phi: float, model: DetectorModel) -> float:
    return -model.nret * model.visibility * math.sin(phi)


def poisson_classical_fi(phi: float, model: DetectorModel) -> float:
    mu = mean_counts(phi, model)
    dmu = dmean_dphi(phi, model)
    return model.samples * (dmu * dmu) / mu


def max_fi_over_phase(model: DetectorModel, grid_points: int = 2001) -> dict[str, float]:
    phis = np.linspace(0.0, 2.0 * math.pi, grid_points)
    fis = np.array([poisson_classical_fi(float(p), model) for p in phis])
    idx = int(np.argmax(fis))
    return {
        "max_classical_fi": float(fis[idx]),
        "phase_at_max_fi_rad": float(phis[idx]),
        "crlb_phase_variance": float(1.0 / fis[idx]) if fis[idx] > 0 else float("inf"),
        "noise_counts_per_gate": noise_counts(model),
        "sbr": model.nret / noise_counts(model) if noise_counts(model) > 0 else float("inf"),
    }


def fi_curve(model: DetectorModel, grid_points: int = 1001) -> dict[str, NDArray[np.float64]]:
    phis = np.linspace(0.0, 2.0 * math.pi, grid_points)
    fis = np.array([poisson_classical_fi(float(p), model) for p in phis])
    mus = np.array([mean_counts(float(p), model) for p in phis])
    return {"phi_rad": phis, "classical_fi": fis, "mean_counts": mus}


def scan_detector(nret_values, dark_rates, background_rates, gate_time_s=1e-6, visibility=0.5, samples=1):
    rows = []
    for nret in nret_values:
        for rd in dark_rates:
            for rb in background_rates:
                m = DetectorModel(nret=float(nret), dark_rate_hz=float(rd), background_rate_hz=float(rb),
                                  gate_time_s=gate_time_s, visibility=visibility, samples=samples)
                rows.append({"nret": float(nret), "dark_rate_hz": float(rd), "background_rate_hz": float(rb),
                             "gate_time_s": gate_time_s, "visibility": visibility, "samples": samples,
                             **max_fi_over_phase(m)})
    return rows
