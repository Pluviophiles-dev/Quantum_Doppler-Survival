#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Conservative stress screens for missing propagation and instrument physics.

These functions do not implement full turbulence, multiple-scattering, or
instrument-level modeling.  They expose the missing physics as auditable risk
indices that can be swept and recorded next to the local quantum-bound outputs.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math


@dataclass(frozen=True)
class RefractiveIndexStress:
    """Order-of-magnitude refractive-index wandering screen."""

    wavelength_nm: float = 1064.0
    path_length_m: float = 0.01
    correlation_length_m: float = 1.0e-3
    sigma_n: float = 1.0e-7
    shape_factor: float = 1.0
    gamma_coefficient: float = 1.0


@dataclass(frozen=True)
class MultipleScatteringStress:
    """Single-pass optical-depth screen for Rayleigh multiple-scattering risk."""

    number_density_m3: float
    cross_section_m2: float
    path_length_m: float
    warning_optical_depth: float = 1.0e-3
    stop_optical_depth: float = 1.0e-2


@dataclass(frozen=True)
class InstrumentStress:
    """Minimal receiver/instrument readiness screen."""

    calibration_relative_uncertainty: float = 0.05
    timing_jitter_to_gate_ratio: float = 0.1
    background_to_signal_ratio: float = 1.0
    detector_dead_time_to_gate_ratio: float = 0.0
    max_calibration_relative_uncertainty: float = 0.02
    max_timing_jitter_to_gate_ratio: float = 0.2
    max_background_to_signal_ratio: float = 1.0
    max_dead_time_to_gate_ratio: float = 0.05


def refractive_phase_variance(stress: RefractiveIndexStress) -> float:
    """Return a coarse phase-variance proxy from refractive-index wandering.

    The scaling is k^2 sigma_n^2 L Lc times a shape factor.  It is a screening
    estimate, not a turbulence spectrum or CFD-coupled propagation solution.
    """
    if stress.wavelength_nm <= 0:
        raise ValueError("wavelength_nm must be positive.")
    if stress.path_length_m < 0 or stress.correlation_length_m < 0:
        raise ValueError("path and correlation lengths must be non-negative.")
    if stress.sigma_n < 0 or stress.shape_factor < 0 or stress.gamma_coefficient < 0:
        raise ValueError("sigma_n, shape_factor, and gamma_coefficient must be non-negative.")
    k = 2.0 * math.pi / (stress.wavelength_nm * 1e-9)
    return float((k * k) * (stress.sigma_n ** 2) * stress.path_length_m *
                 stress.correlation_length_m * stress.shape_factor)


def gamma_from_refractive_stress(stress: RefractiveIndexStress) -> float:
    return float(stress.gamma_coefficient * refractive_phase_variance(stress))


def optical_depth(stress: MultipleScatteringStress) -> float:
    if stress.number_density_m3 < 0 or stress.cross_section_m2 < 0 or stress.path_length_m < 0:
        raise ValueError("density, cross section, and path length must be non-negative.")
    return float(stress.number_density_m3 * stress.cross_section_m2 * stress.path_length_m)


def multiple_scattering_verdict(stress: MultipleScatteringStress) -> dict[str, float | str | bool]:
    tau = optical_depth(stress)
    if tau >= stress.stop_optical_depth:
        verdict = "stop"
    elif tau >= stress.warning_optical_depth:
        verdict = "guarded"
    else:
        verdict = "pass_single_scattering_screen"
    return {
        **asdict(stress),
        "optical_depth": tau,
        "multiple_scattering_screen": verdict,
        "scope_note": "optical-depth screen only; not a radiative-transfer or wave multiple-scattering solver",
    }


def mode_purity_from_stress(
    turbulence_mode_purity: float,
    multiple_scattering_tau: float,
    aperture_scintillation_index: float = 0.0,
    floor: float = 0.0,
) -> float:
    """Combine simple stress penalties into a conservative mode-purity factor."""
    if not (0.0 <= turbulence_mode_purity <= 1.0):
        raise ValueError("turbulence_mode_purity must be in [0,1].")
    if multiple_scattering_tau < 0 or aperture_scintillation_index < 0:
        raise ValueError("stress indices must be non-negative.")
    if not (0.0 <= floor <= 1.0):
        raise ValueError("floor must be in [0,1].")
    penalty = math.exp(-multiple_scattering_tau) / (1.0 + aperture_scintillation_index)
    return float(max(floor, min(1.0, turbulence_mode_purity * penalty)))


def instrument_readiness_verdict(stress: InstrumentStress) -> dict[str, float | str | bool]:
    """Return a conservative instrument-readiness screen."""
    values = asdict(stress)
    for key, value in values.items():
        if value < 0:
            raise ValueError(f"{key} must be non-negative.")
    failures = []
    if stress.calibration_relative_uncertainty > stress.max_calibration_relative_uncertainty:
        failures.append("calibration_relative_uncertainty")
    if stress.timing_jitter_to_gate_ratio > stress.max_timing_jitter_to_gate_ratio:
        failures.append("timing_jitter_to_gate_ratio")
    if stress.background_to_signal_ratio > stress.max_background_to_signal_ratio:
        failures.append("background_to_signal_ratio")
    if stress.detector_dead_time_to_gate_ratio > stress.max_dead_time_to_gate_ratio:
        failures.append("detector_dead_time_to_gate_ratio")
    verdict = "instrument_guarded" if failures else "instrument_screen_pass"
    return {
        **values,
        "instrument_screen": verdict,
        "failed_terms": ";".join(failures) if failures else "none",
        "scope_note": "readiness screen only; not calibration, validation, or GUM uncertainty certification",
    }


def stress_summary(
    refractive: RefractiveIndexStress,
    scattering: MultipleScatteringStress,
    instrument: InstrumentStress,
) -> dict[str, float | str | bool]:
    ms = multiple_scattering_verdict(scattering)
    ins = instrument_readiness_verdict(instrument)
    return {
        "gamma_refractive_proxy": gamma_from_refractive_stress(refractive),
        "phase_variance_refractive_proxy": refractive_phase_variance(refractive),
        "multiple_scattering_optical_depth": ms["optical_depth"],
        "multiple_scattering_screen": ms["multiple_scattering_screen"],
        "instrument_screen": ins["instrument_screen"],
        "instrument_failed_terms": ins["failed_terms"],
        "stress_scope_note": "combined conservative screens; not full turbulence, multiple-scattering, or instrument validation",
    }
