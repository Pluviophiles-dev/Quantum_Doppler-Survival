#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Parameterized Rayleigh photon-budget to effective signal-channel bridge.

This module keeps the macroscopic Rayleigh return and the conditional
single-mode quantum-channel parameters separate.  In particular, the returned
photon number Nret is a radiometric/multimode photon-count budget, whereas the
TMSV benchmark requires a defined signal mode.  The bridge below therefore
computes and reports explicit spatial and temporal mode-overlap factors instead
of hiding them in a magic scalar.

The overlap model is still a diagnostic model, not a full CFD/turbulence or
multiple-scattering solver.  It is meant to make the required single-mode
transduction assumptions auditable and easy to stress test.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Optional
import numpy as np

H = 6.62607015e-34
C = 299792458.0
KB = 1.380649e-23
NA = 6.02214076e23


@dataclass(frozen=True)
class GasState:
    pressure_mpa: float = 30.0
    temperature_k: float = 298.15
    methane_mole_fraction: float = 1.0
    hydrogen_mole_fraction: float = 0.0
    compressibility_z: float = 0.86  # rough high-pressure methane proxy; replace by GERG/PR for final paper.


@dataclass(frozen=True)
class OpticalBudget:
    wavelength_nm: float = 1064.0
    pulse_energy_j: float = 1e-6
    probe_length_m: float = 0.01
    collection_fraction: float = 1e-7  # Omega / 4pi
    window_transmission: float = 0.95
    optical_train_transmission: float = 0.80
    detector_efficiency: float = 0.65

    # Optional legacy/manual override.  If None, the spatial overlap is computed
    # from the Gaussian-mode parameters below.  Keeping the override is useful
    # for reproducing older sensitivity tables, but the returned diagnostics
    # explicitly label it as an override.
    mode_overlap: Optional[float] = None

    polarization_factor: float = 0.50
    king_factor: float = 1.04
    refractive_index_ref: float = 1.000444  # methane-ish visible proxy; replace with wavelength/gas model.

    # Spatial single-mode coupling model.  The amplitude modes are modeled as
    # circular Gaussian modes at the receiver/mixing plane.  Turbulence and
    # multiple scattering should be represented by reducing these overlaps or by
    # replacing the model, not by re-identifying Nret with eta_s.
    local_oscillator_waist_m: float = 1.0e-3
    return_mode_waist_m: float = 1.5e-3
    transverse_offset_m: float = 0.8e-3
    angular_mismatch_rad: float = 1.0e-4
    turbulence_mode_purity: float = 0.70

    # Temporal mode/gate overlap for the signal return.  These parameters
    # prevent the conditional eta_s from silently assuming perfect time-mode
    # matching.
    signal_pulse_rms_s: float = 2.0e-9
    receiver_gate_rms_s: float = 5.0e-9
    signal_gate_offset_s: float = 0.0
    detector_jitter_rms_s: float = 0.5e-9


def _clip01(x: float) -> float:
    return float(max(0.0, min(1.0, x)))


def _require_fraction(name: str, value: float) -> None:
    if not (0.0 <= float(value) <= 1.0):
        raise ValueError(f"{name} must be in [0,1].")


def validate_gas_state(gas: GasState) -> None:
    """Validate gas inputs used by the diagnostic transduction bridge."""
    if gas.pressure_mpa < 0:
        raise ValueError("pressure_mpa must be non-negative.")
    if gas.temperature_k <= 0:
        raise ValueError("temperature_k must be positive.")
    if gas.compressibility_z <= 0:
        raise ValueError("compressibility_z must be positive.")
    _require_fraction("methane_mole_fraction", gas.methane_mole_fraction)
    _require_fraction("hydrogen_mole_fraction", gas.hydrogen_mole_fraction)
    if gas.methane_mole_fraction + gas.hydrogen_mole_fraction > 1.0 + 1e-12:
        raise ValueError("Specified mole fractions must not sum above 1.")


def validate_optical_budget(opt: OpticalBudget) -> None:
    """Validate optical inputs used by the diagnostic transduction bridge."""
    if opt.wavelength_nm <= 0:
        raise ValueError("wavelength_nm must be positive.")
    if opt.pulse_energy_j < 0 or opt.probe_length_m < 0:
        raise ValueError("pulse_energy_j and probe_length_m must be non-negative.")
    for name in (
        "collection_fraction",
        "window_transmission",
        "optical_train_transmission",
        "detector_efficiency",
        "polarization_factor",
        "turbulence_mode_purity",
    ):
        _require_fraction(name, getattr(opt, name))
    if opt.king_factor <= 0:
        raise ValueError("king_factor must be positive.")
    if opt.mode_overlap is not None:
        _require_fraction("mode_overlap", opt.mode_overlap)
    if opt.refractive_index_ref <= 0:
        raise ValueError("refractive_index_ref must be positive.")
    if opt.local_oscillator_waist_m <= 0 or opt.return_mode_waist_m <= 0:
        raise ValueError("Gaussian waists must be positive.")
    if opt.transverse_offset_m < 0 or opt.angular_mismatch_rad < 0:
        raise ValueError("Offsets and angular mismatch must be non-negative.")
    if opt.signal_pulse_rms_s <= 0 or opt.receiver_gate_rms_s <= 0 or opt.detector_jitter_rms_s < 0:
        raise ValueError("Temporal RMS widths must be positive and jitter must be non-negative.")


def emitted_photons(pulse_energy_j: float, wavelength_nm: float) -> float:
    lam = wavelength_nm * 1e-9
    return pulse_energy_j / (H * C / lam)


def ideal_gas_number_density(gas: GasState) -> float:
    P = gas.pressure_mpa * 1e6
    return P / (gas.compressibility_z * KB * gas.temperature_k)


def rayleigh_cross_section_proxy(wavelength_nm: float, n_ref: float = 1.000444, king_factor: float = 1.04) -> float:
    """
    Lorentz-Lorenz style proxy:
        sigma = 24*pi^3 / (N0^2 * lambda^4) * ((n^2-1)/(n^2+2))^2 * F_K

    N0 is Loschmidt number at STP. This is a proxy for boundary analysis; for publication,
    cite and replace with gas- and wavelength-specific molecular cross sections when possible.
    """
    lam = wavelength_nm * 1e-9
    N0 = 2.686780111e25
    n2 = n_ref ** 2
    return (24.0 * math.pi ** 3 / (N0 ** 2 * lam ** 4)) * ((n2 - 1.0) / (n2 + 2.0)) ** 2 * king_factor


def gaussian_spatial_mode_overlap(
    wavelength_nm: float,
    local_oscillator_waist_m: float,
    return_mode_waist_m: float,
    transverse_offset_m: float = 0.0,
    angular_mismatch_rad: float = 0.0,
    turbulence_mode_purity: float = 1.0,
) -> float:
    """Return a normalized 2D Gaussian spatial-mode overlap efficiency.

    The expression is the squared overlap of two circular Gaussian amplitude
    modes, multiplied by simple penalties for transverse displacement, angular
    mismatch, and an externally supplied turbulence/multiple-scattering mode
    purity factor:

        ASCII equivalent: eta = | integral u_LO*(r) u_ret(r) exp(i k theta x) d^2r |^2.

        eta = |∫ u_LO*(r) u_ret(r) exp(i k theta x) d^2r|^2.

    It is not a turbulence propagation solver; it is an auditable single-mode
    coupling model that replaces unexplained constants in the transduction
    bridge.
    """
    if local_oscillator_waist_m <= 0 or return_mode_waist_m <= 0:
        raise ValueError("Gaussian waists must be positive.")
    if transverse_offset_m < 0 or angular_mismatch_rad < 0:
        raise ValueError("Offsets and angular mismatch must be non-negative magnitudes.")

    w1 = float(local_oscillator_waist_m)
    w2 = float(return_mode_waist_m)
    lam = float(wavelength_nm) * 1e-9
    k = 2.0 * math.pi / lam
    denom = w1 * w1 + w2 * w2

    waist_match = (2.0 * w1 * w2 / denom) ** 2
    offset_penalty = math.exp(-2.0 * transverse_offset_m * transverse_offset_m / denom)
    # Fourier overlap of a tilted Gaussian mode.  The coefficient is written as
    # a conservative diagnostic penalty rather than a precision optical design.
    tilt_penalty = math.exp(-(k * angular_mismatch_rad) ** 2 * (w1 * w1 * w2 * w2) / (2.0 * denom))
    return _clip01(waist_match * offset_penalty * tilt_penalty * turbulence_mode_purity)


def gaussian_temporal_mode_overlap(
    signal_pulse_rms_s: float,
    receiver_gate_rms_s: float,
    time_offset_s: float = 0.0,
    detector_jitter_rms_s: float = 0.0,
) -> float:
    """Return a Gaussian temporal-mode/gate overlap efficiency.

    This is the squared overlap of two normalized Gaussian temporal amplitudes
    after broadening the receiver gate by detector timing jitter.  It makes the
    assumed time-mode purity explicit and can be tightened for fast-flow or
    high-jitter receiver models.
    """
    if signal_pulse_rms_s <= 0 or receiver_gate_rms_s <= 0 or detector_jitter_rms_s < 0:
        raise ValueError("Temporal RMS widths must be positive and jitter must be non-negative.")
    s1 = float(signal_pulse_rms_s)
    s2 = math.sqrt(float(receiver_gate_rms_s) ** 2 + float(detector_jitter_rms_s) ** 2)
    denom = s1 * s1 + s2 * s2
    width_match = 2.0 * s1 * s2 / denom
    timing_penalty = math.exp(-(float(time_offset_s) ** 2) / denom)
    return _clip01(width_match * timing_penalty)


def computed_signal_mode_overlap(opt: OpticalBudget) -> dict[str, float | str]:
    validate_optical_budget(opt)
    spatial_computed = gaussian_spatial_mode_overlap(
        opt.wavelength_nm,
        opt.local_oscillator_waist_m,
        opt.return_mode_waist_m,
        opt.transverse_offset_m,
        opt.angular_mismatch_rad,
        opt.turbulence_mode_purity,
    )
    temporal = gaussian_temporal_mode_overlap(
        opt.signal_pulse_rms_s,
        opt.receiver_gate_rms_s,
        opt.signal_gate_offset_s,
        opt.detector_jitter_rms_s,
    )
    if opt.mode_overlap is None:
        spatial_used = spatial_computed
        source = "computed_gaussian_spatial_overlap"
    else:
        spatial_used = _clip01(float(opt.mode_overlap))
        source = "explicit_spatial_overlap_override"
    total = _clip01(spatial_used * temporal)
    return {
        "spatial_mode_overlap_computed": spatial_computed,
        "spatial_mode_overlap_used": spatial_used,
        "temporal_mode_overlap": temporal,
        "single_mode_overlap_total": total,
        "mode_overlap_source": source,
    }


def idler_time_gate_efficiency(
    signal_gate_rms_s: float,
    idler_gate_rms_s: float,
    relative_delay_error_s: float = 0.0,
    joint_jitter_rms_s: float = 0.0,
    idler_memory_efficiency: float = 1.0,
) -> float:
    """Return a diagnostic idler/signal time-registration efficiency.

    This factor is intended to multiply the independently specified idler-path
    transmission eta_i when a protocol requires signal and idler detection gates
    to refer to the same temporal mode.  It catches the otherwise hidden
    assumption that the local idler/reference arm is perfectly aligned with the
    Rayleigh-return gate.
    """
    gate_overlap = gaussian_temporal_mode_overlap(
        signal_gate_rms_s,
        idler_gate_rms_s,
        relative_delay_error_s,
        joint_jitter_rms_s,
    )
    return _clip01(gate_overlap * idler_memory_efficiency)


def rayleigh_return_photons(gas: GasState, opt: OpticalBudget) -> dict[str, float]:
    validate_gas_state(gas)
    validate_optical_budget(opt)
    Nin = emitted_photons(opt.pulse_energy_j, opt.wavelength_nm)
    validate_gas_state(gas)
    validate_optical_budget(opt)
    ndens = ideal_gas_number_density(gas)
    sigma = rayleigh_cross_section_proxy(opt.wavelength_nm, opt.refractive_index_ref, opt.king_factor)
    eta_macro = opt.window_transmission * opt.optical_train_transmission * opt.detector_efficiency
    Nret = Nin * ndens * opt.probe_length_m * sigma * opt.collection_fraction * eta_macro
    zero_prob = math.exp(-Nret) if Nret < 700 else 0.0
    return {
        "Nin": Nin,
        "number_density_m3": ndens,
        "sigma_rayleigh_m2": sigma,
        "eta_macro_no_scatter_no_collection": eta_macro,
        "Nret": Nret,
        "zero_count_probability": zero_prob,
        "model_scope": "single_scattering_homogeneous_photon_budget_proxy",
        "claim_level_max": "engineering_screen",
    }


def effective_signal_channel_eta(gas: GasState, opt: OpticalBudget) -> dict[str, float | str]:
    """
    Construct a transparent effective conditional eta_s bridge.

    eta_s_eff here is not identical to Nret/Nin. It is a mode-defined
    transduction estimate:
        eta_s_eff = scatter_probability * collection_fraction * spatial_overlap
                    * temporal_overlap * polarization_factor * window
                    * optical_train * detector_efficiency.

    For TMSV channel modeling, the returned
    eta_conditional_after_collection excludes scattering and collection and
    describes the conditional collected mode after an auditable single-mode
    projection has been defined.
    """
    ndens = ideal_gas_number_density(gas)
    sigma = rayleigh_cross_section_proxy(opt.wavelength_nm, opt.refractive_index_ref, opt.king_factor)
    p_scatter = ndens * opt.probe_length_m * sigma
    eta_common = opt.window_transmission * opt.optical_train_transmission * opt.detector_efficiency
    overlaps = computed_signal_mode_overlap(opt)
    single_mode_overlap = float(overlaps["single_mode_overlap_total"])
    eta_total_source_to_detector_mode = (
        p_scatter * opt.collection_fraction * single_mode_overlap * opt.polarization_factor * eta_common
    )
    eta_conditional_after_collection = single_mode_overlap * opt.polarization_factor * eta_common
    eta_conditional_after_scattering = opt.collection_fraction * single_mode_overlap * opt.polarization_factor * eta_common
    return {
        "scatter_probability_single_pass": p_scatter,
        "eta_total_source_to_detector_mode": eta_total_source_to_detector_mode,
        "eta_conditional_after_scattering": eta_conditional_after_scattering,
        "eta_conditional_after_collection": eta_conditional_after_collection,
        "eta_common_window_train_detector": eta_common,
        "model_scope": "auditable_single_mode_coupling_screen",
        "claim_level_max": "conditional_local_boundary",
        "not_implemented": "CFD turbulence propagation; multiple-scattering radiative transfer; calibrated receiver validation",
        **overlaps,
    }


def photon_regime(Nret: float) -> str:
    p0 = math.exp(Nret * -1.0) if Nret < 700 else 0.0
    if p0 < 0.01:
        return "photon-rich"
    if p0 < 0.37:
        return "low-return"
    if p0 < 0.90:
        return "photon-starved"
    return "extreme-photon-starved"


def scan_pressure_wavelength(pressures_mpa, wavelengths_nm, gas_template: GasState, opt_template: OpticalBudget):
    rows = []
    for P in pressures_mpa:
        for lam in wavelengths_nm:
            gas = GasState(
                pressure_mpa=float(P),
                temperature_k=gas_template.temperature_k,
                methane_mole_fraction=gas_template.methane_mole_fraction,
                hydrogen_mole_fraction=gas_template.hydrogen_mole_fraction,
                compressibility_z=gas_template.compressibility_z,
            )
            opt = OpticalBudget(**{**opt_template.__dict__, "wavelength_nm": float(lam)})
            ret = rayleigh_return_photons(gas, opt)
            eta = effective_signal_channel_eta(gas, opt)
            rows.append({
                "pressure_mpa": float(P),
                "wavelength_nm": float(lam),
                **ret,
                **eta,
                "regime": photon_regime(ret["Nret"]),
            })
    return rows
