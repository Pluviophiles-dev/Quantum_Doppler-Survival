#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Integrated multilayer classification.

Combines:
- Rayleigh photon budget,
- effective transduction bridge with explicit spatial/temporal mode overlap,
- detector SBR and classical FI,
- Gaussian covariance-fidelity high-NS guard,
- idler preservation with time-gate registration,
- phase diffusion.

Important classification convention
-----------------------------------
Hard physical vetoes (receiver inadmissibility, no pure-loss local advantage,
explicit idler loss) are kept separate from heuristic guardrails.  The
phenomenological Geff = GQ exp(-Gamma) envelope and the Gaussianized diffusion
surrogate are not allowed to masquerade as precision hard thresholds; they move
points into labelled guarded/heuristic regions unless a stricter policy is
explicitly requested.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Literal
from .transduction import (
    GasState,
    OpticalBudget,
    rayleigh_return_photons,
    effective_signal_channel_eta,
    photon_regime,
    idler_time_gate_efficiency,
)
from .detector_fi import DetectorModel, max_fi_over_phase
from .gaussian_fidelity import GaussianPoint, summarize_point, pure_loss_ratio_equal_signal
from .audit import claim_check

HeuristicPolicy = Literal["guarded", "hard_veto"]


@dataclass(frozen=True)
class ClassificationThresholds:
    min_sbr: float = 1.0
    min_eta_i: float = 0.7
    min_gaussian_qfi_ratio: float = 1.0
    min_geff_candidate: float = 1.0
    min_geff_interpretable: float = 1.25
    max_qzzb_to_local_variance_ratio: float = 10.0
    use_conditional_eta_after_collection: bool = True
    heuristic_policy: HeuristicPolicy = "guarded"


@dataclass(frozen=True)
class Scenario:
    gas: GasState
    optics: OpticalBudget
    ns: float = 10.0
    eta_i: float = 0.9
    gamma: float = 0.3
    dark_rate_hz: float = 25.0
    background_rate_hz: float = 100.0
    gate_time_s: float = 1e-6
    visibility: float = 0.5
    samples: int = 1

    # Idler/reference timing registration.  These factors address the fact that
    # an idler can be optically efficient but still unusable if it is not in the
    # same temporal mode as the signal-return gate.
    idler_gate_rms_s: float = 5.0e-9
    idler_delay_error_s: float = 0.0
    idler_joint_jitter_rms_s: float = 0.5e-9
    idler_memory_efficiency: float = 1.0


def classify_scenario(s: Scenario, th: ClassificationThresholds = ClassificationThresholds()) -> dict:
    ret = rayleigh_return_photons(s.gas, s.optics)
    eta_bridge = effective_signal_channel_eta(s.gas, s.optics)

    eta_s_model = (
        eta_bridge["eta_conditional_after_collection"]
        if th.use_conditional_eta_after_collection
        else eta_bridge["eta_total_source_to_detector_mode"]
    )
    eta_s_model = max(0.0, min(1.0, float(eta_s_model)))

    idler_gate_eff = idler_time_gate_efficiency(
        signal_gate_rms_s=s.optics.receiver_gate_rms_s,
        idler_gate_rms_s=s.idler_gate_rms_s,
        relative_delay_error_s=s.idler_delay_error_s,
        joint_jitter_rms_s=s.idler_joint_jitter_rms_s,
        idler_memory_efficiency=s.idler_memory_efficiency,
    )
    eta_i_effective = max(0.0, min(1.0, s.eta_i * idler_gate_eff))

    detector = DetectorModel(
        nret=ret["Nret"],
        dark_rate_hz=s.dark_rate_hz,
        background_rate_hz=s.background_rate_hz,
        gate_time_s=s.gate_time_s,
        visibility=s.visibility,
        samples=s.samples,
    )
    det = max_fi_over_phase(detector)

    gpt = GaussianPoint(ns=s.ns, eta_s=eta_s_model, eta_i=eta_i_effective, gamma=s.gamma)
    gsum = summarize_point(gpt)
    qfi_diag = float(gsum["surrogate_gaussian_local_bures_qfi_fd"])
    local_var = 1.0 / qfi_diag if qfi_diag > 0 else float("inf")
    guard_ratio = gsum["diagnostic_gaussian_qzzb_squared_fidelity"] / local_var if local_var > 0 else float("inf")

    pure_loss_ratio = pure_loss_ratio_equal_signal(s.ns, eta_s_model)
    geff_heuristic = pure_loss_ratio * math.exp(-s.gamma) if math.isfinite(pure_loss_ratio) else float("nan")

    hard_veto = False
    heuristic_guard = False
    first_failed = "none"
    verdict = "local-channel-valid"
    classification_basis = "all configured hard screens and diagnostic guardrails passed"

    if det["sbr"] < th.min_sbr or ret["Nret"] < det["noise_counts_per_gate"]:
        verdict, first_failed = "detector-limited", "receiver admissibility"
        hard_veto = True
        classification_basis = "hard receiver screen"
    elif eta_i_effective < th.min_eta_i:
        verdict, first_failed = "idler-limited", "idler preservation/time registration"
        hard_veto = True
        classification_basis = "hard idler preservation and timing screen"
    elif pure_loss_ratio <= 1.0:
        verdict, first_failed = "stop-extrapolation", "pure-loss QFI boundary"
        hard_veto = True
        classification_basis = "analytic pure-loss local benchmark has no TMSV advantage"
    elif geff_heuristic < th.min_geff_candidate:
        if th.heuristic_policy == "hard_veto":
            verdict, first_failed = "stop-extrapolation", "heuristic diffusion envelope"
            hard_veto = True
            classification_basis = "user-selected hard-veto policy for phenomenological Geff"
        else:
            verdict, first_failed = "heuristic-guarded", "heuristic diffusion envelope"
            heuristic_guard = True
            classification_basis = "phenomenological Geff is below candidate threshold; not treated as a first-principles veto"
    elif geff_heuristic < th.min_geff_interpretable:
        verdict, first_failed = "guarded", "near heuristic diffusion boundary"
        heuristic_guard = True
        classification_basis = "near-boundary Geff guard band"
    elif float(gsum["surrogate_gaussian_qfi_ratio_vs_coherent"]) <= th.min_gaussian_qfi_ratio:
        verdict, first_failed = "gaussian-guarded", "Gaussianized surrogate QFI guard"
        heuristic_guard = True
        classification_basis = "Gaussianized diffusion surrogate failed; diagnostic guard only"
    elif guard_ratio > th.max_qzzb_to_local_variance_ratio:
        verdict, first_failed = "guarded", "global distinguishability"
        heuristic_guard = True
        classification_basis = "finite-prior/QZZB diagnostic guard"

    return {
        "pressure_mpa": s.gas.pressure_mpa,
        "wavelength_nm": s.optics.wavelength_nm,
        "Nret": ret["Nret"],
        "zero_count_probability": ret["zero_count_probability"],
        "photon_regime": photon_regime(ret["Nret"]),
        "eta_s_model": eta_s_model,
        "eta_i_nominal": s.eta_i,
        "eta_i_time_gate_efficiency": idler_gate_eff,
        "eta_i_effective": eta_i_effective,
        "gamma": s.gamma,
        "ns": s.ns,
        "sbr": det["sbr"],
        "max_detector_classical_fi": det["max_classical_fi"],
        "detector_phase_crlb": det["crlb_phase_variance"],
        "surrogate_gaussian_qfi_ratio_vs_coherent": gsum["surrogate_gaussian_qfi_ratio_vs_coherent"],
        "diagnostic_gaussian_qzzb_squared_fidelity": gsum["diagnostic_gaussian_qzzb_squared_fidelity"],
        "qzzb_to_local_variance_ratio": guard_ratio,
        "pure_loss_ratio_equal_signal_no_diffusion": pure_loss_ratio,
        "geff_heuristic_equal_signal": geff_heuristic,
        "geff_status": (
            "not_applicable_no_pure_loss_advantage" if pure_loss_ratio <= 1.0 else
            "below_heuristic_candidate" if geff_heuristic < th.min_geff_candidate else
            "near_boundary_guard_band" if geff_heuristic < th.min_geff_interpretable else
            "above_interpretable_guard_band"
        ),
        "hard_veto": hard_veto,
        "heuristic_guard": heuristic_guard,
        "classification_basis": classification_basis,
        "max_supported_claim_level": claim_check("conditional_local_boundary").safe_wording,
        "instrument_validation_status": claim_check("instrument_validation").status,
        "instrument_validation_reason": claim_check("instrument_validation").reason,
        "verdict": verdict,
        "first_failed_layer": first_failed,
        **eta_bridge,
    }
