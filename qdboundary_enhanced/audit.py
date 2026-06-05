#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Model-scope and claim-integrity audit helpers.

The enhanced package is a conditional boundary-audit codebase.  These helpers
make that scope machine-readable so generated CSV/JSON outputs do not silently
promote diagnostic screens into turbulence-resolved, multiple-scattering, or
instrument-validated claims.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal
import csv
import json
import re

EvidenceLevel = Literal[
    "standard_formula",
    "numerical_diagnostic",
    "phenomenological_heuristic",
    "illustrative_interface",
    "not_implemented",
]

ClaimLevel = Literal[
    "conditional_local_boundary",
    "diagnostic_guardrail",
    "engineering_screen",
    "instrument_validation",
]


@dataclass(frozen=True)
class ScopeItem:
    """A single model-scope statement suitable for CSV/JSON export."""

    component: str
    layer: str
    evidence_level: EvidenceLevel
    implemented_as: str
    not_implemented: str
    safe_claim: str
    unsafe_claim: str
    required_for_stronger_claim: str


@dataclass(frozen=True)
class ClaimCheck:
    claim_level: ClaimLevel
    status: str
    reason: str
    safe_wording: str


STRONG_CLAIM_PATTERNS: tuple[tuple[str, str], ...] = (
    (r"\bend[- ]to[- ]end\b", "End-to-end wording requires a calibrated source-to-detector and estimator chain."),
    (r"\binstrument[- ]level\b", "Instrument-level wording requires calibration, uncertainty budget, and hardware validation."),
    (r"\bvalidated\b", "Validation wording requires external experimental or benchmark evidence."),
    (r"\bcertified\b", "Certification wording requires an accredited metrology or standardization basis."),
    (r"\bturbulence[- ]resolved\b", "Turbulence-resolved wording requires a turbulence propagation model or data."),
    (r"\bmultiple[- ]scattering\b", "Multiple-scattering wording requires radiative-transfer or wave-transport modeling."),
    (r"\buniversal\b", "Universal wording is too broad for a conditional parameter screen."),
    (r"\bguarantee[sd]?\b", "Guarantee wording is too strong for local bounds and diagnostic guardrails."),
    (r"\bproves?\b", "Proof wording must be reserved for stated mathematical propositions only."),
    (r"\bbreakthrough\b", "Breakthrough wording is promotional and unsupported by this code."),
)


def scope_register() -> list[ScopeItem]:
    """Return the package's explicit model-scope registry."""
    return [
        ScopeItem(
            component="Rayleigh photon budget",
            layer="radiometric return",
            evidence_level="standard_formula",
            implemented_as="single-scattering homogeneous-volume photon-count proxy",
            not_implemented="radiative-transfer, window fouling, beam clipping, calibrated gas-mixture cross sections",
            safe_claim="computes reproducible photon-budget anchors under stated proxy assumptions",
            unsafe_claim="validates a field Rayleigh velocimeter photon return",
            required_for_stronger_claim="gas-specific cross sections, calibrated optical train, uncertainty budget, validation data",
        ),
        ScopeItem(
            component="TMSV pure-loss QFI",
            layer="microscopic quantum channel",
            evidence_level="standard_formula",
            implemented_as="one-sided pure-loss local QFI benchmark with equal signal-energy comparison",
            not_implemented="optimal receiver design, finite-prior attainability, source engineering",
            safe_claim="tests whether a conditional local channel can exceed the coherent SQL benchmark",
            unsafe_claim="demonstrates an attainable Rayleigh-scattering quantum advantage",
            required_for_stronger_claim="receiver POVM model, finite-prior bound at target photon number, experiment or independent simulation",
        ),
        ScopeItem(
            component="Phase diffusion envelope",
            layer="channel stress screen",
            evidence_level="phenomenological_heuristic",
            implemented_as="Geff = GQ exp(-a Gamma) with labelled heuristic policy",
            not_implemented="exact non-Gaussian dephasing QFI or turbulence-derived phase statistics",
            safe_claim="flags parameter regions where local pure-loss extrapolation should be guarded",
            unsafe_claim="solves turbulence-induced quantum decoherence",
            required_for_stronger_claim="derived stochastic refractive-index model, exact/non-Gaussian QFI or validated Monte Carlo",
        ),
        ScopeItem(
            component="Finite-dimensional QZZB",
            layer="global distinguishability diagnostic",
            evidence_level="numerical_diagnostic",
            implemented_as="truncated-Fock low-photon-number QZZB guard with cutoff reports",
            not_implemented="high-NS global optimality certificate",
            safe_claim="falsifies unsafe local-QFI extrapolation in diagnostic regimes",
            unsafe_claim="certifies high-photon-number global performance",
            required_for_stronger_claim="cutoff-converged high-NS calculation or analytic Gaussian/non-Gaussian global bound",
        ),
        ScopeItem(
            component="Gaussian-fidelity diagnostic",
            layer="high-NS covariance stress check",
            evidence_level="numerical_diagnostic",
            implemented_as="zero-mean Gaussian covariance fidelity with cross-covariance damping surrogate",
            not_implemented="exact number-dephasing channel in covariance form",
            safe_claim="provides a labelled covariance-level stress test",
            unsafe_claim="proves exact phase-diffusion-limited quantum performance",
            required_for_stronger_claim="exact channel map or benchmark against independent non-Gaussian calculations",
        ),
        ScopeItem(
            component="Mode transduction bridge",
            layer="single-mode coupling",
            evidence_level="phenomenological_heuristic",
            implemented_as="Gaussian spatial overlap, temporal gate overlap, and explicit purity factors",
            not_implemented="CFD, turbulence propagation, speckle statistics, multiple-scattering wave transport",
            safe_claim="makes single-mode assumptions auditable and stress-testable",
            unsafe_claim="derives a complete source-to-detector TMSV Rayleigh channel",
            required_for_stronger_claim="optical propagation model, measured mode overlap, turbulence/multiple-scattering validation",
        ),
        ScopeItem(
            component="Detector FI benchmark",
            layer="receiver likelihood",
            evidence_level="engineering_screen",
            implemented_as="simple Poisson count model with dark/background counts and visibility",
            not_implemented="detector dead time, afterpulsing, saturation, full receiver POVM",
            safe_claim="screens whether count-level receiver assumptions are admissible",
            unsafe_claim="models an optimal instrument receiver",
            required_for_stronger_claim="detector characterization, calibration model, likelihood validation",
        ),
        ScopeItem(
            component="Macro uncertainty transfer",
            layer="future instrument interface",
            evidence_level="illustrative_interface",
            implemented_as="simple 1D uncertainty propagation demo",
            not_implemented="industrial EnKF, flow profile validation, traceability chain",
            safe_claim="shows how a local phase variance could enter a later uncertainty budget",
            unsafe_claim="validates high-pressure pipeline flow measurement",
            required_for_stronger_claim="GUM budget, flow facility data, calibrated reconstruction algorithm",
        ),
    ]


def claim_check(claim_level: ClaimLevel) -> ClaimCheck:
    """Return whether the requested claim level is supported by this package."""
    if claim_level == "conditional_local_boundary":
        return ClaimCheck(
            claim_level=claim_level,
            status="supported_with_assumptions",
            reason="The package implements the stated local QFI, photon-budget, detector, idler, and diagnostic guard screens.",
            safe_wording="conditional local-channel admissibility boundary under stated proxy assumptions",
        )
    if claim_level == "diagnostic_guardrail":
        return ClaimCheck(
            claim_level=claim_level,
            status="supported_with_labels",
            reason="QZZB, Gaussian-fidelity, diffusion, and transduction outputs are labelled as diagnostics or heuristics.",
            safe_wording="diagnostic guardrail against unsafe local-QFI extrapolation",
        )
    if claim_level == "engineering_screen":
        return ClaimCheck(
            claim_level=claim_level,
            status="partially_supported",
            reason="The photon and detector screens are useful engineering filters but are not calibrated instrument models.",
            safe_wording="engineering pre-screen, not validation",
        )
    return ClaimCheck(
        claim_level=claim_level,
        status="not_supported",
        reason="The package lacks calibrated hardware, turbulence-resolved propagation, multiple-scattering transport, and full uncertainty validation.",
        safe_wording="future work toward instrument validation",
    )


def scan_claim_text(text: str) -> list[dict[str, str]]:
    """Find phrases that can overstate the model's evidence level."""
    hits: list[dict[str, str]] = []
    for pattern, reason in STRONG_CLAIM_PATTERNS:
        for match in re.finditer(pattern, text, flags=re.IGNORECASE):
            hits.append({
                "matched_text": match.group(0),
                "start": str(match.start()),
                "end": str(match.end()),
                "reason": reason,
            })
    return hits


def write_integrity_audit(outdir: str | Path) -> dict[str, Path]:
    """Write scope and claim-risk tables to an output directory."""
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)

    scope_items = [asdict(item) for item in scope_register()]
    scope_json = out / "model_scope_register.json"
    scope_csv = out / "model_scope_register.csv"
    scope_json.write_text(json.dumps(scope_items, indent=2), encoding="utf-8")
    with scope_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(scope_items[0].keys()))
        writer.writeheader()
        writer.writerows(scope_items)

    checks = [asdict(claim_check(level)) for level in (
        "conditional_local_boundary",
        "diagnostic_guardrail",
        "engineering_screen",
        "instrument_validation",
    )]
    checks_json = out / "claim_level_checks.json"
    checks_csv = out / "claim_level_checks.csv"
    checks_json.write_text(json.dumps(checks, indent=2), encoding="utf-8")
    with checks_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(checks[0].keys()))
        writer.writeheader()
        writer.writerows(checks)

    return {
        "scope_json": scope_json,
        "scope_csv": scope_csv,
        "claim_checks_json": checks_json,
        "claim_checks_csv": checks_csv,
    }
