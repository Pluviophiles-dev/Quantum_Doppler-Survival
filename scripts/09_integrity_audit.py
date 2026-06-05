#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Generate model-scope, claim-level, and stress-screen audit outputs."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd

from qdboundary_enhanced.audit import (
    scan_claim_text,
    write_integrity_audit,
)
from qdboundary_enhanced.stressors import (
    InstrumentStress,
    MultipleScatteringStress,
    RefractiveIndexStress,
    gamma_from_refractive_stress,
    instrument_readiness_verdict,
    mode_purity_from_stress,
    multiple_scattering_verdict,
)
from qdboundary_enhanced.transduction import (
    GasState,
    OpticalBudget,
    effective_signal_channel_eta,
    rayleigh_return_photons,
)


def build_stress_table() -> pd.DataFrame:
    rows = []
    for wavelength_nm in [532.0, 1064.0, 1550.0]:
        gas = GasState(pressure_mpa=30.0, compressibility_z=0.86)
        opt = OpticalBudget(wavelength_nm=wavelength_nm)
        ret = rayleigh_return_photons(gas, opt)
        eta = effective_signal_channel_eta(gas, opt)
        ms = MultipleScatteringStress(
            number_density_m3=ret["number_density_m3"],
            cross_section_m2=ret["sigma_rayleigh_m2"],
            path_length_m=opt.probe_length_m,
        )
        refr = RefractiveIndexStress(
            wavelength_nm=wavelength_nm,
            path_length_m=opt.probe_length_m,
            correlation_length_m=1.0e-3,
            sigma_n=1.0e-7,
        )
        ins = InstrumentStress(
            calibration_relative_uncertainty=0.05,
            timing_jitter_to_gate_ratio=opt.detector_jitter_rms_s / opt.receiver_gate_rms_s,
            background_to_signal_ratio=1.0 / max(ret["Nret"], 1e-300),
        )
        ms_row = multiple_scattering_verdict(ms)
        ins_row = instrument_readiness_verdict(ins)
        rows.append({
            "pressure_mpa": gas.pressure_mpa,
            "wavelength_nm": wavelength_nm,
            "Nret": ret["Nret"],
            "eta_conditional_after_collection": eta["eta_conditional_after_collection"],
            "gamma_refractive_proxy": gamma_from_refractive_stress(refr),
            "optical_depth": ms_row["optical_depth"],
            "multiple_scattering_screen": ms_row["multiple_scattering_screen"],
            "stress_mode_purity": mode_purity_from_stress(
                turbulence_mode_purity=opt.turbulence_mode_purity,
                multiple_scattering_tau=float(ms_row["optical_depth"]),
            ),
            "instrument_screen": ins_row["instrument_screen"],
            "instrument_failed_terms": ins_row["failed_terms"],
            "scope_note": "stress screens only; not full turbulence, multiple-scattering, or instrument validation",
        })
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", default="outputs_integrity")
    parser.add_argument("--scan-text", default="")
    args = parser.parse_args()

    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)
    write_integrity_audit(out)

    build_stress_table().to_csv(out / "propagation_instrument_stress_screens.csv", index=False)

    if args.scan_text:
        hits = scan_claim_text(args.scan_text)
        pd.DataFrame(hits).to_csv(out / "claim_text_risk_hits.csv", index=False)

    print(f"Integrity audit outputs written to {out.resolve()}")


if __name__ == "__main__":
    main()
