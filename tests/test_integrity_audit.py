from qdboundary_enhanced.audit import claim_check, scan_claim_text, scope_register
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


def test_claim_level_guardrails():
    assert claim_check("conditional_local_boundary").status == "supported_with_assumptions"
    assert claim_check("instrument_validation").status == "not_supported"
    hits = scan_claim_text("This validated end-to-end turbulence-resolved instrument guarantees advantage.")
    matched = {h["matched_text"].lower() for h in hits}
    assert "validated" in matched
    assert any("end" in h["matched_text"].lower() for h in hits)
    assert any(item.component == "Mode transduction bridge" for item in scope_register())


def test_transduction_outputs_scope_metadata():
    ret = rayleigh_return_photons(GasState(), OpticalBudget())
    eta = effective_signal_channel_eta(GasState(), OpticalBudget())
    assert ret["claim_level_max"] == "engineering_screen"
    assert eta["claim_level_max"] == "conditional_local_boundary"
    assert "multiple-scattering" in eta["not_implemented"]
    assert 0.0 <= eta["eta_conditional_after_collection"] <= 1.0


def test_stress_screens_are_conservative_and_labelled():
    gamma = gamma_from_refractive_stress(RefractiveIndexStress(sigma_n=1e-7))
    assert gamma >= 0
    ms = multiple_scattering_verdict(
        MultipleScatteringStress(number_density_m3=1e25, cross_section_m2=1e-30, path_length_m=0.01)
    )
    assert ms["multiple_scattering_screen"] == "pass_single_scattering_screen"
    purity = mode_purity_from_stress(0.8, float(ms["optical_depth"]), aperture_scintillation_index=0.1)
    assert 0.0 <= purity <= 0.8
    inst = instrument_readiness_verdict(InstrumentStress(calibration_relative_uncertainty=0.05))
    assert inst["instrument_screen"] == "instrument_guarded"
