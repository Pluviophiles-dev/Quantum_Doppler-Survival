# Scientific hardening changes in v7

This refactor is intentionally conservative. It does not pretend to complete
turbulence, multiple-scattering, or instrument-level quantum velocimetry
modeling. Instead, it makes those missing layers explicit in code outputs.

## Added

- `qdboundary_enhanced/audit.py`: model-scope registry, claim-level checks, and
  strong-claim text scanning.
- `qdboundary_enhanced/stressors.py`: labelled refractive-index phase-noise,
  optical-depth multiple-scattering, mode-purity, and instrument-readiness
  screens.
- `scripts/09_integrity_audit.py`: writes CSV/JSON files that reviewers can use
  to see which claims are supported, partially supported, or unsupported.

## Changed

- `transduction.py` validates gas and optical inputs and emits explicit
  `model_scope`, `claim_level_max`, and `not_implemented` fields.
- `multilayer.py` now reports the maximum supported claim level and states that
  instrument validation is not supported by this package.
- `scripts/08_enhanced_analysis.py` writes model-scope and stress-screen outputs
  next to the enhanced diagnostics.
- `scripts/run_all.py` includes the integrity-audit script.

## Interpretation

The package supports conditional local-channel admissibility and diagnostic
guardrails. It does not support claims of end-to-end Rayleigh-scattering quantum
advantage, turbulence-resolved single-mode transduction, radiative
multiple-scattering completion, or calibrated instrument validation.
