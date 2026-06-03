# Quantum_Doppler-Survival

Full LaTeX manuscript and Python reproduction package for the paper on **conditional admissibility boundaries for TMSV-assisted Rayleigh--Doppler phase estimation in photon-starved high-pressure gases**.

The Python package is intended to reproduce the numerical boundary maps, detector-admissibility scans, idler-loss sensitivity maps, diagnostic QZZB guardrails, scenario verdict tables, and the multilayer admissibility map used in the manuscript.

## Repository structure

```text
configs/                 default numerical configuration
qdboundary/              reusable Python modules
qdboundary_enhanced/     optional enhanced diagnostics, scope audit, stress screens
scripts/                 figure/table generation scripts, including 00--09
figures/                 generated figures used by the manuscript
data/                    generated CSV/table outputs
tests/                   unit tests for core formulas
main.tex                 manuscript source
main.pdf                 compiled manuscript check
references.bib           bibliography, when applicable
```

## Installation

```bash
pip install -r requirements.txt
```

## Run tests

```bash
PYTHONPATH=. pytest -q tests
```

## Reproduce numerical outputs

The scripts can be run individually. The recommended quick checks are:

```bash
PYTHONPATH=. python scripts/00_conceptual_schematic.py
PYTHONPATH=. python scripts/01_phase_diffusion_envelope.py --config configs/default.json
PYTHONPATH=. python scripts/04_scenario_mapping_table.py --config configs/default.json
PYTHONPATH=. python scripts/05_dark_count_admissibility.py --config configs/default.json
PYTHONPATH=. python scripts/06_macro_uncertainty_demo.py --config configs/default.json
PYTHONPATH=. python scripts/07_multilayer_survival_map.py --outdir figures
PYTHONPATH=. python scripts/08_enhanced_analysis.py --outdir outputs_enhanced --quick
PYTHONPATH=. python scripts/09_integrity_audit.py --outdir outputs_integrity
```

The finite-dimensional QZZB and idler-loss scans are heavier diagnostic calculations. The complete workflow is:

```bash
PYTHONPATH=. python scripts/run_all.py --config configs/default.json
```

For a faster run that skips the heaviest QZZB diagnostic grid:

```bash
PYTHONPATH=. python scripts/run_all.py --config configs/default.json --skip-qzzb
```

## Compile manuscript

```bash
pdflatex main.tex
pdflatex main.tex
```

## Interpretation note

This repository supports a boundary-audit manuscript. The finite-dimensional QZZB module is used to falsify unsafe local-QFI extrapolations, not to certify full high-photon-number optimality. The dark/background-count module is a detector-level admissibility screen, not an additional quantum-channel decoherence model. The macro uncertainty-transfer example is illustrative and not an industrial EnKF or pipeline-flow validation.


## Script categories

Quantitative scripts generate photon-budget tables, QFI maps, detector screens, QZZB diagnostic maps, scenario tables, and data files. The conceptual schematic script is visualization-only and is not driven by numerical simulation data.


## Optional enhanced diagnostics

The `qdboundary_enhanced/` modules and `scripts/08_enhanced_analysis.py` provide optional cross-checks based on covariance-matrix Gaussian fidelity, a parameterized Rayleigh transduction bridge, and a simple Poisson detector Fisher-information benchmark. These outputs are supplementary diagnostics and are not used as end-to-end instrument validation.

## v6.1 scientific-hardening notes

This revision addresses two common review objections in the enhanced diagnostic layer.

1. **No hidden single-mode magic number.**  `qdboundary_enhanced/transduction.py` now computes an explicit Gaussian spatial-mode overlap and a temporal gate overlap.  The bridge reports `spatial_mode_overlap_computed`, `spatial_mode_overlap_used`, `temporal_mode_overlap`, `single_mode_overlap_total`, and `mode_overlap_source`.  A manual `mode_overlap` may still be supplied for legacy sensitivity studies, but the CSV output labels it as `explicit_spatial_overlap_override` rather than presenting it as a derived optical coupling.

2. **Idler timing is not assumed perfect.**  `qdboundary_enhanced/multilayer.py` now multiplies the nominal idler efficiency by an explicit signal--idler time-registration factor, returned as `eta_i_time_gate_efficiency` and `eta_i_effective`.

3. **Heuristics are guarded, not silently promoted to hard first-principles boundaries.**  The phenomenological `Geff = GQ exp(-Gamma)` screen is reported as `geff_heuristic_equal_signal` with a `geff_status`.  By default, failure of this heuristic moves a point into `heuristic-guarded`/`guarded`; only analytic pure-loss failure, detector inadmissibility, or explicit idler failure is treated as a hard veto.  A user can select `heuristic_policy="hard_veto"` only as an explicit sensitivity option.

The package remains a conditional boundary-audit tool.  It does not claim an end-to-end Rayleigh-scattering quantum advantage or a turbulence-resolved single-mode transduction proof.

## v7 integrity-audited diagnostics

This refactor adds a machine-readable scope registry and conservative stress
screens:

- `qdboundary_enhanced/audit.py` exports model-scope records, claim-level checks,
  and strong-claim text scanning.
- `qdboundary_enhanced/stressors.py` adds refractive-index phase-noise,
  optical-depth multiple-scattering, mode-purity, and instrument-readiness
  screens. These are labelled screens, not turbulence/CFD/radiative-transfer or
  instrument-validation models.
- `scripts/09_integrity_audit.py` writes `model_scope_register.*`,
  `claim_level_checks.*`, and `propagation_instrument_stress_screens.csv`.
- `scripts/08_enhanced_analysis.py` now includes the same scope and stress
  outputs beside the enhanced numerical diagnostics.

The strongest supported claim remains a conditional local-channel boundary with
diagnostic guardrails. The code now records why instrument-level validation,
turbulence-resolved transduction, and multiple-scattering completion are not
supported by the present implementation.

## v8 exact-QFI update

This version integrates the exact finite-Fock non-Gaussian phase-diffusion QFI audit and corrects the pure-loss TMSV benchmark convention. The older factor-2 denominator remains only in legacy/audit functions for reproducibility. The recommended validation command is:

```bash
pytest -q tests -p no:cacheprovider
```

For formula auditing:

```bash
python scripts/09_audit_pure_loss_tmsv_qfi_formula.py --outdir outputs_formula_audit
python scripts/10_exact_dephasing_qfi_cutoff_sweep.py --preset smoke --outdir outputs_exact_dephasing_qfi_smoke
```

## v8.1 unified full script package

This full package keeps the v8 refactored code and adds the v8.1 audit as a normal script:

```bash
PYTHONPATH=. python scripts/11_v8_1_dense_low_ns_audit.py --outdir outputs_v8_1_audit_run
```

It also places copies of the old legacy scripts under `scripts/legacy/` while keeping the original `legacy/` directory.  Nothing was intentionally removed to make a compact package.  See `SCRIPT_INVENTORY_UNIFIED_v8_1.md` for a runnable inventory.

### v8.2 engineering-audit add-on

The full unified package now includes `scripts/12_v8_2_engineering_audit.py`. It keeps all prior v7/v8/v8.1 scripts and adds engineering-facing audits for multimode-to-single-mode coupling loss, idler delay-line efficiency/phase drift, gate-based detector probabilities, a classical receiver FI baseline, random-medium Gamma scaling, EOS sensitivity, and QZZB cutoff convergence.

Quick run:

```bash
PYTHONPATH=. python scripts/12_v8_2_engineering_audit.py --outdir outputs_v8_2_engineering_audit
```
