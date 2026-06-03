# Unified Python script inventory (v8.1, full non-compact package)

This package keeps the original refactored Python project structure and does **not** delete the older scripts.  The goal is only to place version-related scripts into one runnable package so they can share the same `qdboundary/` and `qdboundary_enhanced/` modules.

## Main runnable scripts

Run from the repository root with `PYTHONPATH=.`.

```bash
PYTHONPATH=. python scripts/00_conceptual_schematic.py --config configs/default.json
PYTHONPATH=. python scripts/01_phase_diffusion_envelope.py --config configs/default.json
PYTHONPATH=. python scripts/02_idler_loss_sensitivity.py --config configs/default.json
PYTHONPATH=. python scripts/03_qzzb_diagnostic_phase_diagram.py --config configs/default.json
PYTHONPATH=. python scripts/04_scenario_mapping_table.py --config configs/default.json
PYTHONPATH=. python scripts/05_dark_count_admissibility.py --config configs/default.json
PYTHONPATH=. python scripts/06_macro_uncertainty_demo.py --config configs/default.json
PYTHONPATH=. python scripts/07_multilayer_survival_map.py --outdir figures
PYTHONPATH=. python scripts/08_enhanced_analysis.py --outdir outputs_enhanced --quick
PYTHONPATH=. python scripts/09_integrity_audit.py --outdir outputs_integrity
PYTHONPATH=. python scripts/09_audit_pure_loss_tmsv_qfi_formula.py --outdir outputs_formula_audit
PYTHONPATH=. python scripts/10_exact_dephasing_qfi_cutoff_sweep.py --preset smoke --outdir outputs_exact_dephasing_qfi_smoke
PYTHONPATH=. python scripts/11_v8_1_dense_low_ns_audit.py --outdir outputs_v8_1_audit_run
```

## One-command workflow

```bash
PYTHONPATH=. python scripts/run_all.py --config configs/default.json --skip-qzzb --skip-macro-toy --include-v81-audit
```

The `--skip-macro-toy` option exists because the macro uncertainty plot is a single-seed illustrative toy and should not be used as a claim-bearing figure unless replaced by a Monte Carlo uncertainty band.

## Legacy scripts

The original legacy scripts remain in `legacy/`.  Copies are also placed under `scripts/legacy/` so all Python scripts live under the same script tree without changing their contents.

## What changed compared with the compact v8.1 package

The compact v8.1 file has been inserted as `scripts/11_v8_1_dense_low_ns_audit.py`.  It was **not** used to replace the existing package modules or scripts.  All original v8 modules, tests, enhanced diagnostics, exact-QFI scripts, legacy scripts, figures, and generated data remain present.

## v8.2 engineering-audit add-on

`scripts/12_v8_2_engineering_audit.py` adds the conservative engineering checks requested after v8.1:

- `mode_bridge_multimode_to_singlemode.csv`: multimode Rayleigh return to single-mode occupancy penalty with explicit `eta_spatial`, `eta_spectral`, `eta_temporal`, polarization and overlap factors.
- `idler_delay_line_engineering_scan.csv`: idler transmission and phase-stability requirements as functions of fiber delay length or memory time.
- `detector_gate_classical_baseline.csv`: detector dark/background rates converted to per-gate probabilities plus a classical sinusoidal Poisson receiver FI baseline.
- `gamma_phase_structure_bridge.csv`: random-medium phase-variance bridge for the diffusion coordinate `Gamma`.
- `eos_density_sensitivity.csv`: EOS/PR density-proxy sensitivity as a multiplicative photon-budget uncertainty, not a calibrated gas-mixture model.
- `qzzb_cutoff_convergence_audit_v8_2.csv`: finite-Fock QZZB cutoff-convergence table.

Run directly:

```bash
PYTHONPATH=. python scripts/12_v8_2_engineering_audit.py --outdir outputs_v8_2_engineering_audit
```

Or through the unified runner:

```bash
PYTHONPATH=. python scripts/run_all.py --skip-qzzb --skip-macro-toy --include-v82-audit
```

The v8.2 audit does not delete or replace any earlier scripts. It adds engineering-facing checks around the existing theory and diagnostic scripts.
