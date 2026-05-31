# Quantum_Doppler-Survival

Code repository for the paper on **quantum-enhanced Doppler velocimetry survival bounds** in photon-starved high-pressure gas measurements.

This repository contains the Python reproduction package used to generate the numerical boundary maps, QZZB-certified validity diagrams, idler-loss sensitivity maps, detector-level admissibility scans, and illustrative uncertainty-transfer examples.

The manuscript source files are not included in this public repository.

## Repository structure

```text
configs/
  default.json

qdboundary/
  __init__.py
  classification.py
  config.py
  covariance.py
  fock.py
  formulas.py
  plotting.py
  qzzb.py
  rayleigh.py

scripts/
  00_conceptual_schematic.py
  01_phase_diffusion_envelope.py
  02_idler_loss_sensitivity.py
  03_qzzb_certified_phase_diagram.py
  04_scenario_mapping_table.py
  05_dark_count_admissibility.py
  06_macro_uncertainty_demo.py
  run_all.py

tests/
  test_core_formulas.py
