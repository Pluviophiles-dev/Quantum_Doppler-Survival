# Quantum_Doppler-Survival

The Python package is intended to reproduce the numerical boundary maps, detector-admissibility scans, idler-loss sensitivity maps, diagnostic QZZB guardrails, scenario verdict tables, and the multilayer survival map used in the manuscript.
DOI: https://doi.org/10.5281/zenodo.20477832
## Repository structure

```text
configs/                 default numerical configuration
qdboundary/              reusable Python modules
scripts/                 figure/table generation scripts, including 00--07
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
```

The finite-dimensional QZZB and idler-loss scans are heavier diagnostic calculations. The complete workflow is:

```bash
PYTHONPATH=. python scripts/run_all.py --config configs/default.json
```

For a faster run that skips the heaviest QZZB-certified grid:

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
