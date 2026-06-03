# Scientific hardening changes in v6.1

This patch was made to reduce two risks: an unexplained multi-mode-to-single-mode transduction constant, and pseudo-precision from using a phenomenological diffusion envelope as a hard admissibility boundary.

## 1. Explicit spatial and temporal mode matching

`qdboundary_enhanced/transduction.py` now separates these quantities:

- `Nret`: macroscopic Rayleigh return photon count.
- `eta_total_source_to_detector_mode`: full source-to-detector single-mode estimate.
- `eta_conditional_after_collection`: conditional post-collection signal-mode estimate.
- `spatial_mode_overlap_computed`: normalized Gaussian spatial mode overlap.
- `temporal_mode_overlap`: Gaussian time-gate overlap.
- `single_mode_overlap_total`: product of spatial and temporal mode overlap.
- `mode_overlap_source`: either `computed_gaussian_spatial_overlap` or `explicit_spatial_overlap_override`.

The spatial overlap is computed from Gaussian waist mismatch, transverse offset, angular mismatch, and a turbulence/multiple-scattering mode-purity factor.  This is still a diagnostic coupling model, not a CFD/turbulence proof, but it removes the unexplained scalar coupling from the default transduction bridge.

## 2. Idler timing registration

`qdboundary_enhanced/multilayer.py` now distinguishes:

- `eta_i_nominal`: user-specified idler optical efficiency.
- `eta_i_time_gate_efficiency`: signal--idler temporal registration efficiency.
- `eta_i_effective`: product used in the Gaussian diagnostic point.

This catches cases where an idler path has acceptable optical throughput but fails the time-mode alignment needed for a signal--idler protocol.

## 3. Heuristic diffusion envelope is no longer a default hard veto

The phenomenological envelope

```text
Geff = GQ exp(-Gamma)
```

is now reported as a diagnostic field `geff_heuristic_equal_signal`.  The default classification policy is:

- receiver inadmissibility: hard veto;
- explicit idler preservation/timing failure: hard veto;
- analytic pure-loss no-advantage condition: hard stop;
- `Geff` failure or near-boundary behavior: guarded/heuristic-guarded, not a first-principles hard veto;
- Gaussianized diffusion-QFI failure: Gaussian diagnostic guard, not exact non-Gaussian proof.

A hard-veto interpretation of `Geff` is available only by explicitly setting `ClassificationThresholds(heuristic_policy="hard_veto")` for sensitivity testing.

## Verification

The patched package was checked with:

```bash
PYTHONPATH=. python3 -m pytest tests -q
PYTHONPATH=. python3 scripts/enhanced_self_test.py
PYTHONPATH=. python3 scripts/08_enhanced_analysis.py --quick --outdir outputs_enhanced_v2
```

All tests passed in the patched workspace.
