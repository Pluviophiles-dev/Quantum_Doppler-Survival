# Exact finite-Fock non-Gaussian phase-diffusion QFI diagnostic

Date: 2026-06-03

This is an objective diagnostic run.  No manuscript claims were changed.

## What was added

New script:

```text
work/quantum_doppler_refactored_v7/scripts/10_exact_dephasing_qfi_cutoff_sweep.py
```

It computes exact finite-Fock signal-phase QFI after:

1. truncated TMSV preparation;
2. independent signal/idler loss;
3. exact signal-mode number dephasing;
4. SLD spectral-formula QFI using the signal number operator.

It outputs:

```text
exact_dephasing_qfi_cutoff_sweep.csv
exact_dephasing_qfi_vs_envelope.csv
summary.json
figures/fig_exact_dephasing_qfi_cutoff.png
figures/fig_exact_dephasing_qfi_vs_geff.png
figures/fig_exact_dephasing_qfi_vs_gamma.png
```

## Tests

`pytest -q tests -p no:cacheprovider`:

```text
14 passed
```

## Runs completed

### Smoke

Directory:

```text
work/quantum_doppler_refactored_v7/outputs_exact_dephasing_qfi_smoke_v2
```

Summary:

```text
points: 108
highest_cutoff_points: 54
fraction_highest_exact_le_package_envelope: 0.7778
fraction_highest_exact_le_fock_consistent_envelope: 1.0000
fraction_highest_converged_by_5pct_rule: 0.6296
max_tail_probability_highest: 0.0317
```

### Coarse diagnostic

Directory:

```text
work/quantum_doppler_refactored_v7/outputs_exact_dephasing_qfi_coarse_v2
```

Grid:

```text
N_S = 0.5, 1.5, 3, 5
cutoff = 8, 12, 16, 24
eta_s = 0.5, 0.7, 0.9, 1.0
Gamma = 0, 0.5, 1, 2, 3
eta_i = 1.0, 0.9, 0.7
```

Summary:

```text
points: 960
highest_cutoff_points: 240
fraction_highest_exact_le_package_envelope: 0.9125
fraction_highest_exact_le_fock_consistent_envelope: 1.0000
fraction_highest_converged_by_5pct_rule: 0.9000
max_tail_probability_highest: 0.01258
```

By `N_S`:

| N_S | exact advantage fraction | package-envelope advantage fraction | fock-consistent-envelope advantage fraction | exact <= package envelope | exact <= fock-consistent envelope | convergence fraction | max tail |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.5 | 0.1333 | 0.1500 | 0.2000 | 0.9333 | 1.0000 | 1.0000 | 3.54e-12 |
| 1.5 | 0.1833 | 0.2500 | 0.3500 | 0.9167 | 1.0000 | 1.0000 | 4.74e-06 |
| 3.0 | 0.2000 | 0.3000 | 0.4500 | 0.9000 | 1.0000 | 0.8000 | 1.00e-03 |
| 5.0 | 0.2000 | 0.3500 | 0.5000 | 0.9000 | 1.0000 | 0.8000 | 1.26e-02 |

Most important coarse observation:

```text
There were zero G_exact > 1 cases at Gamma > 0.
```

All exact-advantage cases in the coarse highest-cutoff table occurred at:

```text
Gamma = 0
```

### N_S = 10 stress test

Directory:

```text
work/quantum_doppler_refactored_v7/outputs_exact_dephasing_qfi_ns10_stress
```

Grid:

```text
N_S = 10
cutoff = 24, 32
eta_s = 0.7, 0.9
Gamma = 0, 0.5, 1, 2, 3
eta_i = 1.0, 0.9, 0.7
```

Summary:

```text
points: 60
highest_cutoff_points: 30
fraction_highest_exact_le_package_envelope: 0.9333
fraction_highest_exact_le_fock_consistent_envelope: 1.0000
fraction_highest_converged_by_5pct_rule: 0.8000
max_tail_probability_highest: 0.04736
```

Important caution:

```text
For N_S=10 and cutoff=32, NS_cut/NS_target = 0.8409.
```

So this is a stress test, not a fully converged high-`N_S` result.

Again, every `Gamma > 0` row had `G_exact << 1`.

Examples at cutoff 32, eta_i=1:

| eta_s | Gamma | G_exact | package G_eff | fock-consistent G_eff | converged? |
|---:|---:|---:|---:|---:|---|
| 0.7 | 0.0 | 2.005 | 1.571 | 2.750 | no |
| 0.7 | 0.5 | 0.0616 | 0.953 | 1.668 | yes |
| 0.7 | 1.0 | 0.0301 | 0.578 | 1.012 | yes |
| 0.9 | 0.0 | 3.532 | 3.667 | 5.500 | no |
| 0.9 | 0.5 | 0.0501 | 2.224 | 3.336 | yes |
| 0.9 | 1.0 | 0.0242 | 1.349 | 2.023 | yes |

## Major finding: pure-loss formula discrepancy

The exact finite-Fock SLD QFI at `Gamma=0` converges to:

```text
G_Q = (N_S + 1) / [1 + (1 - eta_s) N_S]
```

The existing package/manuscript envelope uses:

```text
G_Q = (N_S + 1) / [1 + 2(1 - eta_s) N_S]
```

This is not a small numerical issue.  It changes the pure-loss reference
boundary.  Therefore the new diagnostic script reports both:

```text
G_Q_pure_loss
G_eff_envelope
G_Q_pure_loss_fock_consistent
G_eff_fock_consistent_envelope
```

No result has been overwritten or reinterpreted silently.

## Objective interpretation

1. The exact non-Gaussian number-dephasing QFI is much harsher than the original heuristic envelope once `Gamma > 0`.
2. In the completed coarse and `N_S=10` stress runs, `G_exact > 1` survived only at `Gamma = 0`.
3. The original `G_eff = G_Q exp(-Gamma)` envelope should not be described as a rigorous upper or lower bound.
4. The exact finite-Fock calculation is valuable, but high-`N_S` use must be limited by the reported tail probability and convergence flag.
5. Before adding this to the paper, the pure-loss formula discrepancy must be resolved.  It may require revisiting the analytic TMSV pure-loss QFI derivation and resource-accounting convention.

## Recommendation before manuscript inclusion

Do not insert these results as a polished conclusion yet.

Recommended next step:

1. Audit the analytic pure-loss TMSV QFI formula.
2. Decide whether the manuscript benchmark should use the current package formula or the Fock-consistent formula.
3. If the Fock-consistent formula is correct for the stated model, update the theory section and rerun all dependent figures.
4. If the current package formula corresponds to a different resource or measurement convention, state that convention explicitly and explain why the finite-Fock SLD QFI is not the same benchmark.

Until that is resolved, the new script should be cited only as an internal audit that revealed a formula/convention mismatch.
