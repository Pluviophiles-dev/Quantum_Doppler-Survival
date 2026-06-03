# v8 exact-QFI integration notes

Main changes:

1. Corrected the one-sided pure-loss TMSV benchmark to the finite-Fock SLD-QFI-consistent convention:

   G_Q = (N_S + 1) / [1 + (1 - eta_s) N_S].

2. Retained the older factor-2 denominator only as a legacy audit function/column:

   G_Q_legacy = (N_S + 1) / [1 + 2(1 - eta_s) N_S].

3. Updated the equal-total-energy threshold to

   eta_c = 0.5 * (1 + 1/N_S), for N_S > 1.

4. Added a pure-loss formula audit script:

   scripts/09_audit_pure_loss_tmsv_qfi_formula.py

5. Kept and updated the exact finite-Fock non-Gaussian number-dephasing QFI sweep:

   scripts/10_exact_dephasing_qfi_cutoff_sweep.py

6. Updated tests. Current result in this package:

   pytest -q tests -p no:cacheprovider
   17 passed

Interpretation:

- G_eff = G_Q exp(-Gamma) is now an optimistic candidate envelope, not a rigorous bound.
- The exact finite-Fock dephasing-QFI audit is the stricter veto layer.
- The low-transduction eta_s regime should be interpreted as a rejection/target-requirement result, not as a demonstrated physical operating point.
