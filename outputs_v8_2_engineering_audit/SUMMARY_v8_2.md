# v8.2 engineering audit summary

{
  "version": "v8.2 engineering audit",
  "outputs": {
    "mode_bridge": 675,
    "idler_delay": 72,
    "detector_gate_classical": 80,
    "gamma_structure_bridge": 108,
    "eos_sensitivity": 21,
    "qzzb_cutoff_convergence": 12
  },
  "major_changes": [
    "explicit multimode-to-single-mode mode-coupling penalty chi_mode",
    "idler transmission and phase-stability requirements as functions of delay length/time",
    "gate-based dark/background probabilities plus classical receiver FI baseline",
    "Gamma linked to a random-medium phase-structure-function proxy",
    "PR/EOS density handled as multiplicative sensitivity, not calibrated gas-mixture prediction",
    "finite-Fock QZZB cutoff convergence made visible"
  ],
  "claim_scope": "engineering-audit outputs; no end-to-end quantum advantage is certified"
}