import importlib.util
from pathlib import Path

import numpy as np


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "10_exact_dephasing_qfi_cutoff_sweep.py"
spec = importlib.util.spec_from_file_location("exact_dephasing_qfi_cutoff_sweep", SCRIPT)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


def test_truncated_tmsv_tail_and_mean_are_reportable():
    ns_cut = mod.truncated_tmsv_mean_ns(1.5, 12)
    assert 0 < ns_cut < 1.5
    assert mod.tmsv_tail_probability(1.5, 12) > 0


def test_smoke_rows_include_required_integrity_fields():
    params = {
        "Ns": [0.5],
        "cutoffs": [8],
        "eta_s": [0.7],
        "Gamma": [0.5],
        "eta_i": [1.0],
    }
    rows = mod.compute_rows(params)
    assert len(rows) == 1
    row = rows[0]
    for key in [
        "NS_target",
        "cutoff",
        "NS_cut",
        "tail_probability",
        "F_Q_exact_TMSV",
        "G_exact",
        "G_eff_envelope",
        "G_exact_minus_G_eff",
        "G_exact_over_G_eff",
        "G_Q_pure_loss_fock_consistent",
        "G_eff_fock_consistent_envelope",
        "pure_loss_formula_audit_note",
    ]:
        assert key in row
    assert np.isfinite(row["G_exact"])


def test_fock_consistent_formula_matches_current_package_formula_at_loss():
    from qdboundary.formulas import gq_pure_loss
    assert mod.fock_consistent_pure_loss_ratio(0.5, 1.5) == gq_pure_loss(0.5, 1.5)
