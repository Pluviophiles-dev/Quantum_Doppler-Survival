import importlib.util
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "09_audit_pure_loss_tmsv_qfi_formula.py"
spec = importlib.util.spec_from_file_location("audit_pure_loss", SCRIPT)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


def test_truncated_mean_is_reportable():
    val = mod.truncated_tmsv_mean_ns(1.5, 12)
    assert 0 < val < 1.5
