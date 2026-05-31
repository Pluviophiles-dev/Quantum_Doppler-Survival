from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def run(cmd: list[str]) -> None:
    print("Running:", " ".join(cmd))
    subprocess.check_call(cmd)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.json")
    parser.add_argument("--skip-qzzb", action="store_true", help="Skip the heaviest QZZB scan.")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    scripts = [
        "00_conceptual_schematic.py",
        "01_phase_diffusion_envelope.py",
        "02_idler_loss_sensitivity.py",
        "04_scenario_mapping_table.py",
        "05_dark_count_admissibility.py",
        "06_macro_uncertainty_demo.py",
    ]
    for s in scripts:
        run([sys.executable, str(root / "scripts" / s), "--config", args.config])
    if not args.skip_qzzb:
        run([sys.executable, str(root / "scripts" / "03_qzzb_certified_phase_diagram.py"), "--config", args.config])


if __name__ == "__main__":
    main()
