from __future__ import annotations

import argparse
import os
import runpy
import sys
from pathlib import Path


def run_script(path: Path, argv: list[str]) -> None:
    old_argv = sys.argv[:]
    sys.argv = [str(path)] + argv
    try:
        print('Running:', ' '.join(sys.argv), flush=True)
        runpy.run_path(str(path), run_name='__main__')
    finally:
        sys.argv = old_argv


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/default.json')
    parser.add_argument('--skip-qzzb', action='store_true', help='Skip the heaviest QZZB scan.')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    os.chdir(root)
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
    os.environ.setdefault('OMP_NUM_THREADS', '1')
    os.environ.setdefault('MKL_NUM_THREADS', '1')
    os.environ.setdefault('NUMEXPR_NUM_THREADS', '1')
    os.environ.setdefault('MPLBACKEND', 'Agg')

    jobs = [
        ('00_conceptual_schematic.py', ['--config', args.config]),
        ('05_dark_count_admissibility.py', ['--config', args.config]),
        ('06_macro_uncertainty_demo.py', ['--config', args.config]),
        ('01_phase_diffusion_envelope.py', ['--config', args.config]),
        ('02_idler_loss_sensitivity.py', ['--config', args.config]),
        ('04_scenario_mapping_table.py', ['--config', args.config]),
        ('07_multilayer_survival_map.py', ['--outdir', 'figures']),
    ]
    if not args.skip_qzzb:
        jobs.append(('03_qzzb_certified_phase_diagram.py', ['--config', args.config]))

    for name, argv in jobs:
        run_script(root / 'scripts' / name, argv)
    print('All requested scripts completed.', flush=True)


if __name__ == '__main__':
    main()
