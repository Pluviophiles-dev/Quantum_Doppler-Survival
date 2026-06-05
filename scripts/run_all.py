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
    parser.add_argument('--skip-macro-toy', action='store_true', help='Skip the single-seed macro uncertainty toy figure.')
    parser.add_argument('--include-v81-audit', action='store_true', help='Run the v8.1 dense-gas/mode-bridge/low-NS/idler-phase audit script.')
    parser.add_argument('--include-v82-audit', action='store_true', help='Run the v8.2 engineering audit script (mode coupling, idler delay, detector gate, QZZB convergence).')
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
        ('01_phase_diffusion_envelope.py', ['--config', args.config]),
        ('02_idler_loss_sensitivity.py', ['--config', args.config]),
        ('04_scenario_mapping_table.py', ['--config', args.config]),
        ('08_enhanced_analysis.py', ['--outdir', 'outputs_enhanced', '--quick']),
        ('09_integrity_audit.py', ['--outdir', 'outputs_integrity']),
        ('07_multilayer_survival_map.py', ['--outdir', 'figures']),
    ]
    if not args.skip_macro_toy:
        jobs.insert(2, ('06_macro_uncertainty_demo.py', ['--config', args.config]))
    if args.include_v81_audit:
        jobs.append(('11_v8_1_dense_low_ns_audit.py', ['--outdir', 'outputs_v8_1_audit_run']))
    if args.include_v82_audit:
        jobs.append(('12_v8_2_engineering_audit.py', ['--outdir', 'outputs_v8_2_engineering_audit']))
    if not args.skip_qzzb:
        jobs.append(('03_qzzb_diagnostic_phase_diagram.py', ['--config', args.config]))

    for name, argv in jobs:
        run_script(root / 'scripts' / name, argv)
    print('All requested scripts completed.', flush=True)


if __name__ == '__main__':
    main()
