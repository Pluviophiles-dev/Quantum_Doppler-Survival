from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from qdboundary.classification import classify_boundary_point
from qdboundary.config import load_config
from qdboundary.formulas import geff, gamma_max, gq_pure_loss, local_phase_variance_from_qfi, tmsv_pure_loss_qfi, wrapping_probability_gaussian
from qdboundary.rayleigh import photon_regime, return_photons, zero_count_probability


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.json")
    args = parser.parse_args()
    cfg = load_config(args.config)
    r = cfg["rayleigh"]
    Ns = float(cfg["model"]["Ns_main"])
    M = int(cfg["model"]["M"])
    rows = []
    for P in r["pressure_MPa_values"]:
        for lam in r["wavelength_nm_values"]:
            Nret = return_photons(
                pressure_MPa=P,
                temperature_K=r["temperature_K"],
                lambda_nm=lam,
                pulse_energy_J=r["pulse_energy_J"],
                probe_length_m=r["probe_length_m"],
                collection_fraction=r["collection_fraction"],
                eta_sys=r["eta_sys"],
                n_ref=r["n_ref"],
                king_factor=r["king_factor"],
            )
            P0 = zero_count_probability(Nret)
            eta_s = float(r["eta_s_assumed"])
            eta_i = float(r["eta_i_assumed"])
            Gamma = float(r["gamma_assumed"])
            # Do not equate Nret and Ns. This is a scenario-level assumed microscopic probe setting.
            g_eff = geff(eta_s, Ns, Gamma, a=1.0)
            Fq = tmsv_pure_loss_qfi(eta_s, Ns) * float(pd.Series([eta_i]).clip(0, 1).iloc[0])  # conservative idler bookkeeping proxy for table only
            local_var = local_phase_variance_from_qfi(Fq, M)
            pwrap = wrapping_probability_gaussian(local_var)
            verdict = classify_boundary_point(
                g_eff, guard_ratio=1.0, wrap_probability=pwrap,
                guard_ratio_threshold=cfg["classification"]["guard_ratio_threshold"],
                wrap_probability_threshold=cfg["classification"]["wrap_probability_threshold"],
                local_ratio_tolerance=cfg["classification"]["local_ratio_tolerance"],
            )
            rows.append({
                "pressure_MPa": P,
                "lambda_nm": lam,
                "Nret_rayleigh_budget": Nret,
                "P0_zero_count": P0,
                "photon_regime": photon_regime(P0),
                "assumed_mode_matching": r["assumed_mode_matching"],
                "assumed_micro_Ns_not_equal_Nret": Ns,
                "eta_s_assumed": eta_s,
                "eta_i_assumed": eta_i,
                "Gamma_assumed": Gamma,
                "GQ_pure_loss": gq_pure_loss(eta_s, Ns),
                "Gamma_max_baseline": gamma_max(pd.Series([eta_s]).to_numpy(), Ns, a=1.0)[0],
                "Geff_baseline": g_eff,
                "local_wrap_probability_proxy": pwrap,
                "scenario_verdict_local_proxy": verdict
            })
    df = pd.DataFrame(rows)
    out_csv = Path(cfg["paths"]["data"]) / "scenario_mapping_table.csv"
    out_tex = Path(cfg["paths"]["data"]) / "scenario_mapping_table.tex"
    df.to_csv(out_csv, index=False)
    df.to_latex(out_tex, index=False, float_format="%.3g")
    print(f"Wrote {out_csv}")
    print(f"Wrote {out_tex}")


if __name__ == "__main__":
    main()
