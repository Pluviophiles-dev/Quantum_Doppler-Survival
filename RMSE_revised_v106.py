import numpy as np
import matplotlib.pyplot as plt


def doppler_vector_magnitude_backscatter(lambda_0=532e-9, n_index=1.000444):
    """
    Backscatter Doppler vector magnitude |K_D| in rad/m.

    k = 2*pi*n/lambda_0, |K_D| = 2k = 4*pi*n/lambda_0.
    With this convention, the accumulated Doppler phase is
        phi_D = |K_D| * v * tau_int,
    not 2*pi * |K_D| * v * tau_int.
    """
    return 4.0 * np.pi * n_index / lambda_0


def calculate_GQ_advantage(eta, N_S):
    """
    Pure-loss TMSV advantage ratio:
        G_Q = (N_S + 1) / [1 + 2(1 - eta)N_S].
    """
    eta = np.asarray(eta, dtype=float)
    N_S = np.asarray(N_S, dtype=float)
    if np.any((eta <= 0) | (eta > 1)):
        raise ValueError("eta must satisfy 0 < eta <= 1.")
    if np.any(N_S <= 0):
        raise ValueError("N_S must be positive.")
    return (N_S + 1.0) / (1.0 + 2.0 * (1.0 - eta) * N_S)


def calculate_Geff(eta, N_S, Gamma):
    """
    First-order effective advantage under phase diffusion:
        G_eff = G_Q * exp(-Gamma).

    This is a first-order survival-boundary proxy, not a full non-Gaussian QFI/QZZB calculation.
    """
    return calculate_GQ_advantage(eta, N_S) * np.exp(-Gamma)


def phase_variances(eta, N_S, Gamma, M=1):
    """
    Coherent-state and TMSV phase-estimation variances.

    Classical coherent-state benchmark:
        Var_CS = 1 / (4 M eta N_S)

    TMSV first-order effective model:
        Var_TMSV = Var_CS / G_eff

    If G_eff < 1, the TMSV protocol is worse than the coherent-state benchmark. We do NOT cap it at 1,
    because Fig. 5 should show the actual disappearance of the TMSV advantage.
    """
    var_cs = 1.0 / (4.0 * M * eta * N_S)
    g_eff = calculate_Geff(eta, N_S, Gamma)
    var_tmsv = var_cs / g_eff
    return var_cs, var_tmsv, g_eff


def analytic_velocity_rmse(lambda_0, tau_int, n_index, eta, N_S, Gamma, M=1):
    """
    Analytic velocity RMSE predicted by the phase variance model.
    """
    K_D = doppler_vector_magnitude_backscatter(lambda_0=lambda_0, n_index=n_index)
    var_cs, var_tmsv, g_eff = phase_variances(eta, N_S, Gamma, M=M)
    rmse_cs = np.sqrt(var_cs) / (tau_int * K_D)
    rmse_tmsv = np.sqrt(var_tmsv) / (tau_int * K_D)
    return rmse_cs, rmse_tmsv, g_eff


def monte_carlo_velocity_rmse(
    v0,
    lambda_0,
    tau_int,
    n_index,
    eta,
    N_S,
    Gamma,
    M=1,
    num_trials=30000,
    rng=None,
):
    """
    Monte Carlo sampling of velocity RMSE from Gaussian phase-estimation noise.

    The true Doppler phase is phi_0 = |K_D| * v0 * tau_int.
    The velocity estimator is v_hat = phi_hat / (|K_D| * tau_int).
    """
    if rng is None:
        rng = np.random.default_rng(42)

    K_D = doppler_vector_magnitude_backscatter(lambda_0=lambda_0, n_index=n_index)
    phi_0 = v0 * tau_int * K_D

    var_cs, var_tmsv, g_eff = phase_variances(eta, N_S, Gamma, M=M)

    phi_cs = rng.normal(loc=phi_0, scale=np.sqrt(var_cs), size=num_trials)
    phi_tmsv = rng.normal(loc=phi_0, scale=np.sqrt(var_tmsv), size=num_trials)

    v_cs = phi_cs / (tau_int * K_D)
    v_tmsv = phi_tmsv / (tau_int * K_D)

    rmse_cs = np.sqrt(np.mean((v_cs - v0) ** 2))
    rmse_tmsv = np.sqrt(np.mean((v_tmsv - v0) ** 2))
    return rmse_cs, rmse_tmsv, g_eff


def eta_boundary_for_Geff_one(N_S, Gamma):
    """
    Solve G_eff(eta, N_S, Gamma) = 1 for eta.
    If no valid boundary exists in (0,1], return np.nan.
    """
    target = np.exp(Gamma)
    denom_needed = (N_S + 1.0) / target
    eta = 1.0 - (denom_needed - 1.0) / (2.0 * N_S)
    if eta <= 0 or eta > 1:
        return np.nan
    return eta


def gamma_boundary_for_Geff_one(eta, N_S):
    """
    Solve G_eff(eta, N_S, Gamma) = 1 for Gamma.
        Gamma_max = ln(G_Q)
    Only meaningful when G_Q > 1.
    """
    gq = calculate_GQ_advantage(eta, N_S)
    return np.log(gq) if gq > 1 else np.nan


def run_unit_test():
    """Minimal checks for phase-to-velocity mapping and advantage boundary."""
    print("--- Unit test for Fig. 5 RMSE model ---")
    lambda_0 = 532e-9
    tau_int = 1e-6
    n_index = 1.000444
    eta = 0.85
    N_S = 100
    Gamma = 0.2
    M = 1

    rmse_cs, rmse_tmsv, g_eff = analytic_velocity_rmse(
        lambda_0, tau_int, n_index, eta, N_S, Gamma, M=M
    )
    print(f"G_eff(eta={eta}, N_S={N_S}, Gamma={Gamma}) = {g_eff:.4f}")
    print(f"Analytic CS RMSE   = {rmse_cs:.6e} m/s")
    print(f"Analytic TMSV RMSE = {rmse_tmsv:.6e} m/s")
    print(f"RMSE ratio CS/TMSV = {rmse_cs / rmse_tmsv:.4f} = sqrt(G_eff)")

    gamma_b = gamma_boundary_for_Geff_one(eta, N_S)
    eta_b = eta_boundary_for_Geff_one(N_S, Gamma)
    print(f"Gamma boundary at eta={eta}, N_S={N_S}: {gamma_b:.4f}")
    print(f"Eta boundary at Gamma={Gamma}, N_S={N_S}: {eta_b:.4f}")


def plot_fig5_monte_carlo_rmse(save_path=None):
    # Representative measurement parameters
    v0 = 15.0
    lambda_0 = 532e-9
    tau_int = 1e-6
    n_index = 1.000444
    N_S_base = 100
    Gamma_base = 0.2
    M = 1
    num_trials = 30000
    rng = np.random.default_rng(42)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15.5, 6.3))

    # --- Panel A: RMSE vs transmittance eta ---
    eta_scan = np.linspace(0.1, 1.0, 90)
    rmse_cs_eta = []
    rmse_tmsv_eta = []
    g_eff_eta = []

    for eta in eta_scan:
        rcs, rtmsv, geff = monte_carlo_velocity_rmse(
            v0, lambda_0, tau_int, n_index, eta, N_S_base, Gamma_base,
            M=M, num_trials=num_trials, rng=rng
        )
        rmse_cs_eta.append(rcs)
        rmse_tmsv_eta.append(rtmsv)
        g_eff_eta.append(geff)

    rmse_cs_eta = np.asarray(rmse_cs_eta)
    rmse_tmsv_eta = np.asarray(rmse_tmsv_eta)
    g_eff_eta = np.asarray(g_eff_eta)

    eta_b = eta_boundary_for_Geff_one(N_S_base, Gamma_base)

    ax1.plot(eta_scan, rmse_cs_eta, color="#d62728", linestyle="--", lw=2.5,
             label="Coherent-state benchmark")
    ax1.plot(eta_scan, rmse_tmsv_eta, color="#1f77b4", linestyle="-", lw=2.5,
             label="TMSV effective model")
    ax1.fill_between(
        eta_scan,
        rmse_cs_eta,
        rmse_tmsv_eta,
        where=rmse_tmsv_eta < rmse_cs_eta,
        color="#1f77b4",
        alpha=0.18,
        label="TMSV RMSE reduction",
    )
    if not np.isnan(eta_b):
        ax1.axvline(eta_b, color="gray", linestyle=":", lw=2.0,
                    label=fr"$G_{{eff}}=1$ boundary ($\eta\approx{eta_b:.2f}$)")

    ax1.set_xlabel(r"System transmittance $\eta$", fontsize=13)
    ax1.set_ylabel(r"Velocity RMSE (m/s)", fontsize=13)
    ax1.set_title(fr"RMSE vs $\eta$ ($N_S={N_S_base}$, $\Gamma={Gamma_base}$)",
                  fontsize=14, fontweight="bold")
    ax1.grid(True, ls="--", alpha=0.4)
    ax1.legend(fontsize=9, loc="upper right")

    # --- Panel B: RMSE vs phase diffusion Gamma ---
    Gamma_scan = np.linspace(0.0, 3.2, 90)
    eta_fixed = 0.85
    rmse_cs_gam = []
    rmse_tmsv_gam = []
    g_eff_gam = []

    for Gamma in Gamma_scan:
        rcs, rtmsv, geff = monte_carlo_velocity_rmse(
            v0, lambda_0, tau_int, n_index, eta_fixed, N_S_base, Gamma,
            M=M, num_trials=num_trials, rng=rng
        )
        rmse_cs_gam.append(rcs)
        rmse_tmsv_gam.append(rtmsv)
        g_eff_gam.append(geff)

    rmse_cs_gam = np.asarray(rmse_cs_gam)
    rmse_tmsv_gam = np.asarray(rmse_tmsv_gam)
    g_eff_gam = np.asarray(g_eff_gam)

    gamma_b = gamma_boundary_for_Geff_one(eta_fixed, N_S_base)

    ax2.plot(Gamma_scan, rmse_cs_gam, color="#d62728", linestyle="--", lw=2.5,
             label="Coherent-state benchmark")
    ax2.plot(Gamma_scan, rmse_tmsv_gam, color="#1f77b4", linestyle="-", lw=2.5,
             label="TMSV effective model")
    ax2.fill_between(
        Gamma_scan,
        rmse_cs_gam,
        rmse_tmsv_gam,
        where=rmse_tmsv_gam < rmse_cs_gam,
        color="#1f77b4",
        alpha=0.18,
        label="TMSV RMSE reduction",
    )
    if not np.isnan(gamma_b):
        ax2.axvline(gamma_b, color="gray", linestyle=":", lw=2.0,
                    label=fr"$G_{{eff}}=1$ boundary ($\Gamma\approx{gamma_b:.2f}$)")

    ax2.set_xlabel(r"Accumulated phase diffusion $\Gamma$", fontsize=13)
    ax2.set_ylabel(r"Velocity RMSE (m/s)", fontsize=13)
    ax2.set_title(fr"RMSE vs $\Gamma$ ($N_S={N_S_base}$, $\eta={eta_fixed}$)",
                  fontsize=14, fontweight="bold")
    ax2.grid(True, ls="--", alpha=0.4)
    ax2.legend(fontsize=9, loc="upper left")

    fig.suptitle(
        "Fig. 5: Monte Carlo Velocity RMSE under the First-Order TMSV Survival Model",
        fontsize=16,
        fontweight="bold",
        y=1.02,
    )
    fig.text(
        0.5,
        -0.02,
        r"Monte Carlo samples use $\phi_D=|K_D|v\tau_{int}$ and "
        r"$\mathrm{Var}_{TMSV}=\mathrm{Var}_{CS}/G_{eff}$; "
        r"no clipping is applied when $G_{eff}<1$; no wrapped-phase unwrapping is included.",
        ha="center",
        fontsize=10,
    )

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Figure saved to: {save_path}")

    plt.show()


if __name__ == "__main__":
    run_unit_test()
    plot_fig5_monte_carlo_rmse(save_path="fig5_monte_carlo_velocity_rmse_v106.png")
