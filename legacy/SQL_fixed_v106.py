import numpy as np
import matplotlib.pyplot as plt


def calculate_classical_sql_error(N_S, M=1, eta=1.0):
    """
    Calculate the SQL-limited phase standard deviation for an optimal coherent-state
    phase estimate under a single-sided loss channel.

    Formula:
        J_Q,coh = 4 * eta * N_S
        Var(phi_hat) >= 1 / (M * J_Q,coh)
                    = 1 / (4 * M * eta * N_S)

    Parameters
    ----------
    N_S : float or ndarray
        Mean transmitted signal photons.
    M : int or float
        Number of independent statistical samples.
    eta : float
        One-sided transmissivity / detection efficiency, 0 < eta <= 1.

    Returns
    -------
    sigma_phi : float or ndarray
        Phase standard deviation in radians.
    """
    N_S = np.asarray(N_S, dtype=float)

    if np.any(N_S <= 0):
        raise ValueError("N_S must be positive.")
    if M <= 0:
        raise ValueError("M must be positive.")
    if not (0 < eta <= 1):
        raise ValueError("eta must satisfy 0 < eta <= 1.")

    var_phi = 1.0 / (4.0 * M * eta * N_S)
    return np.sqrt(var_phi)


def doppler_vector_magnitude_backscatter(lambda_0=532e-9, n_index=1.000444):
    """
    Magnitude of Doppler scattering vector |K_D| for backscatter geometry.

    Convention used here:
        k = 2*pi*n/lambda_0     [rad/m]
        |K_D| = |k_s - k_i| = 2k = 4*pi*n/lambda_0  [rad/m]

    With this rad/m convention:
        phi_D = |K_D| * v * tau_int
        sigma_v = sigma_phi / (tau_int * |K_D|)

    Note:
        If one defines K_D in cycles/m instead of rad/m, an extra 2*pi factor
        appears in the denominator. The present code uses the rad/m convention.
    """
    k_i = 2.0 * np.pi * n_index / lambda_0
    return 2.0 * k_i


def convert_phase_to_velocity_error(
    sigma_phi,
    tau_int,
    lambda_0=532e-9,
    n_index=1.000444,
    geometry="backscatter",
):
    """
    Convert phase standard deviation to velocity standard deviation.

    For backscatter geometry with K_D in rad/m:
        sigma_v = sigma_phi / (tau_int * |K_D|)

    Parameters
    ----------
    sigma_phi : float or ndarray
        Phase standard deviation in radians.
    tau_int : float
        Integration time in seconds.
    lambda_0 : float
        Vacuum wavelength in meters.
    n_index : float
        Refractive index of gas medium.
    geometry : str
        Currently supports "backscatter".

    Returns
    -------
    sigma_v : float or ndarray
        Velocity standard deviation in m/s.
    """
    if tau_int <= 0:
        raise ValueError("tau_int must be positive.")

    if geometry != "backscatter":
        raise NotImplementedError("Only backscatter geometry is implemented in this first-round script.")

    K_D = doppler_vector_magnitude_backscatter(lambda_0=lambda_0, n_index=n_index)
    sigma_v = sigma_phi / (tau_int * K_D)
    return sigma_v


def add_low_photon_signal_background(ax):
    """Add the low-photon signal-mode window used in the manuscript.

This refers to signal-mode photons N_S in the SQL benchmark, not the macroscopic Rayleigh return photons N_ret used in Fig. 1.
"""
    ax.axvspan(
        1, 100,
        color="orange",
        alpha=0.15,
        label=r"Low-photon signal regime ($1<N_S\leq100$)"
    )


def plot_fig2_classical_sql(save_path=None):
    """
    Plot the classical coherent-state SQL phase and velocity error.

    The figure has two panels:
        left: SQL phase standard deviation vs N_S
        right: velocity standard deviation vs N_S

    This supports Section 3 of the manuscript:
        Var(phi_hat) >= 1 / (4 M eta N_S)
        sigma_v = sigma_phi / (tau_int |K_D|)
    """
    N_S_scan = np.logspace(0, 3, 300)  # 1 to 1000 photons

    # Scenarios are chosen to illustrate how loss and statistical averaging affect
    # the SQL baseline. They are not meant to represent a completed instrument.
    scenarios = [
        {
            "M": 1,
            "tau": 1e-6,
            "eta": 0.10,
            "label": r"$M=1,\ \eta=0.10,\ \tau=1\,\mu s$",
            "style": "-",
            "color": "#d62728",
        },
        {
            "M": 10,
            "tau": 1e-6,
            "eta": 0.10,
            "label": r"$M=10,\ \eta=0.10,\ \tau=1\,\mu s$",
            "style": "--",
            "color": "#ff7f0e",
        },
        {
            "M": 100,
            "tau": 1e-6,
            "eta": 0.30,
            "label": r"$M=100,\ \eta=0.30,\ \tau=1\,\mu s$",
            "style": "-.",
            "color": "#2ca02c",
        },
        {
            "M": 1000,
            "tau": 10e-6,
            "eta": 0.50,
            "label": r"$M=1000,\ \eta=0.50,\ \tau=10\,\mu s$",
            "style": ":",
            "color": "#1f77b4",
        },
    ]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14.5, 6))

    print("\nClassical SQL benchmark samples")
    print("=" * 78)
    print("Formula: sigma_phi = 1 / sqrt(4 M eta N_S)")
    print("Backscatter convention: sigma_v = sigma_phi / (tau_int |K_D|)")
    print("=" * 78)

    for s in scenarios:
        sig_phi = calculate_classical_sql_error(N_S_scan, s["M"], s["eta"])
        sig_v = convert_phase_to_velocity_error(sig_phi, s["tau"])

        ax1.plot(
            N_S_scan,
            sig_phi,
            linestyle=s["style"],
            color=s["color"],
            linewidth=2.5,
            label=s["label"],
        )
        ax2.plot(
            N_S_scan,
            sig_v,
            linestyle=s["style"],
            color=s["color"],
            linewidth=2.5,
            label=s["label"],
        )

        # Print representative values for sanity checks.
        for N0 in [1, 10, 100]:
            phi0 = calculate_classical_sql_error(N0, s["M"], s["eta"])
            v0 = convert_phase_to_velocity_error(phi0, s["tau"])
            print(
                f"{s['label']}, N_S={N0:3d}: "
                f"sigma_phi={phi0:.3g} rad, sigma_v={v0:.3g} m/s"
            )
        print("-" * 78)

    # Left panel: phase error
    add_low_photon_signal_background(ax1)
    ax1.axhline(
        np.pi,
        color="red",
        alpha=0.55,
        linestyle="-",
        linewidth=1.5,
        label=r"Phase wrapping scale ($\pi$)",
    )
    ax1.axhline(
        1.0,
        color="gray",
        alpha=0.45,
        linestyle="--",
        linewidth=1.3,
        label=r"$1$ rad reference",
    )
    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.set_xlabel(r"Transmitted signal photons $N_S$", fontsize=13)
    ax1.set_ylabel(r"Phase standard deviation $\sigma_{\hat{\phi}}$ (rad)", fontsize=13)
    ax1.set_title("Classical coherent-state SQL phase error", fontsize=14, fontweight="bold")
    ax1.grid(True, which="both", ls="--", alpha=0.4)
    ax1.legend(fontsize=9, loc="lower left")

    # Right panel: velocity error
    add_low_photon_signal_background(ax2)
    ax2.set_xscale("log")
    ax2.set_yscale("log")
    ax2.set_xlabel(r"Transmitted signal photons $N_S$", fontsize=13)
    ax2.set_ylabel(r"Velocity standard deviation $u(v)$ (m/s)", fontsize=13)
    ax2.set_title("Velocity error propagated from phase SQL", fontsize=14, fontweight="bold")
    ax2.grid(True, which="both", ls="--", alpha=0.4)
    ax2.legend(fontsize=9, loc="lower left")

    fig.suptitle(
        "Fig. 2: Classical SQL Error Propagation in the Low-Photon Signal Regime",
        fontsize=16,
        fontweight="bold",
        y=1.02,
    )
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"\nFigure saved to: {save_path}")

    plt.show()


if __name__ == "__main__":
    plot_fig2_classical_sql(save_path="fig2_classical_sql_v106.png")
