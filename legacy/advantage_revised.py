
import numpy as np
import matplotlib.pyplot as plt


def calculate_GQ_advantage(eta, N_S):
    """
    Pure-loss TMSV quantum advantage ratio.

    G_Q(eta, N_S) = (N_S + 1) / [1 + 2(1 - eta)N_S]
    """
    eta = np.asarray(eta, dtype=float)
    N_S = np.asarray(N_S, dtype=float)

    if np.any((eta <= 0) | (eta > 1)):
        raise ValueError("eta must satisfy 0 < eta <= 1.")
    if np.any(N_S <= 0):
        raise ValueError("N_S must be positive.")

    return (N_S + 1.0) / (1.0 + 2.0 * (1.0 - eta) * N_S)


def calculate_G_eff_diffusion(eta, N_S, Gamma):
    """
    First-order effective advantage under pure loss and phase diffusion.

    G_eff(eta, N_S, Gamma) ≈ G_Q(eta, N_S) * exp(-Gamma)

    This is a first-order analytic interface for the main-text survival map.
    It is not a full non-Gaussian QZZB calculation.
    """
    Gamma = np.asarray(Gamma, dtype=float)
    if np.any(Gamma < 0):
        raise ValueError("Gamma must be non-negative.")

    G_Q = calculate_GQ_advantage(eta, N_S)
    return G_Q * np.exp(-Gamma)


def survival_boundary_gamma(eta, N_S):
    """
    Analytic boundary for G_eff = 1:
        G_Q(eta, N_S) * exp(-Gamma) = 1
        Gamma_boundary = ln[G_Q(eta, N_S)]

    If G_Q <= 1, no positive-Gamma survival region exists.
    """
    G_Q = calculate_GQ_advantage(eta, N_S)
    boundary = np.where(G_Q > 1.0, np.log(G_Q), np.nan)
    return boundary


def run_unit_test():
    """
    Unit tests for several manuscript anchor points.

    For N_S=100:
        eta=0.90: G_Q≈4.8095, Gamma_boundary≈ln(4.8095)=1.5706
        eta=0.50: G_Q=1, Gamma_boundary=0
        eta=0.30: G_Q<1, no survival region
    """
    test_cases = [
        {"N_S": 100, "eta": 0.90, "Gamma": 0.0},
        {"N_S": 100, "eta": 0.90, "Gamma": 1.0},
        {"N_S": 100, "eta": 0.50, "Gamma": 0.0},
        {"N_S": 100, "eta": 0.30, "Gamma": 0.0},
    ]

    print("--- Unit test results for effective advantage G_eff ---")
    for tc in test_cases:
        GQ = calculate_GQ_advantage(tc["eta"], tc["N_S"])
        Geff = calculate_G_eff_diffusion(tc["eta"], tc["N_S"], tc["Gamma"])
        boundary = survival_boundary_gamma(tc["eta"], tc["N_S"])
        print(
            f"N_S={tc['N_S']:>3}, eta={tc['eta']:.2f}, Gamma={tc['Gamma']:.2f} | "
            f"G_Q={GQ:.4f}, G_eff={Geff:.4f}, "
            f"Gamma_boundary={boundary if np.isfinite(boundary) else 'no survival'}"
        )


def plot_fig4_survival_island(save_path=None):
    """
    Fig. 4: Loss-diffusion survival map.

    Left panel:
        Heatmap of G_eff(eta, Gamma) for N_S=100.
        Cyan line marks G_eff=1 survival boundary.

    Right panel:
        Analytic survival-boundary curves for several N_S values.
        This shows how the survival window depends on signal photon number.
    """
    N_S_target = 100

    # Include the loss threshold eta=0.5 and the high-transmittance tail.
    eta_scan = np.linspace(0.30, 1.00, 450)
    Gamma_scan = np.linspace(0.0, 5.0, 420)
    eta_grid, Gamma_grid = np.meshgrid(eta_scan, Gamma_scan)

    G_eff_grid = calculate_G_eff_diffusion(eta_grid, N_S_target, Gamma_grid)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15.5, 6.8))

    # -------------------------
    # Panel A: heatmap
    # -------------------------
    mesh = ax1.pcolormesh(
        eta_grid,
        Gamma_grid,
        G_eff_grid,
        cmap="magma",
        shading="auto",
        vmin=0.0,
        vmax=4.0,
    )
    cbar = fig.colorbar(mesh, ax=ax1)
    cbar.set_label(r"Effective advantage $G_{\rm eff}$ (clipped at 4)", fontsize=12)

    # G_eff = 1 survival boundary
    contour = ax1.contour(
        eta_grid,
        Gamma_grid,
        G_eff_grid,
        levels=[1.0],
        colors="cyan",
        linewidths=2.8,
    )
    ax1.clabel(contour, inline=True, fontsize=11, fmt={1.0: r"$G_{\rm eff}=1$"})

    # Pure-loss threshold eta=0.5
    ax1.axvline(
        0.5,
        color="white",
        linestyle=":",
        lw=1.8,
        alpha=0.9,
        label=r"Pure-loss threshold $\eta=0.5$",
    )

    # Mild visual region guidance
    ax1.text(
        0.83,
        0.65,
        "Survival region\n" + r"($G_{\rm eff}>1$)",
        color="cyan",
        fontsize=12,
        fontweight="bold",
        ha="center",
        va="center",
    )
    ax1.text(
        0.63,
        3.8,
        "No effective\nTMSV advantage",
        color="white",
        fontsize=11,
        ha="center",
        va="center",
        alpha=0.85,
    )

    ax1.set_xlabel(r"System transmittance $\eta$", fontsize=13)
    ax1.set_ylabel(r"Accumulated phase diffusion $\Gamma=\gamma_\phi\tau_{\rm int}$", fontsize=13)
    ax1.set_title(
        fr"(a) Survival map for $N_S={N_S_target}$",
        fontsize=14,
        fontweight="bold",
    )
    ax1.set_xlim(0.30, 1.00)
    ax1.set_ylim(0.0, 5.0)
    ax1.grid(True, ls="--", alpha=0.25)
    ax1.legend(fontsize=9, loc="upper left")

    # -------------------------
    # Panel B: analytic boundary curves for selected N_S
    # -------------------------
    NS_list = [10, 30, 100]
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]

    eta_line = np.linspace(0.3001, 1.0, 600)
    for NS, color in zip(NS_list, colors):
        gamma_boundary = survival_boundary_gamma(eta_line, NS)
        ax2.plot(
            eta_line,
            gamma_boundary,
            color=color,
            lw=2.7,
            label=fr"$N_S={NS}$",
        )

    ax2.axhline(
        0.0,
        color="red",
        linestyle="--",
        lw=1.7,
        label=r"$\Gamma=0$",
    )
    ax2.axvline(
        0.5,
        color="gray",
        linestyle=":",
        lw=1.8,
        label=r"$\eta=0.5$",
    )

    # Shade regions
    ax2.axvspan(0.30, 0.50, color="salmon", alpha=0.10)
    ax2.axvspan(0.50, 1.00, color="lightgreen", alpha=0.08)

    ax2.set_xlabel(r"System transmittance $\eta$", fontsize=13)
    ax2.set_ylabel(r"Maximum tolerable diffusion $\Gamma_{\rm max}=\ln G_Q$", fontsize=13)
    ax2.set_title(
        r"(b) Survival-boundary curves $G_{\rm eff}=1$",
        fontsize=14,
        fontweight="bold",
    )
    ax2.set_xlim(0.30, 1.00)
    ax2.set_ylim(0.0, 5.0)
    ax2.grid(True, ls="--", alpha=0.35)
    ax2.legend(fontsize=10, loc="upper left")

    fig.suptitle(
        r"Fig. 4: Loss-Diffusion Survival Island of the Effective TMSV Advantage",
        fontsize=17,
        fontweight="bold",
        y=1.02,
    )

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Figure saved to: {save_path}")

    plt.show()


if __name__ == "__main__":
    run_unit_test()
    plot_fig4_survival_island(save_path="fig4_loss_diffusion_survival_island.png")
