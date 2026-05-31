import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


def calculate_GQ_advantage(eta, N_S):
    """
    Pure-loss TMSV quantum advantage ratio.

    G_Q(eta, N_S) = (N_S + 1) / [1 + 2(1 - eta)N_S]

    Valid for the idealized single-sided pure-loss channel with:
    - zero thermal noise,
    - preserved idler mode,
    - equal signal-mode photon number N_S,
    - locally optimal phase estimation.
    """
    eta = np.asarray(eta, dtype=float)
    N_S = np.asarray(N_S, dtype=float)

    if np.any((eta <= 0) | (eta > 1)):
        raise ValueError("eta must satisfy 0 < eta <= 1.")
    if np.any(N_S <= 0):
        raise ValueError("N_S must be positive.")

    return (N_S + 1.0) / (1.0 + 2.0 * (1.0 - eta) * N_S)


def run_unit_test():
    """Check the numerical anchor points used in the manuscript."""
    test_cases = [
        {"N_S": 100, "eta": 0.90, "Expected_G_Q": 101 / 21},
        {"N_S": 100, "eta": 0.65, "Expected_G_Q": 101 / 71},
        {"N_S": 100, "eta": 0.50, "Expected_G_Q": 1.0},
        {"N_S": 100, "eta": 0.30, "Expected_G_Q": 101 / 141},
    ]

    print("--- Unit test results for pure-loss TMSV advantage ---")
    for tc in test_cases:
        calc_val = calculate_GQ_advantage(tc["eta"], tc["N_S"])
        expected = tc["Expected_G_Q"]
        passed = np.isclose(calc_val, expected, rtol=1e-10, atol=1e-12)
        status = "PASS" if passed else "FAIL"
        print(
            f"{status} | N_S={tc['N_S']:>3}, eta={tc['eta']:.2f} | "
            f"Expected G_Q={expected:.4f}, Calculated G_Q={calc_val:.4f}"
        )


def plot_fig3_pure_loss_tmsv_revised(save_path="fig3_pure_loss_tmsv_revised.png"):
    """
    Revised Fig. 3 for manuscript use.

    Main improvement over the previous version:
    1. The left panel is zoomed to 0 <= G_Q <= 6 to show the transition near G_Q=1.
    2. A small inset preserves the full 0-105 range, so the high-transmittance tail is not hidden.
    3. The heatmap explicitly states that color values above G_Q=6 are clipped.
    4. The eta=0.5 line is described as the SQL-equivalence boundary, not a physical breakdown point.
    """
    eta_scan = np.linspace(0.01, 1.0, 600)
    NS_list = [10, 30, 100]
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15.5, 6.5))

    # -----------------------------
    # Panel A: line plot, zoomed
    # -----------------------------
    for NS, col in zip(NS_list, colors):
        GQ_vals = calculate_GQ_advantage(eta_scan, NS)
        ax1.plot(
            eta_scan,
            GQ_vals,
            label=fr"$N_S={NS}$",
            color=col,
            lw=2.6,
        )

    ax1.axhline(
        1.0,
        color="red",
        linestyle="--",
        lw=2.0,
        label=r"Coherent-state boundary ($G_Q=1$)",
    )
    ax1.axvline(
        0.5,
        color="gray",
        linestyle=":",
        lw=1.8,
        label=r"SQL-equivalence boundary ($\eta=0.5$)",
    )
    ax1.axvspan(0.5, 1.0, color="lightgreen", alpha=0.12)
    ax1.axvspan(0.01, 0.5, color="salmon", alpha=0.08)

    ax1.text(
        0.73,
        0.35,
        r"pure-loss advantage\n$G_Q>1$",
        color="darkgreen",
        fontsize=10,
        ha="center",
        va="bottom",
    )
    ax1.text(
        0.25,
        0.35,
        r"no pure-loss\nadvantage",
        color="darkred",
        fontsize=10,
        ha="center",
        va="bottom",
    )

    ax1.set_xlabel(r"System transmittance $\eta$", fontsize=14)
    ax1.set_ylabel(r"Quantum advantage ratio $G_Q$", fontsize=14)
    ax1.set_title(r"Zoomed pure-loss advantage near $G_Q=1$", fontsize=15, fontweight="bold")
    ax1.set_xlim(0.01, 1.0)
    ax1.set_ylim(0, 6)
    ax1.grid(True, ls="--", alpha=0.4)
    ax1.legend(fontsize=9, loc="upper left")

    # Inset: full range, to avoid hiding the high-eta tail
    axins = inset_axes(ax1, width="36%", height="36%", loc="upper right", borderpad=1.3)
    for NS, col in zip(NS_list, colors):
        axins.plot(eta_scan, calculate_GQ_advantage(eta_scan, NS), color=col, lw=1.5)
    axins.axhline(1.0, color="red", linestyle="--", lw=1.0)
    axins.axvline(0.5, color="gray", linestyle=":", lw=1.0)
    axins.set_xlim(0.01, 1.0)
    axins.set_ylim(0, 105)
    axins.set_title("full range", fontsize=8)
    axins.tick_params(axis="both", labelsize=7)
    axins.grid(True, ls="--", alpha=0.25)

    # -----------------------------
    # Panel B: heatmap, clipped at 6
    # -----------------------------
    eta_grid, NS_grid = np.meshgrid(
        np.linspace(0.05, 1.0, 300),
        np.linspace(1, 150, 300),
    )
    GQ_grid = calculate_GQ_advantage(eta_grid, NS_grid)

    mesh = ax2.pcolormesh(
        eta_grid,
        NS_grid,
        GQ_grid,
        cmap="viridis",
        shading="auto",
        vmin=0,
        vmax=6,
    )
    cbar = fig.colorbar(mesh, ax=ax2)
    cbar.set_label(r"Advantage ratio $G_Q$ (values $>6$ clipped)", fontsize=12)

    contour = ax2.contour(
        eta_grid,
        NS_grid,
        GQ_grid,
        levels=[1.0],
        colors="red",
        linewidths=2.5,
    )
    ax2.clabel(contour, inline=True, fontsize=11, fmt={1.0: r"$G_Q=1$"})

    ax2.axvline(0.5, color="white", linestyle=":", lw=1.5, alpha=0.9)
    ax2.text(
        0.515,
        145,
        r"$\eta=0.5$",
        color="white",
        fontsize=11,
        va="top",
        ha="left",
    )
    ax2.text(
        0.82,
        12,
        r"color clipped\nat $G_Q=6$",
        color="white",
        fontsize=10,
        ha="center",
        va="bottom",
        bbox=dict(facecolor="black", alpha=0.25, edgecolor="none", boxstyle="round,pad=0.25"),
    )

    ax2.set_xlabel(r"System transmittance $\eta$", fontsize=14)
    ax2.set_ylabel(r"Signal-mode photons $N_S$", fontsize=14)
    ax2.set_title(r"Topology of $G_Q(\eta,N_S)$", fontsize=15, fontweight="bold")
    ax2.set_xlim(0.05, 1.0)
    ax2.set_ylim(1, 150)
    ax2.grid(False)

    plt.suptitle("Fig. 3: Pure-Loss TMSV Advantage Window", fontsize=18, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Figure saved to: {save_path}")

    plt.show()


if __name__ == "__main__":
    run_unit_test()
    plot_fig3_pure_loss_tmsv_revised()
