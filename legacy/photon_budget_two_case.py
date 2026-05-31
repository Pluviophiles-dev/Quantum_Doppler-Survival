import numpy as np
import matplotlib.pyplot as plt
import scipy.constants as const


def solve_pr_eos_density(P_MPa, T_K):
    """
    基于 Peng-Robinson 状态方程计算甲烷数密度。

    输入:
        P_MPa: 压力，单位 MPa
        T_K: 温度，单位 K

    输出:
        number_density: 分子数密度，单位 molecules/m^3

    说明:
        这里以纯甲烷作为天然气主成分近似。
        后续若需要多组分天然气，可在此函数外层加入组分加权或替换为更高精度物性模型。
    """
    # Methane critical parameters
    Tc = 190.56       # K
    Pc = 4.5992       # MPa
    omega = 0.011     # acentric factor
    R = const.R

    P_Pa = P_MPa * 1e6
    Pc_Pa = Pc * 1e6

    # Peng-Robinson parameters
    kappa = 0.37464 + 1.54226 * omega - 0.26992 * omega**2
    alpha = (1 + kappa * (1 - np.sqrt(T_K / Tc)))**2

    a = 0.45724 * (R * Tc)**2 / Pc_Pa * alpha
    b = 0.07780 * R * Tc / Pc_Pa

    A = a * P_Pa / (R * T_K)**2
    B = b * P_Pa / (R * T_K)

    # PR EOS compressibility factor cubic:
    # Z^3 - (1-B)Z^2 + (A - 3B^2 - 2B)Z - (AB - B^2 - B^3) = 0
    coeffs = [
        1.0,
        -(1.0 - B),
        A - 3.0 * B**2 - 2.0 * B,
        -(A * B - B**2 - B**3),
    ]

    roots = np.roots(coeffs)

    # Numerical cubic roots may carry tiny imaginary residuals.
    real_roots = np.real(roots[np.abs(np.imag(roots)) < 1e-8])
    positive_roots = real_roots[real_roots > 0]

    if len(positive_roots) == 0:
        # Conservative fallback: ideal gas
        Z = 1.0
    else:
        # Gas-phase root: largest positive real root
        Z = np.max(positive_roots)

    molar_volume = Z * R * T_K / P_Pa
    number_density = const.Avogadro / molar_volume

    return number_density


def get_rayleigh_cross_section(lambda_nm):
    """
    计算甲烷的瑞利散射截面。

    输入:
        lambda_nm: 真空波长，单位 nm

    输出:
        sigma_R: 瑞利散射截面，单位 m^2

    说明:
        使用标准状态折射率近似和 King 修正因子。
        该函数适合用于第一轮趋势仿真。
    """
    lambda_m = lambda_nm * 1e-9

    # Approximate methane refractive index at standard conditions
    n0 = 1.000444

    # Loschmidt constant, molecules/m^3
    rho_0 = 2.68678e25

    # King correction factor for methane, approximate value
    F_k = 1.04

    term1 = (24 * np.pi**3) / (rho_0**2 * lambda_m**4)
    term2 = ((n0**2 - 1) / (n0**2 + 2))**2

    sigma_R = term1 * term2 * F_k
    return sigma_R


def compute_return_photons(P_scan, wavelength_nm, T_K, L, Omega_frac, eta_sys, E_pulse):
    """
    根据光子预算公式计算不同压力下的返回光子数。

    N_ret = N_tx * sigma_R * rho_N * L * (Omega / 4pi) * eta_sys
    """
    E_photon = const.h * const.c / (wavelength_nm * 1e-9)
    N_tx = E_pulse / E_photon
    sigma_R = get_rayleigh_cross_section(wavelength_nm)

    N_ret_array = np.zeros_like(P_scan, dtype=float)

    for i, P in enumerate(P_scan):
        rho_N = solve_pr_eos_density(P, T_K)
        N_ret_array[i] = N_tx * sigma_R * rho_N * L * Omega_frac * eta_sys

    return N_ret_array


def add_photon_regime_background(ax):
    """
    添加 photon-rich / low-return / photon-starved / extreme photon-starved 背景分区。
    保留你原代码中颜色清楚、可读性好的优点。
    """
    ax.axhspan(1e4, 1e8, color="lightgreen", alpha=0.15,
               label=r"Photon-rich ($N_{\rm ret}>10^4$)")
    ax.axhspan(1e2, 1e4, color="yellow", alpha=0.15,
               label=r"Low-return ($10^2<N_{\rm ret}\leq10^4$)")
    ax.axhspan(1, 1e2, color="orange", alpha=0.25,
               label=r"Photon-starved ($1<N_{\rm ret}\leq100$)")
    ax.axhspan(1e-4, 1, color="red", alpha=0.15,
               label=r"Extreme photon-starved ($N_{\rm ret}\leq1$)")


def plot_fig1_photon_budget_two_cases(save_path=None):
    """
    生成双场景 photon budget 图。

    左图: 较乐观场景，说明系统不一定总是 photon-starved。
    右图: 受限场景，说明在低能量、低收集角、有限效率下会进入 photon-starved 区间。

    这比单一参数图更适合论文，因为它体现了边界条件和可证伪性。
    """
    P_scan = np.linspace(1, 35, 200)

    wavelengths = [532, 633, 1064, 1550]
    colors = ["#1f77b4", "#2ca02c", "#ff7f0e", "#d62728"]

    T = 298.15
    L = 0.01  # 10 mm probe volume length

    cases = [
        {
            "title": "Optimistic high-return case",
            "E_pulse": 1e-3,       # 1 mJ
            "Omega_frac": 1e-6,
            "eta_sys": 0.10,
            "ylim": (1e-2, 1e6),
        },
        {
            "title": "Constrained photon-starved case",
            "E_pulse": 1e-6,       # 1 microjoule
            "Omega_frac": 1e-7,
            "eta_sys": 0.05,
            "ylim": (1e-4, 1e4),
        },
    ]

    fig, axes = plt.subplots(1, 2, figsize=(15, 6), sharex=True)

    for ax, case in zip(axes, cases):
        add_photon_regime_background(ax)

        print("\n" + "=" * 72)
        print(case["title"])
        print(f"E_pulse     = {case['E_pulse']:.2e} J")
        print(f"Omega_frac  = {case['Omega_frac']:.2e}")
        print(f"eta_sys     = {case['eta_sys']:.2f}")
        print("-" * 72)

        for wl, color in zip(wavelengths, colors):
            N_ret_array = compute_return_photons(
                P_scan=P_scan,
                wavelength_nm=wl,
                T_K=T,
                L=L,
                Omega_frac=case["Omega_frac"],
                eta_sys=case["eta_sys"],
                E_pulse=case["E_pulse"],
            )

            ax.plot(
                P_scan,
                N_ret_array,
                label=fr"$\lambda={wl}$ nm",
                color=color,
                lw=2.5,
            )

            # Print representative values for easy checking
            idx_30 = np.argmin(np.abs(P_scan - 30.0))
            idx_35 = np.argmin(np.abs(P_scan - 35.0))
            print(
                f"{wl:4d} nm: "
                f"N_ret(30 MPa) = {N_ret_array[idx_30]:.3g}, "
                f"N_ret(35 MPa) = {N_ret_array[idx_35]:.3g}"
            )

        ax.set_yscale("log")
        ax.set_xlim(1, 35)
        ax.set_ylim(case["ylim"])
        ax.set_xlabel("Pipeline Pressure $P$ (MPa)", fontsize=13)
        ax.set_title(case["title"], fontsize=14, fontweight="bold")
        ax.grid(True, which="both", ls="--", alpha=0.4)

    axes[0].set_ylabel(r"Expected return photons $N_{\rm ret}$ per pulse", fontsize=13)

    # Put one clean legend outside
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="center right",
        bbox_to_anchor=(1.18, 0.5),
        fontsize=10,
        frameon=True,
    )

    fig.suptitle(
        "Photon Budget vs Pressure and Wavelength under Two Optical Constraints",
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
    plot_fig1_photon_budget_two_cases(
        save_path="fig1_photon_budget_two_cases.png"
    )
