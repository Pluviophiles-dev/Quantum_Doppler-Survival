from __future__ import annotations

import numpy as np

AVOGADRO = 6.02214076e23
BOLTZMANN = 1.380649e-23
R = 8.31446261815324
C = 299792458.0
H = 6.62607015e-34
N0 = 2.68678e25


def methane_number_density_peng_robinson(P_Pa: float, T_K: float) -> float:
    """Methane number density using a simple Peng-Robinson EOS vapor root."""
    Tc = 190.564
    Pc = 4.5992e6
    omega = 0.01142
    kappa = 0.37464 + 1.54226 * omega - 0.26992 * omega**2
    alpha = (1.0 + kappa * (1.0 - np.sqrt(T_K / Tc))) ** 2
    a = 0.45724 * R**2 * Tc**2 / Pc * alpha
    b = 0.07780 * R * Tc / Pc
    A = a * P_Pa / (R**2 * T_K**2)
    B = b * P_Pa / (R * T_K)
    # PR cubic: Z^3 -(1-B)Z^2 +(A-3B^2-2B)Z -(AB-B^2-B^3)=0
    coeff = [1.0, -(1.0 - B), A - 3.0 * B**2 - 2.0 * B, -(A * B - B**2 - B**3)]
    roots = np.roots(coeff)
    real_roots = sorted([r.real for r in roots if abs(r.imag) < 1e-8])
    Z = max(real_roots) if real_roots else max(roots, key=lambda x: x.real).real
    mol_density = P_Pa / (Z * R * T_K)
    return float(mol_density * AVOGADRO)


def rayleigh_cross_section(lambda0_m: float, n_ref: float = 1.000444, king_factor: float = 1.04, n0: float = N0) -> float:
    """Single-molecule equivalent Rayleigh cross section used as a budget-layer proxy."""
    ratio = ((n_ref**2 - 1.0) / (n_ref**2 + 2.0)) ** 2
    return float((24.0 * np.pi**3 / (n0**2 * lambda0_m**4)) * ratio * king_factor)


def emitted_photons(pulse_energy_J: float, lambda0_m: float) -> float:
    return float(pulse_energy_J / (H * C / lambda0_m))


def return_photons(
    pressure_MPa: float,
    temperature_K: float,
    lambda_nm: float,
    pulse_energy_J: float,
    probe_length_m: float,
    collection_fraction: float,
    eta_sys: float,
    n_ref: float = 1.000444,
    king_factor: float = 1.04,
) -> float:
    lambda0_m = lambda_nm * 1e-9
    n = methane_number_density_peng_robinson(pressure_MPa * 1e6, temperature_K)
    sigma = rayleigh_cross_section(lambda0_m, n_ref=n_ref, king_factor=king_factor)
    Nin = emitted_photons(pulse_energy_J, lambda0_m)
    return float(Nin * n * probe_length_m * sigma * collection_fraction * eta_sys)


def zero_count_probability(Nret: float) -> float:
    return float(np.exp(-Nret))


def photon_regime(P0: float) -> str:
    if P0 < 0.01:
        return "photon-rich"
    if P0 < 0.37:
        return "low-return"
    if P0 < 0.90:
        return "photon-starved"
    return "extreme photon-starved"
