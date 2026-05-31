from __future__ import annotations

import numpy as np


def tmsv_covariance(Ns: float, eta_s: float = 1.0, eta_i: float = 1.0) -> np.ndarray:
    """Two-mode TMSV covariance matrix after independent signal/idler losses.

    Convention: hbar=2, vacuum covariance is identity.
    The returned matrix is ordered as (x_s, p_s, x_i, p_i).
    """
    I2 = np.eye(2)
    Z2 = np.diag([1.0, -1.0])
    a = 1.0 + 2.0 * eta_s * Ns
    b = 1.0 + 2.0 * eta_i * Ns
    c = 2.0 * np.sqrt(max(0.0, eta_s * eta_i * Ns * (Ns + 1.0)))
    return np.block([[a * I2, c * Z2], [c * Z2, b * I2]])


def epr_correlation_strength(V: np.ndarray) -> float:
    """Simple covariance-level diagnostic: Frobenius norm of signal-idler block."""
    C = V[:2, 2:]
    return float(np.linalg.norm(C, ord="fro"))


def covariance_purity(V: np.ndarray) -> float:
    """Gaussian purity under hbar=2 convention: mu = 1/sqrt(det V)."""
    det = max(float(np.linalg.det(V)), np.finfo(float).tiny)
    return 1.0 / np.sqrt(det)
