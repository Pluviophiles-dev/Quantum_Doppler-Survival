from __future__ import annotations


def classify_boundary_point(
    geff_value: float,
    guard_ratio: float,
    wrap_probability: float,
    guard_ratio_threshold: float = 2.0,
    wrap_probability_threshold: float = 0.05,
    local_ratio_tolerance: float = 1.25,
) -> str:
    """Objective classification logic.

    stop-extrapolation: no local advantage, or wrapping risk is high.
    guarded: local advantage exists but QZZB floor substantially exceeds local variance.
    local-valid: local advantage exists and QZZB does not materially raise the lower bound.
    """
    if geff_value <= 1.0 or wrap_probability > wrap_probability_threshold:
        return "stop-extrapolation"
    if guard_ratio > guard_ratio_threshold:
        return "guarded"
    if guard_ratio <= local_ratio_tolerance:
        return "local-valid"
    return "guarded"


CLASS_TO_INT = {"local-valid": 0, "guarded": 1, "stop-extrapolation": 2}
INT_TO_CLASS = {v: k for k, v in CLASS_TO_INT.items()}
