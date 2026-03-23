"""Example custom physics model: Fresnel reflection at sample surface.

Drop this file (without the leading underscore) into data/custom_models/
to activate it, or use it as a template for your own models.

The Interaction Analyzer will call ``compute()`` and display the returned
metrics alongside the built-in analysis.
"""
from __future__ import annotations

import math

# ── Required metadata ──────────────────────────────────────────────

MODEL_NAME = "Fresnel Surface Reflection"
MODEL_DESCRIPTION = "Computes single-surface Fresnel reflection for s, p, and unpolarized light at normal incidence."
MODEL_VERSION = "1.0.0"


# ── Required entry point ──────────────────────────────────────────

def compute(
    laser: dict,
    material: dict,
    thickness_m: float,
    z_position_m: float = 0.0,
) -> dict:
    """Fresnel reflection at the air-material interface.

    Parameters are plain dicts (dataclasses.asdict output) so this
    module has *no dependency* on harrington-lmi internals.

    Returns
    -------
    dict
        Keys become metric labels.  Values can be float, str, or
        a plot dict ``{"x": [...], "y": [...], "label": str}``.
    """
    n1 = 1.0  # air
    n2 = material.get("refractive_index", 1.0)
    polarization = laser.get("polarization", "linear")

    if n2 <= 0:
        return {"error": "Invalid refractive index"}

    # Normal incidence reflectance (same for s and p)
    R_normal = ((n1 - n2) / (n1 + n2)) ** 2
    T_normal = 1.0 - R_normal

    # Brewster's angle
    theta_B_rad = math.atan2(n2, n1)
    theta_B_deg = math.degrees(theta_B_rad)

    # Angle-resolved R for a quick plot (0-85 deg)
    angles_deg = list(range(0, 86))
    Rs_vals = []
    Rp_vals = []
    for theta_deg in angles_deg:
        theta = math.radians(theta_deg)
        sin_t = n1 * math.sin(theta) / n2
        if abs(sin_t) >= 1.0:
            Rs_vals.append(1.0)
            Rp_vals.append(1.0)
            continue
        cos_t = math.sqrt(1 - sin_t ** 2)
        cos_i = math.cos(theta)

        rs = (n1 * cos_i - n2 * cos_t) / (n1 * cos_i + n2 * cos_t)
        rp = (n2 * cos_i - n1 * cos_t) / (n2 * cos_i + n1 * cos_t)
        Rs_vals.append(rs ** 2)
        Rp_vals.append(rp ** 2)

    results = {
        "R (normal incidence)": f"{R_normal:.4f}  ({R_normal*100:.2f}%)",
        "T (normal incidence)": f"{T_normal:.4f}  ({T_normal*100:.2f}%)",
        f"Brewster Angle": f"{theta_B_deg:.1f}°",
        "Reflected Power (W)": round(laser.get("power_w", 0) * R_normal, 4),
        "Fresnel_Rs_vs_angle": {
            "x": angles_deg,
            "y": Rs_vals,
            "label": "Rs (s-pol)",
        },
        "Fresnel_Rp_vs_angle": {
            "x": angles_deg,
            "y": Rp_vals,
            "label": "Rp (p-pol)",
        },
    }

    return results
