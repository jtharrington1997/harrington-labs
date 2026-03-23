"""Laser-material interaction regime analysis.

Determines whether an interaction is in the linear or nonlinear regime
based on laser parameters and material properties, and predicts the
dominant physical processes.
"""
from __future__ import annotations
from dataclasses import dataclass
import math
from .lasers import LaserSource
from .materials import Material


@dataclass
class InteractionResult:
    """Results of a laser-material interaction analysis."""
    regime: str             # "linear", "nonlinear_kerr", "multiphoton", "ablation", "plasma"
    dominant_processes: list[str]
    warnings: list[str]
    metrics: dict

    @property
    def is_safe(self) -> bool:
        return "ablation" not in self.regime and "plasma" not in self.regime


def classify_regime(laser: LaserSource, material: Material) -> InteractionResult:
    """Classify the interaction regime for a laser-material pair."""
    warnings: list[str] = []
    processes: list[str] = []
    metrics: dict = {}

    irradiance = laser.irradiance_w_cm2
    fluence = laser.fluence_j_cm2

    metrics["irradiance_w_cm2"] = irradiance
    metrics["fluence_j_cm2"] = fluence
    metrics["peak_power_w"] = laser.peak_power_w

    # ── Damage threshold check ──
    if material.damage_threshold_j_cm2 > 0 and fluence > 0:
        damage_ratio = fluence / material.damage_threshold_j_cm2
        metrics["damage_ratio"] = damage_ratio
        if damage_ratio > 1.0:
            warnings.append(
                f"Fluence ({fluence:.2e} J/cm²) exceeds LIDT "
                f"({material.damage_threshold_j_cm2:.1f} J/cm²) by {damage_ratio:.1f}x"
            )

    # ── Kerr nonlinearity: delta_n = n2 * I ──
    if material.nonlinear_index_cm2_w > 0 and irradiance > 0:
        delta_n = material.nonlinear_index_cm2_w * irradiance
        metrics["delta_n"] = delta_n
        metrics["delta_n_over_n"] = delta_n / material.refractive_index if material.refractive_index > 0 else 0

        # B-integral proxy (for a 1cm path)
        wavelength_cm = laser.wavelength_nm * 1e-7
        b_integral_per_cm = (2 * math.pi / wavelength_cm) * delta_n
        metrics["b_integral_per_cm"] = b_integral_per_cm

    # ── Multiphoton absorption threshold ──
    if material.bandgap_ev > 0 and laser.wavelength_nm > 0:
        photon_ev = 1240.0 / laser.wavelength_nm
        photons_needed = math.ceil(material.bandgap_ev / photon_ev) if photon_ev > 0 else 0
        metrics["photon_energy_ev"] = photon_ev
        metrics["photons_for_bandgap"] = photons_needed
        if photons_needed == 1:
            processes.append("single-photon absorption")
        elif photons_needed <= 5:
            processes.append(f"{photons_needed}-photon absorption")
        else:
            processes.append("tunneling ionization likely above MPA")

    # ── Thermal effects ──
    if material.absorption_coeff_cm > 0:
        processes.append("linear absorption → heating")
        depth_cm = 1.0 / material.absorption_coeff_cm
        metrics["penetration_depth_um"] = depth_cm * 1e4

    # ── Regime classification ──
    regime = "linear"

    # Check for nonlinear Kerr
    delta_n = metrics.get("delta_n", 0)
    if delta_n > 1e-6:
        regime = "nonlinear_kerr"
        processes.append("self-phase modulation")
        if delta_n > 1e-4:
            processes.append("self-focusing")
            warnings.append("Strong self-focusing expected — risk of filamentation")

    # Check for multiphoton / ablation
    damage_ratio = metrics.get("damage_ratio", 0)
    if damage_ratio > 1.0:
        if laser.pulse_width_s > 0 and laser.pulse_width_s < 1e-12:
            regime = "multiphoton"
            processes.append("ultrafast ablation (non-thermal)")
        elif damage_ratio > 10:
            regime = "plasma"
            processes.append("plasma formation")
            processes.append("shock wave generation")
        else:
            regime = "ablation"
            processes.append("thermal ablation")
            if material.boiling_point_k > 0:
                processes.append("material vaporization")

    # Check for plasma at extreme irradiance
    if irradiance > 1e12:
        if regime not in ("plasma",):
            regime = "plasma"
            processes.append("optical breakdown")

    return InteractionResult(
        regime=regime,
        dominant_processes=processes,
        warnings=warnings,
        metrics=metrics,
    )
