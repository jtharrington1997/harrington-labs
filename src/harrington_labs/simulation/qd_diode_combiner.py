"""QD-Doped Direct Diode + Beam Combining testbed engine.

Models a diode laser array where quantum dots replace conventional
QW active regions, combined with spectral beam combining (SBC),
coherent beam combining (CBC), and hybrid SBC+CBC architectures.
No Streamlit imports.

Key physics:
- QD size-dependent gain (reuses empirical sizing curves)
- QD diode L-I with inhomogeneous broadening and thermal effects
- Spectral beam combining: diffraction grating, channel spacing,
  spectral fill factor, angular dispersion
- Coherent beam combining: phase locking, Strehl ratio, tiled
  aperture far-field, piston/tip/tilt errors
- Hybrid: SBC groups of CBC sub-arrays
- Beam quality (M²) and brightness calculations
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from harrington_labs.domain import SimulationResult, C_M_S, H_J_S, K_B

# Reuse QD sizing from the fiber laser engine
from harrington_labs.simulation.qd_fiber_laser import (
    qd_bandgap_ev, qd_emission_wavelength_nm, qd_gain_bandwidth_nm,
    qd_emission_cross_section_cm2, _QD_BULK,
)


# ── Dataclass ──────────────────────────────────────────────────────────


@dataclass
class QDDiodeCombinerParams:
    """Parameters for the QD direct diode + beam combining testbed."""
    # QD active region
    qd_material: str = "InAs"
    qd_diameter_nm: float = 5.0
    qd_size_distribution_pct: float = 5.0
    qd_layers: int = 5                 # stacked QD layers in active region
    qd_areal_density_cm2: float = 5e10  # per layer, ~5×10¹⁰ cm⁻² typical
    qd_quantum_yield: float = 0.6

    # Single emitter
    emitter_width_um: float = 100.0     # stripe width
    emitter_length_mm: float = 2.0      # cavity length
    fast_axis_divergence_deg: float = 30.0
    slow_axis_divergence_deg: float = 8.0
    confinement_factor: float = 0.03    # optical confinement per QD layer
    internal_loss_cm: float = 2.0       # internal optical loss
    mirror_reflectivity_front: float = 0.05  # AR-coated front facet
    mirror_reflectivity_back: float = 0.95
    injection_efficiency: float = 0.85
    thermal_resistance_k_w: float = 5.0  # K/W for single emitter
    t0_k: float = 80.0                  # characteristic temperature (QDs: higher than QW)
    operating_current_a: float = 2.0

    # Array
    n_emitters: int = 19               # emitters per bar
    emitter_pitch_um: float = 500.0     # center-to-center
    n_bars: int = 1                     # bars in stack (for SBC)
    fill_factor: float = 0.2           # emitter width / pitch
    heatsink_temp_c: float = 25.0

    # Beam combining
    combining_method: str = "SBC"      # "SBC", "CBC", "Hybrid"

    # SBC parameters
    grating_efficiency: float = 0.93
    grating_lines_per_mm: float = 1200.0
    channel_spacing_nm: float = 2.0     # wavelength offset per emitter
    transform_lens_f_mm: float = 100.0

    # CBC parameters
    phase_error_rms_rad: float = 0.3    # residual phase error after locking
    tip_tilt_error_urad: float = 5.0    # pointing error per element
    fill_factor_cbc: float = 0.7        # aperture fill factor
    feedback_bandwidth_hz: float = 10000.0

    # Hybrid: n_cbc_groups of coherently-combined sub-arrays, then SBC'd
    n_cbc_per_group: int = 7
    n_sbc_groups: int = 3


# ── Single QD emitter physics ─────────────────────────────────────────


def qd_diode_li(params: QDDiodeCombinerParams, n_points: int = 200) -> dict:
    """Compute L-I curve for a single QD diode emitter.

    Models: modal gain from stacked QD layers, inhomogeneous broadening,
    temperature-dependent threshold and slope, thermal rollover.
    """
    eg_ev = qd_bandgap_ev(params.qd_material, params.qd_diameter_nm)
    lam_nm = qd_emission_wavelength_nm(params.qd_material, params.qd_diameter_nm)
    sigma_em = qd_emission_cross_section_cm2(params.qd_material, params.qd_diameter_nm)

    # Modal gain per QD layer
    # g_modal = Γ × N_QD × σ_em × (2f - 1), where f = inversion fraction
    # At threshold: g_modal × n_layers = α_i + α_mirror
    gamma_total = params.confinement_factor * params.qd_layers
    n_qd_per_cm2 = params.qd_areal_density_cm2 * params.qd_layers

    # Mirror loss
    l_cm = params.emitter_length_mm * 0.1
    alpha_mirror = -math.log(params.mirror_reflectivity_front * params.mirror_reflectivity_back) / (2 * l_cm)
    total_cavity_loss = params.internal_loss_cm + alpha_mirror

    # Material gain at full inversion
    g_max = gamma_total * params.qd_areal_density_cm2 * sigma_em  # cm⁻¹ per layer, summed
    g_max_total = g_max * params.qd_layers

    # Threshold inversion fraction
    f_th = 0.5 * (1.0 + total_cavity_loss / g_max_total) if g_max_total > 0 else 1.0
    f_th = min(f_th, 0.99)

    # Threshold current density
    # J_th = q × N_QD × f_th / (τ_rad × η_inj)
    tau_rad_s = 1e-9  # ~1 ns for QD spontaneous emission in a diode
    e_charge = 1.602e-19
    j_th_a_cm2 = e_charge * n_qd_per_cm2 * f_th / (tau_rad_s * params.injection_efficiency)

    # Convert to threshold current
    stripe_area_cm2 = params.emitter_width_um * 1e-4 * params.emitter_length_mm * 0.1
    i_th = j_th_a_cm2 * stripe_area_cm2

    # Slope efficiency
    # η_slope = η_inj × (α_mirror / total_cavity_loss) × (hν / q) × η_rad
    photon_energy_ev = 1240.0 / lam_nm if lam_nm > 0 else 1.0
    eta_d = params.injection_efficiency * (alpha_mirror / total_cavity_loss) if total_cavity_loss > 0 else 0
    eta_slope_w_a = eta_d * photon_energy_ev * params.qd_quantum_yield

    # T0 for QDs is typically higher than QW (less thermal sensitivity)
    t_hs = params.heatsink_temp_c + 273.15
    current = np.linspace(0, params.operating_current_a * 2, n_points)
    power = np.zeros(n_points)
    voltage = np.zeros(n_points)
    junction_temp = np.zeros(n_points)
    efficiency = np.zeros(n_points)

    v_photon = photon_energy_ev
    for i, I in enumerate(current):
        t_j = t_hs
        for _ in range(5):
            i_th_t = i_th * math.exp((t_j - t_hs) / params.t0_k)
            eta_s_t = eta_slope_w_a * math.exp(-(t_j - t_hs) / (params.t0_k * 3))
            p_out = max(0, eta_s_t * (I - i_th_t)) if I > i_th_t else 0.0
            r_s = 0.02
            v_diode = v_photon + 0.4 + r_s * I
            p_elec = v_diode * I
            p_diss = max(p_elec - p_out, 0)
            t_j = t_hs + p_diss * params.thermal_resistance_k_w

        power[i] = p_out
        voltage[i] = v_diode
        junction_temp[i] = t_j - 273.15
        efficiency[i] = power[i] / (voltage[i] * I) if I > 0 else 0.0

    # Single emitter output at operating current
    idx_op = np.searchsorted(current, params.operating_current_a)
    if idx_op >= n_points:
        idx_op = n_points - 1
    p_single = float(power[idx_op])

    return {
        "current_a": current,
        "power_w": power,
        "voltage_v": voltage,
        "efficiency": efficiency,
        "junction_temp_c": junction_temp,
        "threshold_current_a": float(i_th),
        "slope_efficiency_w_a": float(eta_slope_w_a),
        "single_emitter_power_w": p_single,
        "emission_nm": lam_nm,
        "modal_gain_cm": g_max_total,
        "cavity_loss_cm": total_cavity_loss,
    }


# ── Beam combining engines ─────────────────────────────────────────────


def spectral_beam_combine(
    n_emitters: int,
    per_emitter_power_w: float,
    emission_nm: float,
    gain_bandwidth_nm: float,
    channel_spacing_nm: float,
    grating_efficiency: float,
    grating_lines_per_mm: float,
    fast_div_deg: float,
    slow_div_deg: float,
    fill_factor: float = 0.2,
) -> dict:
    """Spectral beam combining via diffraction grating.

    Each emitter operates at a slightly different wavelength within
    the QD gain bandwidth. The grating overlaps all beams spatially.
    """
    # How many emitters fit within the gain bandwidth?
    max_emitters_bw = int(gain_bandwidth_nm / channel_spacing_nm) if channel_spacing_nm > 0 else n_emitters
    usable_emitters = min(n_emitters, max_emitters_bw)

    if usable_emitters < n_emitters:
        bw_limited = True
    else:
        bw_limited = False

    raw_power = usable_emitters * per_emitter_power_w
    total_spectral_width_nm = usable_emitters * channel_spacing_nm

    # Grating diffraction efficiency
    eta_grating = grating_efficiency

    # Spectral fill factor (fraction of channel occupied by emission linewidth)
    emitter_linewidth_nm = max(0.5, channel_spacing_nm * 0.3)  # locked linewidth
    spectral_fill = emitter_linewidth_nm / channel_spacing_nm
    spectral_fill = min(spectral_fill, 0.95)

    # Angular dispersion crosstalk loss
    # Higher channel spacing → less crosstalk
    crosstalk_loss = max(0.0, 1.0 - 0.01 * usable_emitters / max(max_emitters_bw, 1))

    combined_power = raw_power * eta_grating * crosstalk_loss

    # Output beam quality
    # SBC preserves single-emitter M² in slow axis, fast axis unchanged
    m2_fast = 1.0  # single-mode fast axis after FAC
    m2_slow = fast_div_deg / slow_div_deg  # approximation

    # Brightness
    bpp_fast = emission_nm * 1e-6 / (math.pi * m2_fast)  # mm·mrad
    bpp_slow = emission_nm * 1e-6 / (math.pi * m2_slow)
    brightness = combined_power / (bpp_fast * bpp_slow * math.pi**2) if bpp_fast > 0 and bpp_slow > 0 else 0

    # Spectrum: each emitter as a narrow line
    lam_start = emission_nm - total_spectral_width_nm / 2
    channel_wavelengths = np.array([lam_start + i * channel_spacing_nm for i in range(usable_emitters)])
    wl_axis = np.linspace(emission_nm - gain_bandwidth_nm, emission_nm + gain_bandwidth_nm, 512)
    spectrum = np.zeros_like(wl_axis)
    for lam_c in channel_wavelengths:
        sigma = emitter_linewidth_nm / (2 * math.sqrt(2 * math.log(2)))
        spectrum += per_emitter_power_w * np.exp(-((wl_axis - lam_c)**2) / (2 * sigma**2))

    return {
        "method": "SBC",
        "usable_emitters": usable_emitters,
        "bw_limited": bw_limited,
        "raw_power_w": raw_power,
        "combined_power_w": combined_power,
        "combining_efficiency": combined_power / raw_power if raw_power > 0 else 0,
        "total_spectral_width_nm": total_spectral_width_nm,
        "m2_fast": m2_fast,
        "m2_slow": m2_slow,
        "brightness_w_mm2_mrad2": brightness,
        "channel_wavelengths_nm": channel_wavelengths,
        "spectrum_nm": wl_axis,
        "spectrum_power": spectrum,
    }


def coherent_beam_combine(
    n_emitters: int,
    per_emitter_power_w: float,
    emission_nm: float,
    phase_error_rms_rad: float,
    tip_tilt_error_urad: float,
    fill_factor: float,
    emitter_pitch_um: float,
    fast_div_deg: float,
    slow_div_deg: float,
) -> dict:
    """Coherent beam combining via active phase locking.

    All emitters operate at the same wavelength. Phase control
    overlaps beams constructively. Strehl ratio quantifies quality.
    """
    # Strehl from phase errors: S ≈ exp(-σ²_φ)
    strehl_phase = math.exp(-phase_error_rms_rad**2)

    # Strehl from tip/tilt: S_tt ≈ exp(-2 * (θ_err / θ_diff)²)
    # Diffraction angle for single emitter
    theta_diff = emission_nm * 1e-9 / (emitter_pitch_um * 1e-6)  # rad
    strehl_tiptilt = math.exp(-2 * (tip_tilt_error_urad * 1e-6 / theta_diff)**2)

    # Fill factor Strehl (tiled aperture)
    strehl_fill = fill_factor**2

    strehl_total = strehl_phase * strehl_tiptilt * strehl_fill
    strehl_total = max(strehl_total, 0.01)

    # Combined power
    # Ideal CBC: N² × P_single (coherent addition in far field)
    # Real: N² × P_single × Strehl + (1-Strehl) × N × P_single (incoherent residual)
    coherent_power = n_emitters**2 * per_emitter_power_w * strehl_total
    incoherent_residual = n_emitters * per_emitter_power_w * (1 - strehl_total)
    total_on_axis = coherent_power  # peak irradiance equivalent

    # But total power is conserved: combined_power = N × P_single
    # The advantage of CBC is brightness, not power
    combined_power = n_emitters * per_emitter_power_w
    power_in_bucket = combined_power * strehl_total

    # Effective aperture
    array_width_mm = n_emitters * emitter_pitch_um * 1e-3
    effective_aperture_mm = array_width_mm * math.sqrt(fill_factor)

    # M² for coherently combined beam
    m2_combined = 1.0 / math.sqrt(strehl_total) if strehl_total > 0 else 100
    m2_combined = min(m2_combined, 50)

    # Brightness gain over single emitter
    brightness_gain = n_emitters * strehl_total

    # Far-field pattern (1D, slow axis)
    n_ff = 512
    theta_max = 5 * theta_diff * 1e6  # µrad
    theta = np.linspace(-theta_max, theta_max, n_ff)  # µrad
    theta_rad = theta * 1e-6

    # Array factor for tiled aperture
    pitch_m = emitter_pitch_um * 1e-6
    k = 2 * math.pi / (emission_nm * 1e-9)
    array_factor = np.zeros(n_ff)
    for j in range(n_emitters):
        x_j = (j - (n_emitters - 1) / 2) * pitch_m
        phase_err = np.random.default_rng(42 + j).normal(0, phase_error_rms_rad)
        array_factor += np.cos(k * x_j * np.sin(theta_rad) + phase_err)
    array_factor = (array_factor / n_emitters)**2

    # Single element envelope
    w_emitter = emitter_pitch_um * 1e-6 * math.sqrt(fill_factor) / 2
    envelope = np.exp(-2 * (theta_rad / (emission_nm * 1e-9 / (math.pi * w_emitter)))**2)
    far_field = array_factor * envelope

    return {
        "method": "CBC",
        "n_emitters": n_emitters,
        "combined_power_w": combined_power,
        "power_in_bucket_w": power_in_bucket,
        "strehl_phase": strehl_phase,
        "strehl_tiptilt": strehl_tiptilt,
        "strehl_fill": strehl_fill,
        "strehl_total": strehl_total,
        "brightness_gain": brightness_gain,
        "m2_combined": m2_combined,
        "effective_aperture_mm": effective_aperture_mm,
        "combining_efficiency": strehl_total,
        "far_field_angle_urad": theta,
        "far_field_intensity": far_field,
    }


def hybrid_beam_combine(params: QDDiodeCombinerParams, li_data: dict) -> dict:
    """Hybrid SBC + CBC architecture.

    Multiple CBC sub-arrays, each phase-locked internally,
    then spectrally combined via a grating. Maximizes both
    brightness (CBC) and power scaling (SBC).
    """
    n_cbc = params.n_cbc_per_group
    n_sbc = params.n_sbc_groups
    n_total = n_cbc * n_sbc
    p_single = li_data["single_emitter_power_w"]
    lam_nm = li_data["emission_nm"]
    gain_bw = qd_gain_bandwidth_nm(params.qd_material, params.qd_diameter_nm, params.qd_size_distribution_pct)

    # CBC within each group
    cbc_results = []
    for g in range(n_sbc):
        cbc = coherent_beam_combine(
            n_emitters=n_cbc,
            per_emitter_power_w=p_single,
            emission_nm=lam_nm + g * params.channel_spacing_nm * n_cbc,
            phase_error_rms_rad=params.phase_error_rms_rad,
            tip_tilt_error_urad=params.tip_tilt_error_urad,
            fill_factor=params.fill_factor_cbc,
            emitter_pitch_um=params.emitter_pitch_um,
            fast_div_deg=params.fast_axis_divergence_deg,
            slow_div_deg=params.slow_axis_divergence_deg,
        )
        cbc_results.append(cbc)

    # SBC of the CBC sub-arrays
    per_group_power = n_cbc * p_single
    sbc = spectral_beam_combine(
        n_emitters=n_sbc,
        per_emitter_power_w=per_group_power,
        emission_nm=lam_nm,
        gain_bandwidth_nm=gain_bw,
        channel_spacing_nm=params.channel_spacing_nm * n_cbc,
        grating_efficiency=params.grating_efficiency,
        grating_lines_per_mm=params.grating_lines_per_mm,
        fast_div_deg=params.fast_axis_divergence_deg,
        slow_div_deg=params.slow_axis_divergence_deg,
    )

    avg_strehl = float(np.mean([c["strehl_total"] for c in cbc_results]))
    hybrid_brightness_gain = avg_strehl * n_cbc * n_sbc

    return {
        "method": "Hybrid",
        "n_cbc_per_group": n_cbc,
        "n_sbc_groups": n_sbc,
        "n_total_emitters": n_total,
        "cbc_strehl_avg": avg_strehl,
        "sbc_efficiency": sbc["combining_efficiency"],
        "combined_power_w": sbc["combined_power_w"],
        "total_raw_power_w": n_total * p_single,
        "overall_efficiency": sbc["combined_power_w"] / (n_total * p_single) if p_single > 0 else 0,
        "brightness_gain": hybrid_brightness_gain,
        "sbc_result": sbc,
        "cbc_results": cbc_results,
    }


# ── Main simulation ────────────────────────────────────────────────────


def simulate_qd_diode_combiner(params: QDDiodeCombinerParams) -> SimulationResult:
    """Run the full QD direct diode + beam combining testbed."""
    warnings = []

    # QD properties
    eg_ev = qd_bandgap_ev(params.qd_material, params.qd_diameter_nm)
    lam_nm = qd_emission_wavelength_nm(params.qd_material, params.qd_diameter_nm)
    gain_bw = qd_gain_bandwidth_nm(params.qd_material, params.qd_diameter_nm, params.qd_size_distribution_pct)
    sigma_em = qd_emission_cross_section_cm2(params.qd_material, params.qd_diameter_nm)

    # Single emitter L-I
    li = qd_diode_li(params)

    if li["single_emitter_power_w"] <= 0:
        warnings.append("Single emitter below threshold at operating current")

    # Array raw power
    n_total = params.n_emitters * params.n_bars
    raw_array_power = n_total * li["single_emitter_power_w"]

    # Beam combining
    if params.combining_method == "SBC":
        combining = spectral_beam_combine(
            n_emitters=n_total,
            per_emitter_power_w=li["single_emitter_power_w"],
            emission_nm=lam_nm,
            gain_bandwidth_nm=gain_bw,
            channel_spacing_nm=params.channel_spacing_nm,
            grating_efficiency=params.grating_efficiency,
            grating_lines_per_mm=params.grating_lines_per_mm,
            fast_div_deg=params.fast_axis_divergence_deg,
            slow_div_deg=params.slow_axis_divergence_deg,
            fill_factor=params.fill_factor,
        )
        if combining.get("bw_limited"):
            warnings.append(
                f"QD gain bandwidth ({gain_bw:.0f} nm) limits SBC to {combining['usable_emitters']} / {n_total} emitters"
            )

    elif params.combining_method == "CBC":
        combining = coherent_beam_combine(
            n_emitters=n_total,
            per_emitter_power_w=li["single_emitter_power_w"],
            emission_nm=lam_nm,
            phase_error_rms_rad=params.phase_error_rms_rad,
            tip_tilt_error_urad=params.tip_tilt_error_urad,
            fill_factor=params.fill_factor_cbc,
            emitter_pitch_um=params.emitter_pitch_um,
            fast_div_deg=params.fast_axis_divergence_deg,
            slow_div_deg=params.slow_axis_divergence_deg,
        )

    else:  # Hybrid
        combining = hybrid_beam_combine(params, li)

    # QD advantage metrics: compare to hypothetical QW version
    # QD T0 is typically 2-3× higher than QW
    qd_advantage = {
        "gain_bandwidth_nm": gain_bw,
        "gain_bandwidth_vs_qw": gain_bw / 5.0,  # QW typically ~3-5 nm
        "t0_advantage": params.t0_k / 40.0,  # QW T0 ~ 40-60K
        "size_tunable_range_nm": (
            qd_emission_wavelength_nm(params.qd_material, 2.0),
            qd_emission_wavelength_nm(params.qd_material, 15.0),
        ),
    }

    # Size sweep for this material
    diameters = np.linspace(2.0, 15.0, 100)
    emission_sweep = np.array([qd_emission_wavelength_nm(params.qd_material, d) for d in diameters])
    bandwidth_sweep = np.array([qd_gain_bandwidth_nm(params.qd_material, d, params.qd_size_distribution_pct) for d in diameters])

    return SimulationResult(
        name="QD Direct Diode + Beam Combining",
        data={
            "qd": {
                "material": params.qd_material,
                "diameter_nm": params.qd_diameter_nm,
                "bandgap_ev": eg_ev,
                "emission_nm": lam_nm,
                "gain_bandwidth_nm": gain_bw,
                "sigma_em_cm2": sigma_em,
            },
            "single_emitter": li,
            "array": {
                "n_emitters_total": n_total,
                "raw_power_w": raw_array_power,
                "fill_factor": params.fill_factor,
            },
            "combining": combining,
            "qd_advantage": qd_advantage,
            "size_sweep": {
                "diameter_nm": diameters,
                "emission_nm": emission_sweep,
                "gain_bandwidth_nm": bandwidth_sweep,
            },
        },
        warnings=warnings,
    )
