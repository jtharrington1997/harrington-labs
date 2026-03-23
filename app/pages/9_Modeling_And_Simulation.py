from __future__ import annotations

import math
import tempfile
import urllib.parse
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from harrington_labs.ui import render_header
from harrington_labs.ui import lab_panel, make_figure, show_figure, COLORS
from harrington_labs.lmi.ui.formatting import (
    fmt_absorption_cm_inv,
    fmt_ev,
    fmt_fluence_j_cm2,
    fmt_frequency_hz,
    fmt_irradiance_w_cm2,
    fmt_length_m,
    fmt_n2_cm2_w,
    fmt_power_w,
    fmt_refractive_index,
    fmt_temp_k,
    fmt_wavelength_nm,
)
from harrington_labs.lmi.domain.lasers import LaserSource, SpatialMode, all_lasers
from harrington_labs.lmi.domain.materials import all_materials, Material
from harrington_labs.lmi.domain.interactions import classify_regime
from harrington_labs.lmi.io.campaign_import import parse_sheet
from harrington_labs.lmi.io.exporters import export_plot_bundle
from harrington_labs.lmi.domain.plot_spec import PlotSpec, SeriesSpec
from harrington_labs.lmi.simulation.beam_propagation import (
    BeamParams,
    compute_focus,
    propagate_in_material,
)
from harrington_labs.lmi.simulation.thermal import thermal_analysis, two_temperature_model
from harrington_labs.lmi.simulation.nonlinear import nonlinear_analysis
from harrington_labs.lmi.simulation.custom_models import MODELS_DIR, load_models, run_model

POLARIZATION_TYPES = ("Linear", "Circular", "Elliptical", "Unpolarized")

REGIME_COLORS = {
    "linear": "#3FB950",
    "nonlinear_kerr": "#D29922",
    "multiphoton": "#E63946",
    "ablation": "#FF6B6B",
    "plasma": "#FF0000",
}


def _qp_str(qp, key: str, default: str) -> str:
    value = qp.get(key)
    return default if value is None else str(value)


def _qp_float(qp, key: str, default: float) -> float:
    value = qp.get(key)
    if value is None or value == "":
        return float(default)
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _first_attr(obj: Any, *names: str, default: Any = 0.0) -> Any:
    for name in names:
        if hasattr(obj, name):
            value = getattr(obj, name)
            if value is not None:
                return value
    return default


def _n2_correction(pol_type: str, ellipticity_deg: float = 0.0) -> float:
    if pol_type == "Linear":
        return 1.0
    if pol_type == "Circular":
        return 2.0 / 3.0
    if pol_type == "Elliptical":
        chi_rad = math.radians(ellipticity_deg)
        return 1.0 - (1.0 / 3.0) * math.sin(2.0 * chi_rad) ** 2
    return 1.0


def _polarization_label(
    pol_type: str,
    handedness: str = "",
    angle_deg: float = 0.0,
    ellipticity_deg: float = 0.0,
) -> str:
    if pol_type == "Linear":
        return f"Linear @ {angle_deg:.0f}°"
    if pol_type == "Circular":
        return "LCP" if handedness == "Left" else "RCP"
    if pol_type == "Elliptical":
        handed = "L" if handedness == "Left" else "R"
        return f"Elliptical ({handed}, χ={ellipticity_deg:.1f}°)"
    return "Unpolarized"


def _material_n(material, wavelength_nm: float) -> float:
    if hasattr(material, "get_n"):
        try:
            return float(material.get_n(wavelength_nm))
        except Exception:
            pass
    return float(_first_attr(material, "refractive_index", "n", default=1.0))


def _material_alpha_cm(material, wavelength_nm: float) -> float:
    if hasattr(material, "get_alpha"):
        try:
            return float(material.get_alpha(wavelength_nm))
        except Exception:
            pass
    return float(_first_attr(material, "absorption_coeff_cm", "alpha_cm", default=0.0))


def _material_n2_cm2_per_w(material, wavelength_nm: float) -> float:
    if hasattr(material, "get_n2"):
        try:
            return float(material.get_n2(wavelength_nm))
        except Exception:
            pass
    return float(_first_attr(material, "n2_cm2_w", "n2_cm2_per_w", default=0.0) or 0.0)


def _material_beta_cm_per_w(material) -> float:
    return float(
        _first_attr(
            material,
            "two_photon_abs_cm_w",
            "beta_cm_per_w",
            default=0.0,
        )
        or 0.0
    )


def _damage_threshold_j_cm2(material) -> float:
    return float(
        _first_attr(
            material,
            "damage_threshold_j_cm2",
            "lidt_j_cm2",
            default=0.0,
        )
        or 0.0
    )


def _propagate_material(
    *,
    focus,
    beam,
    n_material: float,
    alpha_cm: float,
    thickness_m: float,
    surface_position_m: float,
    beta_cm_per_w: float,
    n2_cm2_per_w: float,
):
    try:
        return propagate_in_material(
            focus=focus,
            beam=beam,
            n_material=n_material,
            alpha_cm=alpha_cm,
            thickness_m=thickness_m,
            surface_position_m=surface_position_m,
            beta_cm_per_w=beta_cm_per_w,
            n2_cm2_per_w=n2_cm2_per_w,
        )
    except TypeError:
        return propagate_in_material(
            focus=focus,
            beam=beam,
            n_material=n_material,
            alpha_cm=alpha_cm,
            thickness_m=thickness_m,
            surface_position_m=surface_position_m,
        )


def _transmission_fraction(prop) -> float:
    value = getattr(prop, "transmission_fraction", None)
    if value is not None:
        return float(value)
    absorbed = np.asarray(getattr(prop, "absorbed_fraction", np.array([0.0])), dtype=float)
    return float(1.0 - absorbed[-1]) if absorbed.size else 1.0


def _peak_irradiance_in_prop(prop) -> float:
    value = getattr(prop, "peak_irradiance_w_cm2", None)
    if value is not None:
        return float(value)
    irr = np.asarray(getattr(prop, "irradiance_z_w_cm2", np.array([0.0])), dtype=float)
    return float(np.max(irr)) if irr.size else 0.0


def _simulate_open_aperture_zscan(
    *,
    focus,
    beam,
    n_material: float,
    alpha_cm: float,
    thickness_m: float,
    z_positions_m: np.ndarray,
    beta_cm_per_w: float,
    n2_cm2_per_w: float,
) -> dict[str, np.ndarray]:
    peak_irradiance_w_cm2 = np.zeros_like(z_positions_m, dtype=float)
    transmission_fraction = np.zeros_like(z_positions_m, dtype=float)
    peak_radius_m = np.zeros_like(z_positions_m, dtype=float)
    b_integral_rad = np.zeros_like(z_positions_m, dtype=float)
    self_focusing_ratio = np.zeros_like(z_positions_m, dtype=float)

    for i, z_center_m in enumerate(z_positions_m):
        surface_pos_m = z_center_m - 0.5 * thickness_m

        prop = _propagate_material(
            focus=focus,
            beam=beam,
            n_material=n_material,
            alpha_cm=alpha_cm,
            thickness_m=thickness_m,
            surface_position_m=surface_pos_m,
            beta_cm_per_w=beta_cm_per_w,
            n2_cm2_per_w=n2_cm2_per_w,
        )

        irr = np.asarray(getattr(prop, "irradiance_z_w_cm2", np.array([0.0])), dtype=float)
        w_z = np.asarray(getattr(prop, "w_z_m", np.array([0.0])), dtype=float)

        transmission_fraction[i] = _transmission_fraction(prop)
        peak_irradiance_w_cm2[i] = _peak_irradiance_in_prop(prop)
        b_integral_rad[i] = float(getattr(prop, "b_integral_rad", 0.0) or 0.0)
        self_focusing_ratio[i] = float(getattr(prop, "self_focusing_ratio", 0.0) or 0.0)

        if irr.size > 0 and w_z.size > 0:
            peak_idx = int(np.argmax(irr))
            peak_radius_m[i] = float(w_z[min(peak_idx, w_z.size - 1)])

    return {
        "peak_irradiance_w_cm2": peak_irradiance_w_cm2,
        "transmission_fraction": transmission_fraction,
        "peak_radius_m": peak_radius_m,
        "b_integral_rad": b_integral_rad,
        "self_focusing_ratio": self_focusing_ratio,
    }


def _apply_pub_layout(fig, *, height: int = 420, showlegend: bool = True) -> None:
    fig.update_layout(
        template="simple_white",
        height=height,
        paper_bgcolor="white",
        plot_bgcolor="white",
        font=dict(size=15),
        margin=dict(l=24, r=24, t=44, b=20),
        showlegend=showlegend,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1.0,
            bgcolor="rgba(255,255,255,0.9)",
            bordercolor="rgba(0,0,0,0.15)",
            borderwidth=1,
        ),
    )
    fig.update_xaxes(
        showline=True,
        linewidth=1.1,
        linecolor="black",
        mirror=True,
        ticks="outside",
        tickwidth=1.1,
        ticklen=6,
        showgrid=True,
        gridcolor="rgba(0,0,0,0.10)",
        zeroline=False,
        exponentformat="power",
        showexponent="all",
    )
    fig.update_yaxes(
        showline=True,
        linewidth=1.1,
        linecolor="black",
        mirror=True,
        ticks="outside",
        tickwidth=1.1,
        ticklen=6,
        showgrid=True,
        gridcolor="rgba(0,0,0,0.10)",
        zeroline=False,
        exponentformat="power",
        showexponent="all",
    )


def _laser_override_defaults(base_laser) -> dict[str, float]:
    pulse_energy_uj = (
        (base_laser.power_w / base_laser.rep_rate_hz * 1e6)
        if (not base_laser.is_cw and base_laser.rep_rate_hz > 0)
        else 0.0
    )

    pulse_width_s = float(base_laser.pulse_width_s)
    if pulse_width_s <= 0:
        pulse_width_s = 170e-15

    rep_rate_hz = float(base_laser.rep_rate_hz)
    if rep_rate_hz <= 0:
        rep_rate_hz = 10e3

    beam_diameter_mm = float(base_laser.beam_diameter_mm)
    if beam_diameter_mm <= 0:
        beam_diameter_mm = 5.0

    wavelength_nm = float(base_laser.wavelength_nm)
    if wavelength_nm <= 0:
        wavelength_nm = 1030.0

    m2 = float(base_laser.m_squared)
    if m2 < 1.0:
        m2 = 1.0

    return {
        "wavelength_nm": wavelength_nm,
        "power_w": float(base_laser.power_w),
        "pulse_energy_uj": float(pulse_energy_uj),
        "beam_diameter_mm": beam_diameter_mm,
        "pulse_width_s": pulse_width_s,
        "rep_rate_hz": rep_rate_hz,
        "m2": m2,
    }


st.set_page_config(page_title="Modeling & Simulation", layout="wide")
render_header("Modeling & Simulation", "Beam propagation • Nonlinear optics • z-Scan • Thermal analysis")

qp = st.query_params
lasers = all_lasers()
materials = all_materials()

with lab_panel():
    st.subheader("Modeling & Simulation")
    st.caption(
        "Unified workspace for interaction analysis, beam propagation, z-scan preview, "
        "thermal / nonlinear digital twin studies, campaign overlay, and export tools."
    )

with lab_panel():
    st.subheader("Shared Source / Material State")
    c1, c2 = st.columns(2)

    laser_names = [l.name for l in lasers]
    default_laser_name = _qp_str(qp, "laser", laser_names[0])
    laser_index = laser_names.index(default_laser_name) if default_laser_name in laser_names else 0

    material_names = [m.name for m in materials]
    default_material_name = _qp_str(qp, "material", material_names[0])
    material_index = material_names.index(default_material_name) if default_material_name in material_names else 0

    with c1:
        laser_name = st.selectbox("Laser Source", laser_names, index=laser_index)
    with c2:
        material_name = st.selectbox("Target Material", material_names, index=material_index)

base_laser = next(l for l in lasers if l.name == laser_name)
material: Material = next(m for m in materials if m.name == material_name)

override_key = f"workspace_override_state::{base_laser.name}"
if "workspace_override_active_key" not in st.session_state:
    st.session_state.workspace_override_active_key = override_key

if (
    "workspace_override_state" not in st.session_state
    or st.session_state.workspace_override_active_key != override_key
):
    st.session_state.workspace_override_state = _laser_override_defaults(base_laser)
    st.session_state.workspace_override_active_key = override_key

ovs = st.session_state.workspace_override_state

with st.expander("Override laser parameters for this workspace"):
    with st.form("workspace_override_form"):
        r1c1, r1c2, r1c3 = st.columns(3)
        r2c1, r2c2, r2c3 = st.columns(3)

        with r1c1:
            override_wl_nm = st.number_input(
                "Wavelength (nm)",
                min_value=100.0,
                max_value=20000.0,
                value=float(ovs["wavelength_nm"]),
                format="%.6g",
            )

        with r1c2:
            if base_laser.is_cw:
                override_power_w = st.number_input(
                    "Power (W)",
                    min_value=0.0,
                    max_value=1e6,
                    value=float(ovs["power_w"]),
                    format="%.6g",
                )
                override_energy_uj = 0.0
            else:
                override_energy_uj = st.number_input(
                    "Pulse Energy (µJ)",
                    min_value=0.0,
                    max_value=1e6,
                    value=float(ovs["pulse_energy_uj"]),
                    format="%.6g",
                )

        with r1c3:
            override_beam_mm = st.number_input(
                "Beam Diameter (mm, 1/e²)",
                min_value=0.001,
                max_value=100.0,
                value=float(ovs["beam_diameter_mm"]),
                format="%.6g",
            )

        with r2c1:
            override_pulse_fs = st.number_input(
                "Pulse Width (fs)",
                min_value=1.0,
                max_value=1e9,
                value=float(ovs["pulse_width_s"]) * 1e15 if float(ovs["pulse_width_s"]) > 0 else 170.0,
                format="%.6g",
            )

        with r2c2:
            override_rep_rate_khz = st.number_input(
                "Rep Rate (kHz)",
                min_value=0.001,
                max_value=1e9,
                value=float(ovs["rep_rate_hz"]) / 1e3 if float(ovs["rep_rate_hz"]) > 0 else 10.0,
                format="%.6g",
            )

        with r2c3:
            override_m2 = st.number_input(
                "M²",
                min_value=1.0,
                max_value=50.0,
                value=float(ovs["m2"]),
                step=0.1,
                format="%.3f",
            )

        a1, a2 = st.columns(2)
        with a1:
            apply_overrides = st.form_submit_button("Apply overrides", width="stretch")
        with a2:
            reset_overrides = st.form_submit_button("Reset defaults", width="stretch")

    if reset_overrides:
        st.session_state.workspace_override_state = _laser_override_defaults(base_laser)
        st.rerun()

    if apply_overrides:
        ovs["wavelength_nm"] = float(override_wl_nm)
        ovs["beam_diameter_mm"] = float(override_beam_mm)
        ovs["pulse_width_s"] = float(override_pulse_fs) * 1e-15
        ovs["rep_rate_hz"] = float(override_rep_rate_khz) * 1e3
        ovs["m2"] = float(override_m2)

        if base_laser.is_cw:
            ovs["power_w"] = float(override_power_w)
            ovs["pulse_energy_uj"] = 0.0
        else:
            ovs["pulse_energy_uj"] = float(override_energy_uj)
            ovs["power_w"] = float(override_energy_uj) * 1e-6 * (float(override_rep_rate_khz) * 1e3)

        st.rerun()

with lab_panel():
    st.subheader("Polarization / Focusing / Sample")
    g1, g2, g3 = st.columns(3)

    with g1:
        qp_pol_type = _qp_str(qp, "pol_type", "Linear")
        pol_index = POLARIZATION_TYPES.index(qp_pol_type) if qp_pol_type in POLARIZATION_TYPES else 0
        pol_type = st.selectbox("Polarization Type", POLARIZATION_TYPES, index=pol_index)

        if pol_type == "Linear":
            pol_angle = st.number_input("Angle (°)", 0.0, 180.0, _qp_float(qp, "pol_angle", 0.0), step=1.0, format="%.0f")
            pol_handedness = ""
            pol_ellipticity = 0.0
        elif pol_type == "Circular":
            pol_handedness = st.selectbox("Handedness", ("Left", "Right"), index=0 if _qp_str(qp, "pol_hand", "Left") == "Left" else 1)
            pol_angle = 0.0
            pol_ellipticity = 45.0
        elif pol_type == "Elliptical":
            pol_handedness = st.selectbox("Handedness", ("Left", "Right"), index=0 if _qp_str(qp, "pol_hand", "Left") == "Left" else 1)
            pol_angle = 0.0
            pol_ellipticity = st.number_input("Ellipticity χ (°)", 0.0, 45.0, _qp_float(qp, "pol_chi", 15.0), step=1.0, format="%.1f")
        else:
            pol_angle = 0.0
            pol_handedness = ""
            pol_ellipticity = 0.0

    with g2:
        focal_length_mm = st.number_input(
            "Focal Length (mm)",
            min_value=1.0,
            max_value=5000.0,
            value=_qp_float(qp, "focal", 10.0) * 10.0 if qp.get("focal") else 100.0,
            format="%.6g",
        )
        surface_offset_um = st.number_input(
            "Focus Offset from Surface (µm)",
            min_value=-5000.0,
            max_value=5000.0,
            value=_qp_float(qp, "offset", 0.0),
            format="%.6g",
        )

    with g3:
        thickness_mm = st.number_input(
            "Thickness (mm)",
            min_value=0.001,
            max_value=100.0,
            value=_qp_float(qp, "thickness", 0.5),
            format="%.6g",
        )
        t_ambient_k = st.number_input("Ambient Temperature (K)", 4.0, 1000.0, 300.0, step=10.0, format="%.6g")

pol_factor = _n2_correction(pol_type, pol_ellipticity)
pol_label = _polarization_label(pol_type, pol_handedness, pol_angle, pol_ellipticity)

laser = LaserSource(
    name=base_laser.name,
    wavelength_nm=float(ovs["wavelength_nm"]),
    power_w=float(ovs["power_w"]),
    rep_rate_hz=float(ovs["rep_rate_hz"]),
    pulse_width_s=float(ovs["pulse_width_s"]),
    beam_diameter_mm=float(ovs["beam_diameter_mm"]),
    m_squared=float(ovs["m2"]),
    gain_medium=getattr(base_laser, "gain_medium", None),
    polarization=pol_type.lower(),
)

beam = BeamParams(
    wavelength_m=laser.wavelength_nm * 1e-9,
    beam_diameter_1e2_m=laser.beam_diameter_mm * 1e-3,
    m_squared=laser.m_squared,
    pulse_energy_j=laser.pulse_energy_j if not laser.is_cw else 0.0,
    pulse_width_s=laser.pulse_width_s,
    rep_rate_hz=laser.rep_rate_hz,
    spatial_mode=getattr(laser, "spatial_mode", SpatialMode.TEM00.value) or SpatialMode.TEM00.value,
)

thickness_m = thickness_mm * 1e-3
focal_length_m = focal_length_mm * 1e-3
surface_offset_m = surface_offset_um * 1e-6

n_mat = _material_n(material, laser.wavelength_nm)
alpha_cm = _material_alpha_cm(material, laser.wavelength_nm)
n2_cm2_per_w = _material_n2_cm2_per_w(material, laser.wavelength_nm) * pol_factor
beta_cm_per_w = _material_beta_cm_per_w(material)
damage_threshold_j_cm2 = _damage_threshold_j_cm2(material)

focus = compute_focus(beam, focal_length_m, n_medium=1.0)
prop = _propagate_material(
    focus=focus,
    beam=beam,
    n_material=n_mat,
    alpha_cm=alpha_cm,
    thickness_m=thickness_m,
    surface_position_m=surface_offset_m,
    beta_cm_per_w=beta_cm_per_w,
    n2_cm2_per_w=n2_cm2_per_w,
)

result = classify_regime(laser, material)
if "delta_n" in result.metrics and pol_factor != 1.0:
    result.metrics["delta_n"] *= pol_factor
    result.metrics["delta_n_over_n"] = result.metrics["delta_n"] / n_mat if n_mat > 0 else 0.0
    if "b_integral_per_cm" in result.metrics:
        result.metrics["b_integral_per_cm"] *= pol_factor

with lab_panel():
    st.subheader("Shared Summary")
    color = REGIME_COLORS.get(result.regime, "#8B949E")
    st.markdown(
        (
            f'<div style="background:{color}18; border:2px solid {color}; '
            'border-radius:10px; padding:14px; text-align:center; margin-bottom:14px;">'
            f'<span style="color:{color}; font-size:1.45rem; font-weight:700;">'
            f'{result.regime.upper().replace("_", " ")}</span></div>'
        ),
        unsafe_allow_html=True,
    )

    s1, s2, s3, s4 = st.columns(4)
    s1.metric("Wavelength", fmt_wavelength_nm(laser.wavelength_nm))
    s2.metric("Peak Power", fmt_power_w(beam.peak_power_w))
    s3.metric("Avg Power", fmt_power_w(beam.avg_power_w))
    s4.metric("Rep Rate", fmt_frequency_hz(beam.rep_rate_hz))

    s5, s6, s7, s8 = st.columns(4)
    s5.metric("Fluence", fmt_fluence_j_cm2(result.metrics.get("fluence_j_cm2", 0.0)))
    s6.metric("Irradiance", fmt_irradiance_w_cm2(result.metrics.get("irradiance_w_cm2", 0.0)))
    s7.metric("Spot Radius w₀", fmt_length_m(focus.w0_m))
    s8.metric("Rayleigh Range zR", fmt_length_m(focus.rayleigh_range_m))

    s9, s10, s11, s12 = st.columns(4)
    s9.metric("n", fmt_refractive_index(n_mat))
    s10.metric("α", fmt_absorption_cm_inv(alpha_cm))
    s11.metric("n₂", fmt_n2_cm2_w(n2_cm2_per_w))
    s12.metric("β", f"{beta_cm_per_w:.4g} cm/W")

    if pol_type != "Linear" or pol_angle != 0:
        st.info(f"Polarization: {pol_label} — n₂ correction {pol_factor:.4g}×")

tabs = st.tabs(
    [
        "Overview / Regime",
        "Beam & Z-Scan",
        "Thermal & Nonlinear",
        "Campaign Overlay",
        "Export & Automation",
    ]
)

with tabs[0]:
    with lab_panel():
        st.subheader("Interaction Overview")
        if result.dominant_processes:
            st.markdown("**Dominant Processes**")
            for process in result.dominant_processes:
                st.markdown(f"- {process}")

        if result.warnings:
            st.markdown("**Warnings**")
            for warning in result.warnings:
                st.warning(warning)

        c1, c2, c3 = st.columns(3)
        c1.metric("Photon Energy", fmt_ev(result.metrics.get("photon_energy_ev", 0.0)))
        c2.metric("Photons for Bandgap", str(result.metrics.get("photons_for_bandgap", "n/a")))
        c3.metric("Transmission Through Slab", f"{100.0 * _transmission_fraction(prop):.4g}%")

with tabs[1]:
    with lab_panel():
        st.subheader("Focused Beam & Slab Preview")
        fig_prop = make_subplots(
            rows=1,
            cols=2,
            subplot_titles=("Peak Irradiance Through Slab", "1/e² Beam Radius Through Slab"),
        )

        z_depth_mm = np.asarray(prop.z_material_m) * 1e3
        fig_prop.add_trace(
            go.Scatter(x=z_depth_mm, y=np.asarray(prop.irradiance_z_w_cm2), mode="lines", line=dict(width=2)),
            row=1,
            col=1,
        )
        fig_prop.add_trace(
            go.Scatter(x=z_depth_mm, y=np.asarray(prop.w_z_m) * 1e6, mode="lines", line=dict(width=2)),
            row=1,
            col=2,
        )
        fig_prop.update_xaxes(title_text="Depth in sample (mm)", row=1, col=1)
        fig_prop.update_xaxes(title_text="Depth in sample (mm)", row=1, col=2)
        fig_prop.update_yaxes(title_text="Peak irradiance (W cm⁻²)", type="log", tickformat=".3e", row=1, col=1)
        fig_prop.update_yaxes(title_text="Beam radius, w(z) (µm)", tickformat=".3e", row=1, col=2)
        _apply_pub_layout(fig_prop, height=420, showlegend=False)
        st.plotly_chart(fig_prop, width="stretch")

    with lab_panel():
        st.subheader("Open-Aperture Z-Scan Preview")
        zc1, zc2 = st.columns(2)
        with zc1:
            zscan_range_mm = st.number_input("Scan Range (± mm)", min_value=0.1, max_value=100.0, value=5.0, format="%.6g")
        with zc2:
            zscan_points = st.number_input("Number of Points", min_value=51, max_value=1001, value=201, step=50)

        z_positions_m = np.linspace(-zscan_range_mm * 1e-3, zscan_range_mm * 1e-3, int(zscan_points))
        z_positions_mm = z_positions_m * 1e3

        zscan = _simulate_open_aperture_zscan(
            focus=focus,
            beam=beam,
            n_material=n_mat,
            alpha_cm=alpha_cm,
            thickness_m=thickness_m,
            z_positions_m=z_positions_m,
            beta_cm_per_w=beta_cm_per_w,
            n2_cm2_per_w=n2_cm2_per_w,
        )

        transmission_frac = zscan["transmission_fraction"]
        peak_irr_in_sample = zscan["peak_irradiance_w_cm2"]

        edge_n = min(10, max(1, len(transmission_frac) // 10))
        t_edges = (transmission_frac[:edge_n].mean() + transmission_frac[-edge_n:].mean()) / 2.0
        t_normalised = transmission_frac / t_edges if t_edges > 0 else transmission_frac

        fig_oa = go.Figure()
        fig_oa.add_trace(go.Scatter(x=z_positions_mm, y=t_normalised, mode="lines", line=dict(width=2)))
        fig_oa.add_hline(y=1.0, line_dash="dot", line_color="gray")
        fig_oa.add_vline(x=0.0, line_dash="dot", line_color="gray")
        fig_oa.update_layout(
            xaxis_title="Sample-center position, z (mm)",
            yaxis_title="Normalized transmission, T(z)/T₀",
        )
        fig_oa.update_yaxes(tickformat=".4f")
        _apply_pub_layout(fig_oa, height=420, showlegend=False)
        st.plotly_chart(fig_oa, width="stretch")

        fig_zirr = go.Figure()
        fig_zirr.add_trace(go.Scatter(x=z_positions_mm, y=peak_irr_in_sample, mode="lines", line=dict(width=2)))
        fig_zirr.add_vline(x=0.0, line_dash="dot", line_color="gray")
        fig_zirr.update_layout(
            xaxis_title="Sample-center position, z (mm)",
            yaxis_title="Peak irradiance in sample (W cm⁻²)",
        )
        fig_zirr.update_yaxes(type="log", tickformat=".3e")
        _apply_pub_layout(fig_zirr, height=420, showlegend=False)
        st.plotly_chart(fig_zirr, width="stretch")

with tabs[2]:
    photon_ev = 1240.0 / laser.wavelength_nm if laser.wavelength_nm > 0 else 0.0
    nl = nonlinear_analysis(
        wavelength_nm=laser.wavelength_nm,
        bandgap_ev=material.bandgap_ev,
        n2_cm2_w=n2_cm2_per_w,
        refractive_index=n_mat,
        alpha_linear_cm=alpha_cm,
        peak_power_w=beam.peak_power_w,
        irradiance_w_cm2=result.metrics.get("irradiance_w_cm2", 0.0),
        pulse_width_s=beam.pulse_width_s,
        beam_radius_m=focus.w0_m,
        thickness_m=thickness_m,
    )

    D_mat = material.get_thermal_diffusivity()
    therm = thermal_analysis(
        fluence_j_cm2=result.metrics.get("fluence_j_cm2", 0.0),
        pulse_width_s=beam.pulse_width_s,
        rep_rate_hz=beam.rep_rate_hz,
        spot_radius_m=focus.w0_m,
        alpha_cm=alpha_cm,
        thermal_conductivity_w_mk=material.thermal_conductivity_w_mk,
        density_kg_m3=material.density_kg_m3,
        specific_heat_j_kgk=material.specific_heat_j_kgk,
        thermal_diffusivity_m2_s=D_mat,
        melting_point_k=material.melting_point_k,
        t_ambient_k=t_ambient_k,
        n_pulses_max=500,
    )

    with lab_panel():
        st.subheader("Thermal Summary")
        t1, t2, t3, t4 = st.columns(4)
        t1.metric("Surface ΔT", fmt_temp_k(therm.delta_t_surface_k))
        t2.metric("Diffusion Length", fmt_length_m(therm.thermal_diffusion_length_m))
        t3.metric("Heat Confined?", "Yes" if therm.heat_confined else "No")
        t4.metric("Melt Threshold", fmt_fluence_j_cm2(therm.melt_threshold_fluence_j_cm2))

        fig_t = go.Figure()
        fig_t.add_trace(go.Scatter(x=therm.z_m * 1e6, y=therm.delta_t_z_k + t_ambient_k, mode="lines", line=dict(width=2)))
        fig_t.add_hline(y=material.melting_point_k, line_dash="dash", line_color="#b8860b")
        fig_t.update_layout(xaxis_title="Depth (µm)", yaxis_title="Temperature (K)")
        fig_t.update_yaxes(tickformat=".3e")
        _apply_pub_layout(fig_t, height=360, showlegend=False)
        st.plotly_chart(fig_t, width="stretch")

        fig_acc = go.Figure()
        fig_acc.add_trace(go.Scatter(x=therm.n_pulses, y=therm.t_surface_vs_pulses, mode="lines", line=dict(width=2)))
        fig_acc.add_hline(y=material.melting_point_k, line_dash="dash", line_color="#b8860b")
        fig_acc.update_layout(xaxis_title="Pulse Number", yaxis_title="Surface Temperature (K)")
        fig_acc.update_yaxes(tickformat=".3e")
        _apply_pub_layout(fig_acc, height=360, showlegend=False)
        st.plotly_chart(fig_acc, width="stretch")

    with lab_panel():
        st.subheader("Nonlinear Summary")
        n1, n2, n3, n4 = st.columns(4)
        n1.metric("MPA Order", f"{nl.mpa.photon_order}-photon")
        n2.metric("P / Pcritical", f"{nl.self_focusing.p_over_pcr:.4g}")
        n3.metric("B-integral", f"{nl.self_focusing.b_integral_rad:.4g} rad")
        n4.metric("Peak Δn", fmt_refractive_index(nl.self_focusing.delta_n_peak))

        fig_nl = make_subplots(rows=1, cols=2, subplot_titles=["Irradiance (with NL absorption)", "Kerr Δn(z)"])
        z_um = nl.z_m * 1e6
        fig_nl.add_trace(go.Scatter(x=z_um, y=nl.irradiance_z_w_cm2, mode="lines", line=dict(width=2)), row=1, col=1)
        fig_nl.add_trace(go.Scatter(x=z_um, y=nl.delta_n_z, mode="lines", line=dict(width=2)), row=1, col=2)
        fig_nl.update_xaxes(title_text="Depth (µm)", row=1, col=1)
        fig_nl.update_xaxes(title_text="Depth (µm)", row=1, col=2)
        fig_nl.update_yaxes(title_text="Irradiance (W cm⁻²)", type="log", tickformat=".3e", row=1, col=1)
        fig_nl.update_yaxes(title_text="Δn", tickformat=".3e", row=1, col=2)
        _apply_pub_layout(fig_nl, height=420, showlegend=False)
        st.plotly_chart(fig_nl, width="stretch")

        if material.electron_phonon_coupling_w_m3k > 0:
            ttm = two_temperature_model(
                fluence_j_cm2=result.metrics.get("fluence_j_cm2", 0.0),
                pulse_width_s=beam.pulse_width_s,
                alpha_cm=alpha_cm,
                electron_phonon_coupling_w_m3k=material.electron_phonon_coupling_w_m3k,
                density_kg_m3=material.density_kg_m3,
                specific_heat_j_kgk=material.specific_heat_j_kgk,
                t_ambient_k=t_ambient_k,
                t_max_ps=50.0,
            )
            fig_ttm = go.Figure()
            fig_ttm.add_trace(go.Scatter(x=ttm.t_ps, y=ttm.t_electron_k, mode="lines", line=dict(width=2), name="Electron"))
            fig_ttm.add_trace(go.Scatter(x=ttm.t_ps, y=ttm.t_lattice_k, mode="lines", line=dict(width=2), name="Lattice"))
            fig_ttm.add_hline(y=material.melting_point_k, line_dash="dash", line_color="#b8860b")
            fig_ttm.update_layout(xaxis_title="Time (ps)", yaxis_title="Temperature (K)")
            fig_ttm.update_yaxes(tickformat=".3e")
            _apply_pub_layout(fig_ttm, height=400, showlegend=True)
            st.plotly_chart(fig_ttm, width="stretch")

with tabs[3]:
    with lab_panel():
        st.subheader("Campaign Overlay")
        uploaded = st.file_uploader("Upload Excel campaign (.xlsx)", type=["xlsx"])

        if uploaded is None:
            st.info("Upload a campaign spreadsheet to compare experiment and model.")
        else:
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".xlsx")
            tmp.write(uploaded.read())
            tmp.close()

            xls = pd.ExcelFile(tmp.name)
            sheet = st.selectbox("Select sheet", xls.sheet_names)

            parsed_materials = parse_sheet(tmp.name, sheet)
            if not parsed_materials:
                st.warning("No materials detected in the selected sheet.")
            else:
                fig = go.Figure()

                for material_entry in parsed_materials:
                    d = material_entry["data"].copy()
                    fig.add_trace(
                        go.Scatter(
                            x=d["position_mm"],
                            y=d["absorption"],
                            mode="markers",
                            name=f"{material_entry['name']} (exp)",
                        )
                    )

                    z_vals_m = d["position_mm"].to_numpy(dtype=float) * 1e-3
                    sim_absorption = []

                    for z in z_vals_m:
                        prop_i = _propagate_material(
                            focus=focus,
                            beam=beam,
                            n_material=n_mat,
                            alpha_cm=alpha_cm,
                            thickness_m=thickness_m,
                            surface_position_m=z,
                            beta_cm_per_w=beta_cm_per_w,
                            n2_cm2_per_w=n2_cm2_per_w,
                        )
                        sim_absorption.append(1.0 - _transmission_fraction(prop_i))

                    fig.add_trace(
                        go.Scatter(
                            x=d["position_mm"],
                            y=sim_absorption,
                            mode="lines",
                            name=f"{material_entry['name']} (sim)",
                        )
                    )

                fig.update_layout(
                    xaxis_title="Sample position (mm)",
                    yaxis_title="Absorption / transmission-derived loss",
                )
                fig.update_yaxes(tickformat=".3e")
                _apply_pub_layout(fig, height=560, showlegend=True)
                st.plotly_chart(fig, width="stretch")

with tabs[4]:
    with lab_panel():
        st.subheader("Plot Export")
        plot_spec = PlotSpec(
            title="z_scan_normalized_transmission",
            x_label="Sample-center position, z (mm)",
            y_label="Normalized transmission, T(z)/T₀",
            series=[
                SeriesSpec(
                    name="T_over_T0",
                    x=z_positions_mm.tolist() if "z_positions_mm" in locals() else [],
                    y=t_normalised.tolist() if "t_normalised" in locals() else [],
                    x_label="Sample-center position, z (mm)",
                    y_label="Normalized transmission, T(z)/T₀",
                )
            ],
        )
        if st.button("Export current z-scan plot bundle", width="stretch"):
            outdir = Path("data/exports")
            files = export_plot_bundle(plot_spec, outdir, "z_scan_normalized_transmission")
            st.success("Exported: " + ", ".join(str(path) for path in files.values()))

    with lab_panel():
        st.subheader("Custom Physics Hooks")
        custom_models = load_models()

        if not custom_models:
            st.info(f"No custom models found. Directory: {MODELS_DIR}")
        else:
            enabled = st.multiselect(
                "Enable custom model previews",
                [m.name for m in custom_models],
                default=[m.name for m in custom_models if not m.error],
            )

            laser_dict = asdict(laser)
            laser_dict["_polarization_type"] = pol_type
            laser_dict["_polarization_angle_deg"] = pol_angle
            laser_dict["_polarization_handedness"] = pol_handedness
            laser_dict["_polarization_ellipticity_deg"] = pol_ellipticity
            laser_dict["_polarization_label"] = pol_label
            laser_dict["_n2_polarization_factor"] = pol_factor
            material_dict = asdict(material)

            for model in custom_models:
                if model.name not in enabled:
                    continue

                st.markdown("---")
                st.markdown(f"**{model.name}** `v{model.version}`")
                st.caption(model.description)

                if model.error:
                    st.error(model.error)
                    continue

                output = run_model(
                    model,
                    laser_dict=laser_dict,
                    material_dict=material_dict,
                    thickness_m=thickness_m,
                    z_position_m=surface_offset_m,
                )

                metric_items: dict[str, Any] = {}
                plot_items: dict[str, Any] = {}

                for key, value in output.items():
                    if isinstance(value, dict) and "x" in value and "y" in value:
                        plot_items[key] = value
                    elif isinstance(value, (int, float, str)):
                        metric_items[key] = value

                if metric_items:
                    cols = st.columns(min(4, max(1, len(metric_items))))
                    for i, (key, value) in enumerate(metric_items.items()):
                        with cols[i % len(cols)]:
                            st.metric(key, f"{value:.4g}" if isinstance(value, float) else str(value))

                if plot_items:
                    fig_custom = go.Figure()
                    for key, payload in plot_items.items():
                        fig_custom.add_trace(
                            go.Scatter(
                                x=payload["x"],
                                y=payload["y"],
                                mode="lines",
                                name=payload.get("label", key),
                            )
                        )
                    _apply_pub_layout(fig_custom, height=360, showlegend=True)
                    st.plotly_chart(fig_custom, width="stretch")

    with lab_panel():
        st.subheader("Legacy Fallbacks")
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.page_link("pages/30_Interaction_Analyzer_Legacy.py", label="Open Analyzer Legacy")
        with c2:
            st.page_link("pages/50_Simulation_Legacy.py", label="Open Simulation Legacy")
        with c3:
            st.page_link("pages/60_Digital_Twin_Legacy.py", label="Open Digital Twin Legacy")
        with c4:
            st.page_link("pages/70_Gnuplot_Legacy.py", label="Open Gnuplot Legacy")
