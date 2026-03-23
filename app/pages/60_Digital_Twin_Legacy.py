"""
pages/60_Digital_Twin_Legacy.py — Legacy Digital Twin

Deep analysis page for ultrafast laser–material interaction.

This page owns:
- beam propagation
- thermal response
- nonlinear optics
- absorption mapping

It accepts query parameters from the Interaction Analyzer Legacy page.
"""

from __future__ import annotations

import math
import urllib.parse

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from harrington_labs.ui import render_header
from harrington_labs.ui import lab_panel, make_figure, show_figure, COLORS
from harrington_labs.lmi.ui.formatting import (
    fmt_absorption_cm_inv,
    fmt_ev,
    fmt_fluence_j_cm2,
    fmt_irradiance_w_cm2,
    fmt_length_m,
    fmt_n2_cm2_w,
    fmt_power_w,
    fmt_refractive_index,
    fmt_temp_k,
    fmt_wavelength_nm,
)
from harrington_labs.lmi.domain.lasers import SpatialMode
from harrington_labs.lmi.domain.materials import all_materials, Material
from harrington_labs.lmi.simulation.beam_propagation import BeamParams, compute_focus, propagate_in_material
from harrington_labs.lmi.simulation.thermal import thermal_analysis, two_temperature_model
from harrington_labs.lmi.simulation.nonlinear import nonlinear_analysis


POLARIZATION_TYPES = ("Linear", "Circular", "Elliptical", "Unpolarized")


def _qp_float(qp, key: str, default: float) -> float:
    value = qp.get(key)
    if value is None or value == "":
        return float(default)
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _qp_str(qp, key: str, default: str) -> str:
    value = qp.get(key)
    return default if value is None else str(value)


def _n2_correction(pol_type: str, ellipticity_deg: float = 0.0) -> float:
    if pol_type == "Linear":
        return 1.0
    if pol_type == "Circular":
        return 2.0 / 3.0
    if pol_type == "Elliptical":
        chi_rad = math.radians(ellipticity_deg)
        return 1.0 - (1.0 / 3.0) * math.sin(2 * chi_rad) ** 2
    return 1.0


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


st.set_page_config(page_title="Digital Twin Legacy", layout="wide")
render_header()

qp = st.query_params
st.sidebar.header("Experimental Parameters")

st.sidebar.subheader("Laser")
wavelength_nm = st.sidebar.number_input(
    "Wavelength (nm)",
    100.0,
    20000.0,
    _qp_float(qp, "wl", 8500.0),
    step=100.0,
    format="%.6g",
)
pulse_energy_uj = st.sidebar.number_input(
    "Pulse Energy (µJ)",
    0.0,
    1e6,
    _qp_float(qp, "energy", 20.0),
    step=1.0,
    format="%.6g",
)
pulse_width_fs = st.sidebar.number_input(
    "Pulse Width (fs)",
    1.0,
    1e9,
    _qp_float(qp, "pulse", 170.0),
    step=10.0,
    format="%.6g",
)
rep_rate_khz = st.sidebar.number_input(
    "Rep Rate (kHz)",
    0.001,
    1e9,
    _qp_float(qp, "rep", 10.0),
    step=1.0,
    format="%.6g",
)

spatial_mode = st.sidebar.selectbox("Spatial Mode", [m.value for m in SpatialMode], index=0)
default_m2 = SpatialMode(spatial_mode).default_m_squared
m_squared = st.sidebar.number_input(
    "M²",
    1.0,
    50.0,
    _qp_float(qp, "m2", default_m2),
    step=0.1,
    format="%.3f",
)

st.sidebar.subheader("Polarization")
qp_pol_type = _qp_str(qp, "pol_type", "Linear")
pol_index = POLARIZATION_TYPES.index(qp_pol_type) if qp_pol_type in POLARIZATION_TYPES else 0
pol_type = st.sidebar.selectbox("Type", POLARIZATION_TYPES, index=pol_index)

if pol_type == "Linear":
    pol_angle = st.sidebar.number_input("Angle (°)", 0.0, 180.0, _qp_float(qp, "pol_angle", 0.0), step=1.0, format="%.0f")
    pol_handedness = ""
    pol_ellipticity = 0.0
elif pol_type == "Circular":
    pol_handedness = st.sidebar.selectbox(
        "Handedness",
        ("Left", "Right"),
        index=0 if _qp_str(qp, "pol_hand", "Left") == "Left" else 1,
    )
    pol_angle = 0.0
    pol_ellipticity = 45.0
elif pol_type == "Elliptical":
    pol_handedness = st.sidebar.selectbox(
        "Handedness",
        ("Left", "Right"),
        index=0 if _qp_str(qp, "pol_hand", "Left") == "Left" else 1,
    )
    pol_ellipticity = st.sidebar.number_input(
        "Ellipticity χ (°)",
        0.0,
        45.0,
        _qp_float(qp, "pol_chi", 15.0),
        step=1.0,
        format="%.1f",
    )
    pol_angle = 0.0
else:
    pol_angle = 0.0
    pol_handedness = ""
    pol_ellipticity = 0.0

pol_factor = _n2_correction(pol_type, pol_ellipticity)
if pol_type != "Linear" or pol_angle != 0:
    st.sidebar.caption(f"n₂ correction: {pol_factor:.4g}×")

st.sidebar.subheader("Focusing")
focal_length_cm = st.sidebar.number_input(
    "Focal Length (cm)",
    0.1,
    100.0,
    _qp_float(qp, "focal", 10.0),
    step=1.0,
    format="%.6g",
)
beam_diameter_mm = st.sidebar.number_input(
    "Input Beam Diameter (mm, 1/e²)",
    0.01,
    50.0,
    _qp_float(qp, "beam", 5.0),
    step=0.5,
    format="%.6g",
)
spot_override = st.sidebar.checkbox("Override Spot Size", value=True)
spot_diameter_um = st.sidebar.number_input(
    "Measured Spot Diameter (µm, 1/e²)",
    1.0,
    5000.0,
    200.0,
    step=10.0,
    format="%.6g",
    disabled=not spot_override,
)

st.sidebar.subheader("Sample")
materials = all_materials()
material_names = [m.name for m in materials]
qp_mat = _qp_str(qp, "material", material_names[0])
mat_index = material_names.index(qp_mat) if qp_mat in material_names else 0
mat_name = st.sidebar.selectbox("Material", material_names, index=mat_index)
material: Material = next(m for m in materials if m.name == mat_name)

thickness_mm = st.sidebar.number_input(
    "Thickness (mm)",
    0.001,
    50.0,
    _qp_float(qp, "thickness", 0.1),
    step=0.05,
    format="%.6g",
)
surface_offset_um = st.sidebar.number_input(
    "Focus Offset from Surface (µm)",
    -5000.0,
    5000.0,
    _qp_float(qp, "offset", 0.0),
    step=10.0,
    format="%.6g",
)
t_ambient_k = st.sidebar.number_input(
    "Ambient Temperature (K)",
    4.0,
    1000.0,
    300.0,
    step=10.0,
    format="%.6g",
)

pulse_energy_j = pulse_energy_uj * 1e-6
pulse_width_s = pulse_width_fs * 1e-15
rep_rate_hz = rep_rate_khz * 1e3
focal_length_m = focal_length_cm * 1e-2
thickness_m = thickness_mm * 1e-3
surface_offset_m = surface_offset_um * 1e-6

beam = BeamParams(
    wavelength_m=wavelength_nm * 1e-9,
    beam_diameter_1e2_m=beam_diameter_mm * 1e-3,
    m_squared=m_squared,
    pulse_energy_j=pulse_energy_j,
    pulse_width_s=pulse_width_s,
    rep_rate_hz=rep_rate_hz,
    spatial_mode=spatial_mode,
)

n_mat = material.get_n(wavelength_nm)
alpha_mat = material.get_alpha(wavelength_nm)
n2_mat = material.get_n2(wavelength_nm) * pol_factor
beta_mat = float(getattr(material, "two_photon_abs_cm_w", 0.0) or 0.0)

focus = compute_focus(beam, focal_length_m, n_medium=1.0)
effective_w0_m = (spot_diameter_um * 1e-6) / 2 if spot_override else focus.w0_m
area_cm2 = math.pi * (effective_w0_m * 100.0) ** 2
peak_irradiance = beam.peak_power_w / area_cm2 if area_cm2 > 0 else 0.0
fluence = pulse_energy_j / area_cm2 if area_cm2 > 0 else 0.0

st.subheader(f"Digital Twin Legacy — {fmt_wavelength_nm(wavelength_nm)}")

with lab_panel():
    back_params = {
        "laser": _qp_str(qp, "laser", ""),
        "material": mat_name,
        "wl": f"{wavelength_nm:.6g}",
        "energy": f"{pulse_energy_uj:.6g}",
        "pulse": f"{pulse_width_fs:.6g}",
        "rep": f"{rep_rate_khz:.6g}",
        "beam": f"{beam_diameter_mm:.6g}",
        "m2": f"{m_squared:.6g}",
        "focal": f"{focal_length_cm:.6g}",
        "thickness": f"{thickness_mm:.6g}",
        "offset": f"{surface_offset_um:.6g}",
        "pol_type": pol_type,
        "pol_angle": f"{pol_angle:.6g}",
        "pol_hand": pol_handedness,
        "pol_chi": f"{pol_ellipticity:.6g}",
    }
    back_qs = urllib.parse.urlencode({k: v for k, v in back_params.items() if v != ""})

    c1, c2 = st.columns(2)
    with c1:
        st.link_button("Open Interaction Analyzer Legacy", f"/Interaction_Analyzer_Legacy?{back_qs}", width="stretch")
    with c2:
        st.link_button("Open Simulation Legacy", f"/Simulation_Legacy?{back_qs}", width="stretch")

if pol_type != "Linear" or pol_angle != 0:
    st.info(f"Polarization: {pol_type} — n₂ correction {pol_factor:.4g}×")

with lab_panel():
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Peak Power", fmt_power_w(beam.peak_power_w))
    c2.metric("Avg Power", fmt_power_w(beam.avg_power_w))
    c3.metric("Spot (1/e²)", fmt_length_m(effective_w0_m * 2))
    c4.metric("Fluence", fmt_fluence_j_cm2(fluence))
    c5.metric("Irradiance", fmt_irradiance_w_cm2(peak_irradiance))

    photon_ev = 1240.0 / wavelength_nm if wavelength_nm > 0 else 0.0
    n_photons = math.ceil(material.bandgap_ev / photon_ev) if photon_ev > 0 else 0
    disp_info = material.dispersion_info(wavelength_nm)
    n_source = "Sellmeier" if disp_info.get("sellmeier_used") else "stored"

    c1b, c2b, c3b, c4b = st.columns(4)
    c1b.metric("Photon Energy", fmt_ev(photon_ev))
    c2b.metric("Bandgap", fmt_ev(material.bandgap_ev))
    c3b.metric("MPA Order", f"{n_photons}-photon")
    c4b.metric(f"n ({n_source})", fmt_refractive_index(n_mat))

    c5b, c6b, c7b = st.columns(3)
    c5b.metric("α", fmt_absorption_cm_inv(alpha_mat))
    c6b.metric("n₂", fmt_n2_cm2_w(n2_mat))
    c7b.metric("β", f"{beta_mat:.4g} cm/W")

tab_beam, tab_thermal, tab_nonlinear, tab_absorption = st.tabs(
    ["Beam Propagation", "Thermal Analysis", "Nonlinear Optics", "Absorption Map"]
)

with tab_beam:
    with lab_panel():
        st.subheader("Focused Beam Profile")
        col1, col2, col3 = st.columns(3)
        col1.metric("Calculated w₀", fmt_length_m(focus.w0_m))
        col2.metric("Rayleigh Range", fmt_length_m(focus.rayleigh_range_m))
        col3.metric("f/#", f"{focus.f_number:.4g}")

    if focus.r_m is not None and focus.intensity_r is not None:
        with lab_panel():
            st.subheader(f"Transverse Profile — {spatial_mode}")
            fig_mode = go.Figure()
            fig_mode.add_trace(
                go.Scatter(
                    x=focus.r_m * 1e6,
                    y=focus.intensity_r,
                    mode="lines",
                    line=dict(width=2),
                    name=f"{spatial_mode}",
                )
            )
            if spatial_mode != "TEM00":
                gauss = np.exp(-2.0 * (focus.r_m / focus.w0_m) ** 2)
                fig_mode.add_trace(
                    go.Scatter(
                        x=focus.r_m * 1e6,
                        y=gauss,
                        mode="lines",
                        line=dict(width=1, dash="dash"),
                        name="TEM00 reference",
                    )
                )
            fig_mode.update_layout(
                xaxis_title="Radius (µm)",
                yaxis_title="Normalized intensity",
            )
            fig_mode.update_yaxes(tickformat=".3e")
            _apply_pub_layout(fig_mode, height=320, showlegend=True)
            st.plotly_chart(fig_mode, width="stretch")

    with lab_panel():
        fig = go.Figure()
        z_mm = focus.z_m * 1e3
        w_um = focus.w_z_m * 1e6
        fig.add_trace(go.Scatter(x=z_mm, y=w_um, mode="lines", line=dict(width=2), name="Beam radius"))
        fig.add_trace(go.Scatter(x=z_mm, y=-w_um, mode="lines", line=dict(width=2), showlegend=False))
        z_surf = surface_offset_m * 1e3
        z_back = z_surf + thickness_mm
        fig.add_vrect(x0=z_surf, x1=z_back, fillcolor="rgba(230,57,70,0.13)", line_width=0)
        fig.add_vline(x=z_surf, line_dash="dash", line_color="gray")
        fig.update_layout(
            xaxis_title="z from focus (mm)",
            yaxis_title="Beam radius (µm)",
        )
        fig.update_yaxes(tickformat=".3e")
        _apply_pub_layout(fig, height=400, showlegend=True)
        st.plotly_chart(fig, width="stretch")

    with lab_panel():
        st.subheader("Propagation Through Sample")
        prop = _propagate_material(
            focus=focus,
            beam=beam,
            n_material=n_mat,
            alpha_cm=alpha_mat,
            thickness_m=thickness_m,
            surface_position_m=surface_offset_m,
            beta_cm_per_w=beta_mat,
            n2_cm2_per_w=n2_mat,
        )
        fig2 = make_subplots(rows=1, cols=2, subplot_titles=["Beam Radius in Material", "Irradiance vs Depth"])
        z_um = prop.z_material_m * 1e6
        fig2.add_trace(go.Scatter(x=z_um, y=prop.w_z_m * 1e6, mode="lines", line=dict(width=2), name="w(z)"), row=1, col=1)
        fig2.add_trace(go.Scatter(x=z_um, y=prop.irradiance_z_w_cm2, mode="lines", line=dict(width=2), name="I(z)"), row=1, col=2)
        fig2.update_xaxes(title_text="Depth (µm)", row=1, col=1)
        fig2.update_xaxes(title_text="Depth (µm)", row=1, col=2)
        fig2.update_yaxes(title_text="Radius (µm)", tickformat=".3e", row=1, col=1)
        fig2.update_yaxes(title_text="Irradiance (W cm⁻²)", type="log", tickformat=".3e", row=1, col=2)
        _apply_pub_layout(fig2, height=420, showlegend=False)
        st.plotly_chart(fig2, width="stretch")

with tab_thermal:
    D_mat = material.get_thermal_diffusivity()
    therm = thermal_analysis(
        fluence_j_cm2=fluence,
        pulse_width_s=pulse_width_s,
        rep_rate_hz=rep_rate_hz,
        spot_radius_m=effective_w0_m,
        alpha_cm=alpha_mat,
        thermal_conductivity_w_mk=material.thermal_conductivity_w_mk,
        density_kg_m3=material.density_kg_m3,
        specific_heat_j_kgk=material.specific_heat_j_kgk,
        thermal_diffusivity_m2_s=D_mat,
        melting_point_k=material.melting_point_k,
        t_ambient_k=t_ambient_k,
        n_pulses_max=500,
    )

    with lab_panel():
        st.subheader("Single-Pulse Thermal Response")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Surface ΔT", fmt_temp_k(therm.delta_t_surface_k))
        c2.metric("Diffusion Length", fmt_length_m(therm.thermal_diffusion_length_m))
        c3.metric("Heat Confined?", "Yes" if therm.heat_confined else "No")
        c4.metric("Melt Threshold", fmt_fluence_j_cm2(therm.melt_threshold_fluence_j_cm2))
        fig_t = go.Figure()
        fig_t.add_trace(go.Scatter(x=therm.z_m * 1e6, y=therm.delta_t_z_k + t_ambient_k, mode="lines", line=dict(width=2), name="T(z)"))
        fig_t.add_hline(y=material.melting_point_k, line_dash="dash", line_color="#b8860b")
        fig_t.update_layout(xaxis_title="Depth (µm)", yaxis_title="Temperature (K)")
        fig_t.update_yaxes(tickformat=".3e")
        _apply_pub_layout(fig_t, height=360, showlegend=False)
        st.plotly_chart(fig_t, width="stretch")

    with lab_panel():
        st.subheader("Multi-Pulse Accumulation")
        fig_acc = go.Figure()
        fig_acc.add_trace(go.Scatter(x=therm.n_pulses, y=therm.t_surface_vs_pulses, mode="lines", line=dict(width=2), name="Surface temperature"))
        fig_acc.add_hline(y=material.melting_point_k, line_dash="dash", line_color="#b8860b")
        fig_acc.update_layout(xaxis_title="Pulse Number", yaxis_title="Surface Temperature (K)")
        fig_acc.update_yaxes(tickformat=".3e")
        _apply_pub_layout(fig_acc, height=360, showlegend=False)
        st.plotly_chart(fig_acc, width="stretch")

    if material.electron_phonon_coupling_w_m3k > 0:
        with lab_panel():
            st.subheader("Two-Temperature Model")
            ttm = two_temperature_model(
                fluence_j_cm2=fluence,
                pulse_width_s=pulse_width_s,
                alpha_cm=alpha_mat,
                electron_phonon_coupling_w_m3k=material.electron_phonon_coupling_w_m3k,
                density_kg_m3=material.density_kg_m3,
                specific_heat_j_kgk=material.specific_heat_j_kgk,
                t_ambient_k=t_ambient_k,
                t_max_ps=50.0,
            )
            c1, c2, c3 = st.columns(3)
            c1.metric("Peak Electron Temp", fmt_temp_k(ttm.peak_electron_temp_k))
            c2.metric("Final Lattice Temp", fmt_temp_k(ttm.final_lattice_temp_k))
            c3.metric("Equilibration Time", f"{ttm.equilibrium_time_ps:.4g} ps")
            fig_ttm = go.Figure()
            fig_ttm.add_trace(go.Scatter(x=ttm.t_ps, y=ttm.t_electron_k, mode="lines", line=dict(width=2), name="Electron"))
            fig_ttm.add_trace(go.Scatter(x=ttm.t_ps, y=ttm.t_lattice_k, mode="lines", line=dict(width=2), name="Lattice"))
            fig_ttm.add_hline(y=material.melting_point_k, line_dash="dash", line_color="#b8860b")
            fig_ttm.update_layout(xaxis_title="Time (ps)", yaxis_title="Temperature (K)")
            fig_ttm.update_yaxes(tickformat=".3e")
            _apply_pub_layout(fig_ttm, height=400, showlegend=True)
            st.plotly_chart(fig_ttm, width="stretch")

with tab_nonlinear:
    nl = nonlinear_analysis(
        wavelength_nm=wavelength_nm,
        bandgap_ev=material.bandgap_ev,
        n2_cm2_w=n2_mat,
        refractive_index=n_mat,
        alpha_linear_cm=alpha_mat,
        peak_power_w=beam.peak_power_w,
        irradiance_w_cm2=peak_irradiance,
        pulse_width_s=pulse_width_s,
        beam_radius_m=effective_w0_m,
        thickness_m=thickness_m,
    )

    with lab_panel():
        st.subheader("Multiphoton Absorption")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("MPA Order", f"{nl.mpa.photon_order}-photon")
        c2.metric("MPA Dominant?", "Yes" if nl.mpa.is_dominant else "No")
        c3.metric("MPA Absorption Depth", fmt_length_m(nl.mpa.mpa_absorption_depth_m) if nl.mpa.mpa_absorption_depth_m < 1.0 else "> 1 m")
        c4.metric("Energy Absorbed", f"{nl.mpa.energy_deposited_fraction * 100:.4g}%")

    with lab_panel():
        st.subheader("Self-Focusing / Kerr Effect")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("P / Pcritical", f"{nl.self_focusing.p_over_pcr:.4g}")
        c2.metric("Critical Power", fmt_power_w(nl.self_focusing.critical_power_w))
        c3.metric("Self-Focusing?", "YES" if nl.self_focusing.self_focusing_occurs else "No")
        c4.metric("B-integral", f"{nl.self_focusing.b_integral_rad:.4g} rad")
        c5, c6 = st.columns(2)
        c5.metric("Peak Δn (Kerr)", fmt_refractive_index(nl.self_focusing.delta_n_peak))
        c6.metric("n₂", fmt_n2_cm2_w(n2_mat))

    with lab_panel():
        st.subheader("Depth-Resolved Nonlinear Effects")
        fig_nl = make_subplots(rows=1, cols=2, subplot_titles=["Irradiance (with NL absorption)", "Kerr Δn(z)"])
        z_um = nl.z_m * 1e6
        fig_nl.add_trace(go.Scatter(x=z_um, y=nl.irradiance_z_w_cm2, mode="lines", line=dict(width=2), name="I(z) with MPA"), row=1, col=1)
        fig_nl.add_trace(go.Scatter(x=z_um, y=nl.delta_n_z, mode="lines", line=dict(width=2), name="Δn(z)"), row=1, col=2)
        fig_nl.update_xaxes(title_text="Depth (µm)", row=1, col=1)
        fig_nl.update_xaxes(title_text="Depth (µm)", row=1, col=2)
        fig_nl.update_yaxes(title_text="Irradiance (W cm⁻²)", type="log", tickformat=".3e", row=1, col=1)
        fig_nl.update_yaxes(title_text="Δn", tickformat=".3e", row=1, col=2)
        _apply_pub_layout(fig_nl, height=420, showlegend=False)
        st.plotly_chart(fig_nl, width="stretch")

with tab_absorption:
    with lab_panel():
        st.subheader("Combined Absorption Profile")
        prop = _propagate_material(
            focus=focus,
            beam=beam,
            n_material=n_mat,
            alpha_cm=alpha_mat,
            thickness_m=thickness_m,
            surface_position_m=surface_offset_m,
            beta_cm_per_w=beta_mat,
            n2_cm2_per_w=n2_mat,
        )

        fig_abs = make_subplots(
            rows=2,
            cols=2,
            subplot_titles=[
                "Transmission vs Depth",
                "Fluence vs Depth",
                "Free Carrier Density",
                "Absorbed Energy Density",
            ],
        )

        z_um = prop.z_material_m * 1e6
        fig_abs.add_trace(go.Scatter(x=z_um, y=(1.0 - prop.absorbed_fraction) * 100, mode="lines", line=dict(width=2), name="T(%)"), row=1, col=1)
        fig_abs.add_trace(go.Scatter(x=z_um, y=prop.fluence_z_j_cm2 * 1000, mode="lines", line=dict(width=2), name="Fluence"), row=1, col=2)

        z_nl_um = nl.z_m * 1e6
        fig_abs.add_trace(go.Scatter(x=z_nl_um, y=nl.carrier_density_z_cm3, mode="lines", line=dict(width=2), name="Free carriers"), row=2, col=1)

        abs_energy = alpha_mat * prop.irradiance_z_w_cm2 * pulse_width_s * 100.0
        fig_abs.add_trace(go.Scatter(x=z_um, y=abs_energy, mode="lines", line=dict(width=2), name="Absorbed"), row=2, col=2)

        for r in (1, 2):
            for c in (1, 2):
                fig_abs.update_xaxes(title_text="Depth (µm)", row=r, col=c)

        fig_abs.update_yaxes(title_text="Transmission (%)", tickformat=".3e", row=1, col=1)
        fig_abs.update_yaxes(title_text="Fluence (mJ cm⁻²)", tickformat=".3e", row=1, col=2)
        fig_abs.update_yaxes(title_text="Carriers (cm⁻³)", type="log", tickformat=".3e", row=2, col=1)
        fig_abs.update_yaxes(title_text="Energy (J cm⁻³)", tickformat=".3e", row=2, col=2)
        _apply_pub_layout(fig_abs, height=620, showlegend=False)
        st.plotly_chart(fig_abs, width="stretch")

    with lab_panel():
        st.subheader("Summary")
        summary_items = [
            f"Wavelength: {fmt_wavelength_nm(wavelength_nm)} ({fmt_ev(photon_ev)}/photon)",
            f"Material: {mat_name} (Eg = {fmt_ev(material.bandgap_ev)}, n = {fmt_refractive_index(n_mat)})",
            f"Spatial mode: {spatial_mode} (M² = {m_squared:.4g})",
            f"MPA order: {n_photons}-photon absorption",
            f"Peak irradiance: {fmt_irradiance_w_cm2(peak_irradiance)}",
            f"Fluence: {fmt_fluence_j_cm2(fluence)}",
            f"Single-pulse surface ΔT: {fmt_temp_k(therm.delta_t_surface_k)}",
            f"Thermal diffusion length: {fmt_length_m(therm.thermal_diffusion_length_m)}",
            f"Heat confined: {'Yes' if therm.heat_confined else 'No'}",
            f"P/Pcritical: {nl.self_focusing.p_over_pcr:.4g}",
            f"B-integral: {nl.self_focusing.b_integral_rad:.4g} rad",
        ]
        for item in summary_items:
            st.markdown(f"- {item}")
