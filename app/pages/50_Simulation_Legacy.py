"""
pages/50_Simulation_Legacy.py — Legacy Campaign Overlay

Campaign-import and experiment-vs-simulation overlay workflow.

This page owns:
- campaign spreadsheet import
- sheet parsing
- experimental absorption / transmission overlay
- quick comparison against slab propagation

Deep nonlinear / thermal analysis remains in 60_Digital_Twin_Legacy.py.
"""

from __future__ import annotations

import math
import tempfile
import urllib.parse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from harrington_labs.lmi.io.campaign_import import parse_sheet
from harrington_labs.lmi.domain.lasers import all_lasers
from harrington_labs.lmi.domain.materials import all_materials
from harrington_labs.lmi.simulation.beam_propagation import (
    BeamParams,
    compute_focus,
    propagate_in_material,
)
from harrington_labs.lmi.ui.layout import render_header
from harrington_labs.lmi.ui.branding import lmi_panel
from harrington_labs.lmi.ui.formatting import (
    fmt_absorption_cm_inv,
    fmt_fluence_j_cm2,
    fmt_irradiance_w_cm2,
    fmt_length_m,
    fmt_n2_cm2_w,
    fmt_power_w,
    fmt_refractive_index,
    fmt_wavelength_nm,
)


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


def _first_attr(obj: Any, *names: str, default: Any = 0.0) -> Any:
    for name in names:
        if hasattr(obj, name):
            value = getattr(obj, name)
            if value is not None:
                return value
    return default


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


def _apply_pub_layout(fig, *, height: int = 500, showlegend: bool = True) -> None:
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


def _pack_query_string(
    *,
    laser,
    material_name: str,
    thickness_mm: float,
) -> str:
    pulse_energy_uj = laser.pulse_energy_j * 1e6 if not laser.is_cw else 0.0
    pulse_width_fs = laser.pulse_width_s * 1e15
    rep_rate_khz = laser.rep_rate_hz / 1e3

    params = {
        "laser": laser.name,
        "material": material_name,
        "wl": f"{laser.wavelength_nm:.6g}",
        "energy": f"{pulse_energy_uj:.6g}",
        "pulse": f"{pulse_width_fs:.6g}",
        "rep": f"{rep_rate_khz:.6g}",
        "beam": f"{laser.beam_diameter_mm:.6g}",
        "m2": f"{laser.m_squared:.6g}",
        "thickness": f"{thickness_mm:.6g}",
    }
    return urllib.parse.urlencode(params)


st.set_page_config(page_title="Simulation Legacy", layout="wide")
render_header()

qp = st.query_params

with lmi_panel():
    st.subheader("Simulation Legacy")
    st.caption(
        "Campaign import and experiment-vs-simulation overlay. "
        "Use Digital Twin Legacy for deeper nonlinear / thermal analysis."
    )

uploaded = st.file_uploader("Upload Excel campaign (.xlsx)", type=["xlsx"])

if uploaded is None:
    with lmi_panel():
        st.info("Upload a campaign spreadsheet to begin.")
    st.stop()

tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".xlsx")
tmp.write(uploaded.read())
tmp.close()

xls = pd.ExcelFile(tmp.name)

with lmi_panel():
    sheet = st.selectbox("Select sheet", xls.sheet_names)

materials_from_sheet = parse_sheet(tmp.name, sheet)

if not materials_from_sheet:
    with lmi_panel():
        st.warning("No materials detected in the selected sheet.")
    st.stop()

lasers = all_lasers()
material_models = all_materials()

laser_names = [l.name for l in lasers]
default_laser_name = _qp_str(qp, "laser", laser_names[0])
laser_index = laser_names.index(default_laser_name) if default_laser_name in laser_names else 0

material_names = [m.name for m in material_models]
default_material_name = _qp_str(qp, "material", material_names[0])
material_index = material_names.index(default_material_name) if default_material_name in material_names else 0

with lmi_panel():
    st.subheader("Simulation Controls")
    c1, c2, c3, c4 = st.columns(4)

    with c1:
        laser = st.selectbox("Laser", lasers, index=laser_index, format_func=lambda l: l.name)
    with c2:
        material_model = st.selectbox(
            "Material Model",
            material_models,
            index=material_index,
            format_func=lambda m: m.name,
        )
    with c3:
        thickness_mm = st.number_input(
            "Thickness (mm)",
            value=_qp_float(qp, "thickness", 0.5),
            min_value=0.001,
            step=0.1,
            format="%.6g",
        )
    with c4:
        focal_length_mm = st.number_input(
            "Focal Length (mm)",
            value=100.0,
            min_value=1.0,
            step=10.0,
            format="%.6g",
        )

beam = BeamParams(
    wavelength_m=laser.wavelength_nm * 1e-9,
    beam_diameter_1e2_m=laser.beam_diameter_mm * 1e-3,
    m_squared=laser.m_squared,
    pulse_energy_j=laser.pulse_energy_j if not laser.is_cw else 0.0,
    pulse_width_s=laser.pulse_width_s,
    rep_rate_hz=laser.rep_rate_hz,
    spatial_mode=getattr(laser, "spatial_mode", "TEM00") or "TEM00",
)

n_mat = _material_n(material_model, laser.wavelength_nm)
alpha_cm = _material_alpha_cm(material_model, laser.wavelength_nm)
n2_cm2_per_w = _material_n2_cm2_per_w(material_model, laser.wavelength_nm)
beta_cm_per_w = _material_beta_cm_per_w(material_model)

focus = compute_focus(beam, focal_length_m=focal_length_mm * 1e-3, n_medium=1.0)
thickness_m = thickness_mm * 1e-3

with lmi_panel():
    st.subheader("Applied Source / Material State")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Laser", laser.name)
    m2.metric("Wavelength", fmt_wavelength_nm(laser.wavelength_nm))
    m3.metric("Peak Power", fmt_power_w(beam.peak_power_w))
    m4.metric("Avg Power", fmt_power_w(beam.avg_power_w))

    m5, m6, m7, m8 = st.columns(4)
    m5.metric("n", fmt_refractive_index(n_mat))
    m6.metric("α", fmt_absorption_cm_inv(alpha_cm))
    m7.metric("n₂", fmt_n2_cm2_w(n2_cm2_per_w))
    m8.metric("β", f"{beta_cm_per_w:.4g} cm/W")

    m9, m10 = st.columns(2)
    m9.metric("Spot Radius w₀", fmt_length_m(focus.w0_m))
    m10.metric("Fluence at Focus", fmt_fluence_j_cm2(focus.fluence_j_cm2))

query_string = _pack_query_string(
    laser=laser,
    material_name=material_model.name,
    thickness_mm=thickness_mm,
)

with lmi_panel():
    st.subheader("Workflow Handoff")
    h1, h2 = st.columns(2)
    with h1:
        st.link_button(
            "Open Interaction Analyzer Legacy",
            f"/Interaction_Analyzer_Legacy?{query_string}",
            width="stretch",
        )
    with h2:
        st.link_button(
            "Open Digital Twin Legacy",
            f"/Digital_Twin_Legacy?{query_string}",
            width="stretch",
        )

with lmi_panel():
    st.subheader("Experimental vs Simulation Overlay")

    fig = go.Figure()

    for material_entry in materials_from_sheet:
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
        sim_transmission = []

        for z in z_vals_m:
            prop = _propagate_material(
                focus=focus,
                beam=beam,
                n_material=n_mat,
                alpha_cm=alpha_cm,
                thickness_m=thickness_m,
                surface_position_m=z,
                beta_cm_per_w=beta_cm_per_w,
                n2_cm2_per_w=n2_cm2_per_w,
            )
            t = _transmission_fraction(prop)
            sim_transmission.append(t)
            sim_absorption.append(1.0 - t)

        fig.add_trace(
            go.Scatter(
                x=d["position_mm"],
                y=sim_absorption,
                mode="lines",
                name=f"{material_entry['name']} (sim absorption)",
            )
        )

    fig.update_layout(
        xaxis_title="Sample position (mm)",
        yaxis_title="Absorption / transmission-derived loss",
    )
    fig.update_yaxes(tickformat=".3e")
    _apply_pub_layout(fig, height=560, showlegend=True)
    st.plotly_chart(fig, width="stretch")

with lmi_panel():
    st.subheader("Slab Preview at z = 0 mm")

    prop0 = _propagate_material(
        focus=focus,
        beam=beam,
        n_material=n_mat,
        alpha_cm=alpha_cm,
        thickness_m=thickness_m,
        surface_position_m=0.0,
        beta_cm_per_w=beta_cm_per_w,
        n2_cm2_per_w=n2_cm2_per_w,
    )

    fig2 = go.Figure()
    fig2.add_trace(
        go.Scatter(
            x=np.asarray(prop0.z_material_m) * 1e3,
            y=np.asarray(prop0.irradiance_z_w_cm2),
            mode="lines",
            name="Peak irradiance",
        )
    )
    fig2.update_layout(
        xaxis_title="Depth in sample (mm)",
        yaxis_title="Peak irradiance (W cm⁻²)",
    )
    fig2.update_yaxes(type="log", tickformat=".3e")
    _apply_pub_layout(fig2, height=420, showlegend=False)
    st.plotly_chart(fig2, width="stretch")

with st.expander("Show Raw Parsed Data"):
    for material_entry in materials_from_sheet:
        st.markdown(f"**{material_entry['name']}**")
        st.dataframe(material_entry["data"], width="stretch", hide_index=True)
