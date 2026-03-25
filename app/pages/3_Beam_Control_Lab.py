"""3_Beam_Control_Lab.py — Beam Control & Atmospheric Propagation Simulator."""
import streamlit as st
import plotly.graph_objects as go
from harrington_labs.ui import render_header, lab_panel, make_figure, show_figure, warning_box, COLORS
from harrington_labs.domain import (
    BeamParams, BeamProfile, PolarizationState,
    PropagationPath, AtmosphericCondition, AdaptiveOpticsParams,
)
from harrington_labs.simulation.beam_control import run_beam_control_simulation

st.set_page_config(page_title="Beam Control Lab", layout="wide")
render_header("Beam Control Lab", "Atmospheric propagation • Turbulence • AO correction • Beam wander")

# ── Sidebar: beam ────────────────────────────────────────────────
from harrington_labs.ui.shared_state import get_shared_beam, push_beam_button, shared_beam_badge
from harrington_labs.ui.db_sidebar import source_and_material_sidebar
sb = get_shared_beam()
shared_beam_badge()
db_laser, db_material = source_and_material_sidebar("beam")

st.sidebar.header("Beam Parameters")
_def_wl = db_laser.wavelength_nm if db_laser else sb["wavelength_nm"]
_def_pwr = db_laser.power_w if db_laser else sb["power_w"]
_def_bd = db_laser.beam_diameter_mm if db_laser else sb["beam_diameter_mm"]
_def_m2 = db_laser.m_squared if db_laser else sb["m_squared"]
wavelength = st.sidebar.number_input("Wavelength (nm)", 300.0, 12000.0, _def_wl, 1.0)
power = st.sidebar.number_input("Power (W)", 0.001, 1e6, _def_pwr, 1.0)
beam_d = st.sidebar.number_input("Beam Diameter (mm)", 0.1, 500.0, _def_bd, 1.0)
m2 = st.sidebar.number_input("M²", 1.0, 20.0, _def_m2, 0.1)

beam = BeamParams(
    wavelength_nm=wavelength, power_w=power,
    beam_diameter_mm=beam_d, m_squared=m2,
)

# ── Sidebar: path ────────────────────────────────────────────────
st.sidebar.header("Propagation Path")
distance = st.sidebar.number_input("Distance (m)", 10.0, 100000.0, 1000.0, 100.0)
condition = st.sidebar.selectbox("Condition", [c.value for c in AtmosphericCondition])
cn2_exp = st.sidebar.slider("log₁₀(Cn²) [m⁻²ᐟ³]", -17.0, -12.0, -15.0, 0.1)
visibility = st.sidebar.number_input("Visibility (km)", 0.1, 50.0, 23.0, 0.5)
wind = st.sidebar.number_input("Wind Speed (m/s)", 0.0, 50.0, 5.0, 0.5)

path = PropagationPath(
    distance_m=distance,
    condition=AtmosphericCondition(condition),
    cn2=10 ** cn2_exp,
    visibility_km=visibility,
    wind_speed_m_s=wind,
)

# ── Sidebar: AO ──────────────────────────────────────────────────
use_ao = st.sidebar.checkbox("Include Adaptive Optics", value=True)
ao = None
if use_ao:
    st.sidebar.header("AO System")
    n_act = st.sidebar.number_input("Actuator Count", 10, 10000, 97)
    bw = st.sidebar.number_input("Bandwidth (Hz)", 10.0, 10000.0, 1000.0, 100.0)
    latency = st.sidebar.number_input("Latency (ms)", 0.1, 20.0, 1.0, 0.1)
    ao = AdaptiveOpticsParams(actuator_count=n_act, bandwidth_hz=bw, latency_ms=latency)

result = run_beam_control_simulation(beam, path, ao)
warning_box(result.warnings)

# ── Summary metrics ──────────────────────────────────────────────
spread = result.data["beam_spread"]
with lab_panel("Propagation Summary"):
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Path Transmission", f"{result.data['path_transmission']:.1%}")
    c2.metric("Fried r₀", f"{spread['r0_m']*100:.1f} cm" if spread['r0_m'] < 100 else ">> 1 m")
    c3.metric("D/r₀", f"{spread['d_over_r0']:.1f}")
    c4.metric("Rytov σ²ᵣ", f"{result.data['rytov_variance']:.3f}")

    c5, c6, c7, c8 = st.columns(4)
    c5.metric("Vacuum Spot", f"{spread['w_vacuum_m']*100:.1f} cm")
    c6.metric("Long-Term Spot", f"{spread['w_long_term_m']*100:.1f} cm")
    c7.metric("Short-Term Spot", f"{spread['w_short_term_m']*100:.1f} cm")
    c8.metric("Beam Wander RMS", f"{spread['beam_wander_rms_m']*1e3:.2f} mm")

# ── AO Strehl ────────────────────────────────────────────────────
if "ao_strehl" in result.data:
    with lab_panel("Adaptive Optics Performance"):
        ao_d = result.data["ao_strehl"]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Uncorrected Strehl", f"{ao_d['strehl_uncorrected']:.4f}")
        c2.metric("Corrected Strehl", f"{ao_d['strehl_corrected']:.4f}")
        c3.metric("Improvement Factor", f"{ao_d['strehl_improvement']:.1f}×" if ao_d['strehl_improvement'] < 1e6 else "∞")
        c4.metric("τ₀", f"{ao_d['tau0_ms']:.2f} ms" if ao_d['tau0_ms'] < 1e6 else ">> 1 s")

        c5, c6, c7 = st.columns(3)
        c5.metric("σ² Fitting", f"{ao_d['sigma2_fit']:.3f} rad²")
        c6.metric("σ² Temporal", f"{ao_d['sigma2_temp']:.3f} rad²")
        c7.metric("σ² Total", f"{ao_d['sigma2_total']:.3f} rad²")

# ── Beam radius along path ───────────────────────────────────────
with lab_panel("Beam Radius Along Path"):
    prof = result.data["profile"]
    fig = make_figure("Beam Radius vs Propagation Distance")
    fig.add_trace(go.Scatter(
        x=prof["z_m"], y=prof["w_vacuum_m"] * 100,
        name="Vacuum", line=dict(color=COLORS[0], width=2, dash="dash"),
    ))
    fig.add_trace(go.Scatter(
        x=prof["z_m"], y=prof["w_turbulence_m"] * 100,
        name="With Turbulence", line=dict(color=COLORS[1], width=2.5),
    ))
    fig.update_xaxes(title_text="Distance (m)")
    fig.update_yaxes(title_text="Beam Radius (cm)")
    show_figure(fig)

# ── Irradiance along path ────────────────────────────────────────
col1, col2 = st.columns(2)
with col1:
    with lab_panel("On-Axis Irradiance"):
        fig = make_figure("Irradiance vs Distance")
        fig.add_trace(go.Scatter(
            x=prof["z_m"], y=prof["irradiance_vacuum_w_cm2"],
            name="Vacuum", line=dict(color=COLORS[0], width=2, dash="dash"),
        ))
        fig.add_trace(go.Scatter(
            x=prof["z_m"], y=prof["irradiance_turbulence_w_cm2"],
            name="Turbulence", line=dict(color=COLORS[1], width=2.5),
        ))
        fig.update_xaxes(title_text="Distance (m)")
        fig.update_yaxes(title_text="Irradiance (W/cm²)", type="log")
        show_figure(fig)

with col2:
    with lab_panel("Atmospheric Transmission"):
        fig = make_figure("Transmission vs Distance")
        fig.add_trace(go.Scatter(
            x=prof["z_m"], y=prof["transmission"],
            name="Transmission", line=dict(color=COLORS[3], width=2.5),
        ))
        fig.update_xaxes(title_text="Distance (m)")
        fig.update_yaxes(title_text="Transmission", range=[0, 1.05])
        show_figure(fig)

# ── Model Comparison ────────────────────────────────────────────
from harrington_labs.comparison.ui import model_comparison_panel, reference_upload_panel

model_comparison_panel(
    sim_x=prof["z_m"],
    sim_y=prof["w_turbulence_m"] * 100,
    x_label="Distance",
    y_label="Beam Radius",
    x_unit="m",
    y_unit="cm",
    panel_title="Model Comparison — Beam Radius",
    key_prefix="beam_radius",
)

reference_upload_panel(key_prefix="beam_ref", save_dir="data/references")
