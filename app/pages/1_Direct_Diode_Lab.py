"""1_Direct_Diode_Lab.py — Direct Diode Laser Lab Simulator."""
import streamlit as st
import plotly.graph_objects as go
from harrington_labs.ui import render_header, lab_panel, make_figure, show_figure, warning_box, COLORS
from harrington_labs.domain import DiodeLaserParams
from harrington_labs.simulation.direct_diode import (
    run_direct_diode_simulation, spectral_beam_combining,
)

st.set_page_config(page_title="Direct Diode Lab", layout="wide")
render_header("Direct Diode Lab", "L-I curves • Thermal behavior • Far-field • Beam combining")

# ── Sidebar inputs ───────────────────────────────────────────────
from harrington_labs.ui.shared_state import get_shared_beam, push_beam_button, shared_beam_badge
sb = get_shared_beam()
shared_beam_badge()

st.sidebar.header("Diode Parameters")
wavelength = st.sidebar.number_input("Wavelength (nm)", 400.0, 2000.0, sb["wavelength_nm"] if 400 <= sb["wavelength_nm"] <= 2000 else 976.0, 1.0)
power = st.sidebar.number_input("Rated Power (W)", 0.1, 500.0, 50.0, 1.0)
threshold = st.sidebar.number_input("Threshold Current (A)", 0.01, 10.0, 0.5, 0.05)
operating = st.sidebar.number_input("Operating Current (A)", 0.1, 50.0, 5.0, 0.1)
slope_eff = st.sidebar.slider("Slope Efficiency", 0.1, 1.0, 0.65, 0.01)
fast_div = st.sidebar.number_input("Fast Axis Divergence (°)", 5.0, 60.0, 35.0, 1.0)
slow_div = st.sidebar.number_input("Slow Axis Divergence (°)", 1.0, 30.0, 10.0, 1.0)
t0 = st.sidebar.number_input("T₀ Characteristic Temp (K)", 50.0, 500.0, 150.0, 10.0)
t1 = st.sidebar.number_input("T₁ Slope Char. Temp (K)", 100.0, 1000.0, 400.0, 10.0)
r_th = st.sidebar.number_input("Thermal Resistance (K/W)", 0.1, 10.0, 1.5, 0.1)
heatsink_t = st.sidebar.number_input("Heatsink Temp (°C)", -10.0, 80.0, 25.0, 1.0)

params = DiodeLaserParams(
    wavelength_nm=wavelength,
    power_w=power,
    threshold_current_a=threshold,
    operating_current_a=operating,
    slope_efficiency=slope_eff,
    beam_divergence_fast_deg=fast_div,
    beam_divergence_slow_deg=slow_div,
    t0_k=t0,
    t1_k=t1,
    thermal_resistance_k_w=r_th,
)

result = run_direct_diode_simulation(params, heatsink_temp_c=heatsink_t)
li = result.data["li_curve"]
ff = result.data["far_field"]
wl_temp = result.data["wavelength_temp"]

# ── Warnings ─────────────────────────────────────────────────────
warning_box(result.warnings)

# ── L-I Curve ────────────────────────────────────────────────────
with lab_panel("L-I Characteristic"):
    fig = make_figure("Optical Power vs Drive Current")
    fig.add_trace(go.Scatter(x=li["current_a"], y=li["power_w"],
                             name="Power", line=dict(color=COLORS[0], width=2.5)))
    fig.update_xaxes(title_text="Current (A)")
    fig.update_yaxes(title_text="Power (W)")
    show_figure(fig)

col1, col2 = st.columns(2)

with col1:
    with lab_panel("Wall-Plug Efficiency"):
        fig = make_figure("WPE vs Current")
        fig.add_trace(go.Scatter(x=li["current_a"], y=li["efficiency"],
                                 name="WPE", line=dict(color=COLORS[2], width=2)))
        fig.update_xaxes(title_text="Current (A)")
        fig.update_yaxes(title_text="Efficiency", tickformat=".0%")
        show_figure(fig)

with col2:
    with lab_panel("Junction Temperature"):
        fig = make_figure("Junction Temperature vs Current")
        fig.add_trace(go.Scatter(x=li["current_a"], y=li["junction_temp_c"],
                                 name="T_j", line=dict(color=COLORS[1], width=2)))
        fig.update_xaxes(title_text="Current (A)")
        fig.update_yaxes(title_text="Temperature (°C)")
        show_figure(fig)

# ── Far-field ────────────────────────────────────────────────────
with lab_panel("Far-Field Pattern"):
    fig = make_figure("Far-Field Intensity Profile")
    fig.add_trace(go.Scatter(x=ff["angle_deg"], y=ff["fast_axis"],
                             name="Fast Axis", line=dict(color=COLORS[0], width=2)))
    fig.add_trace(go.Scatter(x=ff["angle_deg"], y=ff["slow_axis"],
                             name="Slow Axis", line=dict(color=COLORS[1], width=2)))
    fig.update_xaxes(title_text="Angle (°)")
    fig.update_yaxes(title_text="Normalized Intensity")
    show_figure(fig)

# ── Wavelength drift ─────────────────────────────────────────────
with lab_panel("Wavelength vs Temperature"):
    fig = make_figure("Emission Wavelength Drift")
    fig.add_trace(go.Scatter(x=wl_temp["temperature_c"], y=wl_temp["wavelength_nm"],
                             name="λ", line=dict(color=COLORS[3], width=2)))
    fig.update_xaxes(title_text="Junction Temperature (°C)")
    fig.update_yaxes(title_text="Wavelength (nm)")
    show_figure(fig)

# ── Beam combining ───────────────────────────────────────────────
with lab_panel("Spectral Beam Combining"):
    st.caption("Estimate combined output from multiple emitters through a diffraction grating.")
    c1, c2, c3 = st.columns(3)
    n_emit = c1.number_input("Emitter Count", 2, 100, 10)
    per_emit_w = c2.number_input("Power per Emitter (W)", 1.0, 200.0, 10.0, 1.0)
    grating_eff = c3.slider("Grating Efficiency", 0.5, 0.99, 0.92, 0.01)

    sbc = spectral_beam_combining(n_emit, per_emit_w, grating_eff)
    m1, m2, m3 = st.columns(3)
    m1.metric("Raw Power", f"{sbc['raw_power_w']:.0f} W")
    m2.metric("Combined Power", f"{sbc['combined_power_w']:.1f} W")
    m3.metric("Overall Efficiency", f"{sbc['combining_efficiency']:.1%}")

# ── Model Comparison ────────────────────────────────────────────
from harrington_labs.comparison.ui import model_comparison_panel, reference_upload_panel

model_comparison_panel(
    sim_x=li["current_a"],
    sim_y=li["power_w"],
    x_label="Current",
    y_label="Power",
    x_unit="A",
    y_unit="W",
    panel_title="Model Comparison — L-I Curve",
    key_prefix="diode_li",
)

reference_upload_panel(key_prefix="diode_ref", save_dir="data/references")
