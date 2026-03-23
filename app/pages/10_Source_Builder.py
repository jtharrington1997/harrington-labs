
"""
pages/4_Source_Builder.py — Laser Source Builder
Design a laser system from gain medium, pump, resonator, and output coupler.
"""
from __future__ import annotations

import math

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from harrington_labs.ui import render_header
from harrington_labs.ui import lab_panel, make_figure, show_figure, COLORS
from harrington_labs.lmi.ui.formatting import (
    fmt_energy_j,
    fmt_frequency_hz,
    fmt_fluence_j_cm2,
    fmt_power_w,
)

st.set_page_config(page_title="Source Builder", layout="wide")
render_header("Source Builder", "Gain medium • Pump source • Resonator • Output coupler design")

PLOT_KW = dict(template="plotly_white", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")

GAIN_MEDIA = {
    "Nd:YAG": {"wavelength_nm": 1064, "cross_section_cm2": 2.8e-19, "upper_lifetime_us": 230, "saturation_fluence_j_cm2": 0.66, "bandwidth_nm": 0.45},
    "Nd:YVO4": {"wavelength_nm": 1064, "cross_section_cm2": 25e-19, "upper_lifetime_us": 100, "saturation_fluence_j_cm2": 0.076, "bandwidth_nm": 0.8},
    "Ti:Sapphire": {"wavelength_nm": 800, "cross_section_cm2": 3.0e-19, "upper_lifetime_us": 3.2, "saturation_fluence_j_cm2": 0.9, "bandwidth_nm": 230},
    "Yb:YAG": {"wavelength_nm": 1030, "cross_section_cm2": 2.0e-20, "upper_lifetime_us": 951, "saturation_fluence_j_cm2": 9.5, "bandwidth_nm": 6},
    "Yb:Glass (fiber)": {"wavelength_nm": 1070, "cross_section_cm2": 5e-21, "upper_lifetime_us": 800, "saturation_fluence_j_cm2": 38, "bandwidth_nm": 40},
    "Er:Glass (fiber)": {"wavelength_nm": 1550, "cross_section_cm2": 8e-21, "upper_lifetime_us": 8000, "saturation_fluence_j_cm2": 24, "bandwidth_nm": 55},
    "Cr:LiSAF": {"wavelength_nm": 850, "cross_section_cm2": 4.8e-20, "upper_lifetime_us": 67, "saturation_fluence_j_cm2": 4.7, "bandwidth_nm": 180},
    "Tm:YAG": {"wavelength_nm": 2013, "cross_section_cm2": 1.5e-21, "upper_lifetime_us": 10000, "saturation_fluence_j_cm2": 130, "bandwidth_nm": 40},
}

with lab_panel():
    st.subheader("Laser Source Builder")
    st.caption("Build a first-pass resonator estimate with cleaner engineering units and mobile-friendly plots.")

with lab_panel():
    st.subheader("1. Gain Medium")
    medium_name = st.selectbox("Select gain medium", list(GAIN_MEDIA.keys()))
    medium = GAIN_MEDIA[medium_name]

    st.markdown(
        f"""
| Property | Value |
|---|---|
| Lasing wavelength | **{medium['wavelength_nm']:.0f} nm** |
| Stimulated emission cross-section | **{medium['cross_section_cm2']:.2e} cm²** |
| Upper-state lifetime | **{medium['upper_lifetime_us']:.1f} µs** |
| Saturation fluence | **{medium['saturation_fluence_j_cm2']:.3g} J/cm²** |
| Gain bandwidth | **{medium['bandwidth_nm']:.3g} nm** |
"""
    )

with lab_panel():
    st.subheader("2. Resonator")
    col1, col2 = st.columns(2)
    with col1:
        cavity_length_m = st.number_input("Cavity length (m)", 0.01, 10.0, 0.3, step=0.05)
        output_coupler_r = st.slider("Output coupler reflectivity (%)", 1, 99, 80) / 100
        internal_loss_pct = st.slider("Internal round-trip loss (%)", 0, 50, 5) / 100
    with col2:
        rod_length_mm = st.number_input("Gain medium length (mm)", 1.0, 200.0, 50.0)
        rod_diameter_mm = st.number_input("Gain medium diameter (mm)", 0.5, 25.0, 3.0)
        num_passes = st.selectbox("Passes per round trip", [1, 2], index=1)

    fsr_hz = 3e8 / (2 * cavity_length_m)
    rt_time_ns = 2 * cavity_length_m / 3e8 * 1e9
    est_modes = int(max(1, medium["bandwidth_nm"] * 1e-9 * 3e8 / (medium["wavelength_nm"] * 1e-9) ** 2 / fsr_hz))

    c1, c2, c3 = st.columns(3)
    c1.metric("Free spectral range", fmt_frequency_hz(fsr_hz))
    c2.metric("Round-trip time", f"{rt_time_ns:.2f} ns")
    c3.metric("Estimated longitudinal modes", f"{est_modes:,}")

with lab_panel():
    st.subheader("3. Pump Source")
    col1, col2 = st.columns(2)
    with col1:
        pump_type = st.selectbox("Pump type", ["Diode (end-pumped)", "Diode (side-pumped)", "Flashlamp", "Another laser"])
        pump_power_w = st.number_input("Pump power (W)", 0.1, 10000.0, 20.0)
    with col2:
        pump_wavelength = st.number_input("Pump wavelength (nm)", 100.0, 2000.0, 808.0)
        pump_efficiency = st.slider("Pump absorption efficiency (%)", 10, 100, 85) / 100

    quantum_defect = 1 - pump_wavelength / medium["wavelength_nm"]
    absorbed_pump = pump_power_w * pump_efficiency
    quantum_limit_w = absorbed_pump * (1 - quantum_defect)

    c1, c2, c3 = st.columns(3)
    c1.metric("Quantum defect", f"{quantum_defect * 100:.1f}%")
    c2.metric("Absorbed pump", fmt_power_w(absorbed_pump))
    c3.metric("Quantum-limited output", fmt_power_w(quantum_limit_w))

with lab_panel():
    st.subheader("4. Estimated Output")

    total_loss = -math.log(output_coupler_r) + internal_loss_pct
    rod_area_cm2 = math.pi * (rod_diameter_mm / 20) ** 2
    eta_slope = (1 - quantum_defect) * pump_efficiency * (-math.log(output_coupler_r)) / total_loss if total_loss > 0 else 0
    eta_slope = min(eta_slope, 0.8)

    threshold_pump_w = medium["saturation_fluence_j_cm2"] * rod_area_cm2 * total_loss / (
        medium["upper_lifetime_us"] * 1e-6 * pump_efficiency * (1 - quantum_defect)
    ) if medium["upper_lifetime_us"] > 0 else pump_power_w * 0.5
    output_power_w = max(0, eta_slope * (absorbed_pump - threshold_pump_w * pump_efficiency))

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Slope efficiency", f"{eta_slope * 100:.1f}%")
    c2.metric("Threshold pump", fmt_power_w(threshold_pump_w))
    c3.metric("Estimated output", fmt_power_w(output_power_w))
    c4.metric("Overall efficiency", f"{(output_power_w / pump_power_w * 100):.1f}%" if pump_power_w > 0 else "N/A")

    pump_range = np.linspace(0, pump_power_w * 1.5, 200)
    output_range = np.maximum(0, eta_slope * (pump_range * pump_efficiency - threshold_pump_w * pump_efficiency))

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=pump_range, y=output_range,
        mode="lines", line=dict(width=2),
        name="Output power",
    ))
    fig.add_trace(go.Scatter(
        x=[pump_power_w], y=[output_power_w],
        mode="markers", marker=dict(size=11, symbol="star"),
        name="Operating point",
    ))
    fig.add_vline(x=threshold_pump_w, line_dash="dash", annotation_text="Threshold")
    fig.update_layout(
        xaxis_title="Pump power (W)",
        yaxis_title="Estimated output power (W)",
        height=360,
        **PLOT_KW,
    )
    st.plotly_chart(fig, width="stretch")

with lab_panel():
    st.subheader("5. Quick Design Summary")
    summary = [
        f"Gain medium: {medium_name}",
        f"Pump architecture: {pump_type}",
        f"Cavity FSR: {fmt_frequency_hz(fsr_hz)}",
        f"Saturation fluence: {fmt_fluence_j_cm2(medium['saturation_fluence_j_cm2'])}",
        f"Estimated threshold pump: {fmt_power_w(threshold_pump_w)}",
        f"Estimated operating output: {fmt_power_w(output_power_w)}",
    ]
    for item in summary:
        st.markdown(f"- {item}")
