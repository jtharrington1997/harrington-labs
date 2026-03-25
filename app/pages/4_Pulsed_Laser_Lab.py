"""4_Pulsed_Laser_Lab.py — Pulsed Laser Lab Simulator."""
import streamlit as st
import plotly.graph_objects as go
from harrington_labs.ui import render_header, lab_panel, make_figure, show_figure, warning_box, COLORS
from harrington_labs.domain import BeamParams, PulsedSource, PulseShape, LaserType
from harrington_labs.simulation.pulsed_laser import (
    run_pulsed_laser_simulation, open_aperture_zscan,
)

st.set_page_config(page_title="Pulsed Laser Lab", layout="wide")
render_header("Pulsed Laser Lab", "Ultrafast pulses • Temporal & spectral profiles • Dispersion • z-Scan")

# ── Sidebar ──────────────────────────────────────────────────────
from harrington_labs.ui.shared_state import get_shared_beam, push_beam_button, shared_beam_badge
sb = get_shared_beam()
shared_beam_badge()

st.sidebar.header("Laser Source")
wavelength = st.sidebar.number_input("Wavelength (nm)", 200.0, 12000.0, sb["wavelength_nm"], 1.0)
avg_power = st.sidebar.number_input("Average Power (W)", 0.001, 100.0, sb["power_w"], 0.1)
rep_rate_khz = st.sidebar.number_input("Rep Rate (kHz)", 0.001, 10000.0, sb["rep_rate_hz"] / 1e3, 0.1)
pulse_width_fs = st.sidebar.number_input("Pulse Width (fs)", 1.0, 100000.0, sb["pulse_width_s"] * 1e15, 1.0)
beam_d = st.sidebar.number_input("Beam Diameter (mm)", 0.01, 50.0, sb["beam_diameter_mm"], 0.1)
shape = st.sidebar.selectbox("Pulse Shape", [s.value for s in PulseShape])
st.sidebar.markdown("---")
push_beam_button(
    wavelength_nm=wavelength, power_w=avg_power,
    beam_diameter_mm=beam_d, m_squared=1.0,
    rep_rate_hz=rep_rate_khz * 1e3, pulse_width_s=pulse_width_fs * 1e-15,
    key="pulsed_push",
)

beam = BeamParams(wavelength_nm=wavelength, power_w=avg_power, beam_diameter_mm=beam_d)
pulse = PulsedSource(
    beam=beam,
    rep_rate_hz=rep_rate_khz * 1e3,
    pulse_width_s=pulse_width_fs * 1e-15,
    pulse_shape=PulseShape(shape),
)

result = run_pulsed_laser_simulation(pulse)
warning_box(result.warnings)

# ── Pulse summary ────────────────────────────────────────────────
with lab_panel("Pulse Summary"):
    ps = result.data["pulse_summary"]
    c1, c2, c3, c4 = st.columns(4)

    # Smart formatting
    pe = ps["pulse_energy_j"]
    if pe >= 1e-3:
        c1.metric("Pulse Energy", f"{pe*1e3:.2f} mJ")
    elif pe >= 1e-6:
        c1.metric("Pulse Energy", f"{pe*1e6:.2f} µJ")
    else:
        c1.metric("Pulse Energy", f"{pe*1e9:.2f} nJ")

    pp = ps["peak_power_w"]
    if pp >= 1e9:
        c2.metric("Peak Power", f"{pp/1e9:.2f} GW")
    elif pp >= 1e6:
        c2.metric("Peak Power", f"{pp/1e6:.2f} MW")
    elif pp >= 1e3:
        c2.metric("Peak Power", f"{pp/1e3:.2f} kW")
    else:
        c2.metric("Peak Power", f"{pp:.2f} W")

    c3.metric("Fluence", f"{ps['fluence_j_cm2']:.3f} J/cm²")
    c4.metric("TBP Bandwidth", f"{ps['bandwidth_nm']:.2f} nm")

# ── Temporal profile ─────────────────────────────────────────────
col1, col2 = st.columns(2)

with col1:
    with lab_panel("Temporal Profile"):
        temp = result.data["temporal"]
        fig = make_figure("Pulse Temporal Profile")
        fig.add_trace(go.Scatter(
            x=temp["time_s"] * 1e15, y=temp["intensity"],
            name="Intensity", line=dict(color=COLORS[0], width=2.5),
        ))
        fig.update_xaxes(title_text="Time (fs)")
        fig.update_yaxes(title_text="Normalized Intensity")
        show_figure(fig)

with col2:
    with lab_panel("Spectral Profile"):
        spec = result.data["spectral"]
        fig = make_figure(f"Transform-Limited Spectrum (Δλ = {spec['bandwidth_nm']:.2f} nm)")
        fig.add_trace(go.Scatter(
            x=spec["wavelength_nm"], y=spec["spectrum"],
            name="Spectrum", line=dict(color=COLORS[1], width=2.5),
        ))
        fig.update_xaxes(title_text="Wavelength (nm)")
        fig.update_yaxes(title_text="Normalized Spectral Intensity")
        show_figure(fig)

# ── Autocorrelation ──────────────────────────────────────────────
with lab_panel("Intensity Autocorrelation"):
    ac = result.data["autocorrelation"]
    fig = make_figure(f"Autocorrelation (deconv. factor = {ac['deconvolution_factor']:.3f})")
    fig.add_trace(go.Scatter(
        x=ac["delay_s"] * 1e15, y=ac["autocorrelation"],
        name="IAC", line=dict(color=COLORS[2], width=2),
    ))
    fig.update_xaxes(title_text="Delay (fs)")
    fig.update_yaxes(title_text="Normalized IAC Signal")
    show_figure(fig)

# ── Dispersion management ────────────────────────────────────────
with lab_panel("Dispersion Management"):
    disp = result.data["dispersion_scan"]
    fig = make_figure("Pulse Width vs Group Delay Dispersion")
    fig.add_trace(go.Scatter(
        x=disp["gdd_fs2"], y=disp["pulse_width_s"] * 1e15,
        name="τ(GDD)", line=dict(color=COLORS[3], width=2.5),
    ))
    fig.add_hline(y=pulse_width_fs, line_dash="dot", line_color=COLORS[0],
                  annotation_text="Transform limit")
    fig.update_xaxes(title_text="GDD (fs²)")
    fig.update_yaxes(title_text="Pulse Width (fs)", type="log")
    show_figure(fig)

# ── z-Scan ───────────────────────────────────────────────────────
with lab_panel("Open-Aperture z-Scan"):
    st.caption("Simulate two-photon absorption z-scan measurement.")
    c1, c2, c3 = st.columns(3)
    beta = c1.number_input("β (cm/W)", 0.0, 1e-6, 1e-10, 1e-11, format="%.2e")
    thickness = c2.number_input("Sample Thickness (mm)", 0.01, 10.0, 1.0, 0.1)
    z_range = c3.number_input("Scan Range (mm)", 1.0, 100.0, 20.0, 1.0)

    zscan = open_aperture_zscan(pulse, beta, thickness / 10, z_range)
    fig = make_figure("Open-Aperture z-Scan")
    fig.add_trace(go.Scatter(
        x=zscan["z_mm"], y=zscan["transmission"],
        name="T(z)", line=dict(color=COLORS[0], width=2.5),
    ))
    fig.add_hline(y=1.0, line_dash="dot", line_color="gray")
    fig.update_xaxes(title_text="Sample Position (mm)")
    fig.update_yaxes(title_text="Normalized Transmission")
    show_figure(fig)
    st.caption(f"Rayleigh range: {zscan['z_rayleigh_mm']:.2f} mm")

# ── Model Comparison ────────────────────────────────────────────
from harrington_labs.comparison.ui import model_comparison_panel, reference_upload_panel

comparison_result = model_comparison_panel(
    sim_x=zscan["z_mm"],
    sim_y=zscan["transmission"],
    x_label="Stage Position",
    y_label="Normalized Transmission",
    x_unit="mm",
    panel_title="Model Comparison — z-Scan",
    key_prefix="pulsed_zscan",
)

# ── Reference Library ───────────────────────────────────────────
reference_upload_panel(key_prefix="pulsed_ref", save_dir="data/references")
