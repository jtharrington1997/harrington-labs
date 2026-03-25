"""2_Fiber_Laser_Lab.py — Fiber Laser Lab Simulator."""
import streamlit as st
import plotly.graph_objects as go
from harrington_labs.ui import render_header, lab_panel, make_figure, show_figure, warning_box, COLORS
from harrington_labs.domain import FiberLaserParams, FiberType
from harrington_labs.simulation.fiber_laser import run_fiber_laser_simulation

st.set_page_config(page_title="Fiber Laser Lab", layout="wide")
render_header("Fiber Laser Lab", "Gain modeling • Nonlinear limits • Thermal analysis • Mode properties • Beam combining")

# ── Sidebar inputs ───────────────────────────────────────────────
from harrington_labs.ui.shared_state import get_shared_beam, shared_beam_badge
from harrington_labs.ui.db_sidebar import source_and_material_sidebar
sb = get_shared_beam()
shared_beam_badge()
db_laser, db_material = source_and_material_sidebar("fiber")

st.sidebar.header("Fiber Parameters")
fiber_type = st.sidebar.selectbox("Fiber Type", [f.value for f in FiberType], index=3)
core_d = st.sidebar.number_input("Core Diameter (µm)", 1.0, 100.0, 25.0, 1.0)
clad_d = st.sidebar.number_input("Cladding Diameter (µm)", 50.0, 800.0, 250.0, 10.0)
na = st.sidebar.number_input("NA", 0.01, 0.50, 0.065, 0.005, format="%.3f")
length = st.sidebar.number_input("Fiber Length (m)", 0.1, 30.0, 3.0, 0.1)

st.sidebar.header("Pump & Signal")
_def_pump_wl = db_laser.wavelength_nm if db_laser and 800 <= db_laser.wavelength_nm <= 1100 else 976.0
_def_pump_pwr = db_laser.power_w if db_laser else 50.0
pump_wl = st.sidebar.number_input("Pump Wavelength (nm)", 800.0, 1100.0, _def_pump_wl, 1.0)
pump_pwr = st.sidebar.number_input("Pump Power (W)", 0.1, 500.0, _def_pump_pwr, 1.0)
sig_wl = st.sidebar.number_input("Signal Wavelength (nm)", 900.0, 2200.0, 1064.0, 1.0)
seed_pwr = st.sidebar.number_input("Seed Power (W)", 0.0001, 10.0, 0.01, 0.001, format="%.4f")

st.sidebar.header("Doping")
dopant = st.sidebar.selectbox("Dopant", ["Yb", "Er", "Tm", "Ho"])
doping = st.sidebar.number_input("Doping (ppm)", 100, 50000, 1000, 100)
bg_loss = st.sidebar.number_input("Background Loss (dB/m)", 0.0, 0.1, 0.005, 0.001, format="%.3f")

params = FiberLaserParams(
    fiber_type=FiberType(fiber_type),
    core_diameter_um=core_d,
    cladding_diameter_um=clad_d,
    na=na,
    fiber_length_m=length,
    pump_wavelength_nm=pump_wl,
    pump_power_w=pump_pwr,
    signal_wavelength_nm=sig_wl,
    signal_seed_power_w=seed_pwr,
    doping_concentration_ppm=doping,
    dopant=dopant,
    background_loss_db_m=bg_loss,
)

result = run_fiber_laser_simulation(params)
warning_box(result.warnings)

# ── Fiber mode summary ───────────────────────────────────────────
with lab_panel("Fiber Mode Properties"):
    fp = result.data["fiber_params"]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("V-Number", f"{fp['v_number']:.2f}")
    c2.metric("MFD", f"{fp['mfd_um']:.1f} µm")
    c3.metric("Effective Area", f"{fp['a_eff_um2']:.0f} µm²")
    single = "Yes" if fp['v_number'] <= 2.405 else "No"
    c4.metric("Single-Mode", single)

# ── Amplifier performance ────────────────────────────────────────
with lab_panel("Amplifier Performance"):
    amp = result.data["amplifier"]
    c1, c2, c3 = st.columns(3)
    c1.metric("Output Power", f"{amp['signal_out_w']:.2f} W")
    c2.metric("Gain", f"{amp['gain_db']:.1f} dB")
    c3.metric("Optical Efficiency", f"{amp['optical_efficiency']:.1%}")

    c4, c5, c6 = st.columns(3)
    c4.metric("QE Limit", f"{amp['quantum_efficiency_limit']:.1%}")
    c5.metric("Pump Absorbed", f"{amp['pump_absorbed_w']:.1f} W")
    c6.metric("Heat Load", f"{amp['heat_load_w']:.1f} W")

# ── Gain profile along fiber ─────────────────────────────────────
with lab_panel("Power Evolution Along Fiber"):
    prof = result.data["gain_profile"]
    fig = make_figure("Pump & Signal Power vs Position")
    fig.add_trace(go.Scatter(x=prof["z_m"], y=prof["pump_w"],
                             name="Pump", line=dict(color=COLORS[1], width=2)))
    fig.add_trace(go.Scatter(x=prof["z_m"], y=prof["signal_w"],
                             name="Signal", line=dict(color=COLORS[0], width=2.5)))
    fig.update_xaxes(title_text="Position along fiber (m)")
    fig.update_yaxes(title_text="Power (W)", type="log")
    show_figure(fig)

# ── Nonlinear thresholds ─────────────────────────────────────────
col1, col2 = st.columns(2)

with col1:
    with lab_panel("Nonlinear Thresholds"):
        nl = result.data["nonlinear"]
        c1, c2 = st.columns(2)
        sbs_val = nl["sbs_threshold_w"]
        srs_val = nl["srs_threshold_w"]
        c1.metric("SBS Threshold", f"{sbs_val:.1f} W" if sbs_val < 1e6 else ">> 1 MW")
        c2.metric("SRS Threshold", f"{srs_val:.0f} W" if srs_val < 1e6 else ">> 1 MW")

        out_pwr = amp["signal_out_w"]
        if out_pwr > 0:
            sbs_margin = sbs_val / out_pwr if sbs_val < 1e6 else float("inf")
            srs_margin = srs_val / out_pwr if srs_val < 1e6 else float("inf")
            st.caption(f"SBS margin: {sbs_margin:.1f}× | SRS margin: {srs_margin:.1f}×")

with col2:
    with lab_panel("Thermal Summary"):
        th = result.data["thermal"]
        c1, c2, c3 = st.columns(3)
        c1.metric("Core Temp Rise", f"{th['core_temp_rise_k']:.1f} K")
        c2.metric("Surface Temp Rise", f"{th['surface_temp_rise_k']:.1f} K")
        c3.metric("Heat Load/m", f"{th['linear_heat_load_w_m']:.1f} W/m")

# ── Beam combining ───────────────────────────────────────────
from harrington_labs.simulation.direct_diode import spectral_beam_combining

with lab_panel("Spectral Beam Combining"):
    st.caption("Estimate combined output from multiple fiber amplifier channels through a diffraction grating.")
    c1, c2, c3 = st.columns(3)
    n_channels = c1.number_input("Channel Count", 2, 100, 4, key="fiber_sbc_n")
    per_channel_w = c2.number_input(
        "Power per Channel (W)", 1.0, 5000.0,
        float(round(amp["signal_out_w"], 1)), 1.0,
        key="fiber_sbc_pwr",
    )
    grating_eff = c3.slider("Grating Efficiency", 0.5, 0.99, 0.92, 0.01, key="fiber_sbc_eff")

    sbc = spectral_beam_combining(n_channels, per_channel_w, grating_eff)
    m1, m2, m3 = st.columns(3)
    m1.metric("Raw Power", f"{sbc['raw_power_w']:.0f} W")
    m2.metric("Combined Power", f"{sbc['combined_power_w']:.1f} W")
    m3.metric("Overall Efficiency", f"{sbc['combining_efficiency']:.1%}")

# ── Model Comparison ────────────────────────────────────────────
from harrington_labs.comparison.ui import model_comparison_panel, reference_upload_panel

model_comparison_panel(
    sim_x=prof["z_m"],
    sim_y=prof["signal_w"],
    x_label="Fiber Position",
    y_label="Signal Power",
    x_unit="m",
    y_unit="W",
    panel_title="Model Comparison — Signal Evolution",
    key_prefix="fiber_signal",
)

reference_upload_panel(key_prefix="fiber_ref", save_dir="data/references")
