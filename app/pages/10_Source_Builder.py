
"""
pages/10_Source_Builder.py — Laser Source Builder
Design a laser system from gain medium, pump, resonator, and output coupler.
Includes QD-doped pulsed fiber laser testbed.
"""
from __future__ import annotations

import math

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from harrington_labs.ui import render_header
from harrington_labs.ui import lab_panel, make_figure, show_figure, COLORS
from harrington_labs.ui.db_sidebar import source_and_material_sidebar
from harrington_labs.lmi.ui.formatting import (
    fmt_energy_j,
    fmt_frequency_hz,
    fmt_fluence_j_cm2,
    fmt_power_w,
)

st.set_page_config(page_title="Source Builder", layout="wide")
render_header("Source Builder", "Resonator design • QD fiber laser testbed • Pulsed source engineering")

db_laser, db_material = source_and_material_sidebar("srcbuild")

from harrington_labs.ui import PLOT_LAYOUT as PLOT_KW

tabs = st.tabs(["Resonator Builder", "QD Fiber Laser Testbed"])

# ════════════════════════════════════════════════════════════════════
# TAB 1: ORIGINAL RESONATOR BUILDER
# ════════════════════════════════════════════════════════════════════
with tabs[0]:
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
            _def_pump_w = db_laser.power_w if db_laser else 20.0
            pump_power_w = st.number_input("Pump power (W)", 0.1, 10000.0, _def_pump_w)
        with col2:
            _def_pump_wl = db_laser.wavelength_nm if db_laser else 808.0
            pump_wavelength = st.number_input("Pump wavelength (nm)", 100.0, 2000.0, _def_pump_wl)
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
        fig.add_trace(go.Scatter(x=pump_range, y=output_range, mode="lines", line=dict(width=2), name="Output power"))
        fig.add_trace(go.Scatter(x=[pump_power_w], y=[output_power_w], mode="markers", marker=dict(size=11, symbol="star"), name="Operating point"))
        fig.add_vline(x=threshold_pump_w, line_dash="dash", annotation_text="Threshold")
        fig.update_layout(xaxis_title="Pump power (W)", yaxis_title="Estimated output power (W)", height=360, **PLOT_KW)
        st.plotly_chart(fig, width="stretch")

# ════════════════════════════════════════════════════════════════════
# TAB 2: QD FIBER LASER TESTBED
# ════════════════════════════════════════════════════════════════════
with tabs[1]:
    from harrington_labs.simulation.qd_fiber_laser import (
        QDFiberLaserParams, simulate_qd_fiber_laser,
        _QD_BULK, qd_emission_wavelength_nm,
    )

    st.sidebar.header("QD Fiber Laser")

    # QD material
    qd_mat = st.sidebar.selectbox("QD Material", list(_QD_BULK.keys()), index=list(_QD_BULK.keys()).index("PbS"), key="qdf_mat")
    qd_diam = st.sidebar.number_input("QD Diameter (nm)", 1.0, 20.0, 5.0, 0.5, key="qdf_diam")
    qd_dist = st.sidebar.slider("Size Distribution (%)", 1.0, 20.0, 5.0, 0.5, key="qdf_dist")
    qd_conc_exp = st.sidebar.slider("log₁₀(QD concentration / cm⁻³)", 14.0, 19.0, 17.0, 0.5, key="qdf_conc")
    qd_qy = st.sidebar.slider("Intrinsic QY", 0.01, 1.0, 0.3, 0.01, key="qdf_qy")

    # Fiber
    st.sidebar.markdown("---")
    core_d = st.sidebar.number_input("Core Diameter (µm)", 1.0, 50.0, 6.0, 0.5, key="qdf_core")
    fiber_na = st.sidebar.number_input("Fiber NA", 0.05, 0.5, 0.12, 0.01, key="qdf_na")
    fiber_len = st.sidebar.number_input("Fiber Length (m)", 0.01, 10.0, 1.0, 0.1, key="qdf_len")
    bg_loss = st.sidebar.number_input("Background Loss (dB/m)", 0.01, 10.0, 0.5, 0.1, key="qdf_loss")

    # Pump
    st.sidebar.markdown("---")
    _def_pump_wl_qd = db_laser.wavelength_nm if db_laser else 808.0
    _def_pump_mw_qd = db_laser.power_w * 1e3 if db_laser else 500.0
    pump_wl = st.sidebar.number_input("Pump λ (nm)", 400.0, 1500.0, _def_pump_wl_qd, 1.0, key="qdf_pwl")
    pump_mw = st.sidebar.number_input("Pump Power (mW)", 1.0, 10000.0, min(_def_pump_mw_qd, 10000.0), 10.0, key="qdf_pmw")
    pump_eff = st.sidebar.slider("Pump Coupling Efficiency (%)", 10, 100, 70, key="qdf_peff") / 100

    # Pulse mode
    st.sidebar.markdown("---")
    pulse_mode = st.sidebar.selectbox("Operation Mode", ["Q-switched", "Mode-locked", "CW"], key="qdf_mode")
    rep_rate_khz = 100.0
    q_hold_us = 10.0
    sa_depth = 0.3
    if pulse_mode == "Q-switched":
        rep_rate_khz = st.sidebar.number_input("Rep Rate (kHz)", 0.1, 10000.0, 100.0, 10.0, key="qdf_rr")
        q_hold_us = st.sidebar.number_input("Q-switch Hold Time (µs)", 0.1, 1000.0, 10.0, 1.0, key="qdf_qhold")
    elif pulse_mode == "Mode-locked":
        sa_depth = st.sidebar.slider("SA Modulation Depth", 0.01, 0.9, 0.3, 0.01, key="qdf_sa")

    oc_r = st.sidebar.slider("Output Coupler Reflectivity (%)", 1, 99, 50, key="qdf_oc") / 100

    # Run simulation
    params = QDFiberLaserParams(
        qd_material=qd_mat, qd_diameter_nm=qd_diam, qd_size_distribution_pct=qd_dist,
        qd_concentration_cm3=10**qd_conc_exp, qd_quantum_yield=qd_qy,
        core_diameter_um=core_d, cladding_diameter_um=125.0, fiber_na=fiber_na,
        fiber_length_m=fiber_len, background_loss_db_m=bg_loss,
        pump_wavelength_nm=pump_wl, pump_power_mw=pump_mw, pump_coupling_efficiency=pump_eff,
        pulse_mode=pulse_mode, rep_rate_khz=rep_rate_khz,
        q_switch_hold_time_us=q_hold_us, saturable_absorber_modulation_depth=sa_depth,
        output_coupler_reflectivity=oc_r,
    )
    result = simulate_qd_fiber_laser(params)
    d = result.data
    from harrington_labs.ui import warning_box
    warning_box(result.warnings)

    # ── QD Gain Medium Summary ──
    with lab_panel("QD Gain Medium"):
        qd = d["qd"]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Emission λ", f"{qd['emission_nm']:.0f} nm")
        c2.metric("Bandgap", f"{qd['bandgap_ev']:.3f} eV")
        c3.metric("Gain BW", f"{qd['gain_bandwidth_nm']:.0f} nm")
        c4.metric("Effective QY", f"{qd['effective_qy']:.1%}")

        c5, c6, c7, c8 = st.columns(4)
        c5.metric("σ_abs", f"{qd['sigma_abs_cm2']:.2e} cm²")
        c6.metric("σ_em", f"{qd['sigma_em_cm2']:.2e} cm²")
        c7.metric("τ_rad", f"{qd['radiative_lifetime_ns']:.1f} ns")
        c8.metric("τ_Auger", f"{qd['auger_lifetime_ns']:.1f} ns")

    # ── Fiber Properties ──
    with lab_panel("Fiber Properties"):
        fb = d["fiber"]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("V-number", f"{fb['v_number']:.2f}")
        c2.metric("MFD", f"{fb['mfd_um']:.1f} µm")
        c3.metric("A_eff", f"{fb['a_eff_um2']:.0f} µm²")
        c4.metric("Single-Mode", "Yes" if fb['single_mode'] else "No")
        st.caption(f"Overlap factor Γ = {fb['overlap_factor']:.3f}")

    # ── Gain & Threshold ──
    with lab_panel("Gain & Threshold"):
        g = d["gain"]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Net Gain", f"{g['total_gain_db']:.1f} dB")
        c2.metric("Inversion", f"{g['inversion']:.1%}")
        c3.metric("Threshold", f"{g['threshold_pump_mw']:.1f} mW")
        c4.metric("Slope η", f"{g['slope_efficiency']:.1%}")

        c5, c6, c7 = st.columns(3)
        c5.metric("Pump Absorbed", f"{g['absorbed_pump_mw']:.1f} mW")
        c6.metric("Absorption", f"{g['pump_absorbed_fraction']:.1%}")
        c7.metric("P_sat", f"{g['saturation_power_mw']:.1f} mW")

    # ── Output ──
    with lab_panel(f"Output — {pulse_mode}"):
        o = d["output"]
        if pulse_mode == "CW":
            c1, c2 = st.columns(2)
            c1.metric("CW Output", f"{o['cw_output_mw']:.2f} mW")
            c2.metric("Overall η", f"{o['avg_power_mw'] / pump_mw * 100:.1f}%" if pump_mw > 0 else "—")
        else:
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Avg Power", f"{o['avg_power_mw']:.2f} mW")
            if o['pulse_energy_nj'] >= 1:
                c2.metric("Pulse Energy", f"{o['pulse_energy_nj']:.2f} nJ")
            else:
                c2.metric("Pulse Energy", f"{o['pulse_energy_nj']*1e3:.1f} pJ")
            if o['pulse_width_ns'] >= 1:
                c3.metric("Pulse Width", f"{o['pulse_width_ns']:.1f} ns")
            else:
                c3.metric("Pulse Width", f"{o['pulse_width_ns']*1e3:.0f} ps")
            if o['peak_power_w'] >= 1:
                c4.metric("Peak Power", f"{o['peak_power_w']:.1f} W")
            elif o['peak_power_w'] >= 1e-3:
                c4.metric("Peak Power", f"{o['peak_power_w']*1e3:.1f} mW")
            else:
                c4.metric("Peak Power", f"{o['peak_power_w']*1e6:.1f} µW")

            if pulse_mode == "Mode-locked":
                st.caption(f"Rep rate: {o['rep_rate_hz']/1e6:.1f} MHz (fundamental FSR)")
            else:
                st.caption(f"Rep rate: {rep_rate_khz:.0f} kHz")

    # ── Plots ──
    col1, col2 = st.columns(2)
    with col1:
        with lab_panel("Gain Spectrum"):
            sp = d["spectra"]
            fig = make_figure(f"Small-Signal Gain — {qd_mat} {qd_diam:.1f} nm QDs")
            fig.add_trace(go.Scatter(x=sp["wavelength_nm"], y=sp["gain_spectrum_db"],
                                     name="Net Gain", line=dict(color=COLORS[0], width=2.5)))
            fig.add_hline(y=0, line_dash="dot", line_color="gray")
            fig.update_xaxes(title_text="Wavelength (nm)")
            fig.update_yaxes(title_text="Net Gain (dB)")
            show_figure(fig)

    with col2:
        with lab_panel("Pump Slope"):
            ps = d["pump_sweep"]
            fig = make_figure("Output vs Pump Power")
            fig.add_trace(go.Scatter(x=ps["pump_mw"], y=ps["output_mw"],
                                     name="Output", line=dict(color=COLORS[0], width=2.5)))
            fig.add_trace(go.Scatter(x=[pump_mw], y=[d["output"]["avg_power_mw"]],
                                     mode="markers", marker=dict(size=11, symbol="star", color=COLORS[1]),
                                     name="Operating point"))
            fig.add_vline(x=d["gain"]["threshold_pump_mw"], line_dash="dash", line_color=COLORS[3],
                          annotation_text="Threshold")
            fig.update_xaxes(title_text="Pump Power (mW)")
            fig.update_yaxes(title_text="Output Power (mW)")
            show_figure(fig)

    col1, col2 = st.columns(2)
    with col1:
        with lab_panel("Emission vs QD Size"):
            ss = d["size_sweep"]
            fig = make_figure("Size-Tunable Emission")
            fig.add_trace(go.Scatter(x=ss["diameter_nm"], y=ss["emission_nm"],
                                     name="λ_em", line=dict(color=COLORS[1], width=2.5)))
            fig.add_vline(x=qd_diam, line_dash="dot", line_color=COLORS[0],
                          annotation_text=f"{qd_diam} nm")
            fig.update_xaxes(title_text="QD Diameter (nm)")
            fig.update_yaxes(title_text="Emission Wavelength (nm)")
            show_figure(fig)

    with col2:
        with lab_panel("Gain Bandwidth vs QD Size"):
            fig = make_figure("Inhomogeneous Gain Bandwidth")
            fig.add_trace(go.Scatter(x=ss["diameter_nm"], y=ss["gain_bandwidth_nm"],
                                     name="Gain BW", line=dict(color=COLORS[2], width=2.5)))
            fig.add_vline(x=qd_diam, line_dash="dot", line_color=COLORS[0])
            fig.update_xaxes(title_text="QD Diameter (nm)")
            fig.update_yaxes(title_text="Gain Bandwidth (nm)")
            show_figure(fig)

    # ── Thermal ──
    with lab_panel("Thermal"):
        th = d["thermal"]
        c1, c2, c3 = st.columns(3)
        c1.metric("Quantum Defect", f"{th['quantum_defect']:.1%}")
        c2.metric("Heat Load", f"{th['heat_load_mw']:.1f} mW")
        c3.metric("Heat/Length", f"{th['heat_per_length_mw_m']:.1f} mW/m")

    # ── Model Comparison ──
    from harrington_labs.comparison.ui import model_comparison_panel, reference_upload_panel
    model_comparison_panel(
        sim_x=d["spectra"]["wavelength_nm"],
        sim_y=d["spectra"]["gain_spectrum_db"],
        x_label="Wavelength", y_label="Gain", x_unit="nm", y_unit="dB",
        panel_title="Model Comparison — QD Fiber Laser",
        key_prefix="qdf_compare",
    )
    reference_upload_panel(key_prefix="qdf_ref", save_dir="data/references")
