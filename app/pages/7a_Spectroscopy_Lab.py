"""7a_Spectroscopy_Lab.py — Spectroscopy Lab Simulator."""
import numpy as np
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from harrington_labs.ui import render_header, lab_panel, make_figure, show_figure, warning_box, COLORS
from harrington_labs.domain import SimulationResult
from harrington_labs.domain.spectroscopy import (
    RamanParams, BrillouinParams, DUVRRParams,
    LIBSParams, FTIRParams, HyperspectralParams,
    SamplePhase, RamanExcitation,
)
from harrington_labs.simulation.spectroscopy import (
    spontaneous_raman, stimulated_raman,
    spontaneous_brillouin, stimulated_brillouin,
    duvrr_spectrum, libs_spectrum, ftir_spectrum,
    hyperspectral_image,
)

st.set_page_config(page_title="Spectroscopy Lab", layout="wide")
render_header(
    "Spectroscopy Lab",
    "Raman • Brillouin • DUVRR • LIBS • FTIR • Hyperspectral Imaging",
)

# ── Database access ──────────────────────────────────────────────
from harrington_labs.ui.db_sidebar import source_and_material_sidebar
db_laser, db_material = source_and_material_sidebar("spectro")

# ── Technique selector ──────────────────────────────────────────────
tabs = st.tabs([
    "Raman",
    "Brillouin",
    "DUVRR",
    "LIBS",
    "FTIR",
    "Hyperspectral",
])

# ════════════════════════════════════════════════════════════════════
# TAB 1: RAMAN (Spontaneous + Stimulated)
# ════════════════════════════════════════════════════════════════════
with tabs[0]:
    st.sidebar.header("Raman Parameters")
    raman_mode = st.sidebar.radio("Mode", ["Spontaneous", "Stimulated (SRS)"], key="raman_mode")

    exc_wl_options = {e.value: float(e.value.replace(" nm", "")) for e in RamanExcitation}
    exc_sel = st.sidebar.selectbox("Excitation Wavelength", list(exc_wl_options.keys()), index=4, key="raman_exc")
    exc_wl = exc_wl_options[exc_sel]
    if db_laser:
        exc_wl = db_laser.wavelength_nm  # override with database source

    raman_power = st.sidebar.number_input("Laser Power (mW)", 0.1, 1000.0, db_laser.power_w * 1e3 if db_laser else 50.0, 1.0, key="raman_pwr")
    raman_int_time = st.sidebar.number_input("Integration Time (s)", 0.01, 300.0, 1.0, 0.1, key="raman_int")
    raman_na = st.sidebar.number_input("Objective NA", 0.1, 1.4, 0.75, 0.05, key="raman_na")
    raman_res = st.sidebar.number_input("Spectral Resolution (cm⁻¹)", 0.5, 20.0, 4.0, 0.5, key="raman_res")
    raman_temp = st.sidebar.number_input("Temperature (K)", 4.0, 1000.0, 293.0, 10.0, key="raman_temp")
    raman_phase = st.sidebar.selectbox("Sample Phase", [p.value for p in SamplePhase], key="raman_phase")

    # Material preset
    raman_preset = st.sidebar.selectbox("Material Preset", [
        "Silicon (520 cm⁻¹)",
        "Diamond (1332 cm⁻¹)",
        "Fused Silica",
        "Polystyrene",
        "Calcite (CaCO₃)",
        "Water",
        "Custom",
    ], key="raman_preset")

    _RAMAN_PRESETS = {
        "Silicon (520 cm⁻¹)": ([520.0], [8.0], [1.0]),
        "Diamond (1332 cm⁻¹)": ([1332.0], [3.5], [1.0]),
        "Fused Silica": ([440.0, 490.0, 800.0, 1060.0], [40.0, 30.0, 50.0, 60.0], [0.7, 0.3, 0.5, 0.4]),
        "Polystyrene": ([621.0, 1001.0, 1031.0, 1155.0, 1450.0, 1583.0, 1602.0, 2852.0, 2904.0, 3054.0],
                        [12.0, 6.0, 10.0, 8.0, 15.0, 8.0, 6.0, 20.0, 15.0, 25.0],
                        [0.3, 1.0, 0.7, 0.2, 0.3, 0.4, 0.8, 0.3, 0.4, 0.6]),
        "Calcite (CaCO₃)": ([156.0, 282.0, 712.0, 1086.0, 1436.0], [10.0, 8.0, 12.0, 5.0, 15.0], [0.3, 0.5, 0.4, 1.0, 0.15]),
        "Water": ([1640.0, 3250.0, 3450.0], [80.0, 200.0, 200.0], [0.3, 0.7, 1.0]),
    }

    if raman_preset != "Custom" and raman_preset in _RAMAN_PRESETS:
        shifts, widths, intensities = _RAMAN_PRESETS[raman_preset]
    else:
        n_peaks = st.sidebar.number_input("Number of peaks", 1, 10, 1, key="raman_npeaks")
        shifts, widths, intensities = [], [], []
        for i in range(int(n_peaks)):
            s = st.sidebar.number_input(f"Peak {i+1} position (cm⁻¹)", 50.0, 4000.0, 520.0, 1.0, key=f"raman_s{i}")
            w = st.sidebar.number_input(f"Peak {i+1} FWHM (cm⁻¹)", 1.0, 200.0, 8.0, 1.0, key=f"raman_w{i}")
            a = st.sidebar.number_input(f"Peak {i+1} intensity", 0.01, 10.0, 1.0, 0.1, key=f"raman_a{i}")
            shifts.append(s)
            widths.append(w)
            intensities.append(a)

    raman_params = RamanParams(
        excitation_wavelength_nm=exc_wl,
        laser_power_mw=raman_power,
        integration_time_s=raman_int_time,
        numerical_aperture=raman_na,
        spectral_resolution_cm_inv=raman_res,
        sample_phase=SamplePhase(raman_phase),
        temperature_k=raman_temp,
        raman_shifts_cm_inv=shifts,
        raman_widths_cm_inv=widths,
        raman_intensities=intensities,
    )

    if raman_mode == "Spontaneous":
        data = spontaneous_raman(raman_params)

        with lab_panel("Spontaneous Raman Spectrum"):
            fig = make_figure(f"Raman Spectrum — λ_exc = {exc_wl:.0f} nm")
            fig.add_trace(go.Scatter(
                x=data["shift_cm_inv"], y=data["spectrum"],
                name="Measured", line=dict(color=COLORS[0], width=1.5),
            ))
            fig.add_trace(go.Scatter(
                x=data["shift_cm_inv"], y=data["background"],
                name="Background", line=dict(color=COLORS[3], width=1, dash="dash"),
            ))
            fig.update_xaxes(title_text="Raman Shift (cm⁻¹)")
            fig.update_yaxes(title_text="Intensity (a.u.)")
            show_figure(fig)

        col1, col2 = st.columns(2)
        with col1:
            with lab_panel("Peak Table"):
                for i, (s, w, a) in enumerate(zip(shifts, widths, intensities)):
                    wl_s = 1e7 / (1e7 / exc_wl - s)
                    st.caption(f"**{s:.1f} cm⁻¹** — FWHM: {w:.1f} cm⁻¹, λ_Stokes: {wl_s:.1f} nm")

        with col2:
            with lab_panel("Stokes / Anti-Stokes"):
                if data["stokes_anti_stokes_ratio"]:
                    st.metric("S/AS Ratio", f"{data['stokes_anti_stokes_ratio']:.1f}")
                    st.caption(f"Sample temperature: {raman_temp:.0f} K")
                else:
                    st.caption("No peaks defined for S/AS analysis.")

    else:  # SRS
        srs_pump = st.sidebar.number_input("Pump Power (mW)", 1.0, 5000.0, 100.0, 10.0, key="srs_pump")
        srs_seed = st.sidebar.number_input("Stokes Seed (mW)", 0.001, 10.0, 0.01, 0.001, key="srs_seed", format="%.3f")
        srs_length = st.sidebar.number_input("Interaction Length (m)", 0.01, 100.0, 1.0, 0.1, key="srs_len")
        raman_params.pump_power_mw = srs_pump
        raman_params.stokes_seed_power_mw = srs_seed

        data = stimulated_raman(raman_params, fiber_length_m=srs_length)

        with lab_panel("SRS Gain Spectrum"):
            fig = make_figure("Stimulated Raman Gain")
            fig.add_trace(go.Scatter(
                x=data["shift_cm_inv"], y=data["stokes_gain"],
                name="Stokes Gain", line=dict(color=COLORS[0], width=2.5),
            ))
            fig.update_xaxes(title_text="Raman Shift (cm⁻¹)")
            fig.update_yaxes(title_text="Gain Factor", type="log")
            show_figure(fig)

        with lab_panel("SRS Summary"):
            c1, c2, c3 = st.columns(3)
            c1.metric("Peak Gain", f"{data['peak_gain_m_per_w']:.2e} m/W")
            c2.metric("Max Stokes Power", f"{float(np.max(data['stokes_power_mw'])):.2f} mW")
            c3.metric("Pump Remaining", f"{data['pump_remaining_fraction']:.1%}")

    # Model comparison
    from harrington_labs.comparison.ui import model_comparison_panel, reference_upload_panel
    if raman_mode == "Spontaneous":
        model_comparison_panel(
            sim_x=data["shift_cm_inv"], sim_y=data["spectrum"],
            x_label="Raman Shift", y_label="Intensity", x_unit="cm⁻¹", y_unit="a.u.",
            panel_title="Model Comparison — Raman", key_prefix="raman_compare",
        )

# ════════════════════════════════════════════════════════════════════
# TAB 2: BRILLOUIN
# ════════════════════════════════════════════════════════════════════
with tabs[1]:
    st.sidebar.header("Brillouin Parameters")
    brill_mode = st.sidebar.radio("Mode", ["Spontaneous", "Stimulated (SBS)"], key="brill_mode")
    brill_wl = st.sidebar.number_input("Wavelength (nm)", 400.0, 1600.0, db_laser.wavelength_nm if db_laser else 532.0, 1.0, key="brill_wl")
    brill_power = st.sidebar.number_input("Power (mW)", 1.0, 5000.0, db_laser.power_w * 1e3 if db_laser else 100.0, 10.0, key="brill_pwr")
    brill_angle = st.sidebar.number_input("Scattering Angle (°)", 1.0, 180.0, 180.0, 1.0, key="brill_angle")
    brill_v = st.sidebar.number_input("Sound Velocity (m/s)", 100.0, 20000.0, 5960.0, 10.0, key="brill_v")
    _def_n = db_material.refractive_index if db_material and db_material.refractive_index > 0 else 1.46
    _def_rho = db_material.density_kg_m3 if db_material and db_material.density_kg_m3 > 0 else 2200.0
    brill_n = st.sidebar.number_input("Refractive Index", 1.0, 4.0, _def_n, 0.01, key="brill_n")
    brill_rho = st.sidebar.number_input("Density (kg/m³)", 500.0, 20000.0, _def_rho, 100.0, key="brill_rho")
    brill_alpha = st.sidebar.number_input("Acoustic Atten. (dB/cm/GHz²)", 0.01, 10.0, 0.5, 0.1, key="brill_alpha")

    brill_params = BrillouinParams(
        excitation_wavelength_nm=brill_wl, laser_power_mw=brill_power,
        scattering_angle_deg=brill_angle, sound_velocity_m_s=brill_v,
        refractive_index=brill_n, density_kg_m3=brill_rho,
        acoustic_attenuation_db_cm_ghz2=brill_alpha,
    )

    if brill_mode == "Spontaneous":
        data_b = spontaneous_brillouin(brill_params)

        with lab_panel("Brillouin Spectrum"):
            fig = make_figure(f"Brillouin Spectrum — θ = {brill_angle:.0f}°")
            fig.add_trace(go.Scatter(x=data_b["frequency_ghz"], y=data_b["stokes"], name="Stokes", line=dict(color=COLORS[0], width=2)))
            fig.add_trace(go.Scatter(x=data_b["frequency_ghz"], y=data_b["anti_stokes"], name="Anti-Stokes", line=dict(color=COLORS[1], width=2)))
            fig.add_trace(go.Scatter(x=data_b["frequency_ghz"], y=data_b["rayleigh"], name="Rayleigh", line=dict(color=COLORS[3], width=1, dash="dash")))
            fig.update_xaxes(title_text="Frequency Shift (GHz)")
            fig.update_yaxes(title_text="Intensity (a.u.)")
            show_figure(fig)

        with lab_panel("Brillouin Summary"):
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("ν_B", f"{data_b['brillouin_shift_ghz']:.3f} GHz")
            c2.metric("Linewidth", f"{data_b['linewidth_ghz']:.3f} GHz")
            c3.metric("M (Long.)", f"{data_b['longitudinal_modulus_gpa']:.1f} GPa")
            c4.metric("v_s", f"{data_b['sound_velocity_m_s']:.0f} m/s")

    else:  # SBS
        brill_fiber_d = st.sidebar.number_input("Fiber Core (µm)", 1.0, 100.0, 8.0, 1.0, key="sbs_core")
        brill_length = st.sidebar.number_input("Fiber Length (m)", 0.1, 1000.0, 10.0, 1.0, key="sbs_len")
        brill_params.fiber_core_diameter_um = brill_fiber_d
        brill_params.interaction_length_m = brill_length

        data_sbs = stimulated_brillouin(brill_params)

        with lab_panel("SBS Threshold & Reflectivity"):
            fig = make_figure("SBS Reflectivity vs Input Power")
            fig.add_trace(go.Scatter(
                x=data_sbs["input_power_mw"], y=data_sbs["sbs_reflectivity"],
                name="Reflectivity", line=dict(color=COLORS[1], width=2.5),
            ))
            fig.add_vline(x=data_sbs["threshold_power_mw"], line_dash="dot", line_color=COLORS[0],
                          annotation_text=f"Threshold: {data_sbs['threshold_power_mw']:.1f} mW")
            fig.update_xaxes(title_text="Input Power (mW)")
            fig.update_yaxes(title_text="Backward Reflectivity", type="log")
            show_figure(fig)

        with lab_panel("SBS Summary"):
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("g_B", f"{data_sbs['gain_coefficient_m_per_w']:.2e} m/W")
            c2.metric("Threshold", f"{data_sbs['threshold_power_mw']:.1f} mW")
            c3.metric("A_eff", f"{data_sbs['effective_area_um2']:.1f} µm²")
            c4.metric("ν_B", f"{data_sbs['brillouin_shift_ghz']:.3f} GHz")

# ════════════════════════════════════════════════════════════════════
# TAB 3: DUVRR
# ════════════════════════════════════════════════════════════════════
with tabs[2]:
    st.sidebar.header("DUVRR Parameters")
    duvrr_wl = st.sidebar.number_input("Excitation λ (nm)", 190.0, 270.0, 244.0, 1.0, key="duvrr_wl")
    duvrr_power = st.sidebar.number_input("Power (µW)", 1.0, 5000.0, 500.0, 10.0, key="duvrr_pwr")
    duvrr_int = st.sidebar.number_input("Integration (s)", 1.0, 600.0, 60.0, 10.0, key="duvrr_int")
    duvrr_conc = st.sidebar.number_input("Concentration (mg/mL)", 0.1, 100.0, 10.0, 1.0, key="duvrr_conc")
    duvrr_trans = st.sidebar.number_input("Electronic Transition (nm)", 200.0, 300.0, 260.0, 1.0, key="duvrr_trans")
    duvrr_enh = st.sidebar.number_input("Enhancement Factor", 1e2, 1e7, 1e4, 100.0, format="%.0e", key="duvrr_enh")

    duvrr_params = DUVRRParams(
        excitation_wavelength_nm=duvrr_wl, laser_power_uw=duvrr_power,
        integration_time_s=duvrr_int, concentration_mg_ml=duvrr_conc,
        electronic_transition_nm=duvrr_trans, resonance_enhancement_factor=duvrr_enh,
    )
    data_d = duvrr_spectrum(duvrr_params)

    with lab_panel("DUVRR Spectrum"):
        fig = make_figure(f"Deep-UV Resonance Raman — λ = {duvrr_wl:.0f} nm")
        fig.add_trace(go.Scatter(x=data_d["shift_cm_inv"], y=data_d["spectrum"], name="DUVRR", line=dict(color=COLORS[4], width=1.5)))
        for mode in data_d["mode_assignments"]:
            fig.add_vline(x=mode["position"], line_dash="dot", line_color="gray", line_width=0.5,
                          annotation_text=mode["label"], annotation_font_size=9)
        fig.update_xaxes(title_text="Raman Shift (cm⁻¹)")
        fig.update_yaxes(title_text="Intensity (a.u.)")
        show_figure(fig)

    col1, col2 = st.columns(2)
    with col1:
        with lab_panel("Excitation Profile"):
            fig2 = make_figure("Resonance Enhancement vs λ")
            fig2.add_trace(go.Scatter(x=data_d["excitation_profile_nm"], y=data_d["excitation_profile"],
                                       name="Enhancement", line=dict(color=COLORS[1], width=2)))
            fig2.add_vline(x=duvrr_wl, line_dash="dot", line_color=COLORS[0], annotation_text="λ_exc")
            fig2.update_xaxes(title_text="Excitation Wavelength (nm)")
            fig2.update_yaxes(title_text="Relative Enhancement")
            show_figure(fig2)
    with col2:
        with lab_panel("Mode Assignments"):
            st.metric("Resonance Enhancement", f"{data_d['resonance_enhancement']:.2e}×")
            for m in data_d["mode_assignments"]:
                st.caption(f"**{m['label']}** — {m['position']:.0f} cm⁻¹ (rel. int. {m['intensity']:.2f})")

    model_comparison_panel(
        sim_x=data_d["shift_cm_inv"], sim_y=data_d["spectrum"],
        x_label="Raman Shift", y_label="Intensity", x_unit="cm⁻¹",
        panel_title="Model Comparison — DUVRR", key_prefix="duvrr_compare",
    )

# ════════════════════════════════════════════════════════════════════
# TAB 4: LIBS
# ════════════════════════════════════════════════════════════════════
with tabs[3]:
    st.sidebar.header("LIBS Parameters")
    libs_energy = st.sidebar.number_input("Pulse Energy (mJ)", 0.1, 500.0, 50.0, 1.0, key="libs_e")
    libs_pulse_ns = st.sidebar.number_input("Pulse Width (ns)", 0.1, 100.0, 8.0, 0.5, key="libs_pw")
    libs_spot = st.sidebar.number_input("Spot Diameter (µm)", 10.0, 1000.0, 100.0, 10.0, key="libs_spot")
    libs_delay = st.sidebar.number_input("Gate Delay (µs)", 0.01, 100.0, 1.0, 0.1, key="libs_delay")
    libs_gate = st.sidebar.number_input("Gate Width (µs)", 0.1, 100.0, 10.0, 1.0, key="libs_gate")

    libs_preset = st.sidebar.selectbox("Material", [
        "Stainless Steel 304",
        "Aluminum 6061",
        "Copper Alloy",
        "Soil / Environmental",
        "Custom",
    ], key="libs_preset")

    _LIBS_COMPOSITIONS = {
        "Stainless Steel 304": {"Fe": 0.70, "Cr": 0.18, "Ni": 0.08, "Mn": 0.02, "Si": 0.01, "C": 0.01},
        "Aluminum 6061": {"Al": 0.97, "Mg": 0.01, "Si": 0.006, "Cu": 0.003, "Fe": 0.007, "Mn": 0.001, "Cr": 0.002, "Ti": 0.001},
        "Copper Alloy": {"Cu": 0.88, "Si": 0.04, "Mn": 0.02, "Fe": 0.03, "Ni": 0.02, "Al": 0.01},
        "Soil / Environmental": {"Si": 0.30, "Al": 0.08, "Fe": 0.05, "Ca": 0.04, "Mg": 0.02, "Na": 0.01, "Ti": 0.005, "Mn": 0.001},
    }

    if libs_preset in _LIBS_COMPOSITIONS:
        composition = _LIBS_COMPOSITIONS[libs_preset]
    else:
        composition = {"Fe": 0.5, "Cr": 0.2, "Ni": 0.1}

    libs_params = LIBSParams(
        pulse_energy_mj=libs_energy, pulse_width_ns=libs_pulse_ns,
        spot_diameter_um=libs_spot, gate_delay_us=libs_delay,
        gate_width_us=libs_gate, composition=composition,
    )
    data_l = libs_spectrum(libs_params)

    with lab_panel("LIBS Emission Spectrum"):
        fig = make_figure(f"LIBS Spectrum — {libs_preset}")
        fig.add_trace(go.Scatter(x=data_l["wavelength_nm"], y=data_l["spectrum"], name="Emission", line=dict(color=COLORS[0], width=1.2)))
        fig.add_trace(go.Scatter(x=data_l["wavelength_nm"], y=data_l["continuum"], name="Continuum", line=dict(color=COLORS[3], width=1, dash="dash")))
        fig.update_xaxes(title_text="Wavelength (nm)")
        fig.update_yaxes(title_text="Intensity (a.u.)")
        show_figure(fig)

    with lab_panel("Plasma & Line ID"):
        c1, c2, c3 = st.columns(3)
        c1.metric("Plasma Temperature", f"{data_l['plasma_temperature_k']:.0f} K")
        c2.metric("Irradiance", f"{data_l['irradiance_gw_cm2']:.2f} GW/cm²")
        c3.metric("Lines Detected", f"{len(data_l['line_data'])}")

        with st.expander("Top emission lines"):
            for line in data_l["line_data"][:15]:
                st.caption(f"**{line['element']}** {line['wavelength_nm']:.1f} nm — E_upper: {line['upper_ev']:.1f} eV")

    model_comparison_panel(
        sim_x=data_l["wavelength_nm"], sim_y=data_l["spectrum"],
        x_label="Wavelength", y_label="Intensity", x_unit="nm",
        panel_title="Model Comparison — LIBS", key_prefix="libs_compare",
    )

# ════════════════════════════════════════════════════════════════════
# TAB 5: FTIR
# ════════════════════════════════════════════════════════════════════
with tabs[4]:
    st.sidebar.header("FTIR Parameters")
    ftir_res = st.sidebar.number_input("Resolution (cm⁻¹)", 0.5, 16.0, 4.0, 0.5, key="ftir_res")
    ftir_scans = st.sidebar.number_input("Number of Scans", 1, 1024, 32, key="ftir_scans")
    ftir_thick = st.sidebar.number_input("Sample Thickness (µm)", 0.1, 1000.0, 10.0, 1.0, key="ftir_thick")
    ftir_wn_min = st.sidebar.number_input("Min Wavenumber (cm⁻¹)", 200.0, 2000.0, 400.0, 50.0, key="ftir_wn_min")
    ftir_wn_max = st.sidebar.number_input("Max Wavenumber (cm⁻¹)", 2000.0, 6000.0, 4000.0, 100.0, key="ftir_wn_max")

    ftir_preset = st.sidebar.selectbox("Material", [
        "Polymer Film (generic)",
        "Protein (Amide bands)",
        "Silicone",
        "Custom",
    ], key="ftir_preset")

    _FTIR_PRESETS = {
        "Polymer Film (generic)": [(3400, 200, 0.8), (2920, 30, 0.6), (2850, 25, 0.4), (1740, 20, 0.9), (1460, 15, 0.3), (1050, 40, 0.7)],
        "Protein (Amide bands)": [(3300, 250, 0.6), (2960, 30, 0.3), (1650, 30, 1.0), (1540, 30, 0.8), (1300, 25, 0.4), (1100, 40, 0.3)],
        "Silicone": [(2960, 25, 0.5), (1260, 15, 1.0), (1090, 60, 0.9), (1020, 40, 0.8), (800, 20, 0.7)],
    }

    if ftir_preset in _FTIR_PRESETS:
        ir_modes = _FTIR_PRESETS[ftir_preset]
    else:
        ir_modes = [(1740, 20, 0.9)]

    ftir_params = FTIRParams(
        wavenumber_min_cm_inv=ftir_wn_min, wavenumber_max_cm_inv=ftir_wn_max,
        resolution_cm_inv=ftir_res, n_scans=int(ftir_scans),
        thickness_um=ftir_thick, ir_modes=ir_modes,
    )
    data_f = ftir_spectrum(ftir_params)

    with lab_panel("FTIR Spectrum"):
        display_mode = st.radio("Display", ["Absorbance", "Transmittance"], horizontal=True, key="ftir_display")
        if display_mode == "Absorbance":
            fig = make_figure("FTIR Absorbance")
            fig.add_trace(go.Scatter(x=data_f["wavenumber_cm_inv"], y=data_f["absorbance"], name="Absorbance", line=dict(color=COLORS[0], width=1.5)))
            fig.update_yaxes(title_text="Absorbance (AU)")
        else:
            fig = make_figure("FTIR Transmittance")
            fig.add_trace(go.Scatter(x=data_f["wavenumber_cm_inv"], y=data_f["transmittance"] * 100, name="Transmittance", line=dict(color=COLORS[0], width=1.5)))
            fig.update_yaxes(title_text="Transmittance (%)", range=[0, 105])
        fig.update_xaxes(title_text="Wavenumber (cm⁻¹)", autorange="reversed")
        show_figure(fig)

    with lab_panel("FTIR Summary"):
        st.metric("SNR", f"{data_f['snr']:.0f}")
        st.caption(f"Resolution: {ftir_res} cm⁻¹ | Scans: {ftir_scans} | Thickness: {ftir_thick} µm")

    model_comparison_panel(
        sim_x=data_f["wavenumber_cm_inv"], sim_y=data_f["absorbance"],
        x_label="Wavenumber", y_label="Absorbance", x_unit="cm⁻¹",
        panel_title="Model Comparison — FTIR", key_prefix="ftir_compare",
    )

# ════════════════════════════════════════════════════════════════════
# TAB 6: HYPERSPECTRAL IMAGING
# ════════════════════════════════════════════════════════════════════
with tabs[5]:
    st.sidebar.header("Hyperspectral Params")
    hyper_size = st.sidebar.number_input("Image Size (px)", 16, 128, 64, 8, key="hyper_size")
    hyper_px = st.sidebar.number_input("Pixel Size (µm)", 0.1, 10.0, 1.0, 0.1, key="hyper_px")
    hyper_dwell = st.sidebar.number_input("Dwell Time (ms)", 1.0, 1000.0, 100.0, 10.0, key="hyper_dwell")
    hyper_snr = st.sidebar.number_input("SNR (dB)", 5.0, 60.0, 30.0, 5.0, key="hyper_snr")
    hyper_nc = st.sidebar.number_input("Components", 2, 6, 3, key="hyper_nc")

    hyper_params = HyperspectralParams(
        image_size_px=int(hyper_size), pixel_size_um=hyper_px,
        pixel_dwell_time_ms=hyper_dwell, snr_db=hyper_snr,
        n_components=int(hyper_nc),
    )
    data_h = hyperspectral_image(hyper_params)

    with lab_panel("Hyperspectral Image"):
        fov = hyper_size * hyper_px
        fig = make_figure(f"Integrated Intensity — {fov:.0f}×{fov:.0f} µm FOV")
        fig.add_trace(go.Heatmap(
            z=data_h["intensity_image"],
            colorscale="Viridis",
            colorbar=dict(title="Intensity"),
        ))
        fig.update_xaxes(title_text=f"x (pixels, {hyper_px} µm/px)")
        fig.update_yaxes(title_text=f"y (pixels, {hyper_px} µm/px)")
        fig.update_layout(yaxis_scaleanchor="x")
        show_figure(fig)

    with lab_panel("Component Maps & Spectra"):
        cols = st.columns(min(int(hyper_nc), 3))
        for i in range(int(hyper_nc)):
            with cols[i % len(cols)]:
                st.caption(f"**{data_h['component_names'][i]}**")
                fig_c = go.Figure(go.Heatmap(z=data_h["component_maps"][i], colorscale="Hot", showscale=False))
                fig_c.update_layout(height=200, margin=dict(l=10, r=10, t=10, b=10), yaxis_scaleanchor="x")
                st.plotly_chart(fig_c, width="stretch")

        with lab_panel("Component Spectra"):
            fig_s = make_figure("Component Reference Spectra")
            for i, (spec, name) in enumerate(zip(data_h["component_spectra"], data_h["component_names"])):
                fig_s.add_trace(go.Scatter(
                    x=data_h["wavenumber_cm_inv"], y=spec,
                    name=name, line=dict(color=COLORS[i % len(COLORS)], width=2),
                ))
            fig_s.update_xaxes(title_text="Wavenumber (cm⁻¹)")
            fig_s.update_yaxes(title_text="Intensity (a.u.)")
            show_figure(fig_s)

# ── Reference Library (shared) ──────────────────────────────────────
reference_upload_panel(key_prefix="spectro_ref", save_dir="data/references")
