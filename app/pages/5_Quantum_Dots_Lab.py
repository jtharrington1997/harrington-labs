"""5_Quantum_Dots_Lab.py — Quantum Dots Lab Simulator."""
import streamlit as st
import plotly.graph_objects as go
from harrington_labs.ui import render_header, lab_panel, make_figure, show_figure, warning_box, COLORS
from harrington_labs.domain import QuantumDotParams, QDMaterial
from harrington_labs.simulation.quantum_dots import run_quantum_dot_simulation

st.set_page_config(page_title="Quantum Dots Lab", layout="wide")
render_header("Quantum Dots Lab", "Brus bandgap • PL & absorption spectra • Exciton dynamics • Temperature dependence")

# ── Sidebar ──────────────────────────────────────────────────────
from harrington_labs.ui.db_sidebar import source_and_material_sidebar
db_laser, db_material = source_and_material_sidebar("qd")

st.sidebar.header("QD Parameters")
material = st.sidebar.selectbox("Material", [m.value for m in QDMaterial], index=1)
diameter = st.sidebar.number_input("QD Diameter (nm)", 1.0, 20.0, 5.0, 0.1)
size_dist = st.sidebar.slider("Size Distribution (%)", 1.0, 20.0, 5.0, 0.5)
shell = st.sidebar.number_input("Shell Thickness (nm)", 0.0, 10.0, 2.0, 0.5)

st.sidebar.header("Optical Properties")
qy = st.sidebar.slider("Quantum Yield", 0.01, 1.0, 0.5, 0.01)
fwhm = st.sidebar.number_input("Emission FWHM (nm)", 5.0, 100.0, 30.0, 1.0)
lifetime = st.sidebar.number_input("Exciton Lifetime (ns)", 0.1, 200.0, 20.0, 1.0)
concentration = st.sidebar.number_input("Concentration (nmol/mL)", 0.01, 100.0, 1.0, 0.1)

params = QuantumDotParams(
    material=QDMaterial(material),
    diameter_nm=diameter,
    size_distribution_pct=size_dist,
    shell_thickness_nm=shell,
    quantum_yield=qy,
    fwhm_emission_nm=fwhm,
    exciton_lifetime_ns=lifetime,
    concentration_nmol_ml=concentration,
)

result = run_quantum_dot_simulation(params)
warning_box(result.warnings)

# ── Summary ──────────────────────────────────────────────────────
with lab_panel("QD Summary"):
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Bandgap", f"{result.data['bandgap_ev']:.3f} eV")
    c2.metric("Peak Emission", f"{result.data['peak_emission_nm']:.0f} nm")
    c3.metric("Quantum Yield", f"{qy:.0%}")
    em = result.data["emission"]
    c4.metric("Total FWHM", f"{em['fwhm_nm']:.1f} nm")

# ── Size-dependent bandgap ───────────────────────────────────────
with lab_panel("Size-Dependent Bandgap (Brus Equation)"):
    ss = result.data["size_scan"]
    fig = make_figure("Bandgap vs QD Diameter")
    fig.add_trace(go.Scatter(
        x=ss["diameter_nm"], y=ss["bandgap_ev"],
        name="Bandgap", line=dict(color=COLORS[0], width=2.5),
    ))
    fig.add_vline(x=diameter, line_dash="dot", line_color=COLORS[1],
                  annotation_text=f"{diameter} nm")
    fig.update_xaxes(title_text="Diameter (nm)")
    fig.update_yaxes(title_text="Bandgap (eV)")
    show_figure(fig)

    # Secondary axis: emission wavelength
    fig2 = make_figure("Peak Emission Wavelength vs QD Diameter")
    fig2.add_trace(go.Scatter(
        x=ss["diameter_nm"], y=ss["peak_wavelength_nm"],
        name="λ_em", line=dict(color=COLORS[1], width=2.5),
    ))
    fig2.add_vline(x=diameter, line_dash="dot", line_color=COLORS[0])
    fig2.update_xaxes(title_text="Diameter (nm)")
    fig2.update_yaxes(title_text="Emission Wavelength (nm)")
    show_figure(fig2)

# ── Emission & absorption ────────────────────────────────────────
col1, col2 = st.columns(2)

with col1:
    with lab_panel("PL Emission Spectrum"):
        em = result.data["emission"]
        fig = make_figure(f"Emission (peak = {em['peak_nm']:.0f} nm)")
        fig.add_trace(go.Scatter(
            x=em["wavelength_nm"], y=em["intensity"],
            name="PL", line=dict(color=COLORS[1], width=2.5),
            fill="tozeroy", fillcolor="rgba(139,35,50,0.15)",
        ))
        fig.update_xaxes(title_text="Wavelength (nm)")
        fig.update_yaxes(title_text="PL Intensity (a.u.)")
        show_figure(fig)

with col2:
    with lab_panel("Absorption Spectrum"):
        ab = result.data["absorption"]
        fig = make_figure("Absorption Spectrum")
        fig.add_trace(go.Scatter(
            x=ab["wavelength_nm"], y=ab["absorbance"],
            name="Absorbance", line=dict(color=COLORS[0], width=2.5),
        ))
        fig.update_xaxes(title_text="Wavelength (nm)")
        fig.update_yaxes(title_text="Absorbance (a.u.)")
        show_figure(fig)

# ── Exciton dynamics ─────────────────────────────────────────────
with lab_panel("Exciton Decay Dynamics"):
    dec = result.data["exciton_decay"]
    fig = make_figure("Time-Resolved PL Decay")
    fig.add_trace(go.Scatter(
        x=dec["time_ns"], y=dec["single_exciton"],
        name="Single Exciton", line=dict(color=COLORS[0], width=2.5),
    ))
    fig.add_trace(go.Scatter(
        x=dec["time_ns"], y=dec["biexciton"],
        name="Biexciton (Auger)", line=dict(color=COLORS[1], width=2, dash="dash"),
    ))
    fig.update_xaxes(title_text="Time (ns)")
    fig.update_yaxes(title_text="Population (normalized)", type="log")
    show_figure(fig)

    c1, c2, c3 = st.columns(3)
    c1.metric("k_rad", f"{dec['k_rad_ns']:.4f} ns⁻¹")
    c2.metric("k_nr", f"{dec['k_nr_ns']:.4f} ns⁻¹")
    c3.metric("τ_total", f"{1/dec['k_total_ns']:.1f} ns")

# ── Temperature dependence ───────────────────────────────────────
with lab_panel("Temperature-Dependent PL"):
    td = result.data["temp_dependence"]
    col1, col2 = st.columns(2)

    with col1:
        fig = make_figure("Peak Wavelength vs Temperature")
        fig.add_trace(go.Scatter(
            x=td["temperature_k"], y=td["peak_wavelength_nm"],
            name="λ_peak", line=dict(color=COLORS[3], width=2.5),
        ))
        fig.update_xaxes(title_text="Temperature (K)")
        fig.update_yaxes(title_text="Peak Wavelength (nm)")
        show_figure(fig)

    with col2:
        fig = make_figure("Quantum Yield vs Temperature")
        fig.add_trace(go.Scatter(
            x=td["temperature_k"], y=td["quantum_yield"],
            name="QY", line=dict(color=COLORS[2], width=2.5),
        ))
        fig.update_xaxes(title_text="Temperature (K)")
        fig.update_yaxes(title_text="Quantum Yield", range=[0, 1.05])
        show_figure(fig)

# ── Model Comparison ────────────────────────────────────────────
from harrington_labs.comparison.ui import model_comparison_panel, reference_upload_panel

model_comparison_panel(
    sim_x=em["wavelength_nm"],
    sim_y=em["intensity"],
    x_label="Wavelength",
    y_label="PL Intensity",
    x_unit="nm",
    y_unit="a.u.",
    panel_title="Model Comparison — PL Emission",
    key_prefix="qd_pl",
)

reference_upload_panel(key_prefix="qd_ref", save_dir="data/references")
