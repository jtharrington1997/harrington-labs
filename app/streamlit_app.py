"""
streamlit_app.py — Harrington Labs
Photonics Lab Simulators & Laser-Material Interaction Platform.
"""
import streamlit as st
from harrington_labs.ui import render_header, lab_panel

st.set_page_config(
    page_title="Harrington Labs",
    layout="wide",
    initial_sidebar_state="expanded",
)

render_header(
    title="Harrington Labs",
    subtitle="Photonics Lab Simulators • Laser-Material Interaction • Modeling & Simulation",
)

# ── Overview ─────────────────────────────────────────────────────
with lab_panel("Dashboard"):
    cols = st.columns(4)
    try:
        from harrington_labs.lmi.domain.lasers import all_lasers
        from harrington_labs.lmi.domain.materials import all_materials
        lasers = all_lasers()
        materials = all_materials()
        cols[0].metric("Lab Simulators", "6")
        cols[1].metric("LMI Engines", "4")
        cols[2].metric("Lasers in DB", len(lasers))
        cols[3].metric("Materials in DB", len(materials))
    except Exception:
        cols[0].metric("Lab Simulators", "6")
        cols[1].metric("LMI Engines", "4")
        cols[2].metric("Lasers in DB", "—")
        cols[3].metric("Materials in DB", "—")

# ── Lab Simulators ───────────────────────────────────────────────
with lab_panel("Lab Environment Simulators"):
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("### Direct Diode Lab")
        st.caption("L-I curves, thermal rollover, wavelength drift, far-field patterns, beam combining.")
        st.page_link("pages/1_Direct_Diode_Lab.py", label="Open Direct Diode Lab")
    with col2:
        st.markdown("### Fiber Laser Lab")
        st.caption("Gain modeling, ASE buildup, nonlinear thresholds (SBS/SRS/SPM), thermal limits.")
        st.page_link("pages/2_Fiber_Laser_Lab.py", label="Open Fiber Laser Lab")
    with col3:
        st.markdown("### Beam Control Lab")
        st.caption("Atmospheric propagation, turbulence, Fried parameter, scintillation, adaptive optics.")
        st.page_link("pages/3_Beam_Control_Lab.py", label="Open Beam Control Lab")

with lab_panel():
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("### Pulsed Laser Lab")
        st.caption("Ultrafast pulse dynamics, temporal/spectral profiles, dispersion, z-scan modeling.")
        st.page_link("pages/4_Pulsed_Laser_Lab.py", label="Open Pulsed Laser Lab")
    with col2:
        st.markdown("### Quantum Dots Lab")
        st.caption("Size-dependent bandgap (Brus), PL/absorption spectra, exciton dynamics.")
        st.page_link("pages/5_Quantum_Dots_Lab.py", label="Open Quantum Dots Lab")
    with col3:
        st.markdown("### Coatings Lab")
        st.caption("Thin-film transfer matrix, spectral/angular response, E-field profiles, GDD.")
        st.page_link("pages/6_Coatings_Lab.py", label="Open Coatings Lab")

# ── LMI Platform ─────────────────────────────────────────────────
with lab_panel("Laser-Material Interaction Platform"):
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("### Laser Library")
        st.caption("Commercial lasers, custom sources, OPA chaining, spatial beam modes.")
        st.page_link("pages/7_Laser_Library.py", label="Open Laser Library")
    with col2:
        st.markdown("### Material Database")
        st.caption("Optical, thermal, mechanical properties with Sellmeier dispersion models.")
        st.page_link("pages/8_Material_Database.py", label="Open Material Database")
    with col3:
        st.markdown("### Modeling & Simulation")
        st.caption("Beam propagation, nonlinear optics, z-scan, thermal analysis, campaign comparison.")
        st.page_link("pages/9_Modeling_And_Simulation.py", label="Open Modeling & Simulation")

with lab_panel():
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("### Source Builder")
        st.caption("Design laser systems from gain medium, pump, resonator, and output coupler.")
        st.page_link("pages/10_Source_Builder.py", label="Open Source Builder")
    with col2:
        st.markdown("### Admin")
        st.caption("API keys, custom data management, system settings.")
        st.page_link("pages/90_Admin.py", label="Open Admin")
    with col3:
        pass

# ── Footer ───────────────────────────────────────────────────────
st.markdown("---")
st.caption(
    "Harrington Labs v1.0.0 • "
    "Photonics Lab Simulators + Laser-Material Interaction • "
    "Built with Streamlit"
)
