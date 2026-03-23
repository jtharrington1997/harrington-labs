"""
streamlit_app.py — Harrington Labs
Lab Environment Simulators for Photonics Research.
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
    subtitle="Photonics Lab Environment Simulators • Direct Diode • Fiber • Beam Control • Pulsed • Quantum Dots • Coatings",
)

# ── Overview metrics ─────────────────────────────────────────────
with lab_panel("Dashboard"):
    cols = st.columns(6)
    labs = [
        ("Direct Diode", "pages/1_Direct_Diode_Lab.py"),
        ("Fiber Laser", "pages/2_Fiber_Laser_Lab.py"),
        ("Beam Control", "pages/3_Beam_Control_Lab.py"),
        ("Pulsed Laser", "pages/4_Pulsed_Laser_Lab.py"),
        ("Quantum Dots", "pages/5_Quantum_Dots_Lab.py"),
        ("Coatings", "pages/6_Coatings_Lab.py"),
    ]
    for col, (name, _) in zip(cols, labs):
        col.metric(name, "Active")

# ── Lab navigation ───────────────────────────────────────────────
with lab_panel("Lab Environments"):
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("### Direct Diode Lab")
        st.caption(
            "L-I curves, thermal rollover, wavelength drift, "
            "far-field patterns, and beam combining analysis."
        )
        st.page_link("pages/1_Direct_Diode_Lab.py", label="Open Direct Diode Lab")

    with col2:
        st.markdown("### Fiber Laser Lab")
        st.caption(
            "Gain modeling, ASE buildup, nonlinear thresholds "
            "(SBS/SRS/SPM), thermal limits, and mode analysis."
        )
        st.page_link("pages/2_Fiber_Laser_Lab.py", label="Open Fiber Laser Lab")

    with col3:
        st.markdown("### Beam Control Lab")
        st.caption(
            "Atmospheric propagation, turbulence effects, Fried parameter, "
            "scintillation, beam wander, and adaptive optics correction."
        )
        st.page_link("pages/3_Beam_Control_Lab.py", label="Open Beam Control Lab")

with lab_panel():
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("### Pulsed Laser Lab")
        st.caption(
            "Ultrafast pulse dynamics, temporal/spectral profiles, "
            "autocorrelation, dispersion management, and z-scan modeling."
        )
        st.page_link("pages/4_Pulsed_Laser_Lab.py", label="Open Pulsed Laser Lab")

    with col2:
        st.markdown("### Quantum Dots Lab")
        st.caption(
            "Size-dependent bandgap (Brus), PL/absorption spectra, "
            "exciton dynamics, and temperature-dependent emission."
        )
        st.page_link("pages/5_Quantum_Dots_Lab.py", label="Open Quantum Dots Lab")

    with col3:
        st.markdown("### Coatings Lab")
        st.caption(
            "Thin-film coating design via transfer matrix, "
            "spectral/angular response, E-field profiles, and GDD."
        )
        st.page_link("pages/6_Coatings_Lab.py", label="Open Coatings Lab")

# ── Footer ───────────────────────────────────────────────────────
st.markdown("---")
st.caption(
    "Harrington Labs v0.1.0 • "
    "Photonics Lab Environment Simulators • "
    "Built with Streamlit"
)
