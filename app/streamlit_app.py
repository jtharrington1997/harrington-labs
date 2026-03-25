"""
streamlit_app.py — Harrington Labs
Photonics Lab Simulators & Laser-Material Interaction Platform.
"""
import streamlit as st
from pathlib import Path
from harrington_labs.ui import render_header, lab_panel

st.set_page_config(
    page_title="Harrington Labs",
    layout="wide",
    initial_sidebar_state="expanded",
)

render_header(
    title="Harrington Labs",
    subtitle="Reducing geometric footprint • Increasing power & energy • Advancing science, technology & medicine",
)

# ── Overview metrics ─────────────────────────────────────────────
with lab_panel("Platform Status"):
    cols = st.columns(5)
    try:
        from harrington_labs.lmi.domain.lasers import all_lasers
        from harrington_labs.lmi.domain.materials import all_materials
        lasers = all_lasers()
        materials = all_materials()
        cols[0].metric("Lab Simulators", "7")
        cols[1].metric("LMI Engines", "4")
        cols[2].metric("Lasers in DB", len(lasers))
        cols[3].metric("Materials in DB", len(materials))
    except Exception:
        cols[0].metric("Lab Simulators", "7")
        cols[1].metric("LMI Engines", "4")
        cols[2].metric("Lasers in DB", "—")
        cols[3].metric("Materials in DB", "—")

    # Reference file count
    ref_dir = Path("data/references")
    ref_count = len(list(ref_dir.iterdir())) if ref_dir.exists() else 0
    cols[4].metric("Reference Files", ref_count)

# ── Research Pipeline ────────────────────────────────────────────
with lab_panel("Research Pipeline"):
    st.caption(
        "The labs form a progression: build and characterize light sources, "
        "control their propagation, then study how they interact with matter. "
        "Each lab has **Model Comparison** to validate simulations against real experiments."
    )

    c1, c2, c3 = st.columns(3)

    with c1:
        st.page_link("pages/1_Direct_Diode_Lab.py", label="Direct Diode Lab", icon=":material/laser:")
        st.caption("L-I curves, thermal rollover, beam combining")
        st.page_link("pages/2_Fiber_Laser_Lab.py", label="Fiber Laser Lab", icon=":material/cable:")
        st.caption("Gain modeling, nonlinear thresholds, power scaling, beam combining")
        st.page_link("pages/10_Demonstrator_Builder.py", label="Demonstrator Builder", icon=":material/build:")
        st.caption("Full-chain laser demonstrator — resonator, QD fiber, QD diode + beam combining")

    with c2:
        st.page_link("pages/3_Beam_Control_Lab.py", label="Beam Control Lab", icon=":material/track_changes:")
        st.caption("Atmospheric propagation, AO, turbulence")
        st.page_link("pages/4_Pulsed_Laser_Lab.py", label="Pulsed Laser Lab", icon=":material/electric_bolt:")
        st.caption("Ultrafast pulses, dispersion, z-scan")
        st.page_link("pages/6_Coatings_Lab.py", label="Coatings Lab", icon=":material/layers:")
        st.caption("Thin-film design, spectral/angular response")

    with c3:
        st.page_link("pages/7a_Spectroscopy_Lab.py", label="Spectroscopy Lab", icon=":material/biotech:")
        st.caption("Raman, Brillouin, DUVRR, LIBS, FTIR, hyperspectral")
        st.page_link("pages/5_Quantum_Dots_Lab.py", label="Quantum Dots Lab", icon=":material/blur_on:")
        st.caption("Size-dependent bandgap, PL, exciton dynamics")
        st.page_link("pages/9_Modeling_And_Simulation.py", label="Modeling & Simulation", icon=":material/analytics:")
        st.caption("Beam propagation, nonlinear, thermal, z-scan")

# ── Databases & Admin ────────────────────────────────────────────
with lab_panel("Databases & Tools"):
    c1, c2, c3 = st.columns(3)
    with c1:
        st.page_link("pages/7_Laser_Library.py", label="Laser Library")
        st.caption("Commercial lasers, custom sources, OPA chaining")
    with c2:
        st.page_link("pages/8_Material_Database.py", label="Material Database")
        st.caption("Optical, thermal, mechanical properties with Sellmeier dispersion")
    with c3:
        st.page_link("pages/90_Admin.py", label="Admin")
        st.caption("API keys, compute backend, data management")

# ── Footer ───────────────────────────────────────────────────────
st.markdown("---")
st.caption(
    "Harrington Labs v1.2.0 • "
    "Photonics Lab Simulators + Laser-Material Interaction • "
    "Built with Streamlit"
)
