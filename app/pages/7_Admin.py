"""7_Admin.py — Admin page for Harrington Labs."""
import streamlit as st
from harrington_labs.ui import render_header, lab_panel

st.set_page_config(page_title="Admin — Harrington Labs", layout="wide")
render_header("Admin", "System settings and diagnostics")

try:
    from harrington_common.admin.keys import admin_panel
    admin_panel()
except ImportError:
    with lab_panel("Admin"):
        st.info("harrington-common admin module not available.")

with lab_panel("System Info"):
    import sys
    import numpy as np
    st.caption(f"Python {sys.version}")
    st.caption(f"NumPy {np.__version__}")
    try:
        import scipy
        st.caption(f"SciPy {scipy.__version__}")
    except ImportError:
        st.caption("SciPy: not installed")
    try:
        import plotly
        st.caption(f"Plotly {plotly.__version__}")
    except ImportError:
        st.caption("Plotly: not installed")
