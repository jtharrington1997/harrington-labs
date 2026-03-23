"""Layout helpers for Harrington LMI — delegates to harrington_common."""
from __future__ import annotations

from pathlib import Path

import streamlit as st

from harrington_common.theme import apply_theme, st_svg

_TITLE = "Harrington LMI"
_SUBTITLE = "Laser-Material Interaction \u2022 Modeling & Simulation \u2022 Photonics"
_LOGO = "app/assets/lmi-logo.svg"

def render_header(title: str = ""):
    if title:
        st.title(title)
    else:
        st.title("Harrington LMI")
