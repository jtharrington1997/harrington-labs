"""Shared session state for cross-lab beam parameter continuity.

Allows beam parameters configured in one lab to carry forward
to other labs. Stored in st.session_state under a 'shared_beam' key.
"""
from __future__ import annotations

from typing import Optional
import streamlit as st


_KEY = "harrington_shared_beam"

_DEFAULTS = {
    "wavelength_nm": 1030.0,
    "power_w": 2.0,
    "beam_diameter_mm": 2.0,
    "m_squared": 1.0,
    "rep_rate_hz": 1e3,
    "pulse_width_s": 170e-15,
}


def get_shared_beam() -> dict:
    """Get current shared beam state, initializing if needed."""
    if _KEY not in st.session_state:
        st.session_state[_KEY] = dict(_DEFAULTS)
    return st.session_state[_KEY]


def update_shared_beam(**kwargs) -> None:
    """Update shared beam state with new values."""
    state = get_shared_beam()
    state.update(kwargs)


def shared_beam_badge() -> None:
    """Show a small indicator if shared beam state is active."""
    state = get_shared_beam()
    if state != _DEFAULTS:
        st.sidebar.caption(
            f"Shared beam: {state['wavelength_nm']:.0f} nm, "
            f"{state['power_w']:.2g} W, "
            f"D={state['beam_diameter_mm']:.1f} mm"
        )


def push_beam_button(
    wavelength_nm: float,
    power_w: float,
    beam_diameter_mm: float,
    m_squared: float = 1.0,
    rep_rate_hz: float = 1e3,
    pulse_width_s: float = 170e-15,
    key: str = "push_beam",
) -> bool:
    """Render a 'Share to other labs' button in the main content area. Returns True if clicked."""
    if st.button("Share beam to other labs", key=key, help="Sets these beam parameters as defaults in other lab pages"):
        update_shared_beam(
            wavelength_nm=wavelength_nm,
            power_w=power_w,
            beam_diameter_mm=beam_diameter_mm,
            m_squared=m_squared,
            rep_rate_hz=rep_rate_hz,
            pulse_width_s=pulse_width_s,
        )
        st.toast("Beam parameters shared across labs")
        return True
    return False
