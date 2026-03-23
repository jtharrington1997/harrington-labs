"""UI helpers for Harrington Labs — delegates to harrington-common.

Provides lab-specific plotting defaults and reusable display functions.
"""
from __future__ import annotations

from contextlib import contextmanager
from typing import Optional

import streamlit as st
import plotly.graph_objects as go

try:
    from harrington_common.theme import (
        apply_theme,
        render_header as _common_header,
        aw_panel,
        hero_banner,
        metric_card,
        BRAND,
    )
    _HAS_COMMON = True
except ImportError:
    _HAS_COMMON = False
    BRAND = {
        "primary": "#1a3a5c",
        "accent": "#8b2332",
        "gold": "#b8860b",
        "cream": "#faf8f5",
    }


# ── Standard plot template ───────────────────────────────────────

PLOT_TEMPLATE = "plotly_white"
PLOT_LAYOUT = dict(
    template=PLOT_TEMPLATE,
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Source Sans 3, Helvetica Neue, sans-serif"),
    margin=dict(l=60, r=20, t=40, b=50),
)

# Color sequence for multi-trace plots
COLORS = [
    "#1a3a5c",  # navy
    "#8b2332",  # accent red
    "#b8860b",  # gold
    "#2d6a4f",  # green
    "#7b4f8a",  # purple
    "#c96e12",  # orange
    "#4a7c8f",  # teal
]


def make_figure(title: str = "", **kwargs) -> go.Figure:
    """Create a Plotly figure with standard Harrington styling."""
    layout = {**PLOT_LAYOUT, **kwargs}
    if title:
        layout["title"] = dict(text=title, font=dict(size=16))
    return go.Figure(layout=layout)


def show_figure(fig: go.Figure) -> None:
    """Display figure with standard width."""
    st.plotly_chart(fig, width="stretch")


# ── Header / panel delegates ─────────────────────────────────────

def render_header(title: str = "Harrington Labs", subtitle: str = "") -> None:
    if _HAS_COMMON:
        _common_header(title=title, subtitle=subtitle)
    else:
        st.title(title)
        if subtitle:
            st.caption(subtitle)


@contextmanager
def lab_panel(title: str = ""):
    """Context manager for a styled lab panel."""
    if _HAS_COMMON:
        with aw_panel():
            if title:
                st.subheader(title)
            yield
    else:
        if title:
            st.subheader(title)
        with st.container(border=True):
            yield


def warning_box(warnings: list[str]) -> None:
    """Display simulation warnings."""
    for w in warnings:
        st.warning(w, icon="⚠️")
