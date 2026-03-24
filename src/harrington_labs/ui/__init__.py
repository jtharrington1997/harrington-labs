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
        plotly_layout as _base_layout,
        is_dark_mode,
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

def _get_plot_layout(**overrides) -> dict:
    """Build adaptive Plotly layout for labs pages."""
    if _HAS_COMMON:
        base = _base_layout(margin=dict(l=60, r=20, t=40, b=50), **overrides)
    else:
        base = dict(
            template="plotly_white",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(family="Source Sans 3, Helvetica Neue, sans-serif"),
            margin=dict(l=60, r=20, t=40, b=50),
        )
        base.update(overrides)
    return base


# Backward-compatible static dict (computed at import time)
PLOT_LAYOUT = _get_plot_layout()
PLOT_TEMPLATE = PLOT_LAYOUT.get("template", "plotly_white")

# Color sequence — adapts to mode
def _get_colors() -> list[str]:
    if _HAS_COMMON and is_dark_mode():
        return ["#7eaed4", "#d4626f", "#d4a843", "#4ade80", "#a78bfa", "#f97316", "#67e8f9"]
    return ["#1a3a5c", "#8b2332", "#b8860b", "#2d6a4f", "#7b4f8a", "#c96e12", "#4a7c8f"]

COLORS = _get_colors()


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
