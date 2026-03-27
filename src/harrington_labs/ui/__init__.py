"""UI helpers for Harrington Labs — delegates to harrington-common.

Provides lab-specific plotting defaults and reusable display functions.
"""
from __future__ import annotations

from contextlib import contextmanager
from typing import Optional

import plotly.graph_objects as go
import streamlit as st

try:
    from harrington_common.theme import (
        render_header as _common_header,
        aw_panel,
        BRAND,
        plotly_layout as _base_layout,
        is_dark_mode,
    )
    _HAS_COMMON = True
except ImportError:
    _HAS_COMMON = False
    _common_header = None
    aw_panel = None
    _base_layout = None

    def is_dark_mode() -> bool:
        return False

    BRAND = {
        "primary": "#1a3a5c",
        "accent": "#8b2332",
        "gold": "#b8860b",
        "cream": "#faf8f5",
    }


def _get_plot_layout(**overrides) -> dict:
    """Build adaptive Plotly layout for labs pages."""
    if _HAS_COMMON and _base_layout is not None:
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


PLOT_LAYOUT = _get_plot_layout()
PLOT_TEMPLATE = PLOT_LAYOUT.get("template", "plotly_white")


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
    st.plotly_chart(fig, use_container_width=True)


def render_header(title: str = "Harrington Labs", subtitle: str = "") -> None:
    """Render the standard page header."""
    if _HAS_COMMON and _common_header is not None:
        _common_header(title=title, subtitle=subtitle)
    else:
        st.title(title)
        if subtitle:
            st.caption(subtitle)


@contextmanager
def lab_panel(title: str = ""):
    """Context manager for a styled lab panel."""
    if _HAS_COMMON and aw_panel is not None:
        with aw_panel():
            if title:
                st.subheader(title)
            yield
    else:
        if title:
            st.subheader(title)
        with st.container(border=True):
            yield


def _as_message_list(messages: str | list[str] | tuple[str, ...] | None) -> list[str]:
    if not messages:
        return []
    if isinstance(messages, str):
        return [messages]
    return [str(m) for m in messages if m]


def info_box(messages: str | list[str] | tuple[str, ...] | None) -> None:
    for msg in _as_message_list(messages):
        st.info(msg, icon="ℹ")


def success_box(messages: str | list[str] | tuple[str, ...] | None) -> None:
    for msg in _as_message_list(messages):
        st.success(msg, icon="✓")


def warning_box(messages: str | list[str] | tuple[str, ...] | None) -> None:
    for msg in _as_message_list(messages):
        st.warning(msg, icon="⚠")


def error_box(messages: str | list[str] | tuple[str, ...] | None) -> None:
    for msg in _as_message_list(messages):
        st.error(msg, icon="✖")
