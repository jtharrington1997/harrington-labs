"""Model Comparison UI component for lab pages.

Provides a drop-in Streamlit panel for uploading experimental data,
selecting datasets, overlaying on simulation curves, and displaying
comparison metrics. Reusable across all lab pages.
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from harrington_labs.comparison import (
    ComparisonResult,
    ExperimentalDataset,
    compare_curves,
    detect_and_parse,
)
from harrington_labs.ui import COLORS, make_figure, show_figure, lab_panel


# ── Comparison plot builders ──────────────────────────────────────────────


def _overlay_figure(
    result: ComparisonResult,
    title: str = "Model vs Experiment",
    x_label: str = "",
    y_label: str = "",
) -> go.Figure:
    """Build an overlay plot of measured vs simulated."""
    fig = make_figure(title)
    fig.add_trace(go.Scatter(
        x=result.x_common, y=result.y_measured,
        mode="markers",
        name="Measured",
        marker=dict(color=COLORS[1], size=8, symbol="circle-open", line=dict(width=1.5)),
    ))
    fig.add_trace(go.Scatter(
        x=result.x_common, y=result.y_simulated,
        mode="lines",
        name="Simulated",
        line=dict(color=COLORS[0], width=2.5),
    ))
    fig.update_xaxes(title_text=x_label)
    fig.update_yaxes(title_text=y_label)
    return fig


def _residual_figure(
    result: ComparisonResult,
    x_label: str = "",
) -> go.Figure:
    """Build a residual plot."""
    fig = make_figure("Residuals (Measured − Simulated)")
    fig.add_trace(go.Scatter(
        x=result.x_common, y=result.residuals,
        mode="markers+lines",
        name="Residuals",
        marker=dict(color=COLORS[2], size=5),
        line=dict(color=COLORS[2], width=1, dash="dot"),
    ))
    fig.add_hline(y=0, line_dash="dash", line_color="gray", line_width=1)
    fig.update_xaxes(title_text=x_label)
    fig.update_yaxes(title_text="Residual")
    return fig


# ── Main panel ──────────────────────────────────────────────────────────


def model_comparison_panel(
    sim_x: np.ndarray,
    sim_y: np.ndarray,
    x_label: str = "Position",
    y_label: str = "Signal",
    x_unit: str = "",
    y_unit: str = "",
    panel_title: str = "Model Comparison",
    key_prefix: str = "compare",
    extra_parsers: Optional[dict[str, Callable]] = None,
) -> Optional[ComparisonResult]:
    """Render a Model Comparison panel for any lab page.

    Parameters
    ----------
    sim_x, sim_y : array
        Current simulation output to compare against.
    x_label, y_label : str
        Axis labels for plots.
    key_prefix : str
        Unique prefix for Streamlit widget keys (avoids collisions).
    extra_parsers : dict, optional
        Additional file-type → parser functions beyond the built-in ones.

    Returns
    -------
    ComparisonResult or None if no comparison was made.
    """
    with lab_panel(panel_title):
        uploaded = st.file_uploader(
            "Upload experimental data",
            type=["xlsx", "xls", "csv", "tsv", "txt"],
            key=f"{key_prefix}_upload",
            help="Supports xlsx (including messy lab formats), CSV, TSV, and TXT files.",
        )

        if uploaded is None:
            st.caption("Upload a data file to compare simulation output against measured results.")
            return None

        # Save to temp and parse
        tmp_path = Path(f"/tmp/{uploaded.name}")
        tmp_path.write_bytes(uploaded.getvalue())
        datasets = detect_and_parse(tmp_path)

        if not datasets:
            st.warning("Could not detect any numeric datasets in the uploaded file.")
            return None

        # Dataset selector
        names = [ds.name for ds in datasets]
        if len(names) > 1:
            selected_name = st.selectbox(
                "Select dataset",
                names,
                key=f"{key_prefix}_dataset",
            )
            ds = datasets[names.index(selected_name)]
        else:
            ds = datasets[0]
            st.caption(f"Dataset: **{ds.name}** — {ds.n_points} points")

        # Show metadata expander
        if ds.metadata:
            with st.expander("Dataset metadata"):
                for k, v in ds.metadata.items():
                    st.text(f"{k}: {v}")

        # Run comparison
        try:
            result = compare_curves(
                x_measured=ds.x,
                y_measured=ds.y,
                x_simulated=sim_x,
                y_simulated=sim_y,
                y_uncertainty=ds.y_uncertainty,
                metric_name=ds.name,
            )
        except ValueError as e:
            st.error(f"Comparison failed: {e}")
            return None

        # Metrics scorecard
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("R²", f"{result.r_squared:.4f}")
        m2.metric("RMSE", f"{result.rmse:.4g}")
        m3.metric("NRMSE", f"{result.nrmse:.2%}")
        m4.metric("Max Error", f"{result.max_abs_error:.4g}")

        if result.mean_residual != 0:
            bias_dir = "high" if result.mean_residual > 0 else "low"
            st.caption(
                f"Mean residual: {result.mean_residual:.4g} "
                f"(experiment reads {bias_dir} relative to model)"
            )

        # Overlay and residual plots
        col1, col2 = st.columns([2, 1])
        with col1:
            x_ax = f"{x_label} ({x_unit})" if x_unit else x_label
            y_ax = f"{y_label} ({y_unit})" if y_unit else y_label
            fig = _overlay_figure(result, "Model vs Experiment", x_ax, y_ax)
            show_figure(fig)
        with col2:
            fig_r = _residual_figure(result, x_ax)
            show_figure(fig_r)

        return result

    return None


# ── Reference data upload (papers, datasheets, etc.) ──────────────────────


def reference_upload_panel(
    key_prefix: str = "ref",
    save_dir: str = "data/references",
) -> Optional[Path]:
    """Panel for uploading reference papers, datasheets, or supporting documents.

    Saves files to data/references/ within the repo for future use.
    Returns the path to the saved file, or None.
    """
    with lab_panel("Reference Library"):
        st.caption(
            "Upload research papers, datasheets, or other reference material. "
            "These will be saved to inform and improve model development."
        )
        uploaded = st.file_uploader(
            "Upload reference file",
            type=["pdf", "xlsx", "csv", "docx", "txt", "png", "jpg"],
            key=f"{key_prefix}_ref_upload",
        )
        if uploaded is None:
            return None

        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        dest = save_path / uploaded.name
        dest.write_bytes(uploaded.getvalue())
        st.success(f"Saved: `{dest}`")

        # Show file listing
        existing = sorted(save_path.iterdir())
        if existing:
            with st.expander(f"Reference files ({len(existing)})"):
                for f in existing:
                    st.text(f"  {f.name} ({f.stat().st_size / 1024:.0f} KB)")

        return dest
