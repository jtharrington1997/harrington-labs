"""
pages/80_Run_Campaign_Legacy.py — Legacy Campaign Runner

Legacy execution page for campaign jobs.

This page remains intentionally operational rather than ambitious:
- build / inspect a campaign payload
- run the configured campaign through run_campaign_api
- inspect saved JSON results
- view quick diagnostic plots

Deeper experiment-vs-simulation overlay belongs in 50_Simulation_Legacy.py.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from harrington_labs.lmi.io.run_campaign_api import run_campaign_blocking
from harrington_labs.ui import render_header
from harrington_labs.ui import lab_panel, make_figure, show_figure, COLORS
from harrington_labs.lmi.ui.formatting import (
    fmt_frequency_hz,
    fmt_power_w,
    fmt_wavelength_nm,
)


BASE = Path.home() / "Projects" / "harrington-lmi"
RESULT_DIR = BASE / "campaigns" / "si_midIR_ablation" / "results"

DEFAULT_CAMPAIGN = {
    "experiment": {"name": "si_full"},
    "beam": {
        "wavelength_um": 8.5,
        "pulse_duration_fs": 170.0,
        "pulse_energy_uJ": 200.0,
        "rep_rate_hz": 10000.0,
        "beam_diameter_mm": 5.0,
        "m_squared": 1.1,
    },
    "samples": [
        {"id": "p", "type": "p", "thickness_um": 500.0, "rho": [1, 10, 100]}
    ],
}


def _load_results(result_dir: Path) -> pd.DataFrame | None:
    rows: list[dict] = []
    if not result_dir.exists():
        return None

    for fpath in sorted(result_dir.glob("*.json")):
        try:
            with open(fpath, "r", encoding="utf-8") as f:
                payload = json.load(f)
        except Exception:
            continue

        if isinstance(payload, dict):
            payload["_filename"] = fpath.name
            rows.append(payload)

    return pd.DataFrame(rows) if rows else None


def _apply_pub_layout(fig, *, height: int = 420, showlegend: bool = True) -> None:
    fig.update_layout(
        template="simple_white",
        height=height,
        paper_bgcolor="white",
        plot_bgcolor="white",
        font=dict(size=15),
        margin=dict(l=24, r=24, t=44, b=20),
        showlegend=showlegend,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1.0,
            bgcolor="rgba(255,255,255,0.9)",
            bordercolor="rgba(0,0,0,0.15)",
            borderwidth=1,
        ),
    )
    fig.update_xaxes(
        showline=True,
        linewidth=1.1,
        linecolor="black",
        mirror=True,
        ticks="outside",
        tickwidth=1.1,
        ticklen=6,
        showgrid=True,
        gridcolor="rgba(0,0,0,0.10)",
        zeroline=False,
        exponentformat="power",
        showexponent="all",
    )
    fig.update_yaxes(
        showline=True,
        linewidth=1.1,
        linecolor="black",
        mirror=True,
        ticks="outside",
        tickwidth=1.1,
        ticklen=6,
        showgrid=True,
        gridcolor="rgba(0,0,0,0.10)",
        zeroline=False,
        exponentformat="power",
        showexponent="all",
    )


st.set_page_config(page_title="Run Campaign Legacy", layout="wide")
render_header()

with lab_panel():
    st.subheader("Run Campaign Legacy")
    st.caption(
        "Legacy execution page for campaign payloads and saved result inspection."
    )

with lab_panel():
    st.subheader("Workflow Handoff")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.link_button("Open Unified Workspace", "/Modeling_And_Simulation", width="stretch")
    with c2:
        st.link_button("Open Interaction Analyzer Legacy", "/Interaction_Analyzer_Legacy", width="stretch")
    with c3:
        st.link_button("Open Simulation Legacy", "/Simulation_Legacy", width="stretch")
    with c4:
        st.link_button("Open Digital Twin Legacy", "/Digital_Twin_Legacy", width="stretch")

with lab_panel():
    st.subheader("Campaign Configuration")

    c1, c2, c3 = st.columns(3)
    with c1:
        experiment_name = st.text_input("Experiment name", value=DEFAULT_CAMPAIGN["experiment"]["name"])
    with c2:
        wavelength_um = st.number_input(
            "Wavelength (µm)",
            min_value=0.1,
            max_value=100.0,
            value=float(DEFAULT_CAMPAIGN["beam"]["wavelength_um"]),
            format="%.6g",
        )
    with c3:
        pulse_duration_fs = st.number_input(
            "Pulse duration (fs)",
            min_value=1.0,
            max_value=1e9,
            value=float(DEFAULT_CAMPAIGN["beam"]["pulse_duration_fs"]),
            format="%.6g",
        )

    c4, c5, c6 = st.columns(3)
    with c4:
        pulse_energy_uj = st.number_input(
            "Pulse energy (µJ)",
            min_value=0.0,
            max_value=1e9,
            value=float(DEFAULT_CAMPAIGN["beam"]["pulse_energy_uJ"]),
            format="%.6g",
        )
    with c5:
        rep_rate_khz = st.number_input(
            "Rep rate (kHz)",
            min_value=0.001,
            max_value=1e9,
            value=float(DEFAULT_CAMPAIGN["beam"]["rep_rate_hz"]) / 1e3,
            format="%.6g",
        )
    with c6:
        beam_diameter_mm = st.number_input(
            "Beam diameter (mm)",
            min_value=0.001,
            max_value=100.0,
            value=float(DEFAULT_CAMPAIGN["beam"]["beam_diameter_mm"]),
            format="%.6g",
        )

    c7, c8, c9 = st.columns(3)
    with c7:
        m_squared = st.number_input(
            "M²",
            min_value=1.0,
            max_value=50.0,
            value=float(DEFAULT_CAMPAIGN["beam"]["m_squared"]),
            format="%.3f",
        )
    with c8:
        sample_id = st.text_input("Sample ID", value=DEFAULT_CAMPAIGN["samples"][0]["id"])
    with c9:
        sample_type = st.text_input("Sample type", value=DEFAULT_CAMPAIGN["samples"][0]["type"])

    c10, c11 = st.columns(2)
    with c10:
        thickness_um = st.number_input(
            "Thickness (µm)",
            min_value=0.1,
            max_value=1e6,
            value=float(DEFAULT_CAMPAIGN["samples"][0]["thickness_um"]),
            format="%.6g",
        )
    with c11:
        rho_csv = st.text_input(
            "ρ values (ohm·cm, comma-separated)",
            value="1,10,100",
        )

    rho_values = []
    for token in rho_csv.split(","):
        token = token.strip()
        if not token:
            continue
        try:
            rho_values.append(float(token))
        except ValueError:
            pass

    campaign_payload = {
        "experiment": {"name": experiment_name},
        "beam": {
            "wavelength_um": float(wavelength_um),
            "pulse_duration_fs": float(pulse_duration_fs),
            "pulse_energy_uJ": float(pulse_energy_uj),
            "rep_rate_hz": float(rep_rate_khz) * 1e3,
            "beam_diameter_mm": float(beam_diameter_mm),
            "m_squared": float(m_squared),
        },
        "samples": [
            {
                "id": sample_id,
                "type": sample_type,
                "thickness_um": float(thickness_um),
                "rho": rho_values,
            }
        ],
    }

    b1, b2 = st.columns(2)
    with b1:
        run_clicked = st.button("Run Campaign", type="primary", width="stretch")
    with b2:
        st.download_button(
            "Download Payload JSON",
            data=json.dumps(campaign_payload, indent=2),
            file_name="campaign_payload.json",
            mime="application/json",
            width="stretch",
        )

with lab_panel():
    st.subheader("Applied Beam Summary")
    avg_power_w = (pulse_energy_uj * 1e-6) * (rep_rate_khz * 1e3)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Wavelength", fmt_wavelength_nm(wavelength_um * 1000.0))
    c2.metric("Rep rate", fmt_frequency_hz(rep_rate_khz * 1e3))
    c3.metric("Average power", fmt_power_w(avg_power_w))
    c4.metric("ρ points", str(len(rho_values)))

if run_clicked:
    with lab_panel():
        with st.spinner("Running campaign..."):
            run_campaign_blocking(campaign_payload)
        st.success("Campaign completed.")

with lab_panel():
    st.subheader("Result Directory")
    st.code(str(RESULT_DIR), language="text")

df = _load_results(RESULT_DIR)

if df is None:
    with lab_panel():
        st.info("No JSON results found yet in the configured result directory.")
    st.stop()

with lab_panel():
    st.subheader("Loaded Results")
    st.dataframe(df, width="stretch", hide_index=True)

numeric_cols = set(df.select_dtypes(include=["number"]).columns)

with lab_panel():
    st.subheader("Quick Plots")

    if {"rho_ohm_cm", "alpha_cm"}.issubset(numeric_cols):
        fig_alpha = px.scatter(
            df,
            x="rho_ohm_cm",
            y="alpha_cm",
            hover_data=["_filename"] if "_filename" in df.columns else None,
        )
        fig_alpha.update_layout(
            xaxis_title="Resistivity, ρ (ohm·cm)",
            yaxis_title="Absorption coefficient, α (cm⁻¹)",
        )
        fig_alpha.update_xaxes(type="log")
        fig_alpha.update_yaxes(type="log", tickformat=".3e")
        _apply_pub_layout(fig_alpha, height=430, showlegend=False)
        st.plotly_chart(fig_alpha, width="stretch")

    candidate_pairs = [
        ("rho_ohm_cm", "transmission_fraction"),
        ("rho_ohm_cm", "absorbed_fraction"),
        ("rho_ohm_cm", "fluence_j_cm2"),
        ("rho_ohm_cm", "peak_irradiance_w_cm2"),
    ]

    plotted_any = False
    for xcol, ycol in candidate_pairs:
        if {xcol, ycol}.issubset(numeric_cols):
            plotted_any = True
            fig = px.scatter(
                df,
                x=xcol,
                y=ycol,
                hover_data=["_filename"] if "_filename" in df.columns else None,
            )
            fig.update_layout(
                xaxis_title=xcol.replace("_", " "),
                yaxis_title=ycol.replace("_", " "),
            )
            fig.update_xaxes(type="log")
            fig.update_yaxes(type="log", tickformat=".3e")
            _apply_pub_layout(fig, height=430, showlegend=False)
            st.plotly_chart(fig, width="stretch")

    if not plotted_any and not {"rho_ohm_cm", "alpha_cm"}.issubset(numeric_cols):
        st.info("No recognized numeric result columns were found for quick plotting.")
