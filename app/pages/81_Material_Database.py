"""
pages/2_Material_Database.py — Material Properties Database
Browse and add materials with optical, thermal, and mechanical properties.
Supports wavelength-dependent Sellmeier dispersion lookup.
"""
from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
import streamlit as st
import pandas as pd
from harrington_labs.ui import render_header
from harrington_labs.ui import lab_panel, make_figure, show_figure, COLORS
from harrington_labs.lmi.ui.access import is_admin
from harrington_labs.lmi.ui.formatting import (
    fmt_absorption_cm_inv,
    fmt_density_kg_m3,
    fmt_ev,
    fmt_fluence_j_cm2,
    fmt_length_m,
    fmt_n2_cm2_w,
    fmt_refractive_index,
    fmt_temp_k,
    fmt_thermal_conductivity,
)
from harrington_labs.lmi.domain.materials import (
    all_materials, Material, save_custom_materials, load_custom_materials,
    SELLMEIER_VALID_RANGE,
)

st.set_page_config(page_title="Material Database", layout="wide")
render_header("Material Database", "Optical • Thermal • Mechanical properties • Sellmeier dispersion")


materials = all_materials()
categories = sorted(set(m.category for m in materials))

with lab_panel():
    st.subheader("Material Database")
    cat_filter = st.multiselect("Filter by category", categories, default=categories)
    filtered = [m for m in materials if m.category in cat_filter]

    rows = []
    for m in filtered:
        rows.append({
            "Name": m.name,
            "Category": m.category.title(),
            "n": fmt_refractive_index(m.refractive_index),
            "α (cm⁻¹)": fmt_absorption_cm_inv(m.absorption_coeff_cm),
            "Eg (eV)": fmt_ev(m.bandgap_ev) if m.bandgap_ev > 0 else "—",
            "n₂ (cm²/W)": fmt_n2_cm2_w(m.nonlinear_index_cm2_w) if m.nonlinear_index_cm2_w > 0 else "—",
            "LIDT": fmt_fluence_j_cm2(m.damage_threshold_j_cm2) if m.damage_threshold_j_cm2 > 0 else "—",
            "k": fmt_thermal_conductivity(m.thermal_conductivity_w_mk) if m.thermal_conductivity_w_mk > 0 else "—",
            "Tmelt": fmt_temp_k(m.melting_point_k) if m.melting_point_k > 0 else "—",
            "Sellmeier": "Yes" if m.has_sellmeier else "—",
        })

    df = pd.DataFrame(rows)
    st.dataframe(df, width="stretch", hide_index=True)

with lab_panel():
    st.subheader("Material Details")
    selected = st.selectbox("Select material", [m.name for m in filtered])
    mat = next(m for m in filtered if m.name == selected)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Refractive Index", fmt_refractive_index(mat.refractive_index))
    col2.metric("Absorption", fmt_absorption_cm_inv(mat.absorption_coeff_cm))
    col3.metric("Penetration Depth", fmt_length_m(mat.skin_depth_cm * 1e-2))
    col4.metric("LIDT", fmt_fluence_j_cm2(mat.damage_threshold_j_cm2) if mat.damage_threshold_j_cm2 else "N/A")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Thermal Conductivity", fmt_thermal_conductivity(mat.thermal_conductivity_w_mk) if mat.thermal_conductivity_w_mk else "N/A")
    col2.metric("Melting Point", fmt_temp_k(mat.melting_point_k) if mat.melting_point_k else "N/A")
    col3.metric("Density", fmt_density_kg_m3(mat.density_kg_m3) if mat.density_kg_m3 else "N/A")
    col4.metric("Bandgap", fmt_ev(mat.bandgap_ev) if mat.bandgap_ev else "N/A")

    if mat.nonlinear_index_cm2_w > 0:
        st.caption(f"Reference nonlinear index: {fmt_n2_cm2_w(mat.nonlinear_index_cm2_w)}")
    if mat.notes:
        st.caption(mat.notes)

with lab_panel():
    st.subheader("Wavelength-Dependent Properties")

    if mat.has_sellmeier:
        valid = SELLMEIER_VALID_RANGE.get(mat.name, (0, 0))
        st.success(
            f"Sellmeier dispersion model available for {mat.name} "
            f"(valid {valid[0]:.2f} – {valid[1]:.1f} µm)"
        )

        query_wl = st.number_input(
            "Query wavelength (nm)", 100.0, 20000.0, 8500.0,
            step=100.0, format="%.0f",
        )
        info = mat.dispersion_info(query_wl)

        col1, col2, col3 = st.columns(3)
        source_label = "Sellmeier" if info["sellmeier_used"] else "Stored fallback"
        col1.metric(
            f"n at {query_wl:.0f} nm",
            fmt_refractive_index(info["n"]),
            delta=source_label,
            delta_color="off",
        )
        col2.metric(f"α at {query_wl:.0f} nm", fmt_absorption_cm_inv(info["alpha_cm"]))
        col3.metric(f"n₂ at {query_wl:.0f} nm", fmt_n2_cm2_w(info["n2_cm2_w"]))

        if not info.get("in_valid_range", True):
            st.warning(
                f"Wavelength {query_wl:.0f} nm ({query_wl/1000:.3f} µm) is outside "
                f"the Sellmeier validity range ({valid[0]:.2f}–{valid[1]:.1f} µm). "
                f"Using stored reference value."
            )

        wl_min_nm = valid[0] * 1000
        wl_max_nm = valid[1] * 1000
        wl_arr = np.linspace(wl_min_nm, wl_max_nm, 500)
        n_arr = np.array([mat.get_n(w) for w in wl_arr])

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=wl_arr / 1000, y=n_arr,
            mode="lines", line=dict(color="#1a3a5c", width=2),
            name="n(λ)",
        ))
        fig.add_trace(go.Scatter(
            x=[query_wl / 1000], y=[info["n"]],
            mode="markers",
            marker=dict(size=10, color="#8b2332", symbol="star"),
            name=f"Query point",
        ))
        fig.update_layout(
            xaxis_title="Wavelength (µm)",
            yaxis_title="Refractive Index",
            **PLOT_LAYOUT,
            height=350,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1.0),
        )
        st.plotly_chart(fig, width="stretch")
    else:
        st.info(
            f"No Sellmeier model for {mat.name}. Using stored values: "
            f"n = {fmt_refractive_index(mat.refractive_index)} at {mat.ref_wavelength_nm:.0f} nm."
        )

if is_admin():
    with lab_panel():
        st.subheader("Add Custom Material")
        with st.form("new_material"):
            col1, col2 = st.columns(2)
            with col1:
                name = st.text_input("Name")
                category = st.selectbox("Category", ["metal", "semiconductor", "dielectric", "polymer", "biological"])
                n = st.number_input("Refractive index", 0.01, 10.0, 1.5)
                alpha = st.number_input("Absorption coeff (cm⁻¹)", 0.0, 1e7, 0.0, format="%.2e")
                bandgap = st.number_input("Bandgap (eV)", 0.0, 20.0, 0.0)
                n2 = st.number_input("n₂ (cm²/W)", 0.0, 1e-10, 0.0, format="%.2e")
            with col2:
                lidt = st.number_input("LIDT (J/cm²)", 0.0, 1000.0, 0.0)
                k_th = st.number_input("Thermal conductivity (W/m·K)", 0.0, 2000.0, 0.0)
                t_melt = st.number_input("Melting point (K)", 0.0, 5000.0, 0.0)
                t_boil = st.number_input("Boiling point (K)", 0.0, 10000.0, 0.0)
                cp = st.number_input("Specific heat (J/kg·K)", 0.0, 10000.0, 0.0)
                rho = st.number_input("Density (kg/m³)", 0.0, 30000.0, 0.0)
            notes = st.text_area("Notes")

            if st.form_submit_button("Save Material", type="primary"):
                if not name.strip():
                    st.error("Name is required.")
                else:
                    custom = load_custom_materials()
                    custom.append(Material(
                        name=name.strip(), category=category,
                        refractive_index=n, absorption_coeff_cm=alpha,
                        bandgap_ev=bandgap, nonlinear_index_cm2_w=n2,
                        damage_threshold_j_cm2=lidt,
                        thermal_conductivity_w_mk=k_th, melting_point_k=t_melt,
                        boiling_point_k=t_boil, specific_heat_j_kgk=cp,
                        density_kg_m3=rho, notes=notes.strip(),
                    ))
                    save_custom_materials(custom)
                    st.success(f"Saved '{name}'")
                    st.rerun()
