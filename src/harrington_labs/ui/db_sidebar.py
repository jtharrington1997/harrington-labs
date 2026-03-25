"""Sidebar components for laser and material database access.

Provides drop-in sidebar selectors that any lab page can use to
pull parameters from the shared databases instead of manual entry.
"""
from __future__ import annotations

from typing import Optional
import streamlit as st

from harrington_labs.lmi.domain.lasers import LaserSource, all_lasers
from harrington_labs.lmi.domain.materials import Material, all_materials


def laser_source_selector(
    key_prefix: str = "src",
    show_params: bool = True,
    default_source: str = "",
    category_filter: str = "",
) -> Optional[LaserSource]:
    """Render a sidebar light source selector.

    Returns the selected LaserSource, or None if 'Manual Entry' is chosen.
    Displays key parameters in the sidebar when show_params is True.
    Supports optional category_filter to show only e.g. "laser" or "lamp".
    """
    try:
        lasers = all_lasers()
    except Exception:
        return None

    if not lasers:
        return None

    if category_filter:
        lasers = [l for l in lasers if l.category == category_filter]

    # Group by category for display
    _CAT_LABELS = {
        "laser": "Lasers",
        "lamp": "Lamps",
        "led": "LEDs",
        "broadband": "Broadband",
        "synchrotron": "Synchrotron",
    }
    categories_present = sorted(set(l.category for l in lasers), key=lambda c: list(_CAT_LABELS).index(c) if c in _CAT_LABELS else 99)

    # Build grouped name list
    grouped_names = ["Manual Entry"]
    name_to_source = {}
    for cat in categories_present:
        cat_label = _CAT_LABELS.get(cat, cat.title())
        cat_sources = [l for l in lasers if l.category == cat]
        grouped_names.append(f"── {cat_label} ──")
        for l in cat_sources:
            tunable_tag = " ◆" if l.is_tunable else ""
            display = f"  {l.name}{tunable_tag}"
            grouped_names.append(display)
            name_to_source[display] = l

    default_idx = 0
    if default_source:
        for i, n in enumerate(grouped_names):
            if default_source.lower() in n.lower():
                default_idx = i
                break

    with st.sidebar.expander("Source Database", expanded=False):
        selected = st.selectbox(
            "Load from library",
            grouped_names,
            index=default_idx,
            key=f"{key_prefix}_laser_sel",
            help="Select a source to auto-fill parameters. ◆ = tunable.",
        )

        if selected == "Manual Entry" or selected.startswith("──"):
            return None

        laser = name_to_source.get(selected)
        if laser is None:
            return None

        if show_params:
            cat_icon = {"laser": "[L]", "lamp": "[W]", "led": "[D]", "broadband": "[B]"}.get(laser.category, "")
            st.caption(f"{cat_icon} **{laser.name}**")
            cols = st.columns(2)
            cols[0].markdown(f"λ = **{laser.wavelength_nm:.1f} nm**")
            cols[1].markdown(f"P = **{laser.power_w:.2g} W**")

            if not laser.is_cw and laser.category == "laser":
                cols = st.columns(2)
                if laser.pulse_width_s >= 1e-9:
                    cols[0].markdown(f"τ = **{laser.pulse_width_s * 1e9:.1f} ns**")
                else:
                    cols[0].markdown(f"τ = **{laser.pulse_width_s * 1e15:.0f} fs**")
                cols[1].markdown(f"f = **{laser.rep_rate_hz:.0f} Hz**")

            if laser.category == "laser":
                st.markdown(f"⌀ = **{laser.beam_diameter_mm:.1f} mm**, M² = **{laser.m_squared:.2f}**")

            if laser.is_tunable:
                lo, hi = laser.tunable_range_nm
                st.caption(f"Tunable: {lo:.0f} – {hi:.0f} nm")

            if laser.emission_lines_nm:
                st.caption(f"{len(laser.emission_lines_nm)} emission lines")

            if laser.spectral_bandwidth_nm > 0 and laser.category != "laser":
                st.caption(f"Bandwidth: {laser.spectral_bandwidth_nm:.1f} nm FWHM")

            if laser.lifetime_hours > 0:
                st.caption(f"Lifetime: {laser.lifetime_hours:.0f} h")

        return laser


def material_selector(
    key_prefix: str = "mat",
    show_params: bool = True,
    default_material: str = "",
) -> Optional[Material]:
    """Render a sidebar material selector.

    Returns the selected Material, or None if 'Manual Entry' is chosen.
    """
    try:
        materials = all_materials()
    except Exception:
        return None

    if not materials:
        return None

    names = ["Manual Entry"] + [m.name for m in materials]
    default_idx = 0
    if default_material:
        for i, n in enumerate(names):
            if default_material.lower() in n.lower():
                default_idx = i
                break

    with st.sidebar.expander("Material Database", expanded=False):
        selected = st.selectbox(
            "Load from database",
            names,
            index=default_idx,
            key=f"{key_prefix}_mat_sel",
            help="Select a material to auto-fill properties, or use Manual Entry.",
        )

        if selected == "Manual Entry":
            return None

        mat = next(m for m in materials if m.name == selected)

        if show_params:
            st.caption(f"**{mat.name}**")
            cols = st.columns(2)
            cols[0].markdown(f"n = **{mat.refractive_index:.3f}**")
            if mat.bandgap_ev > 0:
                cols[1].markdown(f"E_g = **{mat.bandgap_ev:.2f} eV**")
            else:
                cols[1].markdown(f"E_g = **metallic**")

            if mat.has_sellmeier:
                st.caption("Sellmeier dispersion available")

            props = []
            if mat.density_kg_m3 > 0:
                props.append(f"ρ = {mat.density_kg_m3:.0f} kg/m³")
            if mat.thermal_conductivity_w_mk > 0:
                props.append(f"k = {mat.thermal_conductivity_w_mk:.1f} W/m·K")
            if mat.melting_point_k > 0:
                props.append(f"T_m = {mat.melting_point_k:.0f} K")
            if props:
                st.caption(" | ".join(props))

        return mat


def source_and_material_sidebar(
    key_prefix: str = "db",
    show_source: bool = True,
    show_material: bool = True,
    default_source: str = "",
    default_material: str = "",
) -> tuple[Optional[LaserSource], Optional[Material]]:
    """Convenience: add both selectors to the sidebar.

    Returns (laser_or_None, material_or_None).
    """
    laser = None
    mat = None
    if show_source:
        laser = laser_source_selector(f"{key_prefix}_src", default_source=default_source)
    if show_material:
        mat = material_selector(f"{key_prefix}_mat", default_material=default_material)
    return laser, mat
