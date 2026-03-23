"""
pages/1_Laser_Library.py — Laser Source Library
Browse default commercial lasers and add custom configurations.
Supports spatial beam modes, OPA parent-child relationships, and
wavelength-aware parameter display.
"""
from __future__ import annotations

import pandas as pd
import streamlit as st

from harrington_labs.lmi.domain.lasers import (
    LaserSource,
    SpatialMode,
    all_lasers,
    load_custom_lasers,
    save_custom_lasers,
)
from harrington_labs.lmi.ui.access import is_admin
from harrington_labs.lmi.ui.branding import lmi_panel
from harrington_labs.lmi.ui.formatting import (
    fmt_energy_j,
    fmt_fluence_j_cm2,
    fmt_frequency_hz,
    fmt_irradiance_w_cm2,
    fmt_power_w,
    fmt_time_s,
    fmt_wavelength_nm,
)
from harrington_labs.lmi.ui.layout import render_header

st.set_page_config(page_title="Laser Library", layout="wide")
render_header()

lasers = all_lasers()

with lmi_panel():
    st.subheader("Laser Source Library")

    rows = []
    for l in lasers:
        rows.append(
            {
                "Name": l.name,
                "Wavelength": fmt_wavelength_nm(l.wavelength_nm),
                "Mode": "CW" if l.is_cw else "Pulsed",
                "Average Power": fmt_power_w(l.power_w),
                "Rep Rate": fmt_frequency_hz(l.rep_rate_hz),
                "Pulse Width": fmt_time_s(l.pulse_width_s),
                "Pulse Energy": "—" if l.is_cw else fmt_energy_j(l.pulse_energy_j),
                "Peak Power": fmt_power_w(l.peak_power_w),
                "Beam Ø": f"{l.beam_diameter_mm:.3f} mm",
                "M²": f"{l.m_squared:.1f}",
                "Spatial Mode": l.spatial_mode,
                "Gain Medium": l.gain_medium or "—",
                "Tunable": "Yes" if l.is_tunable else "—",
            }
        )

    df = pd.DataFrame(rows)
    st.dataframe(df, width="stretch", hide_index=True)

with lmi_panel():
    st.subheader("Laser Details")
    selected_name = st.selectbox("Select laser", [l.name for l in lasers])
    laser = next(l for l in lasers if l.name == selected_name)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Wavelength", fmt_wavelength_nm(laser.wavelength_nm))
    col2.metric("Photon Energy", f"{laser.photon_energy_ev:.4f} eV")
    col3.metric("Average Power", fmt_power_w(laser.power_w))
    col4.metric("Peak Power", fmt_power_w(laser.peak_power_w))

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Peak Irradiance", fmt_irradiance_w_cm2(laser.irradiance_w_cm2))
    col2.metric("Fluence", fmt_fluence_j_cm2(laser.fluence_j_cm2) if not laser.is_cw else "N/A (CW)")
    col3.metric("Pulse Energy", fmt_energy_j(laser.pulse_energy_j) if not laser.is_cw else "N/A (CW)")
    col4.metric("Spatial Mode", laser.spatial_mode)

    col1, col2, col3 = st.columns(3)
    col1.metric("Repetition Rate", fmt_frequency_hz(laser.rep_rate_hz))
    col2.metric("Pulse Width", fmt_time_s(laser.pulse_width_s))
    col3.metric("Beam Diameter", f"{laser.beam_diameter_mm:.3f} mm")

    if laser.is_tunable and laser.tunable_range_nm:
        st.info(
            f"Tunable range: {fmt_wavelength_nm(laser.tunable_range_nm[0], dual=False)} – "
            f"{fmt_wavelength_nm(laser.tunable_range_nm[1], dual=False)}"
        )
    if laser.pump_source:
        st.info(f"Pump source: {laser.pump_source}")
    if laser.notes:
        st.caption(laser.notes)

if is_admin():
    with lmi_panel():
        st.subheader("Add Custom Laser")
        with st.form("new_laser"):
            col1, col2 = st.columns(2)
            with col1:
                name = st.text_input("Name")
                wl_unit = st.selectbox("Wavelength unit", ["nm", "µm", "cm⁻¹"])
                wl_value = st.number_input(
                    f"Wavelength ({wl_unit})", 0.001, 1e6, 1064.0, format="%.3f"
                )
                power = st.number_input("Average Power (W)", 0.001, 1e6, 1.0)
                beam_d = st.number_input("Beam diameter (mm)", 0.001, 100.0, 2.0)
                spatial = st.selectbox("Spatial mode", [m.value for m in SpatialMode])
            with col2:
                rep_rate = st.number_input("Rep rate (Hz, 0 = CW)", 0.0, 1e9, 0.0)
                pulse_w = st.number_input("Pulse width (s, 0 = CW)", 0.0, 1.0, 0.0, format="%.2e")
                m2 = st.number_input("M²", 1.0, 100.0, 1.0)
                gain = st.text_input("Gain medium")
                pump_src = st.selectbox("Pump source (for OPA/OPO)", ["None"] + [l.name for l in lasers])
                tunable = st.checkbox("Tunable source")
                if tunable:
                    t_col1, t_col2 = st.columns(2)
                    with t_col1:
                        t_min = st.number_input("Min λ (nm)", 100.0, 20000.0, 630.0)
                    with t_col2:
                        t_max = st.number_input("Max λ (nm)", 100.0, 20000.0, 16000.0)

            notes = st.text_area("Notes")

            if st.form_submit_button("Save Laser", type="primary"):
                if not name.strip():
                    st.error("Name is required.")
                else:
                    if wl_unit == "µm":
                        wavelength_nm = wl_value * 1000
                    elif wl_unit == "cm⁻¹":
                        wavelength_nm = 1e7 / wl_value if wl_value > 0 else 1064.0
                    else:
                        wavelength_nm = wl_value

                    custom = load_custom_lasers()
                    custom.append(
                        LaserSource(
                            name=name.strip(),
                            wavelength_nm=wavelength_nm,
                            power_w=power,
                            rep_rate_hz=rep_rate,
                            pulse_width_s=pulse_w,
                            beam_diameter_mm=beam_d,
                            m_squared=m2,
                            spatial_mode=spatial,
                            gain_medium=gain.strip(),
                            pump_source=pump_src if pump_src != "None" else "",
                            tunable_range_nm=(t_min, t_max) if tunable else None,
                            notes=notes.strip(),
                        )
                    )
                    save_custom_lasers(custom)
                    st.success(f"Saved '{name}'")
                    st.rerun()
