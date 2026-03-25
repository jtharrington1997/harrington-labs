"""6_Coatings_Lab.py — Optical Coatings Lab Simulator."""
import streamlit as st
import plotly.graph_objects as go
from harrington_labs.ui import render_header, lab_panel, make_figure, show_figure, warning_box, COLORS
from harrington_labs.domain import (
    CoatingDesign, ThinFilmLayer, CoatingType, SubstrateType,
)
from harrington_labs.simulation.coatings import (
    run_coating_simulation,
    quarter_wave_ar, v_coat_ar, quarter_wave_stack_hr, broadband_ar,
    COATING_MATERIALS,
)

st.set_page_config(page_title="Coatings Lab", layout="wide")
render_header("Coatings Lab", "Transfer matrix • Spectral/angular response • E-field • GDD")

# ── Sidebar: preset or custom ────────────────────────────────────
st.sidebar.header("Coating Design")
design_mode = st.sidebar.radio("Mode", ["Preset Design", "Custom Stack"])

substrate = st.sidebar.selectbox("Substrate", [s.value for s in SubstrateType])
design_wl = st.sidebar.number_input("Design Wavelength (nm)", 200.0, 12000.0, 1064.0, 1.0)
aoi = st.sidebar.number_input("Angle of Incidence (°)", 0.0, 85.0, 0.0, 1.0)

sub_type = SubstrateType(substrate)

if design_mode == "Preset Design":
    preset = st.sidebar.selectbox("Preset", [
        "Quarter-Wave AR (MgF2)",
        "V-Coat AR (Ta2O5/MgF2)",
        "HR Mirror (TiO2/SiO2)",
        "Broadband AR (4-layer)",
    ])
    if preset == "Quarter-Wave AR (MgF2)":
        design = quarter_wave_ar(design_wl, sub_type)
    elif preset == "V-Coat AR (Ta2O5/MgF2)":
        design = v_coat_ar(design_wl, sub_type)
    elif preset == "HR Mirror (TiO2/SiO2)":
        n_pairs = st.sidebar.number_input("Layer Pairs", 2, 30, 10)
        design = quarter_wave_stack_hr(design_wl, n_pairs, sub_type)
    else:
        design = broadband_ar(design_wl, sub_type)
    design.angle_of_incidence_deg = aoi

else:
    # Custom stack builder
    st.sidebar.subheader("Layer Stack (top → substrate)")
    n_layers = st.sidebar.number_input("Number of Layers", 1, 30, 4)
    avail_mats = list(COATING_MATERIALS.keys())
    layers = []
    for i in range(int(n_layers)):
        with st.sidebar.expander(f"Layer {i+1}", expanded=(i < 3)):
            mat = st.selectbox(f"Material##L{i}", avail_mats, key=f"mat_{i}")
            n_val = COATING_MATERIALS[mat]["n"]
            k_val = COATING_MATERIALS[mat]["k"]
            qwot = design_wl / (4 * n_val)
            thick = st.number_input(
                f"Thickness (nm)##L{i}", 1.0, 5000.0, round(qwot, 1),
                1.0, key=f"thick_{i}",
            )
            layers.append(ThinFilmLayer(mat, thick, n_val, k_val))

    from harrington_labs.simulation.coatings import _SUBSTRATE_N
    design = CoatingDesign(
        name="Custom Stack",
        coating_type=CoatingType.DIELECTRIC_STACK,
        substrate=sub_type,
        substrate_n=_SUBSTRATE_N.get(sub_type, 1.52),
        design_wavelength_nm=design_wl,
        layers=layers,
        angle_of_incidence_deg=aoi,
    )

result = run_coating_simulation(design)
warning_box(result.warnings)

# ── Design performance ───────────────────────────────────────────
with lab_panel("Design Performance"):
    perf = result.data["design_performance"]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric(f"R @ {design_wl:.0f} nm", f"{perf['R_at_design']:.4f}")
    c2.metric(f"T @ {design_wl:.0f} nm", f"{perf['T_at_design']:.4f}")
    c3.metric("Total Thickness", f"{perf['total_thickness_nm']:.0f} nm")
    c4.metric("Layers", f"{perf['n_layers']}")

# ── Spectral response ───────────────────────────────────────────
with lab_panel("Spectral Response"):
    st.caption("Reflectance and transmittance across wavelength range.")
    c1, c2 = st.columns(2)
    wl_min = c1.number_input("Min λ (nm)", 200.0, 10000.0, 400.0, 50.0)
    wl_max = c2.number_input("Max λ (nm)", 200.0, 12000.0, 1200.0, 50.0)

    # Recompute with custom range if needed
    if wl_min != 400.0 or wl_max != 1200.0:
        from harrington_labs.simulation.coatings import spectral_response
        spec = spectral_response(design, (wl_min, wl_max))
    else:
        spec = result.data["spectral"]

    fig = make_figure("Reflectance & Transmittance")
    fig.add_trace(go.Scatter(
        x=spec["wavelength_nm"], y=spec["reflectance"],
        name="Reflectance", line=dict(color=COLORS[0], width=2.5),
    ))
    fig.add_trace(go.Scatter(
        x=spec["wavelength_nm"], y=spec["transmittance"],
        name="Transmittance", line=dict(color=COLORS[2], width=2),
    ))
    fig.add_vline(x=design_wl, line_dash="dot", line_color=COLORS[1],
                  annotation_text=f"Design λ = {design_wl:.0f} nm")
    fig.update_xaxes(title_text="Wavelength (nm)")
    fig.update_yaxes(title_text="R / T", range=[-0.02, 1.05])
    show_figure(fig)

# ── Angular response ─────────────────────────────────────────────
col1, col2 = st.columns(2)

with col1:
    with lab_panel("Angular Response"):
        ang = result.data["angular"]
        fig = make_figure(f"Reflectance vs Angle @ {design_wl:.0f} nm")
        fig.add_trace(go.Scatter(
            x=ang["angle_deg"], y=ang["reflectance_s"],
            name="s-pol", line=dict(color=COLORS[0], width=2),
        ))
        fig.add_trace(go.Scatter(
            x=ang["angle_deg"], y=ang["reflectance_p"],
            name="p-pol", line=dict(color=COLORS[1], width=2),
        ))
        fig.add_trace(go.Scatter(
            x=ang["angle_deg"], y=ang["reflectance_avg"],
            name="Average", line=dict(color=COLORS[2], width=2, dash="dash"),
        ))
        fig.update_xaxes(title_text="Angle of Incidence (°)")
        fig.update_yaxes(title_text="Reflectance", range=[-0.02, 1.05])
        show_figure(fig)

with col2:
    with lab_panel("Electric Field Profile"):
        ef = result.data["e_field"]
        if len(ef["position_nm"]) > 0:
            fig = make_figure("E-Field Through Stack")
            fig.add_trace(go.Scatter(
                x=ef["position_nm"], y=ef["e_field_normalized"],
                name="|E|²", line=dict(color=COLORS[3], width=2),
                fill="tozeroy", fillcolor="rgba(45,106,79,0.12)",
            ))
            # Layer boundaries
            for boundary in ef["layer_boundaries_nm"]:
                fig.add_vline(x=boundary, line_dash="dot", line_color="rgba(0,0,0,0.2)")
            fig.update_xaxes(title_text="Position in Stack (nm)")
            fig.update_yaxes(title_text="Normalized |E|")
            show_figure(fig)
        else:
            st.info("E-field profile not available for this configuration.")

# ── GDD ──────────────────────────────────────────────────────────
with lab_panel("Group Delay Dispersion"):
    gdd = result.data["gdd"]
    fig = make_figure("GDD of Coating")
    fig.add_trace(go.Scatter(
        x=gdd["wavelength_nm"], y=gdd["gdd_fs2"],
        name="GDD", line=dict(color=COLORS[4], width=2),
    ))
    fig.add_hline(y=0, line_dash="dot", line_color="gray")
    fig.update_xaxes(title_text="Wavelength (nm)")
    fig.update_yaxes(title_text="GDD (fs²)")
    show_figure(fig)

# ── Layer table ──────────────────────────────────────────────────
with lab_panel("Layer Stack Details"):
    import pandas as pd
    rows = []
    for i, layer in enumerate(design.layers):
        rows.append({
            "Layer": i + 1,
            "Material": layer.material,
            "Thickness (nm)": f"{layer.thickness_nm:.1f}",
            "n": f"{layer.refractive_index:.3f}",
            "k": f"{layer.extinction_coefficient:.4f}",
            "QWOT": f"{layer.thickness_nm / (design.design_wavelength_nm / (4 * layer.refractive_index)):.3f}",
        })
    if rows:
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

# ── Model Comparison ────────────────────────────────────────────
from harrington_labs.comparison.ui import model_comparison_panel, reference_upload_panel

model_comparison_panel(
    sim_x=spec["wavelength_nm"],
    sim_y=spec["reflectance"],
    x_label="Wavelength",
    y_label="Reflectance",
    x_unit="nm",
    panel_title="Model Comparison — Spectral Reflectance",
    key_prefix="coat_refl",
)

reference_upload_panel(key_prefix="coat_ref", save_dir="data/references")
