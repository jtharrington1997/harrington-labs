"""
pages/70_Gnuplot_Legacy.py — Legacy Gnuplot Workspace

Legacy publication-plot workspace for gnuplot-based rendering.

This page remains useful until plot export and static rendering are fully
migrated into the unified Modeling & Simulation workspace.

It supports:
- canned starter templates
- live gnuplot script editing
- uploaded data plotting
- PNG rendering
- .gp script download
- workflow handoff back to the other legacy pages
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import streamlit as st

from harrington_labs.ui import render_header
from harrington_labs.ui import lab_panel, make_figure, show_figure, COLORS


def _apply_page_header() -> None:
    st.set_page_config(page_title="Gnuplot Legacy", layout="wide")
    render_header()


def _detect_gnuplot() -> str | None:
    gnuplot_bin = shutil.which("gnuplot")
    if gnuplot_bin is not None:
        return gnuplot_bin

    for candidate in [
        r"C:\Program Files\gnuplot\bin\gnuplot.exe",
        r"C:\Program Files (x86)\gnuplot\bin\gnuplot.exe",
        r"C:\gnuplot\bin\gnuplot.exe",
        "/usr/bin/gnuplot",
        "/usr/local/bin/gnuplot",
    ]:
        if os.path.isfile(candidate):
            return candidate
    return None


def _run_gnuplot(
    gnuplot_bin: str,
    script: str,
    data_files: dict[str, str] | None = None,
) -> tuple[bool, bytes | None, str]:
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)

        if data_files:
            for fname, content in data_files.items():
                (tmp_path / fname).write_text(content, encoding="utf-8")

        gp_path = tmp_path / "plot.gp"
        gp_path.write_text(script, encoding="utf-8")

        try:
            result = subprocess.run(
                [gnuplot_bin, str(gp_path)],
                cwd=tmpdir,
                capture_output=True,
                timeout=20,
                check=False,
            )
        except subprocess.TimeoutExpired:
            return False, None, "gnuplot timed out after 20 seconds."

        png_bytes = None
        for fname in tmp_path.iterdir():
            if fname.suffix.lower() == ".png":
                png_bytes = fname.read_bytes()
                break

        stderr_text = result.stderr.decode("utf-8", errors="replace")
        return result.returncode == 0, png_bytes, stderr_text


def _starter_dispersion_script() -> str:
    return """\
# Refractive Index Dispersion Curve
# Replace the inline block with your own wavelength-index data.

set terminal pngcairo enhanced font "Arial,12" size 1400,850
set output "dispersion.png"

set border lw 1.2 linecolor rgb "#222222"
set grid lw 0.8 linecolor rgb "#d9d9d9"
set key opaque box linecolor rgb "#cccccc"
set tics nomirror out
set format x "%.3f"
set format y "%.3e"

set title "Refractive Index Dispersion"
set xlabel "Wavelength ({/Symbol m}m)"
set ylabel "Refractive index, n"

$DATA << EOD
1.0   3.510
2.0   3.453
4.0   3.429
6.0   3.422
8.0   3.419
8.5   3.418
10.0  3.416
12.0  3.410
14.0  3.402
EOD

plot $DATA using 1:2 with linespoints \
    pt 7 ps 1.0 lw 2.2 lc rgb "#1f4e79" \
    title "Example material"
"""


def _starter_beam_script() -> str:
    return """\
# Beam Profile Comparison
# Compare Gaussian and flat-top style transverse profiles.

set terminal pngcairo enhanced font "Arial,12" size 1400,850
set output "beam_profile.png"

set border lw 1.2 linecolor rgb "#222222"
set grid lw 0.8 linecolor rgb "#d9d9d9"
set key opaque box linecolor rgb "#cccccc"
set tics nomirror out
set format x "%.3f"
set format y "%.3e"

set title "Transverse Beam Profile Comparison"
set xlabel "Radial position ({/Symbol m}m)"
set ylabel "Normalized intensity"

w0 = 100.0
gaussian(x) = exp(-2.0*(x/w0)**2)
tophat(x)   = exp(-2.0*(x/w0)**20)

set xrange [-300:300]
set samples 800

plot gaussian(x) with lines lw 2.4 lc rgb "#1f4e79" title "TEM00 Gaussian", \
     tophat(x)   with lines lw 2.4 lc rgb "#8b2332" title "Flat-top proxy"
"""


def _starter_custom_script() -> str:
    return """\
# Custom gnuplot script

set terminal pngcairo enhanced font "Arial,12" size 1400,850
set output "custom_plot.png"

set border lw 1.2 linecolor rgb "#222222"
set grid lw 0.8 linecolor rgb "#d9d9d9"
set key opaque box linecolor rgb "#cccccc"
set tics nomirror out
set format x "%.3e"
set format y "%.3e"

set xlabel "X"
set ylabel "Y"
set title "Custom Plot"

$DATA << EOD
1 1
2 4
3 9
4 16
5 25
EOD

plot $DATA using 1:2 with linespoints \
    pt 7 ps 1.0 lw 2.2 lc rgb "#8b2332" \
    title "Sample"
"""


_apply_page_header()

gnuplot_bin = _detect_gnuplot()

with lab_panel():
    st.subheader("Gnuplot Legacy")
    st.caption(
        "Publication-oriented static plotting workspace for gnuplot scripts, "
        "uploaded data, and export-ready render checks."
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

if gnuplot_bin is None:
    with lab_panel():
        st.error(
            "gnuplot was not found on this system. Install it with "
            "`sudo apt install gnuplot` on Linux or `winget install gnuplot` on Windows, "
            "then restart the app."
        )
    st.stop()

st.sidebar.success(f"gnuplot: `{gnuplot_bin}`")

tab_disp, tab_beam, tab_custom, tab_upload = st.tabs(
    ["Dispersion Curve", "Beam Profile", "Custom Script", "Upload Data"]
)

with tab_disp:
    with lab_panel():
        st.subheader("Dispersion Curve Template")
        left, right = st.columns([1, 1])

        with left:
            disp_script = st.text_area(
                "gnuplot script",
                value=_starter_dispersion_script(),
                height=440,
                key="gp_disp_script",
            )
            b1, b2 = st.columns(2)
            with b1:
                render_disp = st.button("Render", type="primary", key="render_disp", width="stretch")
            with b2:
                st.download_button(
                    "Download .gp",
                    data=disp_script,
                    file_name="dispersion.gp",
                    mime="text/plain",
                    width="stretch",
                )

        with right:
            if render_disp:
                ok, png, stderr = _run_gnuplot(gnuplot_bin, disp_script)
                if ok and png:
                    st.image(png, caption="dispersion.png", width="stretch")
                    st.download_button(
                        "Download PNG",
                        data=png,
                        file_name="dispersion.png",
                        mime="image/png",
                        key="dl_disp_png",
                        width="stretch",
                    )
                else:
                    st.error("gnuplot failed.")
                    st.code(stderr, language="text")
            else:
                st.info("Render the current script to preview the output.")

with tab_beam:
    with lab_panel():
        st.subheader("Beam Profile Template")
        left, right = st.columns([1, 1])

        with left:
            beam_script = st.text_area(
                "gnuplot script",
                value=_starter_beam_script(),
                height=440,
                key="gp_beam_script",
            )
            b1, b2 = st.columns(2)
            with b1:
                render_beam = st.button("Render", type="primary", key="render_beam", width="stretch")
            with b2:
                st.download_button(
                    "Download .gp",
                    data=beam_script,
                    file_name="beam_profile.gp",
                    mime="text/plain",
                    width="stretch",
                )

        with right:
            if render_beam:
                ok, png, stderr = _run_gnuplot(gnuplot_bin, beam_script)
                if ok and png:
                    st.image(png, caption="beam_profile.png", width="stretch")
                    st.download_button(
                        "Download PNG",
                        data=png,
                        file_name="beam_profile.png",
                        mime="image/png",
                        key="dl_beam_png",
                        width="stretch",
                    )
                else:
                    st.error("gnuplot failed.")
                    st.code(stderr, language="text")
            else:
                st.info("Render the current script to preview the output.")

with tab_custom:
    with lab_panel():
        st.subheader("Custom Script")
        left, right = st.columns([1, 1])

        with left:
            custom_script = st.text_area(
                "gnuplot script",
                value=_starter_custom_script(),
                height=440,
                key="gp_custom_script",
            )
            b1, b2 = st.columns(2)
            with b1:
                render_custom = st.button("Render", type="primary", key="render_custom", width="stretch")
            with b2:
                st.download_button(
                    "Download .gp",
                    data=custom_script,
                    file_name="custom_plot.gp",
                    mime="text/plain",
                    width="stretch",
                )

        with right:
            if render_custom:
                ok, png, stderr = _run_gnuplot(gnuplot_bin, custom_script)
                if ok and png:
                    st.image(png, caption="custom_plot.png", width="stretch")
                    st.download_button(
                        "Download PNG",
                        data=png,
                        file_name="custom_plot.png",
                        mime="image/png",
                        key="dl_custom_png",
                        width="stretch",
                    )
                else:
                    st.error("gnuplot failed.")
                    st.code(stderr, language="text")
            else:
                st.info("Render the current script to preview the output.")

with tab_upload:
    with lab_panel():
        st.subheader("Plot from Uploaded Data")
        st.caption(
            "Upload a `.dat`, `.csv`, `.tsv`, or `.txt` file and write a gnuplot "
            "script that references it directly."
        )

        uploaded = st.file_uploader("Upload data file", type=["dat", "csv", "tsv", "txt"])
        if uploaded is None:
            st.info("Upload a data file to activate this tab.")
        else:
            data_content = uploaded.read().decode("utf-8", errors="replace")

            st.text_area(
                "Data preview",
                value=data_content[:2500],
                height=180,
                disabled=True,
                key="gp_upload_preview",
            )

            upload_script = st.text_area(
                "gnuplot script",
                value=f"""\
set terminal pngcairo enhanced font "Arial,12" size 1400,850
set output "data_plot.png"

set border lw 1.2 linecolor rgb "#222222"
set grid lw 0.8 linecolor rgb "#d9d9d9"
set key opaque box linecolor rgb "#cccccc"
set tics nomirror out
set format x "%.3e"
set format y "%.3e"

set xlabel "Column 1"
set ylabel "Column 2"
set title "Uploaded Data Plot"

set datafile separator ","

plot "{uploaded.name}" using 1:2 with linespoints \
    pt 7 ps 1.0 lw 2.2 lc rgb "#1f4e79" \
    title "Uploaded data"
""",
                height=380,
                key="gp_upload_script",
            )

            b1, b2 = st.columns(2)
            with b1:
                render_upload = st.button("Render", type="primary", key="render_upload", width="stretch")
            with b2:
                st.download_button(
                    "Download .gp",
                    data=upload_script,
                    file_name="data_plot.gp",
                    mime="text/plain",
                    width="stretch",
                )

            if render_upload:
                ok, png, stderr = _run_gnuplot(
                    gnuplot_bin,
                    upload_script,
                    data_files={uploaded.name: data_content},
                )
                if ok and png:
                    st.image(png, caption="data_plot.png", width="stretch")
                    st.download_button(
                        "Download PNG",
                        data=png,
                        file_name="data_plot.png",
                        mime="image/png",
                        key="dl_upload_png",
                        width="stretch",
                    )
                else:
                    st.error("gnuplot failed.")
                    st.code(stderr, language="text")
