from __future__ import annotations

from pathlib import Path

from harrington_labs.lmi.domain.plot_spec import PlotSpec


def _sanitize_name(name: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in name).strip("_") or "plot"


def write_gnuplot_bundle(plot: PlotSpec, outdir: str | Path, stem: str | None = None) -> dict[str, Path]:
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    stem = _sanitize_name(stem or plot.title.lower())
    dat_path = outdir / f"{stem}.dat"
    gp_path = outdir / f"{stem}.gp"

    with dat_path.open("w", encoding="utf-8") as f:
        for idx, s in enumerate(plot.series):
            f.write(f"# series: {s.name}\n")
            for x, y in zip(s.x, s.y):
                f.write(f"{x}\t{y}\n")
            if idx < len(plot.series) - 1:
                f.write("\n\n")

    plot_cmds: list[str] = []
    index = 0
    for s in plot.series:
        plot_cmds.append(
            f"'{dat_path.name}' index {index} using 1:2 with lines linewidth 2 title '{s.name}'"
        )
        index += 1

    gp = f"""set terminal svg size 1200,800 dynamic
set output '{stem}.svg'
set title '{plot.title}'
set xlabel '{plot.x_label}'
set ylabel '{plot.y_label}'
set grid
{"set logscale x" if plot.x_log else "unset logscale x"}
{"set logscale y" if plot.y_log else "unset logscale y"}
plot {", ".join(plot_cmds)}
"""
    gp_path.write_text(gp, encoding="utf-8")

    return {"data": dat_path, "script": gp_path}
