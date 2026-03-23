from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path

from harrington_labs.lmi.domain.plot_spec import PlotSpec
from harrington_labs.lmi.io.gnuplot import write_gnuplot_bundle

def timestamped_stem(stem: str) -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{stem}_{ts}"


def export_plot_csv(plot: PlotSpec, outdir: str | Path, stem: str) -> Path:
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    path = outdir / f"{stem}.csv"

    max_len = max((len(s.x) for s in plot.series), default=0)
    headers: list[str] = []
    for s in plot.series:
        headers.extend([f"{s.name}_x", f"{s.name}_y"])

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        for i in range(max_len):
            row: list[object] = []
            for s in plot.series:
                row.append(s.x[i] if i < len(s.x) else "")
                row.append(s.y[i] if i < len(s.y) else "")
            writer.writerow(row)

    return path


def export_plot_json(plot: PlotSpec, outdir: str | Path, stem: str) -> Path:
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    path = outdir / f"{stem}.json"

    payload = {
        "title": plot.title,
        "x_label": plot.x_label,
        "y_label": plot.y_label,
        "x_log": plot.x_log,
        "y_log": plot.y_log,
        "series": [
            {
                "name": s.name,
                "x_label": s.x_label,
                "y_label": s.y_label,
                "x": list(s.x),
                "y": list(s.y),
                "mode": s.mode,
            }
            for s in plot.series
        ],
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def export_plot_bundle(
    plot: PlotSpec,
    outdir: str | Path,
    stem: str,
    timestamp: bool = True,
) -> dict[str, Path]:
    final_stem = timestamped_stem(stem) if timestamp else stem

    files = {}
    files["csv"] = export_plot_csv(plot, outdir, final_stem)
    files["json"] = export_plot_json(plot, outdir, final_stem)
    files |= write_gnuplot_bundle(plot, outdir, final_stem)
    return files
