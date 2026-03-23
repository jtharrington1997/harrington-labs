from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence


@dataclass(slots=True)
class SeriesSpec:
    name: str
    x: Sequence[float]
    y: Sequence[float]
    x_label: str
    y_label: str
    mode: str = "lines"


@dataclass(slots=True)
class PlotSpec:
    title: str
    series: list[SeriesSpec] = field(default_factory=list)
    x_label: str = ""
    y_label: str = ""
    x_log: bool = False
    y_log: bool = False
