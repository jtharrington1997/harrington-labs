"""Model-vs-experiment comparison framework.

Provides statistical metrics, residual analysis, and figure-of-merit
computation for benchmarking simulation engines against measured data.
Designed to be reusable across all lab pages. No Streamlit imports.
"""
from __future__ import annotations

from .metrics import (
    ComparisonResult,
    compare_curves,
    compare_scalar,
    scorecard,
)
from .parsers import (
    ExperimentalDataset,
    parse_knife_edge_xlsx,
    parse_generic_csv,
    parse_generic_xlsx,
    detect_and_parse,
)

__all__ = [
    "ComparisonResult",
    "compare_curves",
    "compare_scalar",
    "scorecard",
    "ExperimentalDataset",
    "parse_knife_edge_xlsx",
    "parse_generic_csv",
    "parse_generic_xlsx",
    "detect_and_parse",
]
