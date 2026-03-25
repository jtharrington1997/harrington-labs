"""Statistical metrics for model-vs-experiment comparison.

All functions operate on plain numpy arrays. No Streamlit imports.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class ComparisonResult:
    """Container for model-vs-experiment comparison metrics."""
    metric_name: str = ""
    rmse: float = 0.0
    nrmse: float = 0.0          # RMSE / (max - min) of measured
    r_squared: float = 0.0
    chi_squared: float = 0.0     # only if uncertainties provided
    max_abs_error: float = 0.0
    max_abs_error_at: float = 0.0  # x-value where max error occurs
    mean_residual: float = 0.0   # bias indicator
    n_points: int = 0
    residuals: np.ndarray = field(default_factory=lambda: np.array([]))
    x_common: np.ndarray = field(default_factory=lambda: np.array([]))
    y_measured: np.ndarray = field(default_factory=lambda: np.array([]))
    y_simulated: np.ndarray = field(default_factory=lambda: np.array([]))


def _interpolate_to_common_grid(
    x_meas: np.ndarray,
    y_meas: np.ndarray,
    x_sim: np.ndarray,
    y_sim: np.ndarray,
    n_points: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Interpolate both datasets onto a common x-grid.

    Uses the overlap region of both datasets. If n_points == 0,
    uses the measured x positions within the overlap.
    """
    x_lo = max(x_meas.min(), x_sim.min())
    x_hi = min(x_meas.max(), x_sim.max())
    if x_hi <= x_lo:
        raise ValueError(
            f"No overlap between measured [{x_meas.min():.4g}, {x_meas.max():.4g}] "
            f"and simulated [{x_sim.min():.4g}, {x_sim.max():.4g}]"
        )

    if n_points > 0:
        x_common = np.linspace(x_lo, x_hi, n_points)
    else:
        mask = (x_meas >= x_lo) & (x_meas <= x_hi)
        x_common = x_meas[mask]

    y_m = np.interp(x_common, x_meas, y_meas)
    y_s = np.interp(x_common, x_sim, y_sim)
    return x_common, y_m, y_s


def compare_curves(
    x_measured: np.ndarray,
    y_measured: np.ndarray,
    x_simulated: np.ndarray,
    y_simulated: np.ndarray,
    y_uncertainty: Optional[np.ndarray] = None,
    metric_name: str = "",
    n_points: int = 0,
) -> ComparisonResult:
    """Compare a simulated curve against measured data.

    Parameters
    ----------
    x_measured, y_measured : array
        Experimental data points.
    x_simulated, y_simulated : array
        Simulation output.
    y_uncertainty : array, optional
        1-sigma measurement uncertainty on y_measured for χ² calculation.
    metric_name : str
        Label for this comparison (e.g., "Normalized Transmission").
    n_points : int
        If > 0, interpolate both to this many common grid points.
        If 0, use measured x-values within the overlap region.
    """
    x_m = np.asarray(x_measured, dtype=np.float64)
    y_m = np.asarray(y_measured, dtype=np.float64)
    x_s = np.asarray(x_simulated, dtype=np.float64)
    y_s = np.asarray(y_simulated, dtype=np.float64)

    # Sort both by x
    sort_m = np.argsort(x_m)
    x_m, y_m = x_m[sort_m], y_m[sort_m]
    sort_s = np.argsort(x_s)
    x_s, y_s = x_s[sort_s], y_s[sort_s]

    x_common, y_mc, y_sc = _interpolate_to_common_grid(x_m, y_m, x_s, y_s, n_points)
    residuals = y_mc - y_sc
    n = len(x_common)

    rmse = float(np.sqrt(np.mean(residuals**2)))
    y_range = float(y_mc.max() - y_mc.min())
    nrmse = rmse / y_range if y_range > 0 else 0.0

    ss_res = float(np.sum(residuals**2))
    ss_tot = float(np.sum((y_mc - y_mc.mean())**2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    max_idx = int(np.argmax(np.abs(residuals)))
    max_abs_err = float(np.abs(residuals[max_idx]))
    max_err_at = float(x_common[max_idx])

    chi2 = 0.0
    if y_uncertainty is not None:
        sigma = np.interp(x_common, x_m, np.asarray(y_uncertainty, dtype=np.float64))
        mask = sigma > 0
        if mask.any():
            chi2 = float(np.sum((residuals[mask] / sigma[mask])**2))

    return ComparisonResult(
        metric_name=metric_name,
        rmse=rmse,
        nrmse=nrmse,
        r_squared=r2,
        chi_squared=chi2,
        max_abs_error=max_abs_err,
        max_abs_error_at=max_err_at,
        mean_residual=float(np.mean(residuals)),
        n_points=n,
        residuals=residuals,
        x_common=x_common,
        y_measured=y_mc,
        y_simulated=y_sc,
    )


def compare_scalar(
    measured: float,
    simulated: float,
    label: str = "",
    uncertainty: float = 0.0,
) -> dict:
    """Compare a single scalar value (threshold, peak location, etc.)."""
    abs_err = abs(measured - simulated)
    pct_err = abs_err / abs(measured) * 100 if measured != 0 else 0.0
    within_sigma = abs_err <= uncertainty if uncertainty > 0 else None
    return {
        "label": label,
        "measured": measured,
        "simulated": simulated,
        "abs_error": abs_err,
        "pct_error": pct_err,
        "uncertainty": uncertainty,
        "within_1sigma": within_sigma,
    }


def scorecard(results: list[ComparisonResult]) -> dict:
    """Aggregate multiple comparison results into a summary scorecard."""
    if not results:
        return {}
    avg_r2 = float(np.mean([r.r_squared for r in results]))
    avg_nrmse = float(np.mean([r.nrmse for r in results]))
    worst = max(results, key=lambda r: r.nrmse)
    best = min(results, key=lambda r: r.nrmse)
    return {
        "n_comparisons": len(results),
        "avg_r_squared": avg_r2,
        "avg_nrmse": avg_nrmse,
        "best_match": best.metric_name,
        "best_nrmse": best.nrmse,
        "worst_match": worst.metric_name,
        "worst_nrmse": worst.nrmse,
    }
