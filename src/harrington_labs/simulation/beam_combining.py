"""Beam combining engines — shared across all source labs.

Spectral beam combining (SBC) and coherent beam combining (CBC)
estimators. Used by Direct Diode, Fiber Laser, Pulsed Laser labs
and the Demonstrator Builder. No Streamlit imports.
"""
from __future__ import annotations

import math

import numpy as np

from harrington_common.compute import jit


# ── Spectral Beam Combining (simple estimate) ──────────────────────────


def spectral_beam_combining(
    n_emitters: int = 10,
    per_emitter_power_w: float = 10.0,
    grating_efficiency: float = 0.92,
    pointing_error_urad: float = 50.0,
    spectral_fill_factor: float = 0.8,
) -> dict:
    """Estimate spectral beam combining performance.

    Quick SBC model used by the individual source labs.
    For the full spectrum-resolved SBC model, see qd_diode_combiner.
    """
    raw_power = n_emitters * per_emitter_power_w
    combining_eff = grating_efficiency * spectral_fill_factor
    pointing_loss = math.exp(-(pointing_error_urad * 1e-6) ** 2 / (1e-3) ** 2)
    combined_power = raw_power * combining_eff * pointing_loss

    return {
        "n_emitters": n_emitters,
        "raw_power_w": raw_power,
        "combined_power_w": combined_power,
        "combining_efficiency": combining_eff * pointing_loss,
        "grating_efficiency": grating_efficiency,
        "pointing_loss": pointing_loss,
    }


# ── Coherent Beam Combining (simple estimate) ──────────────────────────


def coherent_beam_combining(
    n_channels: int = 4,
    per_channel_power_w: float = 10.0,
    phase_error_rms_rad: float = 0.1,
    tip_tilt_error_urad: float = 10.0,
    fill_factor: float = 0.8,
    emitter_pitch_um: float = 250.0,
    emission_nm: float = 1064.0,
) -> dict:
    """Estimate coherent beam combining performance.

    Quick CBC model: Strehl decomposition into phase, tip/tilt,
    and fill-factor contributions. Power is conserved; CBC improves
    brightness, not total power.
    """
    strehl_phase = math.exp(-phase_error_rms_rad ** 2)

    theta_diff = emission_nm * 1e-9 / (emitter_pitch_um * 1e-6)
    strehl_tiptilt = math.exp(-2 * (tip_tilt_error_urad * 1e-6 / theta_diff) ** 2)

    strehl_fill = fill_factor ** 2

    strehl_total = max(strehl_phase * strehl_tiptilt * strehl_fill, 0.01)

    combined_power = n_channels * per_channel_power_w
    power_in_bucket = combined_power * strehl_total
    brightness_gain = n_channels * strehl_total
    m2_combined = min(1.0 / math.sqrt(strehl_total), 50.0)

    return {
        "method": "CBC",
        "n_channels": n_channels,
        "raw_power_w": combined_power,
        "combined_power_w": combined_power,
        "power_in_bucket_w": power_in_bucket,
        "combining_efficiency": strehl_total,
        "strehl_phase": strehl_phase,
        "strehl_tiptilt": strehl_tiptilt,
        "strehl_fill": strehl_fill,
        "strehl_total": strehl_total,
        "brightness_gain": brightness_gain,
        "m2_combined": m2_combined,
    }


# ── JIT-accelerated CBC far-field kernel ───────────────────────────────


@jit
def _cbc_far_field_kernel(
    theta_rad, n_emitters, pitch_m, k, phase_errors
):
    """Compute tiled-aperture array factor — JIT-accelerated."""
    n_ff = len(theta_rad)
    array_factor = np.zeros(n_ff)
    for j in range(n_emitters):
        x_j = (j - (n_emitters - 1) / 2.0) * pitch_m
        for i in range(n_ff):
            array_factor[i] += math.cos(
                k * x_j * math.sin(theta_rad[i]) + phase_errors[j]
            )
    for i in range(n_ff):
        array_factor[i] = (array_factor[i] / n_emitters) ** 2
    return array_factor
