"""Beam propagation and focal analysis with spatial mode support.

Models TEM00 Gaussian, top-hat (super-Gaussian), Hermite-Gaussian,
Laguerre-Gaussian, and Bessel beam propagation through a focusing optic
into a material. Computes spot size, Rayleigh range, and depth-dependent
irradiance for ultrafast pulse interactions.

This version also adds a z-scan-friendly material propagation model that
recomputes local beam radius and peak irradiance through the slab, and
supports optional nonlinear absorption / Kerr diagnostics.

Accelerated via harrington_common.compute:
    CUDA GPU → Numba JIT → NumPy (automatic fallback)
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from harrington_common.compute import (
    jit, parallel_map, to_host,
    _gaussian_w_z, _material_propagation_loop,
)


# ── Small helpers ──────────────────────────────────────────────────────────────

def beam_area_cm2(w_m: np.ndarray | float) -> np.ndarray | float:
    """Beam area in cm^2 from 1/e^2 radius in meters."""
    w = np.asarray(w_m)
    return math.pi * (w * 100.0) ** 2


def beam_radius_at_z(
    focus: "FocusResult",
    z_from_focus_m: np.ndarray | float,
    n_medium: float = 1.0,
) -> np.ndarray | float:
    """Gaussian beam radius at z, using medium-scaled Rayleigh range.

    Uses JIT-accelerated kernel when available.
    """
    z = np.asarray(z_from_focus_m, dtype=np.float64)
    if z.ndim == 0:
        z = z.reshape(1)
        result = _gaussian_w_z(focus.w0_m, focus.rayleigh_range_m, z, n_medium)
        return float(result[0])
    return _gaussian_w_z(focus.w0_m, focus.rayleigh_range_m, z, n_medium)


# ── Transverse profile generators ─────────────────────────────────────────────

@jit
def _gaussian_profile_kernel(r, w0):
    """TEM00 Gaussian — JIT kernel."""
    n = len(r)
    out = np.empty(n)
    for i in range(n):
        out[i] = math.exp(-2.0 * (r[i] / w0) ** 2)
    return out


@jit
def _tophat_profile_kernel(r, w0, order):
    """Super-Gaussian — JIT kernel."""
    n = len(r)
    out = np.empty(n)
    exp = 2 * order
    for i in range(n):
        out[i] = math.exp(-2.0 * (r[i] / w0) ** exp)
    return out


def gaussian_profile(r: np.ndarray, w0: float) -> np.ndarray:
    """TEM00 Gaussian intensity profile I(r) / I_peak."""
    return _gaussian_profile_kernel(np.asarray(r, dtype=np.float64), w0)


def tophat_profile(
    r: np.ndarray,
    w0: float,
    order: int = 10,
) -> np.ndarray:
    """Super-Gaussian (flat-top) profile.

    order=2 → standard Gaussian
    order≥10 → near flat-top
    """
    return _tophat_profile_kernel(np.asarray(r, dtype=np.float64), w0, order)


def hermite_gaussian_profile(
    x: np.ndarray,
    y: np.ndarray,
    w0: float,
    m: int = 0,
    n: int = 0,
) -> np.ndarray:
    """Hermite-Gaussian HG_mn intensity profile on a 2-D grid."""
    from numpy.polynomial.hermite_e import hermeval

    X, Y = np.meshgrid(x, y)
    arg_x = np.sqrt(2.0) * X / w0
    arg_y = np.sqrt(2.0) * Y / w0

    cx = np.zeros(m + 1)
    cx[m] = 1.0
    cy = np.zeros(n + 1)
    cy[n] = 1.0

    Hm = hermeval(arg_x, cx)
    Hn = hermeval(arg_y, cy)

    envelope = np.exp(-(X**2 + Y**2) / w0**2)
    profile = (Hm * Hn) ** 2 * envelope
    peak = profile.max()
    return profile / peak if peak > 0 else profile


def laguerre_gaussian_profile(
    r: np.ndarray,
    phi: np.ndarray,
    w0: float,
    p: int = 0,
    l: int = 0,
) -> np.ndarray:
    """Laguerre-Gaussian LG_pl intensity profile in polar coordinates."""
    from scipy.special import genlaguerre

    rho = (r / w0) ** 2
    L_pl = genlaguerre(p, abs(l))
    amplitude = (
        (np.sqrt(2.0) * r / w0) ** abs(l)
        * L_pl(2.0 * rho)
        * np.exp(-rho)
        * np.cos(l * phi)
    )
    profile = amplitude**2
    peak = profile.max()
    return profile / peak if peak > 0 else profile


def bessel_profile(
    r: np.ndarray,
    half_angle_mrad: float,
    wavelength_m: float,
    order: int = 0,
) -> np.ndarray:
    """Bessel beam J_n intensity profile."""
    from scipy.special import jv

    k_r = 2.0 * math.pi / wavelength_m * math.sin(half_angle_mrad * 1e-3)
    profile = jv(order, k_r * r) ** 2
    peak = profile.max()
    return profile / peak if peak > 0 else profile


# ── 1-D radial profile dispatch ───────────────────────────────────────────────

def radial_intensity_profile(
    r: np.ndarray,
    w0: float,
    spatial_mode: str = "TEM00",
    wavelength_m: float = 1e-6,
    mode_params: dict | None = None,
) -> np.ndarray:
    """Return normalised I(r)/I_peak for the requested spatial mode."""
    mp = mode_params or {}

    if spatial_mode in ("TEM00", "Gaussian"):
        return gaussian_profile(r, w0)

    if spatial_mode == "Top-Hat":
        order = mp.get("tophat_order", 10)
        return tophat_profile(r, w0, order=order)

    if spatial_mode == "Hermite-Gaussian":
        m = mp.get("hg_m", 1)
        n = mp.get("hg_n", 0)
        x = r
        profile_2d = hermite_gaussian_profile(x, np.array([0.0]), w0, m, n)
        return profile_2d.flatten()

    if spatial_mode == "Laguerre-Gaussian":
        p = mp.get("lg_p", 0)
        l_idx = mp.get("lg_l", 1)
        phi = np.zeros_like(r)
        return laguerre_gaussian_profile(r, phi, w0, p, l_idx)

    if spatial_mode == "Bessel":
        ha = mp.get("bessel_half_angle_mrad", 5.0)
        order = mp.get("bessel_order", 0)
        return bessel_profile(r, ha, wavelength_m, order)

    return gaussian_profile(r, w0)


# ── Core dataclasses ──────────────────────────────────────────────────────────

@dataclass
class BeamParams:
    """Input beam parameters before focusing."""
    wavelength_m: float
    beam_diameter_1e2_m: float
    m_squared: float = 1.0
    pulse_energy_j: float = 0.0
    pulse_width_s: float = 0.0
    rep_rate_hz: float = 0.0
    spatial_mode: str = "TEM00"
    mode_params: dict | None = None

    @property
    def beam_radius_m(self) -> float:
        return self.beam_diameter_1e2_m / 2.0

    @property
    def peak_power_w(self) -> float:
        if self.pulse_width_s > 0:
            return self.pulse_energy_j / self.pulse_width_s
        return 0.0

    @property
    def avg_power_w(self) -> float:
        return self.pulse_energy_j * self.rep_rate_hz


@dataclass
class FocusResult:
    """Results of focusing a beam."""
    w0_m: float
    rayleigh_range_m: float
    depth_of_focus_m: float
    f_number: float
    na: float
    peak_irradiance_w_cm2: float
    fluence_j_cm2: float
    z_m: np.ndarray
    w_z_m: np.ndarray
    irradiance_z_w_cm2: np.ndarray
    r_m: np.ndarray | None = None
    intensity_r: np.ndarray | None = None


@dataclass
class PropagationInMaterial:
    """Beam propagation results inside a material slab."""
    z_material_m: np.ndarray
    z_from_focus_m: np.ndarray
    w_z_m: np.ndarray
    irradiance_z_w_cm2: np.ndarray
    fluence_z_j_cm2: np.ndarray
    absorbed_fraction: np.ndarray
    transmission_fraction: float
    peak_irradiance_w_cm2: float
    b_integral_rad: float = 0.0
    self_focusing_ratio: float = 0.0


@dataclass
class ZScanResult:
    """Open-aperture z-scan results using sample-center coordinate."""
    z_positions_m: np.ndarray
    peak_irradiance_w_cm2: np.ndarray
    transmission_fraction: np.ndarray
    peak_radius_m: np.ndarray
    b_integral_rad: np.ndarray
    self_focusing_ratio: np.ndarray


# ── Focus model ───────────────────────────────────────────────────

def compute_focus(
    beam: BeamParams,
    focal_length_m: float,
    n_medium: float = 1.0,
) -> FocusResult:
    """Compute focused beam parameters."""
    lam = beam.wavelength_m
    w_in = beam.beam_radius_m
    m2 = beam.m_squared

    # Focused spot radius (1/e^2)
    w0 = (m2 * lam * focal_length_m) / (math.pi * w_in * n_medium)

    # Rayleigh range in the medium
    z_r = (math.pi * w0**2 * n_medium) / (m2 * lam)

    f_num = focal_length_m / (2.0 * w_in)
    na = n_medium * math.sin(math.atan(w_in / focal_length_m))

    area_cm2 = math.pi * (w0 * 100.0) ** 2
    peak_irr = beam.peak_power_w / area_cm2 if area_cm2 > 0 else 0.0
    fluence = beam.pulse_energy_j / area_cm2 if area_cm2 > 0 else 0.0

    z = np.linspace(-5.0 * z_r, 5.0 * z_r, 500, dtype=np.float64)
    w_z = _gaussian_w_z(w0, z_r, z, 1.0)
    area_z_cm2 = math.pi * (w_z * 100.0) ** 2
    irr_z = np.where(area_z_cm2 > 0, beam.peak_power_w / area_z_cm2, 0.0)

    r = np.linspace(0.0, 3.0 * w0, 200)
    intensity_r = radial_intensity_profile(
        r,
        w0,
        spatial_mode=beam.spatial_mode,
        wavelength_m=lam,
        mode_params=beam.mode_params,
    )

    return FocusResult(
        w0_m=w0,
        rayleigh_range_m=z_r,
        depth_of_focus_m=2.0 * z_r,
        f_number=f_num,
        na=na,
        peak_irradiance_w_cm2=peak_irr,
        fluence_j_cm2=fluence,
        z_m=z,
        w_z_m=w_z,
        irradiance_z_w_cm2=irr_z,
        r_m=r,
        intensity_r=intensity_r,
    )


# ── Material propagation / z-scan core ───────────────────────────────────────

def propagate_in_material(
    focus: FocusResult,
    beam: BeamParams,
    n_material: float,
    alpha_cm: float,
    thickness_m: float,
    surface_position_m: float = 0.0,
    beta_cm_per_w: float = 0.0,
    n2_cm2_per_w: float = 0.0,
) -> PropagationInMaterial:
    """Propagate a focused beam through a material slab.

    Uses JIT-accelerated kernels for the inner propagation loop.
    """
    n_points = 200
    z_mat = np.linspace(0.0, thickness_m, n_points, dtype=np.float64)
    z_from_focus = surface_position_m + z_mat

    w_z = _gaussian_w_z(focus.w0_m, focus.rayleigh_range_m, z_from_focus, n_material)
    area_cm2 = beam_area_cm2(w_z)

    # JIT-accelerated propagation loop
    irradiance_z, fluence_z, final_power, final_energy = _material_propagation_loop(
        n_points, z_mat,
        np.asarray(area_cm2, dtype=np.float64),
        beam.peak_power_w, beam.pulse_energy_j,
        alpha_cm, beta_cm_per_w,
    )

    transmission_fraction = (
        final_power / beam.peak_power_w
        if beam.peak_power_w > 0
        else 1.0
    )
    peak_irradiance = float(np.max(irradiance_z)) if irradiance_z.size else 0.0

    # Absorbed fraction (relative to initial)
    absorbed = np.zeros_like(z_mat)
    if beam.peak_power_w > 0:
        # Approximate from irradiance decay
        area_0 = float(area_cm2[0]) if np.ndim(area_cm2) > 0 else float(area_cm2)
        if area_0 > 0:
            power_z = irradiance_z * area_cm2
            absorbed = 1.0 - power_z / (irradiance_z[0] * area_0) if irradiance_z[0] > 0 else absorbed

    b_integral_rad = 0.0
    self_focusing_ratio = 0.0
    if n2_cm2_per_w > 0:
        n2_m2_per_w = n2_cm2_per_w * 1e-4
        i_w_m2 = irradiance_z * 1e4
        b_integral_rad = float(
            (2.0 * math.pi / beam.wavelength_m)
            * np.trapezoid(n2_m2_per_w * i_w_m2, z_mat)
        )
        p_crit_w = 0.148 * beam.wavelength_m**2 / (n_material * n2_m2_per_w)
        self_focusing_ratio = beam.peak_power_w / p_crit_w if p_crit_w > 0 else 0.0

    return PropagationInMaterial(
        z_material_m=z_mat,
        z_from_focus_m=z_from_focus,
        w_z_m=w_z,
        irradiance_z_w_cm2=irradiance_z,
        fluence_z_j_cm2=fluence_z,
        absorbed_fraction=absorbed,
        transmission_fraction=transmission_fraction,
        peak_irradiance_w_cm2=peak_irradiance,
        b_integral_rad=b_integral_rad,
        self_focusing_ratio=self_focusing_ratio,
    )


def _propagate_single_position(args):
    """Worker for parallel z-scan sweep."""
    (z_center_m, focus, beam, n_material, alpha_cm,
     thickness_m, beta_cm_per_w, n2_cm2_per_w) = args

    surface_pos_m = z_center_m - 0.5 * thickness_m
    prop = propagate_in_material(
        focus=focus, beam=beam,
        n_material=n_material, alpha_cm=alpha_cm,
        thickness_m=thickness_m, surface_position_m=surface_pos_m,
        beta_cm_per_w=beta_cm_per_w, n2_cm2_per_w=n2_cm2_per_w,
    )
    peak_idx = int(np.argmax(prop.irradiance_z_w_cm2))
    return {
        "peak_irr": prop.peak_irradiance_w_cm2,
        "transmission": prop.transmission_fraction,
        "b_integral": prop.b_integral_rad,
        "sf_ratio": prop.self_focusing_ratio,
        "peak_radius": prop.w_z_m[peak_idx],
    }


def simulate_open_aperture_zscan(
    focus: FocusResult,
    beam: BeamParams,
    n_material: float,
    alpha_cm: float,
    thickness_m: float,
    z_positions_m: np.ndarray,
    beta_cm_per_w: float = 0.0,
    n2_cm2_per_w: float = 0.0,
    parallel: bool = True,
) -> ZScanResult:
    """Simulate open-aperture z-scan.

    When parallel=True and len(z_positions) > 50, uses parallel_map
    for multi-core acceleration of the parameter sweep.
    """
    z_positions_m = np.asarray(z_positions_m, dtype=np.float64)
    n_pos = len(z_positions_m)

    # Build argument tuples
    args_list = [
        (float(z), focus, beam, n_material, alpha_cm,
         thickness_m, beta_cm_per_w, n2_cm2_per_w)
        for z in z_positions_m
    ]

    # Use parallel sweep for large scans
    if parallel and n_pos > 50:
        results = parallel_map(
            _propagate_single_position, args_list,
            backend="auto",
        )
    else:
        results = [_propagate_single_position(a) for a in args_list]

    peak_irr = np.array([r["peak_irr"] for r in results])
    transmission = np.array([r["transmission"] for r in results])
    b_integral = np.array([r["b_integral"] for r in results])
    sf_ratio = np.array([r["sf_ratio"] for r in results])
    peak_radius_m = np.array([r["peak_radius"] for r in results])

    return ZScanResult(
        z_positions_m=z_positions_m,
        peak_irradiance_w_cm2=peak_irr,
        transmission_fraction=transmission,
        peak_radius_m=peak_radius_m,
        b_integral_rad=b_integral,
        self_focusing_ratio=sf_ratio,
    )
