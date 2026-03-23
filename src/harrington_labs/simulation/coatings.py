"""Coatings Lab simulator.

Models thin-film optical coatings via transfer matrix method,
computing reflectance/transmittance spectra, phase shifts,
GDD, electric field distribution, and angular dependence.
No Streamlit imports.
"""
from __future__ import annotations

import math
import numpy as np

from harrington_common.compute import jit, parallel_map

from harrington_labs.domain import (
    CoatingDesign, ThinFilmLayer, CoatingType, SubstrateType,
    SimulationResult, C_M_S,
)


# ── Substrate refractive indices ─────────────────────────────────

_SUBSTRATE_N = {
    SubstrateType.BK7: 1.52,
    SubstrateType.FUSED_SILICA: 1.46,
    SubstrateType.SAPPHIRE: 1.77,
    SubstrateType.ZNS: 2.35,
    SubstrateType.ZNSE: 2.40,
    SubstrateType.CAF2: 1.43,
    SubstrateType.SILICON: 3.42,
    SubstrateType.GERMANIUM: 4.00,
}


# ── Common coating materials ─────────────────────────────────────

COATING_MATERIALS = {
    "SiO2":   {"n": 1.46, "k": 0.0, "name": "Silicon Dioxide"},
    "MgF2":   {"n": 1.38, "k": 0.0, "name": "Magnesium Fluoride"},
    "Al2O3":  {"n": 1.63, "k": 0.0, "name": "Aluminum Oxide"},
    "TiO2":   {"n": 2.40, "k": 0.0, "name": "Titanium Dioxide"},
    "Ta2O5":  {"n": 2.10, "k": 0.0, "name": "Tantalum Pentoxide"},
    "HfO2":   {"n": 1.95, "k": 0.0, "name": "Hafnium Dioxide"},
    "Nb2O5":  {"n": 2.30, "k": 0.0, "name": "Niobium Pentoxide"},
    "ZrO2":   {"n": 2.15, "k": 0.0, "name": "Zirconium Dioxide"},
    "ZnS":    {"n": 2.35, "k": 0.0, "name": "Zinc Sulfide"},
    "ZnSe":   {"n": 2.50, "k": 0.0, "name": "Zinc Selenide"},
    "Ag":     {"n": 0.13, "k": 3.99, "name": "Silver"},
    "Au":     {"n": 0.18, "k": 3.07, "name": "Gold"},
    "Al":     {"n": 1.37, "k": 7.62, "name": "Aluminum"},
}


# ── Transfer matrix method ───────────────────────────────────────

def _transfer_matrix_single(
    n_complex: complex,
    thickness_nm: float,
    wavelength_nm: float,
    angle_rad: float,
    polarization: str = "s",
) -> np.ndarray:
    """2×2 transfer matrix for a single layer."""
    # Snell's law in the layer
    n0_sin = math.sin(angle_rad)  # incident medium (air)
    cos_theta = np.sqrt(1 - (n0_sin / n_complex) ** 2 + 0j)

    # Phase thickness
    delta = 2 * math.pi * n_complex * cos_theta * thickness_nm / wavelength_nm

    if polarization == "s":
        eta = n_complex * cos_theta
    else:  # p-polarization
        eta = n_complex / cos_theta

    cos_d = np.cos(delta)
    sin_d = np.sin(delta)

    M = np.array([
        [cos_d, 1j * sin_d / eta],
        [1j * eta * sin_d, cos_d],
    ], dtype=complex)
    return M


def transfer_matrix_stack(
    design: CoatingDesign,
    wavelength_nm: float,
    polarization: str = "s",
) -> tuple[float, float, complex]:
    """Compute reflectance and transmittance for full stack at one wavelength.

    Returns (R, T, r_complex).
    """
    angle_rad = math.radians(design.angle_of_incidence_deg)
    n_inc = 1.0  # air
    n_sub = complex(design.substrate_n, 0)

    # Substrate angle
    cos_sub = np.sqrt(1 - (n_inc * math.sin(angle_rad) / n_sub) ** 2 + 0j)

    if polarization == "s":
        eta_inc = n_inc * math.cos(angle_rad)
        eta_sub = n_sub * cos_sub
    else:
        eta_inc = n_inc / math.cos(angle_rad)
        eta_sub = n_sub / cos_sub

    # Build total transfer matrix
    M = np.eye(2, dtype=complex)
    for layer in design.layers:
        n_layer = complex(layer.refractive_index, -layer.extinction_coefficient)
        M_layer = _transfer_matrix_single(
            n_layer, layer.thickness_nm, wavelength_nm,
            angle_rad, polarization,
        )
        M = M @ M_layer

    # Reflection and transmission coefficients
    num = M[0, 0] * eta_sub + M[0, 1] * eta_sub * eta_sub - M[1, 0] - M[1, 1] * eta_sub
    # Correct formula:
    # r = (M[0,0]*eta_sub + M[0,1]*eta_sub² - M[1,0] - M[1,1]*eta_sub) / (...)
    # Actually use standard formulation:
    a = M[0, 0] + M[0, 1] * eta_sub
    b = M[1, 0] + M[1, 1] * eta_sub

    r = (eta_inc * a - b) / (eta_inc * a + b)
    t = 2 * eta_inc / (eta_inc * a + b)

    R = float(abs(r) ** 2)
    T = float((eta_sub.real / eta_inc.real) * abs(t) ** 2) if eta_inc.real > 0 else 0.0
    T = min(T, 1.0 - R)  # energy conservation clamp

    return R, T, r


# ── Spectral sweep ───────────────────────────────────────────────

def spectral_response(
    design: CoatingDesign,
    wavelength_range_nm: tuple[float, float] = (400, 1200),
    n_points: int = 500,
    polarization: str = "avg",
) -> dict:
    """Reflectance and transmittance over a wavelength range."""
    wavelengths = np.linspace(wavelength_range_nm[0], wavelength_range_nm[1], n_points)
    R = np.zeros(n_points)
    T = np.zeros(n_points)
    phase = np.zeros(n_points)

    def _compute_single_wl(lam):
        if polarization == "avg":
            rs, ts, rc_s = transfer_matrix_stack(design, lam, "s")
            rp, tp, rc_p = transfer_matrix_stack(design, lam, "p")
            return ((rs + rp) / 2, (ts + tp) / 2,
                    (np.angle(rc_s) + np.angle(rc_p)) / 2)
        else:
            r, t, rc = transfer_matrix_stack(design, lam, polarization)
            return (r, t, np.angle(rc))

    # Parallel sweep over wavelengths (auto-selects best backend)
    if n_points > 100:
        results = parallel_map(
            _compute_single_wl, list(wavelengths),
            backend="auto", use_processes=False,
        )
    else:
        results = [_compute_single_wl(lam) for lam in wavelengths]

    for i, (r, t, ph) in enumerate(results):
        R[i] = r
        T[i] = t
        phase[i] = ph

    return {
        "wavelength_nm": wavelengths,
        "reflectance": R,
        "transmittance": T,
        "phase_rad": phase,
    }


# ── Angular sweep ────────────────────────────────────────────────

def angular_response(
    design: CoatingDesign,
    wavelength_nm: float | None = None,
    angle_range_deg: tuple[float, float] = (0, 85),
    n_points: int = 180,
) -> dict:
    """Reflectance vs angle of incidence for s and p polarization."""
    if wavelength_nm is None:
        wavelength_nm = design.design_wavelength_nm
    angles = np.linspace(angle_range_deg[0], angle_range_deg[1], n_points)
    Rs = np.zeros(n_points)
    Rp = np.zeros(n_points)

    original_angle = design.angle_of_incidence_deg
    for i, ang in enumerate(angles):
        design.angle_of_incidence_deg = ang
        Rs[i], _, _ = transfer_matrix_stack(design, wavelength_nm, "s")
        Rp[i], _, _ = transfer_matrix_stack(design, wavelength_nm, "p")
    design.angle_of_incidence_deg = original_angle

    return {
        "angle_deg": angles,
        "reflectance_s": Rs,
        "reflectance_p": Rp,
        "reflectance_avg": (Rs + Rp) / 2,
    }


# ── Electric field distribution ──────────────────────────────────

def electric_field_profile(
    design: CoatingDesign,
    wavelength_nm: float | None = None,
    n_points_per_layer: int = 50,
) -> dict:
    """E-field magnitude through the coating stack."""
    if wavelength_nm is None:
        wavelength_nm = design.design_wavelength_nm

    positions = []
    e_field = []
    layer_boundaries = []
    z = 0.0

    # Compute from top (incident side)
    # We need the full reflection coefficient first
    R, T, r_total = transfer_matrix_stack(design, wavelength_nm, "s")

    for layer in design.layers:
        n_c = complex(layer.refractive_index, -layer.extinction_coefficient)
        dz = layer.thickness_nm / n_points_per_layer

        for j in range(n_points_per_layer):
            z_local = j * dz
            # Phase accumulated in this layer
            delta = 2 * math.pi * n_c * z_local / wavelength_nm
            # Standing wave approximation
            e_fwd = math.exp(-layer.extinction_coefficient * 2 * math.pi * z_local / wavelength_nm)
            e_mag = abs(e_fwd * (1 + abs(r_total) * np.exp(-2j * delta)))
            positions.append(z + z_local)
            e_field.append(e_mag)

        layer_boundaries.append(z)
        z += layer.thickness_nm

    layer_boundaries.append(z)

    return {
        "position_nm": np.array(positions),
        "e_field_normalized": np.array(e_field) / max(e_field) if e_field else np.array([]),
        "layer_boundaries_nm": np.array(layer_boundaries),
    }


# ── GDD from phase ──────────────────────────────────────────────

def group_delay_dispersion(
    design: CoatingDesign,
    wavelength_range_nm: tuple[float, float] = (400, 1200),
    n_points: int = 500,
) -> dict:
    """Compute GDD from second derivative of reflection phase."""
    resp = spectral_response(design, wavelength_range_nm, n_points, "s")
    lam = resp["wavelength_nm"]
    phi = np.unwrap(resp["phase_rad"])

    # Convert to frequency domain for differentiation
    omega = 2 * math.pi * C_M_S / (lam * 1e-9)

    # Second derivative of phase w.r.t. omega (GDD)
    d_omega = np.gradient(omega)
    d_phi = np.gradient(phi, omega)
    gdd = np.gradient(d_phi, omega)

    return {
        "wavelength_nm": lam,
        "gdd_fs2": gdd * 1e30,  # s² to fs²
        "group_delay_fs": d_phi * 1e15,
    }


# ── Preset coating designs ──────────────────────────────────────

def quarter_wave_ar(
    design_wavelength_nm: float = 1064.0,
    substrate: SubstrateType = SubstrateType.BK7,
) -> CoatingDesign:
    """Single-layer quarter-wave AR coating (MgF2)."""
    n_sub = _SUBSTRATE_N.get(substrate, 1.52)
    n_film = 1.38  # MgF2
    qwot = design_wavelength_nm / (4 * n_film)
    return CoatingDesign(
        name=f"QWAR @ {design_wavelength_nm:.0f} nm",
        coating_type=CoatingType.AR,
        substrate=substrate,
        substrate_n=n_sub,
        design_wavelength_nm=design_wavelength_nm,
        layers=[ThinFilmLayer("MgF2", qwot, n_film)],
    )


def v_coat_ar(
    design_wavelength_nm: float = 1064.0,
    substrate: SubstrateType = SubstrateType.BK7,
) -> CoatingDesign:
    """V-coat (two-layer) AR at design wavelength."""
    n_sub = _SUBSTRATE_N.get(substrate, 1.52)
    n_h = 2.10  # Ta2O5
    n_l = 1.38  # MgF2
    t_h = design_wavelength_nm / (4 * n_h) * 0.5
    t_l = design_wavelength_nm / (4 * n_l)
    return CoatingDesign(
        name=f"V-Coat AR @ {design_wavelength_nm:.0f} nm",
        coating_type=CoatingType.AR,
        substrate=substrate,
        substrate_n=n_sub,
        design_wavelength_nm=design_wavelength_nm,
        layers=[
            ThinFilmLayer("Ta2O5", t_h, n_h),
            ThinFilmLayer("MgF2", t_l, n_l),
        ],
    )


def quarter_wave_stack_hr(
    design_wavelength_nm: float = 1064.0,
    n_pairs: int = 10,
    substrate: SubstrateType = SubstrateType.BK7,
) -> CoatingDesign:
    """Quarter-wave stack high reflector."""
    n_sub = _SUBSTRATE_N.get(substrate, 1.52)
    n_h = 2.40  # TiO2
    n_l = 1.46  # SiO2
    t_h = design_wavelength_nm / (4 * n_h)
    t_l = design_wavelength_nm / (4 * n_l)

    layers = []
    for _ in range(n_pairs):
        layers.append(ThinFilmLayer("TiO2", t_h, n_h))
        layers.append(ThinFilmLayer("SiO2", t_l, n_l))

    return CoatingDesign(
        name=f"QWS HR @ {design_wavelength_nm:.0f} nm ({n_pairs} pairs)",
        coating_type=CoatingType.HR,
        substrate=substrate,
        substrate_n=n_sub,
        design_wavelength_nm=design_wavelength_nm,
        layers=layers,
    )


def broadband_ar(
    center_wavelength_nm: float = 800.0,
    substrate: SubstrateType = SubstrateType.BK7,
) -> CoatingDesign:
    """Four-layer broadband AR coating."""
    n_sub = _SUBSTRATE_N.get(substrate, 1.52)
    lam = center_wavelength_nm
    layers = [
        ThinFilmLayer("Al2O3", lam / (4 * 1.63) * 0.3, 1.63),
        ThinFilmLayer("TiO2", lam / (4 * 2.40) * 1.1, 2.40),
        ThinFilmLayer("SiO2", lam / (4 * 1.46) * 0.5, 1.46),
        ThinFilmLayer("MgF2", lam / (4 * 1.38) * 1.0, 1.38),
    ]
    return CoatingDesign(
        name=f"BBAR centered @ {center_wavelength_nm:.0f} nm",
        coating_type=CoatingType.BBAR,
        substrate=substrate,
        substrate_n=n_sub,
        design_wavelength_nm=center_wavelength_nm,
        layers=layers,
    )


# ── Full coating simulation bundle ───────────────────────────────

def run_coating_simulation(design: CoatingDesign) -> SimulationResult:
    """Run complete coatings lab simulation."""
    spec = spectral_response(design)
    angular = angular_response(design)
    efield = electric_field_profile(design)
    gdd = group_delay_dispersion(design)

    # Performance at design wavelength
    R_design, T_design, _ = transfer_matrix_stack(design, design.design_wavelength_nm, "avg")

    warnings = []
    total_thickness = sum(l.thickness_nm for l in design.layers)
    if total_thickness > 10000:
        warnings.append(f"Total stack thickness {total_thickness:.0f} nm — stress/delamination risk")
    if any(l.extinction_coefficient > 0.01 for l in design.layers):
        warnings.append("Absorbing layers present — thermal effects may be significant")

    return SimulationResult(
        name="Coatings Lab",
        data={
            "spectral": spec,
            "angular": angular,
            "e_field": efield,
            "gdd": gdd,
            "design_performance": {
                "R_at_design": R_design,
                "T_at_design": T_design,
                "total_thickness_nm": total_thickness,
                "n_layers": len(design.layers),
            },
        },
        warnings=warnings,
    )
