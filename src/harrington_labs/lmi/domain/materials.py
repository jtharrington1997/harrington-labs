"""Material definitions with optical and thermal properties.

Provides default common materials and supports user-defined additions.
Includes mid-IR and ultrafast-specific properties for fs-pulse modeling.
Supports wavelength-dependent refractive index via Sellmeier equations
for key optical materials (Si, Ge, GaAs, SiO2, BK7, LiNbO3, KTP,
Sapphire, ZnSe, CaF2).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field, asdict
import json
from pathlib import Path

MATERIAL_DB_PATH = Path("data/manual/materials.json")


# ── Sellmeier dispersion models ───────────────────────────────────

def _sellmeier(wavelength_um: float, coeffs: list[tuple[float, float]]) -> float:
    """Evaluate standard Sellmeier equation.

    n²(λ) = 1 + Σ  B_i * λ² / (λ² - C_i)

    Parameters
    ----------
    wavelength_um : Wavelength in micrometers.
    coeffs : List of (B_i, C_i) Sellmeier coefficient pairs.
             C_i in μm² (i.e. already squared wavelengths).

    Returns
    -------
    Refractive index n.
    """
    lam2 = wavelength_um ** 2
    n_sq = 1.0
    for b, c in coeffs:
        n_sq += b * lam2 / (lam2 - c)
    return math.sqrt(max(n_sq, 1.0))


# Sellmeier coefficients: (B_i, C_i) where C_i is in μm²
SELLMEIER_COEFFICIENTS: dict[str, list[tuple[float, float]]] = {
    "Silicon (Si)": [
        (10.6684293, 0.301516485**2),
        (0.0030434748, 1.13475115**2),
        (1.54133408, 1104.0**2),
    ],
    "Germanium (Ge)": [
        (9.28156, 0.44105**2),
        (6.72880, 3870.1**2),
    ],
    "GaAs": [
        (4.372514, 0.4431307**2),
        (2.272507, 0.8746453**2),
        (3.228956, 36.9166**2),
    ],
    "Fused Silica (SiO2)": [
        (0.6961663, 0.0684043**2),
        (0.4079426, 0.1162414**2),
        (0.8974794, 9.896161**2),
    ],
    "BK7 Glass": [
        (1.03961212, 0.00600069867),
        (0.231792344, 0.0200179144),
        (1.01046945, 103.560653),
    ],
    "LiNbO3 (Lithium Niobate)": [
        (2.6734, 0.01764),
        (1.2290, 0.05914),
        (12.614, 474.60),
    ],
    "KTP": [
        (2.12725, 0.05148),
        (1.18431, 0.06603),
        (0.6603, 100.00),
    ],
    "Sapphire (Al2O3)": [
        # Malitson & Dodge 1972, ordinary ray, valid 0.2-5.5 μm
        (1.4313493, 0.0726631**2),
        (0.6505455, 0.1193242**2),
        (5.3414021, 18.028251**2),
    ],
    "ZnSe": [
        # Connolly 1979, valid 0.55-18 μm
        (4.2980149, 0.19824**2),
        (0.62776557, 0.37878**2),
        (2.8955633, 46.994**2),
    ],
    "CaF2": [
        # Daimon & Masumura 2002, valid 0.14-23 μm
        (0.5675888, 0.050263605**2),
        (0.4710914, 0.1003909**2),
        (3.8484723, 34.649040**2),
    ],
}

SELLMEIER_VALID_RANGE: dict[str, tuple[float, float]] = {
    "Silicon (Si)": (1.2, 14.0),
    "Germanium (Ge)": (2.0, 14.0),
    "GaAs": (0.97, 17.0),
    "Fused Silica (SiO2)": (0.21, 3.71),
    "BK7 Glass": (0.3, 2.5),
    "LiNbO3 (Lithium Niobate)": (0.4, 5.0),
    "KTP": (0.4, 3.5),
    "Sapphire (Al2O3)": (0.2, 5.5),
    "ZnSe": (0.55, 18.0),
    "CaF2": (0.14, 23.0),
}


def sellmeier_n(material_name: str, wavelength_nm: float) -> float | None:
    """Compute refractive index via Sellmeier if model is available."""
    coeffs = SELLMEIER_COEFFICIENTS.get(material_name)
    if coeffs is None:
        return None
    wavelength_um = wavelength_nm / 1000.0
    valid = SELLMEIER_VALID_RANGE.get(material_name)
    if valid and not (valid[0] <= wavelength_um <= valid[1]):
        return None
    return _sellmeier(wavelength_um, coeffs)


def scale_n2_miller(
    n2_ref: float, n_ref: float, n_at_lambda: float,
) -> float:
    """Scale n2 using Miller's rule approximation."""
    if n_ref <= 0:
        return n2_ref
    return n2_ref * (n_at_lambda / n_ref) ** 4


@dataclass
class Material:
    """A material with optical, thermal, and mechanical properties."""

    name: str
    category: str  # metal, semiconductor, dielectric, polymer, biological

    # Optical (reference wavelength)
    refractive_index: float = 1.0
    ref_wavelength_nm: float = 1064.0
    absorption_coeff_cm: float = 0.0
    bandgap_ev: float = 0.0
    nonlinear_index_cm2_w: float = 0.0
    damage_threshold_j_cm2: float = 0.0

    # Mid-IR optical (8.5 um) — kept for backward compatibility
    refractive_index_8500nm: float = 0.0
    absorption_coeff_8500nm_cm: float = 0.0
    nonlinear_index_8500nm_cm2_w: float = 0.0

    # Nonlinear absorption
    two_photon_abs_cm_w: float = 0.0
    three_photon_abs_cm3_w2: float = 0.0

    # Thermal
    thermal_conductivity_w_mk: float = 0.0
    melting_point_k: float = 0.0
    boiling_point_k: float = 0.0
    specific_heat_j_kgk: float = 0.0
    density_kg_m3: float = 0.0
    thermal_diffusivity_m2_s: float = 0.0

    # Ultrafast / two-temperature model
    electron_phonon_coupling_w_m3k: float = 0.0

    notes: str = ""

    @property
    def skin_depth_cm(self) -> float:
        if self.absorption_coeff_cm > 0:
            return 1.0 / self.absorption_coeff_cm
        return float("inf")

    @property
    def has_sellmeier(self) -> bool:
        return self.name in SELLMEIER_COEFFICIENTS

    def get_n(self, wavelength_nm: float) -> float:
        n_sell = sellmeier_n(self.name, wavelength_nm)
        if n_sell is not None:
            return n_sell
        if 7500 < wavelength_nm < 9500 and self.refractive_index_8500nm > 0:
            return self.refractive_index_8500nm
        return self.refractive_index

    def get_alpha(self, wavelength_nm: float) -> float:
        if self.bandgap_ev > 0 and wavelength_nm > 0:
            photon_ev = 1240.0 / wavelength_nm
            if photon_ev < self.bandgap_ev and self.category == "semiconductor":
                if 7500 < wavelength_nm < 9500 and self.absorption_coeff_8500nm_cm > 0:
                    return self.absorption_coeff_8500nm_cm
                return max(self.absorption_coeff_cm, 1e-4)
        if 7500 < wavelength_nm < 9500 and self.absorption_coeff_8500nm_cm > 0:
            return self.absorption_coeff_8500nm_cm
        return self.absorption_coeff_cm

    def get_n2(self, wavelength_nm: float) -> float:
        if self.nonlinear_index_cm2_w > 0 and self.has_sellmeier:
            n_ref = sellmeier_n(self.name, self.ref_wavelength_nm)
            n_target = sellmeier_n(self.name, wavelength_nm)
            if n_ref is not None and n_target is not None:
                return scale_n2_miller(
                    self.nonlinear_index_cm2_w, n_ref, n_target,
                )
        if 7500 < wavelength_nm < 9500 and self.nonlinear_index_8500nm_cm2_w > 0:
            return self.nonlinear_index_8500nm_cm2_w
        return self.nonlinear_index_cm2_w

    def get_thermal_diffusivity(self) -> float:
        if self.thermal_diffusivity_m2_s > 0:
            return self.thermal_diffusivity_m2_s
        if (
            self.thermal_conductivity_w_mk > 0
            and self.density_kg_m3 > 0
            and self.specific_heat_j_kgk > 0
        ):
            return self.thermal_conductivity_w_mk / (
                self.density_kg_m3 * self.specific_heat_j_kgk
            )
        return 0.0

    def dispersion_info(self, wavelength_nm: float) -> dict:
        wavelength_um = wavelength_nm / 1000.0
        info: dict = {
            "wavelength_nm": wavelength_nm,
            "wavelength_um": wavelength_um,
            "has_sellmeier": self.has_sellmeier,
            "sellmeier_used": False,
            "n": self.get_n(wavelength_nm),
            "alpha_cm": self.get_alpha(wavelength_nm),
            "n2_cm2_w": self.get_n2(wavelength_nm),
        }
        if self.has_sellmeier:
            valid = SELLMEIER_VALID_RANGE.get(self.name, (0, 0))
            info["sellmeier_valid_range_um"] = valid
            info["in_valid_range"] = valid[0] <= wavelength_um <= valid[1]
            n_sell = sellmeier_n(self.name, wavelength_nm)
            info["sellmeier_used"] = n_sell is not None
            if n_sell is not None:
                info["n_sellmeier"] = n_sell
        return info


# ── Default materials ──────────────────────────────────────────────

DEFAULT_MATERIALS: list[Material] = [
    # ═══ Metals ═══
    Material(
        name="Aluminum (Al)", category="metal",
        refractive_index=1.37, absorption_coeff_cm=1.11e6,
        thermal_conductivity_w_mk=237, melting_point_k=933, boiling_point_k=2743,
        specific_heat_j_kgk=897, density_kg_m3=2700,
        damage_threshold_j_cm2=5.0,
        notes="High reflectivity in UV-vis. Common for mirrors.",
    ),
    Material(
        name="Copper (Cu)", category="metal",
        refractive_index=0.25, absorption_coeff_cm=8.2e5,
        thermal_conductivity_w_mk=401, melting_point_k=1358, boiling_point_k=2835,
        specific_heat_j_kgk=385, density_kg_m3=8960,
        damage_threshold_j_cm2=8.0,
        notes="Excellent thermal conductor. Used in heatsinks.",
    ),
    Material(
        name="Stainless Steel 304", category="metal",
        refractive_index=2.76, absorption_coeff_cm=5.4e5,
        thermal_conductivity_w_mk=16.2, melting_point_k=1673, boiling_point_k=3273,
        specific_heat_j_kgk=500, density_kg_m3=8000,
        damage_threshold_j_cm2=10.0,
        notes="Low thermal conductivity for a metal.",
    ),
    Material(
        name="Gold (Au)", category="metal",
        refractive_index=0.18, ref_wavelength_nm=633,
        absorption_coeff_cm=8.2e5,
        thermal_conductivity_w_mk=317, melting_point_k=1337, boiling_point_k=3129,
        specific_heat_j_kgk=129, density_kg_m3=19300,
        damage_threshold_j_cm2=4.0,
        notes="High IR reflectivity. Common for MIR mirrors and OAPs.",
    ),
    Material(
        name="Titanium (Ti)", category="metal",
        refractive_index=2.16, absorption_coeff_cm=4.8e5,
        thermal_conductivity_w_mk=21.9, melting_point_k=1941, boiling_point_k=3560,
        specific_heat_j_kgk=520, density_kg_m3=4506,
        damage_threshold_j_cm2=6.0,
        notes="Biocompatible. Used in medical implants and aerospace.",
    ),

    # ═══ Semiconductors ═══
    Material(
        name="Silicon (Si)", category="semiconductor",
        refractive_index=3.48, ref_wavelength_nm=1550,
        absorption_coeff_cm=0.001, bandgap_ev=1.12,
        nonlinear_index_cm2_w=4.5e-14,
        thermal_conductivity_w_mk=148, melting_point_k=1687,
        specific_heat_j_kgk=710, density_kg_m3=2329,
        damage_threshold_j_cm2=2.0,
        refractive_index_8500nm=3.42, absorption_coeff_8500nm_cm=0.001,
        nonlinear_index_8500nm_cm2_w=4.0e-14,
        thermal_diffusivity_m2_s=8.8e-5,
        electron_phonon_coupling_w_m3k=1.0e17,
        notes="Transparent above 1.1 um. Indirect gap 1.12 eV. Sellmeier valid 1.2-14 um.",
    ),
    Material(
        name="Germanium (Ge)", category="semiconductor",
        refractive_index=4.0, ref_wavelength_nm=10600,
        absorption_coeff_cm=0.02, bandgap_ev=0.67,
        nonlinear_index_cm2_w=2.0e-13,
        thermal_conductivity_w_mk=60, melting_point_k=1211,
        specific_heat_j_kgk=320, density_kg_m3=5323,
        damage_threshold_j_cm2=1.0,
        refractive_index_8500nm=4.0, absorption_coeff_8500nm_cm=0.02,
        nonlinear_index_8500nm_cm2_w=2.0e-13,
        thermal_diffusivity_m2_s=3.6e-5,
        electron_phonon_coupling_w_m3k=1.0e16,
        notes="Transparent 2-14 um. High n2. 0.67 eV gap. Sellmeier valid 2-14 um.",
    ),
    Material(
        name="GaAs", category="semiconductor",
        refractive_index=3.38, ref_wavelength_nm=1064,
        absorption_coeff_cm=0.5, bandgap_ev=1.42,
        nonlinear_index_cm2_w=1.6e-13,
        thermal_conductivity_w_mk=55, melting_point_k=1511,
        specific_heat_j_kgk=330, density_kg_m3=5317,
        damage_threshold_j_cm2=1.5,
        notes="Direct bandgap. Strong nonlinear response. Sellmeier valid 0.97-17 um.",
    ),
    Material(
        name="InP", category="semiconductor",
        refractive_index=3.17, ref_wavelength_nm=1550,
        absorption_coeff_cm=0.3, bandgap_ev=1.35,
        nonlinear_index_cm2_w=1.0e-13,
        thermal_conductivity_w_mk=68, melting_point_k=1335,
        specific_heat_j_kgk=310, density_kg_m3=4810,
        damage_threshold_j_cm2=1.2,
        notes="Direct bandgap. Telecom substrate material.",
    ),
    Material(
        name="ZnSe", category="semiconductor",
        refractive_index=2.44, ref_wavelength_nm=10600,
        absorption_coeff_cm=5e-4, bandgap_ev=2.7,
        nonlinear_index_cm2_w=1.2e-13,
        thermal_conductivity_w_mk=18, melting_point_k=1798,
        specific_heat_j_kgk=339, density_kg_m3=5266,
        damage_threshold_j_cm2=5.0,
        notes="CO2 laser window/lens material. Transparent 0.5-22 um. Sellmeier valid 0.55-18 um.",
    ),

    # ═══ Dielectrics / Crystals ═══
    Material(
        name="Fused Silica (SiO2)", category="dielectric",
        refractive_index=1.45, ref_wavelength_nm=1064,
        absorption_coeff_cm=1e-5, bandgap_ev=9.0,
        nonlinear_index_cm2_w=2.7e-16,
        thermal_conductivity_w_mk=1.38, melting_point_k=1983,
        specific_heat_j_kgk=740, density_kg_m3=2200,
        damage_threshold_j_cm2=40.0,
        notes="Workhorse optical material. Very high LIDT. Sellmeier valid 0.21-3.71 um.",
    ),
    Material(
        name="BK7 Glass", category="dielectric",
        refractive_index=1.507, ref_wavelength_nm=1064,
        absorption_coeff_cm=5e-4, bandgap_ev=7.5,
        nonlinear_index_cm2_w=3.2e-16,
        thermal_conductivity_w_mk=1.114, melting_point_k=830,
        specific_heat_j_kgk=858, density_kg_m3=2510,
        damage_threshold_j_cm2=25.0,
        notes="Common optical glass. Sellmeier valid 0.3-2.5 um.",
    ),
    Material(
        name="Sapphire (Al2O3)", category="dielectric",
        refractive_index=1.755, ref_wavelength_nm=1064,
        absorption_coeff_cm=1e-4, bandgap_ev=9.9,
        nonlinear_index_cm2_w=3.1e-16,
        thermal_conductivity_w_mk=46, melting_point_k=2323, boiling_point_k=3253,
        specific_heat_j_kgk=761, density_kg_m3=3980,
        damage_threshold_j_cm2=50.0,
        notes="Extremely hard. High LIDT. Sellmeier valid 0.2-5.5 um. Ti:Sapphire host crystal.",
    ),
    Material(
        name="CaF2 (Calcium Fluoride)", category="dielectric",
        refractive_index=1.43, ref_wavelength_nm=1064,
        absorption_coeff_cm=1e-4, bandgap_ev=12.1,
        nonlinear_index_cm2_w=1.9e-16,
        thermal_conductivity_w_mk=9.7, melting_point_k=1691,
        specific_heat_j_kgk=854, density_kg_m3=3180,
        damage_threshold_j_cm2=30.0,
        notes="UV-MIR window (0.13-10 um). Low dispersion. Sellmeier valid 0.14-23 um.",
    ),
    Material(
        name="LiNbO3 (Lithium Niobate)", category="dielectric",
        refractive_index=2.21, ref_wavelength_nm=1064,
        absorption_coeff_cm=0.002, bandgap_ev=3.78,
        nonlinear_index_cm2_w=5.3e-15,
        thermal_conductivity_w_mk=4.6, melting_point_k=1530,
        specific_heat_j_kgk=628, density_kg_m3=4640,
        damage_threshold_j_cm2=5.0,
        notes="chi(2) nonlinear crystal. PPLN for OPO/SHG. Sellmeier valid 0.4-5.0 um.",
    ),
    Material(
        name="KTP", category="dielectric",
        refractive_index=1.74, ref_wavelength_nm=1064,
        absorption_coeff_cm=0.001, bandgap_ev=3.54,
        nonlinear_index_cm2_w=2e-15,
        thermal_conductivity_w_mk=3.3, melting_point_k=1423,
        specific_heat_j_kgk=728, density_kg_m3=3025,
        damage_threshold_j_cm2=15.0,
        notes="Frequency doubling crystal for Nd:YAG -> 532 nm. Sellmeier valid 0.4-3.5 um.",
    ),
    Material(
        name="MgF2", category="dielectric",
        refractive_index=1.38, ref_wavelength_nm=500,
        absorption_coeff_cm=1e-5, bandgap_ev=11.8,
        thermal_conductivity_w_mk=21, melting_point_k=1536,
        specific_heat_j_kgk=1003, density_kg_m3=3148,
        damage_threshold_j_cm2=35.0,
        notes="AR coating material. VUV window (0.11-7 um).",
    ),
    Material(
        name="Diamond (C)", category="dielectric",
        refractive_index=2.417, ref_wavelength_nm=589,
        absorption_coeff_cm=0.1, bandgap_ev=5.47,
        nonlinear_index_cm2_w=1.3e-15,
        thermal_conductivity_w_mk=2200, melting_point_k=3820,
        specific_heat_j_kgk=509, density_kg_m3=3515,
        damage_threshold_j_cm2=100.0,
        notes="Highest thermal conductivity. Raman gain medium. Extreme LIDT.",
    ),
    Material(
        name="BaF2", category="dielectric",
        refractive_index=1.47, ref_wavelength_nm=1064,
        absorption_coeff_cm=1e-4, bandgap_ev=9.1,
        thermal_conductivity_w_mk=11.7, melting_point_k=1641,
        specific_heat_j_kgk=410, density_kg_m3=4890,
        damage_threshold_j_cm2=20.0,
        notes="MIR window material. Transparent 0.15-12 um.",
    ),

    # ═══ Polymers ═══
    Material(
        name="PMMA (Acrylic)", category="polymer",
        refractive_index=1.49, absorption_coeff_cm=0.1,
        thermal_conductivity_w_mk=0.19, melting_point_k=433,
        specific_heat_j_kgk=1466, density_kg_m3=1180,
        damage_threshold_j_cm2=0.5,
        notes="Common polymer for optics prototyping.",
    ),
    Material(
        name="Polycarbonate (PC)", category="polymer",
        refractive_index=1.585, absorption_coeff_cm=0.5,
        thermal_conductivity_w_mk=0.20, melting_point_k=500,
        specific_heat_j_kgk=1200, density_kg_m3=1200,
        damage_threshold_j_cm2=0.3,
        notes="High impact resistance. Used in safety optics.",
    ),
    Material(
        name="PDMS (Silicone)", category="polymer",
        refractive_index=1.41, absorption_coeff_cm=1.0,
        thermal_conductivity_w_mk=0.15, melting_point_k=233,
        specific_heat_j_kgk=1460, density_kg_m3=970,
        notes="Flexible polymer. Common in microfluidics and bio-optics.",
    ),

    # ═══ Biological ═══
    Material(
        name="Water", category="biological",
        refractive_index=1.33, absorption_coeff_cm=0.01,
        thermal_conductivity_w_mk=0.606,
        melting_point_k=273, boiling_point_k=373,
        specific_heat_j_kgk=4186, density_kg_m3=1000,
        notes="Reference for biological tissue absorption.",
    ),
    Material(
        name="Skin Tissue (Dermis)", category="biological",
        refractive_index=1.40, ref_wavelength_nm=633,
        absorption_coeff_cm=2.0,
        thermal_conductivity_w_mk=0.37,
        specific_heat_j_kgk=3391, density_kg_m3=1090,
        damage_threshold_j_cm2=0.1,
        notes="Typical dermis values at visible wavelengths. Highly scattering.",
    ),
    Material(
        name="Cornea", category="biological",
        refractive_index=1.376, ref_wavelength_nm=589,
        absorption_coeff_cm=120.0,  # strong at 2.94 um (Er:YAG)
        thermal_conductivity_w_mk=0.58,
        specific_heat_j_kgk=4178, density_kg_m3=1062,
        damage_threshold_j_cm2=0.05,
        notes="~75% water content. Strong absorption at 2.94 um (Er:YAG). Key for refractive surgery.",
    ),
    Material(
        name="Bone (Cortical)", category="biological",
        refractive_index=1.55, ref_wavelength_nm=633,
        absorption_coeff_cm=5.0,
        thermal_conductivity_w_mk=0.32, melting_point_k=1670,
        specific_heat_j_kgk=1313, density_kg_m3=1900,
        damage_threshold_j_cm2=0.5,
        notes="Hydroxyapatite matrix. Used in laser osteotomy research.",
    ),
    Material(
        name="Dental Enamel", category="biological",
        refractive_index=1.63, ref_wavelength_nm=633,
        absorption_coeff_cm=800.0,  # at 2.94 um
        thermal_conductivity_w_mk=0.93, melting_point_k=1570,
        specific_heat_j_kgk=750, density_kg_m3=2900,
        damage_threshold_j_cm2=0.3,
        notes="96% hydroxyapatite. Strong Er:YAG absorption. Used in laser dentistry.",
    ),
]


def load_custom_materials() -> list[Material]:
    if not MATERIAL_DB_PATH.exists():
        return []
    data = json.loads(MATERIAL_DB_PATH.read_text())
    return [Material(**d) for d in data.get("custom_materials", [])]


def save_custom_materials(materials: list[Material]) -> None:
    MATERIAL_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    existing: dict = {}
    if MATERIAL_DB_PATH.exists():
        existing = json.loads(MATERIAL_DB_PATH.read_text())
    existing["custom_materials"] = [asdict(m) for m in materials]
    MATERIAL_DB_PATH.write_text(json.dumps(existing, indent=2))


def all_materials() -> list[Material]:
    return DEFAULT_MATERIALS + load_custom_materials()
