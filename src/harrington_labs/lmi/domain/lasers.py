"""Laser source definitions and database.

Provides default commercial laser configurations and a builder
for custom sources (gain media, seed light, etc.).  Supports
arbitrary spatial beam modes and OPA parent-child relationships.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from enum import Enum
import json
import math
from pathlib import Path

LASER_DB_PATH = Path("data/manual/lasers.json")


# ── Spatial beam modes ────────────────────────────────────────────

class SpatialMode(str, Enum):
    """Supported transverse spatial beam profiles."""

    TEM00 = "TEM00"                   # Fundamental Gaussian
    TOP_HAT = "Top-Hat"               # Flat-top / super-Gaussian
    HERMITE_GAUSSIAN = "Hermite-Gaussian"  # HG_mn higher-order
    LAGUERRE_GAUSSIAN = "Laguerre-Gaussian"  # LG_pl (includes vortex)
    BESSEL = "Bessel"                 # Non-diffracting Bessel beam

    @property
    def default_m_squared(self) -> float:
        """Characteristic M² for typical use of each mode family."""
        return {
            self.TEM00: 1.0,
            self.TOP_HAT: 1.0,  # varies with fill factor
            self.HERMITE_GAUSSIAN: 2.0,
            self.LAGUERRE_GAUSSIAN: 2.0,
            self.BESSEL: 1.0,  # core ring; effective M² depends on aperture
        }[self]


@dataclass
class BeamModeParams:
    """Extra parameters that qualify the chosen spatial mode.

    Only the fields relevant to the selected mode need to be non-zero.
    """

    hg_m: int = 0           # Hermite-Gaussian x-order
    hg_n: int = 0           # Hermite-Gaussian y-order
    lg_p: int = 0           # Laguerre-Gaussian radial index
    lg_l: int = 0           # Laguerre-Gaussian azimuthal index (OAM)
    bessel_order: int = 0   # Bessel beam order (J_n)
    bessel_half_angle_mrad: float = 0.0  # Axicon half-angle
    tophat_order: int = 10  # Super-Gaussian order (higher = flatter)


@dataclass
class LaserSource:
    """A light source with full beam parameters.

    Covers lasers, calibration lamps, LEDs, and broadband sources.
    """

    name: str
    wavelength_nm: float
    power_w: float
    rep_rate_hz: float = 0.0          # 0 = CW
    pulse_width_s: float = 0.0        # 0 = CW
    beam_diameter_mm: float = 1.0
    m_squared: float = 1.0
    polarization: str = "linear"       # linear, circular, unpolarized
    spatial_mode: str = "TEM00"
    beam_mode_params: dict = field(default_factory=dict)
    gain_medium: str = ""
    pump_source: str = ""              # parent laser name (e.g. for OPA)
    tunable_range_nm: tuple[float, float] | None = None
    notes: str = ""
    # ── New fields (backward-compatible defaults) ──
    category: str = "laser"            # laser, lamp, led, broadband, synchrotron
    source_type: str = ""              # e.g. "calibration", "excimer", "diode", "fiber", ...
    emission_lines_nm: list[float] = field(default_factory=list)  # discrete lines for lamps
    spectral_bandwidth_nm: float = 0.0  # FWHM for broadband / LED sources
    lifetime_hours: float = 0.0        # typical lamp/LED lifetime

    @property
    def is_cw(self) -> bool:
        return self.rep_rate_hz == 0 or self.pulse_width_s == 0

    @property
    def peak_power_w(self) -> float:
        if self.is_cw:
            return self.power_w
        return self.power_w / (self.rep_rate_hz * self.pulse_width_s)

    @property
    def pulse_energy_j(self) -> float:
        if self.is_cw:
            return 0.0
        return self.power_w / self.rep_rate_hz

    @property
    def fluence_j_cm2(self) -> float:
        """Fluence at beam center (assuming Gaussian, 1/e² radius)."""
        radius_cm = (self.beam_diameter_mm / 2) / 10
        area_cm2 = math.pi * radius_cm ** 2
        energy = self.pulse_energy_j if not self.is_cw else self.power_w
        return energy / area_cm2 if area_cm2 > 0 else 0.0

    @property
    def irradiance_w_cm2(self) -> float:
        """Peak irradiance (W/cm²)."""
        radius_cm = (self.beam_diameter_mm / 2) / 10
        area_cm2 = math.pi * radius_cm ** 2
        return self.peak_power_w / area_cm2 if area_cm2 > 0 else 0.0

    @property
    def wavelength_um(self) -> float:
        """Wavelength in micrometers."""
        return self.wavelength_nm / 1000.0

    @property
    def wavelength_m(self) -> float:
        """Wavelength in meters."""
        return self.wavelength_nm * 1e-9

    @property
    def photon_energy_ev(self) -> float:
        """Single-photon energy in eV."""
        return 1240.0 / self.wavelength_nm if self.wavelength_nm > 0 else 0.0

    @property
    def is_tunable(self) -> bool:
        return self.tunable_range_nm is not None

    def get_mode_params(self) -> BeamModeParams:
        """Deserialise beam_mode_params dict into BeamModeParams."""
        return BeamModeParams(**self.beam_mode_params) if self.beam_mode_params else BeamModeParams()


# ── Default commercial lasers ──────────────────────────────────────

DEFAULT_LASERS: list[LaserSource] = [
    # ════════════════════════════════════════════════════════════════
    # ULTRAFAST LASERS
    # ════════════════════════════════════════════════════════════════
    LaserSource(
        name="Light Conversion Pharos 2mJ",
        wavelength_nm=1030.0,
        power_w=20.0,             # 2 mJ @ 10 kHz = 20 W avg
        rep_rate_hz=10_000.0,
        pulse_width_s=190e-15,    # < 190 fs (compressible to < 100 fs)
        beam_diameter_mm=5.0,
        m_squared=1.2,
        gain_medium="Yb:KGW",
        spatial_mode="TEM00",
        tunable_range_nm=(257, 2600),  # with harmonics
        category="laser", source_type="ultrafast",
        notes=(
            "Light Conversion Pharos industrial fs laser. "
            "2 mJ / pulse, < 190 fs, tunable rep rate single-shot to 1 MHz. "
            "Max avg power 20 W. Harmonics: 515, 343, 257 nm. "
            "Typical pump source for Orpheus OPA. Supports burst mode."
        ),
    ),
    LaserSource(
        name="Light Conversion Orpheus OPA",
        wavelength_nm=8500.0,
        power_w=0.2,
        rep_rate_hz=10_000.0,
        pulse_width_s=170e-15,
        beam_diameter_mm=5.0,
        m_squared=1.5,
        gain_medium="OPA (BBO/LGS)",
        pump_source="Light Conversion Pharos 2mJ",
        spatial_mode="TEM00",
        tunable_range_nm=(630, 16000),
        category="laser", source_type="opa",
        notes="Orpheus OPA. Tunable 630 nm – 16 µm (signal + idler).",
    ),
    LaserSource(
        name="Ti:Sapphire 800nm Ultrafast",
        wavelength_nm=800.0,
        power_w=1.0,
        rep_rate_hz=1000.0,
        pulse_width_s=100e-15,
        beam_diameter_mm=5.0,
        m_squared=1.2,
        gain_medium="Ti:Sapphire",
        tunable_range_nm=(680, 1080),
        category="laser", source_type="ultrafast",
        notes="1 kHz, 100 fs, 1 mJ/pulse. Tunable 680–1080 nm.",
    ),
    # ════════════════════════════════════════════════════════════════
    # CW & Q-SWITCHED LASERS
    # ════════════════════════════════════════════════════════════════
    LaserSource(
        name="Nd:YAG 1064nm CW",
        wavelength_nm=1064.0,
        power_w=10.0,
        beam_diameter_mm=2.0,
        m_squared=1.1,
        gain_medium="Nd:YAG",
        category="laser", source_type="dpss",
        notes="Standard CW near-IR laser",
    ),
    LaserSource(
        name="Nd:YAG 1064nm Q-switched",
        wavelength_nm=1064.0,
        power_w=5.0,
        rep_rate_hz=10.0,
        pulse_width_s=10e-9,
        beam_diameter_mm=3.0,
        m_squared=1.5,
        gain_medium="Nd:YAG",
        category="laser", source_type="q-switched",
        notes="10 Hz, 10 ns pulses, 500 mJ/pulse",
    ),
    LaserSource(
        name="Nd:YAG 532nm (2ω)",
        wavelength_nm=532.0,
        power_w=2.5,
        rep_rate_hz=10.0,
        pulse_width_s=8e-9,
        beam_diameter_mm=3.0,
        m_squared=1.5,
        gain_medium="Nd:YAG + KTP",
        category="laser", source_type="q-switched",
        notes="Frequency-doubled Nd:YAG. Common Raman excitation source.",
    ),
    LaserSource(
        name="DPSS 532nm CW",
        wavelength_nm=532.0,
        power_w=0.2,
        beam_diameter_mm=1.2,
        m_squared=1.05,
        gain_medium="Nd:YVO4 + KTP",
        category="laser", source_type="dpss",
        notes="200 mW CW green laser for Raman, alignment, fluorescence excitation.",
    ),
    LaserSource(
        name="CO2 10.6µm CW",
        wavelength_nm=10600.0,
        power_w=100.0,
        beam_diameter_mm=10.0,
        m_squared=1.3,
        gain_medium="CO2",
        tunable_range_nm=(9200, 11400),
        category="laser", source_type="gas",
        notes="Industrial CW CO2 laser. Tunable 9.2–11.4 µm via grating.",
    ),
    # ════════════════════════════════════════════════════════════════
    # FIBER LASERS
    # ════════════════════════════════════════════════════════════════
    LaserSource(
        name="Er:Fiber 1550nm CW",
        wavelength_nm=1550.0,
        power_w=0.5,
        beam_diameter_mm=0.01,
        m_squared=1.05,
        gain_medium="Er:Glass fiber",
        category="laser", source_type="fiber",
        notes="Telecom-grade single-mode fiber laser",
    ),
    LaserSource(
        name="Yb:Fiber 1070nm High-Power",
        wavelength_nm=1070.0,
        power_w=1000.0,
        beam_diameter_mm=14.0,
        m_squared=1.1,
        gain_medium="Yb:Glass fiber",
        category="laser", source_type="fiber",
        notes="kW-class industrial fiber laser",
    ),
    # ════════════════════════════════════════════════════════════════
    # UV / EXCIMER LASERS
    # ════════════════════════════════════════════════════════════════
    LaserSource(
        name="Excimer KrF 248nm",
        wavelength_nm=248.0,
        power_w=20.0,
        rep_rate_hz=100.0,
        pulse_width_s=25e-9,
        beam_diameter_mm=20.0,
        m_squared=50.0,
        gain_medium="KrF",
        spatial_mode="Top-Hat",
        category="laser", source_type="excimer",
        notes="UV excimer, 100 Hz, 25 ns, highly multimode",
    ),
    LaserSource(
        name="Excimer ArF 193nm",
        wavelength_nm=193.0,
        power_w=10.0,
        rep_rate_hz=200.0,
        pulse_width_s=20e-9,
        beam_diameter_mm=15.0,
        m_squared=100.0,
        gain_medium="ArF",
        spatial_mode="Top-Hat",
        category="laser", source_type="excimer",
        notes="Deep-UV excimer for DUVRR, lithography, surface processing.",
    ),
    # ════════════════════════════════════════════════════════════════
    # ALIGNMENT / VISIBLE LASERS
    # ════════════════════════════════════════════════════════════════
    LaserSource(
        name="HeNe 632.8nm",
        wavelength_nm=632.8,
        power_w=0.005,
        beam_diameter_mm=0.8,
        m_squared=1.0,
        gain_medium="HeNe",
        category="laser", source_type="gas",
        notes="Alignment laser, 5 mW",
    ),
    # ════════════════════════════════════════════════════════════════
    # DIODE LASERS (Raman excitation, etc.)
    # ════════════════════════════════════════════════════════════════
    LaserSource(
        name="Diode 785nm (Raman)",
        wavelength_nm=785.0,
        power_w=0.5,
        beam_diameter_mm=3.0,
        m_squared=1.1,
        gain_medium="GaAlAs",
        polarization="linear",
        category="laser", source_type="diode",
        notes="785 nm stabilized diode for NIR Raman. Low fluorescence excitation.",
    ),
    LaserSource(
        name="Diode 405nm (Raman/Fluorescence)",
        wavelength_nm=405.0,
        power_w=0.1,
        beam_diameter_mm=2.0,
        m_squared=1.2,
        gain_medium="InGaN",
        category="laser", source_type="diode",
        notes="405 nm diode for resonance Raman and fluorescence excitation.",
    ),
    LaserSource(
        name="Diode 830nm (Raman)",
        wavelength_nm=830.0,
        power_w=0.3,
        beam_diameter_mm=3.0,
        m_squared=1.1,
        gain_medium="GaAlAs",
        category="laser", source_type="diode",
        notes="830 nm stabilized diode for NIR Raman.",
    ),
    LaserSource(
        name="Diode 976nm (Pump)",
        wavelength_nm=976.0,
        power_w=50.0,
        beam_diameter_mm=0.1,
        m_squared=20.0,
        gain_medium="InGaAs",
        category="laser", source_type="diode",
        notes="976 nm pump diode for Yb fiber/solid-state lasers.",
    ),
    # ════════════════════════════════════════════════════════════════
    # CALIBRATION LAMPS
    # ════════════════════════════════════════════════════════════════
    LaserSource(
        name="Hg(Ar) Calibration Lamp",
        wavelength_nm=546.1,  # strongest visible line
        power_w=0.01,
        beam_diameter_mm=25.0,
        m_squared=100.0,
        polarization="unpolarized",
        spatial_mode="Top-Hat",
        category="lamp", source_type="calibration",
        emission_lines_nm=[253.7, 296.7, 302.2, 313.2, 334.1, 365.0, 404.7, 435.8, 546.1, 577.0, 579.1, 696.5, 706.7, 714.7, 727.3, 738.4, 750.4, 763.5, 772.4, 794.8, 811.5, 826.5, 842.5, 912.3],
        spectral_bandwidth_nm=0.01,
        lifetime_hours=5000,
        notes="Mercury-Argon pen lamp. Standard wavelength calibration for spectrometers. Hg lines + Ar fill gas lines.",
    ),
    LaserSource(
        name="Ne Calibration Lamp",
        wavelength_nm=640.2,
        power_w=0.005,
        beam_diameter_mm=25.0,
        m_squared=100.0,
        polarization="unpolarized",
        spatial_mode="Top-Hat",
        category="lamp", source_type="calibration",
        emission_lines_nm=[585.2, 588.2, 594.5, 597.6, 603.0, 607.4, 614.3, 616.4, 621.7, 626.6, 630.5, 633.4, 638.3, 640.2, 650.7, 659.9, 667.8, 671.7, 692.9, 703.2, 717.4, 724.5, 743.9],
        spectral_bandwidth_nm=0.005,
        lifetime_hours=5000,
        notes="Neon pen lamp. Dense line spectrum in 580–750 nm. Standard for visible-NIR calibration.",
    ),
    LaserSource(
        name="Ar Calibration Lamp",
        wavelength_nm=763.5,
        power_w=0.005,
        beam_diameter_mm=25.0,
        m_squared=100.0,
        polarization="unpolarized",
        spatial_mode="Top-Hat",
        category="lamp", source_type="calibration",
        emission_lines_nm=[696.5, 706.7, 714.7, 727.3, 738.4, 750.4, 763.5, 772.4, 794.8, 800.6, 801.5, 810.4, 811.5, 826.5, 840.8, 842.5, 852.1, 866.8, 912.3, 922.4, 965.8],
        spectral_bandwidth_nm=0.005,
        lifetime_hours=8000,
        notes="Argon pen lamp. Strong lines 700–970 nm. NIR calibration standard.",
    ),
    LaserSource(
        name="Kr Calibration Lamp",
        wavelength_nm=760.2,
        power_w=0.005,
        beam_diameter_mm=25.0,
        m_squared=100.0,
        polarization="unpolarized",
        spatial_mode="Top-Hat",
        category="lamp", source_type="calibration",
        emission_lines_nm=[427.4, 431.9, 436.3, 437.6, 439.9, 445.4, 450.2, 556.2, 557.0, 587.1, 760.2, 768.5, 769.5, 785.5, 805.9, 810.4, 811.3, 819.0, 826.3, 829.8],
        spectral_bandwidth_nm=0.005,
        lifetime_hours=5000,
        notes="Krypton lamp. Lines span UV to NIR. Useful for broadband wavelength calibration.",
    ),
    LaserSource(
        name="D₂ (Deuterium) UV Lamp",
        wavelength_nm=250.0,  # approximate center of UV continuum
        power_w=0.03,
        beam_diameter_mm=10.0,
        m_squared=100.0,
        polarization="unpolarized",
        category="lamp", source_type="broadband",
        spectral_bandwidth_nm=200.0,  # broad UV continuum 160–400 nm
        tunable_range_nm=(160, 400),
        lifetime_hours=2000,
        notes="Deuterium arc lamp. Smooth UV continuum 160–400 nm. FTIR / UV-Vis background source.",
    ),
    LaserSource(
        name="Tungsten-Halogen QTH",
        wavelength_nm=900.0,  # peak of ~3000K blackbody
        power_w=0.5,
        beam_diameter_mm=10.0,
        m_squared=100.0,
        polarization="unpolarized",
        category="lamp", source_type="broadband",
        spectral_bandwidth_nm=2000.0,
        tunable_range_nm=(350, 2500),
        lifetime_hours=10000,
        notes="Quartz-tungsten-halogen lamp. Smooth blackbody continuum 350–2500 nm. VIS-NIR reference source.",
    ),
    LaserSource(
        name="Xe Arc Lamp",
        wavelength_nm=550.0,
        power_w=0.15,
        beam_diameter_mm=5.0,
        m_squared=50.0,
        polarization="unpolarized",
        category="lamp", source_type="broadband",
        spectral_bandwidth_nm=800.0,
        tunable_range_nm=(200, 2000),
        lifetime_hours=1500,
        emission_lines_nm=[764.2, 823.2, 828.0, 881.9, 916.3, 979.9],
        notes="Xenon arc lamp. Near-solar continuum 200–2000 nm with Xe lines in NIR. Fluorescence excitation, solar simulation.",
    ),
    LaserSource(
        name="Globar (SiC) MIR Source",
        wavelength_nm=6000.0,  # peak ~1200 K
        power_w=0.01,
        beam_diameter_mm=6.0,
        m_squared=100.0,
        polarization="unpolarized",
        category="lamp", source_type="broadband",
        spectral_bandwidth_nm=15000.0,
        tunable_range_nm=(1000, 25000),
        lifetime_hours=20000,
        notes="Silicon carbide Globar. Broadband MIR thermal emitter for FTIR spectroscopy. Covers 1–25 µm.",
    ),
    LaserSource(
        name="Nernst Glower (ZrO₂)",
        wavelength_nm=5000.0,
        power_w=0.02,
        beam_diameter_mm=2.0,
        m_squared=100.0,
        polarization="unpolarized",
        category="lamp", source_type="broadband",
        spectral_bandwidth_nm=10000.0,
        tunable_range_nm=(1000, 20000),
        lifetime_hours=5000,
        notes="Zirconia Nernst glower. MIR broadband source for FTIR. Higher emissivity than Globar in 2–10 µm.",
    ),
    # ════════════════════════════════════════════════════════════════
    # LEDs
    # ════════════════════════════════════════════════════════════════
    LaserSource(
        name="UV LED 265nm",
        wavelength_nm=265.0,
        power_w=0.01,
        beam_diameter_mm=5.0,
        m_squared=50.0,
        polarization="unpolarized",
        category="led", source_type="uv",
        spectral_bandwidth_nm=12.0,
        lifetime_hours=10000,
        notes="Deep-UV LED for DUVRR excitation, disinfection, fluorescence.",
    ),
    LaserSource(
        name="UV LED 365nm",
        wavelength_nm=365.0,
        power_w=0.5,
        beam_diameter_mm=8.0,
        m_squared=50.0,
        polarization="unpolarized",
        category="led", source_type="uv",
        spectral_bandwidth_nm=10.0,
        lifetime_hours=20000,
        notes="365 nm UV-A LED. Fluorescence excitation, photocuring.",
    ),
    LaserSource(
        name="White LED (Broadband)",
        wavelength_nm=550.0,
        power_w=1.0,
        beam_diameter_mm=10.0,
        m_squared=100.0,
        polarization="unpolarized",
        category="led", source_type="broadband",
        spectral_bandwidth_nm=300.0,
        tunable_range_nm=(400, 700),
        lifetime_hours=50000,
        notes="Phosphor-converted white LED. Broad visible spectrum 400–700 nm. Illumination source.",
    ),
    LaserSource(
        name="NIR LED 940nm",
        wavelength_nm=940.0,
        power_w=0.1,
        beam_diameter_mm=5.0,
        m_squared=50.0,
        polarization="unpolarized",
        category="led", source_type="nir",
        spectral_bandwidth_nm=50.0,
        lifetime_hours=50000,
        notes="940 nm NIR LED for transmission measurements, moisture sensing.",
    ),
    # ════════════════════════════════════════════════════════════════
    # SUPERCONTINUUM / BROADBAND LASER SOURCES
    # ════════════════════════════════════════════════════════════════
    LaserSource(
        name="Supercontinuum (NKT SuperK)",
        wavelength_nm=1064.0,  # seed wavelength
        power_w=4.0,
        rep_rate_hz=78e6,
        pulse_width_s=5e-12,
        beam_diameter_mm=1.0,
        m_squared=1.1,
        gain_medium="PCF",
        category="laser", source_type="supercontinuum",
        spectral_bandwidth_nm=1600.0,
        tunable_range_nm=(400, 2400),
        notes="NKT SuperK supercontinuum. Broadband 400–2400 nm from PCF. Tunable via AOTF/VARIA. 78 MHz.",
    ),
    LaserSource(
        name="ASE Broadband Source 1550nm",
        wavelength_nm=1550.0,
        power_w=0.02,
        beam_diameter_mm=0.01,
        m_squared=1.05,
        gain_medium="Er:fiber ASE",
        category="laser", source_type="ase",
        spectral_bandwidth_nm=80.0,
        tunable_range_nm=(1520, 1600),
        notes="Erbium ASE broadband source. 1520–1600 nm, ~20 mW. Fiber optic component testing.",
    ),
]


def load_custom_lasers() -> list[LaserSource]:
    """Load user-defined lasers from JSON."""
    if not LASER_DB_PATH.exists():
        return []
    data = json.loads(LASER_DB_PATH.read_text())
    return [LaserSource(**d) for d in data.get("custom_lasers", [])]


def save_custom_lasers(lasers: list[LaserSource]):
    LASER_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    existing = {}
    if LASER_DB_PATH.exists():
        existing = json.loads(LASER_DB_PATH.read_text())
    existing["custom_lasers"] = [asdict(l) for l in lasers]
    LASER_DB_PATH.write_text(json.dumps(existing, indent=2))


def all_lasers() -> list[LaserSource]:
    """Return defaults + custom lasers."""
    return DEFAULT_LASERS + load_custom_lasers()


def get_laser_by_name(name: str) -> LaserSource | None:
    """Look up a laser by name from defaults + custom."""
    for laser in all_lasers():
        if laser.name == name:
            return laser
    return None
