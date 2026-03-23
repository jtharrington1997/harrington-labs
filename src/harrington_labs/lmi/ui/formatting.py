"""
formatting.py — Professional scientific formatting layer

Consistent, SI-aware formatting for UI display.
"""

import math

# =========================
# CORE ENGINE
# =========================

SI_PREFIXES = [
    (1e-9, "n"),
    (1e-6, "µ"),
    (1e-3, "m"),
    (1, ""),
    (1e3, "k"),
    (1e6, "M"),
    (1e9, "G"),
]


def _format_sig(value: float, sig: int = 4) -> str:
    if value == 0:
        return "0"
    return f"{value:.{sig}g}"


def _auto_scale(value: float):
    """Return scaled value + prefix"""
    abs_val = abs(value)

    for factor, prefix in SI_PREFIXES:
        if abs_val < factor * 1000:
            return value / factor, prefix

    return value, ""


# =========================
# ENERGY (J)
# =========================

def fmt_energy_j(value: float, sig: int = 4) -> str:
    scaled, prefix = _auto_scale(value)
    return f"{_format_sig(scaled, sig)} {prefix}J"


# =========================
# LENGTH (m)
# =========================

def fmt_length_m(value: float, sig: int = 4) -> str:
    scaled, prefix = _auto_scale(value)
    return f"{_format_sig(scaled, sig)} {prefix}m"


# =========================
# TIME (s)
# =========================

def fmt_time_s(value: float, sig: int = 4) -> str:
    scaled, prefix = _auto_scale(value)
    return f"{_format_sig(scaled, sig)} {prefix}s"


# =========================
# ABSORPTION (cm⁻¹)
# =========================

def fmt_absorption_cm_inv(value: float, sig: int = 4) -> str:
    return f"{_format_sig(value, sig)} cm⁻¹"


# =========================
# FALLBACK / GENERIC
# =========================

def fmt_number(value: float, sig: int = 4) -> str:
    return _format_sig(value, sig)


def fmt_scientific(value: float, sig: int = 4) -> str:
    return f"{value:.{sig}e}"

# =========================
# FLUENCE (J/cm^2)
# =========================

def fmt_fluence_j_cm2(value: float, sig: int = 4) -> str:
    return f"{_format_sig(value, sig)} J/cm²"


# =========================
# DENSITY (kg/m^3)
# =========================

def fmt_density_kg_m3(value: float, sig: int = 4) -> str:
    return f"{_format_sig(value, sig)} kg/m³"


# =========================
# FREQUENCY (Hz)
# =========================

def fmt_frequency_hz(value: float, sig: int = 4) -> str:
    scaled, prefix = _auto_scale(value)
    return f"{_format_sig(scaled, sig)} {prefix}Hz"

# =========================
# POWER (W)
# =========================

def fmt_power_w(value: float, sig: int = 4) -> str:
    scaled, prefix = _auto_scale(value)
    return f"{_format_sig(scaled, sig)} {prefix}W"


# =========================
# IRRADIANCE (W/cm²)
# =========================

def fmt_irradiance_w_cm2(value: float, sig: int = 4) -> str:
    return f"{_format_sig(value, sig)} W/cm²"


# =========================
# WAVELENGTH (nm-based UI)
# =========================

def fmt_wavelength_nm(value_nm: float, sig: int = 4, dual: bool = True) -> str:
    """
    Display wavelength in nm, optionally with µm secondary.
    """
    if dual and value_nm >= 1000:
        return f"{_format_sig(value_nm, sig)} nm ({_format_sig(value_nm/1000, sig)} µm)"
    return f"{_format_sig(value_nm, sig)} nm"


# =========================
# NONLINEAR INDEX (cm²/W)
# =========================

def fmt_n2_cm2_w(value: float, sig: int = 4) -> str:
    return f"{_format_sig(value, sig)} cm²/W"


# =========================
# TEMPERATURE (K)
# =========================

def fmt_temp_k(value: float, sig: int = 4) -> str:
    return f"{_format_sig(value, sig)} K"


# =========================
# THERMAL CONDUCTIVITY (W/m·K)
# =========================

def fmt_thermal_conductivity(value: float, sig: int = 4) -> str:
    return f"{_format_sig(value, sig)} W/m·K"


#==========================
# REFRACTIVE INDEX
#==========================

def fmt_refractive_index(value: float, sig: int = 4) -> str:
    return _format_sig(value, sig)


#==========================
# ELECTRON VOLT (eV)
#==========================

def fmt_ev(value: float, sig: int = 4) -> str:
    return f"{_format_sig(value, sig)} eV"

