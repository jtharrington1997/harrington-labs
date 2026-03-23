"""
units.py — Canonical unit system for Harrington LMI

All internal calculations should use these base units unless explicitly noted.
UI layers may convert for display, but must convert back before computation.
"""

# =========================
# BASE UNIT DEFINITIONS
# =========================

# Length
M = 1.0
CM = 1e-2 * M
MM = 1e-3 * M
UM = 1e-6 * M
NM = 1e-9 * M

# Time
S = 1.0
MS = 1e-3 * S
US = 1e-6 * S
NS = 1e-9 * S
PS = 1e-12 * S
FS = 1e-15 * S

# Energy
J = 1.0
MJ = 1e-3 * J
UJ = 1e-6 * J
NJ = 1e-9 * J

# Intensity (derived, W/m^2 internally)
W = 1.0
W_PER_M2 = 1.0
W_PER_CM2 = 1e4 * W_PER_M2  # 1 cm^2 = 1e-4 m^2

# =========================
# HELPER CONVERSIONS
# =========================

def um_to_m(x: float) -> float:
    return x * UM

def m_to_um(x: float) -> float:
    return x / UM

def fs_to_s(x: float) -> float:
    return x * FS

def s_to_fs(x: float) -> float:
    return x / FS

def wcm2_to_wm2(x: float) -> float:
    return x * W_PER_CM2

def wm2_to_wcm2(x: float) -> float:
    return x / W_PER_CM2
