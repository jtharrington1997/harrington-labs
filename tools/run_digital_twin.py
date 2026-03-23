from pathlib import Path
import json
import sys
import numpy as np

from harrington_lmi.simulation.beam_propagation import BeamParams
from harrington_lmi.simulation.thermal import thermal_analysis

q = 1.602e-19
eps0 = 8.854e-12
c = 3e8

def carrier_density(rho, t):
    mu = 450 if t == "p" else 1350 if t == "n" else 1000
    return 1 / (q * mu * rho)

def drude_alpha(N, wavelength_m):
    N = N * 1e6
    omega = 2 * np.pi * c / wavelength_m
    m_eff = 0.26 * 9.11e-31
    tau = 1e-13
    sigma = N * q**2 * tau / m_eff
    return sigma / (2 * eps0 * c * omega**2)

def main():
    cfg = json.load(open(sys.argv[1]))
    out = Path(sys.argv[2])

    b = cfg["beam"]

    wavelength_m = b["wavelength_um"] * 1e-6
    energy = b["pulse_energy_uJ"] * 1e-6
    pulse = b["pulse_duration_fs"] * 1e-15
    rep = b["rep_rate_hz"]
    diameter = b["beam_diameter_mm"] * 1e-3

    bp = BeamParams(
        wavelength_m=wavelength_m,
        beam_diameter_1e2_m=diameter,
        m_squared=b["m_squared"],
        pulse_energy_j=energy,
        pulse_width_s=pulse,
        rep_rate_hz=rep,
    )

    peak = bp.peak_power_w

    w0 = 10e-6
    area = np.pi * w0**2

    fluence = energy / area / 1e4
    irradiance = peak / area / 1e4

    rho = cfg["rho"]
    t = cfg["type"]

    N = carrier_density(rho, t)
    alpha_cm = drude_alpha(N, wavelength_m) / 100

    thermal = thermal_analysis(
        fluence_j_cm2=fluence,
        pulse_width_s=pulse,
        rep_rate_hz=rep,
        spot_radius_m=w0,
        alpha_cm=alpha_cm,
        thermal_conductivity_w_mk=150,
        density_kg_m3=2330,
        specific_heat_j_kgk=700,
        thermal_diffusivity_m2_s=150/(2330*700),
        melting_point_k=1687,
    )

    result = {
        "rho_ohm_cm": rho,
        "carrier_density_cm3": N,
        "alpha_cm": alpha_cm,
        "delta_T_K": thermal.delta_t_surface_k,
        "above_melt": thermal.operating_above_melt,
    }

    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump(result, open(out, "w"), indent=2)

if __name__ == "__main__":
    main()
