# harrington-labs

Photonics lab simulators and laser-material interaction (LMI) dissertation platform. All physics engines are JIT-accelerated via `harrington_common.compute` — automatic CUDA GPU → Numba CPU → NumPy fallback.

## Lab Simulators (Pages 1–6)

| Lab | Physics Engines | JIT Status |
|-----|----------------|------------|
| Direct Diode | L-I curves, thermal rollover, wavelength drift, far-field, spectral beam combining | ✓ Compute import |
| Fiber Laser | Gain modeling, pump/signal evolution, SBS/SRS/SPM thresholds, V-number, thermal | ✓ `_fiber_propagation_kernel` |
| Beam Control | Atmospheric propagation, Fried parameter, Rytov scintillation, AO Strehl | ✓ `_turbulence_broadening_kernel` |
| Pulsed Laser | Ultrafast profiles, autocorrelation, GDD dispersion, open-aperture z-scan | ✓ Compute import |
| Quantum Dots | Brus equation, PL/absorption spectra, exciton dynamics, temperature dependence | ✓ `_brus_vectorized` |
| Coatings | Transfer matrix, spectral/angular reflectance, E-field, GDD, custom stacks | ✓ `parallel_map` spectral sweep |

## LMI Platform (Pages 7–10)

Merged from harrington-lmi. Sellmeier dispersion for 10 materials (Si, Ge, GaAs, SiO₂, BK7, LiNbO₃, KTP, Sapphire, ZnSe, CaF₂).

**JIT-accelerated simulation engines** (`src/harrington_labs/lmi/simulation/`):
- **beam_propagation** — `_gaussian_w_z`, `_material_propagation_loop` kernels; z-scan uses `parallel_map` for >50 positions
- **nonlinear** — `_nonlinear_propagation` kernel (MPA + Kerr depth-resolved)
- **thermal** — `_thermal_accumulation` O(N²) kernel, `_euler_two_temp` kernel

**Dissertation target**: sapphire 2100 nm / 5 mm open-aperture z-scan validation.

## Pages

- 1–6: Lab simulators
- 7: Laser Library
- 8: Material Database
- 9: Modeling & Simulation
- 10: Source Builder
- 30–80: Legacy pages (migration continuity)
- 90: Admin (API keys, data management, **Compute backend status**)

## Installation

```bash
# Base (includes numba + joblib)
uv sync

# With CUDA GPU
pip install "harrington-labs[cuda]"
```

Port: **8505**
