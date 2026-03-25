# harrington-labs

Photonics lab simulators and laser-material interaction (LMI) dissertation platform. Reducing geometric footprint of light sources while improving power and energy — advancing science, technology, and medicine through light-matter interaction research.

All physics engines are JIT-accelerated via `harrington_common.compute` — automatic CUDA GPU → Numba CPU → NumPy fallback.

## Lab Simulators (Pages 1–7)

| Lab | Physics Engines | JIT Status |
|-----|----------------|------------|
| Direct Diode | L-I curves, thermal rollover, wavelength drift, far-field, spectral beam combining | ✓ Compute import |
| Fiber Laser | Gain modeling, pump/signal evolution, SBS/SRS/SPM thresholds, V-number, thermal | ✓ `_fiber_propagation_kernel` |
| Beam Control | Atmospheric propagation, Fried parameter, Rytov scintillation, AO Strehl | ✓ `_turbulence_broadening_kernel` |
| Pulsed Laser | Ultrafast profiles, autocorrelation, GDD dispersion, open-aperture z-scan | ✓ Compute import |
| Quantum Dots | Brus equation, PL/absorption spectra, exciton dynamics, temperature dependence | ✓ `_brus_vectorized` |
| Coatings | Transfer matrix, spectral/angular reflectance, E-field, GDD, custom stacks | ✓ `parallel_map` spectral sweep |
| Spectroscopy | Raman (spontaneous + SRS), Brillouin (spontaneous + SBS), DUVRR, LIBS, FTIR, hyperspectral imaging | — |

## Cross-Lab Features

- **Model Comparison** — every lab has a panel to upload experimental data (xlsx, csv, txt) and compare against simulation output with R², RMSE, NRMSE, residual plots. Includes downloadable templates.
- **Shared Beam State** — configure a source in one lab, carry the parameters to other labs via session state.
- **Source Database** — 32 light sources (19 lasers, 9 calibration/broadband lamps, 4 LEDs) with grouped sidebar selector on every page.
- **Material Database** — 27 materials with Sellmeier dispersion for 10 optical materials. Sidebar selector on every page.
- **Reference Library** — upload research papers and datasheets on any lab page; persisted to `data/references/`.

## LMI Platform (Pages 8–10)

Merged from the former harrington-lmi repo. Sellmeier dispersion for 10 materials (Si, Ge, GaAs, SiO₂, BK7, LiNbO₃, KTP, Sapphire, ZnSe, CaF₂).

**JIT-accelerated simulation engines** (`src/harrington_labs/lmi/simulation/`):
- **beam_propagation** — `_gaussian_w_z`, `_material_propagation_loop` kernels; z-scan uses `parallel_map` for >50 positions
- **nonlinear** — `_nonlinear_propagation` kernel (MPA + Kerr depth-resolved)
- **thermal** — `_thermal_accumulation` O(N²) kernel, `_euler_two_temp` kernel

**Dissertation target**: sapphire 2100 nm / 5 mm open-aperture z-scan validation.

## Pages

- 1–6: Lab simulators (Direct Diode, Fiber Laser, Beam Control, Pulsed Laser, Quantum Dots, Coatings)
- 7a: Advanced Spectroscopy Lab
- 7: Laser Library
- 8: Material Database
- 9: Modeling & Simulation
- 10: Source Builder (resonator design, QD fiber laser testbed, QD diode + beam combining)
- 90: Admin (API keys, data management, compute backend status)

## Source Builder Testbeds

The Source Builder (page 10) hosts three design tools:

**Resonator Builder** — first-pass estimate for bulk/slab gain media (Nd:YAG, Ti:Sapph, Yb:YAG, Er:Glass, etc.) with pump architecture, cavity FSR, slope efficiency, and threshold.

**QD Fiber Laser** — quantum-dot-doped fiber laser with 8 QD materials (PbS, PbSe, InAs, CdSe, InP, Si, Perovskite). Empirical sizing curves, Auger recombination, single/multi-exciton gain, Q-switched and mode-locked operation.

**QD Diode + Beam Combining** — QD active region diode array with three beam combining architectures:
- Spectral Beam Combining (SBC): diffraction grating, gain-bandwidth-limited channels, spectral fill
- Coherent Beam Combining (CBC): phase-locked aperture, Strehl decomposition (phase/tip-tilt/fill), far-field pattern
- Hybrid SBC+CBC: CBC sub-arrays spectrally combined for simultaneous brightness and power scaling

## Key Modules

| Module | Purpose |
|--------|---------|
| `src/harrington_labs/simulation/` | Physics engines for all 7 labs + 2 testbeds (no Streamlit imports) |
| `src/harrington_labs/comparison/` | Model-vs-experiment framework: metrics, parsers, templates, UI |
| `src/harrington_labs/lmi/` | LMI platform: beam propagation, nonlinear, thermal engines |
| `src/harrington_labs/domain/` | Dataclasses for all labs including spectroscopy |
| `src/harrington_labs/ui/` | Shared UI: theme, shared beam state, database sidebar selectors |

## Installation

```bash
# Base (includes numba + joblib)
uv sync

# With CUDA GPU
pip install "harrington-labs[cuda]"
```

Port: **8505**
