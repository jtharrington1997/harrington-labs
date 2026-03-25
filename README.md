# harrington-labs

Photonics lab simulators and laser-material interaction (LMI) dissertation platform. Reducing geometric footprint of light sources while improving power and energy — advancing science, technology, and medicine through light-matter interaction research.

All physics engines are JIT-accelerated via `harrington_common.compute` — automatic CUDA GPU → Numba CPU → NumPy fallback.

**~12,800 lines of Python** across 11 simulation engines, 4 LMI engines, and 12 Streamlit pages.

## Pages

| # | Page | Description |
|---|------|-------------|
| 1 | Direct Diode Lab | L-I curves, thermal rollover, wavelength drift, far-field, spectral beam combining |
| 2 | Fiber Laser Lab | Gain modeling, pump/signal evolution, nonlinear thresholds, thermal, beam combining |
| 3 | Pulsed Laser Lab | Ultrafast profiles, autocorrelation, GDD dispersion, z-scan, beam combining (SBC + CBC) |
| 4 | Quantum Dots Lab | Brus equation, PL/absorption spectra, exciton dynamics, temperature dependence |
| 5 | Beam Control Lab | Atmospheric propagation, Fried parameter, Rytov scintillation, AO Strehl |
| 6 | Coatings Lab | Transfer matrix, spectral/angular reflectance, E-field, GDD, custom stacks |
| 7 | Advanced Spectroscopy Lab | Raman (spontaneous + SRS), Brillouin (spontaneous + SBS), DUVRR, LIBS, FTIR, hyperspectral imaging |
| 8 | Laser Library | 35 sources (pulsed + CW lasers, calibration lamps, LEDs, broadband) with grouped sidebar selector |
| 9 | Material Database | 27 materials with Sellmeier dispersion. Optical, thermal, mechanical properties |
| 10 | Demonstrator Builder | Full-chain laser demonstrator — resonator, QD fiber laser, QD diode + beam combining, M&S workspace |
| 11 | Modeling & Simulation | LMI workspace — regime classification, beam propagation, nonlinear, thermal, z-scan, campaign overlay, export. Custom / shared beam source support |
| 90 | Admin | API keys, data management, compute backend status |

## Simulation Engines

All engines live under `src/harrington_labs/simulation/` with no Streamlit imports.

| Engine | JIT | Key Capabilities |
|--------|-----|------------------|
| `direct_diode` | Yes | L-I, WPE, junction temperature, far-field, beam combining (via shared module) |
| `fiber_laser` | Yes | `_fiber_propagation_kernel`, SBS/SRS/SPM thresholds, V-number, thermal |
| `beam_control` | Yes | `_turbulence_broadening_kernel`, Fried r0, Rytov variance, AO correction |
| `pulsed_laser` | Yes | Temporal/spectral profiles, autocorrelation, GDD scan, open-aperture z-scan |
| `quantum_dots` | Yes | `_brus_vectorized`, PL spectra, exciton dynamics, Varshni temperature |
| `coatings` | Yes | TMM via `parallel_map` spectral sweep, angular response, E-field |
| `beam_combining` | Yes | `_cbc_far_field_kernel`, SBC + CBC estimators — shared by all source labs |
| `spectroscopy` | Yes | `_raman_spectrum_kernel`, `_lorentzian_kernel`, `_gaussian_kernel` — 8 techniques |
| `qd_fiber_laser` | Yes | `_pump_sweep_kernel`, 8 QD materials, Auger, Q-switch, mode-lock |
| `qd_diode_combiner` | Yes | `_qd_diode_li_kernel`, SBC/CBC/Hybrid beam combining, Strehl decomposition |

## LMI Platform

Merged from the former harrington-lmi repo. Engines under `src/harrington_labs/lmi/simulation/`:

| Engine | JIT | Key Capabilities |
|--------|-----|------------------|
| `beam_propagation` | Yes | `_gaussian_profile_kernel`, `_tophat_profile_kernel`, material propagation, z-scan via `parallel_map` |
| `nonlinear` | Yes | `_estimate_mpa_coefficient_kernel`, `_nonlinear_propagation` (MPA + Kerr depth-resolved) |
| `thermal` | Yes | `_thermal_accumulation` O(N²), `_euler_two_temp` two-temperature model |
| `custom_models` | — | User-defined Python models loaded from `data/custom_models/` |

Sellmeier dispersion for 10 optical materials (Si, Ge, GaAs, SiO₂, BK7, LiNbO₃, KTP, Sapphire, ZnSe, CaF₂). 27 total materials including metals, polymers, and biological tissue.

**Dissertation target**: sapphire 2100 nm / 5 mm open-aperture z-scan validation.

## Cross-Lab Features

- **Model Comparison** — every lab has a panel to upload experimental data (xlsx, csv, txt) and compare against simulation output with R², RMSE, NRMSE, residual plots. Includes downloadable templates.
- **Shared Beam State** — configure a source in one lab, carry the parameters to other labs via session state. All 7 lab pages receive shared beam; source labs (Direct Diode, Fiber Laser, Pulsed Laser, Quantum Dots) can push.
- **Source Database** — 35 light sources with grouped sidebar selector on every page.
- **Material Database** — 27 materials with sidebar selector on every page.
- **Beam Combining** — all source labs (Direct Diode, Fiber Laser, Pulsed Laser) include spectral and/or coherent beam combining. Demonstrator Builder adds QD-based SBC/CBC/Hybrid architectures.
- **Reference Library** — upload research papers and datasheets on any lab page; persisted to `data/references/`.

## Demonstrator Builder

The Demonstrator Builder (page 10) hosts four design tools:

**Resonator Builder** — first-pass estimate for bulk/slab gain media (Nd:YAG, Ti:Sapph, Yb:YAG, Er:Glass, etc.) with pump architecture, cavity FSR, slope efficiency, and threshold.

**QD Fiber Laser + Beam Combining** — quantum-dot-doped fiber laser with 8 QD materials (PbS, PbSe, InAs, CdSe, InP, Si, Perovskite). Empirical sizing curves, Auger recombination, single/multi-exciton gain, Q-switched and mode-locked operation. SBC and CBC combining of multiple QD fiber channels.

**QD Diode + Beam Combining** — QD active region diode array with three beam combining architectures:
- Spectral Beam Combining (SBC): diffraction grating, gain-bandwidth-limited channels, spectral fill
- Coherent Beam Combining (CBC): phase-locked aperture, Strehl decomposition (phase/tip-tilt/fill), far-field pattern
- Hybrid SBC+CBC: CBC sub-arrays spectrally combined for simultaneous brightness and power scaling

**Modeling & Simulation** — links to the full LMI workspace for regime classification, beam propagation, nonlinear optics, thermal modeling, campaign overlays, and publication-quality export.

## Key Modules

| Module | Purpose |
|--------|---------|
| `src/harrington_labs/simulation/` | 11 physics engines for labs + testbeds (no Streamlit imports) |
| `src/harrington_labs/lmi/simulation/` | 4 LMI engines: beam propagation, nonlinear, thermal, custom models |
| `src/harrington_labs/lmi/domain/` | 35 laser sources, 27 materials, regime classification, plot specs |
| `src/harrington_labs/comparison/` | Model-vs-experiment framework: metrics, parsers, templates, UI |
| `src/harrington_labs/domain/` | Dataclasses for all labs including spectroscopy |
| `src/harrington_labs/ui/` | Shared UI: Americana theme, shared beam state, database sidebar selectors |
| `tools/` | CLI utilities: campaign generation, knife-edge reformatter, digital twin, inbox watcher |

## Installation

```bash
# Base (includes numba + joblib)
uv sync

# With CUDA GPU
pip install "harrington-labs[cuda]"
```

Port: **8505**
