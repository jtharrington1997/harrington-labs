# harrington-labs

Photonics lab environment simulators and laser-material interaction platform. Consolidates six lab simulators with the former `harrington-lmi` dissertation platform into one unified photonics application.

**Port 8505** · Streamlit + harrington-common

## Lab Environment Simulators

| Page | Lab | Physics |
|------|-----|---------|
| 1 | Direct Diode | L-I curves, thermal rollover, wavelength drift, far-field patterns, spectral beam combining |
| 2 | Fiber Laser | Gain modeling, pump/signal evolution, SBS/SRS/SPM thresholds, thermal limits, V-number |
| 3 | Beam Control | Atmospheric propagation, Fried parameter, Rytov scintillation, beam wander, AO Strehl |
| 4 | Pulsed Laser | Ultrafast temporal/spectral profiles, autocorrelation, GDD dispersion, open-aperture z-scan |
| 5 | Quantum Dots | Brus equation, PL/absorption spectra, exciton dynamics, temperature-dependent emission |
| 6 | Coatings | Transfer matrix method, spectral/angular reflectance, E-field profiles, GDD, custom stacks |

## Laser-Material Interaction Platform (merged from harrington-lmi)

| Page | Module | Function |
|------|--------|----------|
| 7 | Laser Library | Commercial lasers, custom sources, OPA chaining, spatial beam modes (TEM00, Top-Hat, HG, LG, Bessel) |
| 8 | Material Database | Sellmeier dispersion for Si, Ge, GaAs, SiO2, BK7, LiNbO3, KTP, Sapphire, ZnSe, CaF2. Optical/thermal/mechanical properties |
| 9 | Modeling & Simulation | Beam propagation through finite-thickness slabs, nonlinear absorption, Kerr/self-focusing, z-scan, thermal analysis, campaign comparison |
| 10 | Source Builder | Gain medium, pump source, resonator, output coupler design |

Legacy pages (30–80) preserve transitional compatibility surfaces from the original LMI migration.

## Package Structure

```
src/harrington_labs/
├── domain/           Lab simulator data structures (BeamParams, PulsedSource, QD, coatings, etc.)
├── simulation/       Lab simulator physics engines (6 modules, no Streamlit imports)
├── ui/               Lab-specific display helpers, plot conventions
└── lmi/              Laser-material interaction platform (merged from harrington-lmi)
    ├── domain/       Lasers, materials, Sellmeier, interactions, plot specs, units
    ├── simulation/   Beam propagation, nonlinear optics, thermal, custom models
    ├── io/           Gnuplot export, campaign import, CSV/JSON exporters
    └── ui/           LMI-specific formatting, branding, access control
```

## Dissertation Alignment

The LMI platform supports dissertation-quality supplemental modeling:

1. **Sapphire SCG / k–ω benchmarking** — through-focus slab propagation, B-integral, self-focusing susceptibility
2. **MIR ablation support** — fluence at focus, threshold/incubation, campaign comparison against morphology
3. **Platform claim** — demonstrates a reusable research software platform, not a one-off calculator

Primary validation dataset: 2100 nm pump, ~30 µJ, 1 kHz, ~170 fs, 5 mm sapphire window.

## Running

```bash
source ~/harrington/activate.sh
cd ~/harrington/harrington-labs
streamlit run app/streamlit_app.py
```

## TODO

- [ ] Finalize beam propagation kernel validation against sapphire z-scan data
- [ ] Add experiment overlay support for simulation vs measured comparison
- [ ] Build sapphire validation workflow with exportable figures
- [ ] Add sensitivity/uncertainty sweeps for key parameters
- [ ] Extract remaining page-local scientific logic into simulation/ and services/
- [ ] Add regression tests for beam propagation and nonlinear kernels
- [ ] Connect lab simulators to LMI material database (shared Sellmeier models)
- [ ] Add fiber laser amplifier chain modeling (seed → preamp → power amp)
- [ ] Add coating LIDT estimation based on E-field profiles
- [ ] Add atmospheric turbulence phase screen generation for beam control lab
