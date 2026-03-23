# harrington-labs

Photonics lab environment simulators and laser-material interaction platform for the Harrington research software ecosystem. Consolidates the former `harrington-lmi` dissertation platform with six lab simulators into one unified photonics application.

## Lab Simulators

| # | Lab | Physics |
|---|-----|---------|
| 1 | **Direct Diode** | L-I curves, thermal rollover, wavelength drift, far-field patterns, spectral beam combining |
| 2 | **Fiber Laser** | Gain modeling, pump/signal evolution, SBS/SRS/SPM thresholds, thermal limits, V-number & mode analysis |
| 3 | **Beam Control** | Atmospheric propagation, Fried parameter, scintillation (Rytov), beam wander/spread, adaptive optics Strehl |
| 4 | **Pulsed Laser** | Ultrafast temporal/spectral profiles, autocorrelation, dispersion management (GDD), open-aperture z-scan |
| 5 | **Quantum Dots** | Brus equation (size-dependent bandgap), PL/absorption spectra, exciton dynamics, temperature-dependent emission |
| 6 | **Coatings** | Transfer matrix method, spectral/angular reflectance, E-field distribution, GDD, preset & custom stacks |

## Laser-Material Interaction Platform (from harrington-lmi)

| # | Module | Function |
|---|--------|----------|
| 7 | **Laser Library** | Commercial lasers, custom sources, OPA chaining, spatial beam modes |
| 8 | **Material Database** | Sellmeier dispersion, optical/thermal/mechanical properties |
| 9 | **Modeling & Simulation** | Beam propagation, nonlinear optics, z-scan, thermal analysis, campaign comparison |
| 10 | **Source Builder** | Gain medium, pump, resonator, output coupler design |

## Architecture

- **Streamlit + `harrington-common`** → GUI, Americana theme, layout
- **`src/harrington_labs/simulation/`** → Lab simulator physics engines
- **`src/harrington_labs/lmi/`** → LMI domain models, simulation kernels, IO/export (merged from harrington-lmi)
- **`src/harrington_labs/domain/`** → Shared lab simulator data structures
- **`src/harrington_labs/ui/`** → App-specific display helpers

## Running

```bash
cd harrington-labs
pip install -e "../harrington-common"
pip install -e .
streamlit run app/streamlit_app.py
```
