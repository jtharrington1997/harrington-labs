# harrington-labs

Photonics lab environment simulators for the Harrington research software ecosystem.

## Labs

| # | Lab | Physics |
|---|-----|---------|
| 1 | **Direct Diode** | L-I curves, thermal rollover, wavelength drift, far-field patterns, spectral beam combining |
| 2 | **Fiber Laser** | Gain modeling, pump/signal evolution, SBS/SRS/SPM thresholds, thermal limits, V-number & mode analysis |
| 3 | **Beam Control** | Atmospheric propagation, Fried parameter, scintillation (Rytov), beam wander/spread, adaptive optics Strehl |
| 4 | **Pulsed Laser** | Ultrafast temporal/spectral profiles, autocorrelation, dispersion management (GDD), open-aperture z-scan |
| 5 | **Quantum Dots** | Brus equation (size-dependent bandgap), PL/absorption spectra, exciton dynamics, temperature-dependent emission |
| 6 | **Coatings** | Transfer matrix method, spectral/angular reflectance, E-field distribution, GDD, preset & custom stacks |

## Architecture

Same ecosystem model as `harrington-lmi`, `automation-station`, and `pax-americana`:

- **Streamlit + `harrington-common`** → GUI, Americana theme, layout
- **`src/harrington_labs/simulation/`** → Pure physics engines (no Streamlit imports)
- **`src/harrington_labs/domain/`** → Typed dataclasses, enums, physical constants
- **`src/harrington_labs/ui/`** → App-specific display helpers

## Running

```bash
cd harrington-labs
pip install -e "../harrington-common"
pip install -e .
streamlit run app/streamlit_app.py
```

## Conventions

- Plotly charts: `plotly_white` template, transparent background, `width="stretch"`
- `st.dataframe` uses `use_container_width=True`
- Admin page is always the last numbered page
- Physics engines never import Streamlit
