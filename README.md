# Gravitational Wave Simulation

A Python simulation framework for modelling gravitational waves from compact binary inspirals using Newtonian and quadrupole approximations.

The project reproduces key observable features of gravitational-wave events such as GW150914, including:

- Orbital inspiral
- Frequency chirp evolution
- Polarized gravitational-wave strain
- Spectrogram generation
- Chirp mass recovery
- Waveform overlap analysis
- Comparison with public LIGO data
  
## Physical Model

The simulation models binary black hole inspirals using:

- Peters–Mathews gravitational radiation formalism
- Quadrupole gravitational-wave emission
- Circular binary approximation
- Inspiral-only waveform generation
- ISCO termination condition

The waveform does NOT currently include:

- Merger phase
- Ringdown phase
- Spin effects
- Eccentric orbits
- Post-Newtonian corrections
- Detector noise modelling

## Repository Structure
```
gw-simulation/
│
├── physics/
│   ├── orbital.py      # Orbital evolution and inspiral dynamics
│   ├── waveform.py     # GW strain generation
│   └── spectrum.py     # FFT and spectrogram analysis
│
├── analysis/
│   ├── comparison.py   # Match filtering and parameter recovery
│   └── gwosc_compare.py# Optional comparison with real LIGO data
│
├── plots/
│   └── visualise.py    # Publication-quality plotting
│
├── results/            # Generated figures
├── main.py             # Main execution script
├── config.py           # Physical constants and configuration
└── requirements.txt
```

## Installation
temp

## Key Equations

Chirp mass:

$$M_c = (m1 m2)^{3/5} / (m1 + m2)^{1/5}$$

Frequency evolution:

$$df/dt = (96/5) π^{8/3} (G M_c / c^3)^{5/3}) f^{11/3}$$

Strain amplitude:

$$h ~ (G M_c)^{5/3} f^{2/3} / (c^4 d_L)$$

## Limitations

This project is intended for educational and research-demonstration purposes.

The simulation uses Newtonian inspiral approximations and does not reproduce the full numerical-relativity waveform used by LIGO/Virgo collaborations.

## Project Structure
- Simulation.py: The core engine handling the physics of the binary orbit.
- GravitationalWaves.py: Logic for calculating strain polarizations and wave energy.
- main.py: Entry point for user configuration and visualization.

## License
This project is open-source and available under the MIT License.
