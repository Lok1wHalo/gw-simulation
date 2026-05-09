# Gravitational Wave Simulation

A Python-based numerical simulation of Binary Black Hole (BBH) systems and the resulting gravitational radiation. This project models the orbital dynamics of compact binaries and computes the spacetime strain ($h_+$ and $h_\times$) based on the quadrupole formula and Post-Newtonian approximations.

## Features
- **Inspiral Modeling:** Simulates the shrinking orbit of binary systems due to energy loss via gravitational waves.
- **Waveform Analysis:** Generates "Chirp" signals showing the evolution of frequency and amplitude over time.
- **Numerical Relativity:** Uses ODE solvers to track the trajectory of two bodies under mutual gravitational influence.
- **Visualization:** 2D and 3D plotting of the orbital paths and the resulting gravitational wave strain.

## Mathematical Foundation
The simulation calculates the gravitational wave strain $h$ using the mass quadrupole moment $Q_{ij}$ of the system:

$$h_{ij} = \frac{2G}{c^4 r} \frac{d^2 Q_{ij}}{dt^2}$$

As the black holes orbit closer, the orbital frequency increases, leading to the characteristic increase in signal amplitude and frequency before the final merger.

## Installation
1. Clone this repository:
```bash
   git clone https://github.com/Lok1wHalo/gw-simulation.git
   cd gw-simulation
   ```
3. Install the required Python packages:
   ```bash
   pip install -r requirements.txt
   ```
## Usage
Execute the main script to run the simulation and generate plots:
   ```bash
   python main.py
   ```
## Project Structure
- Simulation.py: The core engine handling the physics of the binary orbit.
- GravitationalWaves.py: Logic for calculating strain polarizations and wave energy.
- main.py: Entry point for user configuration and visualization.

## License
This project is open-source and available under the MIT License.
