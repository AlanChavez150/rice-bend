# rice-bend

Phase retrieval of mm-wave (150 GHz) beams using a modified Gerchberg-Saxton (MGS)
algorithm. Given field measurements at a receiver plane, reconstruct the transmitter
aperture's phase profile and recover the trajectory of a curved ("accelerating") beam.

Supports both pure simulation and real experimental data captured from an oscilloscope.

## Install

```bash
pip install -e .
```

## Usage

Two console entry points are installed:

### `mgs`

Runs the phase-retrieval pipeline. With no data paths it runs a pure simulation:

```bash
mgs                  # pure simulation
mgs --freq 150e9     # set carrier frequency (Hz)
mgs --debug          # verbose logging
```

To run against experimental `.mat` data, pass both paths:

```bash
mgs --rx-path RX.mat --heatmap-path HEATMAP.mat
```

On a headless machine, force a non-interactive matplotlib backend so it writes
`mgs.png` instead of opening a window:

```bash
MPLBACKEND=Agg mgs
```

### `traj`

Standalone trajectory tool — generate a phase plate from `Ax^2 + Bx + C` and
search to recover the trajectory:

```bash
traj 0.3 0.01 0.025 --accuracy 100
```

## Source layout

All code lives in `src/rice_bend/`:

- `rs.py` — Rayleigh-Sommerfeld wave propagation
- `caustic.py` — phase-plate design for parabolic beam trajectories
- `sim_scene.py` — aperture/scene data structures and experimental `.mat` I/O
- `mgs.py` — modified Gerchberg-Saxton driver (entry point `mgs`)
- `traj.py` — trajectory generation/search tool (entry point `traj`)
