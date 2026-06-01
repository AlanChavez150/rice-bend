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

Each run is persisted to `<output_dir>/<run_name>/` (default `results/data_dump/`;
existing contents are cleared on each run) — `run.npz` numeric arrays, `run.json`
metadata + start/stop conditions, config snapshots, and a copy of the plot. Use
`--seed` for reproducibility and `--no-save` to skip persistence. Tune the
`gerchberg_saxton` / `output` blocks in the config (e.g. `history_stride`, which controls
how often per-iteration state is captured).

### `mgs-animate`

Animate the TX aperture phase estimate as it evolves over Gerchberg-Saxton iterations,
from a saved run (requires `output.save_gs_history: true`, the default; renders `.mp4`
via ffmpeg):

```bash
mgs-animate                              # latest run under results/ -> tx_estimate.mp4
mgs-animate results/<run> -o out.mp4     # specific run / output path
mgs-animate --fps 20 --show
```

For a smoother animation, lower `gerchberg_saxton.history_stride` in the config before
running `mgs` (stride 1 captures every iteration).

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
- `config.py` — pydantic config models (scene, Gerchberg-Saxton, output)
- `data_store.py` — per-run persistence (run.npz / run.json)
- `mgs.py` — modified Gerchberg-Saxton driver (entry point `mgs`)
- `animate.py` — TX-estimate animation from a saved run (entry point `mgs-animate`)
- `traj.py` — trajectory generation/search tool (entry point `traj`)
