# rice-bend

Phase retrieval of mm-wave (150 GHz) beams using a modified Gerchberg-Saxton (MGS)
algorithm. Given field measurements at a receiver plane, reconstruct the transmitter
aperture's phase profile for a curved ("accelerating") caustic or a directional
(steered) beam.

Supports both pure simulation and real experimental data captured from an oscilloscope.

## Install

```bash
pip install -e .
```

## Usage

Three console entry points are installed:

### `mgs`

Runs the phase-retrieval pipeline. With no data paths it runs a pure simulation:

```bash
mgs                                              # pure simulation (default: caustic beam)
mgs --config configs/directional_config.yml      # directional (steered) beam scene
mgs --freq 150e9                                 # set carrier frequency (Hz)
mgs --debug                                      # verbose logging
```

Two general configs ship in `configs/`: `caustic_config.yml` (an accelerating
"caustic" beam — the `mgs` default) and `directional_config.yml` (a steered
plane-wave beam). The emitted beam is selected by the `tx_aperture.beam.type`
field (`caustic` requires `trajectory: [a, b, c]`; `directional` requires
`steer_angle_deg`). The receive aperture is configured by the `rx_aperture` block
(`x_center` + `width`); shrink `width` for a smaller receiver.

Five **test scenarios** (`configs/scenario_*.yml`) exercise beam-vs-RX geometry,
each with a bare-minimum TX aperture and a 6.5 cm RX window, all sharing one
symmetric scene (`x[-0.3,0.3] z[0,0.85]`; `rs.rs` convolves a centered kernel, so
an off-center scene would shift the beam):

| Config | Beam | Result |
| --- | --- | --- |
| `scenario_directional_miss` | directional, slightly steered | near-misses RX (just past the window edge) |
| `scenario_directional_larger` | directional, centered | hits RX, beam **larger** than RX |
| `scenario_directional_smaller_hit` | directional, short stand-off | hits RX, beam **smaller** than RX |
| `scenario_caustic_miss` | caustic (same beam as hit, TX moved right) | misses RX to the right |
| `scenario_caustic_hit` | caustic | hits RX |

`scripts/verify_scenario.py CONFIG.yml [--expect-hit|--expect-miss] [--expect-size
larger|smaller]` simulates a scene and reports where/how the beam lands on the RX
plane (HIT vs MISS, beam size vs RX), for validating these geometries.

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

### `grid-search-mgs`

Localizes a transmitter whose **location is unknown**. The real TX location/trajectory in
the config is used only to synthesize the single shared RX measurement; the tool then
sweeps a speculative `(z, x_center)` grid, runs MGS phase retrieval at each hypothesized
plane (a fixed-width aperture with uniform assumed amplitude) against that one
measurement, and saves each reconstruction as a **candidate beam**.

```bash
grid-search-mgs                                # full sweep from the config's grid_search block
grid-search-mgs --dry-run                       # enumerate the grid (counts + skipped points), no MGS
grid-search-mgs --limit 10 --max-iters 500      # quick partial run
grid-search-mgs --summary                       # also write residual_heatmap.png
grid-search-mgs --scatter                        # also write residual_scatter.png (residual vs. distance to true TX)
grid-search-mgs --scenes                         # one PNG per candidate beam (scenes/) + an averaged scene
grid-search-mgs --anim                           # animate the candidate beams -> candidate_beams.mp4 (ffmpeg)
grid-search-mgs --replot results/grid_search    # re-plot residual heatmap + scatter + the true MGS run (TX known)
grid-search-mgs --replot results/grid_search --scenes          # + the averaged scene and per-candidate scenes
grid-search-mgs --replot results/grid_search --scenes --anim   # standalone scenes + animation for a saved run
```

Configure the sweep in the `grid_search` block of the config: the `z` / `x_center` sweeps,
the assumed aperture `width`/`dx`, a fixed `seed` (reused across candidates so residuals are
comparable), and `gs_overrides.max_iters` for a cheaper search. Each run is saved to
`<output_dir>/<run_name>/` (default `results/grid_search/`):

- `candidate_beams.json` — manifest: ground-truth TX location, the grid spec, and the
  indexed candidate list (z, x_center, final residual, iters, stop reason) plus any skipped points.
- `measurement.npz` — the shared RX field, error weighting, and RX / real-TX apertures.
- `candidates/cand_####.{npz,json}` — each reconstructed aperture + loss curve, and its metadata.
- `residual_heatmap.png` — (with `--summary`) GS residual over `(z, x_center)`; lower = better
  data fit, with the true location and best candidate marked.
- `residual_scatter.png` — (with `--scatter`) a scatter of every candidate's GS residual (fixed
  `[0, 0.1]` axis, no colorbar) against its distance to the true TX, so the trend (a lower residual
  marking a candidate closer to the real transmitter) is visible.
- `true_mgs_scene.png` — (on `--replot`) the single baseline MGS run at the *known* (true) TX
  location: the real scene vs. its MGS reconstruction plus the TX aperture phase/amplitude, exactly
  as plain `mgs` would produce. Unlike the residual plots this recomputes one full MGS solve.
- `scenes/cand_####.png` + `scene_average.png` — (with `--scenes`) one 4-panel PNG per candidate
  (candidate beam, real beam at the same color scale, candidate aperture phase, candidate aperture
  amplitude; scenes mark the RX aperture in red and the TX aperture in blue), plus a single plot
  averaging every candidate beam. Use `--scene-top N` to render only the N lowest-residual candidates,
  and `--scene-z-planes` to trade resolution for speed.
- `candidate_beams.mp4` — (with `--anim`) an animation sweeping the candidate beams, one frame per
  candidate re-illuminating the scene (needs ffmpeg; `--fps` sets the frame rate, `--scene-top N` /
  `--scene-z-planes` apply as for scenes).

Lower residual = better data fit = more likely TX location, which is the input to the
candidate-ranking step. On a headless machine set `MPLBACKEND=Agg` when using `--summary`/`--scatter`/`--replot`.

### `mgs-animate`

Animate the TX aperture phase estimate as it evolves over Gerchberg-Saxton iterations,
from a saved run (requires `output.save_gs_history: true`, the default; renders `.mp4`
via ffmpeg):

```bash
mgs-animate                              # latest run under results/ -> tx_estimate.mp4
mgs-animate --mode scene                 # 2D scene re-illuminated by the TX estimate
mgs-animate results/<run> -o out.mp4     # specific run / output path
mgs-animate --fps 20 --show
```

How many frames each mode renders:

- `--mode phase` animates **every captured iteration**, so its frame count is set by
  `gerchberg_saxton.history_stride` (`1` = every iteration). Capturing more enlarges
  `run.npz`; raise `mgs-animate --fps` to keep the video short.
- `--mode scene` recomputes a full Rayleigh-Sommerfeld propagation per frame (expensive),
  so it **subsamples the captured iterations to ~60 frames by default** regardless of
  `history_stride`. Use `--frame-stride 1` for every captured iteration (slow), and
  `--z-stride` to trade scene resolution for speed.

## Source layout

All code lives in `src/rice_bend/`:

- `rs.py` — Rayleigh-Sommerfeld wave propagation
- `caustic.py` — phase-plate design for parabolic beam trajectories
- `sim_scene.py` — aperture/scene data structures and experimental `.mat` I/O
- `config.py` — pydantic config models (scene, TX beam, RX aperture, Gerchberg-Saxton, output, grid search)
- `data_store.py` — per-run persistence (run.npz / run.json)
- `mgs.py` — modified Gerchberg-Saxton driver (entry point `mgs`)
- `grid_search.py` — speculative TX-location sweep producing candidate beams (entry point `grid-search-mgs`)
- `animate.py` — TX-estimate animation from a saved run (entry point `mgs-animate`)
