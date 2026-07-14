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
grid-search-mgs --freq 140e9 150e9 160e9        # run the whole sweep at each frequency (see below)
grid-search-mgs --dry-run                       # enumerate the grid (counts + skipped points), no MGS
grid-search-mgs --limit 10 --max-iters 500      # quick partial run
grid-search-mgs --summary                       # also write residual_heatmap.png
grid-search-mgs --scatter                        # also write residual_scatter.png (residual vs. distance to true TX)
grid-search-mgs --scenes                         # one PNG per candidate beam (scenes/) + an averaged scene
grid-search-mgs --anim                           # animate the candidate beams -> candidate_beams.mp4 (ffmpeg)
grid-search-mgs --replot results/grid_search    # re-plot residual heatmap + scatter (+hc twins) + the true MGS run (TX known)
grid-search-mgs --replot results/grid_search --skip-true-mgs   # plots only: skip the true-MGS recompute (no MGS solve)
grid-search-mgs --replot results/grid_search --scenes          # + the averaged scene and per-candidate scenes
grid-search-mgs --replot results/grid_search --scenes --anim   # standalone scenes + animation for a saved run
```

Configure the sweep in the `grid_search` block of the config: the `z` / `x_center` sweeps,
the assumed aperture `width`/`dx`, a fixed `seed` (reused across candidates so residuals are
comparable), and `gs_overrides.max_iters` for a cheaper search. Each run is saved to
`<output_dir>/<run_name>/` — the run name comes from the config's `output.run_name`
(overridable with `--run-name`; fallback `grid_search`), e.g. `results/scenario_caustic_hit/`.
**Caveat:** plain `mgs` uses the same `output.run_name`, and a run directory is cleared when a
new run starts saving into it — so running `mgs` with a scenario config replaces that
scenario's saved grid-search run (and vice versa). Use `--run-name` to keep them apart:

- `candidate_beams.json` — manifest: ground-truth TX location, the grid spec, and the
  indexed candidate list (z, x_center, final residual, iters, stop reason) plus any skipped points.
- `measurement.npz` — the shared RX field, error weighting, and RX / real-TX apertures.
- `candidates/cand_####.{npz,json}` — each reconstructed aperture + loss curve, and its metadata.
- `residual_heatmap.png` — (with `--summary`) GS residual over `(z, x_center)`; lower = better
  data fit, with the true location and best candidate marked.
- `residual_scatter.png` — (with `--scatter`) a scatter of every candidate's GS residual (fixed
  `[0, 0.06]` axis, no colorbar) against its distance to the true TX, so the trend (a lower residual
  marking a candidate closer to the real transmitter) is visible.
- `residual_heatmap_hc.png` / `residual_scatter_hc.png` — **high-contrast** twins written alongside
  every residual plot: a log residual scale whose floor adapts to the data (that run's minimum loss,
  rounded down to the nearest decade) with a fixed `0.06` ceiling, so each populated decade gets an
  equal share of the colormap (or y-axis). The lowest (best-fit) residuals differentiate maximally
  while high residuals compress into nearly one dark color. Because the floor is data-driven, hc
  colors are NOT comparable across runs — use the linear `[0, 0.06]` plots for cross-run comparison.
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

#### Multiple frequencies

To test the same scene at several carrier frequencies, list them in the config's top-level
`frequencies` block (Hz), or pass several values to `--freq` (which overrides the config):

```yaml
# null / absent -> single --freq, default 150e9
frequencies: [100.0e9, 110.0e9, 120.0e9, 130.0e9, 140.0e9, 150.0e9, 160.0e9, 170.0e9, 180.0e9, 190.0e9]
```

With more than one frequency the whole sweep runs **independently per frequency** (each
re-illuminates the scene and re-solves MGS at every candidate — note the RX element count
scales with wavelength when `rx_aperture.dx` is null). Results are laid out as:

- `<output_dir>/<run_name>/freq_<GHz>/` — one complete run dir per frequency (`candidate_beams.json`,
  `residual_heatmap.png`, etc., exactly as a single-frequency run), so `--scenes`/`--anim`/`--replot`
  all work per frequency.
- `<output_dir>/<run_name>/frequencies.json` — index tying the per-frequency subdirs together.
- `<output_dir>/<run_name>/residual_scatter_3d.png` — the **3D residual scatter**: every candidate
  plotted with `(x_center, z)` on the bottom plane and frequency rising vertically, colored by its GS
  residual on the same fixed `[0, 0.06]` `viridis_r` scale as the 2D heatmap, with the true TX marked
  on each frequency layer. Dot size shrinks cubically as the residual grows, so high-MSE dots are
  near-invisible and the layers stay see-through. This is the 2D residual data with frequency as the
  third axis. `residual_scatter_3d_hc.png` is its high-contrast (log color scale) twin.
- `<output_dir>/<run_name>/residual_heatmap_avg.png` — the **frequency-averaged residual heatmap**:
  each `(z, x_center)` cell is the mean GS residual across all frequencies where that candidate ran,
  so a location that fits well at *every* frequency stands out. `residual_heatmap_avg_hc.png` is its
  high-contrast twin.
- `<output_dir>/<run_name>/freq_<GHz>/residual_heatmap_vs_avg.png` — per-frequency **vs-average
  comparison**: that frequency's heatmap and the averaged heatmap side by side (shared linear scale),
  plus their difference (frequency − average) on a symmetric `viridis_r` scale — yellow where that
  frequency fits better than the average, dark purple where worse. `residual_heatmap_vs_avg_hc.png`
  is its high-contrast twin (data-driven log scale on the heatmaps, symmetric log on the difference).
- `<output_dir>/<run_name>/residual_scatter_3d_diff.png` — 3D **difference-from-center scatter**:
  every layer shows loss(frequency) − loss(center frequency) (the middle of the sorted list, e.g.
  150 GHz) on a symmetric `viridis_r` scale — yellow = better than the baseline, purple = worse,
  teal = unchanged (the baseline layer is uniformly zero). Dot size grows with the deviation, so
  candidates that behave like the baseline stay see-through.

`grid-search-mgs --replot <run_name>` detects the multi-frequency layout automatically: it regenerates
every per-frequency plot, the frequency-averaged heatmap pair, and both 3D scatters — add
`--skip-true-mgs` for a pure plots-only pass (no MGS solves rerun). A single frequency keeps the
original flat layout (no `freq_<GHz>/` subdirs, no 3D/averaged plots).

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
