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
mgs --freq 140e9 150e9 160e9                     # ONE joint solve across several frequencies
mgs --debug                                      # verbose logging
```

With more than one frequency (several `--freq` values, or a `frequencies:` list in
the config — the CLI overrides it) `mgs` runs **one joint solve**: a single
achromatic phase mask fitted against every frequency's measurement at once, the
loss being the mean of the per-frequency losses. A single frequency is just the
length-1 case of the same path. The scene panels render at the primary (first)
frequency; `run.json` records the joint `final_loss` plus `final_loss_per_freq`.

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

Each run is persisted to `<output_dir>/<run_name>/` (default `results/data_dump/`) —
`run.npz` numeric arrays, `run.json` metadata + start/stop conditions, config
snapshots, and a copy of the plot. Use `-o/--out PATH` to name the run directory
outright, `--seed` for reproducibility and `--no-save` to skip persistence. Tune the
`gerchberg_saxton` / `output` blocks in the config (e.g. `history_stride`, which controls
how often per-iteration state is captured).

An existing run directory is cleared when a new run saves into it, but **only if it
was written by the same tool** — `mgs` will not clear a `grid-search-mgs` run or vice
versa, and neither will clear a directory holding neither's marker file. The check
runs before the solve, so a collision costs you nothing, and the directory is created
only once the solve has finished, so an interrupted run never destroys previous
results without producing new ones.

To run against experimental `.mat` captures, see `--rx-path` / `--heatmap-path` below;
the bench constants for that path (down-conversion chain, aperture edges, coordinate
origins, amplitude normalisation, scene margins) live in the config's `experimental:`
block.

### `grid-search-mgs`

Localizes a transmitter whose **location is unknown**. The real TX location/trajectory in
the config is used only to synthesize the shared RX measurement set (one measurement per
frequency); the tool then sweeps a speculative `(z, x_center)` grid, runs one joint MGS
phase retrieval at each hypothesized plane (a fixed-width aperture with uniform assumed
amplitude, one phase mask fitted against every frequency's measurement at once), and
saves each reconstruction as a **candidate beam**.

```bash
grid-search-mgs                                 # full sweep from the config's grid_search block
grid-search-mgs --freq 140e9 150e9 160e9        # joint sweep: one solve per candidate across all three
grid-search-mgs --dry-run                       # enumerate the grid (counts + per-frequency validity), no MGS
grid-search-mgs --limit 10                      # quick partial run
grid-search-mgs -j 8                            # worker processes (default: all cores; 1 = serial)
grid-search-mgs -o results/my_run               # name the run directory outright
grid-search-mgs --scenes                        # one PNG per candidate beam (scenes/) + an averaged scene
grid-search-mgs --anim                          # animate the candidate beams -> candidate_beams.mp4 (ffmpeg)
grid-search-mgs --surface-anim                  # orbit each residual surface -> residual_surface_orbit.mp4 (ffmpeg)
grid-search-mgs --true-mgs                      # also recompute the baseline MGS run at the KNOWN TX
grid-search-mgs --replot results/grid_search    # re-plot a saved run (joint or legacy layout)
grid-search-mgs --replot results/grid_search --scenes --anim   # standalone scenes + animation for a saved run
```

The residual heatmap, surface and scatter (with their high-contrast twins) are always
written. One rule sets everything else: **cheap plots always run; anything that
re-solves MGS, renders per-candidate PNGs or encodes video is opt-in** (`--true-mgs`,
`--scenes`, `--anim`, `--surface-anim`). A fresh run and a `--replot` of it therefore
produce the same set of files.

Configure the sweep in the `grid_search` block of the config: the `z` / `x_center` sweeps,
the assumed aperture `width` (its `dx` is provenance-only — candidates are sampled on the
scene grid), a fixed `seed` (reused across candidates so residuals are comparable), and
`gs_overrides.max_iters` for a cheaper search. Each run is saved to
`<output_dir>/<run_name>/` — the run name comes from the config's `output.run_name`
(fallback `grid_search`), e.g. `results/scenario_caustic_hit/`, or `-o/--out` names the
directory outright.

Plain `mgs` resolves `output.run_name` to the same path, but the two tools will no longer
clear each other's runs: each refuses to clear a directory that does not carry its own
marker file, and says so before starting work. Use `-o/--out` to keep them apart.

- `candidate_beams.json` — manifest: ground-truth TX location, the grid spec, the frequency
  list (the alignment order for every per-frequency value in the run), and the indexed
  candidate list (z, x_center, joint residual, `per_freq_losses` with `null` where a
  frequency failed the sampling check, `freq_valid`, iters, stop reason) plus any skipped points.
- `measurement.npz` — the per-frequency RX fields and error weightings (stacked `(F, nx)` on
  the scene axis), the ragged per-frequency RX element arrays (`rx_aper_axis_00`, ...), and
  the real-TX aperture (one shared axis, per-frequency profiles).
- `candidates/cand_####.{npz,json}` — each reconstructed aperture + loss curve, and its metadata.
- `residual_heatmap.png` — GS residual over `(z, x_center)`; lower = better
  data fit, with the true location and best candidate marked.
- `residual_surface.png` — the same grid as relief instead of colour: height is the
  **inverted** residual, so the better a candidate fits, the higher the surface stands
  and the best fit becomes a peak over the TX. The flat heatmap saturates (on a dense
  caustic sweep over half the cells sit within 1% of the maximum), which hides how deep
  the basin actually goes. The z-axis is labelled in real residual values, the flat
  heatmap is projected on the floor beneath the surface, and the true TX and best
  candidate carry full-height marker poles so neither can be hidden behind the terrain.
- `residual_scatter.png` — a scatter of every candidate's GS residual (fixed
  `[0, 0.06]` axis, no colorbar) against its distance to the true TX, so the trend (a lower residual
  marking a candidate closer to the real transmitter) is visible.
- `residual_heatmap_hc.png` / `residual_surface_hc.png` / `residual_scatter_hc.png` — **high-contrast**
  twins written alongside every residual plot: a log residual scale whose floor adapts to the data
  (that run's minimum loss, rounded down to the nearest decade) with a fixed `0.06` ceiling, so each
  populated decade gets an equal share of the colormap (or y-axis, or surface height). The lowest
  (best-fit) residuals differentiate maximally while high residuals compress into nearly one dark
  color. Because the floor is data-driven, hc colors are NOT comparable across runs — use the linear
  `[0, 0.06]` plots for cross-run comparison. On the surface this is the difference between a broad
  mesa covering the whole basin (linear) and a single sharp summit on the TX (hc).
- `true_mgs_scene.png` — (with `--true-mgs`) the single baseline MGS run at the *known* (true) TX
  location: the real scene vs. its MGS reconstruction plus the TX aperture phase/amplitude, exactly
  as plain `mgs` would produce. Unlike the residual plots this recomputes one full MGS solve.
- `scenes/cand_####.png` + `scene_average.png` — (with `--scenes`) one 4-panel PNG per candidate
  (candidate beam, real beam at the same color scale, candidate aperture phase, candidate aperture
  amplitude; scenes mark the RX aperture in red and the TX aperture in blue), plus a single plot
  averaging every candidate beam. Use `--scene-top N` to render only the N lowest-residual candidates.
- `residual_surface_orbit.mp4` / `_hc.mp4` — (with `--surface-anim`) the residual surface orbited
  through a full turn (needs ffmpeg). A still is fixed to one viewing angle, and which angle reads
  best depends on where the basin lands, so this is the way past a peak hidden behind a ridge.
  Opt-in because it is ~12 s of render per video per run directory, against a `--replot` that
  otherwise finishes in seconds.
- `candidate_beams.mp4` — (with `--anim`) an animation sweeping the candidate beams, one frame per
  candidate re-illuminating the scene (needs ffmpeg). Every frame is held in memory at once to fix
  a shared colour scale, so `--anim` renders the 100 lowest-residual candidates by default;
  `--scene-top N` overrides that, and warns past 400 frames.

Lower residual = better data fit = more likely TX location, which is the input to the
candidate-ranking step. On a headless machine set `MPLBACKEND=Agg`.

`--jobs/-j` sets the worker-process count for the sweep and for scene/animation rendering
(default: all cores; `1` = serial). Results do not depend on it — every candidate uses the
same fixed seed, and results are assembled by input index, so the manifest and every
candidate `.npz` are byte-identical whatever `-j` you pass.

#### Multiple frequencies

To solve the same scene against several carrier frequencies at once, list them in the
config's top-level `frequencies` block (Hz), or pass several values to `--freq` (which
overrides the config):

```yaml
# null / absent -> single --freq, default 150e9
frequencies: [100.0e9, 110.0e9, 120.0e9, 130.0e9, 140.0e9, 150.0e9, 160.0e9, 170.0e9, 180.0e9, 190.0e9]
```

With more than one frequency each candidate gets **one joint solve**: a single achromatic
phase mask is fitted against every frequency's measurement simultaneously (the
per-frequency forward operators all act on the same aperture field), and the residual the
solver minimizes — and the plots rank by — is the **mean of the per-frequency losses**.
The measurement set is still per-frequency: each frequency synthesizes its own RX field
with its own TX beam phase (∝ wavenumber) and its own RX element count when
`rx_aperture.dx` is null (λ/20 spacing).

A candidate too close to the RX plane may fail the Rayleigh-Sommerfeld sampling check at
the highest frequencies while passing at lower ones; such a candidate is solved jointly
over its **valid frequency subset** (the joint loss is the mean over that subset), with
`freq_valid` and the `null`-padded `per_freq_losses` in the manifest recording exactly
which frequencies contributed. Only a candidate invalid at *every* frequency is skipped.

Results land in **one flat run directory** whatever the frequency count — same layout and
plot names as a single-frequency run (`residual_heatmap.png` is the joint residual), plus
the multi-frequency extras:

- `residual_scatter_3d.png` — the **3D residual scatter**: every candidate plotted with
  `(x_center, z)` on the bottom plane and frequency rising vertically, colored by that
  frequency's residual **component** of the joint solve on the same fixed `[0, 0.06]`
  `viridis_r` scale as the 2D heatmap, with the true TX marked on each frequency layer.
  Dot size shrinks cubically as the residual grows, so high-MSE dots are near-invisible
  and the layers stay see-through. `residual_scatter_3d_hc.png` is its high-contrast
  (log color scale) twin.
- `residual_scatter_3d_diff.png` — 3D **difference-from-center scatter**: every layer
  shows component(frequency) − component(center frequency) (the middle of the sorted
  list, e.g. 150 GHz) on a symmetric `viridis_r` scale — yellow = better than the
  baseline, purple = worse, teal = unchanged (the baseline layer is uniformly zero).
  Dot size grows with the deviation, so candidates that behave like the baseline stay
  see-through.
- `residual_heatmap_vs_joint_<GHz>.png` — one per frequency: that frequency's component
  heatmap and the joint heatmap side by side (shared linear scale), plus their difference
  (frequency − joint) on a symmetric `viridis_r` scale — yellow where that frequency fits
  better than the joint, dark purple where worse. `residual_heatmap_vs_joint_<GHz>_hc.png`
  is its high-contrast twin (data-driven log scale on the heatmaps, symmetric log on the
  difference).

Scene renders (`--scenes`, `--anim`, `--true-mgs`) are inherently monochromatic; on a
multi-frequency run they use the **centre frequency** by default, or `--scene-freq HZ`
(one of the run's frequencies) to pick another view.

`grid-search-mgs --replot <run_name>` regenerates every plot from the saved manifest; it
re-solves nothing unless you pass `--true-mgs`. Pointed at a results directory from the
retired per-frequency layout (`freq_<GHz>/` subdirs + `frequencies.json`, written before
the joint solver), it regenerates each subdir's own plots and explains that the joint
plots require a re-run — averaging independent per-frequency solves after the fact is not
the same measurement as one joint solve.

**When comparing results, re-run rather than `--replot`.** `--replot --true-mgs` regenerates
`true_mgs_scene.png` from a fresh solve while reading residuals from the saved manifest, so
replotting a run made by older code produces a figure pair that silently disagrees with itself.
`schema_version` in `candidate_beams.json` / `run.json` distinguishes result sets (3 = joint
multi-frequency; 2 = the retired per-frequency layout).

### `mgs-animate`

Animate the TX aperture phase estimate as it evolves over Gerchberg-Saxton iterations,
from a saved run (requires `output.save_gs_history: true`, the default; renders `.mp4`
via ffmpeg):

```bash
mgs-animate                              # latest run under results/ -> tx_estimate.mp4
mgs-animate --mode scene                 # 2D scene re-illuminated by the TX estimate
mgs-animate results/<run> -o out.mp4     # specific run / output path
mgs-animate --freq-index 1               # multi-frequency run: which frequency's view
mgs-animate --fps 20 --show
```

On a joint multi-frequency run, `--freq-index` (an index into `run.json`'s
`frequencies_hz`, default 0) picks the scene-mode wavelength and the phase-mode
Real-TX overlay — the reconstructed mask is achromatic, but the real plate's phase
scales with the wavenumber, so the overlay is one frequency's view and is labelled
with it. The loss panel always shows the joint curve.

How many frames each mode renders:

- `--mode phase` animates **every captured iteration**, so its frame count is set by
  `gerchberg_saxton.history_stride`. Capturing more enlarges `run.npz`; raise
  `mgs-animate --fps` to keep the video short. Every shipped config now uses
  `history_stride: 50` — at `1`, a default `mgs` run wrote 576 MB of capture arrays and
  `--mode phase` rendered 10,000 frames rather than 200.
- `--mode scene` recomputes a full Rayleigh-Sommerfeld propagation per frame (expensive),
  so it **subsamples the captured iterations to ~60 frames by default** regardless of
  `history_stride`. Use `--frame-stride 1` for every captured iteration (slow), and
  `--z-stride` to trade scene resolution for speed.

## Source layout

All code lives in `src/rice_bend/`:

Numerics and geometry:

- `rs.py` — Rayleigh-Sommerfeld propagation: the kernel, applying it, and whole-scene illumination
- `caustic.py` — phase-plate design for parabolic beam trajectories
- `sim_scene.py` — aperture and scene geometry
- `interp.py` — the complex-interpolation conventions, named (cartesian for fields, amplitude-only
  for magnitudes, polar for the caustic construction). They are **not** interchangeable
- `config.py` — pydantic config models (scene, TX beam, RX aperture, Gerchberg-Saxton, output,
  grid search, experimental bench constants)

Infrastructure:

- `data_store.py` — per-run persistence (run.npz / run.json / config snapshots) and the small
  I/O helpers both entry points share
- `parallel.py` — one parallel map; `jobs=1` runs the same task function the pool would
- `plotting.py` — the scene panel and line panel both entry points draw
- `cli.py` — shared logging setup
- `exp_data.py` — readers for the experimental `.mat` captures

Entry points:

- `mgs.py` — the MGS solver core and driver (`mgs`)
- `animate.py` — TX-estimate animation from a saved run (`mgs-animate`)
- `grid_search.py` — orchestration + CLI for the sweep (`grid-search-mgs`), over three modules:
  - `grid_sweep.py` — enumerate the grid, solve at each point, persist. No matplotlib
  - `residual_plots.py` — the residual grid and its six plots
  - `candidate_scenes.py` — re-illuminated candidate beams and the true-MGS baseline

## Tests

There is no test suite; `scripts/characterize.sh` is the net. It runs a fixed command list
into a scratch directory, extracts a numeric digest (final losses, iteration counts, stop
reasons, npz key lists and content hashes, output file lists, the grid's joint and
per-frequency loss vectors, the joint-equals-mean-of-components invariant, and the
index → (z, x_center) map) and diffs it against `scripts/characterize_expected.json`:

```bash
scripts/characterize.sh            # run + diff, non-zero exit on any difference
scripts/characterize.sh --bless    # regenerate the expected digest
```

Takes ~25 s. The expected values are generated, never transcribed — hand-typed float literals
rot the moment nobody re-blesses them.
