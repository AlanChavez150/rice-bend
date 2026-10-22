# rice-bend

Phase retrieval of mm-wave (300 GHz) beams using a modified Gerchberg-Saxton (MGS)
algorithm. Given field measurements at a receiver plane, reconstruct the transmitter
aperture's phase profile for a curved ("accelerating") caustic, a directional
(steered), a Bessel (axicon) or a focused (lens) beam.

Supports both pure simulation and real experimental data captured from an oscilloscope.
The simulations default to a 300 GHz carrier, and the test scenarios add 30 dB of
receiver noise; the experimental captures come from a 150 GHz bench.

## Install

```bash
pip install -e .
```

Optional: the Rust solver engine (`--engine rust`; ~5× faster per candidate when
benchmarked before the 300 GHz retarget, see `rust/README.md`) is a separate extension
module under `rust/`. It needs a Rust toolchain and maturin:

```bash
uv pip install maturin
VIRTUAL_ENV=$PWD/.venv .venv/bin/maturin develop --release -m rust/Cargo.toml
```

## Usage

Six console entry points are installed (`mgs`, `grid-search-mgs`, `mgs-animate`,
`mgs-study`, `topt-parser`, `topt-heatmap`):

### `mgs`

Runs the phase-retrieval pipeline. With no data paths it runs a pure simulation:

```bash
mgs                                                       # pure simulation (default: configs/caustic_config.yml, 300 GHz)
mgs --config configs/scenario_caustic_hit_1f_grid2l.yml   # a campaign base config: caustic beam, on-grid RX, 30 dB
mgs --config configs/scenario_bessel_hit_1f_grid2l.yml    # the same scene with a Bessel (axicon) beam
mgs --config configs/scenario_focus_hit_1f_grid2l.yml     # the same scene with a focused (lens) beam
mgs --config configs/archive/directional_config.yml       # an archived config, by its new path
mgs --freq 300e9                                          # set carrier frequency (Hz)
mgs --freq 280e9 300e9 320e9                              # ONE joint solve across several frequencies
mgs --debug                                               # verbose logging
```

With neither `--freq` nor a `frequencies:` list in the config, the carrier is 300 GHz
(`rice_bend.config.DEFAULT_FREQ_HZ`), the fallback `mgs`, `grid-search-mgs` and
`mgs-study` share. The experimental path is the one exception (see below).

With more than one frequency (several `--freq` values, or a `frequencies:` list in
the config — the CLI overrides it) `mgs` runs **one joint solve**: a single shared
aperture profile — a delay plate whose phase scales with frequency (see the
physical-model section below) — fitted against every frequency's measurement at
once, the loss being the mean of the per-frequency losses. A single frequency is
just the length-1 case of the same path. The scene panels render at the primary
(first) frequency; `run.json` records the joint `final_loss` plus
`final_loss_per_freq` and `gs_result.ref_freq_hz`.

Right after the solve, a simulated run logs a `LANDING` block: where the energy lands
on the RX plane, the real beam against the solver's reconstruction, and the final loss
in context. One true / reconstructed / match triple per tone, in the run's frequency
order:

```
LANDING target: <beam type> x=<mm> mm (<how the geometric aim is defined>)
LANDING true          <f> GHz: peak, centroid, 5-95% [lo, hi] mm, in-window <fraction> | in-window peak, centroid
LANDING reconstructed <f> GHz: the same numbers for the reconstruction
LANDING match         <f> GHz: corr, rel-L2 vs true, power ratio, rel-L2 vs target
LANDING loss solver=<loss> true_plate=<loss> zero_model=<loss> ratio=<solver/true_plate> iters=<n> stop=<reason>
LANDING note: positions carry the propagator's +dx/2 lateral offset (...); outside the RX window the reconstruction is unconstrained
```

- `target` is the beam's geometric aim on the RX plane (ray optics: where the beam is
  aimed, not where a finite-wavelength field peaks).
- `true` is the clean field of the real plate at the RX plane, before the RX elements
  sample it and before any noise. `reconstructed` is the solver's own model, propagated
  the solver's own way. Outside the RX window the reconstruction is unconstrained, so
  only its in-window numbers are held to the measurement.
- `match` compares the two inside the RX window (`corr`, `rel-L2 vs true`,
  `power ratio`), and the reconstruction with what the solver was actually given to fit
  (`rel-L2 vs target`: the measurement as the RX elements sampled it, noise included).
  With an on-grid receiver (`rx_aperture.grid_stride`, below) the target exists at the
  element samples only, so `rel-L2 vs target` is taken there; the comparison with the
  true field still runs over every grid point in the window, which also covers the
  points between elements, where nothing constrained the fit.
- `loss` sets the solver's final loss against two references evaluated with the same
  expression: `true_plate`, what the real plate's phase scores under the solver's
  model, and `zero_model`, what radiating nothing scores. `ratio` is
  `solver / true_plate`. The true plate does not score zero, because the target is the
  field as the RX elements sampled (and noised) it; with a sparse or noisy receiver a
  ratio below 1 is the ordinary case. A ratio well above 1 means the solve stopped
  short of the true plate: cut off by `max_iters` (`stop` says so), or converged in
  another basin (see Initialization, under "Multiple frequencies"). An on-grid
  receiver changes the reference: its target is the true field at the elements, so on
  a **noiseless** run the true plate scores almost nothing, far below the loss at
  which the solver stops (its convergence test or its iteration cap), and the ratio
  is large for a solve that sits in the right basin. Read it next to the match
  figures there. With receiver noise the true plate's loss is the noise it cannot
  fit, and a **single-tone** solve reaches that floor: the ratio is about 1 (0.95,
  0.93 and 1.01 at 30 dB on the caustic, bessel and focus base configs, under the
  −1e-8 stop they had before). Several tones get there or not depending on the start
  (see Initialization, under "Multiple frequencies").

The block is logged, not written to `run.json`; `grid-search-mgs --true-mgs` logs the
same block for its baseline solve. The numbers come from `src/rice_bend/landing.py`.

#### Beam types

The emitted beam is selected by the `tx_aperture.beam.type` field:

| `type` | Parameters | Beam |
| --- | --- | --- |
| `caustic` | `trajectory: [a, b, c]` | accelerating beam following `x(d) = a·d² + b·d + c`, `d` the distance travelled from the TX |
| `directional` | `steer_angle_deg` | plane wave steered off the −Z axis |
| `bessel` | `cone_angle_deg`, optional `steer_angle_deg` | two plane-wave halves crossing at the cone angle about the aperture centre |
| `focus` | `focal_length`, optional `steer_angle_deg` | lens converging on the point `focal_length` from the aperture centre |

A positive `steer_angle_deg` steers toward **−x** for all three types that take one. It
tilts the Bessel axis, and for `focus` it is the angle of the ray from the aperture
centre to the focal point, which sits at `(x_c − F·sin θ, z_tx − F·cos θ)`. The `bessel`
and `focus` beams are laid out about the centre of `[x_min, x_max]`, so they move with
the TX window. An unknown key in the `beam` block is an error.

The scene is 2D, so the `bessel` beam is the two-wave counterpart of J0: **equal-height
fringes** of period `λ / (2·sin(cone))` rather than one dominant core. The two halves
overlap down to `(W/2) / tan(cone)` below a plate of width `W`, and the overlap is
widest (`W/2`) at half that range. The `focus` spot is about `λ·F / W` wide.

#### Configs

`configs/` holds the eight files that the 300 GHz campaign, the defaults and the test
harness need. Everything else is in `configs/archive/`.

| Config | Kept for | Tones (GHz) | Receiver | Start | Candidate grid |
| --- | --- | --- | --- | --- | --- |
| `scenario_caustic_hit_1f_grid2l.yml` | **campaign base config**, caustic beam. Also the `mgs-study` default, characterize check 6 and three parity slices | 300 | on-grid λ/2 (`grid_stride: 2`) | `init: backprop` | 327×151 (49,377) |
| `scenario_bessel_hit_1f_grid2l.yml` | **campaign base config**, bessel beam | 300 | on-grid λ/2 | `init: backprop` | 327×151 |
| `scenario_focus_hit_1f_grid2l.yml` | **campaign base config**, focus beam | 300 | on-grid λ/2 | `init: backprop` | 327×151 |
| `scenario_caustic_hit.yml` | characterize check 2, the net's full-resolution single solve | ten-tone comb, 200 to 380 | on-grid λ/2 | legacy start | 64×48 (3,072) |
| `scenario_caustic_hit_pm5.yml` | the input of `rust/python/bench.py`, which times the Rust engine's own warm start | 285, 300, 315 (±5 %) | on-grid λ/2 | legacy start | 327×151 |
| `scenario_caustic_hit_sparse.yml` | characterize checks 3 and 4 and two parity slices | ten-tone comb | **interpolated** λ/2 (`dx: 0.000499654`) | legacy start | 12×12 (144) |
| `tiny_check.yml` | characterize checks 1 and 5, two parity slices, the Rust golden-vector generator. A coarse copy of the caustic scene, not a research scenario | 280, 300 | interpolated, λ/20 of each tone (`dx: null`) | legacy start | 3×3 |
| `caustic_config.yml` | the `mgs` and `grid-search-mgs` default: a commented general config on a larger scene | none listed (300 by default) | interpolated, λ/20 of each tone | legacy start | 20×20 |

The three base configs are the ones to run and to copy from. They share the scene, the
TX window, the RX array, the noise, the seeds and the candidate grid, and differ in the
beam block alone, so the true TX location is the same and the residual maps compare
like for like. They start every solve from the data (`gerchberg_saxton.init: backprop`,
described under `grid-search-mgs`) and run every solve under one setting:
`gerchberg_saxton.max_iters: 3000` and `convergence_threshold: -1.0e-10`, with no
`grid_search.gs_overrides`, so each sweep candidate gets exactly one solve, under the
same solver settings as a plain `mgs` run, `--true-mgs` and `scripts/true_tx_table.py`
(which solve at the real TX, not at a grid point). The other
five keep the legacy start (`init` left at its default, `warm_start`). For the three
scenario configs and `tiny_check.yml`
that is what they are kept for: the characterize checks on them pin the solver the
earlier results were made with, and the Rust benchmark times the engine's own warm
start. `caustic_config.yml` is unchanged.

`configs/archive/` holds the 17 configs of earlier work: the general configs of the
other three beam types (`configs/archive/{directional,bessel,focus}_config.yml`), the
beam-versus-RX geometry scenarios (`scenario_directional_{miss,larger,smaller_hit}`,
`scenario_caustic_{miss,partial}`, `scenario_{bessel,focus}_hit`), the tone and pitch
variants of the caustic hit (`scenario_caustic_hit_{1f,pm10,lambda2,lambda4}`) and the
λ/20 controls of the base configs
(`scenario_{caustic,bessel,focus}_hit_1f_grid2l_lambda20`). They are not maintained:
they keep the interpolated or λ/20 receiver and the legacy start, and their header
comments describe the tree as it was. They still load, and they run when passed by
path: `--config configs/archive/<name>`.

The six `scenario_*` configs share one 300 GHz setup (`tiny_check.yml` and
`caustic_config.yml` do not follow it):

- **Scene.** One symmetric scene (`x[-0.3,0.3] z[0,0.85]`; `rs.rs` convolves a centered
  kernel, so an off-center scene would shift the beam), a 0.072 m TX plate at
  `z = 0.3` centred on `x = -0.101`, and a 0.04 m RX window centred on `x = 0`.
- **Tones.** The base configs list a single tone, 300 GHz; `mgs-study` builds the
  combs of a `frequency` series about it. `scenario_caustic_hit` and `_sparse` list a
  ten-tone comb, 200 to 380 GHz in 20 GHz steps, whose centre (by value) is 300 GHz,
  and `_pm5` lists the ±5 % comb.
- **RX.** An 80-element λ/2 array, the same physical array for every tone, so it is
  λ/3 at 200 GHz and 0.63 λ at 380 GHz. Where its elements sit is "The receiver",
  below.
- **TX.** The plate is sampled at λ/20: `tx_aperture.dx: 0.0000499654` m, 1440 samples
  across the 0.072 m plate.
- **Solver grid.** `sim_scene.spacing` is 0.25 mm, λ/4 at 300 GHz.
- **Noise.** `noise: {snr_db: 30.0, seed: 0}`; `--snr-db off` gives the noiseless run
  (see "Receiver noise (SNR)").
- **Candidate grid.** The 327×151 grid (49,377 candidates) has ≈ 2 mm ≈ 2λ cells at
  300 GHz (z step 1.99 mm, x step 2.00 mm). It is a nested refinement of the 164×76
  grid the fine configs used before the retarget: every second point in each axis.

#### The receiver

The receive aperture is configured by the `rx_aperture` block
(`x_center` + `width`); shrink `width` for a smaller receiver. Where the elements sit
inside the window is set by one of two fields. They are mutually exclusive: setting
both is a config error, and so is an unknown key in the block.

- `rx_aperture.dx` is the element pitch in metres, on an axis of the receiver's own: a
  number is one fixed physical array, the same for every tone, while `null` means λ/20
  of each tone at runtime. The elements do not line up with the solver grid, so the
  RX-plane field is linearly interpolated onto them, and their samples are linearly
  interpolated back onto the grid, where the solver's loss is scored at every grid
  point in the window.
- `rx_aperture.grid_stride` puts the elements **on** the solver grid: every
  `grid_stride`-th scene x sample inside the window, centred in it, the same set for
  every tone. It is an integer of at least 1 and has to be written as one: `true`, `2.0`
  and `"2"` are config errors. The field is read at those samples and the loss is scored
  at those samples alone; the weighting is zero at every other grid point, including the
  ones inside the window that carry no element. Nothing is interpolated. The pitch is
  `grid_stride` scene x steps, so it follows `sim_scene.spacing` rather than a number in
  metres. Each element stands for `grid_stride` grid samples, and its weight carries a
  factor `√grid_stride` so that the loss stays the same sum over the window that a dense
  receiver gives: residuals remain comparable between the two receivers, and
  `convergence_threshold` keeps its meaning.

With both left `null` the receiver is the `dx` one at λ/20. Which kept config uses
which:

| Receiver | Configs | Elements across the 0.04 m window |
| --- | --- | --- |
| on-grid, `grid_stride: 2` | the three base configs, `scenario_caustic_hit.yml`, `scenario_caustic_hit_pm5.yml` | every second one of the 160 grid samples in the window: 80, at a 0.5002 mm pitch (λ/2 at 300 GHz to 0.1 %) |
| interpolated at a fixed pitch, `dx: 0.000499654` (λ/2 at 300 GHz) | `scenario_caustic_hit_sparse.yml` | `sampled_axis` turns the nominal pitch into `int(width / dx)` elements: 80, at a realized 0.506 mm |
| interpolated at λ/20 of each tone, `dx: null` | `tiny_check.yml`, `caustic_config.yml` | 800 at 300 GHz, 747 at 280 GHz (`caustic_config.yml` has a wider, 0.065 m window) |

The experimental `.mat` path goes through the interpolated receiver too.
`scenario_caustic_hit_sparse.yml` stays on it on purpose: it is the only noisy run of
that path in `scripts/characterize.sh`, and the path is still in use, by the bench
captures and by `tiny_check.yml`.

Known limitation of the interpolated (`dx`) receiver at a λ/2 pitch: the measurement
passes through two linear interpolations of a complex field. First the RX-plane field
on the 0.25 mm grid is interpolated onto the elements, which fall between grid points;
the element samples come out about −31 dB off the true field at 300 GHz. Then the 80
samples are interpolated back onto the grid, and the loss is scored at all 160 grid
points in the window. The field's phase turns about 60° from one element to the next,
so the straight-line fill between elements dips. The solver's target ends up about
−18 dB off the true RX-plane field at 300 GHz (relative error power inside the RX
window, noiseless) and carries about 20 % less power. That is above the 30 dB noise
floor, and it grows with frequency across the comb: −24.6 dB at 200 GHz, −13.8 dB at
380 GHz on `scenario_caustic_hit_sparse`. At λ/20 (`dx: null`) the same round trip
costs little, because the elements are denser than the grid: about −42 to −51 dB
depending on the beam (caustic −45.0, bessel −41.7, focus −51.1, measured on the
archived `_lambda20` controls).

`rx_aperture.grid_stride` is the receiver with neither interpolation: the elements are
grid samples, read directly and scored alone, so the solver's target is the field the
elements sampled (plus the receiver noise) and nothing else.

What that does to localisation was measured on the full 327×151 grid, for a single
tone at 300 GHz, when the on-grid receiver was added: pilot runs from the legacy
start, under the sweep's 800-iteration cap. Each cell is the
top candidate's distance from the true TX (mm) / the mean distance of the top
candidates (mm) / the size of that tied set, which is the lowest-loss cell's connected
cluster of cells within 1.5× its loss (`analysis.json`):

| Receiver | caustic | bessel | focus |
| --- | --- | --- | --- |
| 30 dB, interpolated λ/2 (`dx`) | 4.6 / 22.9 / 230 | 50.6 / 54.6 / 250 | 21.0 / 45.0 / 223 |
| 30 dB, on-grid λ/2 (`grid_stride: 2`) | 2.65 / 13.3 / 66 | 2.65 / 21.5 / 43 | 2.65 / 9.6 / 18 |
| 30 dB, λ/20 control (`dx: null`) | 2.65 / 9.6 / 38 | 2.65 / 23.7 / 43 | 2.65 / 10.2 / 23 |
| noiseless, on-grid λ/2 | 2.65 / 2.65 / 1 | 2.65 / 3.6 / 2 | 2.65 / 2.65 / 1 |

At 30 dB the on-grid λ/2 array lands on the same top candidate as the λ/20 control for
all three beams, where the interpolated λ/2 array was 4.6 to 50.6 mm off. Two things
qualify these figures. The on-grid measurement is produced by the solver's own forward
model on the solver's own grid, so noise is the only mismatch left between the
measurement and what the solver can represent; that makes them somewhat optimistic for
a real array. And the 800-iteration cap of those runs is below what the on-grid solve
at the true TX needs from a random start (1488 / 1765 / 1424 iterations at 30 dB for
caustic / bessel / focus), so the on-grid rows rank candidates near the truth on
solves that were cut off. The base configs have since been given a start from the
data and one setting for every solve, the sweep's included (3000 iterations, stop
threshold −1e-10; see "Configs"). These are single-tone runs; they say nothing about
multi-tone localisation.

#### Scenario geometry

`scripts/verify_scenario.py CONFIG.yml [--freq HZ] [--expect-hit|--expect-miss]
[--expect-size larger|smaller]` propagates the real beam to the RX plane and reports
where/how it lands (HIT vs MISS, beam size vs RX), for validating a scenario's
geometry. It is forward-only: no solve, and no receiver noise even when the config asks
for some. `--freq` defaults to the config's centre frequency (then the 300 GHz project
default), and the one used is printed as `freq:`. Next to the peak (`beam_peak_x`), the
half-amplitude extent (`beam_FWHM`) and `rx_peak/global_peak` (HIT at 0.5 or above) it
prints `beam_centroid_x`, `energy_5_95=[lo,hi]` (the span holding the middle 90 % of
the energy), `target_x` (the beam's geometric aim) and `energy_frac_in_rx`, a plain
sum over the scene x-axis: the number `analysis.json` records as `fraction_in_window`.
Measured on the kept scenario configs at 300 GHz:

| Config | Beam | Peak x (m) | Half-amp extent (m) | rx_peak/peak | In-window energy | Class |
| --- | --- | --- | --- | --- | --- | --- |
| the four `scenario_caustic_hit*` configs | caustic | +0.0031 | 0.0178 | 1.000 | 0.840 | HIT; the peak is not centred, 3.9 mm short of the geometric caustic at +0.0070 |
| `scenario_bessel_hit_1f_grid2l` | bessel: the caustic's TX window, axis steered onto the RX | +0.0001 | 0.0445 (wider than the 0.04 m window) | 1.000 | 0.796 | HIT; five bright fringes in the window, about 6 mm apart |
| `scenario_focus_hit_1f_grid2l` | focus: the caustic's TX window, focal point on the RX | +0.0001 | 0.0055 | 1.000 | 0.977 | HIT, focal spot smaller than the RX |

The archived scenarios cover the other geometries: a directional beam that
near-misses, is larger than, or is smaller than its 0.065 m RX window, and the caustic
plate moved 115 mm (miss) and 22 mm (partial hit) to the right.

#### Experimental data and run output

To run against experimental `.mat` data, pass both paths:

```bash
mgs --rx-path RX.mat --heatmap-path HEATMAP.mat
```

With no `--freq` (and no `frequencies:` in the config) this path does not fall back to
300 GHz. It uses the bench carrier `experimental.lo_freq * experimental.trx_n`, 150 GHz
with the default bench constants (25 GHz × 6), because the captures are 150 GHz: the
reader takes the carrier from the FFT bin at `freq − lo_freq·trx_n`, and a carrier the
rig did not transmit points that read at a bin the signal is not in.

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

To run against experimental `.mat` captures, see `--rx-path` / `--heatmap-path` above;
the bench constants for that path (down-conversion chain, aperture edges, coordinate
origins, amplitude normalisation, scene margins) live in the config's `experimental:`
block.

### `grid-search-mgs`

Localizes a transmitter whose **location is unknown**. The real TX location/trajectory in
the config is used only to synthesize the shared RX measurement set (one measurement per
frequency); the tool then sweeps a speculative `(z, x_center)` grid, runs one joint MGS
phase retrieval at each hypothesized plane (a fixed-width aperture with uniform assumed
amplitude, one phase mask fitted against every frequency's measurement at once), and
saves each reconstruction as a **candidate beam**. To localize a TX from a measured
THz-TDS raster instead, see "Measured THz-TDS data" below.

```bash
grid-search-mgs                                 # full sweep from the config's grid_search block
grid-search-mgs --freq 280e9 300e9 320e9        # joint sweep: one solve per candidate across all three
grid-search-mgs --dry-run                       # enumerate the grid (counts + per-frequency validity), no MGS
grid-search-mgs --limit 10                      # quick partial run
grid-search-mgs -j 8                            # worker processes (default: all cores; 1 = serial)
grid-search-mgs --engine rust                   # solve with the Rust engine (see "Solver engines")
grid-search-mgs -o results/my_run               # name the run directory outright
grid-search-mgs --scenes                        # one PNG per candidate beam (scenes/) + an averaged scene
grid-search-mgs --anim                          # animate the candidate beams -> candidate_beams.mp4 (ffmpeg)
grid-search-mgs --surface-anim                  # orbit each residual surface -> residual_surface_orbit.mp4 (ffmpeg)
grid-search-mgs --true-mgs                      # also recompute the baseline MGS run at the KNOWN TX
grid-search-mgs --replot results/grid_search    # re-plot a saved run (joint or legacy layout)
grid-search-mgs --replot results/grid_search --scenes --anim   # standalone scenes + animation for a saved run
grid-search-mgs --replot results/grid_search --plot-window -0.15 0.15 0 0.4 --scenes   # re-window a saved run's plots
```

The residual heatmap, surface and scatter (with their high-contrast twins) are always
written. One rule sets everything else: **cheap plots always run; anything that
re-solves MGS, renders per-candidate PNGs or encodes video is opt-in** (`--true-mgs`,
`--scenes`, `--anim`, `--surface-anim`). A fresh run and a `--replot` of it therefore
produce the same set of files.

**Plot window.** By default the plots show the whole candidate grid and the whole
scene. A top-level `plot_window: {x_min, x_max, z_min, z_max}` in the config (metres, in
the solver frame: RX centre at (0, 0), TX above at +z), or `--plot-window XMIN XMAX
ZMIN ZMAX` (which overrides it), limits them to that rectangle. It is a view only: the
search, the manifest's numbers and `analysis.json` do not change.

- **Residual heatmap, scatter, surface (and `_hc`).** Cells outside the window are not
  drawn and do not set the colour limits, and the surface is built from the cells inside
  it. The heatmap shows exactly the window at equal aspect (1 m of x = 1 m of z). The
  scatter's axes are a distance and a residual, so it keeps its shape and plots the
  in-window candidates only. The true TX, best-candidate and top-candidate markers come
  from the whole grid; one outside the window is named in the legend with its position,
  and the title counts the candidates left out.
- **`--scenes`.** Every scene panel (the average's too) shows exactly the window at equal
  aspect, with the same number of z planes spread over the window's z range. The beams
  are still propagated across the whole scene width, then cut to the window, and the
  phase/amplitude panels span the window's x range. A candidate whose TX lies outside the
  window says so in its title. `--anim`, `--true-mgs` and the multi-frequency extras
  always show the whole grid / scene.
- **Recording.** A run saved with a window records it in `candidate_beams.json` as
  `plot_window` (a run without one has no such key), and a `--replot` draws with the
  recorded window unless `--plot-window` is given. `--replot ... --plot-window` is the
  only way to re-window a saved run. It does not rewrite the manifest, so pass it again on
  the next replot. Bad values (max ≤ min, a non-number, not 4 numbers) are refused.

Configure the sweep in the `grid_search` block of the config: the `z` / `x_center` sweeps,
the assumed aperture `width` (its `dx` is provenance-only — candidates are sampled on the
scene grid), a fixed `seed` (reused across candidates so residuals are comparable),
and `gs_overrides.max_iters` for a cheaper search (the campaign base configs set none,
so the sweep solves under `gerchberg_saxton`'s own cap). Every usable candidate is
solved exactly once, from `gerchberg_saxton.init`. An unknown key in `grid_search`,
`gs_overrides` or `gerchberg_saxton` is a config error, so a misspelt `init` cannot
silently run the legacy path. Each run is saved to `<output_dir>/<run_name>/` — the
run name comes from the config's `output.run_name` (fallback `grid_search`), e.g.
`results/scenario_caustic_hit/`, or `-o/--out` names the directory outright.

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
- `residual_heatmap_maxiters.png` — the high-contrast heatmap again, with every candidate
  whose solve hit `max_iters` (`stop_reason`) hatched and counted in the legend against the
  run's effective limit (skipped on `--replot` of a manifest without stop reasons).
- `residual_surface.png` — the same grid as relief instead of colour: height is the
  **inverted** residual, so the better a candidate fits, the higher the surface stands
  and the best fit becomes a peak over the TX. The flat heatmap saturates (on a dense
  caustic sweep made before the 300 GHz retarget, over half the cells sat within 1% of
  the maximum), which hides how deep the basin actually goes. The z-axis is labelled in
  real residual values, the flat heatmap is projected on the floor beneath the surface,
  and the true TX and best candidate carry full-height marker poles so neither can be
  hidden behind the terrain.
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
  as plain `mgs` would produce. Unlike the residual plots this recomputes one full MGS solve, and
  it logs that solve's `LANDING` block (see `mgs`). A run with `gerchberg_saxton.amplitude:
  descend` or `level` writes `true_mgs.json` / `true_mgs.npz` instead (see "Descending the
  amplitude").
- `scenes/cand_####.png` + `scene_average.png` — (with `--scenes`) one 4-panel PNG per candidate
  (candidate beam, real beam at the same color scale, candidate aperture phase, candidate aperture
  amplitude; scenes mark the RX aperture in red and the TX aperture in blue), plus a single plot
  averaging every candidate beam. Use `--scene-top N` to render only the N lowest-residual candidates
  (the average then spans exactly those). A measured run gets a 3-panel PNG instead, with no
  real-beam panel (see "Measured THz-TDS data").
- `residual_surface_orbit.mp4` / `_hc.mp4` — (with `--surface-anim`) the residual surface orbited
  through a full turn (needs ffmpeg). A still is fixed to one viewing angle, and which angle reads
  best depends on where the basin lands, so this is the way past a peak hidden behind a ridge.
  Opt-in because it adds a full render per video per run directory to a `--replot` that otherwise
  only redraws stills (~12 s per video, against a replot that otherwise finished in seconds, when
  timed before the 300 GHz retarget).
- `candidate_beams.mp4` — (with `--anim`) an animation sweeping the candidate beams, one frame per
  candidate re-illuminating the scene (needs ffmpeg). Every frame is held in memory at once to fix
  a shared colour scale, so `--anim` renders the 100 lowest-residual candidates by default;
  `--scene-top N` overrides that, and warns past 400 frames.

Lower residual = better data fit = more likely TX location, which is the input to the
candidate-ranking step. On a headless machine set `MPLBACKEND=Agg`.

`--jobs/-j` sets the worker-process count for the sweep and scene/animation rendering
(default: all cores; `1` = serial). Results do not depend on it — every candidate uses
the same fixed seed, and results are assembled by input index, so the manifest and
every candidate `.npz` are byte-identical whatever `-j` you pass.

#### Solver engines

`--engine {python,rust}` (on `grid-search-mgs` and `mgs-study`, or
`gerchberg_saxton.engine` in the config) picks who runs each candidate's solve.
`python` is the default and the reference implementation. `rust` runs the same
algorithm in the `rice_bend_core` extension (build it per Install). It is checked
against python by `rust/python/parity.py` and `rust/python/compare_runs.py`. As
measured before the 300 GHz retarget (the tables in `rust/README.md` are from the
150 GHz configs and have not been re-run), it was about 5× faster and agreed far
inside the parity bar: per-candidate losses within ~1e-12, with identical argmin and
top candidates.
The rest of the run (measurement, persistence, analysis, plots) is python either way.
Solves that record GS history (single-run `mgs`, `--true-mgs`) always use python.
With `init: backprop` the start itself (the back-propagated phase and, for several
tones, the offset scan) is computed in python for either engine, and the engine runs
the descent from it; the legacy warm start has a twin inside the Rust engine.
The engine a run used is recorded in its `config_snapshot.json`. Asking for `rust`
without the extension built fails immediately; it never falls back to python.

#### Multiple frequencies

To solve the same scene against several carrier frequencies at once, list them in the
config's top-level `frequencies` block (Hz), or pass several values to `--freq` (which
overrides the config):

```yaml
# null / absent -> single --freq, default 300e9
frequencies: [200.0e9, 220.0e9, 240.0e9, 260.0e9, 280.0e9, 300.0e9, 320.0e9, 340.0e9, 360.0e9, 380.0e9]
```

With more than one frequency each candidate gets **one joint solve**: a single shared
aperture profile is fitted against every frequency's measurement simultaneously, and the
residual the solver minimizes — and the plots rank by — is the **mean of the per-frequency
losses**. The measurement set is still per-frequency: each frequency synthesizes its own
RX field with its own TX beam phase (∝ wavenumber) and, when `rx_aperture.dx` is null
(λ/20 of each tone), its own RX element count. The scenario configs set a fixed pitch
or a `grid_stride` instead, so there every tone is sampled by the same array.

##### The physical model

The solver reconstructs a *plate with one shape*: the unknown ψ(x) is the aperture's
delay/optical-path profile, expressed as phase at a reference frequency, and each
frequency's phase is `(f / f_ref) · ψ(x)` — the way a real plate's phase scales with
wavenumber. Every simulated beam type is exactly this kind of source (the caustic plate,
the steered ramp, the axicon and the lens all bake the wavenumber into their phase), so
the true TX can fit
every frequency simultaneously and extra bandwidth adds constraint. It also resolves the
single-frequency 2π ambiguity: only the true delay is consistent with every frequency at
once (the multi-wavelength interferometry effect). The reference frequency is the
centre-by-value of the run's frequency list — the same convention as the scene views and
the 3D-diff baseline — and is recorded as `ref_freq_hz` (manifest `gs` block /
`run.json` `gs_result`). Every candidate in a sweep shares the run-level reference,
including candidates solved on a valid-subset of frequencies, so their profiles stay in
the same units. The stored reconstruction (`aper_profile`, `gs_tx_aper_profile`) is the
aperture *field at the reference frequency*; `mgs-animate --freq-index` rescales it to
show the phase the plate presents at any selected frequency, and candidate scene renders
are exact at the default centre-frequency view (the usual monochromatic approximation at
other `--scene-freq` choices).

**Initialization** (`gerchberg_saxton.init: random | warm_start | backprop`): a
multi-frequency solve must recover the profile in an *absolute* sense — the 2π
degeneracy that makes single-frequency retrieval easy is exactly what the
multi-frequency constraint breaks — and plain gradient descent from a random start
gets stuck in a wrong basin (measured before the 300 GHz retarget: ~100× above the
reachable floor). Where the descent starts is therefore part of the method:

- `random` is the seeded random draw, at any number of frequencies.
- `warm_start` is the default and the **legacy** start. It runs the multi-wavelength
  initialization inside every multi-frequency solve: the channel nearest the reference
  frequency is solved alone from the random draw (its degenerate landscape is easy —
  that recovers the SHAPE), the result is unwrapped over the support (collapsing the
  ambiguity to one scalar), and that absolute offset is scanned over
  `2π·f_ref/Δf_min` against the joint loss before the joint descent starts. At a
  single frequency it is the random start.
- `backprop` takes the start from the **data**, at any number of frequencies. The
  reference tone's weighted measurement is back-propagated to the candidate plane with
  the solver's own adjoint kernel. Its phase is unwrapped where the back-propagated
  amplitude is at least 10 % of its peak, which is the part of the plate that lights
  the RX window, and continued at constant slope outside it, where the data says
  nothing. With several tones the absolute offset is then scanned over the comb's
  **true** common period, and the ordinary descent runs from there. There is no
  reference-frequency solve.

`backprop` exists because the legacy multi-tone start has two measured defects:

1. **Its scan range is not always a period.** `2π·f_ref/Δf_min` is a period of the
   joint loss only when `f_ref/Δf` is an integer. For the study's combs
   `(100−p):100:(100+p)` the true period is `2π·100/gcd(100, p)`: the legacy range is
   the period at ±1, 2, 5, 10 and 20 %, and a fraction of it at ±3, 7, 13 and 16 %.
   Those are the four bandwidths at which the focus beam fails: the offset that lines
   the tones up lies outside what was scanned. `backprop` scans the period that
   `mgs.comb_period` computes from the tone ratios as fractions,
   `2π·lcm(denominators)/gcd(numerators)`.
2. **Its reference solve is not unique.** Started from a random phase, a single-tone
   solve is not unique where part of the beam misses the RX window: nothing measured
   constrains the part of the plate that feeds the missing energy. The scaled copy of
   what that solve returns misses the other tones, and the joint descent stalls (the
   bessel beam). The back-propagated phase is read from the measurement instead.

Measured at the true TX on the on-grid receiver at 30 dB (noise seed 0), as the final
loss over the loss of the true plate's own phase, worst over the `frequency` study's
ten bandwidths:

| Beam | `warm_start` | `backprop` |
| --- | --- | --- |
| caustic | 1.66 | 1.11 |
| bessel | 165 | 1.00 |
| focus | 31.5 | 1.01 |

The data-driven start gets there in a half to a tenth of the iterations.

**Near-tied offsets.** A comb that almost repeats inside its period leaves two offsets
the scan cannot tell apart. The ±13 % comb (87:100:113) repeats after 100 cycles and
almost after 77, so two offsets score within about 1e-4 of each other, only one of
them lies in the basin the other tones agree with, and which of the two scores higher
depends on the noise draw. The offset is chosen from the 32 best local maxima of a
closed-form score, each polished by Newton steps, so the choice does not depend on
where the 0.02 rad scan grid falls. It is still the best-scoring offset, and the
descent cannot cross from one of the pair to the other: on the caustic config at
±13 % the solve ended at 2.1 to 2.3 times the true plate's loss for two of nine noise
seeds. That is a known limit of the single data-driven start, which every solve
(`mgs`, `--true-mgs`, each sweep candidate) runs.

**The legacy start is kept unchanged**, so that earlier runs can be reproduced:
`warm_start` is still the default, with its original scan range, on both engines, and
a config that does not set `init` solves exactly as it did before. The derivations
and the measurements behind all of this are in `src/rice_bend/mgs.py` (the notes above
`comb_period`, and the docstring of `_backprop_start`). `scripts/true_tx_table.py`
(see `mgs-study`) tabulates the solve at the true TX per bandwidth for a given config.

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
  list, e.g. 300 GHz) on a symmetric `viridis_r` scale — yellow = better than the
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
The 300 GHz retarget is such a change: the caustic generator now builds its plate by resampling
the unwrapped phase, so a caustic run saved before it gets its baseline re-solved against a plate
built differently from the one its residuals were measured on.
`schema_version` in `candidate_beams.json` distinguishes result sets (3 = joint; 4 =
joint, written by the since-removed refinement pass, whose extra keys the readers
ignore; 2 = the retired per-frequency layout).
`run.json` is 3.

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
Real-TX overlay, and rescales the captured plate profile to the phase it presents
at that frequency — so the estimate and the overlay are compared at the same
frequency, labelled with it. The loss panel always shows the joint curve.

How many frames each mode renders:

- `--mode phase` animates **every captured iteration**, so its frame count is set by
  `gerchberg_saxton.history_stride`. Capturing more enlarges `run.npz`; raise
  `mgs-animate --fps` to keep the video short. Every shipped config except
  `tiny_check.yml` (10) now uses `history_stride: 50` — at `1`, a default `mgs` run wrote
  576 MB of capture arrays and `--mode phase` rendered 10,000 frames rather than 200.
- `--mode scene` recomputes a full Rayleigh-Sommerfeld propagation per frame (expensive),
  so it **subsamples the captured iterations to ~60 frames by default** regardless of
  `history_stride`. Use `--frame-stride 1` for every captured iteration (slow), and
  `--z-stride` to trade scene resolution for speed.

### `mgs-study`

Runs a **parameter study**: a code-defined series of grid searches that varies one
parameter of a base config, measures each point's localization error (from the point's
`analysis.json`) and plots it against the study's x-axis.

```bash
mgs-study --study frequency                       # bandwidth series on the default base config (scenario_caustic_hit_1f_grid2l.yml)
mgs-study --study frequency --config configs/scenario_bessel_hit_1f_grid2l.yml --out results/study_frequency_bessel
mgs-study --study snr --config configs/scenario_caustic_hit_1f_grid2l.yml
mgs-study --study frequency --limit 10 --out results/study_smoke   # per-point candidate cap, for smoke tests
mgs-study --replot results/study_frequency_snr30dB                 # rebuild study.json + the plots, no solving
```

The default base config is the caustic campaign base config, because a study is a
series of full sweeps and should run them from a config with the on-grid receiver and
the data-driven start. The 300 GHz campaign is six studies, the two series below for
each `<beam>` of `caustic`, `bessel` and `focus`, each from that beam's base config and
into its own root:

```bash
mgs-study --study frequency --config configs/scenario_<beam>_hit_1f_grid2l.yml --engine rust -j 32 --out results/campaign_300ghz/frequency_<beam>
mgs-study --study snr --config configs/scenario_<beam>_hit_1f_grid2l.yml --engine rust -j 32 --out results/campaign_300ghz/snr_<beam>
```

The roots already under `results/campaign_300ghz/` were made with the 800-iteration
sweep plus the refinement pass that has since been removed (manifest schema 4; the pass
ran at every point of the frequency series and at 15 dB and above in the snr series,
and was skipped at 0, 5 and 10 dB). The current code cannot reproduce them, and the
resume guard below refuses to resume them: a re-run needs new `--out` roots. They
still replot.

| `--study` | Varies | Points |
| --- | --- | --- |
| `frequency` | bandwidth: a comb of both edges and the centre, ± % about the base config's centre frequency | 0 (the single tone), 1, 2, 3, 5, 7, 10, 13, 16, 20 % |
| `tx_shift` | a rigid TX translation (the window and the caustic trajectory move together, so caustic base configs only), solved with the ±5 % comb about the same centre and plotted against the energy received in the RX window | nine shifts, 0 to 130 mm |
| `snr` | receiver SNR, with whatever frequencies the base config lists | 0 to 40 dB in 5 dB steps |
| `snr_fine` | the same | 0 to 30 dB in 1 dB steps |

No carrier is hard-coded. The combs are centred on the base config's centre frequency
(the centre-by-value of its `frequencies`, else the 300 GHz default), so the same
series runs from any base config; with the shipped scenario configs the centre is
300 GHz and ±5 % is 285/300/315 GHz.

The study root holds `study.json` (written first and updated after every point, so an
interrupted study resumes), `configs/point_*.yml` (each point's effective config), one
ordinary grid-run directory per point, the study plot (`study_error_vs_bandwidth.png`,
`study_error_vs_energy.png` or `study_error_vs_snr.png`) and
`study_metrics_vs_ndof.png`. The default root is `<output_dir>/study_<study>`, plus an
`_snr<DB>dB` suffix when noise is on (see "Receiver noise (SNR)").

`study.json` is schema 9: it records the base config's `beam` block and
`center_freq_hz` at the top level and stores `x_label` with the centre filled in
(`bandwidth (± % of 300 GHz)`). Plot titles name the beam and the carrier, e.g.
`frequency study — caustic beam, 300 GHz — SNR 30 dB`. A schema-8 root (written while
the refinement pass existed) still replots; its points come back without their
`refine_status` / `n_refined`.

A point whose `candidate_beams.json` exists is not solved again, and its directory
name is all that keys it. So before anything is written, a pre-flight guard refuses
(exit 2) to resume a root whose saved points are not the ones this invocation would
solve: different frequencies, scene, TX (beam included), RX, candidate grid, solver
settings, a different `--limit`, another study, or different noise — or a point solved
with the removed refinement pass, whatever its settings. The default root names neither
the beam nor the carrier, so different beams or carriers need different `--out` roots.
A root solved before the retarget (150 GHz) is refused by the retargeted configs.
`--replot` is not guarded: such a root still replots from what its own points
recorded, and keeps its 150 GHz labels.

`scripts/true_tx_table.py CONFIG.yml --out FILE.md [--snr-db DB|off] [--noise-seed N]
[--max-iters N]` is the companion check for the `frequency`
series. A grid search can only rank the true location first if the solve *at* the true
location gets down to what the true plate scores, and whether a multi-tone joint solve
does depends on its start (see Initialization, under "Multiple frequencies"). For each
study bandwidth the script runs one solve at the true TX location with the study's
exact comb, from the config's own `init` and under its own cap and stop threshold,
and tabulates the solver's loss against the true plate's loss (the `LANDING` loss
numbers). With the base configs (no `gs_overrides`) it runs under the same solver
settings as the sweep's one solve per candidate; the true TX is not a grid point.

A ratio above 2 flags the row as `solver-limited`: a localization error at that
bandwidth reflects the solver, not the physics. On a noiseless run with an on-grid
receiver the flag column reads `n/a` on every row: the target is then the true field
at the elements, the true plate scores almost zero, and the ratio is not a basin test.
The ratios are still written; run with receiver noise for a ratio that tests the
basin. `--max-iters` overrides the iteration cap. The solve captures history, so it
runs on python whatever `gerchberg_saxton.engine` says. The script writes the
Markdown table to `FILE.md` and the same rows to `FILE.json` beside it.

### Receiver noise (SNR)

With `noise.snr_db: null`, the code default, the simulated RX measurement is exact
(infinite SNR). Setting an SNR adds complex white Gaussian noise to it, the way a
receiver would. Every `configs/scenario_*.yml` sets one (`noise: {snr_db: 30.0,
seed: 0}`), so the test scenarios run at 30 dB; `tiny_check.yml` and
`caustic_config.yml` stay noiseless. The model is one noise floor for every antenna
and every frequency, referenced to the brightest RX element:

```
P_pk    = max over tones f and RX elements n of |s_f[n]|^2      (clean field)
sigma^2 = P_pk / 10^(snr_db / 10)                               (same for all f, n)
n_f[n]  ~ CN(0, sigma^2), independent per (frequency, antenna)
```

The noise is added on the RX **element** axis, before the element samples are put on
the solver grid (interpolated onto it, or placed at their own grid points for an
on-grid receiver), and the error weighting is built from the noisy measurement. Noise
is **off** unless a config or a flag asks for it: with `snr_db: null` (or no `noise:`
block) no random number is drawn and every result is identical to a run made before
this option existed. `--snr-db off` does the same over a config that enables it.

Set it in the config's `noise:` block, or on the command line of `mgs`,
`grid-search-mgs` and `mgs-study` (the flags override the config):

```yaml
noise:
  snr_db: 30.0    # null -> noiseless
  seed: 0         # null -> drawn at runtime and recorded
```

```bash
mgs --config configs/tiny_check.yml --snr-db 20                     # one noisy solve
grid-search-mgs --config configs/tiny_check.yml --snr-db 20 --noise-seed 3 -o results/g20
grid-search-mgs --config configs/scenario_caustic_hit_sparse.yml --snr-db off -o results/sparse_noiseless   # force noiseless over a 30 dB config
mgs-study --study frequency --snr-db 20                             # every point at 20 dB
```

`--snr-db DB|off` takes a number of dB or `off`, which forces a noiseless run even when
the config enables noise. `--noise-seed INT` picks the realization. Invalid command-line
values (`nan`, `inf`, a negative seed) are rejected with exit code 2. The same values in
the `noise:` block, or a misspelled key there, fail config validation before anything
runs, so a typo cannot silently run noiseless. Write a negative SNR as
`--snr-db=-10`: argparse reads `--snr-db -1e1` as a flag.

What "SNR" means here, and three things to know before reading results:

1. **Nominal vs. realized.** The dB value is relative to the *peak* RX element. The mean
   per-antenna SNR is lower by the beam's peak-to-average ratio: on the interpolated
   receiver of `scenario_caustic_hit_sparse` with the ±5 % comb
   (`--freq 285e9 300e9 315e9`), the nominal 30 dB is about 24.9 dB per antenna. The
   interpolation onto the solver grid then averages some of the noise away, so the
   solver sees 25.2 to 26.4 dB there, depending on the tone (seed 0). Nothing is
   compensated. Both figures are logged and recorded per frequency (`mean_snr_db`,
   `effective_snr_db`). They count noise only: with the interpolated λ/2 array, the
   sampling error of the target is larger (see "The receiver" under `mgs`).
   With an on-grid receiver (`rx_aperture.grid_stride`) nothing is interpolated, so
   nothing is averaged: `effective_snr_db` is then the realized per-element SNR of the
   draw, the same quantity as `mean_snr_db` taken against the noise actually drawn
   instead of the nominal `sigma^2` (25.1 and 25.5 dB on
   `scenario_caustic_hit_1f_grid2l`, seed 0).
2. **The floor tracks the beam.** The peak is taken over the RX antennas, so a beam that
   walks off the window lowers the absolute noise floor with it. Every run sits at the
   configured SNR, but the absolute noise is not constant across geometries.
3. **Common random numbers.** Each tone's draw is keyed by its frequency value and the
   seed, never its position in the list. Runs that share a tone and a seed share that
   tone's normalized draw, scaled by their own `sigma`. SNR sweeps therefore vary
   smoothly, but their points are not independent realizations. Change `--noise-seed`
   for a different draw.

**What is saved.** `run.json` and `candidate_beams.json` carry a `noise` block: `snr_db`,
`seed` (a drawn seed is written back), `reference`, `peak_power`, `sigma2`,
`numpy_version`, and `per_freq` realized SNRs. It is `null` when noise is off and absent
in runs saved before this option existed. With noise on, `rx_aper_profile_NN` (both
files), `rx_field` / `error_weighting` in `measurement.npz` and `gs_target_rx_field` /
`gs_error_weighting` in `run.npz` are the **noisy** measurement the solver saw, and
`rx_aper_profile_clean_NN` holds the noiseless element profile beside it.
`rx_plane_row` stays noiseless, and so do the energy and N_E metrics in
`analysis.json`. `grid-search-mgs --replot DIR --true-mgs` restores the noise settings
from the manifest, so the baseline solve replays the same noisy measurement.
`--snr-db`/`--noise-seed` are ignored (with a warning) together with `--replot` on both
`grid-search-mgs` and `mgs-study`: a replot always uses the noise recorded in the saved
run. Residual and study plot titles gain a ` — SNR 20 dB` suffix.

**Studies.** On `mgs-study` the flags apply to every point of the series. With noise on
and no `--out`, the default root gains a suffix, so noisy and noiseless studies do not
collide: `study_frequency_snr30dB` from a scenario config as shipped,
`study_frequency_snr20dB` with `--snr-db 20`, plain `study_frequency` with
`--snr-db off`. Resuming a study root whose points were solved with different noise
settings is refused (exit 2), next to the experiment guard described under
`mgs-study`. `study.json` records the noise block and a per-point `snr_db`. With
`seed: null`, one seed is drawn for the whole study and recorded in `study.json`;
resuming the same root at the same SNR adopts that recorded seed.

The `snr` series sweeps the SNR itself: 0 to 40 dB in 5 dB steps, one grid search per
value, all with the same seed. Nothing else changes, so it solves with whatever
frequencies the base config lists, and the base config's own `snr_db` (30 dB in the
scenario configs) is overridden at every point. Its default root is `study_snr` (no
suffix), and it plots error against SNR (`study_error_vs_snr.png`). `--snr-db` cannot
be combined with it (exit 2) because the series sets every point's SNR, but
`--noise-seed` can:

```bash
mgs-study --study snr --config configs/scenario_caustic_hit_1f_grid2l.yml   # 300 GHz, 327x151 grid
```

**Engines and data.** The noise is added in Python before the solver engine is called,
so `--engine rust` needs nothing extra: both engines receive the identical noisy
arrays. The experimental `.mat` path ignores `snr_db` with a warning, because a real
capture already carries its own receiver noise.

### `topt-parser`

Turns a folder of Toptica THz-TDS pulse traces into **one spectra CSV**, the input the
measured grid search reads (see "Measured THz-TDS data" below). It is the one place a
pulse is processed: `grid-search-mgs` never opens a trace.

```bash
topt-parser ~/Desktop/experiments -j 24     # -> ~/Desktop/experiments/topt_spectra.csv
topt-parser FOLDER -o spectra.csv --force   # another path; overwrite an existing file
```

- **Input.** Every `{idx}-z{z}-x{x}.pulse.csv` in `FOLDER`. `z` and `x` are the stage
  position in file units of `--unit-mm` (default 0.1 mm, so `z1770` is 177 mm), and `idx`
  is an acquisition counter that is ignored. Each trace starts with the header
  `Time_abs/ps, Signal 1/nA, Signal 2/nA`. Only `Signal 1` is used, and the time column
  is **absolute**: the window start moves from row to row. Other files (`probe-*`,
  `asdf*`, `af`) are skipped and listed; two traces at one position are an error.
- **Output.** `FOLDER/topt_spectra.csv` by default, `-o` for another path. An existing
  file is refused unless `--force`. The CSV is written to a temporary file and renamed
  into place, so an interrupted run leaves no partial file.
- **`-j N`** reads the traces in N worker processes (default 1). The output does not
  depend on it. `--fmax HZ` sets the last column (default 1e12).

**The format.** Plain comma-separated text, one row per trace, sorted by Z, then X:

```
X,Z,0GHz,5GHz,10GHz,...,1000GHz
<X mm>,<Z mm>,<V(0)>,<V(5 GHz)>,<V(10 GHz)>,...,<V(1000 GHz)>
```

`X` and `Z` are the stage position in **mm**, written like `177` or `200.5`. Each cell
is `V(f)` as a Python complex literal at 17 significant digits (`12.3-0.45j`), which
`complex()` reads back to the identical float64 pair. The first raster gives 40,501 rows,
201 frequency columns (0, 5, …, 1000 GHz) and a file of about 320 MB.

**The frame.** Every trace of the first raster is 4001 samples 0.05 ps apart, the
window's closing endpoint included. The parser **drops the last sample**, so each FFT
frame is N = 4000 samples, exactly 200 ps, and its bins land exactly on multiples of
**5 GHz** (300 GHz is bin 60, 1 THz bin 200). With all 4001 samples they would be
4.99875 GHz apart, and no carrier would land on one. Every trace must have the same
sample count and step, on a uniform time grid. The traces that do not are all named in
one error, and nothing is written.

**The cell.** For a trace with absolute sample times `t_n = t0 + n*dt` and Signal 1
samples `s_n`:

```
V(f_k) = conj( exp(-2j*pi*f_k*t0) * rfft(s[:4000])[k] ) = conj( sum_n s_n * exp(-2j*pi*f_k*t_n) )
f_k    = k / (4000 * dt)
```

- **Absolute time.** The `exp(-2j*pi*f*t0)` factor puts every phase on the same t = 0,
  so rows with different window starts stay phase-consistent.
- **The conjugate.** A pulse delayed by τ comes out with phase **+2πfτ**. That is the
  `exp(+ikr)` (`exp(−iωt)`) convention of the RS kernel in `rs.py`, in which the
  simulated measurement is built.
- **No scaling.** There is no dt factor, gate, taper or amplitude scale. The unit is nA
  summed over the samples, and the measured config's `amplitude_scale` absorbs it.
- **`0GHz`** is the plain sum of the 4000 samples: real, with no time correction (f =
  0). It is kept as a record only, because the grid search refuses to solve at 0 Hz,
  which has no wavelength.

**Reading it back.** In plain Python, with the standard `csv` module and `complex()`:

```python
import csv
from pathlib import Path

with open(Path("~/Desktop/experiments/topt_spectra.csv").expanduser(), newline="") as f:
    rows = csv.reader(f)
    header = next(rows)                 # ['X', 'Z', '0GHz', '5GHz', ..., '1000GHz']
    col = header.index("300GHz")
    for row in rows:
        x_mm, z_mm = float(row[0]), float(row[1])
        v = complex(row[col])           # V(300 GHz) at (x_mm, z_mm)
```

The grid search's own reader returns one row at the frequencies asked for. It streams
the file once and parses only that row's cells:

```python
from rice_bend.topt_parser import read_spectra_csv

r = read_spectra_csv(Path("~/Desktop/experiments/topt_spectra.csv").expanduser(),
                     177.0, [300e9])
r.x_mm, r.values[:, 0]                  # row Z = 177 mm: ascending X (mm), V(300 GHz)
```

### `topt-heatmap`

Plots one frequency column of a `topt-parser` spectra CSV as a heatmap of the **log
magnitude**, over the whole raster:

```bash
topt-heatmap ~/Desktop/experiments/topt_spectra.csv --freq 300GHz          # -> ~/Desktop/experiments/topt_spectra_300GHz.png
topt-heatmap CSV --freq 100GHz -o map.png --range 40                        # another path; a 40 dB colour range
topt-heatmap CSV --freq 300GHz --config configs/measured/measured_rx177.yml # draw that config's TX and RX on top
```

- **`--freq`** must be a column of the CSV (to 1 Hz). Give it in Hz (`300e9`) or with
  a unit (`300GHz`, `0.3THz`, `3e11Hz`). A frequency with no column is an error that
  names the nearest columns, as in the grid search. `0GHz` (DC) is allowed here.
- **The scale.** dB re the map's peak, `20*log10(|V| / max|V|)`, with the max taken
  over every cell of the map. The peak cell is 0 dB, and a cell at half its amplitude
  is −6.02 dB. The colour floor is `−--range` dB (default 60). A cell below it, or at
  `|V| = 0`, gets the floor colour. A (Z, X) with no row in the CSV is drawn grey. The
  colour bar notes the peak `|V|` and where it is.
- **The axes.** The CSV's stage `X` (across) and `Z` (up) in mm, at equal aspect, so
  1 mm of X is as long as 1 mm of Z. Each cell reaches halfway to its neighbours, and
  half a step past the outer samples.
- **`--config`** takes a measured config (`base_config` is honoured) and draws its
  `measured.tx_aperture` (cyan, dashed) and `measured.rx_aperture` (red) as written,
  in stage mm. The axes grow to take in an aperture outside the scan: the TX of
  `measured_rx177.yml` is at z = −65 mm, below the first row. A config without a
  `measured:` block is refused. A `data_file` other than the CSV plotted is only
  noted.
- **Output.** `<csv stem>_<freq>.png` beside the CSV (`topt_spectra_300GHz.png`) by
  default, `-o` for another path. An existing PNG is overwritten. `--show` also opens
  the figure in a window.

From Python, `read_spectra_map` streams the file once and returns the one column as a
(Z, X) grid (NaN where there is no row), and `plot_heatmap` draws it:

```python
from pathlib import Path

from rice_bend.topt_parser import read_spectra_map
from rice_bend.topt_heatmap import measured_overlays, plot_heatmap

m = read_spectra_map(Path("~/Desktop/experiments/topt_spectra.csv").expanduser(), 300e9)
m.x_mm, m.z_mm, m.values                # (401,), (101,), (101, 401) complex
overlays, _ = measured_overlays(Path("configs/measured/measured_rx177.yml"))
plot_heatmap(m, Path("map.png"), db_range=60, overlays=overlays)
```

### Measured THz-TDS data

`grid-search-mgs` can localize a TX from a measured THz-TDS raster instead of a
simulated measurement. One constant-z row of the raster is the receiver, and each of
its positions gives one element's complex value per frequency. The search then runs
exactly as on the on-grid simulated configs, and the result is scored against the TX
position measured on the bench. The TX **beam** is the unknown: nothing is synthesized
from it, and nothing reports it. The output is **localization only**.

**The data.** The spectra CSV that `topt-parser` writes from the folder of pulse traces
(see "`topt-parser`" above). Parse the folder once, then point the config at the file:

```bash
topt-parser ~/Desktop/experiments -j 24     # writes ~/Desktop/experiments/topt_spectra.csv
```

The grid search reads that one file and never opens a pulse trace. It takes every
row's Z, to check the RX row against, and the values of the RX row alone, at the run's
frequencies.

**The config.** `configs/measured/` holds `base.yml`, which is shared, and one thin
file per acquisition laid over it:

```yaml
# configs/measured/measured_rx177.yml
base_config: base.yml           # relative to this file; deep-merged underneath it
measured:
  data_file: ~/Desktop/experiments/topt_spectra.csv                # topt-parser's output
  tx_aperture: {left: {x: 85,  z: -65}, right: {x: 185, z: -65}}   # stage frame, mm
  rx_aperture: {left: {x: 175, z: 177}, right: {x: 225, z: 177}}   # z: a row of the CSV
  amplitude_scale: 0.0081728    # see "amplitude_scale" below
output: {run_name: measured_rx177}
```

- **`data_file`.** The spectra CSV; `~` is expanded, and a relative path is relative to
  the working directory. A config that still sets the old `data_dir` (or
  `file_unit_mm`) is refused with what to do instead: run `topt-parser <that folder>`
  and set `data_file` to its output. The file-name unit is now `topt-parser --unit-mm`.
- **Points.** Every point is in the **stage frame, in mm**, with named `x` / `z` keys,
  because the order of a pair is easy to swap. The CSV's `X` and `Z` are in the same
  stage mm. The two points of an aperture must share one `z` (a tilted aperture is
  refused). Their order along `x` does not matter.
- **`base_config`.** The path is relative to the file's own directory (`~` expanded).
  The file is deep-merged over that base: mappings merge key by key, and a list, scalar
  or null replaces the base's value whole. So any other block (`grid_search`,
  `gerchberg_saxton`, `frequencies`, `output`, ...) can be overridden per acquisition.
  It is one level only: a base that names a base of its own is refused. Files without
  the key load exactly as before.
- **`base.yml`.** The field is x −150…150 mm by z 0…400 mm in the solver frame (RX
  centre at the bottom middle), and it is the `plot_window` of every measured run. The
  scene is x ±0.3 m (the propagation needs the lateral margin) and z 0–0.40 m at
  0.25 mm with `exact_step: true`. The candidate grid covers the whole field, z 0–0.40 m
  by x_center −0.15–0.15 m at exactly 2 mm in both: 201×151 = 30,351 grid points. The
  z = 0 row (the RX plane itself) is skipped automatically, which leaves **30,200
  candidates**. Runs before 2026-10-19's extension searched z 0.15–0.40 m only (126×151 =
  19,026). The solver and the single 300 GHz tone are the campaign's
  (`scenario_caustic_hit_1f_grid2l.yml`). It sets `noise.snr_db: null` and
  `output_dir: results/measured`. Runs saved before this change used a z 0–0.85 m scene
  and the campaign's 327×151 grid; `--replot <run> --plot-window -0.15 0.15 0 0.4`
  redraws their plots on the field.
- **What a measured config must not set.** It must not set `tx_aperture` or
  `rx_aperture`: both are derived from the `measured` block and the data, as is
  `grid_search.aperture.width` (the measured TX width). It also needs
  `sim_scene.exact_step: true`, `noise.snr_db: null` and a `grid_search` block.
- **What the data must allow.** The RX `z` must be **exactly a row of the CSV** (to
  1e-6 mm). Otherwise the error names the nearest rows: z = 175 gives "174 or 177 mm",
  since the rows are 3 mm apart. Both RX edges must be `X` values of that row, every
  position between them must be present, and their pitch must be uniform and a whole
  number of scene spacings. Every frequency of the run must be a column of the CSV (to
  1 Hz; the error names the nearest columns, e.g. 299.5 GHz gives "295GHz and 300GHz"),
  and must be > 0: 0 Hz has no wavelength, so the `0GHz` column cannot be solved at.
  Any of these failing is an error (positions in mm) before anything is solved.

**The frame transform.** With `z_rx` the RX row and `x_c` the RX centre
(`(x_left + x_right)/2`), stage mm become simulation metres as

```
z_sim = (z_rx - z) * 1e-3        the RX row at z = 0, the TX above it at +z
x_sim = (x - x_c) * 1e-3         the RX centre at x = 0
```

`x` is **not** mirrored: inverting z is the only coordinate inversion, and a 2D
propagation does not depend on the mirror. For `measured_rx177.yml` the TX becomes
x −0.115…−0.015 m (centre −0.065 m, width 0.1 m) at z 0.242 m. The RX becomes 101
elements at −25…+25 mm on the 0.5 mm stage pitch, an on-grid receiver with
`grid_stride: 2`. The exact-step axis is `spacing · arange(−N, N+1)`: x = 0 is a
sample, and every element **is** a scene sample, placed by index and checked to 1e-9 m
(samples 1100, 1102, …, 1300 of 2401). The linspace axis of the simulated configs steps
by 0.250104 mm and has no sample at the stage positions.

**From the CSV to the measurement.** Each RX element's measurement at frequency `f` is

```
y = amplitude_scale * V(f)        V(f): the CSV cell at the element's (X, Z), column f
```

`V(f)` is `topt-parser`'s cell (see "The cell" above): the conjugated, absolute-time
spectrum of the trace's first 4000 samples. A pulse delayed by τ therefore has phase
**+2πfτ**, the convention of the RS kernel the simulated measurement is built with.

**`amplitude_scale`.** It is required, finite and > 0, and it is applied as written:
nothing is calibrated. The raw values are in **nA·samples**, an arbitrary unit. The
solver's TX is a **unit-amplitude box** with no fitted gain, so the scale sets the
power the candidates must reproduce. A wrong scale therefore **biases the candidate z**:
the loss has no gain to absorb it, and it favours the planes whose box can deliver that
power. The value is fixed by hand.

The shipped value is a **placeholder**: 1 / max|V(300 GHz)| on that RX in
`topt_spectra.csv` (122.356, at x = 209 mm), which puts the RX peak at 1.0.
Every run logs the raw max and median |V| it read, to choose a value from.

**Descending the amplitude (`gerchberg_saxton.amplitude: descend`, `amp_max`).** The
default, `amplitude: fixed`, holds every solve's TX at the unit-amplitude box above.
With `amplitude: descend` the solve fits the plate's amplitude too: one amplitude a(x)
over the candidate window, shared by every tone, started from the **raw** magnitude of
the reference tone's back-propagated measurement (not rescaled) and moved in the same
backtracking step as the phase, clipped to [0, `amp_max`] after every step. `amp_max`
is required with `descend` and refused with `fixed`; it is finite and > 0, in the units
of the box's amplitude 1 (so relative to the data as `amplitude_scale` scales it).
The candidates' `aper_profile` then carries amplitude and phase, the `--scenes`
amplitude panel scales to the data, and `candidate_beams.json`'s `gs` block and
`true_mgs.json` record `amplitude` and `amp_max` (a fixed run writes neither). It runs
on both engines (rust is parity-checked against python, see rust/README.md), but not
on a solve that captures GS history: `mgs` refuses it. `grid-search-mgs --true-mgs` then
skips the plain-`mgs` baseline (`true_mgs_scene.png`) and makes **one solve at the TX,
like a candidate**, as on a measured run (see "`--true-mgs` on a measured run" below):
at the true TX window (`tx_aperture` x_min..x_max at its z) on a simulated config, under
the sweep's solver settings and engine, written to `true_mgs.json` / `true_mgs.npz` with
the measured run's keys (`kind` says "true TX") and logged as true/best. `--replot
--true-mgs` of such a run exits 2: the solve is the sweep's, as on a measured run.
`configs/measured/measured_rx{153,177,225}_wide_amp.yml` use it with the whole stage
row as the RX (x 50–250 mm, 401 elements, RX centre at stage x 150 mm), `amp_max: 10`,
and an `amplitude_scale` from the 2D energy rule on that row,
sqrt(0.1 / (Σ|V(300 GHz)|² · 0.5e-3)): the RX's own power, not a calibration.

**A uniform level (`gerchberg_saxton.amplitude: level`).** Between the two: the plate
stays a **uniform** box, but its level A ≥ 0 (one number, shared by every tone) is fitted
with the phase rather than assumed, so nothing is guessed about the plate's brightness:
no cap, and `amp_max` is refused. A starts **power-matched**: the start phase's
unit-amplitude plate is propagated to the RX, and A0 = sqrt(Σ|y|² / Σ|p|²) over the RX
elements of every tone, unweighted (a start that reaches no element falls back to A0 = 1).
The descent then moves A by the **mean** over the window of the per-sample amplitude
gradient (the exact gradient is the sum, n_window times larger; the mean keeps A's steps
the size of one sample's under `descend`), clamped at 0, in the same backtracking step as
the phase. `aper_profile` carries the level, and the `gs` blocks record `amplitude:
level` (no `amp_max`). It runs on both engines (rust is parity-checked against python,
see rust/README.md), is refused where `descend` is (`mgs`), and its `--true-mgs` is the
same candidate-style solve at the TX.
With several tones it needs `init: backprop`: the legacy warm start would match A0 to
the seeded random phase, which lands it on a wrong level (8.33 against a true 3.7), so
`grid-search-mgs` refuses that pairing before the sweep. One tone under `warm_start`
is the random start, and runs.
`configs/measured/rxsize/rx177_level_w{200,165,140,120,100,80}_coarse.yml` run the
receiver-size study on row 177 with it.

**Excess-power penalty (`gerchberg_saxton.power_penalty`).** A fitted amplitude can
radiate more power than the receiver caught and still fit it, by sending the excess
past the RX window, where the plain loss does not look. With `power_penalty: gamma`
(finite, > 0; default null = off) the descent minimizes J = L · max(1, P_plate /
P_rx)^gamma instead of the plain joint loss L: P_plate is the power leaving the
candidate plate (Σ a² dx over the window; A² · n_window · dx for a level) and P_rx the
power measured on the RX elements (Σ|y|² times the element pitch, the mean over the
tones), a constant of the solve. The **floor at 1** is essential: a plate dimmer than
the receiver is not rewarded (an unfloored L · R would collapse onto dark plates), so
the penalty only charges the excess. It needs a fitted amplitude and runs with
`amplitude: level` and `amplitude: descend`; with `fixed` (where P_plate is the same
for every solve) it is refused. The descent direction is ∇J / R^gamma: the phase
gradient is unchanged and the amplitude gradient gains one term where R > 1. Under
`level` that makes the floor (P_plate = P_rx) a kink of J in A, and where the slopes on
its two sides bracket zero the kink is J's minimum: a plain step for the level crossed
it from either side, the joint step was rejected at every halving, and the flat loss
stopped the solve as "converged" with J 1.5-4.5x too high (96% of the row-177 / 120 mm
cells). There the level's step is stopped at the floor, from above or below, so the
level can sit on the floor while the phase keeps descending (`level` only; `descend`
keeps the plain term). Both engines run it with the same recipe (python computes P_rx
and hands it to rust; parity in rust/README.md). `final_loss` (and the loss curve) is
then J, the number that ranks candidates, while `per_freq_losses` stay the plain
per-tone losses;
`candidate_beams.json`'s `gs` block records `power_penalty` (only when set), and
every candidate entry, `cand_####.json` and `true_mgs.json`'s `solve` block carry
`final_mse` (the plain L) and `power_ratio` (R; null without the penalty). Residual
plot titles say `× excess-power factor^gamma`.

**Unknown TX power in simulation (`tx_aperture.amplitude`).** The simulated counterpart
of `amplitude_scale`: one real factor s > 0 on the simulated plate of every tone. The
grid candidates never see s: under `amplitude: fixed` each holds the unit box, under
`descend` / `level` the amplitude it fits to the data. The capturing `mgs` baseline of a
fixed-amplitude run (`--true-mgs`, `scripts/true_tx_table.py`) does see it: it holds the
true, scaled plate's own amplitude, as plain `mgs` does. So a fixed-amplitude grid search
at s != 1 is a deliberate wrong-power test (no candidate, the true position's included,
can match the data's power), and `grid-search-mgs` warns once, up front, when a simulated
run asks for one. A number is s itself (finite and > 0; a YAML `5e-1` or `1e0`, which
PyYAML loads as a string, is taken as the number). `rx_peak` sets s = 1/P,
P the largest |clean RX element| over every tone from the unscaled plate, so the clean RX
peak element is 1, as `amplitude_scale` = 1/max|V| puts the measured peak at 1. It is
applied once, to the plate, when the simulated scene is built: the measurement, the noise
(drawn on the scaled field, so one seed gives s times the unscaled run's noisy profile),
the real-beam panels, `run.npz`/`measurement.npz` and the `mgs` baseline all carry it.
The run logs `TX amplitude rx_peak: plate x 0.38169 (clean RX peak element 2.62 -> 1)`,
and `candidate_beams.json` (and `run.json`) gain a top-level `tx_amplitude` block
(`setting`, `scale`, `clean_rx_peak_unscaled`). The default 1.0 changes nothing, and is
left out of every dump, so older snapshots and `mgs-study` roots still match. The loss
scales as s^2 while `convergence_threshold` is absolute, so a run at s != 1 stops at a
different relative flatness than the same run at 1: as a measured run scaled to peak 1
does, which is the point of `rx_peak`. It does not apply on the measured and
experimental paths, which have no simulated plate.
`configs/twin/twin_caustic_rx120.yml` is the measured 120 mm run's simulated twin with
`rx_peak` and `amplitude: descend`.

**Only the RX aperture's data is used**, for the solve and for anything computed from
the data (the logged |V| statistics). The rest of the CSV is only scanned for its rows'
`Z`, to check the RX row against.

**Running it.**

```bash
grid-search-mgs --config configs/measured/measured_rx177.yml --dry-run    # derived geometry + grid, no solve
grid-search-mgs --config configs/measured/measured_rx177.yml --limit 50   # quick partial run
grid-search-mgs --config configs/measured/measured_rx177.yml --engine rust -j 32 --true-mgs
grid-search-mgs --replot results/measured/measured_rx177 --scenes --scene-top 10   # the 10 best candidates' scenes
```

The derived geometry is logged first (the data file, RX row, elements, pitch, TX
window in both frames), and then, per frequency, the column read and the raw |V|.
`--engine`, `-j`, `--limit`, `--dry-run`, `--true-mgs`, `--freq` (each one a column of
the CSV), `--surface-anim`, `--scenes`, `--scene-top`, `--plot-window` and `-o` work as
on a simulated config. `--anim`, `--snr-db` and `--noise-seed` are refused (exit 2): the
candidate animation has no measured-run variant, and the data carries its own noise.
`mgs` and `mgs-study` refuse a measured config. `--replot` of a measured run redraws
its residual plots and `analysis.json`, and renders `--scenes`, but refuses
`--true-mgs` and `--anim`.

**`--scenes` on a measured run.** The TX's beam is unknown, so there is no real beam to
draw beside a candidate. Each `scenes/cand_####.png` therefore has **3 panels**, not 4:

- **Left (tall):** the candidate's beam, re-propagated through the scene, on a colour
  scale set by that candidate's own field. The candidate TX aperture is in blue and the
  RX elements are in red, as on a simulated run. The **measured TX window** (z =
  `real_tx_z`, x = `real_tx_x_min`..`real_tx_x_max`) is a dashed cyan box, "measured TX"
  in the legend. The title gives the MSE and the stop reason (when not converged), then
  the candidate's **distance to the measured TX in mm**: centre to window centre, as
  `analysis.json` measures it.
- **Right, top and bottom:** the candidate TX phase (unwrapped) and amplitude.

`--scene-top N` renders the N lowest-loss candidates only. `scene_average.png` averages
exactly the candidates rendered. It outlines the measured TX window the same way and
draws no TX aperture; its title says "TX shown = measured TX window".

`--true-mgs` on a measured run is **one solve at the measured TX position, made exactly
like a candidate**. It uses the measured TX window with the uniform assumed amplitude,
the run's own measurement, and the sweep's solver settings and engine. Its loss
therefore compares directly with the candidates': it is logged beside the best
candidate's, with the ratio measured/best. It replaces `true_mgs_scene.png`, the plain
`mgs` baseline of a simulated run, which needs the real plate.

**What lands in `results/measured/<run_name>/`.** The run has the simulated grid run's
layout, with these differences:

- **`candidate_beams.json`.** `ground_truth` is the measured TX **position** (`real_tx_z`,
  `real_tx_x_min`, `real_tx_x_max`), as on every run. A `measured` block is added, giving
  the data file (`data_file`, its absolute path; `data_file_bytes` and
  `data_file_sha256`, since `topt-parser` rewrites the file under the same name), the
  `columns` read (e.g. `["300GHz"]`), the `frame` formulas (`z_rx_mm`, `x_center_mm`,
  no mirror) and the stage-frame TX/RX as configured. It also has `tx` (`beam:
  "unknown"` and the derived window), `rx` (`row_mm`, element count, pitch,
  `grid_stride`, the element positions `x_mm` and their scene indices), the `phasor`
  formula and `amplitude_scale`. `per_freq` holds the raw |V| max and median. Runs saved
  before the CSV input have the older keys in this block (`data_dir`, `file_unit_mm`,
  trace file names) and still replot: `--replot` reads only whether the block is there.
- **`measurement.npz`.** There is no `real_tx_aper_axis` or `real_tx_aper_profile`.
  `rx_field` is non-zero at the element samples only, and `rx_aper_profile_00` holds the
  scaled values `amplitude_scale * V(f)` (stored as complex64, like every profile).
- **`analysis.json`.** The argmin and the top candidates are scored against the measured
  TX. `energy` and `n_dof` are null: there is no RX-plane field off the elements and no
  beam.
- **Residual plots.** These are as on a simulated run, windowed by `base.yml`'s
  `plot_window` (recorded in the manifest). Their "True TX location" marker is the
  measured position.
- **`config_snapshot.json`.** It holds the derived `tx_aperture` (with `beam: null`; the
  config's stand-in beam only lets the block validate and is never used), the derived
  `rx_aperture`, and the `measured` block. `config_source.yml` is the thin file.
  Neither loads back, because a measured config may not set the derived blocks and the
  base path is relative to the original file.
- **`true_mgs.json` / `true_mgs.npz`.** These are written with `--true-mgs`.
  `true_mgs.json` holds the TX window, the solver settings, the solve's loss, iterations
  and stop reason, the best candidate and its distance to the measured TX, and
  `loss_ratio_to_best`. `true_mgs.npz` holds the solved aperture and loss curve, with a
  candidate npz's keys.
- **`scenes/cand_####.png` / `scene_average.png`.** These are written with `--scenes`,
  as 3-panel PNGs with the measured TX outlined (see "`--scenes` on a measured run"
  above). They show the candidates' beams. The TX's own beam is still unknown and is
  never drawn.

## Source layout

All code lives in `src/rice_bend/`:

Numerics and geometry:

- `rs.py` — Rayleigh-Sommerfeld propagation: the kernel, applying it, and whole-scene illumination
- `beam_aperatures.py` — every TX beam generator (caustic phase plate for parabolic trajectories,
  steered plane wave, Bessel axicon, focusing lens) and the `beam.type` dispatch. The caustic
  plate resamples its **unwrapped** phase and only then takes `exp(1j·phase)`
- `sim_scene.py` — aperture and scene geometry, including which scene samples an on-grid
  receiver's elements are (`grid_element_indices`)
- `noise.py` — receiver noise: peak-referenced complex AWGN on the simulated RX elements,
  one seeded stream per tone
- `landing.py` — where a beam lands on the RX plane, as numbers (peak, centroid, the 5–95 %
  energy span, the in-window fraction), the beam's geometric target, and how closely two
  RX-plane fields match inside the window. Pure measurement: it feeds the `LANDING` block and
  `scripts/verify_scenario.py`
- `interp.py` — the interpolation conventions, named (`interp_real_imag`, cartesian, for fields;
  `interp_amplitude` for magnitudes). They are **not** interchangeable, and no polar
  (amplitude-and-angle) interpolation of a field remains
- `analysis.py` — numeric per-run analysis of a grid search (top candidates, localization error,
  energy in the RX window), written to `analysis.json`. It stores at most 1,200 top candidates
  (`TOP_MAX_STORED`, sized for the 327×151 grid); its statistics always use the full set
- `config.py` — pydantic config models (scene, TX beam, RX aperture, Gerchberg-Saxton, output,
  grid search, receiver noise, experimental bench constants) and the default carrier
  (`DEFAULT_FREQ_HZ`)

Infrastructure:

- `data_store.py` — per-run persistence (run.npz / run.json / config snapshots) and the small
  I/O helpers both entry points share
- `parallel.py` — one parallel map; `jobs=1` runs the same task function the pool would
- `plotting.py` — the scene panel and line panel both entry points draw
- `cli.py` — shared logging setup
- `exp_data.py` — readers for the experimental `.mat` captures
- `tds_data.py` — readers for a measured THz-TDS raster: the directory index and one pulse
  trace (used by `topt-parser`). numpy and the standard library only
- `measured.py` — a `measured:` config block to the simulation: the stage → simulation
  frame, the derived TX/RX blocks (`apply_measured`), the element placement and the RX
  values read from the spectra CSV (see "Measured THz-TDS data")

Entry points:

- `mgs.py` — the MGS solver core and driver (`mgs`): the descent, the three starts
  (`init`) and the offset scan over the comb's true period
- `animate.py` — TX-estimate animation from a saved run (`mgs-animate`)
- `grid_search.py` — orchestration + CLI for the sweep (`grid-search-mgs`), over three modules:
  - `grid_sweep.py` — enumerate the grid, solve once at each point, persist. No
    matplotlib
  - `residual_plots.py` — the residual grid and its six plots
  - `candidate_scenes.py` — re-illuminated candidate beams and the true-MGS baseline
- `study.py` — the parameter-study driver (`mgs-study`): the series tables, the resume guards,
  `study.json` and the study plots
- `topt_parser.py` — pulse traces → the spectra CSV (`topt-parser`), and its readers
  (`read_spectra_csv`, one row; `read_spectra_map`, one column), the format's one home
  (see "`topt-parser`")
- `topt_heatmap.py` — one column of the spectra CSV as a log-magnitude map (`topt-heatmap`)

Scripts, in `scripts/`:

- `characterize.sh`, `characterize.py` — the regression net (see Tests)
- `verify_scenario.py` — where a config's beam lands on the RX plane, forward-only
- `true_tx_table.py` — the solve at the true TX per study bandwidth (see `mgs-study`)
- `archive/report_figures.py` — the figures and rank-of-truth numbers of the final
  report. The 150 GHz result roots it reads were deleted on 2026-10-13, so it is kept
  for reference only

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

The six checks split by noise, receiver and start:

| Check | Command | Noise | Receiver | Start |
| --- | --- | --- | --- | --- |
| 1 | `mgs` on `tiny_check.yml` (280 and 300 GHz, one joint solve) | none | interpolated, λ/20 at runtime | legacy |
| 2 | `mgs` on `scenario_caustic_hit.yml`, pinned to 300 GHz | 30 dB, seed 0 | on-grid λ/2 | legacy |
| 3, 4 | `grid-search-mgs --limit 6` on `scenario_caustic_hit_sparse.yml` at 300 GHz, serially and across 4 workers | 30 dB, seed 0 | interpolated λ/2 | legacy |
| 5 | `grid-search-mgs` on `tiny_check.yml`, 280 and 300 GHz | none | interpolated, λ/20 at runtime | legacy |
| 6 | `grid-search-mgs --limit 6` on `scenario_caustic_hit_1f_grid2l.yml`, serially and across 4 workers | 30 dB, seed 0 | on-grid λ/2 | `backprop` |

Checks 1 and 5 pin the noiseless path and the runtime λ/20 RX sampling. The others pin
the receiver-noise path: their residuals and hashes depend on the seeded draw, that
is on numpy's `Generator` stream (`noise.py`) as well as on the solver. Checks 3 and 4
are the net's only noisy run of the interpolated receiver, which is why
`scenario_caustic_hit_sparse.yml` stays on it.

Check 6 is what puts the data-driven start in the net, with the campaign's one-solve
sweep: each of its six candidates is solved once, under the base config's 3000-iteration
cap and −1e-10 stop threshold (no `gs_overrides`). Its digest holds what checks 3 and 4
digest (`init`, every residual and iteration count, candidate 3's and the measurement's
hashes), and as in check 4 the parallel run's manifest has to equal the serial one's.

The expected values are generated, never transcribed — hand-typed float literals rot
the moment nobody re-blesses them. After a change that is meant to move the numbers
(the 300 GHz retarget is one), the digest has to be regenerated with `--bless` before
a diff means anything.

**The expected digest is awaiting the owner's bless.** `scripts/characterize_expected.json`
still dates from before the 300 GHz retarget. Until it is blessed, `characterize.sh`
runs to the end, exits 1 and reports differences on every check, with check 6 as two
`ADDED` entries: a run shows that every command still completes, not that the numbers
are held.

The run took ~25 s before the retarget; check 6 adds about 20 seconds. It pins the
python engine only.
The Rust engine has its own gates: `cd rust && cargo test`, and
`rust/python/parity.py`, whose fourteen slices run `tiny_check.yml` and
`scenario_caustic_hit_sparse.yml` noiseless and again at 20 and 10 dB, and
`scenario_caustic_hit_1f_grid2l.yml` on the on-grid receiver at that config's 30 dB,
from the legacy start and from the data-driven start at one tone and at three. Four
more descend the amplitude: that config from the data-driven start at one tone and
from the legacy start at three, again with zeros in the start amplitude, and the
measured `measured_rx177_wide_amp.yml` row (skipped without its data file). Three
fit a uniform level: that config from the data-driven start at one tone and at three,
and the measured `rxsize/rx177_level_w200_coarse.yml` row (skipped without its data
file). See `rust/README.md`.
