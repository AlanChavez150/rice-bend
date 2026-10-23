# rice-bend-core — Rust engine for the MGS solver hot path

> Milestone status and the remaining work order live in [PLAN.md](PLAN.md).

The Python implementation under `src/rice_bend/` is **the reference**:
`scripts/characterize.sh` pins it byte-for-byte, and this crate is a second,
tolerance-validated engine for the sweep's compute (`gs_reconstruct`, ~90% of a
grid search's multi-hour cost). Select it with `--engine rust` on
`grid-search-mgs` / `mgs-study` (or `gerchberg_saxton.engine: rust`). The
dispatch lives at one point in `gs_reconstruct`, after the shared seed/kernel
setup, so both engines start from identical arrays. Compare two saved runs with
`python/compare_runs.py <python_run> <rust_run>`.

## Parity bar (settled)

rustfft ≠ pocketfft, so bit-identity is out of reach by design and NOT the bar.
The Rust engine is accepted when:

- every candidate's `final_loss` agrees within **rtol 1e-3**;
- the sweep-level results match **exactly**: argmin cell, top-candidate set;
- `n_iters_run` / `stop_reason` are *reported, never gated* — ulp-level loss
  differences near the convergence threshold legitimately move the stop point.

The one bit-level requirement is the complex64 quantization at the two
`rs_apply` outputs (rs.py:79-82 — "changing this quantisation flips candidate
rankings"): ≥99% of quantized elements bit-equal on the conv golden vectors,
the rest within 1 ulp.

**Receiver noise** (`noise.snr_db`, `--snr-db`) is added in Python by
`MGS.measure()`, before the engine boundary, so the engine is unchanged: both
engines receive the identical noisy `rx_field` / `error_weighting` arrays. The
golden vectors stay noiseless, and `python/compare_runs.py` refuses two runs
whose `(snr_db, seed)` differ. The on-grid receiver (`rx_aperture.grid_stride`)
is built there too: the element samples placed on the scene axis and a weighting
that is zero at every sample without an element are still just an `rx_field` and
an `error_weighting`, so the engine does not know about it either.

**The data-driven start** (`gerchberg_saxton.init: backprop`) did not touch
the crate. `gs_solve` knows two modes. `warm_start` runs the engine's own twin
of the legacy warm start (`src/warm_start.rs`), which is frozen with it so that
earlier runs reproduce. `random` descends from the phase it is handed,
unchanged, and that is how a start computed in Python reaches the engine: the
back-propagated phase and, for several tones, the offset scan over the comb's
true period are built in `src/rice_bend/mgs.py` for either engine, and Rust runs
the descent from the result.

**The descended amplitude** (`gerchberg_saxton.amplitude: descend`, with
`amp_max`; added 2026-10-20) runs on both engines. The descent then fits ONE
amplitude a(x), shared by every tone, together with psi. Its gradient is
2·Re(g·conj(e^{iρψ})) from the same back-propagated residual g, averaged over the
tones and zeroed off the support. Every backtracking trial takes the same step on
both halves, with `clip(a − step·∇a, 0, amp_max)` (`npmath::np_clip`, numpy's NaN
semantics), and the pair is accepted or shrunk as one (`src/solve.rs`). The start
amplitude a0 (`mgs._backprop_amplitude`) is built in Python and crosses as
`aper_amp`. `gs_solve` takes three trailing arguments, `amplitude` ('fixed' |
'descend' | 'level'), `amp_max` and `support`, whose defaults are the legacy behaviour, so
older positional callers run unchanged. **The support must cross explicitly**
under 'descend', and `gs_solve` refuses 'descend' without it. Python's support
is `|orig_aper_amp| > 0`, the window, while a0 can be 0 inside the window. The
legacy derivation (`aper_amp > 0`) would then drop samples Python still descends.
With a0 zeroed at 78 of 288 window samples, that one change moves the final loss
by 7e-6 rel and the field by 8e-3. Real a0 is nonzero across the window on the
measured row, but the clip drives 14 to 35 window samples to exactly 0 by the
end of a solve. Those samples keep descending because rust fixes the support
once, on entry. Under the legacy warm start, stage 1 stays phase-only at a0
(`AmplitudeMode::Fixed`). It runs on `|a0| > 0`, the support of Python's child
`gs_reconstruct`, and the scan uses a0 (`src/warm_start.rs`). The 'fixed' path
is unchanged bit for bit. 455 result arrays (curr_aper_f, the losses, the loss
curves, n_iters, stop) are byte-equal between the builds before and after the
change. They come from slices 1, 2, 5, 6 and 7 below, a 3-tone legacy warm start
on the slice-5 config, and five measured candidates with the amplitude forced to
'fixed'. The slice cases ran through both `gs_reconstruct` and the hand-marshaled
call.

**The uniform level** (`gerchberg_saxton.amplitude: level`, no cap; added
2026-10-20) runs on both engines too (`AmplitudeMode::Level`). The plate is ONE
level A ≥ 0 on the support, shared by every tone. The per-sample amplitude
gradient is built as under 'descend', but from the residual weighted twice (below);
A moves by its mean over the support (`np_mean_f64`, support samples in order),
clipped at 0 with no upper bound (`np_clip(A − step·∇A, 0, ∞)`), in the same joint
backtracking step as psi. The level back-propagates w·r = w²(prop − y) (one more
adjoint per tone, `Work::r2` / `g2`), whose mean is the exact slope
(2N/n_sup)·dL/dA, where the phase and 'descend' keep r: under the measured row's
tapered weighting the mean of the singly weighted gradient points against dL/dA in
about half the cells, and the joint step stalled with the plate far below the power
floor (2026-10-22). The start A0 (`mgs._power_matched_level`) is computed in
Python before the dispatch and crosses exactly as a0 does, as `aper_amp`: A0 on
the support and 0 off it.
Rust reads A0 back from any support sample, since A0·1.0 is A0 bit for bit, and
the boundary checks that the plate is one value there. It also needs the support,
as 'descend' does. **Several tones under `init: warm_start` are refused** on both
engines with "amplitude: level with several tones needs init: backprop". The
check is `mgs.check_amplitude_mode`, which `grid-search-mgs` and
`run_grid_search` call before the sweep with the run's tone count, with a second
guard in `gs_solve`. That start would match A0 to the seeded random phase, which
lands it on a wrong level (8.33 against a true 3.7). One tone under `warm_start`
is the random start and runs. 'fixed' and 'descend' are unchanged bit for bit
by this port: 637 result arrays from the builds before and after it are
byte-equal. They are the fixed set above plus descend from the backprop and
warm starts, with and without zeros in a0, and the five measured candidates
under both modes.

**The excess-power penalty** (`gerchberg_saxton.power_penalty`, the exponent γ;
added 2026-10-22) runs on both engines under 'level' and 'descend'
(`SolveParams::power_penalty`, `types::PowerPenalty`). The descent then minimizes
J = L·R^γ with R = max(1, P_plate/P_rx), instead of the joint loss L. P_plate is
the power the candidate plate sends out, Σ a_j²·dx over the support: (A·A)·n_sup·dx
for a level, and `np_sum_f64` of the squared support samples, in order, times dx
under 'descend'. P_rx is the power on the RX elements. The floor at 1 is what
makes it a penalty: an unfloored L·R collapses onto dark plates. P_rx is a
constant of the solve, computed once in Python (`mgs._received_power`), and
crosses as a scalar, so the two engines divide by the same number. `gs_solve`
takes it with γ as two more trailing keywords, `power_penalty` and `rx_power`.
They go together (both or neither), must be finite and > 0, and are refused under
'fixed', where R is a constant of the grid (the config refuses that too). The
phase gradient is untouched. The amplitude gradient gains one term when R > 1,
added after the mean over the tones (and, for the level, after the mean over the
support); the expressions and their evaluation order are in the `src/solve.rs`
module doc. Each backtracking trial takes R from its own amplitude and is accepted
on a strict decrease of J. Under 'level' J has a kink in A at the floor
A_floor = √(P_rx/(n_sup·dx)), where R leaves 1; on the measured row-177 grid the
level reached it in 96% of the cells within a few iterations, after which its
step kept crossing the floor, the joint step was rejected at every halving and
the flatness rule reported a false "converged" with J 1.5–4.5× too high. So
when J's one-sided slopes in A bracket zero (∇A < 0 < ∇A plus the penalty's
term), the level's step is stopped at the floor from the side it is on, and the
level can sit there while the phase keeps descending; elsewhere the gradient is
the one above, bit for bit. This clamp is part of the penalty under 'level'
(no flag); 'descend' keeps the plain penalty. `loss_full` and `final_loss` are then J, while
`final_loss_per_freq` and `loss_full_per_freq` stay the plain per-tone losses.
The return tuple grows by two at the end: `final_mse`, the post-accept L (equal
to `final_loss` with the penalty off), and `power_ratio`, the post-accept R (None
with it off). With the penalty off the loop does not change: `gs_core` branches
on it and never multiplies by a factor of 1.0. The legacy warm start's phase-only
stage 1 runs unpenalized on both engines. Python drops the penalty from its child
solve, and `gs_core` ignores it under `AmplitudeMode::Fixed`, which is what
`warm_start.rs` hands stage 1.

**The relative stop rule** (`gerchberg_saxton.convergence_rel`, added 2026-10-23;
`SolveParams::convergence_rel`) replaces the absolute flatness test when set, and
`convergence_threshold` is then ignored. The absolute threshold has units of loss,
so it means a different thing in every cell: on the measured valley floor
(J ~ 2e-5) it stops cells before the ranking is decided, on the walls (J ~ 7e-3)
f32 spacing (~5e-10) makes it fire on quantization. The relative rule runs at the
same place, under the same `iter_idx > convergence_count`, on an f64 history of the
same pre-step objective the f32 `loss_full` records. With w the window
`hist64[iter_idx − count .. iter_idx]` (current index excluded) and r its last
value, the solve converges when r ≤ 0 or mean(diff(w))/r > −`convergence_rel`. The
mean is `np_mean_f64`, numpy's pairwise sum over n (bit-equal to `np.mean` on the
9-element windows of the default count 10). `gs_solve` takes it as one more
trailing keyword, `convergence_rel` (None by default, else finite and > 0). Off,
`gs_core` branches on the Option and runs the absolute block verbatim, with no
floating-point operation added. Runs under different stop rules are not
comparable.

`python/parity.py` runs seventeen slices under the same bar:

| Slice | Config | Tones (GHz) | Noise | Receiver | Start |
|---|---|---|---|---|---|
| 1 | `tiny_check.yml`, 3 candidates | 280, 300 | off | interpolated, λ/20 | legacy warm start (Rust's own twin) |
| 2 | `scenario_caustic_hit_sparse.yml`, 6 candidates | 300 | off | interpolated, λ/2 | legacy (one tone: the seeded random draw) |
| 3 | slice 1 again | 280, 300 | 20 dB | interpolated, λ/20 | legacy warm start |
| 4 | slice 2 again | 300 | 10 dB | interpolated, λ/2 | legacy |
| 5 | `scenario_caustic_hit_1f_grid2l.yml`, 6 candidates | 300 | the config's 30 dB | on-grid, λ/2 | legacy, forced in memory |
| 6 | the same | 300 | the config's 30 dB | on-grid, λ/2 | `backprop` |
| 7 | the same | 270, 300, 330 (±10 %) | the config's 30 dB | on-grid, λ/2 | `backprop`, with the offset scan in front |
| 8 | the same, `amplitude: descend`, `amp_max` 2 | 300 | the config's 30 dB | on-grid, λ/2 | `backprop` |
| 9 | the same | 270, 300, 330 | the config's 30 dB | on-grid, λ/2 | legacy warm start (stage 1 phase-only at a0) |
| 10 | slice 9, a0 zeroed inside the window | 270, 300, 330 | the config's 30 dB | on-grid, λ/2 | legacy warm start |
| 11 | `measured/measured_rx177_wide_amp.yml` as written (descend, `amp_max` 10, 6000 iterations), 6 candidates spread over the grid; skipped without its data file | 300 | measured | measured row, 401 elements | `backprop` |
| 12 | slice 5's config, `amplitude: level`, 6 candidates spread over the grid | 300 | the config's 30 dB | on-grid, λ/2 | `backprop` |
| 13 | the same | 270, 300, 330 | the config's 30 dB | on-grid, λ/2 | `backprop` |
| 14 | `measured/rxsize/rx177_level_w200_coarse.yml` as written (level, 6000 iterations), 6 candidates spread over its 4 mm grid; skipped without its data file | 300 | measured | measured row, 401 elements | `backprop` |
| 15 | slice 5's config, `amplitude: level`, `power_penalty` 1, 6 candidates spread over the grid | 300 | the config's 30 dB | on-grid, λ/2 | `backprop` |
| 16 | the same, `amplitude: descend`, `amp_max` 2, `power_penalty` 1 | 300 | the config's 30 dB | on-grid, λ/2 | `backprop` |
| 17 | `twin/twin_caustic_rx120_level_pp1.yml` (level + `power_penalty` 1, as configured), `convergence_rel` 1e-6, 4 cells around the true TX | 300 | the config's 30 dB | the twin's 120 mm RX | `backprop` |

Results (2026-10-20, with the level port; slice 11 at 6000 iterations): all
fourteen pass, and n_iters_run and stop_reason are identical on every candidate
of every slice. The worst rel Δloss is 1.2e-10 on slices 8-11 (on the measured
slice) and 2.5e-10 on slices 12-14 (on the measured slice; 1.8e-14 on the
simulated ones). Slices 15 and 16 came with the power penalty (2026-10-22) and
slice 17 with the relative stop rule (2026-10-23); they are not in these results.

The scenario configs carry a 30 dB noise block of their own, so the noiseless
slices force the noise off explicitly (`SNR_OFF`, what `--snr-db off` passes)
rather than inherit it. The start of the on-grid slices is set in memory and
never taken from the file: the caustic base config says `init: backprop`, so
slice 5 overrides it to `warm_start` to stay the legacy on-grid slice, and
slices 6 and 7 pass `backprop` explicitly. For the legacy starts `rust_solve`
marshals the candidate for `gs_solve` by hand, an independent second copy of
what `mgs._gs_solve_rust` does. For `backprop` it is `gs_reconstruct` with the
engine forced to Rust, the path a sweep takes, so slices 6 and 7 check the
descent from one identical, structured start. The descend slices go the same
way, because a0 is built in `mgs.py`, and so do the level slices (A0 is built
there too). Slice 10 wraps `mgs._backprop_amplitude`
in memory, for both engines, to zero a0 at every fifth window sample and over one
run of 25.

## Build / dev loop

```bash
# pure-Rust cycle (no Python involved — the `python` feature stays off):
cd rust && cargo build && cargo test

# golden vectors (already committed; regenerate only after an approved change
# to the reference solver — meta.json records the generating commit):
.venv/bin/python rust/python/gen_golden.py
# ...or named cases alone, leaving the rest and meta.json as they are:
.venv/bin/python rust/python/gen_golden.py solve_descend solve_descend_joint
.venv/bin/python rust/python/gen_golden.py solve_level solve_level_joint
.venv/bin/python rust/python/gen_golden.py solve_level_penalty solve_descend_penalty
.venv/bin/python rust/python/gen_golden.py solve_level_relstop

# extension module into the project venv (Python 3.8.10). Rebuild after any
# change under src/: mgs._gs_solve_rust passes amplitude=/amp_max=/support=,
# power_penalty=/rx_power= and convergence_rel= by keyword and unpacks a 9-tuple,
# which a build older than the relative stop rule refuses (a TypeError, with the
# rule off too):
uv pip install maturin
VIRTUAL_ENV=$PWD/.venv .venv/bin/maturin develop --release -m rust/Cargo.toml

# end-to-end acceptance (M5):
.venv/bin/python rust/python/parity.py

# per-candidate timing, python against rust:
.venv/bin/python rust/python/bench.py
```

The golden vectors committed under `tests/data/` predate the 300 GHz retarget
(generated 2026-09-27, at the commit `meta.json` records): `solve_single` at
150 GHz and `solve_joint` at 140/150 GHz, N=400. They remain valid for
`cargo test`. They are stored arrays, the tests read no config, and what they
pin is the solver's arithmetic, which does not care which carrier produced its
inputs. `gen_golden.py` run today would not reproduce them: from the retargeted
`tiny_check.yml` it generates 300 GHz and 280/300 GHz vectors of a different
size (N=800). The exceptions are `solve_descend` (F=1, backprop start) and
`solve_descend_joint` (F=2, legacy warm start). They were generated 2026-10-20
at 300 and 280/300 GHz, N=800, with `amplitude: descend`, `amp_max` 2 and a0
zeroed inside the window, and their manifests carry their own provenance. The
golden support crosses as stored, and the test asserts that it differs from
a0 > 0. Rust matches Python on both: final loss rel 1.5e-16 and 0, n_iters 12/12,
max |field − python| 2e-16. `solve_level` (F=1) and `solve_level_joint` (F=2,
280/300 GHz), both from the backprop start with `amplitude: level`, were added
the same day. Their start plate is A0 on the support, which the test reads from
the binary array: serde_json's parse of the manifest scalar is 1 ulp off. Final
loss rel 1.9e-16 and 0, n_iters 12/12, max |field − python| 0.
`solve_level_penalty` and `solve_descend_penalty` (2026-10-22) are `solve_level`
and `solve_descend` again (F=1 at 300 GHz, backprop start; descend with `amp_max`
2 and a0 zeroed inside the window) under `power_penalty` 1. Their manifests add
the scalars `power_penalty`, `rx_power`, `final_mse` and `power_ratio`. The tests
run `gs_core` with that penalty and keep the bars above, on J (first-iteration
loss rtol 1e-6, final_loss rtol 1e-3), plus final_mse and power_ratio at rtol 1e-3.
They also check the result against itself, bit for bit: final_mse is the mean of
the plain per-tone losses, and final_loss = final_mse·power_ratio^γ. P_rx is read
from the manifest scalar, so serde_json's parse can move R by 1 ulp, far below
these bars. `penalty_floor_is_the_plain_solve` needs no new data. On the
`solve_level` and `solve_descend` inputs, a P_rx far above any plate's power
(R = 1 throughout) reproduces the unpenalized solve bit for bit. A P_rx at a
quarter of the start plate's power, γ 2, raises the first objective above L, moves
the solve and keeps the same identities.
`solve_level_relstop` (2026-10-23) is a live descent, unlike the tiny_check cases,
which never accept a step: the digital twin (`twin_caustic_rx120_level_pp1.yml`,
2401-sample odd axis, level + `power_penalty` 1, backprop start, 30 dB noise) with
one candidate behind a 100 mm 0/1 window, stopped by `convergence_rel`. Its files
and keys are those of `solve_level_penalty`, plus the scalars `convergence_rel` and
`n_iters_abs_rule`, where the absolute rule stops the same solve in Python (Rust's
absolute rule stops it at the same iteration). The test keeps the penalty bars and
gates the stop loosely, as the other solve tests do: Converged, n_iters_run within
1% of Python's, and far below `n_iters_abs_rule` (so the relative rule is what
fired). An exact gate would be tighter than the losses: they are bit-equal for the
first 319 iterations, then differ by up to rel 7e-5 (about 5e-7 by the stop), and at
`convergence_rel` 1e-5 this solve stopped at 1961 in Rust and 1962 in Python. The
golden uses 4e-5 (stop 1237 in both; the absolute rule 2687).

Version pins: **pyo3 0.22 + numpy(rust) 0.22, abi3-py38, maturin ≥1.4,<2** — the
verified pairing for CPython 3.8.10. Do not bump without revisiting the floor.

## Work split and gates

| Package | Files | Gate | Status |
|---|---|---|---|
| FFT/convolution | `src/fft.rs`, `src/conv.rs` | `cargo test --test conv_parity` | green — 100% post-downcast bit-equal |
| numpy mirrors | `src/npmath.rs` | `cargo test --test numerics_parity` | green — bit-exact |
| Solver core | `src/solve.rs`, `src/warm_start.rs` | `cargo test --test solve_parity` | green — final loss bit-identical (descend and level cases: 1 ulp or 0) |
| PyO3 boundary | `src/py.rs` | `maturin develop` + `python/parity.py` | green — worst rel Δloss 5e-14 (descend slices: 1.2e-10; level slices: 2.5e-10) |

Alan wrote `next_fast_len`; Claude wrote the harness and, at Alan's request,
the rest of the implementation. Every module's doc comment carries its
semantics contract, with `src/rice_bend/mgs.py` / `rs.py` line references as
the source of truth. The load-bearing traps, in one place:

- **complex64 quantization** exactly at the two `rs_apply` outputs, nowhere else.
- **Adjoint spectrum from the time-domain conjugate** (`FFT(pad(conj(h)))`),
  never `conj(FFT(pad(h)))`.
- **f32 convergence window**: losses stored f32, diff/mean at f32, strictly
  `iter_idx > convergence_count`, window excludes the current index. The
  relative rule (`convergence_rel`) is the exception: it reads an f64 history of
  the same values, never the f32 one.
- **`loss_full` holds PRE-step losses**; `final_loss` is the last iteration's
  POST-accept loss — they differ whenever the last iteration accepted a step.
- **Under the power penalty the "loss" is J = L·R^γ**: the accept test,
  `loss_full` and `final_loss` run on it, while the per-tone losses stay plain and
  `final_mse` carries L. Off, the loop branches around it (no factor of 1.0).
- **Warm-start δ grid** derived from `freqs`/`ref_freq` with the exact op
  sequence in `warm_start.rs` (sort → positive gaps → min → `2π·f_ref/gap`;
  `n = ceil(period/(2π/64))` capped at 100 000; `δ_k = k·(period/n)`).
- **Single-threaded solve** — no rayon; the sweep parallelizes over candidates
  in worker processes (`src/rice_bend/parallel.py` pins BLAS threads to 1 for
  the same reason).
- **RNG stays in Python**; the drawn initial phase crosses the boundary, and the
  warm start's stage-1 solve reuses the same array (same seed ⇒ same draw).

> **Note (2026-10-12, updated 2026-10-13): everything below predates the 300 GHz
> retarget.** The timings, speedups and parity figures in the sections below (and
> the `parity.py` figure in the table above) were measured on the 150 GHz configs
> and have not been re-run. The command lines are kept as the record of what was
> timed; run today they would time a different workload. The scenario configs now
> sample the TX plate at λ/20 and the RX with a fixed λ/2 array, carry 30 dB of
> receiver noise, and the fine grid has grown about 4× (164×76 = 12 464
> candidates then, 327×151 = 49 377 now, so `scenario_caustic_hit_pm5.yml` is no
> longer the 12 464-point grid named below). `scenario_caustic_hit.yml` and
> `scenario_caustic_hit_pm5.yml` have also moved to the on-grid receiver
> (`rx_aperture.grid_stride: 2`), so the target and the weighting they hand the
> engines are non-zero at the 80 element samples only. Today's equivalents of the
> frequencies are `--freq 300e9` for the single-tone workloads and
> `--freq 285e9 300e9 315e9` for the pm5 comb. `python/bench.py` times exactly
> those on the 327×151 grid of `scenario_caustic_hit_pm5.yml`, with the noise
> forced off. That config is kept as the benchmark's input and keeps the legacy
> start for it: workload A times the engine's own warm start, which a config with
> `init: backprop` would replace by a start computed in Python for both engines.
> `python/parity.py` runs the seventeen slices listed above.

## Amplitude descend on the measured row (2026-10-20)

`configs/measured/measured_rx177_wide_amp.yml` as it stood for these runs: 300 GHz,
backprop start, `amplitude: descend`, `amp_max` 10, 3000 iterations, N=2401. (It
now sets `max_iters: 6000`; the runs below are at 3000.)

**Parity, Python against Rust** (`gs_reconstruct` on each engine). The five
candidates are the measured TX window (z 0.242, x −0.015) and the grid points
(0.24, −0.012), (0.10, 0), (0.01, 0) and (0.35, −0.01). Every one runs to the
3000-iteration cap on both engines. The worst rel Δloss is 3.6e-9, at
(0.10, 0); two candidates are bit-equal. The worst max |Δcurr_aper_f| is
6.6e-9, at (0.01, 0).

**`grid-search-mgs --limit 40 -j 4 --true-mgs --scenes --scene-top 3`.**
Against the Python run of the same 40 candidates:

- worst rel Δloss 2.7e-10, and 10 of the 40 losses bit-equal;
- n_iters and stop_reason identical on all 40;
- the same argmin cell;
- the stored complex64 apertures within 1 f32 ulp;
- the measured-TX solve bit-equal.

Wall time 6.1 s against 17.1 s.

**Timing.** Single process, single thread, 24 random grid candidates: Python
1.445 s per candidate, Rust 0.296 s per candidate (**4.9×**; 4.0× to 5.0× per
candidate). Through the sweep's own `map_workers` at `-j 24` on this 16-core,
32-thread machine:

| Engine | Sample | Wall | Per worker | Full grid (30 200) |
|---|---|---|---|---|
| Rust | 960 random candidates | 15.3 s | 0.38 s/cand | **≈ 8 min** |
| Python | 480 random candidates | 39.4 s | 1.97 s/cand | ≈ 41 min |

SMT contention costs about 1.7× per worker against the single-thread figures.
The full-grid figures add no setup, persistence or plots.

## Uniform level on the measured row (2026-10-20)

`configs/measured/rxsize/rx177_level_w200_coarse.yml` (the whole row) and
`rx177_level_w120_coarse.yml` (x 95–215 mm, so x_sim = x − 155 mm) as written:
300 GHz, backprop start, `amplitude: level`, 6000 iterations, N=2401.

**Parity, Python against Rust** (`gs_reconstruct` on each engine). Five points
per config: the measured TX window, and (0.24, −0.012), (0.10, 0), (0.01, 0) and
(0.35, −0.01) in the w200 frame, the same physical points in w120's (x 5 mm
lower). Those four are off the 4 mm grid, so they are built as windows of the
grid's aperture width, the way the measured `--true-mgs` point is. Every one of
the ten runs to the 6000-iteration cap on both engines, and the fitted level
agrees to the digits printed (0.48 to 0.76). The worst rel Δloss is 5.9e-11
(w120 at (0.01, −0.005)), and eight of the ten are bit-equal or within 2 ulp.
The worst max |Δcurr_aper_f| is 3.0e-9.

**One tone from the random start** (`init: warm_start` at one tone, allowed) is
the loosest case found. On 24 spread candidates of the simulated on-grid config,
two drift past 1e-9: to rel 7.4e-8 (n_iters 3000/3000) and 4.9e-5 (2690/2655).
A third, outside that set at (0.585, −0.15), reaches 5.9e-5 (2323/2339). The
fixed and descend solves of the same 24 candidates from the same start stay
within 5e-11. On the 4.9e-5 candidate the two engines are bit-identical through
iteration 63. At iteration 64 the loss differs by 1 ulp with the aperture still
identical, which is numpy's unmirrored complex abs/exp. The strict-decrease line
search flips on such near-ties, and the one shared level carries the difference
to the whole plate. Every backprop case, the configured start, stays at 2.5e-10
or below.

**`grid-search-mgs --limit 40 -j 4 --true-mgs`** on the w200 config, against the
Python run of the same 40 candidates:

- worst rel Δloss 5.0e-13, and 27 of the 40 losses bit-equal;
- n_iters and stop_reason identical on all 40;
- the same argmin cell;
- the stored apertures within 1 f32 ulp;
- the measured-TX solve within 4.1e-11.

Wall time 3.9 s against 11.4 s.

**Timing** (w200 config, 7 600 usable candidates). Single thread, 24 random
candidates (13 of them at the 6000 cap): Python 1.689 s per candidate, Rust
0.369 s per candidate (**4.6×**; 4.0× to 4.6× per candidate). At `-j 24`
through `map_workers`:

| Engine | Sample | Wall | Per worker | Full grid (7 600) |
|---|---|---|---|---|
| Rust | 960 random candidates | 19.1 s | 0.48 s/cand | **≈ 2.5 min** |
| Python | 480 random candidates | 50.0 s | 2.50 s/cand | ≈ 13 min |

## Baseline timings (M0, this machine, 2026-09-19, python engine)

| Workload | Wall time |
|---|---|
| `mgs --config configs/tiny_check.yml` (joint F=2, 40 iters + plots) | 1.0 s |
| `mgs --config configs/scenario_caustic_hit.yml --freq 150e9` (full-res single solve) | 5.3 s |
| `grid-search-mgs --config configs/scenario_caustic_hit_sparse.yml --freq 150e9 --limit 6 --jobs 1` | 3.9 s |
| `scipy.signal.fftconvolve`, N=2400 (the hot primitive) | ~119 µs/call |
| cached-kernel-spectrum FFT equivalent (what `rs_apply_into` should beat) | ~68 µs/call |

Full-scale context: one `study_frequency_grid2l` point = 12 464 candidates ×
~694 iterations ≈ 10⁸ convolutions ≈ 4.5 core-hours.

## Measured speedup (M5, 2026-09-27, `python/bench.py`)

Single process, single thread, identical inputs; 16 candidates spread evenly
over the 12 464-point `scenario_caustic_hit_pm5.yml` grid. Rust timings include
the Python-side kernel build and marshaling.

| Workload | Python s/cand | Rust s/cand | Speedup (range) | max rel Δloss | same n_iters | argmin |
|---|---|---|---|---|---|---|
| A: pm5 study point (3-tone, warm start) | 1.403 | 0.282 | **5.0×** (4.8–5.2) | 5.2e-12 | 16/16 | same |
| B: 1-frequency anchor (150 GHz) | 0.368 | 0.073 | **5.1×** (4.9–5.2) | 2.0e-13 | 16/16 | same |

Projected single-core cost of one full pm5 study point: **4.86 → 0.98
core-hours**; a 10-point study on 16 cores: ~3 h → ~37 min.

Primitive (N=2400): scipy `fftconvolve` 106.5 µs/call vs Rust
`rs_apply_into` **23.6 µs** (criterion) — 4.5×. The Rust solve is now ~90%
FFT time, so further gains need FFT-level changes (e.g. pruned transforms
exploiting the sparse support/RX window), not more loop tuning.

### End to end through the CLI (2026-09-27)

One pm5 study point on the settled 64×48 grid: `grid-search-mgs --config
configs/scenario_caustic_hit.yml --freq 142.5e9 150e9 157.5e9 --engine {python,rust}`
(3 072 candidates, 3 tones, warm start, 800 iterations, default `-j` = 32).

| | Python | Rust |
|---|---|---|
| Wall time (whole run incl. setup, persistence, plots) | 247.0 s | 56.5 s (**4.4×**) |
| argmin cell / error to truth | #680 / 5.93 mm | #680 / 5.93 mm |
| top-candidate cluster (7 cells) / mean distance | 9.54 mm | identical |

`python/compare_runs.py`: max per-candidate rel Δloss 1.3e-9, 1 506/3 072
bit-identical, n_iters and stop_reason identical 3 072/3 072. The gap from 5× to
4.4× is fixed Python-side work both runs share (estimated ~9 s: measurement,
3 072 `.npz` writes, analysis, plots). It matters less as grids grow.

Parity is far tighter than the rtol-1e-3 bar because the c64 quantization
resynchronizes the two engines: rustfft's ulp differences almost never cross an
f32 rounding boundary (100% of conv golden elements bit-equal), and the
reductions mirror numpy's pairwise summation exactly (`src/npmath.rs`). The
only unmirrored ops are numpy's SIMD complex `exp`/`abs` (1-2 ulp on 0.1-3.6%
of elements), which is why a few candidates differ at the 1e-12 level.
