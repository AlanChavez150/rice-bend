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

## Build / dev loop

```bash
# pure-Rust cycle (no Python involved — the `python` feature stays off):
cd rust && cargo build && cargo test

# golden vectors (already committed; regenerate only after an approved change
# to the reference solver — meta.json records the generating commit):
.venv/bin/python rust/python/gen_golden.py

# extension module into the project venv (Python 3.8.10):
uv pip install maturin
VIRTUAL_ENV=$PWD/.venv .venv/bin/maturin develop --release -m rust/Cargo.toml

# end-to-end acceptance (M5):
.venv/bin/python rust/python/parity.py
```

Version pins: **pyo3 0.22 + numpy(rust) 0.22, abi3-py38, maturin ≥1.4,<2** — the
verified pairing for CPython 3.8.10. Do not bump without revisiting the floor.

## Work split and gates

| Package | Files | Gate | Status |
|---|---|---|---|
| FFT/convolution | `src/fft.rs`, `src/conv.rs` | `cargo test --test conv_parity` | green — 100% post-downcast bit-equal |
| numpy mirrors | `src/npmath.rs` | `cargo test --test numerics_parity` | green — bit-exact |
| Solver core | `src/solve.rs`, `src/warm_start.rs` | `cargo test --test solve_parity` | green — final loss bit-identical |
| PyO3 boundary | `src/py.rs` | `maturin develop` + `python/parity.py` | green — worst rel Δloss 5e-14 |

Alan wrote `next_fast_len`; Claude wrote the harness and, at Alan's request,
the rest of the implementation. Every module's doc comment carries its
semantics contract, with `src/rice_bend/mgs.py` / `rs.py` line references as
the source of truth. The load-bearing traps, in one place:

- **complex64 quantization** exactly at the two `rs_apply` outputs, nowhere else.
- **Adjoint spectrum from the time-domain conjugate** (`FFT(pad(conj(h)))`),
  never `conj(FFT(pad(h)))`.
- **f32 convergence window**: losses stored f32, diff/mean at f32, strictly
  `iter_idx > convergence_count`, window excludes the current index.
- **`loss_full` holds PRE-step losses**; `final_loss` is the last iteration's
  POST-accept loss — they differ whenever the last iteration accepted a step.
- **Warm-start δ grid** derived from `freqs`/`ref_freq` with the exact op
  sequence in `warm_start.rs` (sort → positive gaps → min → `2π·f_ref/gap`;
  `n = ceil(period/(2π/64))` capped at 100 000; `δ_k = k·(period/n)`).
- **Single-threaded solve** — no rayon; the sweep parallelizes over candidates
  in worker processes (`src/rice_bend/parallel.py` pins BLAS threads to 1 for
  the same reason).
- **RNG stays in Python**; the drawn initial phase crosses the boundary, and the
  warm start's stage-1 solve reuses the same array (same seed ⇒ same draw).

## Baseline timings (M0, this machine, 2026-09-19, python engine)

| Workload | Wall time |
|---|---|
| `mgs --config configs/tiny_check.yml` (joint F=2, 40 iters + plots) | 1.0 s |
| `mgs --config configs/scenario_caustic_hit.yml --freq 150e9` (full-res single solve) | 5.3 s |
| `grid-search-mgs scenario_caustic_hit_sparse --freq 150e9 --limit 6 --jobs 1` | 3.9 s |
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
