# rice-bend-core — Rust engine for the MGS solver hot path

> Milestone status and the remaining work order live in [PLAN.md](PLAN.md).

The Python implementation under `src/rice_bend/` is **the reference** and is not
modified by this work: `scripts/characterize.sh` pins it byte-for-byte, and this
crate is a second, tolerance-validated engine for the sweep's compute
(`gs_reconstruct`, ~90% of a grid search's multi-hour cost). Engine wiring into
the CLI (`--engine rust`) is deliberately deferred to a later, separately
approved change; until then the extension is exercised only by
`rust/python/parity.py`.

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

| Package | Files | Author | Gate |
|---|---|---|---|
| FFT/convolution | `src/fft.rs`, `src/conv.rs` | Alan | `cargo test --test conv_parity` |
| Solver core | `src/solve.rs`, `src/warm_start.rs` | Alan | `cargo test --test solve_parity` |
| PyO3 boundary | `src/py.rs` (+`src/types.rs` if fields need adjusting) | Alan | `maturin develop` builds, `parity.py` green |
| Scaffolding, contracts, golden vectors, parity/bench harness | everything else here | Claude | `cargo build` clean; vectors committed |

Suggested order: conv → solve → warm start → py (each test layer builds on the
previous one). Every stub's doc comment carries the exact semantics contract,
with `src/rice_bend/mgs.py` / `rs.py` line references as the source of truth.
The load-bearing traps, in one place:

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
~694 iterations ≈ 10⁸ convolutions ≈ 4.5 core-hours; target is 5–10× on that.
Record the M5 numbers (criterion + `parity.py` timings + one full 64×48 point)
here when they exist.
