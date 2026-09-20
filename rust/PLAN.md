# Plan: Rust engine for the rice-bend solver core

Status-tracked version of the approved plan (2026-09-19). `README.md` covers
how to build, the trap list, and baseline timings; this file tracks what
remains and in what order. Update the checkboxes as gates go green.

## Goal

Grid-search studies are the multi-hour cost center: up to 12,464 candidate
solves per study point, ~694 solver iterations each, every iteration bottoming
out in `rs.rs_apply` (a 2400-point complex fftconvolve, ~10 calls/iteration)
— ~10⁸ convolutions ≈ 4.5 core-hours per study point. A Rust engine for this
path targets **5–10× on sweeps** (~45 core-hours → ~6–10).

## Settled decisions

- **Python is the golden reference.** Nothing under `src/rice_bend/` changes in
  this phase; `scripts/characterize.sh` must stay byte-identical. Golden
  vectors and parity checks import `rice_bend` read-only.
- **Parity bar**: per-candidate `final_loss` rtol ≤ 1e-3 + exact argmin /
  top-candidate set. `n_iters_run`/`stop_reason` reported, never gated.
  Bit-identity is NOT required (rustfft ≠ pocketfft) — except the complex64
  quantization at the two `rs_apply` outputs (≥99% bit-equal, rest ≤1 ulp).
- **Engine selection**: explicit opt-in only, wired in a later plan. No silent
  fallback.
- **Alan writes the Rust** (all three work packages); Claude built the harness
  (stubs + contracts, golden vectors, gate tests, parity script, benches) and
  reviews each package as it lands.
- Pins: pyo3 0.22 + rust-numpy 0.22, abi3-py38 (CPython 3.8.10 venv),
  maturin ≥1.4,<2. Single-threaded solve — no rayon; parallelism stays at the
  sweep's process-pool-over-candidates level.
- RNG stays in Python; `freqs` + `ref_freq` cross the boundary (not rho or the
  δ grid) so ulp-sensitive derivations happen once, in Rust, with the pinned
  op sequence.

## Milestones

### Done

- [x] **M0 — Baseline** (Claude): characterize green before any work; timings
      recorded in README.md (tiny 1.0 s, full-res solve 5.3 s, limit-6 sweep
      3.9 s, fftconvolve ~119 µs / cached-spectrum ~68 µs).
- [x] **M1 — Harness** (Claude): crate scaffolding; six stub files with the
      semantics contracts in doc comments; golden vectors
      (`tests/data/`, generated at reference commit `edff357`); gate tests
      red only on `todo!()`; `parity.py` + criterion benches; maturin/pyo3
      pinning verified live (stub extension builds abi3-py38 and imports).

### In progress

- [ ] **M2 — FFT/convolution layer** (Alan): `src/fft.rs` + `src/conv.rs`.
      Gate: `cargo test --test conv_parity` — all 3 tests.
  - [x] `next_fast_len` ({2,3,5,7,11}-smooth, matches scipy on all 5 sizes)
  - [ ] `FftPair` (rustfft plans + scratch, unnormalized inverse)
  - [ ] `Convolver::rs_apply_f64_into` → `pre_downcast_rel_err` green
        (rel err ≤ 1e-12 vs scipy at complex128)
  - [ ] `Convolver::rs_apply_into` (c64 round-trip per element) →
        `post_downcast_bit_parity` green (≥99% bit-equal, rest ≤1 ulp)

### Up next (dependency order)

- [ ] **M3 — Solver core** (Alan): `src/solve.rs::gs_core` — the per-iteration
      recipe, backtracking, f32 convergence window. Read `mgs.py:251-301` side
      by side with the stub before writing.
      Gate: `solve_parity::single_freq_solve` — first-iteration loss rtol
      ≤ 1e-6, final_loss rtol ≤ 1e-3.
- [ ] **M4 — Warm start** (Alan): `src/warm_start.rs` — stage-1 solve, unwrap,
      δ-scan (literal or analytic, same grid + strict-< first-win).
      Gate: `solve_parity::joint_delta_scan` (δ equal on the shared grid, or
      scan loss rtol ≤ 1e-5) + `joint_warm_start_solve` (final_loss rtol ≤ 1e-3).
- [ ] **M5 — PyO3 boundary + acceptance** (Alan + Claude): `src/py.rs`
      marshaling per the stub recipe; then
      `VIRTUAL_ENV=$PWD/.venv .venv/bin/maturin develop --release -m rust/Cargo.toml`
      and `.venv/bin/python rust/python/parity.py` green (tiny_check joint +
      the characterize limit-6 slice; rtol 1e-3 + exact argmin). Record
      criterion + parity timings in README.md.

### Deferred — needs a separate approved plan

- [ ] Engine wiring into `src/rice_bend/`: `engine` config field + `--engine`
      CLI flag on `mgs` / `grid-search-mgs` / `mgs-study`.
- [ ] The numpy quick wins in the reference engine: cached kernel spectra in
      the solver loop (measured 1.6×, bit-identical) and the analytic δ-scan
      behind an opt-in config field.
- [ ] Full-scale validation: one full 64×48 study point, python vs rust,
      `--jobs 4` — argmin cell exact, top-5 set equal, timings recorded.
- [ ] Rayon-over-candidates replacing the process pool (maybe never).
- [ ] Scene-render / `hankel1` port (plotting only — likely never).

## Verification

```bash
cd rust && cargo test                 # M2-M4 red→green tracker (no Python)
bash scripts/characterize.sh          # must stay green, byte-identical, always
.venv/bin/python rust/python/parity.py  # M5 acceptance (needs maturin develop)
cargo bench                           # after M2: compare vs README baselines
```

Golden vectors are regenerated only after an approved change to the reference
solver: `.venv/bin/python rust/python/gen_golden.py` (records the generating
commit in `tests/data/meta.json`).
