//! The PyO3 boundary — the ONLY module that speaks Python. Compiled solely under
//! the `python` feature (maturin builds it; `cargo test` never does).
//!
//! WORK PACKAGE: PyO3 boundary. Gate: `maturin develop --release -m rust/Cargo.toml`
//! builds + `python rust/python/parity.py` green.
//!
//! ## What crosses, and what deliberately does not
//!
//! Crossing per solve (once per candidate — microseconds against ~0.3-3 s of
//! solve, so marshaling cost is irrelevant):
//!   - the seeded initial phase, drawn IN PYTHON (`np.random.default_rng(seed)
//!     .random(N) * 2*pi`, mgs.py:207-216). The RNG never moves to Rust: PCG64 +
//!     SeedSequence bit-parity is not worth reimplementing.
//!   - freqs + ref_freq instead of rho or any precomputed scan grid: rho[f] =
//!     freqs[f] / ref_freq is one IEEE division (bit-exact both sides), and the
//!     warm-start delta grid derives from freqs with the exact op sequence
//!     pinned in warm_start.rs — recomputing from first principles in Rust
//!     avoids every "computed slightly differently in two places" drift.
//!   - support is DERIVED here as aper_amp > 0 (mgs.py:215,219 guarantee amp =
//!     |orig_aper_amp|, so the two definitions coincide).
//! Not crossing: capture/history (python engine only), the config object
//! (scalars are exploded into arguments so the crate never parses pydantic),
//! kernels' Hankel build (scipy builds h_fwd once per solve; only the spectra
//! are computed here).
//!
//! ## Marshaling recipe for gs_solve
//!
//! 1. Validate shapes: all (F, N) arrays agree, initial_phase/aper_amp are (N,),
//!    freqs is (F,), F >= 1. Return PyValueError on mismatch, never panic.
//! 2. Copy inputs into owned Vecs (as_array().to_owned() row by row — inputs may
//!    be non-contiguous views).
//! 3. Build ChannelData (rho = freq / ref_freq) and, inside `py.allow_threads`,
//!    the Convolver + KernelPair spectra (adjoint from time-domain conj), then:
//!      - init_mode == "warm_start" && F > 1: psi_start = warm_start::warm_start_psi(...)
//!      - else: psi_start = initial_phase (the "random" path; also the F == 1 path)
//!    then solve::gs_core(psi_start, ...). ALL of this stays inside
//!    allow_threads — the GIL is held only for marshaling.
//! 4. Compose curr_aper_f = amp * exp(i * phase) (mgs.py:302) and hand back
//!    numpy arrays. loss_full_per_freq reshapes row-major to (n_iters_run, F).
//!
//! stop_reason strings are exactly "converged" / "max_iters" (types.rs::as_str).

use num_complex::{Complex32, Complex64};
use numpy::{PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;

/// The full solve for one candidate. Mirrors the array-facing subset of
/// `gs_reconstruct(tx_z, orig_aper_amp, x_axis, rx_z, channels, params,
/// ref_freq, capture=False)` — geometry (tx_z/rx_z/x_axis) stays in Python,
/// which builds h_fwd from it via rs.rs_kernel and passes the kernels in.
///
/// Returns (curr_aper_f, final_loss, final_loss_per_freq, n_iters_run,
/// stop_reason, loss_full, loss_full_per_freq).
#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn gs_solve<'py>(
    py: Python<'py>,
    initial_phase: PyReadonlyArray1<'py, f64>,
    aper_amp: PyReadonlyArray1<'py, f64>,
    freqs: PyReadonlyArray1<'py, f64>,
    ref_freq: f64,
    rx_field: PyReadonlyArray2<'py, Complex64>,
    error_weighting: PyReadonlyArray2<'py, f64>,
    h_fwd: PyReadonlyArray2<'py, Complex64>,
    dx: f64,
    init_mode: &str,
    max_iters: usize,
    convergence_count: usize,
    convergence_threshold: f64,
    lr0: f64,
    bt_shrink: f64,
    bt_tries: usize,
) -> PyResult<(
    Bound<'py, PyArray1<Complex64>>,
    f64,
    Bound<'py, PyArray1<f64>>,
    usize,
    String,
    Bound<'py, PyArray1<f32>>,
    Bound<'py, PyArray2<f32>>,
)> {
    let _ = (
        py, initial_phase, aper_amp, freqs, ref_freq, rx_field, error_weighting, h_fwd, dx,
        init_mode, max_iters, convergence_count, convergence_threshold, lr0, bt_shrink, bt_tries,
    );
    todo!("PyO3 work package: marshaling recipe above")
}

/// Test/smoke hook: the quantized rs_apply for one pair, so parity can be probed
/// from Python without a full solve. Mirrors rs.rs_apply(u0, h, dx) — including
/// its dtype: the return is numpy complex64 (Complex<f32>), narrowed from the
/// internal quantized-c128 representation.
#[pyfunction]
pub fn fftconvolve_same<'py>(
    py: Python<'py>,
    u0: PyReadonlyArray1<'py, Complex64>,
    h: PyReadonlyArray1<'py, Complex64>,
    dx: f64,
) -> PyResult<Bound<'py, PyArray1<Complex32>>> {
    let _ = (py, u0, h, dx);
    todo!("PyO3 work package: crate::conv::fftconvolve_same, each value narrowed to Complex32")
}

#[pymodule]
fn rice_bend_core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(gs_solve, m)?)?;
    m.add_function(wrap_pyfunction!(fftconvolve_same, m)?)?;
    Ok(())
}
