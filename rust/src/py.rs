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
//!
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
//!
//!    then solve::gs_core(psi_start, ...). ALL of this stays inside
//!    allow_threads — the GIL is held only for marshaling.
//! 4. Compose curr_aper_f = amp * exp(i * phase) (mgs.py:302) and hand back
//!    numpy arrays. loss_full_per_freq reshapes row-major to (n_iters_run, F).
//!
//! stop_reason strings are exactly "converged" / "max_iters" (types.rs::as_str).

// pyo3 0.22's #[pyfunction] expansion trips this lint on every PyResult return.
#![allow(clippy::useless_conversion)]

use num_complex::{Complex32, Complex64};
use numpy::ndarray::Array2;
use numpy::{IntoPyArray, PyArray1, PyArray2, PyArrayMethods, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::conv::{fftconvolve_same as conv_fftconvolve_same, Convolver};
use crate::solve::gs_core;
use crate::types::{ChannelData, KernelPair, SolveParams};
use crate::warm_start::warm_start_psi;

fn vec1<T: numpy::Element + Copy>(a: &PyReadonlyArray1<'_, T>) -> Vec<T> {
    a.as_array().iter().copied().collect()
}

/// Rows of an (F, N) array as owned Vecs (works for non-contiguous views).
fn rows<T: numpy::Element + Copy>(a: &PyReadonlyArray2<'_, T>) -> Vec<Vec<T>> {
    a.as_array().rows().into_iter().map(|r| r.iter().copied().collect()).collect()
}

/// The full solve for one candidate. Mirrors the array-facing subset of
/// `gs_reconstruct(tx_z, orig_aper_amp, x_axis, rx_z, channels, params,
/// ref_freq, capture=False)` — geometry (tx_z/rx_z/x_axis) stays in Python,
/// which builds h_fwd from it via rs.rs_kernel and passes the kernels in.
///
/// Returns (curr_aper_f, final_loss, final_loss_per_freq, n_iters_run,
/// stop_reason, loss_full, loss_full_per_freq).
#[pyfunction]
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
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
    let warm = match init_mode {
        "warm_start" => true,
        "random" => false,
        other => {
            return Err(PyValueError::new_err(format!(
                "init_mode must be 'random' or 'warm_start', got {other:?}"
            )))
        }
    };
    let n = initial_phase.len()?;
    let n_freq = freqs.len()?;
    if n_freq == 0 || n == 0 {
        return Err(PyValueError::new_err("need at least one frequency and one sample"));
    }
    if aper_amp.len()? != n {
        return Err(PyValueError::new_err("aper_amp must have the same length as initial_phase"));
    }
    for (name, dims) in [
        ("rx_field", rx_field.dims()),
        ("h_fwd", h_fwd.dims()),
        ("error_weighting", error_weighting.dims()),
    ] {
        if dims[0] != n_freq || dims[1] != n {
            return Err(PyValueError::new_err(format!(
                "{name} has shape ({}, {}), expected ({n_freq}, {n})",
                dims[0], dims[1]
            )));
        }
    }

    let initial_phase = vec1(&initial_phase);
    let aper_amp = vec1(&aper_amp);
    let freqs = vec1(&freqs);
    let rx_rows = rows(&rx_field);
    let w_rows = rows(&error_weighting);
    let h_rows = rows(&h_fwd);
    let params = SolveParams {
        max_iters,
        convergence_count,
        convergence_threshold,
        lr0,
        bt_shrink,
        bt_tries,
    };

    let (aper_f, res) = py.allow_threads(move || {
        let support: Vec<bool> = aper_amp.iter().map(|&a| a > 0.0).collect();
        let channels: Vec<ChannelData> = freqs
            .iter()
            .zip(rx_rows)
            .zip(w_rows)
            .map(|((&freq, rx_field), error_weighting)| ChannelData {
                freq,
                rho: freq / ref_freq,
                rx_field,
                error_weighting,
            })
            .collect();
        let mut conv = Convolver::new(n);
        let kernels: Vec<KernelPair> = h_rows
            .iter()
            .map(|h| {
                let h_conj: Vec<Complex64> = h.iter().map(|c| c.conj()).collect();
                KernelPair { fwd: conv.kernel_spectrum(h), adj: conv.kernel_spectrum(&h_conj) }
            })
            .collect();

        let start = if warm && n_freq > 1 {
            warm_start_psi(
                &initial_phase, &aper_amp, &support, &channels, &kernels, ref_freq, dx, &params,
                &mut conv,
            )
        } else {
            initial_phase
        };
        let res = gs_core(&start, &aper_amp, &support, &channels, &kernels, dx, &params, &mut conv);
        let aper_f: Vec<Complex64> = aper_amp
            .iter()
            .zip(&res.aper_phase)
            .map(|(&a, &p)| {
                let (s, c) = p.sin_cos();
                Complex64::new(a * c, a * s)
            })
            .collect();
        (aper_f, res)
    });

    let n_iters = res.n_iters_run;
    let loss_pf = Array2::from_shape_vec((n_iters, n_freq), res.loss_full_per_freq)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok((
        PyArray1::from_vec_bound(py, aper_f),
        res.final_loss,
        PyArray1::from_vec_bound(py, res.final_loss_per_freq),
        n_iters,
        res.stop_reason.as_str().to_string(),
        PyArray1::from_vec_bound(py, res.loss_full),
        loss_pf.into_pyarray_bound(py),
    ))
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
    let (u0, h) = (vec1(&u0), vec1(&h));
    if u0.len() != h.len() || u0.is_empty() {
        return Err(PyValueError::new_err("u0 and h must be non-empty and the same length"));
    }
    let out: Vec<Complex32> = py.allow_threads(move || {
        conv_fftconvolve_same(&u0, &h, dx)
            .into_iter()
            .map(|v| Complex32::new(v.re as f32, v.im as f32))
            .collect()
    });
    Ok(PyArray1::from_vec_bound(py, out))
}

#[pymodule]
fn rice_bend_core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(gs_solve, m)?)?;
    m.add_function(wrap_pyfunction!(fftconvolve_same, m)?)?;
    Ok(())
}
