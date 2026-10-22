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
//!   - the support: mgs.gs_reconstruct's `support = |orig_aper_amp| > 0`, passed
//!     as the trailing `support` argument. Left out (None) it is DERIVED here as
//!     aper_amp > 0, which is the same array under amplitude 'fixed' (amp =
//!     |orig_aper_amp|) — the legacy callers rely on that. Under 'descend' it is
//!     REQUIRED: aper_amp is then the start amplitude a0 (mgs._backprop_amplitude),
//!     which can be 0 at window samples Python still descends, so a0 > 0 would
//!     silently shrink the support. 'level' needs it too: its plate A0 * 1_support
//!     has no support of its own when A0 == 0.
//!   - the amplitude mode: trailing `amplitude` ('fixed' default | 'descend' |
//!     'level') and `amp_max` (required with 'descend', finite and > 0; refused
//!     with 'fixed' and 'level'). 'level' takes its start A0 the way 'descend'
//!     takes a0: as aper_amp, here A0 on the support and 0 off it, which must be
//!     one value on the support (checked). 'level' with several tones and
//!     init_mode 'warm_start' is refused, as mgs.check_amplitude_mode refuses it.
//!   - the excess-power penalty: trailing `power_penalty` (gamma) and `rx_power`
//!     (P_rx), both or neither (default None: the plain loss, bit-identical to a
//!     build without them). P_rx is computed IN PYTHON (mgs._received_power) and
//!     crosses as a scalar, so both engines divide by the identical number rather
//!     than each summing |rx|^2 over the elements its own way. Both must be finite
//!     and > 0, and the penalty is refused with amplitude 'fixed' (R would be a
//!     constant of the grid), as the config refuses it.
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
//! 4. Compose curr_aper_f = amp * exp(i * phase) (mgs.py:302) from the solve's
//!    final amplitude (the input one under 'fixed') and hand back
//!    numpy arrays. loss_full_per_freq reshapes row-major to (n_iters_run, F).
//!    final_mse and power_ratio (None when the penalty is off) trail the tuple.
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
use crate::types::{AmplitudeMode, ChannelData, KernelPair, PowerPenalty, SolveParams};
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
/// stop_reason, loss_full, loss_full_per_freq, final_mse, power_ratio). Under
/// 'descend' curr_aper_f carries the descended amplitude. Under the power penalty
/// final_loss (and loss_full) is the objective J = final_mse * power_ratio^gamma,
/// final_mse the plain joint loss, final_loss_per_freq the plain per-tone losses;
/// off, final_mse == final_loss and power_ratio is None.
///
/// The trailing `amplitude`, `amp_max`, `support`, `power_penalty`, `rx_power`
/// default to the legacy behaviour (fixed amplitude, support = aper_amp > 0, no
/// penalty), so positional callers that predate them run unchanged.
#[pyfunction]
#[pyo3(signature = (
    initial_phase, aper_amp, freqs, ref_freq, rx_field, error_weighting, h_fwd, dx,
    init_mode, max_iters, convergence_count, convergence_threshold, lr0, bt_shrink,
    bt_tries, amplitude = "fixed", amp_max = None, support = None, power_penalty = None,
    rx_power = None
))]
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
    amplitude: &str,
    amp_max: Option<f64>,
    support: Option<PyReadonlyArray1<'py, bool>>,
    power_penalty: Option<f64>,
    rx_power: Option<f64>,
) -> PyResult<(
    Bound<'py, PyArray1<Complex64>>,
    f64,
    Bound<'py, PyArray1<f64>>,
    usize,
    String,
    Bound<'py, PyArray1<f32>>,
    Bound<'py, PyArray2<f32>>,
    f64,
    Option<f64>,
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
    let amplitude = match (amplitude, amp_max) {
        ("fixed", None) => AmplitudeMode::Fixed,
        ("fixed", Some(v)) => {
            return Err(PyValueError::new_err(format!(
                "amp_max ({v}) is set but amplitude is 'fixed', which caps nothing"
            )))
        }
        ("descend", Some(v)) if v.is_finite() && v > 0.0 => AmplitudeMode::Descend { amp_max: v },
        ("descend", v) => {
            return Err(PyValueError::new_err(format!(
                "amplitude 'descend' needs amp_max, a finite number > 0; got {v:?}"
            )))
        }
        ("level", None) => AmplitudeMode::Level,
        ("level", Some(v)) => {
            return Err(PyValueError::new_err(format!(
                "amp_max ({v}) is set but amplitude is 'level', which has no cap"
            )))
        }
        (other, _) => {
            return Err(PyValueError::new_err(format!(
                "amplitude must be 'fixed', 'descend' or 'level', got {other:?}"
            )))
        }
    };
    if amplitude == AmplitudeMode::Level && warm && n_freq > 1 {
        // mgs.check_amplitude_mode's refusal: the warm start's level would come
        // from the seeded random phase, and lands on a wrong level
        return Err(PyValueError::new_err(
            "amplitude: level with several tones needs init: backprop",
        ));
    }
    let power_penalty = match (power_penalty, rx_power) {
        (None, None) => None,
        (Some(gamma), Some(rx_power)) => {
            if !(gamma.is_finite() && gamma > 0.0) {
                return Err(PyValueError::new_err(format!(
                    "power_penalty (gamma) must be a finite number > 0; got {gamma}"
                )));
            }
            if !(rx_power.is_finite() && rx_power > 0.0) {
                return Err(PyValueError::new_err(format!(
                    "rx_power (the received power P_rx) must be a finite number > 0; got {rx_power}"
                )));
            }
            if amplitude == AmplitudeMode::Fixed {
                return Err(PyValueError::new_err(
                    "power_penalty needs a fitted amplitude ('level' or 'descend'): under \
                     'fixed' the plate power, and so the penalty, is a constant of the grid",
                ));
            }
            Some(PowerPenalty { gamma, rx_power })
        }
        (gamma, rx_power) => {
            return Err(PyValueError::new_err(format!(
                "power_penalty and rx_power go together (both or neither); got \
                 power_penalty={gamma:?}, rx_power={rx_power:?}"
            )))
        }
    };
    let support: Option<Vec<bool>> = match support {
        Some(s) => {
            if s.len()? != n {
                return Err(PyValueError::new_err("support must have the same length as initial_phase"));
            }
            Some(vec1(&s))
        }
        None if amplitude != AmplitudeMode::Fixed => {
            return Err(PyValueError::new_err(
                "amplitude 'descend' and 'level' need the support (|orig_aper_amp| > 0): \
                 the start amplitude can be 0 inside it, so aper_amp > 0 is not the support",
            ))
        }
        None => None,
    };
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
    if let Some(s) = &support {
        // u0 is built on the support only, so an amplitude off it would be dropped
        // where Python's u0 = amp * exp(...) keeps it
        if let Some(j) = (0..n).find(|&j| !s[j] && aper_amp[j] != 0.0) {
            return Err(PyValueError::new_err(format!(
                "aper_amp must be 0 off the support; aper_amp[{j}] = {}",
                aper_amp[j]
            )));
        }
        if amplitude == AmplitudeMode::Level {
            // A0 * 1_support: one value on the support, which gs_core reads back
            if let Some(j0) = (0..n).find(|&j| s[j]) {
                if let Some(j) = (j0..n).find(|&j| s[j] && aper_amp[j].to_bits() != aper_amp[j0].to_bits()) {
                    return Err(PyValueError::new_err(format!(
                        "amplitude 'level' needs aper_amp to be one level on the support; \
                         aper_amp[{j0}] = {} but aper_amp[{j}] = {}",
                        aper_amp[j0], aper_amp[j]
                    )));
                }
            }
        }
    }
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
        amplitude,
        power_penalty,
    };

    let (aper_f, res) = py.allow_threads(move || {
        let support: Vec<bool> =
            support.unwrap_or_else(|| aper_amp.iter().map(|&a| a > 0.0).collect());
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
        let aper_f: Vec<Complex64> = res
            .aper_amp
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
        res.final_mse,
        res.power_ratio,
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
