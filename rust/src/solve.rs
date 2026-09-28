//! The MGS descent loop — the Rust twin of the iteration body of
//! `gs_reconstruct` (src/rice_bend/mgs.py:247-335). This is ~90% of all compute
//! in a grid sweep: ~694 iterations x F frequencies x (2 + backtracking tries)
//! convolutions per candidate.
//!
//! WORK PACKAGE: solver core. Gate: single-frequency case in
//! `cargo test --test solve_parity` (first-iteration loss rtol <= 1e-6,
//! final_loss rtol <= 1e-3; n_iters/stop_reason are report-only).
//!
//! ## Per-iteration recipe (mgs.py:251-301), per channel f:
//!
//!   u0[j]      = amp[j] * exp(i * rho[f] * psi[j])           // complex128
//!   prop       = rs_apply(u0, h_fwd[f])                      // QUANTIZED c64
//!   r[j]       = w[f][j] * (prop[j] - rx[f][j])              // c128 (numpy promotes)
//!   loss_pf[f] = 0.5 * mean(|r|^2)                           // f64 accumulate
//!   g          = rs_apply(r, h_adj[f])                       // QUANTIZED c64
//!   grad[j]   += rho[f] * 2.0 * Im(g[j] * conj(u0[j]))       // f64
//!
//! then:
//!   grad /= F  (mean-loss gradient — dividing by F is load-bearing, mgs.py:273-276)
//!   grad[!support] = 0
//!   loss = mean(loss_pf)                                     // f64
//!   loss_full[iter] = loss as f32; loss_full_per_freq[iter][f] = loss_pf[f] as f32
//!     (PRE-step values — recorded BEFORE backtracking, mgs.py:279-281)
//!
//! ## Backtracking line search (mgs.py:285-301)
//!
//!   step = lr0
//!   repeat at most bt_tries times:
//!     theta_trial = psi - step * grad         // full axis, off-support included
//!     recompute the F forward losses for theta_trial (same quantized rs_apply)
//!     if loss_trial < loss (STRICT):          // sufficient decrease, first win
//!       psi = theta_trial; psi[!support] = 0  // zeroed only on ACCEPT
//!       loss = loss_trial; loss_pf = trial_pf
//!       break
//!     step *= bt_shrink
//!   (no accept => psi unchanged for this iteration)
//!
//! Note the initial psi is NOT zeroed off support (mgs.py:216 leaves the random
//! draw there); off-support psi only becomes 0 at the first accepted step. It
//! never affects any computed value because amp is 0 off support — but preserve
//! the behaviour so the returned phase matches.
//!
//! ## Convergence test (mgs.py:308-316) — the f32 trap
//!
//! Checked only when iter_idx > convergence_count (STRICTLY greater). Window =
//! loss_full[iter_idx - convergence_count .. iter_idx] — f32 values, EXCLUDING
//! the current index. Compute diff (f32 subtractions) then the mean, compared as
//! `mean > convergence_threshold` => stop with StopReason::Converged and
//! n_iters_run = iter_idx + 1. The losses MUST be read back at f32; running this
//! window in f64 skews n_iters systematically.
//!
//! The window mean mirrors numpy's pairwise f32 summation exactly (npmath.rs).
//! n_iters can still differ from Python when the LOSSES differ — numpy's SIMD
//! complex exp/abs are not mirrored (1-2 ulp) — so the parity bar keeps n_iters
//! report-only.
//!
//! ## What is returned
//!
//! final_loss / final_loss_per_freq are the POST-accept values of the last
//! iteration; loss_full holds the PRE-step values (so loss_full[last] !=
//! final_loss whenever the last iteration accepted a step). aper_phase is psi
//! after the last iteration.
//!
//! ## Performance notes (why this is worth writing in Rust)
//!
//! - Allocate every buffer ONCE before the loop (u0, residual, gradient, trial
//!   phase, the Convolver's internals). The Python loop builds ~40 temporaries
//!   per iteration; the Rust loop should build none.
//! - Fuse the elementwise passes where convenient (u0 build + forward staging,
//!   residual + loss accumulation). Keep the QUANTIZED rs_apply outputs as the
//!   values downstream math sees — fusing must not skip the c64 round-trip.
//! - Single-threaded by design: the sweep already runs one candidate per worker
//!   process (src/rice_bend/parallel.py pins BLAS threads to 1 for the same
//!   reason). No rayon inside the solve.

use num_complex::Complex64;

use crate::conv::{Convolver, KernelSpectrum};
use crate::npmath::{np_mean_f64, np_sum_f32};
use crate::types::{ChannelData, CoreResult, KernelPair, SolveParams, StopReason};

/// Preallocated per-solve work buffers: the loop body allocates nothing.
struct Work {
    u0: Vec<Complex64>,
    prop: Vec<Complex64>,
    r: Vec<Complex64>,
    abs2: Vec<f64>,
    g: Vec<Complex64>,
}

impl Work {
    fn new(n: usize) -> Self {
        let z = Complex64::new(0.0, 0.0);
        Work { u0: vec![z; n], prop: vec![z; n], r: vec![z; n], abs2: vec![0.0; n], g: vec![z; n] }
    }
}

/// u0 = amp * exp(i * rho * psi), evaluated on the support only: amp is 0 off
/// the support, so u0 is 0 there and the costly sin_cos can be skipped.
fn build_u0(u0: &mut [Complex64], aper_amp: &[f64], support_idx: &[usize], rho: f64, psi: &[f64]) {
    for &j in support_idx {
        let (s, c) = (rho * psi[j]).sin_cos();
        let a = aper_amp[j];
        u0[j] = Complex64::new(a * c, a * s);
    }
}

/// Forward-propagate the current u0 for one channel and return its loss
/// 0.5 * mean(|w (prop - rx)|^2), leaving the weighted residual in `work.r`.
fn channel_loss(work: &mut Work, ch: &ChannelData, fwd: &KernelSpectrum, dx: f64, conv: &mut Convolver) -> f64 {
    conv.rs_apply_into(&work.u0, fwd, dx, &mut work.prop);
    for j in 0..work.r.len() {
        let d = work.prop[j] - ch.rx_field[j];
        let w = ch.error_weighting[j];
        let r = Complex64::new(w * d.re, w * d.im);
        work.r[j] = r;
        work.abs2[j] = r.re * r.re + r.im * r.im;
    }
    0.5 * np_mean_f64(&work.abs2)
}

/// Run the descent from `initial_phase`. Inputs are on the full scene x-axis
/// (length N == conv.n_signal); `channels` and `kernels` are index-aligned.
///
/// `support[j]` == (aper_amp[j] > 0) — computed by the caller from the SAME amp
/// array (mgs.py:219 uses |orig_aper_amp| > 0 and amp = |orig_aper_amp|, so the
/// two are consistent by construction).
#[allow(clippy::too_many_arguments)]
pub fn gs_core(
    initial_phase: &[f64],
    aper_amp: &[f64],
    support: &[bool],
    channels: &[ChannelData],
    kernels: &[KernelPair],
    dx: f64,
    params: &SolveParams,
    conv: &mut Convolver,
) -> CoreResult {
    let n = conv.n_signal;
    let n_freq = channels.len();
    assert!(n_freq >= 1, "gs_core needs at least one channel");
    assert_eq!(kernels.len(), n_freq);
    assert!(initial_phase.len() == n && aper_amp.len() == n && support.len() == n);

    let support_idx: Vec<usize> = (0..n).filter(|&j| support[j]).collect();
    let mut work = Work::new(n);
    let mut psi = initial_phase.to_vec();
    let mut off_support_zeroed = false;
    let mut theta = vec![0.0; n];
    let mut grad = vec![0.0; n];
    let mut loss_pf = vec![0.0; n_freq];
    let mut trial_pf = vec![0.0; n_freq];
    let mut hist = vec![0.0f32; params.max_iters];
    let mut hist_pf = vec![0.0f32; params.max_iters * n_freq];
    let mut diffs = Vec::with_capacity(params.convergence_count);

    let mut stop_reason = StopReason::MaxIters;
    let mut n_iters_run = params.max_iters;
    let mut loss = f64::NAN;

    for iter_idx in 0..params.max_iters {
        // forward + adjoint per channel; gradient of the MEAN loss
        grad.fill(0.0);
        for (f, (ch, kp)) in channels.iter().zip(kernels).enumerate() {
            build_u0(&mut work.u0, aper_amp, &support_idx, ch.rho, &psi);
            loss_pf[f] = channel_loss(&mut work, ch, &kp.fwd, dx, conv);
            conv.rs_apply_into(&work.r, &kp.adj, dx, &mut work.g);
            for &j in &support_idx {
                // imag(g * conj(u0)), in numpy's complex-multiply operand order
                let (g, u) = (work.g[j], work.u0[j]);
                let im = g.re * (-u.im) + g.im * u.re;
                grad[j] += ch.rho * (2.0 * im);
            }
        }
        for &j in &support_idx {
            grad[j] /= n_freq as f64;
        }

        loss = np_mean_f64(&loss_pf);
        hist[iter_idx] = loss as f32;
        for f in 0..n_freq {
            hist_pf[iter_idx * n_freq + f] = loss_pf[f] as f32;
        }

        // backtracking line search on the joint loss (strict decrease, first win)
        let mut step = params.lr0;
        for _ in 0..params.bt_tries {
            for &j in &support_idx {
                theta[j] = psi[j] - step * grad[j];
            }
            for (f, (ch, kp)) in channels.iter().zip(kernels).enumerate() {
                build_u0(&mut work.u0, aper_amp, &support_idx, ch.rho, &theta);
                trial_pf[f] = channel_loss(&mut work, ch, &kp.fwd, dx, conv);
            }
            let loss_trial = np_mean_f64(&trial_pf);
            if loss_trial < loss {
                for &j in &support_idx {
                    psi[j] = theta[j];
                }
                if !off_support_zeroed {
                    for j in 0..n {
                        if !support[j] {
                            psi[j] = 0.0;
                        }
                    }
                    off_support_zeroed = true;
                }
                loss = loss_trial;
                loss_pf.copy_from_slice(&trial_pf);
                break;
            }
            step *= params.bt_shrink;
        }

        // convergence: mean(diff(recent f32 losses)), in f32, current index excluded
        let count = params.convergence_count;
        if iter_idx > count {
            diffs.clear();
            diffs.extend(hist[iter_idx - count..iter_idx].windows(2).map(|p| p[1] - p[0]));
            let flatness = np_sum_f32(&diffs) / diffs.len() as f32;
            if f64::from(flatness) > params.convergence_threshold {
                stop_reason = StopReason::Converged;
                n_iters_run = iter_idx + 1;
                break;
            }
        }
    }

    hist.truncate(n_iters_run);
    hist_pf.truncate(n_iters_run * n_freq);
    CoreResult {
        aper_phase: psi,
        final_loss: loss,
        final_loss_per_freq: loss_pf,
        n_iters_run,
        stop_reason,
        loss_full: hist,
        loss_full_per_freq: hist_pf,
    }
}
