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
//! Known accepted divergence: numpy's mean over a small f32 array uses an 8-way
//! unrolled pairwise sum; a plain sequential f32 sum can differ in the last ulp
//! and (rarely, when the mean sits within an ulp of the threshold) move the stop
//! decision by an iteration. The parity bar treats n_iters as report-only for
//! exactly this reason — do not chase bit-parity here.
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

use crate::conv::Convolver;
use crate::types::{ChannelData, CoreResult, KernelPair, SolveParams};

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
    let _ = (initial_phase, aper_amp, support, channels, kernels, dx, params, conv);
    let _ = Complex64::new(0.0, 0.0);
    todo!("solver work package: the per-iteration recipe above")
}
