//! Multi-wavelength warm start — the Rust twin of `_warm_start_phase`
//! (src/rice_bend/mgs.py:37-116): reference-frequency solve -> unwrap ->
//! absolute-offset scan. Runs once per candidate, BEFORE the joint descent,
//! only when init == "warm_start" and F > 1 (a single frequency always takes
//! the random path — that equivalence is a documented guarantee).
//!
//! WORK PACKAGE: solver core. Gate: joint case in `cargo test --test
//! solve_parity` (chosen delta equal on the shared grid, or the scan loss at
//! the chosen delta within rtol 1e-5; joint final_loss rtol <= 1e-3).
//!
//! ## Stage 1 — reference solve (mgs.py:63-71)
//!
//! ref channel index c = argmin_f |rho[f] - 1.0| (numpy argmin: FIRST index
//! wins ties). Run `gs_core` on that single channel with the SAME params and
//! the SAME initial_phase the parent was given — the Python side draws the
//! random phase once from the resolved seed, and the recursive stage-1 solve
//! reproduces exactly that draw (same seed => same array, mgs.py:68,207-216).
//!
//! ## Stage 2 — unwrap (mgs.py:74-76)
//!
//! psi0 = zeros(N);
//! over the support indices IN ORDER: a = angle(amp * exp(i * theta_stage1))
//!   — compute the complex field first and take atan2(im, re), matching the
//!   Python data flow through np.angle (the result is theta wrapped to
//!   (-pi, pi], but going through the field keeps the same rounding);
//! then np.unwrap over that 1-D sequence, period 2*pi:
//!   dd    = diff(a)
//!   ddmod = ((dd + pi) mod 2pi) - pi        // Python % semantics: result in [0, 2pi)
//!   where ddmod == -pi && dd > 0: ddmod = pi
//!   corr  = ddmod - dd; where |dd| < pi: corr = 0
//!   out[k] = a[k] + cumsum(corr)[k-1]  (out[0] = a[0])
//! finally psi0[support] = out / rho[c].
//! The support is CONNECTED for every shipped scene (contiguous windows), so
//! "support indices in order" is one contiguous run.
//!
//! ## Stage 3 — the absolute-offset scan (mgs.py:78-107)
//!
//! Grid (replicate mgs.py:82-93 op-for-op — the ulp-sensitive part):
//!   freqs_sorted = sort(freqs); gaps = diff(freqs_sorted) keeping only > 0
//!   period = 2*pi * ref_freq / min(gaps)        // f64, this exact expression
//!   n = ceil(period / (2*pi / 64.0)) as usize   // WARM_START_SCAN_STEP
//!   if n > 100_000 { n = 100_000 }              // WARM_START_MAX_SAMPLES
//!   delta_k = (k as f64) * (period / n as f64)  // np.linspace endpoint=False
//!
//! base_props[f] = rs_apply(amp * exp(i * rho[f] * psi0), h_fwd[f]) — QUANTIZED
//! c64, one per channel, computed once (propagation linearity).
//!
//! Reference scan loop (mgs.py:99-107): for each delta, per channel
//!   r = w * (exp(i*rho[f]*delta) * base_props[f] - rx[f])   // c128
//!   loss += 0.5 * mean(|r|^2)
//! loss /= F; keep the STRICTLY smaller loss, first win.
//!
//! ANALYTIC FORM (allowed, and the reason this scan is cheap in Rust):
//!   loss(delta) = C - (1/F) * sum_f |S_f| * cos(rho[f]*delta + arg S_f)
//!   S_f = mean(w^2 * b_f * conj(rx_f)),  C = (1/F) * sum_f 0.5*mean(w^2*(|b_f|^2+|rx_f|^2))
//! O(n*F) scalars instead of O(n*F*N) array work. It is tolerance-level (not
//! bit-level) equal to the literal loop — which is exactly what the gate
//! allows. Either implementation is acceptable; evaluate on the SAME delta
//! grid with the same strict-< first-win rule.
//!
//! ## Result (mgs.py:114-116)
//!
//! psi = psi0 + best_delta everywhere, then psi[!support] = 0.

use crate::conv::Convolver;
use crate::types::{ChannelData, KernelPair, SolveParams};

/// Best (delta, loss) over the offset grid, given the already-propagated
/// per-channel base fields (QUANTIZED c64 values, len N each). Split out from
/// `warm_start_psi` so the parity test can gate the scan on its own.
pub fn delta_scan(
    channels: &[ChannelData],
    base_props: &[Vec<num_complex::Complex64>],
    ref_freq: f64,
) -> (f64, f64) {
    let _ = (channels, base_props, ref_freq);
    todo!("solver work package: stage-3 scan (literal or analytic)")
}

/// Full warm start: stage-1 solve -> unwrap -> scan. Returns the starting psi
/// for the joint descent (zero off support). Caller guarantees F > 1.
#[allow(clippy::too_many_arguments)]
pub fn warm_start_psi(
    initial_phase: &[f64],
    aper_amp: &[f64],
    support: &[bool],
    channels: &[ChannelData],
    kernels: &[KernelPair],
    ref_freq: f64,
    dx: f64,
    params: &SolveParams,
    conv: &mut Convolver,
) -> Vec<f64> {
    let _ = (initial_phase, aper_amp, support, channels, kernels, ref_freq, dx, params, conv);
    todo!("solver work package: stages 1-3 above")
}
