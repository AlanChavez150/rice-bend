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

use std::f64::consts::PI;

use num_complex::Complex64;

use crate::conv::Convolver;
use crate::npmath::{np_mean_f64, np_unwrap};
use crate::solve::gs_core;
use crate::types::{ChannelData, KernelPair, SolveParams};

/// WARM_START_SCAN_STEP (mgs.py:30), as the same expression.
const SCAN_STEP: f64 = 2.0 * PI / 64.0;
/// WARM_START_MAX_SAMPLES (mgs.py:34).
const MAX_SAMPLES: usize = 100_000;

/// The offset grid of mgs.py:82-93: one synthetic-wavelength period of the
/// comb, sampled at SCAN_STEP. Returns (period, n_samples).
fn scan_grid(channels: &[ChannelData], ref_freq: f64) -> (f64, usize) {
    let mut freqs: Vec<f64> = channels.iter().map(|ch| ch.freq).collect();
    freqs.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let min_gap = freqs
        .windows(2)
        .map(|p| p[1] - p[0])
        .filter(|&g| g > 0.0)
        .fold(f64::INFINITY, f64::min);
    let period = if min_gap.is_finite() { 2.0 * PI * ref_freq / min_gap } else { 2.0 * PI };
    let n_samples = ((period / SCAN_STEP).ceil() as usize).min(MAX_SAMPLES);
    (period, n_samples)
}

/// Best (delta, loss) over the offset grid, given the already-propagated
/// per-channel base fields (QUANTIZED c64 values, len N each). Split out from
/// `warm_start_psi` so the parity test can gate the scan on its own.
///
/// Analytic form: channel f's loss at offset delta is A_f - Re(e^{i rho_f delta} S_f),
/// so each channel reduces to two numbers once and every grid point costs O(F).
pub fn delta_scan(
    channels: &[ChannelData],
    base_props: &[Vec<Complex64>],
    ref_freq: f64,
) -> (f64, f64) {
    let n_freq = channels.len();
    let mut a_f = Vec::with_capacity(n_freq);
    let mut s_f = Vec::with_capacity(n_freq);
    for (ch, b) in channels.iter().zip(base_props) {
        let n = b.len();
        let (mut a, mut s_re, mut s_im) = (vec![0.0; n], vec![0.0; n], vec![0.0; n]);
        for j in 0..n {
            let (w2, m) = (ch.error_weighting[j] * ch.error_weighting[j], ch.rx_field[j]);
            a[j] = w2 * (b[j].norm_sqr() + m.norm_sqr());
            let bm = b[j] * m.conj();
            s_re[j] = w2 * bm.re;
            s_im[j] = w2 * bm.im;
        }
        a_f.push(0.5 * np_mean_f64(&a));
        s_f.push(Complex64::new(np_mean_f64(&s_re), np_mean_f64(&s_im)));
    }

    let (period, n_samples) = scan_grid(channels, ref_freq);
    let step = period / n_samples as f64;
    let (mut best_delta, mut best_loss) = (0.0, f64::INFINITY);
    for k in 0..n_samples {
        let delta = k as f64 * step;
        let mut loss = 0.0;
        for f in 0..n_freq {
            let (s, c) = (channels[f].rho * delta).sin_cos();
            loss += a_f[f] - (c * s_f[f].re - s * s_f[f].im);
        }
        loss /= n_freq as f64;
        if loss < best_loss {
            best_loss = loss;
            best_delta = delta;
        }
    }
    (best_delta, best_loss)
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
    let n = aper_amp.len();

    // 1. reference solve: the channel nearest rho == 1 (first wins ties), solved
    //    ALONE against its own frequency, so its rho is exactly 1.0
    let mut c_idx = 0;
    for (f, ch) in channels.iter().enumerate() {
        if (ch.rho - 1.0).abs() < (channels[c_idx].rho - 1.0).abs() {
            c_idx = f;
        }
    }
    let mut ch_ref = channels[c_idx].clone();
    ch_ref.rho = 1.0; // freq / freq, which IEEE makes exactly 1.0
    let stage1 = gs_core(
        initial_phase,
        aper_amp,
        support,
        std::slice::from_ref(&ch_ref),
        std::slice::from_ref(&kernels[c_idx]),
        dx,
        params,
        conv,
    );

    // 2. unwrap angle(amp * exp(i theta)) over the support, into psi units
    let support_idx: Vec<usize> = (0..n).filter(|&j| support[j]).collect();
    let angles: Vec<f64> = support_idx
        .iter()
        .map(|&j| {
            let (s, c) = stage1.aper_phase[j].sin_cos();
            (aper_amp[j] * s).atan2(aper_amp[j] * c)
        })
        .collect();
    let mut psi0 = vec![0.0; n];
    for (&j, u) in support_idx.iter().zip(np_unwrap(&angles)) {
        psi0[j] = u / channels[c_idx].rho;
    }

    // 3. the absolute-offset scan over each channel's once-propagated field
    let mut base_props = Vec::with_capacity(channels.len());
    let mut u0 = vec![Complex64::new(0.0, 0.0); n];
    for (ch, kp) in channels.iter().zip(kernels) {
        for &j in &support_idx {
            let (s, c) = (ch.rho * psi0[j]).sin_cos();
            u0[j] = Complex64::new(aper_amp[j] * c, aper_amp[j] * s);
        }
        let mut prop = vec![Complex64::new(0.0, 0.0); n];
        conv.rs_apply_into(&u0, &kp.fwd, dx, &mut prop);
        base_props.push(prop);
    }
    let (best_delta, _) = delta_scan(channels, &base_props, ref_freq);

    let mut psi = vec![0.0; n];
    for &j in &support_idx {
        psi[j] = psi0[j] + best_delta;
    }
    psi
}
