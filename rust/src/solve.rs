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
//! ## Descended amplitude (`amplitude: descend`, SolveParams::amplitude)
//!
//! The amplitude `a` becomes an unknown too: ONE array shared by every tone,
//! starting from the `aper_amp` handed in (mgs._backprop_amplitude, computed in
//! Python). Per channel f, from the SAME back-propagated residual g:
//!
//!   e[j]         = exp(i * rho[f] * psi[j])     // the unit phasor u0 was built from
//!   grad_a[j]   += 2.0 * Re(g[j] * conj(e[j]))  // no rho: a is shared by every tone
//!
//! then grad_a /= F; grad_a[!support] = 0, like grad. Each backtracking trial
//! moves both halves with the same step:
//!
//!   a_trial = np.clip(a - step * grad_a, 0, amp_max); a_trial[!support] = 0
//!
//! (npmath::np_clip — NaN passes through, as in numpy), the trial field is built
//! from a_trial, and an accepted trial sets a = a_trial along with psi. The
//! support is the CALLER's: the original aperture's nonzero region, NOT a > 0.
//! The start amplitude can be 0 at window samples, and the clip drives samples
//! to 0 during the descent; every support sample keeps being descended.
//!
//! ## Uniform level (`amplitude: level`)
//!
//! ONE level A >= 0 over the support, shared by every tone. The plate is always
//! A * 1_support; its start A0 (mgs._power_matched_level, computed in Python)
//! arrives as aper_amp = A0 on the support, and A is read back from any support
//! sample (A0 * 1.0 is A0 exactly). The per-sample grad_a is built as for Descend
//! (/= F, zeroed off the support), but from the residual weighted TWICE: per
//! channel f, after g (which the phase gradient keeps),
//!
//!   r2[j]       = w[f][j] * r[j]                // whole axis; numpy real * complex
//!   g2          = rs_apply(r2, h_adj[f])        // QUANTIZED c64, as g
//!   grad_a[j]  += 2.0 * Re(g2[j] * conj(e[j]))
//!
//! then
//!
//!   grad_A  = np.mean(grad_a[support])         // np_mean_f64, support in order
//!   A_trial = np.clip(A - step * grad_A, 0, inf)   // NaN passes through
//!   a_trial = A_trial on the support, 0 off it
//!
//! and an accepted trial sets A = A_trial and a = a_trial along with psi.
//!
//! Why w^2 r: g carries ONE factor of the error weighting (the solver's convention,
//! the exact slope only for a 0/1 weighting, where w = w^2), and the phase and
//! descend's per-sample amplitude keep it. The level is one scalar, the mean of that
//! gradient over the support, and under a tapered weighting (the measured row's
//! w = sqrt(2)(|y|/max + 0.25), 0.40..1.77) the mean of the w r gradient has the
//! wrong sign against dL/dA in about half the cells: the joint step then raised L at
//! every halving, nothing was accepted, and the flatness rule stopped the solve
//! "converged" with the plate dimmed far below the power floor. With w^2 r,
//! grad_A = (2N / n_sup) dL/dA exactly. The cost is one more adjoint per tone per
//! iteration, under 'level' only.
//!
//! ## Excess-power penalty (`power_penalty`, SolveParams::power_penalty)
//!
//! With PowerPenalty { gamma, rx_power } (types.rs) and a fitted amplitude, the
//! backtracking descends the OBJECTIVE
//!
//!   J = L * R^gamma,   R = max(1, P_plate / P_rx)
//!
//! where L is the joint loss above, P_rx = rx_power (computed once in Python) and
//! the plate power comes from the amplitude the loss was computed at:
//!
//!   level:    P_plate = (A * A) * (n_sup as f64) * dx
//!   descend:  P_plate = np_sum_f64([a[j] * a[j] for j in support, in order]) * dx
//!   R         = (P_plate / rx_power).max(1.0)   // Python's max(1.0, .); NaN -> 1
//!   J         = L * R.powf(gamma)
//!
//! Per iteration, after the gradient: L = mean(loss_pf) and R from the CURRENT
//! amplitude, then loss = J. loss_full records J (pre-step) and the convergence
//! window runs on it unchanged; loss_full_per_freq keeps the plain loss_pf. The
//! descent direction is J's scaled by 1 / R^gamma, so the phase gradient is
//! untouched and the amplitude's gains ONE term, only when R > 1, added after
//! grad_a /= F (and, for the level, after grad_A = mean(grad_a[support])),
//! evaluated in exactly this order:
//!
//!   level:    grad_A += (4.0 * gamma * L * N) / (A * n_sup)
//!   descend:  coef = (4.0 * gamma * L * N * dx) / P_plate
//!             grad_a[j] += coef * a[j]                      // support samples
//!
//! with N = conv.n_signal. (descend's grad_a is 2N * dL/da_j in the solver's
//! one-w convention, and (gamma L / R) dR/da_j = 2 gamma L a_j dx / P_plate; the
//! level's grad_A is (2N / n_sup) dL/dA exactly, from w^2 r, and (gamma L / R) dR/dA
//! = 2 gamma L / A, so the clamp's d_minus and t below are on one scale.) Each trial
//! computes trial_plain = mean(trial_pf), R_trial from ITS amplitude (a_trial /
//! A_trial) and J_trial = trial_plain * R_trial^gamma; accept iff J_trial < J
//! (strict, first win), and the accept also carries trial_plain and R_trial over
//! as the current L and R.
//!
//! The level at the floor. Under 'level' J has a kink in A at the floor
//! a_floor = (rx_power / (n_sup as f64 * dx)).sqrt() (computed once per solve),
//! where R leaves 1. J's one-sided slopes there are d_minus = grad_A (below) and
//! d_plus = grad_A + t, t the level's term above (+inf at A == 0, which has no
//! kink below it). When d_minus < 0 < d_plus the kink is J's minimum in A, and a
//! plain step for A keeps crossing it: J rises on either side by more than the
//! phase step brings it down, every halving is rejected, and the flatness rule
//! stops the solve as "converged" with J far above its value at the floor. So,
//! when the slopes bracket zero, t is NOT added and the level's trial is stopped
//! AT the floor from the side it is on:
//!
//!   above   = P_plate > rx_power
//!   A_trial = if above { (A - step * d_plus).max(a_floor) }
//!             else       { (A - step * d_minus).min(a_floor) }
//!
//! At the floor (A == a_floor, not above) the trial is a_floor: the level holds
//! and the phase alone steps. A halved step that falls short of the floor is an
//! ordinary gradient step, and a clamped trial is accepted or rejected on J like
//! any other. Otherwise t is added when R > 1, as above, the same expression, so
//! the gradient is bit-identical wherever the rule does not fire. Descend keeps
//! the plain penalty (no clamp).
//!
//! Off (None): not one operation added or reordered. The loop branches on the
//! Option; it never multiplies by a factor of 1.0. Under AmplitudeMode::Fixed the
//! penalty is ignored: the boundary refuses it there (R would be a constant of the
//! grid), and the one Fixed solve that can still carry it, the legacy warm start's
//! phase-only stage 1 (warm_start.rs builds its params as `..params.clone()`),
//! runs the plain loss.
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
//! The relative variant (`convergence_rel`, SolveParams::convergence_rel). The
//! threshold above has units of loss: on a valley floor (J ~ 2e-5) it stops cells
//! before the ranking is decided, on a wall (J ~ 7e-3) f32 spacing (~5e-10) makes it
//! fire on quantization. With Some(tol) the absolute test is REPLACED
//! (convergence_threshold ignored), run at the same place under the same
//! iter_idx > convergence_count, on an f64 history `hist64` of the same pre-step
//! objective the f32 history records (stored beside it, never read back at f32):
//!
//!   w         = hist64[iter_idx - convergence_count .. iter_idx]  // current excluded
//!   mean_diff = np_mean_f64(diff(w))     // p[1] - p[0]; numpy's pairwise sum / n
//!   r         = hist64[iter_idx - 1]     // w[-1]
//!   converged iff r <= 0.0 || mean_diff / r > -tol
//!
//! with the same Converged / n_iters_run = iter_idx + 1. A NaN fails both
//! comparisons and runs to max_iters, as under the absolute rule. Off (None): the
//! loop branches on the Option and runs the absolute block verbatim; hist64 stays
//! empty and no floating-point operation is added or reordered.
//!
//! ## What is returned
//!
//! final_loss / final_loss_per_freq are the POST-accept values of the last
//! iteration; loss_full holds the PRE-step values (so loss_full[last] !=
//! final_loss whenever the last iteration accepted a step). aper_phase is psi
//! after the last iteration, aper_amp the amplitude (the input one under Fixed).
//! Under the power penalty final_loss is the post-accept J, final_mse the
//! post-accept L and power_ratio the post-accept R (so final_loss = final_mse *
//! R^gamma, and final_loss_per_freq stays the plain per-tone losses); off, final_mse
//! is final_loss and power_ratio is None.
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
use crate::npmath::{np_clip, np_mean_f64, np_sum_f32, np_sum_f64};
use crate::types::{AmplitudeMode, ChannelData, CoreResult, KernelPair, SolveParams, StopReason};

/// Preallocated per-solve work buffers: the loop body allocates nothing. r2 / g2
/// are the level's w * r and its back-propagation (written under 'level' only).
struct Work {
    u0: Vec<Complex64>,
    prop: Vec<Complex64>,
    r: Vec<Complex64>,
    abs2: Vec<f64>,
    g: Vec<Complex64>,
    r2: Vec<Complex64>,
    g2: Vec<Complex64>,
}

impl Work {
    fn new(n: usize) -> Self {
        let z = Complex64::new(0.0, 0.0);
        Work {
            u0: vec![z; n],
            prop: vec![z; n],
            r: vec![z; n],
            abs2: vec![0.0; n],
            g: vec![z; n],
            r2: vec![z; n],
            g2: vec![z; n],
        }
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

/// build_u0 that also keeps the unit phasor exp(i * rho * psi) in `phasor` (on
/// the support), which the descended amplitude's gradient takes Re(g * conj(.))
/// against. The same sin_cos feeds both, as one np.exp feeds both in numpy.
fn build_u0_phasor(
    u0: &mut [Complex64],
    phasor: &mut [Complex64],
    aper_amp: &[f64],
    support_idx: &[usize],
    rho: f64,
    psi: &[f64],
) {
    for &j in support_idx {
        let (s, c) = (rho * psi[j]).sin_cos();
        phasor[j] = Complex64::new(c, s);
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

/// The plate power P_plate of a descended amplitude: np.add.reduce(a[support] ** 2)
/// * dx, the squares in support order into `sq` (capacity n_sup, so no allocation).
fn descend_power(amp: &[f64], support_idx: &[usize], sq: &mut Vec<f64>, dx: f64) -> f64 {
    sq.clear();
    sq.extend(support_idx.iter().map(|&j| amp[j] * amp[j]));
    np_sum_f64(sq) * dx
}

/// The plate power P_plate of a uniform level A over n_sup support samples, in
/// mgs.py's order: (A * A) * float(n_sup) * dx.
fn level_power(amp_level: f64, n_sup: usize, dx: f64) -> f64 {
    (amp_level * amp_level) * (n_sup as f64) * dx
}

/// Run the descent from `initial_phase`. Inputs are on the full scene x-axis
/// (length N == conv.n_signal); `channels` and `kernels` are index-aligned.
///
/// `support[j]` is mgs.gs_reconstruct's `support = |orig_aper_amp| > 0`. Under
/// AmplitudeMode::Fixed that equals aper_amp > 0 (amp = |orig_aper_amp|); under
/// Descend it does NOT — the start amplitude can be 0 inside the window — so the
/// caller passes Python's support itself. `aper_amp` must be 0 off the support
/// (u0 is built on the support only).
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
    let (amp_max, level) = match params.amplitude {
        AmplitudeMode::Fixed => (None, false),
        AmplitudeMode::Descend { amp_max } => (Some(amp_max), false),
        AmplitudeMode::Level => (None, true),
    };
    // Descend or Level: the amplitude is fitted (mgs.py's fit_amp)
    let fit_amp = amp_max.is_some() || level;
    // the amplitude the solve runs at: held under Fixed, fitted otherwise
    let mut amp = aper_amp.to_vec();
    // Level's unknown A, read back from its start plate A0 * 1_support (an empty
    // support has no sample to read, and no sample for A to move: 1.0 then)
    let mut amp_level = support_idx.first().map_or(1.0, |&j| aper_amp[j]);
    let mut grad_level = f64::NAN;
    let mut support_grad = Vec::with_capacity(if level { support_idx.len() } else { 0 });
    // the excess-power penalty, with a fitted amplitude only (see the module doc:
    // ignored under Fixed). `loss` is then the objective J, the plain joint loss L
    // and the ratio R are kept beside it, and descend's plate power sums its squares
    // in `power_sq`
    let penalty = params.power_penalty.filter(|_| fit_amp);
    let n_scene = n as f64;
    let n_sup = support_idx.len();
    let mut power_sq = Vec::with_capacity(if penalty.is_some() && !level { n_sup } else { 0 });
    let mut loss_plain = f64::NAN;
    let mut ratio = f64::NAN;
    // the level's floor under the penalty, where R leaves 1 (the module doc's "level
    // at the floor"), and the per-iteration state of its clamp: whether this
    // iteration's trial aims at the floor, from which side, and J's slopes in A
    // below / above it. Untouched (aim_floor stays false) without level && penalty
    let a_floor = match penalty {
        Some(pp) if level => (pp.rx_power / (n_sup as f64 * dx)).sqrt(),
        _ => f64::NAN,
    };
    let mut aim_floor = false;
    let mut above = false;
    let (mut d_minus, mut d_plus) = (f64::NAN, f64::NAN);
    // the fitted modes' buffers (empty under Fixed). amp_trial is never written off
    // the support, so it stays 0 there: a_trial[!support] = 0, as in mgs.py
    let (mut grad_amp, mut amp_trial, mut phasor) = if fit_amp {
        (vec![0.0; n], vec![0.0; n], vec![Complex64::new(0.0, 0.0); n])
    } else {
        (Vec::new(), Vec::new(), Vec::new())
    };
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
    // the relative stop rule's f64 history of the same pre-step objective and its
    // window diffs, both empty when the rule is off (the absolute rule reads hist)
    let rel = params.convergence_rel.is_some();
    let mut hist64 = vec![0.0f64; if rel { params.max_iters } else { 0 }];
    let mut diffs64: Vec<f64> = Vec::with_capacity(if rel { params.convergence_count } else { 0 });

    let mut stop_reason = StopReason::MaxIters;
    let mut n_iters_run = params.max_iters;
    let mut loss = f64::NAN;

    for iter_idx in 0..params.max_iters {
        // forward + adjoint per channel; gradient of the MEAN loss
        grad.fill(0.0);
        if fit_amp {
            grad_amp.fill(0.0);
        }
        for (f, (ch, kp)) in channels.iter().zip(kernels).enumerate() {
            if fit_amp {
                build_u0_phasor(&mut work.u0, &mut phasor, &amp, &support_idx, ch.rho, &psi);
            } else {
                build_u0(&mut work.u0, &amp, &support_idx, ch.rho, &psi);
            }
            loss_pf[f] = channel_loss(&mut work, ch, &kp.fwd, dx, conv);
            conv.rs_apply_into(&work.r, &kp.adj, dx, &mut work.g);
            for &j in &support_idx {
                // imag(g * conj(u0)), in numpy's complex-multiply operand order
                let (g, u) = (work.g[j], work.u0[j]);
                let im = g.re * (-u.im) + g.im * u.re;
                grad[j] += ch.rho * (2.0 * im);
            }
            if fit_amp {
                if level {
                    // the level's gradient back-propagates w * r (w^2 on the residual,
                    // the exact slope; the module doc), over the WHOLE axis as numpy
                    // multiplies the whole array: real * complex is (w re, w im)
                    for j in 0..n {
                        let w = ch.error_weighting[j];
                        let r = work.r[j];
                        work.r2[j] = Complex64::new(w * r.re, w * r.im);
                    }
                    conv.rs_apply_into(&work.r2, &kp.adj, dx, &mut work.g2);
                }
                // descend keeps g (one w, the phase's convention)
                let g_src: &[Complex64] = if level { &work.g2 } else { &work.g };
                for &j in &support_idx {
                    // real(g * conj(e)), in numpy's complex-multiply operand order
                    let (g, e) = (g_src[j], phasor[j]);
                    let re = g.re * e.re - g.im * (-e.im);
                    grad_amp[j] += 2.0 * re;
                }
            }
        }
        for &j in &support_idx {
            grad[j] /= n_freq as f64;
        }
        if fit_amp {
            for &j in &support_idx {
                grad_amp[j] /= n_freq as f64;
            }
        }
        if level {
            // np.mean(grad_amp[support]): the support samples, in order
            support_grad.clear();
            support_grad.extend(support_idx.iter().map(|&j| grad_amp[j]));
            grad_level = np_mean_f64(&support_grad);
        }

        loss = np_mean_f64(&loss_pf);
        if let Some(pp) = penalty {
            // the objective J = L * R^gamma at the CURRENT amplitude; `loss` was L.
            // The gradient term comes after grad /= F and grad_level (the module doc;
            // mgs.gs_reconstruct's `if penalized:` block after grad_level)
            loss_plain = loss;
            let p_plate = if level {
                level_power(amp_level, n_sup, dx)
            } else {
                descend_power(&amp, &support_idx, &mut power_sq, dx)
            };
            ratio = (p_plate / pp.rx_power).max(1.0);
            aim_floor = false;
            if level {
                // J's slopes in A just below the floor (the plain gradient) and just
                // above it (plus the penalty's term; A == 0 has no kink below it).
                // When they bracket zero the kink is the minimum in A, and a plain
                // step keeps jumping across it: the joint step is then rejected at
                // every halving and the solve stalls into a false "converged". The
                // trial below stops the level AT the floor instead, and no term is added
                let t_pen = if amp_level > 0.0 {
                    (4.0 * pp.gamma * loss_plain * n_scene) / (amp_level * n_sup as f64)
                } else {
                    f64::INFINITY
                };
                d_minus = grad_level;
                d_plus = grad_level + t_pen;
                if d_minus < 0.0 && 0.0 < d_plus {
                    aim_floor = true;
                    above = p_plate > pp.rx_power;
                } else if ratio > 1.0 {
                    grad_level += t_pen;
                }
            } else if ratio > 1.0 {
                let coef = (4.0 * pp.gamma * loss_plain * n_scene * dx) / p_plate;
                for &j in &support_idx {
                    grad_amp[j] += coef * amp[j];
                }
            }
            loss = loss_plain * ratio.powf(pp.gamma);
        }
        hist[iter_idx] = loss as f32;
        if rel {
            hist64[iter_idx] = loss;
        }
        for f in 0..n_freq {
            hist_pf[iter_idx * n_freq + f] = loss_pf[f] as f32;
        }

        // backtracking line search on the joint loss, or on J under the power
        // penalty (strict decrease, first win)
        let mut step = params.lr0;
        for _ in 0..params.bt_tries {
            for &j in &support_idx {
                theta[j] = psi[j] - step * grad[j];
            }
            // one JOINT step: the descended amplitude takes the same step, clipped
            // back into [0, amp_max]; the pair is accepted or shrunk together
            let mut level_trial = f64::NAN;
            if let Some(cap) = amp_max {
                for &j in &support_idx {
                    amp_trial[j] = np_clip(amp[j] - step * grad_amp[j], 0.0, cap);
                }
            } else if level {
                // the level's step, kept >= 0 with no cap (level_trial * 1_support).
                // Aiming at the floor, the step on the level's side of it (d_plus
                // above, d_minus below) is stopped AT the floor, never across it: at
                // the floor the level holds and the phase alone steps
                level_trial = if aim_floor {
                    if above {
                        (amp_level - step * d_plus).max(a_floor)
                    } else {
                        (amp_level - step * d_minus).min(a_floor)
                    }
                } else {
                    np_clip(amp_level - step * grad_level, 0.0, f64::INFINITY)
                };
                for &j in &support_idx {
                    amp_trial[j] = level_trial;
                }
            }
            let trial_amp: &[f64] = if fit_amp { &amp_trial } else { &amp };
            for (f, (ch, kp)) in channels.iter().zip(kernels).enumerate() {
                build_u0(&mut work.u0, trial_amp, &support_idx, ch.rho, &theta);
                trial_pf[f] = channel_loss(&mut work, ch, &kp.fwd, dx, conv);
            }
            let mut loss_trial = np_mean_f64(&trial_pf);
            let (mut trial_plain, mut ratio_trial) = (f64::NAN, f64::NAN);
            if let Some(pp) = penalty {
                // J_trial from the trial's OWN amplitude: a_trial, or A_trial
                // (mgs.gs_reconstruct's `if penalized:` block after loss_trial)
                trial_plain = loss_trial;
                let p_trial = if level {
                    level_power(level_trial, n_sup, dx)
                } else {
                    descend_power(&amp_trial, &support_idx, &mut power_sq, dx)
                };
                ratio_trial = (p_trial / pp.rx_power).max(1.0);
                loss_trial = trial_plain * ratio_trial.powf(pp.gamma);
            }
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
                if fit_amp {
                    // whole array: amp_trial is 0 off the support
                    amp.copy_from_slice(&amp_trial);
                }
                if level {
                    amp_level = level_trial;
                }
                loss = loss_trial;
                if penalty.is_some() {
                    loss_plain = trial_plain;
                    ratio = ratio_trial;
                }
                loss_pf.copy_from_slice(&trial_pf);
                break;
            }
            step *= params.bt_shrink;
        }

        // convergence: the absolute rule, or the relative one when convergence_rel is set
        let count = params.convergence_count;
        if iter_idx > count {
            match params.convergence_rel {
                None => {
                    // convergence: mean(diff(recent f32 losses)), in f32, current index excluded
                    diffs.clear();
                    diffs.extend(hist[iter_idx - count..iter_idx].windows(2).map(|p| p[1] - p[0]));
                    let flatness = np_sum_f32(&diffs) / diffs.len() as f32;
                    if f64::from(flatness) > params.convergence_threshold {
                        stop_reason = StopReason::Converged;
                        n_iters_run = iter_idx + 1;
                        break;
                    }
                }
                Some(tol) => {
                    // the relative rule (module doc): the same window on the f64
                    // history, its mean drop per iteration against the window's last
                    // value. r <= 0 has nothing left to descend
                    diffs64.clear();
                    diffs64.extend(hist64[iter_idx - count..iter_idx].windows(2).map(|p| p[1] - p[0]));
                    let mean_diff = np_mean_f64(&diffs64);
                    let r = hist64[iter_idx - 1];
                    if r <= 0.0 || mean_diff / r > -tol {
                        stop_reason = StopReason::Converged;
                        n_iters_run = iter_idx + 1;
                        break;
                    }
                }
            }
        }
    }

    hist.truncate(n_iters_run);
    hist_pf.truncate(n_iters_run * n_freq);
    CoreResult {
        aper_phase: psi,
        aper_amp: amp,
        final_loss: loss,
        final_loss_per_freq: loss_pf,
        final_mse: if penalty.is_some() { loss_plain } else { loss },
        power_ratio: penalty.map(|_| ratio),
        n_iters_run,
        stop_reason,
        loss_full: hist,
        loss_full_per_freq: hist_pf,
    }
}
