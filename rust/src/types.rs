//! Shared data types — the fixed interface between the three work packages.
//!
//! These mirror `GerchbergSaxtonConfig` / `FreqChannel` / `GSResult` on the Python
//! side (src/rice_bend/config.py:87-108, src/rice_bend/mgs.py:24,124-128). Field
//! sets are part of the interface contract; internals of other modules are not.

use num_complex::Complex64;

use crate::conv::KernelSpectrum;

/// What the descent does with the plate's amplitude — `GerchbergSaxtonConfig.amplitude`
/// (+ `amp_max`). The START amplitude is the caller's in either mode: the `aper_amp`
/// handed to gs_core (|orig_aper_amp| for Fixed, mgs._backprop_amplitude for Descend,
/// A0 on the support for Level, all computed in Python).
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum AmplitudeMode {
    /// Held at the amplitude handed in, throughout (the default).
    Fixed,
    /// ONE amplitude shared by every tone, fitted in the same backtracking step as
    /// psi and clipped back into [0, amp_max] (numpy's clip, see npmath::np_clip)
    /// after every trial step. `amp_max` is finite and > 0 (checked at the boundary).
    Descend { amp_max: f64 },
    /// ONE uniform level A >= 0 over the support, shared by every tone, fitted in
    /// the same backtracking step as psi by the MEAN over the support of Descend's
    /// per-sample gradient, kept >= 0 by np.clip(., 0, inf) and uncapped. Its start
    /// A0 (mgs._power_matched_level) arrives as aper_amp = A0 on the support, 0 off
    /// it, which gs_core reads back exactly (A0 * 1.0 == A0).
    Level,
}

/// The excess-power penalty, `GerchbergSaxtonConfig.power_penalty` (gamma) plus the
/// received power it is measured against. With it the descent minimizes the
/// objective
///
///   J = L * R^gamma,   R = max(1, P_plate / P_rx)
///
/// instead of the plain joint loss L: P_plate = sum over the support of a_j^2 * dx
/// is the power leaving the candidate plate, P_rx the power measured on the RX
/// elements. The floor at 1 is load-bearing (an unfloored L * R collapses onto dark
/// plates): a plate that sends out no more than the receiver caught is not
/// penalized at all. P_rx is a constant of the solve, computed ONCE in Python
/// (mgs._received_power) and handed over as a scalar, so both engines use the
/// identical number. The penalty needs a fitted amplitude: the boundary refuses it
/// under 'fixed', where R is a constant of the grid (see solve.rs).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PowerPenalty {
    /// The exponent gamma, finite and > 0 (checked at the boundary).
    pub gamma: f64,
    /// P_rx, finite and > 0 (checked at the boundary).
    pub rx_power: f64,
}

/// Solver hyperparameters, one-to-one with `GerchbergSaxtonConfig` (minus `init`,
/// `seed`, `history_stride`, which stay on the Python side: the RNG draw and the
/// init-mode choice happen before the boundary, and capture=True never crosses it).
#[derive(Debug, Clone)]
pub struct SolveParams {
    pub max_iters: usize,
    /// Window of recent losses checked for flatness (`convergence_count`).
    pub convergence_count: usize,
    /// Converged when mean(diff(recent f32 losses)) > this value (compared in f64).
    /// Ignored when `convergence_rel` is set.
    pub convergence_threshold: f64,
    /// Initial backtracking step size.
    pub lr0: f64,
    /// Backtracking shrink factor, 0 < bt_shrink < 1.
    pub bt_shrink: f64,
    /// Max backtracking reductions per iteration.
    pub bt_tries: usize,
    /// `amplitude` / `amp_max`.
    pub amplitude: AmplitudeMode,
    /// `power_penalty` (+ the received power); None (the default) is the plain
    /// joint loss, every operation of the solve bit-identical to a build without it.
    pub power_penalty: Option<PowerPenalty>,
    /// `convergence_rel`, the relative stop rule (finite and > 0, checked at the
    /// boundary). When set it REPLACES the absolute test and convergence_threshold
    /// is ignored: on an f64 history of the same pre-step objective, converged when
    /// the window's last value r <= 0 or mean(diff(window)) / r > -convergence_rel
    /// (solve.rs, "Convergence test"). An absolute threshold has units of loss and
    /// means a different thing in every cell. None (the default) is the absolute
    /// rule, every operation of the solve bit-identical to a build without it.
    pub convergence_rel: Option<f64>,
}

/// Why the solve stopped. String forms must match mgs.py exactly: the values land
/// in candidate JSON and are compared by the parity harness.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StopReason {
    Converged,
    MaxIters,
}

impl StopReason {
    pub fn as_str(&self) -> &'static str {
        match self {
            StopReason::Converged => "converged",
            StopReason::MaxIters => "max_iters",
        }
    }
}

/// One frequency's measurement, the Rust twin of `FreqChannel` (mgs.py:24).
///
/// `rho` MUST be computed as the single f64 division `freq / ref_freq` (bit-exact
/// with mgs.py:174); do not derive it any other way.
#[derive(Debug, Clone)]
pub struct ChannelData {
    pub freq: f64,
    /// freq / ref_freq — each channel's phase is rho * psi (delay-plate model).
    pub rho: f64,
    /// Measured RX field on the scene x-axis, complex128. (N,)
    pub rx_field: Vec<Complex64>,
    /// Per-sample loss weighting, f64, zero outside the RX window. (N,)
    pub error_weighting: Vec<f64>,
}

/// Forward + adjoint kernel spectra for one channel. The adjoint kernel is
/// bit-exactly conj(h_fwd) in the TIME domain (mgs.py:185-187); its spectrum must
/// be built as FFT(pad(conj(h_fwd))) — NOT as conj(FFT(pad(h_fwd))), which is the
/// conjugate-plus-frequency-reversal trap.
#[derive(Debug, Clone)]
pub struct KernelPair {
    pub fwd: KernelSpectrum,
    pub adj: KernelSpectrum,
}

/// Result of one core solve, mirroring the arrays gs_reconstruct returns
/// (mgs.py:329-335). The complex aperture field is composed by the caller as
/// aper_amp * exp(i * aper_phase) — identical to mgs.py:302.
#[derive(Debug, Clone)]
pub struct CoreResult {
    /// Final psi (phase at ref_freq) on the full x-axis. Off-support values are
    /// whatever the accept/zeroing rules left there (see gs_core contract);
    /// they never affect the field because amp is zero off support.
    pub aper_phase: Vec<f64>,
    /// Final amplitude on the full x-axis: the input amplitude unchanged under
    /// AmplitudeMode::Fixed, the descended one (zero off support once a step has
    /// been accepted) under Descend. The caller composes amp * exp(i * aper_phase).
    pub aper_amp: Vec<f64>,
    /// Post-accept joint loss of the last iteration (f64) — NOT loss_full[last],
    /// which is that iteration's PRE-step loss. Under the power penalty it is the
    /// OBJECTIVE J = final_mse * power_ratio^gamma, the number that ranks candidates.
    pub final_loss: f64,
    /// Post-accept per-frequency losses of the last iteration. (F,) Always the
    /// PLAIN per-tone losses, also under the power penalty (so there final_loss =
    /// mean(final_loss_per_freq) * power_ratio^gamma).
    pub final_loss_per_freq: Vec<f64>,
    /// Post-accept PLAIN joint loss L of the last iteration (the mean of
    /// final_loss_per_freq); equal to final_loss when the penalty is off.
    pub final_mse: f64,
    /// Post-accept R = max(1, P_plate / P_rx) of the last iteration; None when the
    /// penalty is off.
    pub power_ratio: Option<f64>,
    /// iter_idx + 1 when converged, else max_iters.
    pub n_iters_run: usize,
    pub stop_reason: StopReason,
    /// PRE-step joint loss per iteration (the objective J under the power penalty),
    /// stored f32, truncated to n_iters_run.
    pub loss_full: Vec<f32>,
    /// PRE-step per-frequency losses (plain, also under the penalty), row-major
    /// (n_iters_run, F), f32.
    pub loss_full_per_freq: Vec<f32>,
}
