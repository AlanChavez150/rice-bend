//! Shared data types — the fixed interface between the three work packages.
//!
//! These mirror `GerchbergSaxtonConfig` / `FreqChannel` / `GSResult` on the Python
//! side (src/rice_bend/config.py:87-108, src/rice_bend/mgs.py:24,124-128). Field
//! sets are part of the interface contract; internals of other modules are not.

use num_complex::Complex64;

use crate::conv::KernelSpectrum;

/// Solver hyperparameters, one-to-one with `GerchbergSaxtonConfig` (minus `init`,
/// `seed`, `history_stride`, which stay on the Python side: the RNG draw and the
/// init-mode choice happen before the boundary, and capture=True never crosses it).
#[derive(Debug, Clone)]
pub struct SolveParams {
    pub max_iters: usize,
    /// Window of recent losses checked for flatness (`convergence_count`).
    pub convergence_count: usize,
    /// Converged when mean(diff(recent f32 losses)) > this value (compared in f64).
    pub convergence_threshold: f64,
    /// Initial backtracking step size.
    pub lr0: f64,
    /// Backtracking shrink factor, 0 < bt_shrink < 1.
    pub bt_shrink: f64,
    /// Max backtracking reductions per iteration.
    pub bt_tries: usize,
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
/// amp * exp(i * aper_phase) — identical to mgs.py:302.
#[derive(Debug, Clone)]
pub struct CoreResult {
    /// Final psi (phase at ref_freq) on the full x-axis. Off-support values are
    /// whatever the accept/zeroing rules left there (see gs_core contract);
    /// they never affect the field because amp is zero off support.
    pub aper_phase: Vec<f64>,
    /// Post-accept joint loss of the last iteration (f64) — NOT loss_full[last],
    /// which is that iteration's PRE-step loss.
    pub final_loss: f64,
    /// Post-accept per-frequency losses of the last iteration. (F,)
    pub final_loss_per_freq: Vec<f64>,
    /// iter_idx + 1 when converged, else max_iters.
    pub n_iters_run: usize,
    pub stop_reason: StopReason,
    /// PRE-step joint loss per iteration, stored f32, truncated to n_iters_run.
    pub loss_full: Vec<f32>,
    /// PRE-step per-frequency losses, row-major (n_iters_run, F), f32.
    pub loss_full_per_freq: Vec<f32>,
}
