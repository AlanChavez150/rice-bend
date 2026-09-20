//! Gates for the solver-core work package (solve.rs + warm_start.rs).
//!
//! Golden vectors (rust/tests/data/):
//!   solve_single/ — one single-frequency candidate solve, tiny_check geometry
//!                   (N=400, 40 iters): gates gs_core alone.
//!   solve_joint/  — the same candidate at F=2 with init="warm_start": gates
//!                   delta_scan (isolated via the golden psi0) and the full
//!                   warm start + joint descent.
//!
//! Bars (from the approved plan):
//!   - first-iteration loss rtol <= 1e-6 (one conv round-trip, minimal drift);
//!   - final_loss rtol <= 1e-3 (divergence compounds over iterations);
//!   - warm start: chosen delta equal on the shared grid, OR scan loss at the
//!     chosen delta within rtol 1e-5;
//!   - n_iters_run / stop_reason: REPORTED, not gated — ulp-level loss
//!     differences near the convergence threshold legitimately move the stop.
//!
//! These tests exercise conv.rs too, so they only mean anything once
//! `cargo test --test conv_parity` is green.

mod common;

use common::Case;
use num_complex::Complex64;
use rice_bend_core::conv::Convolver;
use rice_bend_core::solve::gs_core;
use rice_bend_core::types::{ChannelData, KernelPair, SolveParams};
use rice_bend_core::warm_start::{delta_scan, warm_start_psi};

struct Inputs {
    initial_phase: Vec<f64>,
    aper_amp: Vec<f64>,
    support: Vec<bool>,
    channels: Vec<ChannelData>,
    kernels: Vec<KernelPair>,
    ref_freq: f64,
    dx: f64,
    params: SolveParams,
    conv: Convolver,
    n: usize,
}

fn load_inputs(case: &Case) -> Inputs {
    let shape = case.shape("rx_field");
    let (n_freq, n) = (shape[0], shape[1]);

    let freqs = case.f64s("freqs");
    let ref_freq = case.scalar_f64("ref_freq");
    let rx = case.c128s("rx_field");
    let w = case.f64s("error_weighting");
    let h_fwd = case.c128s("h_fwd");
    assert_eq!(freqs.len(), n_freq);

    let mut conv = Convolver::new(n);
    let mut channels = Vec::with_capacity(n_freq);
    let mut kernels = Vec::with_capacity(n_freq);
    for f in 0..n_freq {
        channels.push(ChannelData {
            freq: freqs[f],
            rho: freqs[f] / ref_freq, // the pinned single-division definition
            rx_field: rx[f * n..(f + 1) * n].to_vec(),
            error_weighting: w[f * n..(f + 1) * n].to_vec(),
        });
        let h = &h_fwd[f * n..(f + 1) * n];
        let h_conj: Vec<Complex64> = h.iter().map(|c| c.conj()).collect();
        kernels.push(KernelPair {
            fwd: conv.kernel_spectrum(h),
            adj: conv.kernel_spectrum(&h_conj), // time-domain conj, per the contract
        });
    }

    let aper_amp = case.f64s("aper_amp");
    let support: Vec<bool> = aper_amp.iter().map(|&a| a > 0.0).collect();

    Inputs {
        initial_phase: case.f64s("initial_phase"),
        aper_amp,
        support,
        channels,
        kernels,
        ref_freq,
        dx: case.scalar_f64("dx"),
        params: SolveParams {
            max_iters: case.scalar_usize("max_iters"),
            convergence_count: case.scalar_usize("convergence_count"),
            convergence_threshold: case.scalar_f64("convergence_threshold"),
            lr0: case.scalar_f64("lr0"),
            bt_shrink: case.scalar_f64("bt_shrink"),
            bt_tries: case.scalar_usize("bt_tries"),
        },
        conv,
        n,
    }
}

fn rel(a: f64, b: f64) -> f64 {
    (a - b).abs() / b.abs().max(f64::MIN_POSITIVE)
}

#[test]
fn single_freq_solve() {
    let case = Case::load("solve_single");
    let mut inp = load_inputs(&case);
    assert_eq!(inp.channels.len(), 1, "solve_single must be F=1");

    let res = gs_core(
        &inp.initial_phase,
        &inp.aper_amp,
        &inp.support,
        &inp.channels,
        &inp.kernels,
        inp.dx,
        &inp.params,
        &mut inp.conv,
    );

    let gold_loss_full = case.f32s("loss_full");
    let first = rel(res.loss_full[0] as f64, gold_loss_full[0] as f64);
    assert!(
        first <= 1e-6,
        "first-iteration loss rel err {first:.3e} > 1e-6 (got {}, want {})",
        res.loss_full[0],
        gold_loss_full[0]
    );

    let gold_final = case.scalar_f64("final_loss");
    let fin = rel(res.final_loss, gold_final);
    assert!(
        fin <= 1e-3,
        "final_loss rel err {fin:.3e} > 1e-3 (got {}, want {gold_final})",
        res.final_loss
    );

    // report-only
    eprintln!(
        "solve_single: first-iter rel {first:.3e}, final rel {fin:.3e}; n_iters {} vs {} \
         (python), stop {} vs {}",
        res.n_iters_run,
        case.scalar_usize("n_iters_run"),
        res.stop_reason.as_str(),
        case.scalar_str("stop_reason"),
    );
}

#[test]
fn joint_delta_scan() {
    let case = Case::load("solve_joint");
    let mut inp = load_inputs(&case);
    assert!(inp.channels.len() > 1, "solve_joint must be F>1");

    // Isolate the scan from stage-1 drift: base props from the GOLDEN psi0.
    let psi0 = case.f64s("psi0");
    let mut base_props: Vec<Vec<Complex64>> = Vec::new();
    for (ch, kp) in inp.channels.iter().zip(&inp.kernels) {
        let u0: Vec<Complex64> = inp
            .aper_amp
            .iter()
            .zip(&psi0)
            .map(|(&a, &p)| Complex64::from_polar(a, ch.rho * p))
            .collect();
        let mut out = vec![Complex64::new(0.0, 0.0); inp.n];
        inp.conv.rs_apply_into(&u0, &kp.fwd, inp.dx, &mut out);
        base_props.push(out);
    }

    let (delta, loss) = delta_scan(&inp.channels, &base_props, inp.ref_freq);
    let gold_delta = case.scalar_f64("warm_best_delta");
    let gold_loss = case.scalar_f64("warm_best_loss");
    let period = case.scalar_f64("warm_period");

    if (delta - gold_delta).abs() > 1e-9 * period {
        let l = rel(loss, gold_loss);
        assert!(
            l <= 1e-5,
            "delta scan picked {delta} (python {gold_delta}) and its loss differs by \
             rel {l:.3e} > 1e-5 ({loss} vs {gold_loss})"
        );
        eprintln!("solve_joint: different delta ({delta} vs {gold_delta}) but loss within tolerance");
    } else {
        eprintln!("solve_joint: delta matches python exactly ({delta})");
    }
}

#[test]
fn joint_warm_start_solve() {
    let case = Case::load("solve_joint");
    let mut inp = load_inputs(&case);

    let psi = warm_start_psi(
        &inp.initial_phase,
        &inp.aper_amp,
        &inp.support,
        &inp.channels,
        &inp.kernels,
        inp.ref_freq,
        inp.dx,
        &inp.params,
        &mut inp.conv,
    );

    // report-only: psi vs python's warm psi (a 2pi unwrap-branch flip near a
    // wrap boundary is possible and legitimate; the joint solve gate decides)
    let gold_psi = case.f64s("warm_psi");
    let max_dpsi = psi
        .iter()
        .zip(&gold_psi)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0, f64::max);
    eprintln!("solve_joint: max |warm psi - python| = {max_dpsi:.3e}");

    let res = gs_core(
        &psi,
        &inp.aper_amp,
        &inp.support,
        &inp.channels,
        &inp.kernels,
        inp.dx,
        &inp.params,
        &mut inp.conv,
    );

    let gold_final = case.scalar_f64("final_loss");
    let fin = rel(res.final_loss, gold_final);
    assert!(
        fin <= 1e-3,
        "joint final_loss rel err {fin:.3e} > 1e-3 (got {}, want {gold_final})",
        res.final_loss
    );
    eprintln!(
        "solve_joint: final rel {fin:.3e}; n_iters {} vs {} (python), stop {} vs {}",
        res.n_iters_run,
        case.scalar_usize("n_iters_run"),
        res.stop_reason.as_str(),
        case.scalar_str("stop_reason"),
    );
}
