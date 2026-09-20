//! Criterion micro-benches: the N=2400 rs_apply (the primitive ~10^8 calls of a
//! study point are made of) and a synthetic max_iters-bound solve. Run with
//! `cargo bench` once the conv/solve packages are implemented (they panic on
//! todo!() until then). Compare against the Python numbers in rust/README.md.

use criterion::{criterion_group, criterion_main, Criterion};
use num_complex::Complex64;
use rice_bend_core::conv::Convolver;
use rice_bend_core::solve::gs_core;
use rice_bend_core::types::{ChannelData, KernelPair, SolveParams};

/// Deterministic synthetic values — benches measure speed, not physics.
fn xorshift(state: &mut u64) -> f64 {
    let mut x = *state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *state = x;
    (x >> 11) as f64 / (1u64 << 53) as f64
}

fn cvec(n: usize, state: &mut u64) -> Vec<Complex64> {
    (0..n)
        .map(|_| Complex64::new(xorshift(state) - 0.5, xorshift(state) - 0.5))
        .collect()
}

fn bench_rs_apply(c: &mut Criterion) {
    let n = 2400; // the flagship scene size (nfft 4800)
    let mut state = 0x9e3779b97f4a7c15u64;
    let u0 = cvec(n, &mut state);
    let h = cvec(n, &mut state);
    let dx = 0.00025;

    let mut conv = Convolver::new(n);
    let ks = conv.kernel_spectrum(&h);
    let mut out = vec![Complex64::new(0.0, 0.0); n];

    // Python baseline on this machine: scipy fftconvolve ~119 us/call,
    // cached-spectrum equivalent ~68 us.
    c.bench_function("rs_apply n=2400", |b| {
        b.iter(|| conv.rs_apply_into(&u0, &ks, dx, &mut out))
    });
}

fn bench_solve(c: &mut Criterion) {
    // tiny_check-shaped candidate (N=400, F=2) forced to run all 800 iterations:
    // convergence_threshold = +inf can never be exceeded by the flatness mean.
    let n = 400;
    let n_freq = 2;
    let mut state = 0x243f6a8885a308d3u64;

    let mut conv = Convolver::new(n);
    let mut channels = Vec::new();
    let mut kernels = Vec::new();
    let freqs = [140e9, 150e9];
    for f in 0..n_freq {
        let h = cvec(n, &mut state);
        let h_conj: Vec<Complex64> = h.iter().map(|c| c.conj()).collect();
        channels.push(ChannelData {
            freq: freqs[f],
            rho: freqs[f] / 150e9,
            rx_field: cvec(n, &mut state),
            error_weighting: (0..n).map(|_| xorshift(&mut state)).collect(),
        });
        kernels.push(KernelPair {
            fwd: conv.kernel_spectrum(&h),
            adj: conv.kernel_spectrum(&h_conj),
        });
    }

    let initial_phase: Vec<f64> = (0..n)
        .map(|_| 2.0 * std::f64::consts::PI * xorshift(&mut state))
        .collect();
    let aper_amp: Vec<f64> = (0..n).map(|i| if (100..300).contains(&i) { 1.0 } else { 0.0 }).collect();
    let support: Vec<bool> = aper_amp.iter().map(|&a| a > 0.0).collect();

    let params = SolveParams {
        max_iters: 800,
        convergence_count: 10,
        convergence_threshold: f64::INFINITY, // never converges: full 800 iters
        lr0: 0.05,
        bt_shrink: 0.5,
        bt_tries: 8,
    };

    let mut group = c.benchmark_group("solve");
    group.sample_size(10);
    group.bench_function("gs_core 800 iters n=400 f=2", |b| {
        b.iter(|| {
            gs_core(
                &initial_phase,
                &aper_amp,
                &support,
                &channels,
                &kernels,
                0.0015,
                &params,
                &mut conv,
            )
        })
    });
    group.finish();
}

criterion_group!(benches, bench_rs_apply, bench_solve);
criterion_main!(benches);
