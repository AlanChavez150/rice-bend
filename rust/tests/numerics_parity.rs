//! Gate for src/npmath.rs: bit-exact against numpy on the `numerics` golden
//! case (np.add.reduce at f64/f32, the f32 convergence-window mean, np.unwrap).

mod common;

use common::Case;
use rice_bend_core::npmath::{np_sum_f32, np_sum_f64, np_unwrap};

/// Split a flat batch by its `<name>_lens` (f64-encoded counts).
fn batches<T: Copy>(data: &[T], lens: &[f64]) -> Vec<Vec<T>> {
    let mut out = Vec::new();
    let mut at = 0;
    for &l in lens {
        let l = l as usize;
        out.push(data[at..at + l].to_vec());
        at += l;
    }
    assert_eq!(at, data.len(), "lens must cover the data exactly");
    out
}

#[test]
fn sum_f64_bit_exact() {
    let case = Case::load("numerics");
    let arrs = batches(&case.f64s("sum64_data"), &case.f64s("sum64_lens"));
    for (a, want) in arrs.iter().zip(case.f64s("sum64_expected")) {
        let got = np_sum_f64(a);
        assert_eq!(got.to_bits(), want.to_bits(), "n={}: {got:e} vs numpy {want:e}", a.len());
    }
}

#[test]
fn sum_f32_bit_exact() {
    let case = Case::load("numerics");
    let arrs = batches(&case.f32s("sum32_data"), &case.f64s("sum32_lens"));
    for (a, want) in arrs.iter().zip(case.f32s("sum32_expected")) {
        let got = np_sum_f32(a);
        assert_eq!(got.to_bits(), want.to_bits(), "n={}: {got:e} vs numpy {want:e}", a.len());
    }
}

#[test]
fn convergence_window_mean_bit_exact() {
    let case = Case::load("numerics");
    let data = case.f32s("win_data");
    for (w, want) in data.chunks_exact(10).zip(case.f32s("win_expected")) {
        let diff: Vec<f32> = w.windows(2).map(|p| p[1] - p[0]).collect();
        let got = np_sum_f32(&diff) / diff.len() as f32;
        assert_eq!(got.to_bits(), want.to_bits(), "{got:e} vs numpy {want:e}");
    }
}

#[test]
fn unwrap_bit_exact() {
    let case = Case::load("numerics");
    let lens = case.f64s("unwrap_lens");
    let ins = batches(&case.f64s("unwrap_in"), &lens);
    let outs = batches(&case.f64s("unwrap_out"), &lens);
    for (i, (a, want)) in ins.iter().zip(&outs).enumerate() {
        let got = np_unwrap(a);
        for (k, (g, w)) in got.iter().zip(want).enumerate() {
            assert_eq!(g.to_bits(), w.to_bits(), "batch {i}[{k}]: {g} vs numpy {w}");
        }
    }
}
