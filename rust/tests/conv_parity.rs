//! Gate for the FFT/convolution work package (fft.rs + conv.rs).
//!
//! Golden vectors: rust/tests/data/conv_n{7,8,2399,2400,2401}/ — odd/even tiny
//! sizes for the "same"-slice centering, the two real scene sizes around it,
//! and n2400 built from an actual RS kernel + box-aperture field.
//!
//! Bars (from the approved plan):
//!   - nfft equals scipy.fft.next_fast_len(2N-1) on every case;
//!   - pre-downcast (rs_apply_f64): elementwise rel err <= 1e-12 vs
//!     scipy.signal.fftconvolve(u0, h, "same") * dx at complex128;
//!   - post-downcast (rs_apply): >= 99% of elements bit-equal to
//!     rs.rs_apply's complex64 output, every mismatch within 1 ulp.

mod common;

use common::{rel_err_c, ulp_dist_f32, Case};
use rice_bend_core::conv::{fftconvolve_same, fftconvolve_same_f64};
use rice_bend_core::fft::next_fast_len;

const CASES: [&str; 5] = ["conv_n7", "conv_n8", "conv_n2399", "conv_n2400", "conv_n2401"];

#[test]
fn next_fast_len_matches_scipy() {
    for name in CASES {
        let case = Case::load(name);
        let n = case.scalar_usize("n");
        let want = case.scalar_usize("nfft");
        assert_eq!(
            next_fast_len(2 * n - 1),
            want,
            "{name}: next_fast_len(2*{n}-1) must equal scipy's {want}"
        );
    }
}

#[test]
fn pre_downcast_rel_err() {
    for name in CASES {
        let case = Case::load(name);
        let u0 = case.c128s("u0");
        let h = case.c128s("h");
        let dx = case.scalar_f64("dx");
        let expected = case.c128s("expected_f64");

        let got = fftconvolve_same_f64(&u0, &h, dx);
        assert_eq!(got.len(), expected.len(), "{name}: length");

        let floor = expected.iter().map(|c| c.norm()).fold(0.0, f64::max) * 1e-3;
        let mut worst = 0.0f64;
        for (i, (&g, &e)) in got.iter().zip(&expected).enumerate() {
            let err = rel_err_c(g, e, floor);
            if err > worst {
                worst = err;
            }
            assert!(
                err <= 1e-12,
                "{name}[{i}]: rel err {err:.3e} > 1e-12 (got {g}, want {e})"
            );
        }
        eprintln!("{name}: pre-downcast worst rel err {worst:.3e}");
    }
}

#[test]
fn post_downcast_bit_parity() {
    for name in CASES {
        let case = Case::load(name);
        let u0 = case.c128s("u0");
        let h = case.c128s("h");
        let dx = case.scalar_f64("dx");
        let expected = case.c64s("expected_c64");

        let got = fftconvolve_same(&u0, &h, dx);
        assert_eq!(got.len(), expected.len(), "{name}: length");

        let mut exact = 0usize;
        for (i, (&g, &(ere, eim))) in got.iter().zip(&expected).enumerate() {
            // quantized values live in c128 slots; narrow for the bit compare
            let (gre, gim) = (g.re as f32, g.im as f32);
            if gre.to_bits() == ere.to_bits() && gim.to_bits() == eim.to_bits() {
                exact += 1;
            } else {
                let d = ulp_dist_f32(gre, ere).max(ulp_dist_f32(gim, eim));
                assert!(
                    d <= 1,
                    "{name}[{i}]: {d} ulps from python c64 (got {gre}+{gim}i, want {ere}+{eim}i)"
                );
            }
        }
        let frac = exact as f64 / got.len() as f64;
        eprintln!("{name}: post-downcast bit-equal {exact}/{} ({:.2}%)", got.len(), frac * 100.0);
        assert!(
            frac >= 0.99,
            "{name}: only {:.2}% of elements bit-equal after the c64 downcast (need >= 99%)",
            frac * 100.0
        );
    }
}
