//! FFT plumbing: scipy-compatible fast lengths and reusable in-place transforms.
//!
//! Everything numerical funnels through one padded complex FFT per rs_apply, so
//! this module owns the two properties the whole port leans on:
//!   1. the SAME padded length scipy.signal.fftconvolve picks, and
//!   2. plan + scratch reuse so the solver loop allocates nothing.
//!
//! WORK PACKAGE: FFT/convolution layer. Gate: `cargo test --test conv_parity`.

use num_complex::Complex64;

/// Smallest n' >= n whose prime factors are all in {2, 3, 5, 7, 11}.
pub fn next_fast_len(n: usize) -> usize {
    let fast_primes = [2usize, 3, 5, 7, 11];
    let mut fast_n = n;
    loop {
        let mut leftover = fast_n;
        for curr_prime in fast_primes{
            while leftover % curr_prime == 0{
                leftover = leftover / curr_prime;
            }
        }
        if leftover == 1{
            break;
        }
        fast_n += 1;
    }

    return fast_n;
}

/// Reusable forward + inverse complex FFT plans of one fixed length `nfft`,
/// with their rustfft scratch held inside so transforms allocate nothing.
///
/// Built from `rustfft::FftPlanner::<f64>::new()` via `plan_fft_forward(nfft)` /
/// `plan_fft_inverse(nfft)`; run with `process_with_scratch` against a scratch
/// buffer sized by `get_inplace_scratch_len()`. rustfft's inverse is
/// UNNORMALIZED — no 1/nfft here; conv.rs applies the scaling where the
/// quantization contract says (see conv.rs).
///
/// Add whatever private fields the implementation needs; `nfft` and the two
/// methods are the interface `conv::Convolver` compiles against.
pub struct FftPair {
    pub nfft: usize,
    // implementation fields go here (forward/inverse plans + scratch)
}

impl FftPair {
    pub fn new(nfft: usize) -> Self {
        let _ = nfft;
        todo!("FFT work package: build forward/inverse rustfft plans + scratch")
    }

    /// In-place forward FFT of `buf` (len == nfft).
    pub fn fft_in_place(&mut self, buf: &mut [Complex64]) {
        let _ = buf;
        todo!("FFT work package")
    }

    /// In-place inverse FFT of `buf` (len == nfft), UNNORMALIZED (no 1/nfft).
    pub fn ifft_in_place(&mut self, buf: &mut [Complex64]) {
        let _ = buf;
        todo!("FFT work package")
    }
}
