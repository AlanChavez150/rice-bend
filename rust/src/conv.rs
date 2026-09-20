//! rs_apply: scipy-exact "same"-mode complex convolution + the load-bearing
//! complex64 quantization.
//!
//! Python reference: src/rice_bend/rs.py:76-85 —
//!     np.asarray(scipy.signal.fftconvolve(u0, h, mode="same") * dx, dtype=np.complex64)
//!
//! WORK PACKAGE: FFT/convolution layer. Gate: `cargo test --test conv_parity`.
//!
//! ## The exact scipy semantics to replicate (per element)
//!
//! With N = len(u0) = len(h) and nfft = next_fast_len(2N - 1):
//!   1. zero-pad u0 and h to nfft; forward FFT both (the kernel's FFT is done
//!      once per solve in `kernel_spectrum`, never per call);
//!   2. pointwise multiply the spectra;
//!   3. UNNORMALIZED inverse FFT; scale each element by fct = 1.0 / (nfft as f64)
//!      (precompute fct and multiply — scipy's pocketfft applies fct as its own
//!      multiply, so `v * fct` mirrors it more closely than `v / nfft`);
//!   4. take the centered "same" slice: elements [(N-1)/2 .. (N-1)/2 + N) of the
//!      full (2N-1)-long linear convolution (integer division; for N = 2400 the
//!      slice starts at 1199 — note the FULL result occupies the first 2N-1
//!      entries of the nfft buffer, the tail beyond 2N-1 is padding garbage);
//!   5. multiply by dx (a second, separate multiply — same order as rs.py, which
//!      scales AFTER fftconvolve returns);
//!   6. QUANTIZE each element through complex64 and store the quantized value:
//!      re = (re as f32) as f64, im = (im as f32) as f64.
//!
//! Step 6 is the rs.py:79-82 load-bearing downcast ("changing this quantisation
//! flips candidate rankings"). It happens HERE and nowhere else — the solver's
//! downstream arithmetic is f64/complex128 on these quantized values, exactly as
//! numpy promotes complex64 * float64 operands back to complex128.

use num_complex::Complex64;

use crate::fft::FftPair;

/// The forward FFT of one zero-padded kernel (len == nfft), computed once per
/// solve. The ADJOINT kernel's spectrum must be built from the time-domain
/// conjugate — `kernel_spectrum(&conj(h))` — NOT as conj of the forward
/// spectrum, which differs by a frequency reversal.
#[derive(Debug, Clone)]
pub struct KernelSpectrum {
    pub n_signal: usize,
    /// FFT(pad(h, nfft)), len == nfft.
    pub spec: Vec<Complex64>,
}

/// One per solve: FFT plans + all staging buffers for signals of length
/// `n_signal`. Every `rs_apply_*` call reuses the internal buffers — zero heap
/// allocation inside the solver loop.
pub struct Convolver {
    pub n_signal: usize,
    pub nfft: usize,
    // implementation fields go here (FftPair + one nfft staging buffer)
}

impl Convolver {
    pub fn new(n_signal: usize) -> Self {
        let _ = n_signal;
        let _ = FftPair::new; // the intended building block
        todo!("conv work package: nfft = next_fast_len(2 * n_signal - 1)")
    }

    /// FFT of the zero-padded kernel `h` (len == n_signal). Once per solve per
    /// kernel; allocation here is fine.
    pub fn kernel_spectrum(&mut self, h: &[Complex64]) -> KernelSpectrum {
        let _ = h;
        todo!("conv work package")
    }

    /// The rs_apply hot path: steps 1-6 above. `out` has len n_signal and
    /// receives the QUANTIZED values.
    pub fn rs_apply_into(
        &mut self,
        u0: &[Complex64],
        ks: &KernelSpectrum,
        dx: f64,
        out: &mut [Complex64],
    ) {
        let _ = (u0, ks, dx, out);
        todo!("conv work package")
    }

    /// Steps 1-5 WITHOUT the quantization — the pre-downcast value, used only by
    /// the parity tests (gate: rel err <= 1e-12 vs scipy at full precision).
    pub fn rs_apply_f64_into(
        &mut self,
        u0: &[Complex64],
        ks: &KernelSpectrum,
        dx: f64,
        out: &mut [Complex64],
    ) {
        let _ = (u0, ks, dx, out);
        todo!("conv work package")
    }
}

/// One-shot convenience for tests and the Python-side smoke hook: build a
/// Convolver + spectrum, apply once, return the quantized result.
pub fn fftconvolve_same(u0: &[Complex64], h: &[Complex64], dx: f64) -> Vec<Complex64> {
    let _ = (u0, h, dx);
    todo!("conv work package: Convolver::new + kernel_spectrum + rs_apply_into")
}

/// One-shot un-quantized variant (parity tests only).
pub fn fftconvolve_same_f64(u0: &[Complex64], h: &[Complex64], dx: f64) -> Vec<Complex64> {
    let _ = (u0, h, dx);
    todo!("conv work package")
}
