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

use crate::fft::{next_fast_len, FftPair};

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
    fft: FftPair,
    buf: Vec<Complex64>,
    /// 1/nfft, precomputed: pocketfft applies its normalization as a multiply.
    fct: f64,
    /// First index of the centered "same" slice of the linear convolution.
    start: usize,
}

impl Convolver {
    pub fn new(n_signal: usize) -> Self {
        assert!(n_signal > 0, "Convolver needs a non-empty signal");
        let nfft = next_fast_len(2 * n_signal - 1);
        Convolver {
            n_signal,
            nfft,
            fft: FftPair::new(nfft),
            buf: vec![Complex64::new(0.0, 0.0); nfft],
            fct: 1.0 / nfft as f64,
            start: (n_signal - 1) / 2,
        }
    }

    /// FFT of the zero-padded kernel `h` (len == n_signal). Once per solve per
    /// kernel; allocation here is fine.
    pub fn kernel_spectrum(&mut self, h: &[Complex64]) -> KernelSpectrum {
        assert_eq!(h.len(), self.n_signal, "kernel length");
        let mut spec = vec![Complex64::new(0.0, 0.0); self.nfft];
        spec[..self.n_signal].copy_from_slice(h);
        self.fft.fft_in_place(&mut spec);
        KernelSpectrum { n_signal: self.n_signal, spec }
    }

    /// Steps 1-5: leaves the full-precision "same" result in `out`.
    fn convolve_same(&mut self, u0: &[Complex64], ks: &KernelSpectrum, dx: f64, out: &mut [Complex64]) {
        let n = self.n_signal;
        debug_assert_eq!(u0.len(), n);
        debug_assert_eq!(out.len(), n);
        debug_assert_eq!(ks.n_signal, n);

        self.buf[..n].copy_from_slice(u0);
        self.buf[n..].fill(Complex64::new(0.0, 0.0));
        self.fft.fft_in_place(&mut self.buf);
        // signal spectrum on the left: numpy evaluates sp1 * sp2 in this order
        for (b, &k) in self.buf.iter_mut().zip(&ks.spec) {
            *b *= k;
        }
        self.fft.ifft_in_place(&mut self.buf);

        let (fct, start) = (self.fct, self.start);
        for (o, &v) in out.iter_mut().zip(&self.buf[start..start + n]) {
            // two separate multiplies: pocketfft's 1/nfft, then rs.py's * dx
            *o = Complex64::new((v.re * fct) * dx, (v.im * fct) * dx);
        }
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
        self.convolve_same(u0, ks, dx, out);
        for o in out.iter_mut() {
            *o = quantize_c64(*o);
        }
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
        self.convolve_same(u0, ks, dx, out);
    }
}

/// The load-bearing complex64 round-trip (rs.py:79-82), kept in c128 storage.
#[inline]
pub fn quantize_c64(v: Complex64) -> Complex64 {
    Complex64::new((v.re as f32) as f64, (v.im as f32) as f64)
}

/// One-shot convenience for tests and the Python-side smoke hook: build a
/// Convolver + spectrum, apply once, return the quantized result.
pub fn fftconvolve_same(u0: &[Complex64], h: &[Complex64], dx: f64) -> Vec<Complex64> {
    let mut conv = Convolver::new(u0.len());
    let ks = conv.kernel_spectrum(h);
    let mut out = vec![Complex64::new(0.0, 0.0); u0.len()];
    conv.rs_apply_into(u0, &ks, dx, &mut out);
    out
}

/// One-shot un-quantized variant (parity tests only).
pub fn fftconvolve_same_f64(u0: &[Complex64], h: &[Complex64], dx: f64) -> Vec<Complex64> {
    let mut conv = Convolver::new(u0.len());
    let ks = conv.kernel_spectrum(h);
    let mut out = vec![Complex64::new(0.0, 0.0); u0.len()];
    conv.rs_apply_f64_into(u0, &ks, dx, &mut out);
    out
}
