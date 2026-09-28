//! Bit-exact mirrors of the numpy reductions and helpers the solver leans on.
//!
//! numpy 1.24's `np.add.reduce` over a contiguous float array is pairwise
//! summation starting from 0 (verified 600/600 bit-exact on this project's
//! numpy); `np.mean` is that sum divided by n in the array's own precision.
//! Mirroring these costs nothing and removes a source of drift between the
//! engines. numpy's SIMD complex `exp`/`abs` are NOT mirrored — they differ
//! from libm at the 1-2 ulp level and reproducing them is not worth it.
//! Checked against numpy by tests/numerics_parity.rs.

use std::f64::consts::PI;
use std::ops::Add;

/// numpy's pairwise-summation block size (PW_BLOCKSIZE in loops_utils.h).
const PW_BLOCKSIZE: usize = 128;

fn pairwise_sum<T: Copy + Add<Output = T> + Default>(a: &[T]) -> T {
    let n = a.len();
    if n < 8 {
        let mut res = T::default();
        for &x in a {
            res = res + x;
        }
        res
    } else if n <= PW_BLOCKSIZE {
        let mut r = [T::default(); 8];
        r.copy_from_slice(&a[..8]);
        let blocks_end = n - n % 8;
        let mut i = 8;
        while i < blocks_end {
            for j in 0..8 {
                r[j] = r[j] + a[i + j];
            }
            i += 8;
        }
        let mut res = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
        for &x in &a[blocks_end..] {
            res = res + x;
        }
        res
    } else {
        let half = n / 2;
        let n2 = half - half % 8;
        pairwise_sum(&a[..n2]) + pairwise_sum(&a[n2..])
    }
}

/// `np.sum` / `np.add.reduce` of a contiguous f64 array.
pub fn np_sum_f64(a: &[f64]) -> f64 {
    pairwise_sum(a)
}

/// `np.sum` / `np.add.reduce` of a contiguous f32 array, accumulated in f32.
pub fn np_sum_f32(a: &[f32]) -> f32 {
    pairwise_sum(a)
}

/// `np.mean` of a contiguous f64 array.
pub fn np_mean_f64(a: &[f64]) -> f64 {
    np_sum_f64(a) / a.len() as f64
}

/// numpy's float `%` (npy_divmod's remainder): fmod, then shifted into the
/// divisor's sign; an exact zero takes the divisor's sign.
pub fn np_mod(a: f64, b: f64) -> f64 {
    let m = a % b;
    if m != 0.0 {
        if (b < 0.0) != (m < 0.0) {
            m + b
        } else {
            m
        }
    } else {
        0.0f64.copysign(b)
    }
}

/// `np.unwrap(p)` with the default period 2π (numpy 1.24 source semantics).
pub fn np_unwrap(p: &[f64]) -> Vec<f64> {
    let mut out = p.to_vec();
    if p.len() < 2 {
        return out;
    }
    let period = 2.0 * PI;
    let interval_high = period / 2.0;
    let interval_low = -interval_high;
    let discont = period / 2.0;
    let mut cum = 0.0;
    for k in 1..p.len() {
        let dd = p[k] - p[k - 1];
        let mut ddmod = np_mod(dd - interval_low, period) + interval_low;
        if ddmod == interval_low && dd > 0.0 {
            ddmod = interval_high;
        }
        let mut correct = ddmod - dd;
        if dd.abs() < discont {
            correct = 0.0;
        }
        cum += correct;
        out[k] = p[k] + cum;
    }
    out
}
