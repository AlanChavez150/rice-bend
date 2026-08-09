import numpy as np

import scipy
import scipy.special
import scipy.signal

def kernel_rs(x: np.ndarray, wavelength: float, z: float, n: float = 1.0):
    """Rayleigh-Sommerfeld propagation kernel over transverse offset `x` for a
    propagation distance `z`."""
    k = 2 * np.pi  * n / wavelength
    r = np.sqrt(x**2 + z**2) + 1e-16
    hk = scipy.special.hankel1(1, k * r)
    return (0.5j * k * z / r) * hk

def kernel_rs_inverse(x: np.ndarray, wavelength: float, z: float, n: float = 1.0):
    return np.conjugate(kernel_rs(x, wavelength, np.abs(z), n))

def sampling_quality(x_axis: np.ndarray, z_targets: np.ndarray, wavelength: float,
                     z_src: float = 0.0, forward_dir: float = -1.0,
                     background: float = 1.0) -> float:
    """Ratio of the ideal-to-actual transverse sampling for an RS propagation.

    The densest sampling requirement is set by the nearest target plane; the
    propagation is adequately sampled when this ratio is >= 1. rs() raises if it
    drops below 1, so callers can pre-screen geometries (e.g. a grid of candidate
    TX planes) without triggering that error mid-run.
    """
    prop = forward_dir * (np.asarray(z_targets, dtype=float) - z_src)
    dx = x_axis[1] - x_axis[0]
    dr_real = np.sqrt(dx**2)
    rmax = np.sqrt(x_axis**2).max()
    wave_ratio = wavelength / background
    nearest = np.min(np.abs(prop))
    dr_ideal = np.sqrt(wave_ratio**2 + rmax**2 + 2 * wave_ratio * np.sqrt(rmax**2 + nearest**2)) - rmax
    return dr_ideal / dr_real

def rs(x_axis: np.ndarray, z_targets: np.ndarray, u0: np.ndarray, wavelength: float,
       z_src: float = 0.0, forward_dir: float = -1.0) -> np.ndarray:
    """
    Implements Rayleigh-Sommerfeld propogation, both forward and backwards.

    `u0` is the field defined at the source plane `z_src` (an absolute z
    coordinate). For each absolute z coordinate in `z_targets`, the field is
    propagated there. `forward_dir` is the sign of the direction in which energy
    physically flows (-1.0 => -Z is "forward", the project's convention).

    The signed propagation distance to a target is `forward_dir * (z - z_src)`:
      - prop >= 0 => forward propagation (kernel_rs)
      - prop <  0 => adjoint / back-propagation (kernel_rs_inverse)
    """
    assert len(u0) == len(x_axis)

    # signed propagation distance from the source plane to each target plane
    prop = forward_dir * (np.asarray(z_targets, dtype=float) - z_src)
    dx = x_axis[1] - x_axis[0]

    # worst-case (densest) sampling requirement is set by the nearest target plane
    quality = sampling_quality(x_axis, z_targets, wavelength, z_src=z_src, forward_dir=forward_dir)
    if quality < 1:
        raise RuntimeError(f"needs denser sampling. {quality=} {dx=} {wavelength=}")

    # Benchmarked (do not re-litigate): batching this loop into one 2D kernel matrix
    # and a single fftconvolve runs at 0.86-0.93x -- i.e. SLOWER -- at every size
    # tried, and costs a 130 MB kernel matrix. hankel1 is the expense here, not the
    # FFT (0.375 ms vs 0.110 ms per call); hoisting it out is what pays, and
    # gs_reconstruct does exactly that via rs_kernel/rs_apply.
    s_mat = np.zeros(shape=(len(prop), len(x_axis)), dtype=np.complex64)
    for z_idx, curr_prop in enumerate(prop):
        h = None
        if curr_prop >= 0:
            h = kernel_rs(x_axis, wavelength, curr_prop, 1.0)
        else:
            h = kernel_rs_inverse(x_axis, wavelength, curr_prop, 1.0)
        s = scipy.signal.fftconvolve(u0, h, mode="same") * dx
        s_mat[z_idx] = s

    return s_mat
