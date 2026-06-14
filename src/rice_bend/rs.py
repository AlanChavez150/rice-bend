import numpy as np

import scipy
import scipy.special
import scipy.signal

def kernel_rs(x: np.ndarray, wavelength: float, z: float, n: float = 1.0, kind: str = "x"):
    k = 2 * np.pi  * n / wavelength
    r = np.sqrt(x**2 + z**2) + 1e-16
    hk = scipy.special.hankel1(1, k * r)
    if kind == "z":
        return (0.5j * k * z / r) * hk
    elif kind == "x":
        return (0.5j * k * x / r) * hk
    else:
        raise ValueError(f"Invalid axis {x}")

def kernel_rs_inverse(x: np.ndarray, wavelength: float, z: float, n: float = 1.0, kind: str = "x"):
    return np.conjugate(kernel_rs(x, wavelength, np.abs(z), n, kind))

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
    dr_real = np.sqrt(dx**2)
    rmax = np.sqrt(x_axis**2).max()
    background = 1.0
    wave_ratio = wavelength / background
    # worst-case (densest) sampling requirement is set by the nearest target plane
    nearest = np.min(np.abs(prop))
    dr_ideal = np.sqrt(wave_ratio**2 + rmax**2 + 2 * wave_ratio * np.sqrt(rmax**2 + nearest**2)) - rmax
    quality = dr_ideal / dr_real
    if quality < 1:
        raise RuntimeError(f"needs denser sampling. {quality=} {dr_ideal=} {dr_real=} {dx} {wavelength}")

    s_mat = np.zeros(shape=(len(prop), len(x_axis)), dtype=np.complex64)
    for z_idx, curr_prop in enumerate(prop):
        h = None
        if curr_prop >= 0:
            h = kernel_rs(x_axis, wavelength, curr_prop, 1.0, kind="z")
        else:
            h = kernel_rs_inverse(x_axis, wavelength, curr_prop, 1.0, kind="z")
        s = scipy.signal.fftconvolve(u0, h, mode="same") * dx
        s_mat[z_idx] = s

    return s_mat
