import numpy as np
import scipy.integrate
import scipy.constants

from rice_bend.interp import interp_amp_phase

def generate_aperature(
        freq: float,
        x_axis: np.ndarray,
        z_max: float,
        res: int,
        a: float,
        b: float,
        c: float
    ):
    '''
    Generates a phase plate that will create a beam with trajectory ax^2 + bx + c
    '''
    z = np.linspace(0, z_max, res)
    wave_number = (2 * np.pi) / (scipy.constants.c / freq)
    caustic = (a * z**2) + (b * z) + c
    d_caustic = 2 * a* z + b

    x_caustic = caustic - z * d_caustic

    dphi_dy = (wave_number * d_caustic) / np.sqrt(1 + d_caustic**2)
    sort_idx = np.argsort(x_caustic)
    x_sorted = x_caustic[sort_idx]
    dphi_dy_sorted = dphi_dy[sort_idx]

    phi = np.flip(scipy.integrate.cumulative_trapezoid(dphi_dy_sorted, x_sorted, initial=0))
    aper = 1.0 * np.exp(1j * phi)

    # interpolate from caustic x axis to provided x axis. x_caustic is not sorted
    # (it comes out of the trajectory, not a grid), hence assume_sorted=False.
    return interp_amp_phase(x_caustic, aper, x_axis, assume_sorted=False)


