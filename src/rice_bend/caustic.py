import numpy as np
import scipy.integrate
import scipy.constants
import scipy.interpolate

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

    phi = np.flip(scipy.integrate.cumtrapz(dphi_dy_sorted, x_sorted, initial=0))
    aper = 1.0 * np.exp(1j * phi)

    # interpolate from cuastic x axis to provided x axis
    ampl_interp_func = scipy.interpolate.interp1d(
        x_caustic,
        np.abs(aper),
        kind="linear",
        fill_value=0,
        bounds_error=False,
        assume_sorted=False
    )
    phs_interp_func = scipy.interpolate.interp1d(
        x_caustic,
        np.angle(aper),
        kind="linear",
        fill_value=0,
        bounds_error=False,
        assume_sorted=False
    )

    aper_interp = ampl_interp_func(x_axis) * np.exp(1j * phs_interp_func(x_axis))
    return aper_interp


