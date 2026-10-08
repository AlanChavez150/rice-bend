"""Every beam the TX aperture can emit, and the dispatch from the config's
`tx_aperture.beam` block to the generator that builds it. A new beam type is a
change here and to config.BeamConfig, and nowhere else.

Each generator returns the complex aperture profile sampled on `x_axis`, and all of
them keep to one contract the solver relies on:
  - frequency enters the PHASE only. |profile| must not depend on `freq`: a joint
    solve takes the primary frequency's amplitude as the fixed constraint for every
    tone (see MGS.run_gerch_sax)
  - samples outside the beam's support are exactly 0. The solver's support mask is
    |profile| > 0, so a dark sample left nonzero is solved for as if it radiated
"""

import numpy as np
import scipy.integrate

from rice_bend import rs
from rice_bend.config import BeamConfig
from rice_bend.interp import interp_amp_phase


def beam_profile(beam: BeamConfig, freq: float, x_axis: np.ndarray,
                 prop_len: float) -> np.ndarray:
    """The profile of the configured `beam` at `freq`, sampled on `x_axis`.
    `prop_len` is the distance from the TX plane down to the scene floor; only the
    caustic uses it."""
    if beam.type == "caustic":
        # caustic beam x(d) = a*d^2 + b*d + c, d = distance travelled from the TX,
        # so it is parameterised by the downstream propagation length
        a, b, c = beam.trajectory
        return caustic_profile(freq, x_axis, prop_len, a, b, c)
    elif beam.type == "directional":
        # steered plane wave at the configured angle
        return steered_profile(freq, x_axis, theta_deg=beam.steer_angle_deg)
    else:
        raise ValueError(f"unknown beam type: {beam.type}")


def caustic_profile(
        freq: float,
        x_axis: np.ndarray,
        prop_len: float,
        a: float,
        b: float,
        c: float
    ) -> np.ndarray:
    '''
    Generates a phase plate that will create a beam with trajectory
    x(d) = a*d^2 + b*d + c, d the distance travelled from the plate, designed over
    0 <= d <= prop_len. The plate only exists on [c - a*prop_len^2, c]; the rest of
    `x_axis` is zero-filled.

    Assumes a > 0. The np.flip below lines phi up with x_caustic only when x_caustic
    comes out descending, which is the a > 0 case; a < 0 passes the config's
    validation but does not produce the mirrored beam.
    '''
    z = np.linspace(0, prop_len, len(x_axis))
    wave_number = rs.wavenumber(freq)
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


def steered_profile(freq: float, x_axis: np.ndarray, theta_deg: float) -> np.ndarray:
    """
    Implements -k * x * sin(theta).
    Where theta trajectory of the beam, and k is the wavenumber
    """
    k = rs.wavenumber(freq)
    theta_rad = theta_deg * (np.pi / 180)
    phase = -1.0  * k * x_axis * np.sin(theta_rad)
    return 1.0 * np.exp(1j * phase)
