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

    # The bessel and focus plates are laid out about the aperture centre, so the
    # beam moves rigidly with the TX window. x_axis is a linspace over
    # [x_min, x_max], so its end points give that centre exactly. Their steer angle
    # is optional: null means the beam leaves straight down.
    x_center = 0.5 * (x_axis[0] + x_axis[-1])
    theta_deg = beam.steer_angle_deg or 0.0
    if beam.type == "bessel":
        # axicon: two plane-wave halves crossing at the cone angle
        return bessel_profile(freq, x_axis, x_center, beam.cone_angle_deg, theta_deg)
    elif beam.type == "focus":
        # lens converging on the point focal_length away along the steer angle
        return focused_profile(freq, x_axis, x_center, beam.focal_length, theta_deg)
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
    0 <= d <= prop_len. The plate only exists between c - a*prop_len^2 and c; the
    rest of `x_axis` is exactly zero.

    The sign of a picks the side: a > 0 puts the plate on the -x side of c and
    bends the beam toward +x. Negating a and b gives the mirror image about x = c
    (plate on the +x side, bending toward -x), up to a constant phase offset that no
    intensity sees: the phase is integrated from the plate's -x end either way,
    which is the far edge for a > 0 and the c edge for a < 0.
    '''
    z = np.linspace(0, prop_len, len(x_axis))
    wave_number = rs.wavenumber(freq)
    caustic = (a * z**2) + (b * z) + c
    d_caustic = 2 * a* z + b

    x_caustic = caustic - z * d_caustic

    dphi_dy = (wave_number * d_caustic) / np.sqrt(1 + d_caustic**2)
    # x_caustic comes out of the trajectory, not a grid: it runs from c to the far
    # edge, so it is descending for a > 0. Sort it once and stay on the sorted axis
    sort_idx = np.argsort(x_caustic)
    x_sorted = x_caustic[sort_idx]
    dphi_dy_sorted = dphi_dy[sort_idx]

    phi_sorted = scipy.integrate.cumulative_trapezoid(dphi_dy_sorted, x_sorted, initial=0)

    # Resample the UNWRAPPED phase, and only then take exp(1j * phi). The caustic
    # samples are quadratic in d and do not line up with `x_axis`, so interpolating
    # the wrapped phase instead (or |aper| and angle(aper)) averages through the
    # +/-pi branch cut wherever an `x_axis` sample lands in an interval that holds a
    # wrap, and leaves it at full amplitude with an arbitrary phase.
    phi = np.interp(x_axis, x_sorted, phi_sorted)

    # The plate's far edge c - a*prop_len^2 and the aperture edge cut to it (x_min
    # for a > 0) are the same number computed two ways and can land an ULP apart,
    # which would drop the edge sample of a window cut exactly to the support
    # (scenario_caustic_miss, scenario_caustic_partial, any shifted plate). 1e-9 m
    # is far below any dx and far above that rounding. np.interp clamps to the end
    # values, so a sample let in by the tolerance takes the edge phase.
    inside = (x_axis >= x_sorted[0] - 1e-9) & (x_axis <= x_sorted[-1] + 1e-9)
    return np.where(inside, np.exp(1j * phi), 0j)


def steered_profile(freq: float, x_axis: np.ndarray, theta_deg: float) -> np.ndarray:
    """
    Implements -k * x * sin(theta).
    Where theta trajectory of the beam, and k is the wavenumber

    A positive theta steers toward -x. bessel_profile and focused_profile take
    their steer angle in the same sense.
    """
    k = rs.wavenumber(freq)
    theta_rad = theta_deg * (np.pi / 180)
    phase = -1.0  * k * x_axis * np.sin(theta_rad)
    return 1.0 * np.exp(1j * phase)


def bessel_profile(freq: float, x_axis: np.ndarray, x_center: float, cone_deg: float,
                   theta_deg: float = 0.0) -> np.ndarray:
    """
    Axicon phase plate, the 2D Bessel beam: -k * |x - x_center| * sin(cone).
    The half of the plate either side of `x_center` emits a plane wave tilted
    `cone_deg` toward the axis, and where the two cross they interfere into
    cos(k * sin(cone) * (x - x_center)) fringes that do not spread.

    In this 2D scene those are EQUAL-HEIGHT fringes of period
    wavelength / (2 * sin(cone)), not one dominant core (that is J0, the radially
    symmetric 3D case). The halves overlap down to (W / 2) / tan(cone) below a plate
    of width W, and the overlap is widest, W / 2, at half that range.

    `theta_deg` tilts the axis, with steered_profile's sign: the halves leave at
    theta +/- cone, and a vanishing cone leaves the steered plane wave.
    """
    k = rs.wavenumber(freq)
    cone_rad = cone_deg * (np.pi / 180)
    theta_rad = theta_deg * (np.pi / 180)
    u = x_axis - x_center
    phase = -1.0 * k * u * np.where(u >= 0, np.sin(theta_rad + cone_rad),
                                    np.sin(theta_rad - cone_rad))
    return 1.0 * np.exp(1j * phase)


def focused_profile(freq: float, x_axis: np.ndarray, x_center: float,
                    focal_length: float, theta_deg: float = 0.0) -> np.ndarray:
    """
    Lens phase plate: -k * (r - focal_length), r the distance from each sample to
    the focal point, so every sample arrives there in phase. The focal point sits
    `focal_length` from `x_center` along the ray `theta_deg` off the -Z axis:
    (x_center - focal_length * sin(theta), focal_length * cos(theta) below the plate).

    The path length is exact, not the paraxial x^2 / (2 * focal_length): the shipped
    apertures are a good fraction of their stand-off. `theta_deg` has
    steered_profile's sign, and this is that plane wave as focal_length -> inf.
    The focal spot is about wavelength * focal_length / W wide for a plate of
    width W.
    """
    k = rs.wavenumber(freq)
    theta_rad = theta_deg * (np.pi / 180)
    u = x_axis - x_center
    r = np.sqrt((u + focal_length * np.sin(theta_rad))**2
                + (focal_length * np.cos(theta_rad))**2)
    phase = -1.0 * k * (r - focal_length)
    return 1.0 * np.exp(1j * phase)
