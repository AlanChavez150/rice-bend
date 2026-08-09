"""Aperture and scene geometry. No I/O, no plotting -- the .mat readers live in
exp_data.py."""

import numpy as np

from rice_bend import caustic, rs


def sampled_axis(lo: float, hi: float, spacing: float) -> np.ndarray:
    """The project's scene sampling: linspace over [lo, hi] with as many points as
    `spacing` fits into the span.

    Note linspace's actual step is (hi - lo) / (n - 1), NOT `spacing` -- the
    configured spacing sets the point COUNT and is never the step used. The "dx N
    wavelengths" log lines are slightly wrong for the same reason. Changing this is
    a real physics improvement that shifts every saved number in the 2nd-3rd
    significant digit, so it is a decision to make between result sets, not during
    a refactor.
    """
    return np.linspace(lo, hi, int((hi - lo) / spacing))


class SimAperature():
    def __init__(self, x_min: float, x_max: float, z: float, dx: float):
        assert x_max > x_min
        self.x_min = x_min
        self.x_max = x_max
        self.z = z
        self.dx = dx
        # Floor division on purpose (for now): `/` and `//` genuinely disagree in
        # three shipped configs -- 0.11/0.00025 == 440.0 but // gives 439.0 -- so
        # unifying this with sampled_axis perturbs those apertures by one sample.
        # That is Stage 7d, its own commit.
        self.num_points = int((x_max - x_min) // dx)
        self.aper_axis = np.linspace(self.x_min, self.x_max, self.num_points)
        self.aper_profile = np.zeros(len(self.aper_axis), dtype=np.complex128)

    def make_caustic(self, freq: float, z_max: float, a: float, b: float, c: float):
        self.aper_profile = caustic.generate_aperature(
            freq,
            self.aper_axis,
            z_max,
            self.num_points,
            a,
            b,
            c
        )

    def make_steer(self, freq: float, theta_deg: float):
        """
        Implements -k * x * sin(theta).
        Where theta trajectory of the beam, and k is the wavenumber
        """
        k = rs.wavenumber(freq)
        theta_rad = theta_deg * (np.pi / 180)
        phase = -1.0  * k * self.aper_axis * np.sin(theta_rad)
        self.aper_profile = 1.0 * np.exp(1j * phase)

class SimScene():
    """
    Simple class used to keep track of the parameters of the simulation.
    All units are in meters.
    """
    def __init__(self, x_min: float, x_max: float, z_min: float, z_max: float, spacing: float, tx_ap: SimAperature, rx_ap: SimAperature):
        assert x_max > x_min
        assert z_max > z_min
        self.x_min = x_min
        self.x_max = x_max
        self.z_min = z_min
        self.z_max = z_max
        self.spacing = spacing

        self.x_axis = sampled_axis(self.x_min, self.x_max, spacing)
        self.z_axis = sampled_axis(self.z_min, self.z_max, spacing)

        self.tx_ap = tx_ap
        self.rx_ap = rx_ap

        # Lazily filled by MGS.illuminate_real(). Allocating it here cost 130.6 MB
        # of complex128 that the first propagation immediately overwrote with
        # complex64 -- and in the grid-search path was never written at all.
        self.data = None
