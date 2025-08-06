import numpy as np
import scipy.constants
import scipy.special

class SimAperature():
    def __init__(self, x_min: float, x_max: float, z: float, dx: float):
        assert x_max > x_min
        self.x_min = x_min
        self.x_max = x_max
        self.z = z
        self.dx = dx
        num_points = int((x_max - x_min) // dx)
        self.aper_axis = np.linspace(self.x_min, self.x_max, num_points)
        self.aper_profile = np.zeros(len(self.aper_axis), dtype=np.cfloat)

    def _wavenumber(self, freq: float) -> float:
        return 2 * np.pi / (scipy.constants.c / freq)

    def make_caustic(self, freq: float):
        pass

    def make_steer(self, freq: float, theta_deg: float):
        """
        Implements -k * x * sin(theta).
        Where theta trajectory of the beam, and k is the wavenumber
        """
        k = self._wavenumber(freq)
        theta_rad = theta_deg * (np.pi / 180)
        phase = -1.0  * k * self.aper_axis * np.sin(theta_rad)
        self.aper_profile = 1.0 * np.exp(1j * phase)

    def make_bessel(self, freq: float):
        pass

    def make_airy(self, freq: float, x0: float, alpha: float):
        # is freq really not used for anything?
        s = self.aper_axis / x0
        envelope = np.exp(alpha * s)
        airy_func = scipy.special.airy(s)[0]
        self.aper_profile = airy_func * envelope

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

        x_points = int((self.x_max - self.x_min) / spacing)
        self.x_axis = np.linspace(self.x_min, self.x_max, x_points)
        self.dx = self.x_axis[1] - self.x_axis[0]

        z_points = int((self.z_max - self.z_min) / spacing)
        self.z_axis = np.linspace(self.z_min, self.z_max, z_points)
        self.dz = self.z_axis[1] - self.z_axis[0]

        self.tx_ap = tx_ap
        self.rx_ap = rx_ap

        self.data = np.zeros(shape=(len(self.z_axis), len(self.x_axis)), dtype=np.cfloat)