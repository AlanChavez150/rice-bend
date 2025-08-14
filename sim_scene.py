import pathlib as Path
from copy import deepcopy

import numpy as np
import scipy.constants
import scipy.interpolate
import scipy.special
import h5py

import caustic

class SimAperature():
    def __init__(self, x_min: float, x_max: float, z: float, dx: float):
        assert x_max > x_min
        self.x_min = x_min
        self.x_max = x_max
        self.z = z
        self.dx = dx
        self.num_points = int((x_max - x_min) // dx)
        self.aper_axis = np.linspace(self.x_min, self.x_max, self.num_points)
        self.aper_profile = np.zeros(len(self.aper_axis), dtype=np.cfloat)

    def _wavenumber(self, freq: float) -> float:
        return 2 * np.pi / (scipy.constants.c / freq)

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

    def interp_axis(self, new_axis: np.ndarray, assume_sorted: bool = True):
        """
        Interpolates the current aperature onto a different x axis.

        Amplitude and phase interpolated seperately
        """
        ampl_interp_func = scipy.interpolate.interp1d(
            self.aper_axis,
            np.abs(self.aper_profile),
            kind="linear",
            fill_value=0,
            bounds_error=False,
            assume_sorted=assume_sorted
        )
        phs_interp_func = scipy.interpolate.interp1d(
            self.aper_axis,
            np.angle(self.aper_profile),
            kind="linear",
            fill_value=0,
            bounds_error=False,
            assume_sorted=assume_sorted
        )
        return ampl_interp_func(new_axis) * np.exp(1j * phs_interp_func(new_axis))

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

def parse_oscope_rx_data(path: Path, freq_c: float, lo_freq: float = 25e9, trx_n: float = 6) -> SimAperature:
    """
    Reads a .mat file created by experimental_data/heatmap.m
    """
    down_mix_freq = freq_c - (lo_freq * trx_n)
    if down_mix_freq < 0:
        err_msg = f"Incorrect carrier frequency. carrier {freq_c*1e-9:0.1f} GHz. LO Freq {lo_freq*1e-9:0.1f} GHz {trx_n*1e-9:0.1f} GHz"
        raise ValueError(err_msg)

    with h5py.File(path, "r") as f:
        # HDF5 datasets behave like NumPy arrays once opened
        xvec     = f["xvec"][:].squeeze()          # shape (Nx,)
        zvec     = f["zvec"][:].flatten()          # shape (Nz,)
        xax_td   = f["xax_td"][:].flatten()
        tds      = f["tds"]

        tds_plane = tds[:, :, 0]
        tds_plane = tds_plane.T

    sample_rate = 1.0 / (xax_td[1] - xax_td[0])

    # convert from mm to m
    xvec = np.array(xvec * 1e-3)
    zvec = np.array(zvec * 1e-3)

    aper_profile = np.zeros(shape=xvec.shape, dtype=np.cfloat)
    for x_idx in range(xvec.shape[0]):
        curr_fft = np.fft.fft(tds_plane[x_idx, :])
        freq_axis = np.arange(0, len(curr_fft), 1)
        freq_axis = freq_axis * (sample_rate / len(curr_fft))

        freq_carrier_bin = int(down_mix_freq / (sample_rate / len(curr_fft)))
        aper_profile[x_idx] = curr_fft[freq_carrier_bin]

    aper = SimAperature(
        x_min=xvec.min(),
        x_max=xvec.max(),
        z=zvec[0],
        dx=xvec[1] - xvec[0]
    )
    aper.aper_axis = xvec
    aper.aper_profile = aper_profile[::-1]
    return aper

def parse_oscope_heatmap_data(
        path: Path,
        base_scene: SimScene,
        x_off: float,
        freq_c: float,
        lo_freq: float=25e9,
        trx_n: float = 6
    ) -> SimScene:
    """
    Reads a .mat file created by experimental_data/heatmap.m

    Experimental data is interpolated onto a base scene
    """
    down_mix_freq = freq_c - (lo_freq * trx_n)
    if down_mix_freq < 0:
        err_msg = f"Incorrect carrier frequency. carrier {freq_c*1e-9:0.1f} GHz. LO Freq {lo_freq*1e-9:0.1f} GHz {trx_n*1e-9:0.1f} GHz"
        raise ValueError(err_msg)

    with h5py.File(path, "r") as f:
        # HDF5 datasets behave like NumPy arrays once opened
        xvec     = f["xvec"][:].squeeze()          # shape (Nx,)
        zvec     = f["zvec"][:].flatten()          # shape (Nz,)
        xax_td   = f["xax_td"][:].flatten()
        tds      = f["tds"]


        tds = np.array(tds)
        tds_plane = tds.T

    sample_rate = 1.0 / (xax_td[1] - xax_td[0])

    # convert from mm to m
    xvec = np.array(xvec * 1e-3)
    zvec = np.array(zvec * 1e-3)

    exp_data = np.zeros(shape=(zvec.shape[0], xvec.shape[0]), dtype=np.cfloat)
    for z_idx in range(zvec.shape[0]):
        for x_idx in range(xvec.shape[0]):
            curr_fft = np.fft.fft(tds_plane[z_idx, x_idx])
            freq_axis = np.arange(0, len(curr_fft), 1)
            freq_axis = freq_axis * (sample_rate / len(curr_fft))
            freq_carrier_bin = int(down_mix_freq / (sample_rate / len(curr_fft)))
            exp_data[z_idx][x_idx] = curr_fft[freq_carrier_bin]

    zvec = (0.3 - zvec)
    xvec = (0.3 - xvec)
    xvec += x_off
    interp_2d = scipy.interpolate.RegularGridInterpolator(
        (zvec, xvec),
        exp_data,
        bounds_error=False,
        fill_value=0.0+0.0j
    )

    exp_scene = deepcopy(base_scene)
    base_z, base_x = np.meshgrid(base_scene.z_axis, base_scene.x_axis, indexing="ij")
    exp_scene.data = interp_2d((base_z, base_x))
    return exp_scene
