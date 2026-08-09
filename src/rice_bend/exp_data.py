"""Readers for the experimental .mat captures produced by experimental_data/heatmap.m.

Kept out of sim_scene.py, which is geometry and has no business knowing about
oscilloscope captures. Both readers extract the same thing -- the complex amplitude
of the down-converted carrier in a time-domain trace -- one over a line of x
positions, one over an (z, x) raster.

MATLAB's -v7.3 saves an [Nz, Nx, Nt] array that h5py reads back as (Nt, Nx, Nz);
`xvec`, `zvec` and `xax_td` are row vectors. Both readers depend on exactly that.
"""

from copy import deepcopy
from pathlib import Path

import numpy as np
import scipy.interpolate

from rice_bend.sim_scene import SimAperature, SimScene


def _down_mix_freq(freq_c: float, exp) -> float:
    """Where the carrier lands after the receive chain's down-conversion."""
    down_mix_freq = freq_c - (exp.lo_freq * exp.trx_n)
    if down_mix_freq < 0:
        raise ValueError(
            f"Incorrect carrier frequency. carrier {freq_c*1e-9:0.1f} GHz. "
            f"LO Freq {exp.lo_freq*1e-9:0.1f} GHz {exp.trx_n*1e-9:0.1f} GHz")
    return down_mix_freq


def _carrier_bin(trace: np.ndarray, sample_rate: float, down_mix_freq: float):
    """Complex amplitude of the down-converted carrier in one time-domain trace."""
    curr_fft = np.fft.fft(trace)
    return curr_fft[int(down_mix_freq / (sample_rate / len(curr_fft)))]


def _read_capture(path: Path):
    """Common header of both capture formats: (xvec, zvec, sample_rate, tds) with the
    position vectors converted from mm to m.

    h5py is imported HERE and not at module scope: it is needed only on this path,
    and importing it up top made an environment without h5py break both simulated
    workflows at import time.
    """
    import h5py

    with h5py.File(path, "r") as f:
        # HDF5 datasets behave like NumPy arrays once opened
        xvec = f["xvec"][:].squeeze()          # shape (Nx,)
        zvec = f["zvec"][:].flatten()          # shape (Nz,)
        xax_td = f["xax_td"][:].flatten()
        tds = np.array(f["tds"])               # (Nt, Nx, Nz)

    sample_rate = 1.0 / (xax_td[1] - xax_td[0])
    return np.array(xvec * 1e-3), np.array(zvec * 1e-3), sample_rate, tds


def parse_oscope_rx_data(path: Path, freq_c: float, exp) -> SimAperature:
    """One z plane of a capture, as the measured RX aperture profile."""
    down_mix_freq = _down_mix_freq(freq_c, exp)
    xvec, zvec, sample_rate, tds = _read_capture(path)
    tds_plane = tds[:, :, 0].T                 # (Nx, Nt)

    aper_profile = np.zeros(shape=xvec.shape, dtype=np.complex128)
    for x_idx in range(xvec.shape[0]):
        aper_profile[x_idx] = _carrier_bin(tds_plane[x_idx, :], sample_rate, down_mix_freq)

    aper = SimAperature(x_min=xvec.min(), x_max=xvec.max(), z=zvec[0],
                        dx=xvec[1] - xvec[0])
    aper.aper_axis = xvec
    aper.aper_profile = aper_profile[::-1]
    return aper


def parse_oscope_heatmap_data(path: Path, base_scene: SimScene, x_off: float,
                              freq_c: float, exp) -> SimScene:
    """A full (z, x) raster capture, interpolated onto `base_scene`'s grid."""
    down_mix_freq = _down_mix_freq(freq_c, exp)
    xvec, zvec, sample_rate, tds = _read_capture(path)
    tds_plane = tds.T                          # (Nz, Nx, Nt)

    exp_data = np.zeros(shape=(zvec.shape[0], xvec.shape[0]), dtype=np.complex128)
    for z_idx in range(zvec.shape[0]):
        for x_idx in range(xvec.shape[0]):
            exp_data[z_idx][x_idx] = _carrier_bin(tds_plane[z_idx, x_idx],
                                                  sample_rate, down_mix_freq)

    # rig coordinates mirror into scene coordinates, then shift onto the recentred scene
    zvec = exp.rig_z_origin - zvec
    xvec = exp.rig_x_origin - xvec
    xvec += x_off
    interp_2d = scipy.interpolate.RegularGridInterpolator(
        (zvec, xvec), exp_data, bounds_error=False, fill_value=0.0 + 0.0j)

    exp_scene = deepcopy(base_scene)
    base_z, base_x = np.meshgrid(base_scene.z_axis, base_scene.x_axis, indexing="ij")
    exp_scene.data = interp_2d((base_z, base_x))
    return exp_scene
