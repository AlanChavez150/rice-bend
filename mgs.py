import logging
import argparse

import coloredlogs
import numpy as np
import scipy.constants
import scipy.interpolate
import matplotlib.pyplot as plt

import rs
from sim_scene import SimAperature, SimScene

class MGS():
    def __init__(self, freq: float):
        self.log = logging.getLogger()
        # set up simulation scene
        x_min = -0.2
        x_max =  0.2
        z_min = 0.0
        z_max = 0.5
        self.freq = 150e9
        self.wavelength = scipy.constants.c / freq
        rx_spacing_ratio = 1 / 10.0
        rx_spacing = self.wavelength * rx_spacing_ratio

        rx = SimAperature(
            x_min=x_min,
            x_max=x_max,
            z=0.25,
            dx=rx_spacing
        )
        tx = SimAperature(
            x_min=x_min + 0.1,
            #x_max=(x_max - x_min) * 0.75 + x_min,
            x_max=x_max - 0.1,
            z=0,
            dx=0.5e-3
        )
        tx.make_steer(self.freq, theta_deg=15)

        self.scene = SimScene(
            x_min=x_min,
            x_max=x_max,
            z_min=z_min,
            z_max=z_max,
            spacing=0.25e-3,
            rx_ap=rx,
            tx_ap=tx
        )

        # Log parameters
        self.log.info(f"Simulation scene:")
        self.log.info(f" - X axis: {self.scene.x_min:0.3f} - {self.scene.x_max:0.3f}")
        self.log.info(f" - Z axis: {self.scene.z_min:0.3f} - {self.scene.z_max:0.3f}")
        self.log.info(f"RX aperature:")
        rx_s_num = 1
        rx_s_den = 1 / rx_spacing_ratio
        self.log.info(f" - X axis: {self.scene.rx_ap.x_min:0.3f} {self.scene.rx_ap.x_max:0.3f}. {rx_s_num:1.1f}/{rx_s_den:1.1f} wavelength")
        self.log.info(f" - Z: {self.scene.rx_ap.z:0.3f}")
        self.log.info(f"TX aperature")
        self.log.info(f" - X axis: {self.scene.tx_ap.x_min:0.3f} {self.scene.tx_ap.x_max:0.3f}. {rx_s_num:1.1f}/{rx_s_den:1.1f} wavelength")
        self.log.info(f" - Z: {self.scene.tx_ap.z:0.3f}")

    def run_sim(self):
        """
        Simulates RF energy in entire scene for pretty plots.
        Also measures data at RX antennes for later use.
        """
        self.log.info("Interpolating TX aperature into scene coordinates")
        # redefine tx aperature coordinates and interp data.
        #assert self.scene.spacing < self.scene.tx_ap.dx
        assert np.isclose(self.scene.tx_ap.z, 0, 0.001) # Currently only works if tx is at z=0.0
        tx_profile_interp_func = scipy.interpolate.interp1d(
            self.scene.tx_ap.aper_axis,
            self.scene.tx_ap.aper_profile,
            kind="linear",
            fill_value=complex(0, 0),
            bounds_error=False,
            assume_sorted=True
        )
        tx_profile_interp = tx_profile_interp_func(self.scene.x_axis)

        self.log.info("Computing wave propogation across scene")
        self.scene.data = rs.rs(self.scene.x_axis, self.scene.z_axis, tx_profile_interp, self.wavelength)

        # Fill data at recievers
        scene_r_z_idx = np.searchsorted(self.scene.z_axis, self.scene.rx_ap.z)-1
        scene_r_z_idx = np.max([0, scene_r_z_idx]) # prevents indexes under zero
        scene_r_z = self.scene.z_axis[scene_r_z_idx]
        scene_r_z_next = self.scene.z_axis[scene_r_z_idx+1]
        # check if closer to other z point
        if np.abs(scene_r_z - self.scene.rx_ap.z) < np.abs(scene_r_z_next - self.scene.rx_ap.z):
            scene_r_z_idx += 1
            scene_r_z = scene_r_z_next

        # take slice of data from scene matrix
        scene_rx_slice = self.scene.data[scene_r_z_idx]

        # interpolate from scene rx coordinates to rx aperature axis
        rx_interp_func = scipy.interpolate.interp1d(
            self.scene.x_axis,
            scene_rx_slice,
            kind="linear",
            fill_value=complex(0, 0),
            bounds_error=False,
            assume_sorted=True
        )
        self.scene.rx_ap.aper_profile = rx_interp_func(self.scene.rx_ap.aper_axis)

    def plot_scene(self):
        self.log.info("Plotting scene")

        fig = plt.figure(figsize=(20, 10), layout="constrained")
        rows = 2
        cols = 2
        plt_index = 1
        ax_1 = fig.add_subplot(rows, cols, plt_index)
        plt_index += 1
        ax_1.set_title(f"Scene Amplitude")
        ax_1.set_xlabel("x (m)")
        ax_1.set_ylabel("z (m)")
        ax_1.grid(True)

        #ax_1.plot(x_ext, np.abs(s))
        implot = ax_1.imshow(
            np.abs(self.scene.data),
            extent=[
                self.scene.x_axis.min(),
                self.scene.x_axis.max(),
                self.scene.z_axis.min(),
                self.scene.z_axis.max()
            ],
            cmap="inferno",
            #vmin=0.0s,
            #vmax=0.4,
            aspect="auto",
            origin="lower"
        )
        fig.colorbar(implot, orientation="vertical", label="EMW (V/m)")

        rx_z_array = self.scene.rx_ap.z * np.ones(shape=self.scene.rx_ap.aper_axis.shape)
        tx_z_array = self.scene.tx_ap.z * np.ones(shape=self.scene.tx_ap.aper_axis.shape)
        ax_1.scatter(
            self.scene.rx_ap.aper_axis,
            rx_z_array,
            10,
            "r",
            label=f"Rx measurement locations"
        )
        ax_1.scatter(
            self.scene.tx_ap.aper_axis,
            tx_z_array,
            10,
            "r",
            label="TX aperature"
        )

        ax_2 = fig.add_subplot(rows, cols, plt_index)
        plt_index += 1
        ax_2.set_title("TX Aperature Phase")
        ax_2.set_xlabel("x (m)")
        ax_2.set_ylabel("Phase [rad]")
        ax_2.set_xlim(self.scene.x_min, self.scene.x_max)
        #rx_ax.set_ylim(0, )
        ax_2.grid(True)

        tx_interp_func = scipy.interpolate.interp1d(
            self.scene.tx_ap.aper_axis,
            self.scene.tx_ap.aper_profile,
            kind="linear",
            fill_value=complex(0, 0),
            bounds_error=False,
            assume_sorted=True
        )
        tx_interp = tx_interp_func(self.scene.x_axis)
        ax_2.plot(
            self.scene.x_axis,
            np.unwrap(np.angle(tx_interp))
        )

        ax_3 = fig.add_subplot(rows, cols, plt_index)
        plt_index += 1
        ax_3.set_title("RX Aperature Amplitude")
        ax_3.set_xlabel("x (m)")
        ax_3.set_ylabel("Amplitude EMF (V/m)")
        ax_3.set_xlim(self.scene.rx_ap.x_min, self.scene.rx_ap.x_max)
        ax_3.grid(True)

        ax_3.plot(
            self.scene.rx_ap.aper_axis,
            np.abs(self.scene.rx_ap.aper_profile)
        )

        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Program to simulate phase retrieval using a modified gerchberg-saxton algorithm"
    )
    parser.add_argument(
        "--debug",
        help="Enables debug logs",
        action="store_true",
        default=False
    )
    parser.add_argument(
        "--freq", "-f",
        help="Frequency in Hz",
        type=float,
        default=150e9
    )
    args = parser.parse_args()

    fmt = "%(levelname)s: %(message)s"
    level = "INFO"
    if bool(args.debug): level = "DEBUG"
    coloredlogs.install(level=level, fmt=fmt)

    mgs = MGS(args.freq)
    mgs.run_sim()
    mgs.plot_scene()