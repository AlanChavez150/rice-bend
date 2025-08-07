import logging
import argparse
import copy

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
            dx=0.25e-3
        )
        tx.make_steer(self.freq, theta_deg=10)
        #tx.make_airy(self.freq, 3.7e-3, 0.1)

        self.gs_tx = SimAperature(
            x_min=tx.x_min,
            x_max=tx.x_max,
            z=tx.z,
            dx=tx.dx
        )

        self.scene = SimScene(
            x_min=x_min,
            x_max=x_max,
            z_min=z_min,
            z_max=z_max,
            spacing=0.25e-3,
            rx_ap=rx,
            tx_ap=tx
        )

        # reconstructed scene created be re-illumitating the TX aperature computed by gs
        self.gs_rec_scene = copy.deepcopy(self.scene)

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

    def run_sim(self, gs_rec: bool = False):
        """
        Simulates RF energy in entire scene for pretty plots.
        Also measures data at RX antennes for later use.
        """
        if gs_rec:
            self.log.info("Running scene simulation with MGS reconstructed aperature")
        else:
            self.log.info("Running scene simulation with real aperature")

        tx_ap = None
        if not gs_rec:
            tx_ap = self.scene.tx_ap
        else:
            tx_ap = self.gs_tx

        # redefine tx aperature coordinates and interp data.
        #assert self.scene.spacing < self.scene.tx_ap.dx
        assert np.isclose(tx_ap.z, 0, 0.001) # Currently only works if tx is at z=0.0
        tx_profile_interp_func = scipy.interpolate.interp1d(
            tx_ap.aper_axis,
            tx_ap.aper_profile,
            kind="linear",
            fill_value=complex(0, 0),
            bounds_error=False,
            assume_sorted=True
        )
        tx_profile_interp = tx_profile_interp_func(self.scene.x_axis)

        self.log.info("Computing wave propogation across scene")
        data = rs.rs(self.scene.x_axis, self.scene.z_axis, tx_profile_interp, self.wavelength)
        if gs_rec:
            self.gs_rec_scene.data = data
        else:
            self.scene.data = data

        if gs_rec:
            # dont fill receiver data if doing re-construction
            return
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

    def run_gerch_sax(self):
        self.log.info(f"Running modified Gerchberg-saxton algorithm")
        """
        self.run_sim must have already been called otherwise this wont work.
        """
        # for simplicity, the tx profile is computed with the same resolution as the rx profile,
        # and then interpolated into the scene's resolution later
        size = len(self.scene.rx_ap.aper_axis)
        # rayleigh-sommerfeld code only takes an array for Z values. however only 1 value is needed
        z_axis = np.array([self.scene.rx_ap.z])

        # assume incident wave has a magnitude of 1
        max_iters = 1000
        cvrg_count = 10
        best_error = np.inf
        best_profile = None
        best_gs_idx = 0
        gs_attempts = 100
        for gs_idx in range(gs_attempts):
            ms_am_s = np.ones(size, dtype=np.cfloat) # what the heck do these names mean again?
            ms_am_f = self.scene.rx_ap.aper_profile.copy()
            x_axis  = self.scene.rx_ap.aper_axis.copy()

            curr_phase_s = 2 * np.pi * np.random.rand(size)
            curr_s = np.abs(ms_am_s) * np.exp(curr_phase_s * 1j)

            hist_error = np.zeros(max_iters, np.float32)

            curr_error = 0
            for iter_idx in range(max_iters):

                curr_f = rs.rs(x_axis, z_axis, curr_s, self.wavelength)[0]
                curr_phase_f = np.angle(curr_f)
                curr_f = np.abs(ms_am_f) * np.exp(1j * curr_phase_f)

                curr_s = rs.rs(x_axis, -1.0 * z_axis, curr_f, self.wavelength)[0]
                curr_phase_s = np.angle(curr_s)
                curr_s = np.abs(ms_am_s) * np.exp(1j * curr_phase_s)

                curr_error = np.linalg.norm(np.abs(np.abs(curr_f) - np.abs(ms_am_f)))
                hist_error[iter_idx] = curr_error

                # if the last x iterations did not improve the error, GS has "converged"
                if iter_idx > cvrg_count:
                    error_diff = np.diff(hist_error)
                    recent_err = error_diff[iter_idx-cvrg_count: iter_idx]
                    if np.all(recent_err < 1e-3):
                        self.log.info(f"{gs_idx}: GS has converged after {iter_idx+1} iterations")
                        break

                if iter_idx == max_iters-1:
                    self.log.error(f"{gs_idx}: GS did not converge after {iter_idx+1} iterations")

            curr_profile = curr_s
            if curr_error < best_error:
                best_error = curr_error
                best_profile = curr_profile
                best_gs_idx = gs_idx

        self.log.info(f"After {gs_attempts}: found best result on attempt {best_gs_idx+1}")

        # interpolate from rx axis to gs axis
        gs_interp_func = scipy.interpolate.interp1d(
            x_axis,
            best_profile,
            kind="linear",
            fill_value=complex(0, 0),
            bounds_error=False,
            assume_sorted=True
        )
        gs_interp = gs_interp_func(self.gs_tx.aper_axis)
        self.gs_tx.aper_profile = gs_interp

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

        ax_gs = fig.add_subplot(rows, cols, plt_index)
        plt_index += 1
        ax_gs.set_title(f"MGS reconstruction Scene Amplitude")
        ax_gs.set_xlabel("x (m)")
        ax_gs.set_ylabel("z (m)")
        ax_gs.grid(True)

        implot = ax_gs.imshow(
            np.abs(self.gs_rec_scene.data),
            extent=[
                self.gs_rec_scene.x_axis.min(),
                self.gs_rec_scene.x_axis.max(),
                self.gs_rec_scene.z_axis.min(),
                self.gs_rec_scene.z_axis.max()
            ],
            cmap="inferno",
            #vmin=0.0s,
            #vmax=0.4,
            aspect="auto",
            origin="lower"
        )
        fig.colorbar(implot, orientation="vertical", label="EMW (V/m)")

        ax_2 = fig.add_subplot(rows, cols, plt_index)
        plt_index += 1
        ax_2.set_title("TX Aperature Phase")
        ax_2.set_xlabel("x (m)")
        ax_2.set_ylabel("Phase [rad]")
        ax_2.set_xlim(self.scene.x_min, self.scene.x_max)
        #rx_ax.set_ylim(0, )
        ax_2.grid(True)

        # interp from tx axis to scene x_axis
        tx_interp_func = scipy.interpolate.interp1d(
            self.scene.tx_ap.aper_axis,
            self.scene.tx_ap.aper_profile,
            kind="linear",
            fill_value=complex(0, 0),
            bounds_error=False,
            assume_sorted=True
        )
        tx_interp = tx_interp_func(self.scene.x_axis)

        # interp from gs axis to scene x_axis
        gs_interp_func = scipy.interpolate.interp1d(
            self.gs_tx.aper_axis,
            self.gs_tx.aper_profile,
            kind="linear",
            fill_value=complex(0, 0),
            bounds_error=False,
            assume_sorted=True
        )
        gs_interp = gs_interp_func(self.scene.x_axis)
        ax_2.plot(
            self.scene.tx_ap.aper_axis,
            np.unwrap(np.angle(self.scene.tx_ap.aper_profile)),
            label="Real TX"
        )
        ax_2.plot(
            self.gs_tx.aper_axis,
            np.unwrap(np.angle(self.gs_tx.aper_profile)),
            label="MGS TX"
        )

        ax_3 = fig.add_subplot(rows, cols, plt_index)
        plt_index += 1
        ax_3.set_title("TX Aperature")
        ax_3.set_xlabel("x (m)")
        ax_3.set_ylabel("Amplitude EMF (V/m)")
        ax_2.set_xlim(self.scene.x_min, self.scene.x_max)
        ax_3.grid(True)

        ax_3.plot(
            self.scene.x_axis,
            np.abs(tx_interp),
            label="Real TX"
        )
        ax_3.plot(
            self.scene.x_axis,
            np.abs(gs_interp),
            label="MGS TX"
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
    mgs.run_sim(False)
    mgs.run_gerch_sax()
    mgs.run_sim(True)
    mgs.plot_scene()