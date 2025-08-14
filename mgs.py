import logging
import argparse
import copy
import time
from pathlib import Path

import coloredlogs
import numpy as np
import scipy.constants
import scipy.interpolate
import matplotlib.pyplot as plt

import rs
from sim_scene import SimAperature, SimScene, parse_oscope_rx_data, parse_oscope_heatmap_data

class MGS():
    def __init__(self, freq: float):
        self.log = logging.getLogger()
        # set up simulation scene
        x_min = -0.2
        x_max =  0.2
        z_min = 0.0
        z_max = 0.5
        self.freq = freq
        self.wavelength = scipy.constants.c / freq
        rx_spacing_ratio = 1 / 20
        rx_spacing = self.wavelength * rx_spacing_ratio

        rx = SimAperature(
            x_min=x_min+0.1,
            x_max=x_max-0.1,
            z=0.25,
            dx=rx_spacing
        )

        tx = SimAperature(
            x_min=x_min + 0.15,
            #x_max=(x_max - x_min) * 0.75 + x_min,
            x_max=x_max - 0.15,
            z=0,
            dx=0.25e-3
        )
        #tx.make_steer(self.freq, theta_deg=10)
        #tx.make_airy(self.freq, 3.7e-3, 0.1)
        self.real_traj = [0.3, 0.01, 0.025]
        tx.make_caustic(self.freq, z_max, self.real_traj[0], self.real_traj[1], self.real_traj[2])

        self.gs_tx = SimAperature(
            x_min=tx.x_min,
            x_max=tx.x_max,
            z=tx.z,
            dx=tx.dx
        )
        self.rec_traj = []

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

    def run_sim(self, gs_rec: bool = False, measure_rx: bool = True):
        """
        Simulates RF energy in entire scene for pretty plots.
        Also optionally measures data at RX antennes for later use.
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

        if gs_rec or not measure_rx:
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
        # Create a new, high resolution axis to do all math in, then interp back
        size = len(self.scene.x_axis)
        x_axis  = self.scene.x_axis.copy()
        # rayleigh-sommerfeld code only takes an array for Z values. however only 1 value is needed
        z_axis = np.array([self.scene.rx_ap.z])

        max_iters = int(10e3)
        cvrg_count = 10
        # assume incident wave has a magnitude of 1
        orig_aper_amp = np.abs(self.scene.tx_ap.interp_axis(x_axis))
        orig_prop_f  = self.scene.rx_ap.interp_axis(x_axis)
        # error computations are weighted to favor higher amplitude data, and ignore things outside the recieve aperature
        error_weighting = (np.abs(orig_prop_f) / np.abs(orig_prop_f).max()) + 0.25
        for idx, x_val in enumerate(x_axis):
            if x_val < self.scene.rx_ap.x_min or x_val > self.scene.rx_ap.x_max:
                error_weighting[idx] = 0.0

        # track aperature amplitude and phase seperatly
        curr_aper_amp = np.abs(orig_aper_amp.copy())
        curr_aper_phase = 2 * np.pi * np.random.rand(size)

        support = np.abs(orig_aper_amp) > 0 # aperature mask

        lr0 = 0.05         # starting step size (tune: 0.01–0.1)
        bt_shrink = 0.5    # backtracking factor
        bt_tries = 8       # max reductions per iter


        hist_error = np.zeros(max_iters, np.float32)
        for iter_idx in range(max_iters):
            # propogate aperature guess to measurement plane
            u0 = curr_aper_amp * np.exp(1j * curr_aper_phase)
            #self.plot_mgs_helper(x_axis, orig_prop_f, u0, f"{iter_idx}")
            curr_prop_f = rs.rs(x_axis, z_axis, u0, self.wavelength)[0]
            r_cx = error_weighting * (curr_prop_f - orig_prop_f)

            loss = 0.5 * np.mean(np.abs(r_cx)**2)
            hist_error[iter_idx] = loss
            g_meas = r_cx

            # back propogate
            g_u0 = rs.rs(x_axis, -1.0 * z_axis, g_meas, self.wavelength)[0]

            grad_theta = 2.0 * np.imag(g_u0 * np.conj(u0))
            grad_theta[~support] = 0.0

            step = lr0
            for _ in range(bt_tries):
                theta_trial = curr_aper_phase - step * grad_theta
                u0_trial = curr_aper_amp * np.exp(1j * theta_trial)
                um_trial = rs.rs(x_axis, z_axis, u0_trial, self.wavelength)[0]
                r_trial = error_weighting * (um_trial - orig_prop_f)
                loss_trial = 0.5 * np.mean(np.abs(r_trial)**2)
                if loss_trial < loss:  # sufficient decrease
                    curr_aper_phase = theta_trial.copy()
                    curr_aper_phase[~support] = 0.0
                    loss = loss_trial
                    break
                step *= bt_shrink
            curr_aper_f = curr_aper_amp * np.exp(1j * curr_aper_phase)

            if iter_idx % 1000 == 0:
                self.log.info(f"{iter_idx} loss {loss}")

            # if the last x iterations did not improve the error, GS has "converged"
            if iter_idx > cvrg_count:
                recent_err = hist_error[iter_idx-cvrg_count: iter_idx]
                recent_err_flatness = np.mean(np.diff(recent_err))
                if recent_err_flatness > -1e-10:
                    self.log.info(f"MGS has converged after {iter_idx+1} iterations")
                    break

            if iter_idx == max_iters-1:
                self.log.error(f"MGS did not converge after {iter_idx+1} iterations")

        # interpolate from rx axis to gs axis
        gs_interp_func_amp = scipy.interpolate.interp1d(
            x_axis,
            np.abs(curr_aper_f),
            kind="linear",
            fill_value=0,
            bounds_error=False,
            assume_sorted=True
        )
        gs_interp_func_phs = scipy.interpolate.interp1d(
            x_axis,
            np.angle(curr_aper_f),
            kind="linear",
            fill_value=0,
            bounds_error=False,
            assume_sorted=True
        )
        gs_interp_amp = gs_interp_func_amp(self.gs_tx.aper_axis)
        gs_interp_phs = gs_interp_func_phs(self.gs_tx.aper_axis)
        gs_interp = gs_interp_amp * np.exp(1j * gs_interp_phs)
        self.gs_tx.aper_profile = gs_interp

    def plot_mgs_helper(self, x_axis: np.ndarray, real_f: np.ndarray, test_f: np.ndarray, title: str):
        fig = plt.figure(figsize=(20, 10), layout="constrained")
        self.log.info(f"Plotting MGS helper: '{title}'")
        fig.suptitle(title)
        rows = 1
        cols = 2
        plot_index = 1
        ampl_ax = fig.add_subplot(rows, cols, plot_index)
        plot_index += 1
        ampl_ax.set_title(f"Amplitude")
        ampl_ax.set_xlabel("x (m)")
        ampl_ax.set_ylabel("EMF (V/m)")
        ampl_ax.grid(True)
        ampl_ax.plot(
            x_axis,
            np.abs(real_f),
            label="Real F"
        )
        ampl_ax.plot(
            x_axis,
            np.abs(test_f),
            label="Test F"
        )
        ampl_ax.legend()

        phs_ax = fig.add_subplot(rows, cols, plot_index)
        plot_index += 1
        phs_ax.set_title(f"Phase")
        phs_ax.set_xlabel("x (m)")
        phs_ax.set_ylabel("Phase [rad]")
        phs_ax.grid(True)

        phs_ax.plot(
            x_axis,
            np.unwrap(np.angle(real_f)),
            label="Real TX"
        )
        phs_ax.plot(
            x_axis,
            np.unwrap(np.angle(test_f)),
            label="MGS TX"
        )
        phs_ax.legend()
        plt.show()

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

        v_max = np.nanmax([
            np.nanmax(np.abs(self.scene.data)),
            np.nanmax(np.abs(self.gs_rec_scene.data))
        ])

        implot = ax_1.imshow(
            np.abs(self.scene.data),
            extent=[
                self.scene.x_axis.min(),
                self.scene.x_axis.max(),
                self.scene.z_axis.min(),
                self.scene.z_axis.max()
            ],
            cmap="inferno",
            vmin=0.0,
            vmax=v_max,
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
            label=f"Rx aperture location"
        )
        ax_1.scatter(
            self.scene.tx_ap.aper_axis,
            tx_z_array,
            10,
            "b",
            label="TX aperture location"
        )
        ax_1.legend()

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
            #vmin=0.0,
            #vmax=v_max,
            aspect="auto",
            origin="lower"
        )
        fig.colorbar(implot, orientation="vertical", label="EMW (V/m)")

        ax_gs.scatter(
            self.scene.rx_ap.aper_axis,
            rx_z_array,
            10,
            "r",
            label=f"RX aperture location"
        )
        ax_gs.scatter(
            self.scene.tx_ap.aper_axis,
            tx_z_array,
            10,
            "b",
            label="TX aperture location"
        )

        real_a, real_b, real_c = (1e-3, 1e-3, 1e-3)
        if len(self.real_traj) > 0:
            real_a, real_b, real_c = self.real_traj
        rec_a, rec_b, rec_c = self.rec_traj
        traj_x = self.scene.x_axis.copy()
        # recenter x axis
        traj_x += (self.scene.x_max - self.scene.x_min) / 2.0
        traj_z = np.linspace(self.scene.z_min, self.scene.z_max, len(traj_x))
        self.log.error(f"{real_a=} {real_b=} {real_c=}")
        self.log.error(f"{rec_a=} {rec_b=} {rec_c=}")
        real_traj = (-real_b + np.sqrt( real_b**2 - 4*real_a * ( real_c - traj_z))) / (2 * real_a)
        rec_traj  = ( -rec_b + np.sqrt(  rec_b**2 - 4* rec_a * (  rec_c - traj_z))) / (2 *  rec_a)

        for idx in range(len(traj_x)):
            if real_traj[idx] > self.scene.z_max or real_traj[idx] < self.scene.z_min:
                real_traj[idx] = None
            if rec_traj[idx] > self.scene.z_max or rec_traj[idx] < self.scene.z_min:
                rec_traj[idx] = None

        ax_gs.plot(
            traj_x,
            real_traj,
            linewidth=3,
            label="Real Trajectory"
        )
        ax_gs.plot(
            traj_x,
            rec_traj,
            linewidth=3,
            label="Recovered Trajectory"
        )

        ax_gs.legend()

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
        gs_interp_amp_func = scipy.interpolate.interp1d(
            self.gs_tx.aper_axis,
            np.abs(self.gs_tx.aper_profile),
            kind="linear",
            fill_value=0,
            bounds_error=False,
            assume_sorted=True
        )
        gs_interp = gs_interp_amp_func(self.scene.x_axis)
        if len(self.real_traj) > 0:
            ax_2.plot(
                self.scene.tx_ap.aper_axis,
                np.unwrap(np.angle(self.scene.tx_ap.aper_profile)),
                label="Real TX"
            )
        ax_2.plot(
            self.gs_tx.aper_axis,
            np.unwrap(np.angle(self.gs_tx.aper_profile)),
            label="MGS Reconstructed TX"
        )
        ax_2.legend()


        ax_3 = fig.add_subplot(rows, cols, plt_index)
        plt_index += 1
        ax_3.set_title("TX Aperature Amplitude")
        ax_3.set_xlabel("x (m)")
        ax_3.set_ylabel("Amplitude EMF (V/m)")
        ax_2.set_xlim(self.scene.x_min, self.scene.x_max)
        ax_3.grid(True)

        if len(self.real_traj) > 0:
            ax_3.plot(
                self.scene.x_axis,
                np.abs(tx_interp),
                label="Real TX"
            )
        ax_3.plot(
            self.scene.x_axis,
            gs_interp,
            label="MGS Reconstructed TX"
        )
        ax_3.legend()
        fig.savefig("mgs.png")
        plt.show()

    def compute_traj(self):
        self.log.info("Computing trajectory")
        traj_resolution = 100
        search_space = np.linspace(
            start=-0.6,
            stop=0.6,
            num=traj_resolution
        )
        search_space_c = np.linspace(
            start=self.scene.tx_ap.x_min,
            stop=self.scene.tx_ap.x_max,
            num=traj_resolution
        )
        min_a = None
        min_b = None
        min_c = None
        min_mse = np.inf
        min_aper = None
        start_time = time.time()
        total_cnt = len(search_space)**2 * len(search_space_c)
        attempt_cnt = 0
        for search_a in search_space:
            for search_b in search_space:
                for search_c in search_space_c:
                    test_aper = SimAperature(
                        x_min = self.gs_tx.x_min,
                        x_max = self.gs_tx.x_max,
                        z= self.gs_tx.z,
                        dx=self.gs_tx.dx
                    )
                    test_aper.make_caustic(self.freq, self.scene.z_max, search_a, search_b, search_c)
                    #gen_plot = attempt_cnt % 10000 == 0
                    gen_plot = False
                    curr_mse = self.compare_aper(self.gs_tx, test_aper, gen_plot)

                    attempt_cnt += 1
                    if curr_mse < min_mse:
                        min_mse = curr_mse
                        min_a = search_a
                        min_b = search_b
                        min_c = search_c
                        min_aper = test_aper
                        complete_perc = (attempt_cnt+1)/total_cnt
                        complete_perc *= 100
                        self.log.info(f"{complete_perc:02.2f}% a {search_a:03.3f} b {search_b:03.3f} {search_c:03.3f}: MSE {curr_mse:0.3f}")

        dur = time.time() - start_time
        self.log.info(f"Finished after {dur}")
        self.log.info(f"Attempts: {attempt_cnt}")
        self.log.info(f"A {min_a:0.3f}")
        self.log.info(f"B {min_b:0.3f}")
        self.log.info(f"C {min_c:0.3f}")
        self.rec_traj = [min_a, min_b, min_c]

    def compare_aper(self, ap_1: SimAperature, ap_2: SimAperature, plot: bool = False) -> float:
        # make sure both aperatures fit inside the scene then interp them onto the scene.
        # then compares.
        # returns MSE
        assert ap_1.x_min > self.scene.x_min and ap_1.x_max < self.scene.x_max
        assert ap_2.x_min > self.scene.x_min and ap_2.x_max < self.scene.x_max

        x_axis = self.scene.x_axis
        ap_1_scene = ap_1.interp_axis(x_axis, assume_sorted=False)
        ap_2_scene = ap_2.interp_axis(x_axis, assume_sorted=False)
        mse = np.mean(np.abs(ap_1_scene - ap_2_scene)**2)
        if plot:
            self.plot_mgs_helper(x_axis, ap_1_scene, ap_2_scene, f"Aperature comparison: mse {mse}")
        return mse

class ExpMGS(MGS):
    def __init__(self, rx_path: Path, heatmap_path: Path, freq: float):
        self.log = logging.getLogger("ExpMGS")
        self.log.info(f"Carrier Frequency: {freq*1e-9: 0.2f} GHz")
        self.freq = freq

        self.log.info(f"Reading file as RX data: {rx_path}")
        rx = parse_oscope_rx_data(rx_path, freq, 25e9, 6)
        z_adj = 0.35 # measured during experiment setup
        rx.z = z_adj - rx.z


        ap_left_edge = 0.2558
        ap_right_edge = 0.1573

        dx = rx.dx
        tx = SimAperature(
            x_min=0.3 - ap_left_edge,
            x_max=0.3 - ap_right_edge,
            z=0,
            dx=dx
        )

        # recenter everything such that tx data is centered around 0.0
        x_offset = (tx.x_max - tx.x_min) / 2.0 + tx.x_min
        x_offset *= -1.0

        rx_recentered = SimAperature(
            x_min=rx.x_min + x_offset,
            x_max=rx.x_max + x_offset,
            z=rx.z,
            dx=dx
        )
        rx_recentered.aper_profile = rx.aper_profile[0: len(rx_recentered.aper_profile)]
        rx = rx_recentered
        # rescale data
        rx.aper_profile = 6.0 * (rx.aper_profile / np.max(rx.aper_profile))

        tx_recentered = SimAperature(
            x_min=tx.x_min + x_offset,
            x_max=tx.x_max + x_offset,
            z=tx.z,
            dx=dx
        )
        tx = tx_recentered
        tx.make_steer(self.freq, theta_deg=0)

        x_min = tx.x_min - 0.2
        x_max = tx.x_max + 0.2
        z_min = 0.0
        z_max = 0.4
        self.wavelength = scipy.constants.c / freq

        spacing_ratio = round(rx.dx / self.wavelength, 3)
        self.log.info(f"RX measurements are spread {spacing_ratio:0.3f} wavelengths apart")

        self.gs_tx = SimAperature(
            x_min=tx.x_min,
            x_max=tx.x_max,
            z=tx.z,
            dx=dx
        )

        self.log.info(f"Reading file with heatmap data: {heatmap_path}")
        base_scene = SimScene(
            x_min=x_min,
            x_max=x_max,
            z_min=z_min,
            z_max=z_max,
            spacing=dx,
            rx_ap=rx,
            tx_ap=tx
        )
        heatmap_scene = parse_oscope_heatmap_data(heatmap_path, base_scene, x_offset, self.freq)
        self.scene = heatmap_scene

        # reconstructed scene created be re-illumitating the TX aperature computed by gs
        self.gs_rec_scene = copy.deepcopy(self.scene)

        # Log parameters
        self.log.info(f"Simulation scene:")
        self.log.info(f" - X axis: {self.scene.x_min:0.3f} - {self.scene.x_max:0.3f}")
        self.log.info(f" - Z axis: {self.scene.z_min:0.3f} - {self.scene.z_max:0.3f}")
        self.log.info(f"RX aperature:")
        self.log.info(f" - X axis: {self.scene.rx_ap.x_min:0.3f} {self.scene.rx_ap.x_max:0.3f}")
        self.log.info(f" - Z: {self.scene.rx_ap.z:0.3f}")
        self.log.info(f"TX aperature")
        self.log.info(f" - X axis: {self.scene.tx_ap.x_min:0.3f} {self.scene.tx_ap.x_max:0.3f}")
        self.log.info(f" - Z: {self.scene.tx_ap.z:0.3f}")
        self.real_traj = [] # empty list means unknown trajectory
        #self.rec_traj  = None


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
    parser.add_argument(
        "--rx-path",
        type=Path,
        help="Path to .mat with experimentation data. Only a slice of z axis measurements",
        default=None
    )
    parser.add_argument(
        "--heatmap-path",
        type=Path,
        help="Path to .mat experimentation data. Expects full heatmap measurements",
        default=None
    )
    args = parser.parse_args()

    fmt = "%(levelname)s: %(message)s"
    level = "INFO"
    if bool(args.debug): level = "DEBUG"
    coloredlogs.install(level=level, fmt=fmt)

    rx_path = Path(args.rx_path)
    heatmap_path = Path(args.heatmap_path)

    if args.rx_path is not None and not args.heatmap_path is None:
        mgs = ExpMGS(rx_path, heatmap_path, args.freq)
        #mgs.run_sim(False, False)
        mgs.run_gerch_sax()
        mgs.run_sim(True, False)
        mgs.compute_traj()
        mgs.plot_scene()

    else:
        mgs = MGS(args.freq)
        mgs.run_sim(False)
        mgs.run_gerch_sax()
        mgs.run_sim(True)
        mgs.compute_traj()
        mgs.plot_scene()
