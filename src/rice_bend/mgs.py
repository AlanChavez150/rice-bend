import logging
import argparse
import copy
from collections import namedtuple
from pathlib import Path

import coloredlogs
import numpy as np
import scipy.constants
import scipy.interpolate
import matplotlib.pyplot as plt

from rice_bend import rs
from rice_bend.config import SimConfig, load_config
from rice_bend.data_store import GSHistory, make_run_dir, save_run
from rice_bend.sim_scene import SimAperature, SimScene, parse_oscope_rx_data, parse_oscope_heatmap_data

# Default config shipped in the repo's configs/ folder (repo_root/configs/caustic_config.yml)
DEFAULT_CONFIG = Path(__file__).resolve().parents[2] / "configs" / "caustic_config.yml"


# Modified Gerchberg-Saxton hyperparameters bundled as a plain (picklable) tuple so
# the solver core can be handed to worker processes without an MGS/SimConfig instance.
GSParams = namedtuple(
    "GSParams",
    "max_iters convergence_count lr0 bt_shrink bt_tries convergence_threshold history_stride",
)


def gs_params_from_cfg(gs_cfg) -> "GSParams":
    """Pull the GS hyperparameters out of a GerchbergSaxtonConfig into a GSParams."""
    return GSParams(
        max_iters=gs_cfg.max_iters,
        convergence_count=gs_cfg.convergence_count,
        lr0=gs_cfg.lr0,
        bt_shrink=gs_cfg.bt_shrink,
        bt_tries=gs_cfg.bt_tries,
        convergence_threshold=gs_cfg.convergence_threshold,
        history_stride=gs_cfg.history_stride,
    )


# Result of one solver run. `curr_aper_f` is the reconstructed complex aperture on the
# math (scene) x-axis; `history` is a GSHistory when capture=True, else None.
GSResult = namedtuple(
    "GSResult",
    "curr_aper_f final_loss n_iters_run stop_reason seed loss_full history",
)


def interp_complex_to_axis(x_axis: np.ndarray, aper_f: np.ndarray,
                           target_axis: np.ndarray) -> np.ndarray:
    """Interpolate a complex aperture from `x_axis` onto `target_axis`, amplitude and
    phase separately (matches the project's interpolation convention)."""
    amp = scipy.interpolate.interp1d(
        x_axis, np.abs(aper_f), kind="linear", fill_value=0,
        bounds_error=False, assume_sorted=True)(target_axis)
    phs = scipy.interpolate.interp1d(
        x_axis, np.angle(aper_f), kind="linear", fill_value=0,
        bounds_error=False, assume_sorted=True)(target_axis)
    return amp * np.exp(1j * phs)


def gs_reconstruct(tx_z: float, orig_aper_amp: np.ndarray, x_axis: np.ndarray,
                   rx_z: float, rx_field: np.ndarray, error_weighting: np.ndarray,
                   wavelength: float, params: "GSParams", seed,
                   capture: bool = False, log=None) -> "GSResult":
    """Modified Gerchberg-Saxton solver core (no MGS/SimScene instance required).

    Solves for the aperture phase at plane `tx_z` that best reproduces the measured RX
    field `rx_field`, holding the amplitude fixed at `orig_aper_amp` (defined on
    `x_axis`; its nonzero region is the support). Pure given its arguments — every input
    is a plain array/scalar, so this runs unchanged in a worker process.

    With `capture=True` a GSHistory is built and per-iteration state recorded (the
    single-shot `mgs` path, used by animation); with `capture=False` only the dense loss
    curve + final aperture are produced (the grid-search path). The math is identical
    either way, so results do not depend on `capture`. `log` (optional) receives the
    per-iteration progress lines; workers pass None to stay quiet.
    """
    size = len(x_axis)
    rx_plane = np.array([rx_z])   # measurement (RX) plane target
    tx_plane = np.array([tx_z])   # aperture (TX) plane target, used by back-prop

    max_iters = params.max_iters
    cvrg_count = params.convergence_count
    lr0 = params.lr0
    bt_shrink = params.bt_shrink
    bt_tries = params.bt_tries
    conv_threshold = params.convergence_threshold

    # seeded initial phase for reproducibility; draw + record a seed if none given
    if seed is None:
        seed = int(np.random.SeedSequence().entropy % (2**32))
    rng = np.random.default_rng(seed)

    # track aperature amplitude and phase seperatly
    curr_aper_amp = np.abs(orig_aper_amp.copy())
    curr_aper_phase = 2 * np.pi * rng.random(size)
    initial_phase = curr_aper_phase.copy()

    support = np.abs(orig_aper_amp) > 0  # aperature mask

    history = None
    if capture:
        history = GSHistory(
            x_axis=x_axis,
            orig_prop_f=rx_field,
            orig_aper_amp=orig_aper_amp,
            error_weighting=error_weighting,
            support=support,
            initial_phase=initial_phase,
            rx_z=float(rx_z),
            seed=seed,
            history_stride=params.history_stride,
        )

    stop_reason = "max_iters"
    n_iters_run = max_iters
    hist_error = np.zeros(max_iters, np.float32)
    for iter_idx in range(max_iters):
        # propogate aperature guess to measurement plane
        u0 = curr_aper_amp * np.exp(1j * curr_aper_phase)
        curr_prop_f = rs.rs(x_axis, rx_plane, u0, wavelength, z_src=tx_z, forward_dir=-1.0)[0]
        r_cx = error_weighting * (curr_prop_f - rx_field)

        loss = 0.5 * np.mean(np.abs(r_cx)**2)
        hist_error[iter_idx] = loss
        if history is not None:
            history.record_iter(iter_idx, loss, curr_aper_phase, curr_prop_f)
        g_meas = r_cx

        # back propogate the residual from the RX plane to the TX (aperture) plane
        g_u0 = rs.rs(x_axis, tx_plane, g_meas, wavelength, z_src=rx_z, forward_dir=-1.0)[0]

        grad_theta = 2.0 * np.imag(g_u0 * np.conj(u0))
        grad_theta[~support] = 0.0

        step = lr0
        for _ in range(bt_tries):
            theta_trial = curr_aper_phase - step * grad_theta
            u0_trial = curr_aper_amp * np.exp(1j * theta_trial)
            um_trial = rs.rs(x_axis, rx_plane, u0_trial, wavelength, z_src=tx_z, forward_dir=-1.0)[0]
            r_trial = error_weighting * (um_trial - rx_field)
            loss_trial = 0.5 * np.mean(np.abs(r_trial)**2)
            if loss_trial < loss:  # sufficient decrease
                curr_aper_phase = theta_trial.copy()
                curr_aper_phase[~support] = 0.0
                loss = loss_trial
                break
            step *= bt_shrink
        curr_aper_f = curr_aper_amp * np.exp(1j * curr_aper_phase)

        if log is not None and iter_idx % 1000 == 0:
            log.info(f"{iter_idx} loss {loss}")

        # if the last x iterations did not improve the error, GS has "converged"
        if iter_idx > cvrg_count:
            recent_err = hist_error[iter_idx-cvrg_count: iter_idx]
            recent_err_flatness = np.mean(np.diff(recent_err))
            if recent_err_flatness > conv_threshold:
                if log is not None:
                    log.info(f"MGS has converged after {iter_idx+1} iterations")
                stop_reason = "converged"
                n_iters_run = iter_idx + 1
                break

        if iter_idx == max_iters-1:
            if log is not None:
                log.error(f"MGS did not converge after {iter_idx+1} iterations")
            stop_reason = "max_iters"
            n_iters_run = max_iters

    if history is not None:
        history.capture_final(iter_idx, loss, curr_aper_phase, curr_prop_f)
        history.finalize(stop_reason, n_iters_run, loss)

    loss_full = hist_error[:n_iters_run].copy()
    return GSResult(curr_aper_f=curr_aper_f, final_loss=float(loss), n_iters_run=n_iters_run,
                    stop_reason=stop_reason, seed=seed, loss_full=loss_full, history=history)


class MGS():
    def __init__(self, freq: float, config: SimConfig):
        self.log = logging.getLogger()
        # set up simulation scene from config
        scene_cfg = config.sim_scene
        x_min = scene_cfg.x_min
        x_max = scene_cfg.x_max
        z_min = scene_cfg.z_min
        z_max = scene_cfg.z_max
        spacing = scene_cfg.spacing
        self.plot_path = config.plot_path
        self.gs_cfg = config.gerchberg_saxton
        self.output_cfg = config.output
        self.gs_history = None
        self._rx_field = None          # measured RX field on scene.x_axis (set by measure())
        self._error_weighting = None   # phase-retrieval error weighting (set by measure())
        self.freq = freq
        self.wavelength = scipy.constants.c / freq
        rx_spacing_ratio = 1 / 20
        rx_spacing = self.wavelength * rx_spacing_ratio

        # Convention: the RX aperture sits at the origin (z=0), the bottom of the
        # image. The TX aperture sits at the top of the scene (z=z_max) and projects
        # toward -Z, so its beam travels *down* the full scene height to the RX.
        # Move the TX around by changing tx.z (height) and its x_min/x_max (lateral).
        # RX geometry comes from config (rx_aperture): a window of `width` centered
        # at `x_center`. dx defaults to wavelength/20 (rx_spacing) when left null.
        rx_cfg = config.rx_aperture
        rx_dx = rx_cfg.dx if rx_cfg.dx is not None else rx_spacing
        rx = SimAperature(
            x_min=rx_cfg.x_center - rx_cfg.width / 2.0,
            x_max=rx_cfg.x_center + rx_cfg.width / 2.0,
            z=0,
            dx=rx_dx
        )

        # TX aperture location/sampling/trajectory come from config (tx_aperture),
        # defined independently of the scene grid.
        tx_cfg = config.tx_aperture
        tx = SimAperature(
            x_min=tx_cfg.x_min,
            x_max=tx_cfg.x_max,
            z=tx_cfg.z,
            dx=tx_cfg.dx
        )
        assert tx.z > z_min, f"tx_aperture.z ({tx.z}) must be above the scene floor z_min ({z_min})"
        # the caustic is parameterised by *distance from the aperture* along the
        # beam, so feed it the downstream propagation length (TX plane down to the RX)
        prop_length = tx.z - z_min
        # The emitted beam is selected by tx_aperture.beam.type. `has_real_aper`
        # marks that a known real aperture exists (true for both simulated beams;
        # false for ExpMGS, where the TX aperture is unknown).
        beam = tx_cfg.beam
        self.beam_type = beam.type
        self.has_real_aper = True
        if beam.type == "caustic":
            # caustic beam x(d) = a*d^2 + b*d + c (d = distance from TX)
            a, b, c = beam.trajectory
            tx.make_caustic(self.freq, prop_length, a, b, c)
        elif beam.type == "directional":
            # steered plane wave at the configured angle
            tx.make_steer(self.freq, theta_deg=beam.steer_angle_deg)
        else:
            raise ValueError(f"unknown beam type: {beam.type}")

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
            spacing=spacing,
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
        rx_wl_ratio = self.scene.rx_ap.dx / self.wavelength
        rx_width = self.scene.rx_ap.x_max - self.scene.rx_ap.x_min
        self.log.info(f" - X axis: {self.scene.rx_ap.x_min:0.3f} {self.scene.rx_ap.x_max:0.3f} (width {rx_width:0.3f} m, dx {rx_wl_ratio:0.3f} wavelength)")
        self.log.info(f" - Z: {self.scene.rx_ap.z:0.3f}")
        self.log.info(f"TX aperature ({self.beam_type} beam)")
        tx_wl_ratio = self.scene.tx_ap.dx / self.wavelength
        self.log.info(f" - X axis: {self.scene.tx_ap.x_min:0.3f} {self.scene.tx_ap.x_max:0.3f} (dx {tx_wl_ratio:0.3f} wavelength)")
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
        data = rs.rs(self.scene.x_axis, self.scene.z_axis, tx_profile_interp, self.wavelength,
                     z_src=tx_ap.z, forward_dir=-1.0)
        # the aperture only radiates into the -Z half-space; zero the field behind it
        # (planes above the TX plane would otherwise show a back-propagated artifact)
        behind_tx = self.scene.z_axis > tx_ap.z
        data[behind_tx, :] = 0
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

    def measure(self) -> None:
        """Sample the measured RX field onto the scene grid and build the error
        weighting used by phase retrieval. Idempotent.

        For the pure-simulation path, run_sim(False) must have already filled
        scene.rx_ap.aper_profile; for ExpMGS it is filled from experimental data
        in __init__. The result (self._rx_field / self._error_weighting) is the
        single measurement shared across every hypothesized TX location.
        """
        x_axis = self.scene.x_axis
        orig_prop_f = self.scene.rx_ap.interp_axis(x_axis)
        # error computations are weighted to favor higher amplitude data, and ignore things outside the recieve aperature
        error_weighting = (np.abs(orig_prop_f) / np.abs(orig_prop_f).max()) + 0.25
        for idx, x_val in enumerate(x_axis):
            if x_val < self.scene.rx_ap.x_min or x_val > self.scene.rx_ap.x_max:
                error_weighting[idx] = 0.0
        self._rx_field = orig_prop_f
        self._error_weighting = error_weighting

    def run_gerch_sax(self):
        """Reconstruct the aperture phase at the *real* TX plane (single-shot path).

        Thin wrapper over measure() + reconstruct_at() that preserves the original
        behaviour: solve at scene.tx_ap.z using the real aperture's amplitude as the
        fixed support, writing the result into self.gs_tx.
        """
        self.log.info(f"Running modified Gerchberg-saxton algorithm")
        self.measure()
        orig_aper_amp = np.abs(self.scene.tx_ap.interp_axis(self.scene.x_axis))
        self.gs_history = self.reconstruct_at(
            tx_z=self.scene.tx_ap.z,
            orig_aper_amp=orig_aper_amp,
            out_aper=self.gs_tx,
        )

    def reconstruct_at(self, tx_z: float, orig_aper_amp: np.ndarray,
                       out_aper: SimAperature) -> GSHistory:
        """Run modified Gerchberg-Saxton for a hypothesized TX plane.

        Solves for the aperture phase at plane `tx_z` that best reproduces the
        measured RX field (self._rx_field), holding the amplitude fixed at
        `orig_aper_amp` (defined on scene.x_axis; its nonzero region is the
        support). Writes the reconstructed complex aperture into
        out_aper.aper_profile and returns the run's GSHistory. measure() must
        have been called first.
        """
        assert self._rx_field is not None and self._error_weighting is not None, \
            "measure() must be called before reconstruct_at()"
        # Run the solver core (capture=True so the single-shot path keeps its full
        # GSHistory for animation/persistence), then interp the result onto out_aper.
        result = gs_reconstruct(
            tx_z=tx_z,
            orig_aper_amp=orig_aper_amp,
            x_axis=self.scene.x_axis.copy(),
            rx_z=self.scene.rx_ap.z,
            rx_field=self._rx_field,
            error_weighting=self._error_weighting,
            wavelength=self.wavelength,
            params=gs_params_from_cfg(self.gs_cfg),
            seed=self.gs_cfg.seed,
            capture=True,
            log=self.log,
        )
        out_aper.aper_profile = interp_complex_to_axis(
            self.scene.x_axis, result.curr_aper_f, out_aper.aper_axis)
        return result.history

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

    def plot_scene(self, save_path=None, show=True):
        """Save the 4-panel scene plot (real vs MGS-reconstructed scene + TX aperture
        phase/amplitude). Writes to `save_path` if given, else self.plot_path; only
        opens an interactive window when `show` is True (set False for headless/batch)."""
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
        if self.has_real_aper:
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
        ax_3.set_xlim(self.scene.x_min, self.scene.x_max)
        ax_3.grid(True)

        if self.has_real_aper:
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
        out_path = save_path if save_path is not None else self.plot_path
        self.log.info(f"Saving scene plot to {out_path}")
        fig.savefig(out_path)
        if show:
            plt.show()
        plt.close(fig)

class ExpMGS(MGS):
    def __init__(self, rx_path: Path, heatmap_path: Path, freq: float, config: SimConfig):
        self.log = logging.getLogger("ExpMGS")
        self.log.info(f"Carrier Frequency: {freq*1e-9: 0.2f} GHz")
        self.freq = freq
        # ExpMGS derives its scene from measured data, so only plot_path is used
        self.plot_path = config.plot_path
        self.gs_cfg = config.gerchberg_saxton
        self.output_cfg = config.output
        self.gs_history = None
        self._rx_field = None          # measured RX field on scene.x_axis (set by measure())
        self._error_weighting = None   # phase-retrieval error weighting (set by measure())

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
        self.has_real_aper = False  # experimental TX aperture is unknown -> no "Real TX" overlay
        self.beam_type = "directional"  # ExpMGS seeds with a steered guess


def main():
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
    parser.add_argument(
        "--config",
        type=Path,
        help="Path to a simulation config .yml (sim_scene bounds and plot_path)",
        default=DEFAULT_CONFIG
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Override output.output_dir from config (base dir for run folders)",
        default=None
    )
    parser.add_argument(
        "--run-name",
        type=str,
        help="Override output.run_name (suffix appended to the run directory name)",
        default=None
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="Override gerchberg_saxton.seed for a reproducible initial phase",
        default=None
    )
    parser.add_argument(
        "--no-save",
        help="Disable run persistence for this invocation",
        action="store_true",
        default=False
    )
    args = parser.parse_args()

    fmt = "%(levelname)s: %(message)s"
    level = "INFO"
    if bool(args.debug): level = "DEBUG"
    coloredlogs.install(level=level, fmt=fmt)

    config = load_config(args.config)

    # apply CLI overrides onto the config
    if args.output_dir is not None: config.output.output_dir = args.output_dir
    if args.run_name is not None: config.output.run_name = args.run_name
    if args.seed is not None: config.gerchberg_saxton.seed = args.seed
    if args.no_save: config.output.save_run = False

    run_dir = None
    if config.output.save_run:
        run_dir = make_run_dir(config.output.output_dir, config.output.run_name)

    if args.rx_path is not None and not args.heatmap_path is None:
        rx_path = Path(args.rx_path)
        heatmap_path = Path(args.heatmap_path)
        mgs = ExpMGS(rx_path, heatmap_path, args.freq, config)
        #mgs.run_sim(False, False)
        mgs.run_gerch_sax()
        mgs.run_sim(True, False)
        mgs.plot_scene()
        if run_dir is not None:
            save_run(mgs, run_dir, config, args.config, args.freq, vars(args), is_exp=True)

    else:
        mgs = MGS(args.freq, config)
        mgs.run_sim(False)
        mgs.run_gerch_sax()
        mgs.run_sim(True)
        mgs.plot_scene()
        if run_dir is not None:
            save_run(mgs, run_dir, config, args.config, args.freq, vars(args), is_exp=False)

if __name__ == "__main__":
    main()

