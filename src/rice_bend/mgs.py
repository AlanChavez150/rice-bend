import logging
import argparse
from collections import namedtuple
from pathlib import Path

import coloredlogs
import numpy as np
import scipy.interpolate
import matplotlib.pyplot as plt

from rice_bend import rs
from rice_bend.config import DEFAULT_CONFIG, GerchbergSaxtonConfig, SimConfig, load_config
from rice_bend.interp import interp_amp_phase, interp_real_imag
from rice_bend.plotting import draw_line_panel, draw_scene
from rice_bend.data_store import GSHistory, check_run_dir, make_run_dir, save_run
from rice_bend.sim_scene import SimAperature, SimScene, parse_oscope_rx_data, parse_oscope_heatmap_data


# Result of one solver run. `curr_aper_f` is the reconstructed complex aperture on the
# math (scene) x-axis; `history` is a GSHistory when capture=True, else None.
GSResult = namedtuple(
    "GSResult",
    "curr_aper_f final_loss n_iters_run stop_reason seed loss_full history",
)


def gs_reconstruct(tx_z: float, orig_aper_amp: np.ndarray, x_axis: np.ndarray,
                   rx_z: float, rx_field: np.ndarray, error_weighting: np.ndarray,
                   wavelength: float, params: GerchbergSaxtonConfig,
                   capture: bool = False, log=None) -> "GSResult":
    """Modified Gerchberg-Saxton solver core (no MGS/SimScene instance required).

    Solves for the aperture phase at plane `tx_z` that best reproduces the measured RX
    field `rx_field`, holding the amplitude fixed at `orig_aper_amp` (defined on
    `x_axis`; its nonzero region is the support). Pure given its arguments — arrays,
    scalars and a picklable config model — so this runs unchanged in a worker process.

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
    seed = params.seed
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

    if stop_reason == "max_iters" and log is not None:
        log.error(f"MGS did not converge after {max_iters} iterations")

    if history is not None:
        history.capture_final(iter_idx, loss, curr_aper_phase, curr_prop_f)
        history.finalize(stop_reason, n_iters_run, loss)

    loss_full = hist_error[:n_iters_run].copy()
    return GSResult(curr_aper_f=curr_aper_f, final_loss=float(loss), n_iters_run=n_iters_run,
                    stop_reason=stop_reason, seed=seed, loss_full=loss_full, history=history)


class MGS():
    def __init__(self, freq: float, config: SimConfig):
        self.log = logging.getLogger()
        self.freq = freq
        self.wavelength = rs.wavelength(freq)
        self.plot_path = config.plot_path
        self.gs_cfg = config.gerchberg_saxton
        self.output_cfg = config.output
        self.gs_history = None
        self._rx_field = None          # measured RX field on scene.x_axis (set by measure())
        self._error_weighting = None   # phase-retrieval error weighting (set by measure())
        self.gs_rec_data = None        # scene re-illuminated by the reconstruction

        # Convention: the RX aperture sits at the origin (z=0), the bottom of the
        # image. The TX aperture sits above it and projects toward -Z, so its beam
        # travels *down* to the RX. Move the TX by changing tx.z (height) and its
        # x_min/x_max (lateral).
        scene_cfg = config.sim_scene
        rx = self._build_rx(config.rx_aperture)
        tx = self._build_tx(config.tx_aperture, scene_cfg.z_min)
        self.gs_tx = SimAperature(x_min=tx.x_min, x_max=tx.x_max, z=tx.z, dx=tx.dx)
        self.scene = SimScene(
            x_min=scene_cfg.x_min, x_max=scene_cfg.x_max,
            z_min=scene_cfg.z_min, z_max=scene_cfg.z_max,
            spacing=scene_cfg.spacing, rx_ap=rx, tx_ap=tx,
        )
        self._log_geometry()

    def _build_rx(self, rx_cfg) -> SimAperature:
        """The receive aperture: a window of `width` centred at `x_center`, at the
        scene origin (z=0). `dx` defaults to wavelength/20 when left null, and is what
        sets the receiver element count -- the independent variable in the
        scenario_caustic_hit_lambda2/lambda4 experiments."""
        # written as a multiply by 1/20, not a divide by 20: 0.05 is not exactly 1/20
        # in binary, so the two disagree by an ulp (measured on 28 config x frequency
        # combinations, though none of them moved num_points)
        rx_dx = rx_cfg.dx if rx_cfg.dx is not None else self.wavelength * (1 / 20)
        return SimAperature(
            x_min=rx_cfg.x_center - rx_cfg.width / 2.0,
            x_max=rx_cfg.x_center + rx_cfg.width / 2.0,
            z=0,
            dx=rx_dx,
        )

    def _build_tx(self, tx_cfg, z_min: float) -> SimAperature:
        """The transmit aperture and the beam it emits, defined independently of the
        scene grid. Sets self.beam_type and self.has_real_aper."""
        tx = SimAperature(x_min=tx_cfg.x_min, x_max=tx_cfg.x_max, z=tx_cfg.z, dx=tx_cfg.dx)
        assert tx.z > z_min, f"tx_aperture.z ({tx.z}) must be above the scene floor z_min ({z_min})"

        beam = tx_cfg.beam
        self.beam_type = beam.type
        # a known real aperture exists (both simulated beams); false for the
        # experimental path, where the TX aperture is what we are trying to recover
        self.has_real_aper = True
        if beam.type == "caustic":
            # caustic beam x(d) = a*d^2 + b*d + c, d = distance travelled from the TX,
            # so it is parameterised by the downstream propagation length
            a, b, c = beam.trajectory
            tx.make_caustic(self.freq, tx.z - z_min, a, b, c)
        elif beam.type == "directional":
            # steered plane wave at the configured angle
            tx.make_steer(self.freq, theta_deg=beam.steer_angle_deg)
        else:
            raise ValueError(f"unknown beam type: {beam.type}")
        return tx

    def _log_geometry(self) -> None:
        scene = self.scene
        self.log.info(f"Simulation scene:")
        self.log.info(f" - X axis: {scene.x_min:0.3f} - {scene.x_max:0.3f}")
        self.log.info(f" - Z axis: {scene.z_min:0.3f} - {scene.z_max:0.3f}")
        self.log.info(f"RX aperature:")
        rx_wl_ratio = scene.rx_ap.dx / self.wavelength
        rx_width = scene.rx_ap.x_max - scene.rx_ap.x_min
        self.log.info(f" - X axis: {scene.rx_ap.x_min:0.3f} {scene.rx_ap.x_max:0.3f} (width {rx_width:0.3f} m, dx {rx_wl_ratio:0.3f} wavelength)")
        self.log.info(f" - Z: {scene.rx_ap.z:0.3f}")
        self.log.info(f"TX aperature ({self.beam_type} beam)")
        tx_wl_ratio = scene.tx_ap.dx / self.wavelength
        self.log.info(f" - X axis: {scene.tx_ap.x_min:0.3f} {scene.tx_ap.x_max:0.3f} (dx {tx_wl_ratio:0.3f} wavelength)")
        self.log.info(f" - Z: {scene.tx_ap.z:0.3f}")

    # ----------------------------------------------------------------- scene ---
    def _propagate(self, tx_ap: SimAperature) -> np.ndarray:
        """Illuminate the whole scene from `tx_ap`: resample its profile onto the
        scene x-axis, then RS-propagate down through every z plane."""
        u0 = interp_real_imag(tx_ap.aper_axis, tx_ap.aper_profile, self.scene.x_axis)
        self.log.info("Computing wave propogation across scene")
        return rs.illuminate(self.scene.x_axis, self.scene.z_axis, u0,
                             self.wavelength, tx_ap.z)

    def illuminate_real(self) -> None:
        """Fill scene.data with the field radiated by the real TX aperture.

        Only the plots need this. The RX measurement does NOT come from here -- see
        _synthesize_rx, which propagates to one plane instead of all 3400 of them.
        """
        self.log.info("Running scene simulation with real aperature")
        self.scene.data = self._propagate(self.scene.tx_ap)

    def illuminate_reconstructed(self) -> None:
        """Fill gs_rec_data with the field radiated by the MGS-reconstructed aperture."""
        self.log.info("Running scene simulation with MGS reconstructed aperature")
        self.gs_rec_data = self._propagate(self.gs_tx)

    # ----------------------------------------------------------- measurement ---
    def _rx_plane_index(self) -> int:
        """Index of the scene z-plane the RX measurement is taken from.

        KNOWN BUG, preserved here so this stage moves no numbers: the comparison
        steps to the NEXT (farther) plane when the current one is closer, so this
        returns 1 rather than 0 for all 11 shipped configs. The measurement is
        therefore synthesized one grid cell -- 0.25 mm, lambda/8 at 150 GHz -- away
        from the plane gs_reconstruct actually models (rx_z = 0.0). Stage 7a deletes
        this method; do not "tidy" it into an argmin here, because that is the fix
        and it belongs in its own commit.
        """
        z_axis = self.scene.z_axis
        rx_z = self.scene.rx_ap.z
        idx = max(0, int(np.searchsorted(z_axis, rx_z)) - 1)
        if np.abs(z_axis[idx] - rx_z) < np.abs(z_axis[idx + 1] - rx_z):
            idx += 1
        return idx

    def _synthesize_rx(self) -> np.ndarray:
        """Propagate the real TX aperture to the RX measurement plane and sample it
        onto the RX element axis. Returns the complex RX aperture profile.

        Propagates to ONE plane, not all of them. run_grid_search used to call
        run_sim purely to get this, paying a full 3400 x 2400 scene (1.71 s and
        +190 MB RSS per frequency, held in the parent for the whole sweep) to keep a
        single row. Verified bit-identical: rs.rs(x, [z_k], ...)[0] equals
        rs.rs(x, z_axis, ...)[k], since each row is computed independently from u0.

        The round trip through rx_ap.aper_axis is kept deliberately -- it models the
        receiver element spacing (lambda/2 vs lambda/20), which is the whole point of
        scenario_caustic_hit_lambda2.yml.
        """
        scene = self.scene
        tx_ap = scene.tx_ap
        u0 = interp_real_imag(tx_ap.aper_axis, tx_ap.aper_profile, scene.x_axis)
        z_meas = scene.z_axis[self._rx_plane_index()]
        row = rs.rs(scene.x_axis, np.array([z_meas]), u0, self.wavelength,
                    z_src=tx_ap.z, forward_dir=-1.0)[0]
        return interp_real_imag(scene.x_axis, row, scene.rx_ap.aper_axis)

    def measure(self) -> None:
        """Sample the measured RX field onto the scene grid and build the error
        weighting used by phase retrieval. Idempotent.

        On the simulated path the measurement is synthesized here, by propagating
        the known TX aperture to the RX plane. On the experimental path there is no
        known TX aperture (has_real_aper is False) and rx_ap.aper_profile already
        holds the measured data. The result (self._rx_field / self._error_weighting)
        is the single measurement shared across every hypothesized TX location.
        """
        if self.has_real_aper:
            self.scene.rx_ap.aper_profile = self._synthesize_rx()
        x_axis = self.scene.x_axis
        rx_ap = self.scene.rx_ap
        orig_prop_f = interp_amp_phase(rx_ap.aper_axis, rx_ap.aper_profile, x_axis)
        # error computations are weighted to favor higher amplitude data, and ignore things outside the recieve aperature
        error_weighting = (np.abs(orig_prop_f) / np.abs(orig_prop_f).max()) + 0.25
        error_weighting[(x_axis < rx_ap.x_min) | (x_axis > rx_ap.x_max)] = 0.0
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
        # amplitude-only ON PURPOSE: this is the solver's fixed amplitude constraint
        # and its support mask, not a field. Forcing it cartesian changes the
        # constraint by 3.0-5.3% rel L2 with no error raised and no visibly broken plot.
        tx_ap = self.scene.tx_ap
        orig_aper_amp = np.abs(interp_amp_phase(tx_ap.aper_axis, tx_ap.aper_profile,
                                                self.scene.x_axis))
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
            params=self.gs_cfg,
            capture=True,
            log=self.log,
        )
        out_aper.aper_profile = interp_amp_phase(
            self.scene.x_axis, result.curr_aper_f, out_aper.aper_axis)
        return result.history

    def plot_scene(self, save_path=None, show=True):
        """Save the 4-panel scene plot (real vs MGS-reconstructed scene + TX aperture
        phase/amplitude). Writes to `save_path` if given, else self.plot_path; only
        opens an interactive window when `show` is True (set False for headless/batch)."""
        self.log.info("Plotting scene")

        scene, gs_tx = self.scene, self.gs_tx
        bounds = (scene.x_axis.min(), scene.x_axis.max(),
                  scene.z_axis.min(), scene.z_axis.max())
        panel = dict(bounds=bounds, rx_axis=scene.rx_ap.aper_axis, rx_z=scene.rx_ap.z,
                     tx_axis=scene.tx_ap.aper_axis, tx_z=scene.tx_ap.z,
                     colorbar_label="EMW (V/m)")

        if scene.data is None or self.gs_rec_data is None:
            raise RuntimeError(
                "plot_scene needs both scenes illuminated; call illuminate_real() and "
                "illuminate_reconstructed() first")

        fig = plt.figure(figsize=(20, 10), layout="constrained")
        (ax_real, ax_rec), (ax_phase, ax_amp) = fig.subplots(2, 2)

        v_max = np.nanmax([np.nanmax(np.abs(scene.data)),
                           np.nanmax(np.abs(self.gs_rec_data))])
        draw_scene(fig, ax_real, np.abs(scene.data), title="Scene Amplitude",
                   vmax=v_max, **panel)
        # Panel 2 autoscales. v_max above was computed to make the two panels
        # comparable and then commented out at the call; the reconstruction renders
        # dimmer than the real scene, so an independent scale is easier to read.
        # Pass vmax=v_max here instead to put them on one scale.
        draw_scene(fig, ax_rec, np.abs(self.gs_rec_data),
                   title="MGS reconstruction Scene Amplitude", vmax=None, **panel)

        # both aperture panels are drawn against the full scene x extent
        xlim = (scene.x_min, scene.x_max)
        tx_interp = interp_real_imag(scene.tx_ap.aper_axis, scene.tx_ap.aper_profile,
                                     scene.x_axis)
        gs_amp = scipy.interpolate.interp1d(
            gs_tx.aper_axis, np.abs(gs_tx.aper_profile), kind="linear",
            fill_value=0, bounds_error=False, assume_sorted=True)(scene.x_axis)

        phase_series, amp_series = [], []
        if self.has_real_aper:
            phase_series.append(("Real TX", scene.tx_ap.aper_axis,
                                 np.unwrap(np.angle(scene.tx_ap.aper_profile))))
            amp_series.append(("Real TX", scene.x_axis, np.abs(tx_interp)))
        phase_series.append(("MGS Reconstructed TX", gs_tx.aper_axis,
                             np.unwrap(np.angle(gs_tx.aper_profile))))
        amp_series.append(("MGS Reconstructed TX", scene.x_axis, gs_amp))

        draw_line_panel(ax_phase, phase_series, title="TX Aperature Phase",
                        xlabel="x (m)", ylabel="Phase [rad]", xlim=xlim)
        draw_line_panel(ax_amp, amp_series, title="TX Aperature Amplitude",
                        xlabel="x (m)", ylabel="Amplitude EMF (V/m)", xlim=xlim)

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
        self.wavelength = rs.wavelength(freq)

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

        self.gs_rec_data = None   # filled by illuminate_reconstructed()

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
        help="Override output.run_name (the run directory NAME under output_dir; "
             "it replaces the name, it is not appended to it)",
        default=None
    )
    parser.add_argument(
        "--out", "-o",
        type=Path,
        help="Run directory to write this run into, named outright "
             "(overrides output.output_dir + output.run_name)",
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
    if args.out is not None:
        config.output.output_dir = Path(args.out).parent
        config.output.run_name = Path(args.out).name
    if args.seed is not None: config.gerchberg_saxton.seed = args.seed
    if args.no_save: config.output.save_run = False

    # Fail fast on a run-directory collision, before spending the solve.
    if config.output.save_run:
        check_run_dir(config.output.output_dir, config.output.run_name, kind="mgs")

    if args.rx_path is not None and not args.heatmap_path is None:
        rx_path = Path(args.rx_path)
        heatmap_path = Path(args.heatmap_path)
        mgs = ExpMGS(rx_path, heatmap_path, args.freq, config)
        mgs.run_gerch_sax()
        mgs.illuminate_reconstructed()
        mgs.plot_scene()
        is_exp = True
    else:
        mgs = MGS(args.freq, config)
        mgs.illuminate_real()
        mgs.run_gerch_sax()
        mgs.illuminate_reconstructed()
        mgs.plot_scene()
        is_exp = False

    # Deferred to here on purpose: make_run_dir clears the target, so a crashed (or
    # plt.show()-blocked) solve must not have destroyed the previous run already.
    if config.output.save_run:
        run_dir = make_run_dir(config.output.output_dir, config.output.run_name, kind="mgs")
        save_run(mgs, run_dir, config, args.config, args.freq, vars(args), is_exp=is_exp)

if __name__ == "__main__":
    main()

