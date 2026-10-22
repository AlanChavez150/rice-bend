"""Re-illuminating candidate beams: what each reconstructed aperture actually
radiates into the scene, as PNGs, an averaged scene, and an animation.

Also holds make_true_mgs_plot, the baseline single MGS run at the KNOWN TX location
-- the thing every candidate is being compared against.
"""

import json
import logging
from collections import namedtuple
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Rectangle

from rice_bend import rs
from rice_bend.config import NoiseConfig, PlotWindowConfig, center_freq_index
from rice_bend.data_store import load_run_config, write_mp4
from rice_bend.grid_sweep import GridSearchRun
from rice_bend.mgs import MGS, check_amplitude_mode
from rice_bend.parallel import map_workers, round_robin_chunks, worker_shared
from rice_bend.plotting import draw_line_panel, draw_scene

# Demoted from CLI flags: never changed across 90 saved runs, and the tradeoff is
# better documented next to the code than in --help.
SCENE_Z_PLANES = 200      # z planes per re-illumination; lower = faster and coarser


ANIM_FPS = 10             # candidate_beams.mp4 frame rate


ANIM_TOP_DEFAULT = 100    # --anim holds every frame at once (see _anim_top)


ANIM_WARN_FRAMES = 400


def _center_freq_index(freqs: List[float]) -> int:
    """Index of the default display frequency: the centre of the sorted list
    (matching plot_residual_scatter_3d_diff's baseline convention and the delay
    solver's reference frequency). A scene re-illumination is inherently
    monochromatic, so multi-frequency runs need ONE frequency picked for the
    scene views."""
    return center_freq_index(freqs)


def _display_freq_index(freqs: List[float], scene_freq: Optional[float]) -> int:
    """Index of the frequency the scene views render at: --scene-freq when given
    (it must be one of the run's frequencies), else the centre frequency."""
    if scene_freq is None:
        return _center_freq_index(freqs)
    for i, f in enumerate(freqs):
        if abs(f - scene_freq) <= 1e-6 * max(abs(f), 1.0):
            return i
    ghz = ", ".join(f"{f / 1e9:g}" for f in freqs)
    raise SystemExit(f"--scene-freq {scene_freq / 1e9:g} GHz is not one of this "
                     f"run's frequencies ({ghz} GHz)")


def _manifest_freqs(manifest: dict) -> List[float]:
    """The manifest's frequency list — schema 3 and 4 (`frequencies`) or the
    schema-2 scalar `freq_hz` as a length-1 list."""
    if "frequencies" in manifest:
        return [float(e["freq_hz"]) for e in manifest["frequencies"]]
    return [float(manifest["freq_hz"])]


def _restore_run_noise(config, manifest: dict, log: logging.Logger) -> None:
    """Set config.noise to the receiver noise the saved run ACTUALLY measured with.

    load_run_config prefers the verbatim config_source.yml, which knows neither a
    --snr-db/--noise-seed override nor a seed drawn at run time, so re-measuring from
    it alone would replay a different measurement than the one the sweep solved. The
    manifest's `noise` block records what ran; a missing or null block (a noiseless
    run, or one that predates noise) means noiseless. The replay then draws the same
    noise because each tone's stream is keyed by its frequency value, whatever order
    the frequency list is in (--scene-freq rotates it).
    """
    block = manifest.get("noise")
    if block:
        config.noise = NoiseConfig(snr_db=block["snr_db"], seed=block["seed"])
        log.info(f"true MGS run: restored the run's receiver noise "
                 f"(SNR {block['snr_db']:g} dB, seed {block['seed']})")
        recorded = block.get("numpy_version")
        if recorded is not None and recorded != np.__version__:
            log.warning(f"true MGS run: the noise was drawn under numpy {recorded}, this "
                        f"is numpy {np.__version__}; the replayed noise draw may differ")
        return
    if config.noise.snr_db is not None:
        log.info(f"true MGS run: the saved config enables noise (SNR "
                 f"{config.noise.snr_db:g} dB) but the run recorded none; replaying "
                 f"noiseless")
    config.noise = NoiseConfig(snr_db=None, seed=config.noise.seed)


def make_true_mgs_plot(run_dir: Path, log: Optional[logging.Logger] = None,
                       scene_freq: Optional[float] = None) -> Optional[Path]:
    """Reconstruct + plot the single "true" MGS run for a saved grid run.

    The grid search treats the TX as unknown; this is the baseline plain-`mgs` result at
    the KNOWN (true) TX location, for a simulated run at a fixed amplitude (with
    amplitude descend or level, grid-search-mgs --true-mgs solves at the true TX as
    a candidate instead: grid_sweep.solve_at_measured_tx). The saved config carries
    tx_aperture.amplitude, so the replayed plate is scaled as the sweep's was
    (MGS._apply_tx_amplitude). Using the run's saved config + frequency list, it
    illuminates the real scene, runs the joint MGS solve at the true TX plane,
    re-illuminates with the reconstruction, and saves the 4-panel scene plot (identical
    to what `mgs` produces). This recomputes a full MGS solve, so it is more expensive
    than the manifest-only residual plots.

    Output: <run_dir>/true_mgs_scene.png. Returns the path, or None if the config is
    missing from the run dir.
    """
    log = log or logging.getLogger()
    run_dir = Path(run_dir)
    with open(run_dir / "candidate_beams.json") as f:
        manifest = json.load(f)
    freqs = _manifest_freqs(manifest)
    if scene_freq is not None:
        # MGS renders its scene panels at the PRIMARY (first) frequency, so honour
        # --scene-freq by rotating it to the front. The solve stays joint over the
        # same frequency set.
        i = _display_freq_index(freqs, scene_freq)
        freqs = [freqs[i]] + freqs[:i] + freqs[i + 1:]
    config = load_run_config(run_dir, log)
    if config is None:
        return None
    # the baseline is the capturing `mgs` solve, which a fitted amplitude cannot
    # use: a clear refusal now rather than one from inside the solve after the real
    # scene has been illuminated. grid-search-mgs never sends such a run here: a
    # fresh one solves at the true TX as a candidate instead (true_mgs.json), and
    # its --replot --true-mgs is refused up front
    check_amplitude_mode(config.gerchberg_saxton, capture=True)

    # Pin the GS seed so the baseline is reproducible across --replot calls. Plain mgs
    # leaves gerchberg_saxton.seed null -> a fresh random initial phase each run, which
    # would make true_mgs_scene.png differ every time. Reuse the run's recorded sweep
    # seed when available, else 0.
    if config.gerchberg_saxton.seed is None:
        pinned = manifest.get("seed")
        config.gerchberg_saxton.seed = 0 if pinned is None else int(pinned)
        log.info(f"true MGS run: gerchberg_saxton.seed was null; pinned to "
                 f"{config.gerchberg_saxton.seed} for a reproducible baseline")
    # likewise the noise: replay the measurement the sweep actually solved against
    _restore_run_noise(config, manifest, log)

    out_path = run_dir / "true_mgs_scene.png"
    ghz = ", ".join(f"{f / 1e9:.3g}" for f in freqs)
    log.info(f"Reconstructing the true MGS run (TX known) at {ghz} GHz -> {out_path}")
    mgs = MGS(freqs, config)
    mgs.illuminate_real()             # illuminate the real scene
    mgs.run_gerch_sax()               # MGS solve at the true TX plane
    mgs.log_landing()                 # the LANDING block `mgs` logs, for this baseline
    mgs.illuminate_reconstructed()    # re-illuminate with the reconstructed aperture
    mgs.plot_scene(save_path=out_path, show=False)
    log.info(f"Wrote true MGS scene to {out_path}")
    return out_path


@dataclass
class CandidateScene:
    """A candidate's reconstructed aperture, ready to be re-illuminated."""
    index: int
    z: float
    x_center: float
    final_loss: float
    aper_axis: np.ndarray
    aper_profile: np.ndarray
    stop_reason: str = "?"


def _reilluminate(x_axis: np.ndarray, z_axis: np.ndarray, aper_axis: np.ndarray,
                  aper_profile: np.ndarray, wavelength: float, tx_z: float) -> np.ndarray:
    """RS-propagate a candidate aperture from its plane (tx_z) down through the scene.

    The aperture is interpolated onto the scene x-axis, propagated toward -Z, and the
    region behind the TX plane is zeroed (it does not radiate there). Returns |field|.
    """
    # zero-fill beyond the aperture extent. np.interp clamps to the edge values by
    # default, which would smear a spurious ~unit-amplitude source across the whole
    # scene (energy from outside the aperture); match MGS's fill_value=0.
    #
    # Deliberately np.interp and not interp.interp_real_imag: the two disagree at ULP
    # level (1.3e-7 on complex64), and this is the only thing keeping --replot
    # --scenes/--anim working on saved runs whose npz holds a 400-point aper_axis
    # against a 2400-point scene.
    ap = (np.interp(x_axis, aper_axis, aper_profile.real, left=0.0, right=0.0)
          + 1j * np.interp(x_axis, aper_axis, aper_profile.imag, left=0.0, right=0.0))
    # np.abs of a complex64 field is already float32, so no cast is needed here.
    return np.abs(rs.illuminate(x_axis, z_axis, ap, wavelength, tx_z))


@dataclass
class SceneContext:
    """Everything needed to re-illuminate candidate beams into scene fields.

    A measured run carries no real TX profile: its beam is the unknown, so
    real_tx_aper_axis / real_tx_aper_profile are None (`measured` is True) and the
    truth is the measured TX window alone (real_tx_z, real_tx_x_min..real_tx_x_max).
    Its scenes have no real-beam panel; the measured window is outlined instead."""
    items: List[CandidateScene]
    scene_bounds: Tuple[float, float, float, float]   # x_min, x_max, z_min, z_max
    wavelength: float
    scene_x_axis: np.ndarray
    real_tx_z: float
    real_tx_x_center: float
    real_tx_aper_axis: Optional[np.ndarray]   # ground-truth TX aperture (for the real
    real_tx_aper_profile: Optional[np.ndarray]  # beam panel); None on a measured run
    rx_aper_axis: np.ndarray             # RX aperture x-axis (for the RX dots)
    real_tx_x_min: Optional[float] = None   # the TX window; outlined on a measured run
    real_tx_x_max: Optional[float] = None
    rx_z: Optional[float] = None         # the RX plane; None = scene z_min (_scene_panel)

    @property
    def measured(self) -> bool:
        """True on a measured run: there is no real TX profile to re-illuminate."""
        return self.real_tx_aper_profile is None


def scenes_from_run(run: GridSearchRun,
                    scene_freq: Optional[float] = None) -> SceneContext:
    """Adapter: a SceneContext from an in-memory run. Multi-frequency runs are
    rendered at `scene_freq` (must be one of the run's frequencies) or, by
    default, the centre frequency. A measured run's context has no real profile,
    and carries the RX plane from the run."""
    items = [CandidateScene(c.point.index, c.point.z, c.point.x_center, c.final_loss,
                            c.aper_axis, c.aper_profile, c.stop_reason)
             for c in run.candidates]
    i = _display_freq_index(run.freqs, scene_freq)
    measured = run.measured is not None
    return SceneContext(items, run.scene_bounds, run.wavelengths[i], run.scene_x_axis,
                        run.real_tx_z, 0.5 * (run.real_tx_x_min + run.real_tx_x_max),
                        None if measured else run.real_tx_aper_axis,
                        None if measured else run.real_tx_aper_profiles[i],
                        run.rx_aper_axes[i],
                        real_tx_x_min=run.real_tx_x_min, real_tx_x_max=run.real_tx_x_max,
                        rx_z=float(run.rx_z) if measured else None)


def scenes_from_manifest(run_dir: Path,
                         scene_freq: Optional[float] = None) -> SceneContext:
    """Adapter: a SceneContext from a saved run (reads each candidate npz).

    Handles both layouts: a joint run (frequency list, stacked TX profiles,
    indexed RX aperture keys — rendered at `scene_freq`, defaulting to the centre
    frequency) and a schema-2 per-frequency run (scalar keys). A joint run is
    schema 3, or schema 4: the layout the removed refinement pass wrote, whose
    extra keys are ignored. A schema-4 candidate's npz holds its stored solve, so
    it is rendered from the profile its loss belongs to. A measured run (a
    `measured` block; schema 3) has no real TX profile in measurement.npz: its
    context carries none, and takes the RX plane from the manifest."""
    run_dir = Path(run_dir)
    with open(run_dir / "candidate_beams.json") as f:
        m = json.load(f)
    measured = m.get("measured") is not None
    sb = m["scene_bounds"]
    scene_bounds = (sb["x_min"], sb["x_max"], sb["z_min"], sb["z_max"])
    meas = np.load(run_dir / "measurement.npz")
    items = []
    for c in m["candidates"]:
        d = np.load(run_dir / c["npz"])
        items.append(CandidateScene(c["index"], c["z"], c["x_center"], c["final_loss"],
                                    d["aper_axis"], d["aper_profile"], c["stop_reason"]))
    gt = m["ground_truth"]
    i = _display_freq_index(_manifest_freqs(m), scene_freq)
    if "frequencies" in m:   # schema 3 and 4: joint layout
        wavelength = float(m["frequencies"][i]["wavelength_m"])
        tx_profile = None if measured else meas["real_tx_aper_profile"][i]
        rx_axis = meas[f"rx_aper_axis_{i:02d}"]
    else:                    # schema 2: legacy per-frequency layout (i is 0)
        wavelength = m["wavelength_m"]
        tx_profile = meas["real_tx_aper_profile"]
        rx_axis = meas["rx_aper_axis"]
    return SceneContext(items, scene_bounds, wavelength, meas["scene_x_axis"],
                        gt["real_tx_z"], 0.5 * (gt["real_tx_x_min"] + gt["real_tx_x_max"]),
                        None if measured else meas["real_tx_aper_axis"], tx_profile,
                        rx_axis,
                        real_tx_x_min=gt["real_tx_x_min"], real_tx_x_max=gt["real_tx_x_max"],
                        rx_z=float(m["rx_aperture"]["z"]) if measured else None)


@dataclass(frozen=True)
class SceneView:
    """The part of the scene a scene panel shows, and the planes it is propagated to.

    Without a plot window (`window` None) that is the whole scene, as it always was:
    `z_axis` spans the scene's z range, every scene x sample is kept (`cols` None)
    and `extent` is the scene bounds. With one, the same number of z planes is
    spread over the window's z range (the resolution goes where the plot is), the
    fields are propagated on the whole scene x axis (an aperture reaching past the
    window still radiates into it) and then cut to the x samples inside the window,
    and the panel shows exactly the window at equal aspect."""
    z_axis: np.ndarray
    cols: Optional[slice]                          # scene x samples kept; None = all
    extent: Tuple[float, float, float, float]      # imshow extent: x_min, x_max, z_min, z_max
    window: Optional[PlotWindowConfig] = None

    @staticmethod
    def build(ctx: "SceneContext", z_planes: int,
              window: Optional[PlotWindowConfig]) -> "SceneView":
        x_min, x_max, z_min, z_max = ctx.scene_bounds
        if window is None:
            return SceneView(np.linspace(z_min, z_max, z_planes), None, ctx.scene_bounds)
        x_axis = np.asarray(ctx.scene_x_axis)
        tol = window.EDGE_TOL
        inside = np.flatnonzero((x_axis >= window.x_min - tol)
                                & (x_axis <= window.x_max + tol))
        if inside.size < 2:
            raise ValueError(
                f"plot window x [{window.x_min:g}, {window.x_max:g}] m holds "
                f"{inside.size} sample(s) of the scene's x axis [{x_min:g}, "
                f"{x_max:g}] m: no scene to show")
        first, last = int(inside[0]), int(inside[-1])
        return SceneView(np.linspace(window.z_min, window.z_max, z_planes),
                         slice(first, last + 1),
                         (float(x_axis[first]), float(x_axis[last]),
                          window.z_min, window.z_max), window)

    def crop(self, field: np.ndarray) -> np.ndarray:
        """A field on the whole scene x axis cut to the x samples shown."""
        return field if self.cols is None else field[:, self.cols]


def _scene_panel(fig, ax, field: np.ndarray, ctx: SceneContext,
                 tx_axis: Optional[np.ndarray], tx_z: float, vmax: float,
                 title: str, view: Optional[SceneView] = None) -> None:
    """plotting.draw_scene bound to a SceneContext, plus the measured TX window on a
    measured run (_mark_measured_tx).

    On a simulated run the RX plane is passed as scene z_min. That is an assumption,
    not a fact: it holds for every simulated config (rx.z == 0 == z_min), and a
    simulated SceneContext does not carry the RX z, so this is where the assumption
    lives. A measured context carries its RX z (ctx.rx_z) and that is used instead.

    Legend pinned to "upper right" rather than "best": these panels also become mp4
    frames, and a legend that relocates between frames is worse than one that
    occasionally overlaps.

    `field` is on `view` (the whole scene when None or windowless); a windowed view
    limits the panel to exactly the window, at equal aspect.
    """
    rx_z = ctx.scene_bounds[2] if ctx.rx_z is None else ctx.rx_z
    bounds = ctx.scene_bounds if view is None else view.extent
    draw_scene(fig, ax, field, bounds=bounds,
               rx_axis=ctx.rx_aper_axis, rx_z=rx_z,
               tx_axis=tx_axis, tx_z=tx_z, vmax=vmax, title=title,
               legend_loc="upper right")
    if ctx.measured:
        _mark_measured_tx(ax, ctx, bounds)
    if view is not None and view.window is not None:
        w = view.window
        ax.set_xlim(w.x_min, w.x_max)
        ax.set_ylim(w.z_min, w.z_max)
        ax.set_aspect("equal")


def _save_scene_fig(fig, out_path: Path, view: Optional[SceneView]) -> None:
    """Save and close a scene figure. A windowed one is saved with a tight bounding
    box: constrained layout sizes an equal-aspect panel's margins against its slot
    rather than the narrower box it is drawn in, which can push the panel's y label
    off the left edge; the tight box grows the canvas to keep every label."""
    if view is not None and view.window is not None:
        fig.savefig(out_path, dpi=120, bbox_inches="tight", pad_inches=0.1)
    else:
        fig.savefig(out_path, dpi=120)
    plt.close(fig)


# The measured TX window's outline: a dashed box around z = real_tx_z spanning the
# window in x, in a colour that stands off inferno's black-red-yellow and the red RX
# / blue candidate-TX dots. Its half-height is a fraction of the z range shown, so
# the box reads as a box at any scene size (1.5% is ~13 mm on the 0.85 m scene, 6 mm
# on the 0.4 m one).
_MEASURED_TX_COLOR = "cyan"
_MEASURED_TX_HALF_HEIGHT = 0.015


def _mark_measured_tx(ax, ctx: SceneContext,
                      bounds: Tuple[float, float, float, float]) -> None:
    """Outline the measured TX window (z = ctx.real_tx_z, x = ctx.real_tx_x_min ..
    ctx.real_tx_x_max) on a scene panel showing `bounds`, and redraw the legend
    with it as "measured TX" (same placement and style as draw_scene's)."""
    _, _, z_min, z_max = bounds
    h = _MEASURED_TX_HALF_HEIGHT * (z_max - z_min)
    ax.add_patch(Rectangle((ctx.real_tx_x_min, ctx.real_tx_z - h),
                           ctx.real_tx_x_max - ctx.real_tx_x_min, 2.0 * h,
                           fill=False, edgecolor=_MEASURED_TX_COLOR, linestyle="--",
                           linewidth=2.0, zorder=6, label="measured TX"))
    ax.legend(loc="upper right", framealpha=0.9, markerscale=2)


def _measured_tx_dist_mm(item: "CandidateScene", ctx: SceneContext) -> float:
    """A candidate's distance to the measured TX, in mm: centre to window centre,
    the analysis.json convention (dist_to_truth_m)."""
    return 1e3 * float(np.hypot(item.z - ctx.real_tx_z,
                                item.x_center - ctx.real_tx_x_center))


def _candidate_title(item: "CandidateScene", ctx: SceneContext, best: bool,
                     view: Optional[SceneView] = None) -> str:
    """A candidate scene panel's title: index, best tag, MSE and a non-converged stop
    reason; on a measured run also its distance to the measured TX; and, when a plot
    window leaves the candidate's TX centre off the panel, where that TX is."""
    tag = " (best)" if best else ""
    note = "" if item.stop_reason == "converged" else f" [{item.stop_reason}]"
    title = f"Candidate #{item.index}{tag} beam — MSE={item.final_loss:.4g}{note}"
    if ctx.measured:
        title += f" — {_measured_tx_dist_mm(item, ctx):.1f} mm from measured TX"
    if view is not None and view.window is not None \
            and not view.window.contains(item.x_center, item.z):
        title += (f"\nTX at x={item.x_center:+.3f} z={item.z:.3f} m, outside the "
                  "plot window")
    return title


def _on_scene_axis(scene_x: np.ndarray, aper_axis: np.ndarray,
                   values: np.ndarray) -> np.ndarray:
    """Place per-aperture `values` onto the full scene x-axis, NaN outside the aperture.

    The scene is wider than the aperture, so scene points beyond
    [aper_axis[0], aper_axis[-1]] have no data and are left as NaN (rendered as gaps).
    """
    out = np.full(len(scene_x), np.nan)
    inside = (scene_x >= aper_axis[0]) & (scene_x <= aper_axis[-1])
    out[inside] = np.interp(scene_x[inside], aper_axis, values)
    return out


def _plot_candidate_quad(cand_field: np.ndarray, real_field: Optional[np.ndarray],
                         item: CandidateScene, ctx: SceneContext, out_path: Path,
                         best: bool = False, view: Optional[SceneView] = None) -> None:
    """4-panel candidate PNG: candidate beam | real beam (shared scale) over candidate
    aperture phase | candidate aperture amplitude. The fields are on `view`."""
    vmax = float(cand_field.max())
    if real_field is not None:
        vmax = max(vmax, float(real_field.max()))
    vmax = vmax or 1.0

    fig, axes = plt.subplots(2, 2, figsize=(14, 12), layout="constrained")
    (ax_cand, ax_real), (ax_phase, ax_amp) = axes

    _scene_panel(fig, ax_cand, cand_field, ctx, item.aper_axis, item.z, vmax,
                 _candidate_title(item, ctx, best, view), view)

    if real_field is not None:
        _scene_panel(fig, ax_real, real_field, ctx, ctx.real_tx_aper_axis, ctx.real_tx_z,
                     vmax, "Real beam", view)
    else:
        ax_real.text(0.5, 0.5, "real beam unavailable", ha="center", va="center",
                     transform=ax_real.transAxes)
        ax_real.set_title("Real beam")

    _aperture_panels(ax_phase, ax_amp, item, ctx, view)
    _save_scene_fig(fig, out_path, view)


def _plot_candidate_measured(cand_field: np.ndarray, item: CandidateScene,
                             ctx: SceneContext, out_path: Path,
                             best: bool = False, view: Optional[SceneView] = None) -> None:
    """3-panel candidate PNG for a measured run, whose TX beam is unknown (so there
    is no real-beam panel): the candidate beam, tall on the left, with the measured
    TX window outlined and the candidate's distance to it in the title; the
    candidate aperture phase and amplitude stacked on the right. The colour scale
    is the candidate's own field, which is on `view`."""
    vmax = float(cand_field.max()) or 1.0
    fig, axd = plt.subplot_mosaic([["scene", "phase"], ["scene", "amp"]],
                                  figsize=(16, 10), layout="constrained",
                                  gridspec_kw={"width_ratios": [1.15, 1.0]})
    _scene_panel(fig, axd["scene"], cand_field, ctx, item.aper_axis, item.z, vmax,
                 _candidate_title(item, ctx, best, view), view)
    _aperture_panels(axd["phase"], axd["amp"], item, ctx, view)
    _save_scene_fig(fig, out_path, view)


def _aperture_panels(ax_phase, ax_amp, item: CandidateScene, ctx: SceneContext,
                     view: Optional[SceneView] = None) -> None:
    """The candidate aperture's unwrapped phase and amplitude, as line panels: over
    the scene's x range, or over the window's (data and limits) with a window. The
    phase has a gap wherever the amplitude is 0."""
    # phase/amplitude on the full scene x-axis; NaN (gap) where the scene extends
    # beyond the candidate aperture (no data there)
    scene_x = np.asarray(ctx.scene_x_axis)
    x_min, x_max = ctx.scene_bounds[0], ctx.scene_bounds[1]
    if view is not None and view.window is not None:
        # the unwrap runs over the whole aperture first, so the curve inside the
        # window is the same curve, only cut
        scene_x = scene_x[view.cols]
        x_min, x_max = view.window.x_min, view.window.x_max
    # the phase is undefined where the amplitude is 0 (a descended amplitude is
    # clipped to 0 there), and np.angle's 0/±pi at those samples would go through
    # the unwrap as false 2π slips: unwrap over the nonzero samples only, NaN (gap)
    # at the rest. A fixed amplitude is nonzero across the aperture: the plain unwrap.
    lit = np.abs(item.aper_profile) > 0
    aper_phase = np.full(len(lit), np.nan)
    if lit.any():
        aper_phase[lit] = np.unwrap(np.angle(item.aper_profile[lit]))
    phase = _on_scene_axis(scene_x, item.aper_axis, aper_phase)
    amp = _on_scene_axis(scene_x, item.aper_axis, np.abs(item.aper_profile))

    draw_line_panel(ax_phase, [(None, scene_x, phase, "C0")],
                    title="Candidate TX phase", xlabel="x (m)",
                    ylabel="phase unwrapped [rad]", xlim=(x_min, x_max))
    draw_line_panel(ax_amp, [(None, scene_x, amp, "C3")],
                    title="Candidate TX amplitude", xlabel="x (m)",
                    ylabel="amplitude (V/m)", xlim=(x_min, x_max),
                    ylim=_amplitude_ylim(item.aper_profile))


# How far an amplitude peak may sit from 1 and still be the uniform box's: the
# profile is stored as complex64, so a unit phasor's magnitude is 1 to ~1e-7
_UNIT_BOX_TOL = 1e-4


def _amplitude_ylim(aper_profile: np.ndarray) -> Tuple[float, float]:
    """The amplitude panel's y range: 0 to 1.1x the profile's peak, which is the
    fixed (0, 1.1) the panel always had when the peak is 1 — the uniform box of
    every fixed-amplitude solve (and of older runs whose box was resampled, and
    dips between its samples) — and returned as exactly that there, so those
    panels are drawn as they always were. A descended amplitude may peak
    anywhere up to amp_max, a fitted level (amplitude: level) anywhere at all."""
    amp = np.abs(np.asarray(aper_profile))
    peak = float(amp.max()) if amp.size else 0.0
    if not peak > 0 or abs(peak - 1.0) <= _UNIT_BOX_TOL:
        return (0.0, 1.1)
    return (0.0, 1.1 * peak)


# Read-only scene geometry + real beam, handed to the scene tasks once per worker.
# real_field is already on `view`; z_axis is view.z_axis.
SceneShared = namedtuple("SceneShared",
                         "ctx real_field x_axis z_axis best_index scenes_dir view")


def _agg_backend() -> None:
    """Worker setup: each scene task saves its own PNG, so force the headless backend.
    Runs in the pool only — doing it in the parent would permanently kill --show."""
    try:
        plt.switch_backend("Agg")
    except Exception:
        pass


def _render_scene_chunk(chunk: "List[CandidateScene]"):
    """Render every candidate in `chunk` to its PNG; return (partial field-sum, count,
    skipped). Re-illumination + plotting are deterministic, so each PNG is identical
    however the chunks are distributed.

    Chunked rather than one task per candidate ON PURPOSE: a task returns a whole
    scene field, so per-candidate tasks would stream 2304 of them back to the parent.
    """
    s = worker_shared()
    acc = None
    count = 0
    skipped: List[Tuple[int, str]] = []
    log = logging.getLogger()
    for it in chunk:
        try:
            field = s.view.crop(_reilluminate(s.x_axis, s.z_axis, it.aper_axis,
                                              it.aper_profile, s.ctx.wavelength, it.z))
        except RuntimeError as e:
            skipped.append((it.index, str(e)))
            continue
        out_path = s.scenes_dir / f"cand_{it.index:04d}.png"
        if s.ctx.measured:
            _plot_candidate_measured(field, it, s.ctx, out_path,
                                     best=(it.index == s.best_index), view=s.view)
        else:
            _plot_candidate_quad(field, s.real_field, it, s.ctx, out_path,
                                 best=(it.index == s.best_index), view=s.view)
        acc = field.astype(np.float64) if acc is None else acc + field
        count += 1
        if count % 10 == 0:
            log.info(f"  {count} rendered in this chunk (through #{it.index})")
    return acc, count, skipped


def make_candidate_scenes(ctx: SceneContext, out_dir: Path, *, z_planes: int = SCENE_Z_PLANES,
                          top: Optional[int] = None, average: bool = True, jobs: int = 1,
                          log: Optional[logging.Logger] = None,
                          window: Optional[PlotWindowConfig] = None) -> None:
    """Write one 4-panel PNG per candidate beam (scenes/cand_####.png) plus, by default, a
    single PNG averaging every candidate beam (scene_average.png).

    Each candidate PNG shows the candidate beam, the real beam (same color scale), and the
    candidate's reconstructed aperture phase and amplitude; scenes mark the RX aperture
    (red) and TX aperture (blue). With `top` set, only the N lowest-residual candidates are
    rendered, and the average spans exactly those. The average accumulates incrementally.

    On a measured run (ctx.measured) there is no real beam: each PNG has 3 panels
    (_plot_candidate_measured), and both it and the average outline the measured TX
    window instead of showing a real TX.

    With `window` (a config.PlotWindowConfig) every scene panel, the average's
    included, shows exactly that window at equal aspect, its z planes spread over
    the window's z range, and the phase/amplitude panels span the window's x range
    (see SceneView).
    """
    log = log or logging.getLogger()
    items = ctx.items
    if not items:
        log.warning("No candidates to render scenes for")
        return
    if top is not None:
        items = sorted(items, key=lambda it: it.final_loss)[:top]
    try:
        view = SceneView.build(ctx, z_planes, window)
    except ValueError as e:
        log.warning(f"No candidate scenes rendered: {e}")
        return
    out_dir = Path(out_dir)
    scenes_dir = out_dir / "scenes"
    scenes_dir.mkdir(parents=True, exist_ok=True)

    x_axis = np.asarray(ctx.scene_x_axis)
    z_axis = view.z_axis
    best = min(items, key=lambda it: it.final_loss)

    # the real beam is identical for every candidate -> re-illuminate it once (a
    # measured run has none: its TX's beam is the unknown)
    real_field = None
    if not ctx.measured:
        try:
            real_field = view.crop(_reilluminate(x_axis, z_axis, ctx.real_tx_aper_axis,
                                                 ctx.real_tx_aper_profile, ctx.wavelength,
                                                 ctx.real_tx_z))
        except RuntimeError as e:
            log.warning(f"real beam re-illumination failed ({e}); real panel will be "
                        "blank")

    # Each re-illumination + plot is independent and deterministic, so candidates are
    # distributed across worker processes when jobs > 1 (the per-candidate PNGs are
    # identical to the serial render; only the averaged-scene sum order differs, to
    # float round-off). The running field-sum for the average is reduced from per-worker
    # partial sums.
    shared = SceneShared(ctx=ctx, real_field=real_field, x_axis=x_axis, z_axis=z_axis,
                         best_index=best.index, scenes_dir=scenes_dir, view=view)
    chunks = round_robin_chunks(items, max(1, int(jobs)))
    parts = map_workers(_render_scene_chunk, chunks, jobs=jobs, shared=shared,
                        setup=_agg_backend, log=log,
                        desc=f"Rendering {len(items)} candidate scene(s) over "
                             f"{z_planes} z-planes -> {scenes_dir}/")

    # The averaged scene sums in chunk order, which is not candidate order; float
    # addition is not associative, so the last bits of scene_average.png depend on
    # the chunk count. Honest, and unchanged from before.
    acc = None
    count = 0
    for pacc, pcount, skipped in parts:
        if pacc is not None:
            acc = pacc if acc is None else acc + pacc
        count += pcount
        for idx, msg in skipped:
            log.warning(f"  candidate #{idx}: skipped ({msg})")

    log.info(f"Wrote {count} candidate scene(s) to {scenes_dir}/")
    if average and acc is not None:
        avg = (acc / count).astype(np.float32)
        # a measured run has no real TX to show: _scene_panel outlines its window
        shown = "measured TX window" if ctx.measured else "real"
        fig, ax = plt.subplots(figsize=(8, 8), layout="constrained")
        _scene_panel(fig, ax, avg, ctx, ctx.real_tx_aper_axis, ctx.real_tx_z,
                     float(avg.max()) or 1.0,
                     f"Average of {count} candidate beams (TX shown = {shown})", view)
        avg_path = out_dir / "scene_average.png"
        _save_scene_fig(fig, avg_path, view)
        log.info(f"Wrote averaged scene to {avg_path}")


# Read-only scene geometry handed to the frame tasks once per worker.
_AnimShared = namedtuple("_AnimShared", "x_axis z_axis wavelength")


def _anim_frame(item: "CandidateScene"):
    """Re-illuminate one candidate into a scene field (task); returns (field, reason).

    Returns the skip reason rather than swallowing it — this used to discard str(e)
    while the serial arm printed the real message, so the same failure read
    differently depending on --jobs.
    """
    s = worker_shared()
    try:
        return _reilluminate(s.x_axis, s.z_axis, item.aper_axis, item.aper_profile,
                             s.wavelength, item.z), None
    except RuntimeError as e:
        return None, str(e)


def animate_candidate_beams(ctx: SceneContext, out_path: Path, *, z_planes: int = SCENE_Z_PLANES,
                            top: Optional[int] = None, fps: int = 10, jobs: int = 1,
                            show: bool = False, log: Optional[logging.Logger] = None):
    """Animate the candidate beams: one frame per candidate re-illuminating the scene.

    Frames sweep the grid in index order (or the `top` lowest-residual, best first). All
    frames share one color scale; the RX aperture (red) is fixed and the TX aperture (blue)
    moves with each candidate. Renders .mp4 via ffmpeg (reuses animate._write_mp4).
    """
    log = log or logging.getLogger()
    items = ctx.items
    if not items:
        log.warning("No candidates to animate")
        return None
    items = (sorted(items, key=lambda it: it.final_loss)[:top] if top is not None
             else sorted(items, key=lambda it: it.index))

    x_min, x_max, z_min, z_max = ctx.scene_bounds
    x_axis = np.asarray(ctx.scene_x_axis)
    z_axis = np.linspace(z_min, z_max, z_planes)
    extent = [x_min, x_max, z_min, z_max]

    # one (independent) propagation per candidate; precompute frames so the color scale
    # is fixed, distributing them across workers when jobs > 1 (output identical to serial).
    shared = _AnimShared(x_axis=x_axis, z_axis=z_axis, wavelength=ctx.wavelength)
    results = map_workers(
        _anim_frame, items, jobs=jobs, shared=shared, log=log,
        on_done=lambda done, total, _r: (log.info(f"  {done}/{total} propagated")
                                         if done % 10 == 0 else None),
        desc=f"Propagating {len(items)} candidate frame(s) over {z_planes} z-planes")

    frames, frame_items = [], []
    vmax = 0.0
    for it, (f, reason) in zip(items, results):
        if f is None:
            log.warning(f"  candidate #{it.index}: skipped ({reason})")
            continue
        frames.append(f); frame_items.append(it); vmax = max(vmax, float(f.max()))
    if not frames:
        log.warning("No frames to animate")
        return None
    vmax = vmax or 1.0

    dpi = 100
    fig, ax = plt.subplots(figsize=(9.6, 9.6), dpi=dpi, layout="constrained")
    im = ax.imshow(frames[0], extent=extent, origin="lower", aspect="auto",
                   cmap="inferno", vmin=0.0, vmax=vmax)
    fig.colorbar(im, ax=ax, label="|field| (V/m)")
    ax.scatter(ctx.rx_aper_axis, np.full(len(ctx.rx_aper_axis), z_min), s=6, c="red",
               label="RX aperture", zorder=5)
    tx_scatter = ax.scatter(frame_items[0].aper_axis,
                            np.full(len(frame_items[0].aper_axis), frame_items[0].z),
                            s=6, c="blue", label="TX aperture", zorder=5)
    ax.set_xlabel("x (m)"); ax.set_ylabel("z (m)")
    ax.legend(loc="upper right", framealpha=0.9, markerscale=2)

    def update(k):
        it = frame_items[k]
        im.set_data(frames[k])
        tx_scatter.set_offsets(np.column_stack([it.aper_axis,
                                                np.full(len(it.aper_axis), it.z)]))
        note = "" if it.stop_reason == "converged" else f" [{it.stop_reason}]"
        ax.set_title(f"Candidate #{it.index} — MSE={it.final_loss:.4g}{note}")
        return [im, tx_scatter]

    anim = FuncAnimation(fig, update, frames=len(frames), blit=False)
    return write_mp4(anim, fig, out_path, fps, dpi, show, log)
