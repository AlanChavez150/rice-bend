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

from rice_bend import rs
from rice_bend.config import center_freq_index
from rice_bend.data_store import load_run_config, write_mp4
from rice_bend.grid_sweep import GridSearchRun
from rice_bend.mgs import MGS
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
    """The manifest's frequency list — schema 3 (`frequencies`) or the schema-2
    scalar `freq_hz` as a length-1 list."""
    if "frequencies" in manifest:
        return [float(e["freq_hz"]) for e in manifest["frequencies"]]
    return [float(manifest["freq_hz"])]


def make_true_mgs_plot(run_dir: Path, log: Optional[logging.Logger] = None,
                       scene_freq: Optional[float] = None) -> Optional[Path]:
    """Reconstruct + plot the single "true" MGS run for a saved grid run.

    The grid search treats the TX as unknown; this is the baseline plain-`mgs` result at
    the KNOWN (true) TX location. Using the run's saved config + frequency list, it
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

    # Pin the phase model to what THIS sweep actually ran (absent in pre-field
    # manifests means achromatic — historically accurate, they all were). Without
    # this, replotting an old achromatic run under the delay default would produce
    # a baseline that silently disagrees with the manifest's residuals.
    ran_model = manifest.get("gs", {}).get("phase_model", "achromatic")
    if config.gerchberg_saxton.phase_model != ran_model:
        log.info(f"true MGS run: pinning phase_model to {ran_model!r} (what this "
                 "sweep ran) over the config's default")
        config.gerchberg_saxton.phase_model = ran_model

    # Pin the GS seed so the baseline is reproducible across --replot calls. Plain mgs
    # leaves gerchberg_saxton.seed null -> a fresh random initial phase each run, which
    # would make true_mgs_scene.png differ every time. Reuse the run's recorded sweep
    # seed when available, else 0.
    if config.gerchberg_saxton.seed is None:
        pinned = manifest.get("seed")
        config.gerchberg_saxton.seed = 0 if pinned is None else int(pinned)
        log.info(f"true MGS run: gerchberg_saxton.seed was null; pinned to "
                 f"{config.gerchberg_saxton.seed} for a reproducible baseline")

    out_path = run_dir / "true_mgs_scene.png"
    ghz = ", ".join(f"{f / 1e9:.3g}" for f in freqs)
    log.info(f"Reconstructing the true MGS run (TX known) at {ghz} GHz -> {out_path}")
    mgs = MGS(freqs, config)
    mgs.illuminate_real()             # illuminate the real scene
    mgs.run_gerch_sax()               # MGS solve at the true TX plane
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
    """Everything needed to re-illuminate candidate beams into scene fields."""
    items: List[CandidateScene]
    scene_bounds: Tuple[float, float, float, float]   # x_min, x_max, z_min, z_max
    wavelength: float
    scene_x_axis: np.ndarray
    real_tx_z: float
    real_tx_x_center: float
    real_tx_aper_axis: np.ndarray        # ground-truth TX aperture (for the real beam panel)
    real_tx_aper_profile: np.ndarray
    rx_aper_axis: np.ndarray             # RX aperture x-axis (for the RX dots, at z_min)


def scenes_from_run(run: GridSearchRun,
                    scene_freq: Optional[float] = None) -> SceneContext:
    """Adapter: a SceneContext from an in-memory run. Multi-frequency runs are
    rendered at `scene_freq` (must be one of the run's frequencies) or, by
    default, the centre frequency."""
    items = [CandidateScene(c.point.index, c.point.z, c.point.x_center, c.final_loss,
                            c.aper_axis, c.aper_profile, c.stop_reason)
             for c in run.candidates]
    i = _display_freq_index(run.freqs, scene_freq)
    return SceneContext(items, run.scene_bounds, run.wavelengths[i], run.scene_x_axis,
                        run.real_tx_z, 0.5 * (run.real_tx_x_min + run.real_tx_x_max),
                        run.real_tx_aper_axis, run.real_tx_aper_profiles[i],
                        run.rx_aper_axes[i])


def scenes_from_manifest(run_dir: Path,
                         scene_freq: Optional[float] = None) -> SceneContext:
    """Adapter: a SceneContext from a saved run (reads each candidate npz).

    Handles both layouts: a schema-3 joint run (frequency list, stacked TX
    profiles, indexed RX aperture keys — rendered at `scene_freq`, defaulting to
    the centre frequency) and a schema-2 per-frequency run (scalar keys)."""
    run_dir = Path(run_dir)
    with open(run_dir / "candidate_beams.json") as f:
        m = json.load(f)
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
    if "frequencies" in m:   # schema 3: joint layout
        wavelength = float(m["frequencies"][i]["wavelength_m"])
        tx_profile = meas["real_tx_aper_profile"][i]
        rx_axis = meas[f"rx_aper_axis_{i:02d}"]
    else:                    # schema 2: legacy per-frequency layout (i is 0)
        wavelength = m["wavelength_m"]
        tx_profile = meas["real_tx_aper_profile"]
        rx_axis = meas["rx_aper_axis"]
    return SceneContext(items, scene_bounds, wavelength, meas["scene_x_axis"],
                        gt["real_tx_z"], 0.5 * (gt["real_tx_x_min"] + gt["real_tx_x_max"]),
                        meas["real_tx_aper_axis"], tx_profile, rx_axis)


def _scene_panel(fig, ax, field: np.ndarray, ctx: SceneContext, tx_axis: np.ndarray,
                 tx_z: float, vmax: float, title: str) -> None:
    """plotting.draw_scene bound to a SceneContext.

    The RX plane is passed as scene z_min explicitly. That is an assumption, not a
    fact: it holds for every simulated config (rx.z == 0 == z_min) and the grid
    search only ever runs on simulated scenes, but a SceneContext does not carry the
    RX z, so this is where the assumption lives.

    Legend pinned to "upper right" rather than "best": these panels also become mp4
    frames, and a legend that relocates between frames is worse than one that
    occasionally overlaps.
    """
    draw_scene(fig, ax, field, bounds=ctx.scene_bounds,
               rx_axis=ctx.rx_aper_axis, rx_z=ctx.scene_bounds[2],
               tx_axis=tx_axis, tx_z=tx_z, vmax=vmax, title=title,
               legend_loc="upper right")


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
                         best: bool = False) -> None:
    """4-panel candidate PNG: candidate beam | real beam (shared scale) over candidate
    aperture phase | candidate aperture amplitude."""
    vmax = float(cand_field.max())
    if real_field is not None:
        vmax = max(vmax, float(real_field.max()))
    vmax = vmax or 1.0

    fig, axes = plt.subplots(2, 2, figsize=(14, 12), layout="constrained")
    (ax_cand, ax_real), (ax_phase, ax_amp) = axes

    tag = " (best)" if best else ""
    note = "" if item.stop_reason == "converged" else f" [{item.stop_reason}]"
    _scene_panel(fig, ax_cand, cand_field, ctx, item.aper_axis, item.z, vmax,
                f"Candidate #{item.index}{tag} beam — MSE={item.final_loss:.4g}{note}")

    if real_field is not None:
        _scene_panel(fig, ax_real, real_field, ctx, ctx.real_tx_aper_axis, ctx.real_tx_z,
                    vmax, "Real beam")
    else:
        ax_real.text(0.5, 0.5, "real beam unavailable", ha="center", va="center",
                     transform=ax_real.transAxes)
        ax_real.set_title("Real beam")

    # phase/amplitude on the full scene x-axis; NaN (gap) where the scene extends
    # beyond the candidate aperture (no data there)
    scene_x = np.asarray(ctx.scene_x_axis)
    x_min, x_max = ctx.scene_bounds[0], ctx.scene_bounds[1]
    phase = _on_scene_axis(scene_x, item.aper_axis, np.unwrap(np.angle(item.aper_profile)))
    amp = _on_scene_axis(scene_x, item.aper_axis, np.abs(item.aper_profile))

    draw_line_panel(ax_phase, [(None, scene_x, phase, "C0")],
                    title="Candidate TX phase", xlabel="x (m)",
                    ylabel="phase unwrapped [rad]", xlim=(x_min, x_max))
    draw_line_panel(ax_amp, [(None, scene_x, amp, "C3")],
                    title="Candidate TX amplitude", xlabel="x (m)",
                    ylabel="amplitude (V/m)", xlim=(x_min, x_max), ylim=(0.0, 1.1))

    fig.savefig(out_path, dpi=120)
    plt.close(fig)


# Read-only scene geometry + real beam, handed to the scene tasks once per worker.
SceneShared = namedtuple("SceneShared", "ctx real_field x_axis z_axis best_index scenes_dir")


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
            field = _reilluminate(s.x_axis, s.z_axis, it.aper_axis, it.aper_profile,
                                  s.ctx.wavelength, it.z)
        except RuntimeError as e:
            skipped.append((it.index, str(e)))
            continue
        _plot_candidate_quad(field, s.real_field, it, s.ctx,
                             s.scenes_dir / f"cand_{it.index:04d}.png",
                             best=(it.index == s.best_index))
        acc = field.astype(np.float64) if acc is None else acc + field
        count += 1
        if count % 10 == 0:
            log.info(f"  {count} rendered in this chunk (through #{it.index})")
    return acc, count, skipped


def make_candidate_scenes(ctx: SceneContext, out_dir: Path, *, z_planes: int = SCENE_Z_PLANES,
                          top: Optional[int] = None, average: bool = True, jobs: int = 1,
                          log: Optional[logging.Logger] = None) -> None:
    """Write one 4-panel PNG per candidate beam (scenes/cand_####.png) plus, by default, a
    single PNG averaging every candidate beam (scene_average.png).

    Each candidate PNG shows the candidate beam, the real beam (same color scale), and the
    candidate's reconstructed aperture phase and amplitude; scenes mark the RX aperture
    (blue) and TX aperture (red). With `top` set, only the N lowest-residual candidates are
    rendered, and the average spans exactly those. The average accumulates incrementally.
    """
    log = log or logging.getLogger()
    items = ctx.items
    if not items:
        log.warning("No candidates to render scenes for")
        return
    if top is not None:
        items = sorted(items, key=lambda it: it.final_loss)[:top]
    out_dir = Path(out_dir)
    scenes_dir = out_dir / "scenes"
    scenes_dir.mkdir(parents=True, exist_ok=True)

    _, _, z_min, z_max = ctx.scene_bounds
    x_axis = np.asarray(ctx.scene_x_axis)
    z_axis = np.linspace(z_min, z_max, z_planes)
    best = min(items, key=lambda it: it.final_loss)

    # the real beam is identical for every candidate -> re-illuminate it once
    try:
        real_field = _reilluminate(x_axis, z_axis, ctx.real_tx_aper_axis,
                                   ctx.real_tx_aper_profile, ctx.wavelength, ctx.real_tx_z)
    except RuntimeError as e:
        log.warning(f"real beam re-illumination failed ({e}); real panel will be blank")
        real_field = None

    # Each re-illumination + plot is independent and deterministic, so candidates are
    # distributed across worker processes when jobs > 1 (the per-candidate PNGs are
    # identical to the serial render; only the averaged-scene sum order differs, to
    # float round-off). The running field-sum for the average is reduced from per-worker
    # partial sums.
    shared = SceneShared(ctx=ctx, real_field=real_field, x_axis=x_axis, z_axis=z_axis,
                         best_index=best.index, scenes_dir=scenes_dir)
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
        fig, ax = plt.subplots(figsize=(8, 8), layout="constrained")
        _scene_panel(fig, ax, avg, ctx, ctx.real_tx_aper_axis, ctx.real_tx_z,
                    float(avg.max()) or 1.0,
                    f"Average of {count} candidate beams (TX shown = real)")
        avg_path = out_dir / "scene_average.png"
        fig.savefig(avg_path, dpi=120)
        plt.close(fig)
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
