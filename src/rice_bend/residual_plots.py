"""The candidate-residual view of a sweep: the (z, x_center) grid of GS residuals,
and the six plots drawn from it.

A lower residual means a better fit to the measurement, which is the whole premise
of the search -- so these are the figures that say whether it worked.

Every plot takes `hc: bool = False` and builds its own colour norm. The
high-contrast variant is a log scale whose floor adapts to that plot's own data
(rounded down to a decade, ceiling fixed at 0.06), so each populated decade gets an
equal share of the colormap and the lowest residuals spread apart. Because the floor
is data-driven, hc colours are NOT comparable across runs; the linear [0, 0.06]
plots are the comparable view.
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LogNorm, Normalize, SymLogNorm
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers the 3d projection)

from rice_bend.config import AxisSweep
from rice_bend.grid_sweep import GridSearchRun

@dataclass
class ResidualSummary:
    """The (z, x_center) grid of GS residuals, plus ground truth and the best point."""
    z_values: np.ndarray
    x_values: np.ndarray
    loss_grid: np.ndarray              # shape (nz, nx); NaN where not run / skipped
    real_tx_z: float
    real_tx_x_center: float
    best: Optional[dict]               # {index, z, x_center, final_loss} or None


def _assemble_summary(zs: List[float], xs: List[float],
                      cand_index_loss: List[Tuple[int, float]],
                      real_z: float, real_x_center: float) -> ResidualSummary:
    """Place each candidate's residual on the grid using its enumeration index.

    enumerate_grid assigns index = z_idx * nx + x_idx (z outer, x inner), so the
    grid position is recovered exactly without float matching.
    """
    nz, nx = len(zs), len(xs)
    grid = np.full((nz, nx), np.nan, dtype=float)
    best: Optional[dict] = None
    for index, loss in cand_index_loss:
        z_idx, x_idx = divmod(index, nx)
        if 0 <= z_idx < nz and 0 <= x_idx < nx:
            grid[z_idx, x_idx] = loss
            if best is None or loss < best["final_loss"]:
                best = {"index": index, "z": float(zs[z_idx]),
                        "x_center": float(xs[x_idx]), "final_loss": float(loss)}
    return ResidualSummary(np.asarray(zs, float), np.asarray(xs, float), grid,
                           float(real_z), float(real_x_center), best)


def summary_from_run(run: GridSearchRun) -> ResidualSummary:
    """Build a ResidualSummary from an in-memory grid run."""
    cand = [(c.point.index, c.final_loss) for c in run.candidates]
    real_x_center = 0.5 * (run.real_tx_x_min + run.real_tx_x_max)
    return _assemble_summary(run.grid_cfg.z.values(), run.grid_cfg.x_center.values(),
                             cand, run.real_tx_z, real_x_center)


def summary_from_manifest(run_dir: Path) -> ResidualSummary:
    """Rebuild a ResidualSummary from a saved run's candidate_beams.json (no npz needed).

    Works on both schemas: `final_loss` is the joint (mean-over-frequency)
    residual in a schema-3 run and the single frequency's residual in a legacy
    schema-2 run — either way it is THE residual that ranks candidates."""
    with open(Path(run_dir) / "candidate_beams.json") as f:
        manifest = json.load(f)
    spec = manifest["grid_spec"]
    zs = AxisSweep(**spec["z"]).values()
    xs = AxisSweep(**spec["x_center"]).values()
    cand = [(c["index"], c["final_loss"]) for c in manifest["candidates"]]
    gt = manifest["ground_truth"]
    real_x_center = 0.5 * (gt["real_tx_x_min"] + gt["real_tx_x_max"])
    return _assemble_summary(zs, xs, cand, gt["real_tx_z"], real_x_center)


def freq_summaries_from_manifest(run_dir: Path) -> List[Tuple[float, ResidualSummary]]:
    """Per-frequency component summaries of a schema-3 joint run.

    One ResidualSummary per frequency, built from each candidate's
    per_freq_losses (a null component means the candidate failed that
    frequency's RS sampling check, so its cell stays NaN and the plots mask it).
    Returns [] for a schema-2 manifest, which carries no components.
    """
    with open(Path(run_dir) / "candidate_beams.json") as f:
        manifest = json.load(f)
    if "frequencies" not in manifest:
        return []
    spec = manifest["grid_spec"]
    zs = AxisSweep(**spec["z"]).values()
    xs = AxisSweep(**spec["x_center"]).values()
    gt = manifest["ground_truth"]
    real_x_center = 0.5 * (gt["real_tx_x_min"] + gt["real_tx_x_max"])
    out: List[Tuple[float, ResidualSummary]] = []
    for i, entry in enumerate(manifest["frequencies"]):
        cand = [(c["index"], c["per_freq_losses"][i]) for c in manifest["candidates"]
                if c["per_freq_losses"][i] is not None]
        out.append((float(entry["freq_hz"]),
                    _assemble_summary(zs, xs, cand, gt["real_tx_z"], real_x_center)))
    return out


def freq_summaries_from_run(run: GridSearchRun) -> List[Tuple[float, ResidualSummary]]:
    """freq_summaries_from_manifest's in-memory twin (NaN components skipped)."""
    zs = run.grid_cfg.z.values()
    xs = run.grid_cfg.x_center.values()
    real_x_center = 0.5 * (run.real_tx_x_min + run.real_tx_x_max)
    out: List[Tuple[float, ResidualSummary]] = []
    for i, freq in enumerate(run.freqs):
        cand = [(c.point.index, float(c.per_freq_losses[i])) for c in run.candidates
                if np.isfinite(c.per_freq_losses[i])]
        out.append((float(freq),
                    _assemble_summary(zs, xs, cand, run.real_tx_z, real_x_center)))
    return out


def _hc_norm(losses: np.ndarray) -> LogNorm:
    """Fresh high-contrast colour norm: log scale with a data-driven floor.

    vmin is the data's own minimum finite loss rounded down to the nearest decade
    (capped so at least one decade sits below the fixed 0.06 ceiling), so every
    populated decade gets colormap share and the lowest (best-fit) residuals in
    each run differentiate maximally. Because the floor adapts per plot, hc colours
    are NOT comparable across runs — the linear [0, 0.06] plots are the comparable
    view. A new instance per plot — norms are stateful.
    """
    finite = np.asarray(losses)[np.isfinite(losses)]
    lo = float(finite.min()) if finite.size else 1e-5
    vmin = 10.0 ** np.floor(np.log10(max(lo, 1e-12)))
    vmin = min(vmin, 6e-3)               # keep >= 1 decade of range below the ceiling
    return LogNorm(vmin=vmin, vmax=0.06)


def _candidate_points(z_values: np.ndarray, x_values: np.ndarray, grid: np.ndarray,
                      real_z: float, real_x_center: float):
    """Flatten a (z, x_center) grid into per-candidate arrays.

    Returns (z, x_center, value, dist) for every cell that was actually run (finite
    value). `dist` is the Euclidean distance in the (z, x_center) plane from the
    candidate to the true TX; z and x_center share units (metres), so it is
    physically meaningful. enumerate_grid uses z-outer/x-inner ordering, which is
    exactly meshgrid's "ij" indexing, so cells map back without float matching.

    Takes arrays rather than a ResidualSummary because the difference plots pass a
    grid of loss DIFFERENCES. Handing it a summary forced them to fabricate one
    carrying a difference grid and best=None, which meant a reader had to know that
    a ResidualSummary sometimes holds losses and sometimes holds something else.
    """
    zz, xx = np.meshgrid(z_values, x_values, indexing="ij")
    finite = np.isfinite(grid)
    z = zz[finite]
    x = xx[finite]
    value = grid[finite]
    dist = np.hypot(z - real_z, x - real_x_center)
    return z, x, value, dist
# --- shared plot vocabulary --------------------------------------------------
CMAP = "viridis_r"                 # low residual = bright yellow
RESIDUAL_VMAX = 0.06               # fixed residual ceiling -> comparable across runs
RESIDUAL_LABEL = "GS residual (lower = better fit)"
HC_SUFFIX = " (high contrast, log scale)"
DOT_MIN, DOT_MAX = 1.5, 60.0       # 3D dot-size range

# --- residual surface ---------------------------------------------------------
SURFACE_VIEW = (28, -55)           # elev, azim for the still; the orbit mp4 escapes occlusion
SURFACE_FLOOR_FRAC = 0.18          # flat-map plane sits this far below the surface's base
SURFACE_HEAD_FRAC = 0.28           # headroom above the tallest peak for the marker poles
SURFACE_LEVELS = 32                # contour bands in the floor projection
SURFACE_ORBIT_FRAMES = 72          # 5 degrees per frame
SURFACE_ORBIT_FPS = 15


def _residual_norm(values: np.ndarray, hc: bool) -> Normalize:
    """The colour/axis norm for a residual plot: fixed [0, 0.06] normally, or the
    data-driven log scale when hc.

    Every plot function calls this rather than being handed a norm. Passing norms in
    leaked the choice upward -- callers had to know to np.stack a multi-frequency
    list before calling _hc_norm, and one of them reached into _hc_norm(loss).vmin
    to recover a floor it was never handed.
    """
    return _hc_norm(values) if hc else Normalize(vmin=0.0, vmax=RESIDUAL_VMAX)


def _residual_mesh(ax, x_values: np.ndarray, z_values: np.ndarray,
                   grid: np.ndarray, norm: Normalize):
    """pcolormesh one (z, x_center) grid, masking the cells that were never run.

    A LogNorm cannot take zeros, so values are clipped up to the norm's own floor;
    NaNs survive np.maximum and stay masked. Four call sites spelled this out, two
    of them writing the log clip differently to mean the same thing.
    """
    if isinstance(norm, LogNorm):
        grid = np.maximum(grid, norm.vmin)
    mesh_x, mesh_z = np.meshgrid(x_values, z_values)
    return ax.pcolormesh(mesh_x, mesh_z, np.ma.masked_invalid(grid),
                         shading="nearest", cmap=CMAP, norm=norm)


def _mark_true_tx(ax, summary: "ResidualSummary", size: float = 170) -> None:
    ax.scatter([summary.real_tx_x_center], [summary.real_tx_z], marker="X", s=size,
               c="red", edgecolor="white", linewidth=1.5, label="True TX location",
               zorder=5)


def _hc_title(title: str, hc: bool) -> str:
    return title + HC_SUFFIX if hc else title


def plot_residual_heatmap(summary: ResidualSummary, out_path: Path,
                          title: str = "Candidate residual over speculative TX locations",
                          hc: bool = False) -> None:
    """The residual over (z, x_center), with the true TX and the best candidate marked.

    Lower residual = better fit to the measurement = more likely TX location, so this
    is the figure the whole search exists to produce.
    """
    fig, ax = plt.subplots(figsize=(9, 6), layout="constrained")
    norm = _residual_norm(summary.loss_grid, hc)
    mesh = _residual_mesh(ax, summary.x_values, summary.z_values, summary.loss_grid, norm)
    fig.colorbar(mesh, ax=ax, label=RESIDUAL_LABEL)

    _mark_true_tx(ax, summary)
    if summary.best is not None:
        ax.scatter([summary.best["x_center"]], [summary.best["z"]], marker="*", s=260,
                   c="lime", edgecolor="black", linewidth=1.0, zorder=6,
                   label=f"Best candidate (loss {summary.best['final_loss']:.4g})")
    ax.set_xlabel("x_center (m)")
    ax.set_ylabel("z (m)")
    ax.set_title(_hc_title(title, hc))
    ax.legend(loc="upper right", framealpha=0.9)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def _surface_height(values: np.ndarray, norm: Normalize) -> np.ndarray:
    """Turn residuals into surface heights: low residual -> physically high.

    The floor clip mirrors _residual_mesh's `np.maximum(grid, norm.vmin)` -- a LogNorm
    cannot take a zero -- and the ceiling clip keeps height >= 0, matching the way the
    2D heatmap saturates above RESIDUAL_VMAX. NaN survives np.clip, so cells that never
    ran stay NaN and plot_surface leaves them as holes.

    The two branches are the same inversion the 2D pair already draws: linear under
    Normalize, decades under LogNorm. That is what makes residual_surface.png the
    relief of residual_heatmap.png rather than a third, unrelated scaling.
    """
    lo = norm.vmin if isinstance(norm, LogNorm) else 0.0
    clipped = np.clip(values, lo, RESIDUAL_VMAX)
    if isinstance(norm, LogNorm):
        return np.log10(RESIDUAL_VMAX / clipped)   # decades below the 0.06 ceiling
    return RESIDUAL_VMAX - clipped                 # residual units, inverted


def _surface_zticks(norm: Normalize) -> Tuple[np.ndarray, List[str]]:
    """(positions, labels) for the height axis, labelled in REAL residual values.

    The axis is a transform of the residual, so labelling it with the transform's own
    output would make the reader do the inversion in their head. Instead the ticks sit
    at the height each round residual maps to, and read as that residual.

    Decade ticks are built from integer exponents rather than from vmin directly:
    _hc_norm's floor can be 6e-3 (its cap), whose log is not an integer, and rounding
    that for a 10^n label would print 10^-2 next to a tick that is not at 0.01.
    """
    if isinstance(norm, LogNorm):
        exps = np.arange(int(np.ceil(np.log10(norm.vmin))),
                         int(np.floor(np.log10(RESIDUAL_VMAX))) + 1)
        decades = 10.0 ** exps.astype(float)
        return (np.log10(RESIDUAL_VMAX / decades),
                [f"$10^{{{int(e)}}}$" for e in exps])
    ticks = np.arange(0.0, RESIDUAL_VMAX + 1e-9, 0.01)
    return RESIDUAL_VMAX - ticks, [f"{t:g}" for t in ticks]


def _surface_pole(ax, x: float, z: float, floor: float, ceiling: float, *, marker: str,
                  color: str, edgecolor: str, size: float, label: str) -> None:
    """A full-height marker pole at (x, z), capped above the tallest peak.

    The cap sits at the ceiling and NOT at the surface height on purpose: mplot3d
    depth-sorts whole collections, so a marker drawn on the surface gets painted under
    it -- placing it at the surface hid the true-TX X behind the peak, and moving it
    down to the floor plane hid it under the surface instead. Nothing can occlude a
    marker above every point of the surface, at any azimuth, which is also what keeps
    both markers readable through every frame of the orbit animation.
    """
    ax.plot([x, x], [z, z], [floor, ceiling], color=color, linewidth=1.3, alpha=0.85)
    ax.scatter([x], [z], [ceiling], marker=marker, s=size, c=color, edgecolor=edgecolor,
               linewidth=1.0, depthshade=False, label=label)


def _draw_surface(ax, summary: ResidualSummary, hc: bool) -> Normalize:
    """Draw the residual surface, its floor projection and the marker poles onto `ax`.

    Returns the norm so the caller can hang a colorbar on it. Split out from
    plot_residual_surface because the orbit animation needs the identical scene.
    """
    grid = summary.loss_grid
    norm = _residual_norm(grid, hc)
    lo = norm.vmin if isinstance(norm, LogNorm) else 0.0
    clipped = np.clip(grid, lo, RESIDUAL_VMAX)
    height = _surface_height(grid, norm)

    mesh_x, mesh_z = np.meshgrid(summary.x_values, summary.z_values)
    # facecolors, not cmap=: plot_surface's cmap colours by HEIGHT, which would put
    # this figure's colorbar on a different scale from the 2D heatmap's. Routing the
    # colour through _residual_norm instead makes the two colorbars identical, so the
    # flat and relief views of one run are directly comparable. shade=False for the
    # same reason -- lighting would alter the mapped colours.
    # edgecolor traces the real candidate sampling, so the surface reads as sampled
    # data rather than as a smooth analytic function.
    ax.plot_surface(mesh_x, mesh_z, height,
                    facecolors=plt.get_cmap(CMAP)(norm(clipped)), shade=False,
                    rcount=grid.shape[0], ccount=grid.shape[1],
                    edgecolor=(0, 0, 0, 0.22), linewidth=0.15)

    h_max = float(np.nanmax(height))
    if not np.isfinite(h_max) or h_max <= 0.0:
        h_max = 1.0            # every candidate at or above the ceiling: keep zlim sane
    floor = -SURFACE_FLOOR_FRAC * h_max
    ceiling = h_max * (1.0 + SURFACE_HEAD_FRAC)

    levels = (np.logspace(np.log10(norm.vmin), np.log10(RESIDUAL_VMAX), SURFACE_LEVELS)
              if isinstance(norm, LogNorm)
              else np.linspace(0.0, RESIDUAL_VMAX, SURFACE_LEVELS))
    ax.contourf(mesh_x, mesh_z, np.ma.masked_invalid(clipped), levels=levels,
                zdir="z", offset=floor, cmap=CMAP, norm=norm)

    _surface_pole(ax, summary.real_tx_x_center, summary.real_tx_z, floor, ceiling,
                  marker="X", color="red", edgecolor="white", size=130,
                  label="True TX location")
    if summary.best is not None:
        _surface_pole(ax, summary.best["x_center"], summary.best["z"], floor, ceiling,
                      marker="*", color="lime", edgecolor="black", size=260,
                      label=f"Best candidate (loss {summary.best['final_loss']:.4g})")

    ax.set_zlim(floor, ceiling)
    positions, labels = _surface_zticks(norm)
    ax.set_zticks(positions)
    ax.set_zticklabels(labels)
    ax.set_xlabel("x_center (m)")
    ax.set_ylabel("z (m)")
    ax.set_zlabel("GS residual (inverted)")
    return norm


def _finish_surface(fig, ax, norm: Normalize, title: str) -> None:
    """Title, colorbar, legend, footnote and viewpoint — shared by the still and the orbit.

    layout="constrained" is deliberately NOT used on these figures: with a 3D axes plus
    this colorbar it collapses the axes to zero size and warns.
    """
    ax.set_title(title)
    fig.colorbar(ScalarMappable(norm=norm, cmap=CMAP), ax=ax, shrink=0.6, pad=0.10,
                 label=RESIDUAL_LABEL)
    ax.legend(loc="upper left")
    fig.text(0.02, 0.015, "height = residual, inverted (higher = better fit); "
                          "flat heatmap projected on the floor", fontsize=8, alpha=0.7)
    ax.view_init(elev=SURFACE_VIEW[0], azim=SURFACE_VIEW[1])


def _surface_placeholder(out_path: Path, title: str, message: str) -> None:
    """Stand-in figure for a grid that cannot be a surface, so the file always exists.

    Matches plot_residual_scatter's no-candidates behaviour: a run directory's plot set
    should not depend on whether the sweep happened to be degenerate.
    """
    fig, ax = plt.subplots(figsize=(9, 6), layout="constrained")
    ax.text(0.5, 0.5, message, ha="center", va="center", transform=ax.transAxes)
    ax.set_axis_off()
    ax.set_title(title)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def _surface_unplottable(grid: np.ndarray) -> Optional[str]:
    """Why this grid cannot be drawn as a surface, or None if it can."""
    if not np.isfinite(grid).any():
        return "no candidates"
    if min(grid.shape) < 2:
        return (f"grid is {grid.shape[0]}x{grid.shape[1]} — "
                "a surface needs at least 2 points on each axis")
    return None


def plot_residual_surface(summary: ResidualSummary, out_path: Path,
                          title: str = "Candidate residual surface over speculative TX locations",
                          hc: bool = False) -> None:
    """The residual heatmap as relief: the lower the residual, the higher the surface.

    The flat heatmap saturates -- on a dense caustic sweep over half the cells sit
    within 1% of the maximum -- so the depth of the basin, which is the thing the
    search is actually measuring, is invisible there. Inverting it into height makes
    the best fit a peak standing over the TX.

    Same norms as plot_residual_heatmap, so the pair reads as one figure in two views:
    the linear variant shows the shape of the whole basin, the hc variant resolves the
    decades near the bottom of it into a single sharp summit.
    """
    reason = _surface_unplottable(summary.loss_grid)
    if reason is not None:
        _surface_placeholder(out_path, _hc_title(title, hc), reason)
        return
    fig = plt.figure(figsize=(11, 8))
    ax = fig.add_subplot(projection="3d")
    norm = _draw_surface(ax, summary, hc)
    _finish_surface(fig, ax, norm, _hc_title(title, hc))
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def animate_residual_surface(summary: ResidualSummary, log, out_path: Path, *,
                             title: str = "Candidate residual surface over speculative TX locations",
                             hc: bool = False) -> Optional[Path]:
    """Orbit the residual surface through a full turn and write it as .mp4.

    A still is fixed to one azimuth, and which azimuth reads best depends on where the
    basin lands -- so this is the escape hatch when SURFACE_VIEW happens to hide the
    peak behind a ridge for a given scenario. Returns None for a degenerate grid, which
    has no surface to orbit.

    `log` precedes `out_path` because _emit_pair supplies out_path and hc as KEYWORDS
    and everything else positionally; putting out_path second would collide with it.
    """
    from matplotlib.animation import FuncAnimation

    from rice_bend.data_store import write_mp4

    if _surface_unplottable(summary.loss_grid) is not None:
        return None
    fig = plt.figure(figsize=(11, 8))
    ax = fig.add_subplot(projection="3d")
    norm = _draw_surface(ax, summary, hc)
    _finish_surface(fig, ax, norm, _hc_title(title, hc))
    # No bbox_inches="tight" on this path -- FFMpegWriter needs every frame the same
    # size, and a tight box is recomputed per frame. Set the margins once instead.
    fig.subplots_adjust(left=0.02, right=0.90, top=0.94, bottom=0.04)

    elev, azim0 = SURFACE_VIEW

    def _frame(i: int):
        ax.view_init(elev=elev, azim=azim0 + i * 360.0 / SURFACE_ORBIT_FRAMES)
        return ()

    anim = FuncAnimation(fig, _frame, frames=SURFACE_ORBIT_FRAMES, blit=False)
    return write_mp4(anim, fig, out_path, SURFACE_ORBIT_FPS, 120, False, log)


def plot_residual_scatter(summary: ResidualSummary, out_path: Path, *,
                          title: str = "Candidate residual distribution",
                          hc: bool = False) -> None:
    """Every candidate's residual against its distance to the true TX.

    A rising trend is what validates the search's premise: that a lower GS residual
    marks a candidate closer to the real transmitter.
    """
    _, _, loss, dist = _candidate_points(summary.z_values, summary.x_values,
                                         summary.loss_grid, summary.real_tx_z,
                                         summary.real_tx_x_center)
    fig, ax = plt.subplots(figsize=(9, 6), layout="constrained")

    if loss.size == 0:
        ax.text(0.5, 0.5, "no candidates", ha="center", va="center",
                transform=ax.transAxes)
        fig.suptitle(_hc_title(title, hc))
        fig.savefig(out_path, dpi=120)
        plt.close(fig)
        return

    norm = _residual_norm(loss, hc)
    if hc:
        loss = np.maximum(loss, norm.vmin)   # keep sub-floor losses on the log axis
    ax.scatter(dist, loss, s=40, color="C0", edgecolor="black", linewidth=0.3, zorder=3)
    ax.set_xlabel("distance from candidate to true TX (m)")
    ax.set_ylabel("GS residual")
    if hc:
        ax.set_yscale("log")
    ax.set_ylim(norm.vmin, RESIDUAL_VMAX)
    ax.set_title("Residual vs. distance to true TX")
    ax.grid(True, alpha=0.3)

    fig.suptitle(_hc_title(title, hc))
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def _scatter_3d(layers, out_path: Path, *, norm: Normalize, dot_size, title: str,
                colorbar_label: str, footnote: str) -> None:
    """The 3D residual scatter: (x_center, z) flat on the bottom, frequency rising,
    one layer of candidate dots per frequency, coloured by `layers`' scalar.

    `layers` is [(freq_hz, summary, values_grid)] and `dot_size(values) -> sizes`.
    Those two, the norm and the labels are the ENTIRE difference between the
    absolute-residual view and the difference-from-baseline view; everything below
    -- the loop, the red X per layer, the dashed guide, the axis labels, view_init,
    the colorbar and the proxy legend handle -- was line-for-line identical in two
    62-line functions.
    """
    fig = plt.figure(figsize=(11, 8), layout="constrained")
    ax = fig.add_subplot(projection="3d")

    freqs_ghz = sorted(f / 1e9 for f, _, _ in layers)
    real_x = real_z = None
    for freq_hz, summary, values in layers:
        f_ghz = freq_hz / 1e9
        z, x, val, _ = _candidate_points(summary.z_values, summary.x_values, values,
                                         summary.real_tx_z, summary.real_tx_x_center)
        if val.size:
            # Clip colours up to the norm's floor so a LogNorm never sees zero
            # (a no-op under a linear norm whose vmin is at or below the data).
            ax.scatter(x, z, np.full_like(x, f_ghz), c=np.maximum(val, norm.vmin),
                       cmap=CMAP, norm=norm, s=dot_size(val), depthshade=False,
                       edgecolor="none")
        real_x, real_z = summary.real_tx_x_center, summary.real_tx_z
        ax.scatter([summary.real_tx_x_center], [summary.real_tx_z], [f_ghz], marker="X",
                   s=90, c="red", edgecolor="white", linewidth=1.0, depthshade=False,
                   zorder=6)

    # vertical guide connecting the true-TX markers up the frequency axis
    if real_x is not None and len(freqs_ghz) > 1:
        ax.plot([real_x, real_x], [real_z, real_z], [min(freqs_ghz), max(freqs_ghz)],
                color="red", linestyle="--", linewidth=1.0, alpha=0.6)

    ax.set_xlabel("x_center (m)")
    ax.set_ylabel("z (m)")
    ax.set_zlabel("frequency (GHz)")
    ax.set_zticks(freqs_ghz)
    ax.set_title(title)
    ax.text2D(0.02, 0.02, footnote, transform=ax.transAxes, fontsize=8, alpha=0.7)
    ax.view_init(elev=22, azim=-60)

    fig.colorbar(ScalarMappable(norm=norm, cmap=CMAP), ax=ax, shrink=0.6, pad=0.1,
                 label=colorbar_label)
    # a proxy handle, so the legend documents the red X without one entry per layer
    ax.scatter([], [], [], marker="X", s=90, c="red", edgecolor="white",
               linewidth=1.0, label="True TX location")
    ax.legend(loc="upper left")
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_residual_scatter_3d(summaries: List[Tuple[float, ResidualSummary]],
                             out_path: Path, *,
                             title: str = "Candidate residual across frequency",
                             hc: bool = False) -> None:
    """The multi-frequency view of the 2D residual heatmap.

    Dot size falls off cubically as the residual grows, so only genuinely
    low-residual candidates stay large and the high-residual bulk shrinks to dots
    the eye can see through.
    """
    norm = _residual_norm(np.stack([s.loss_grid for _, s in summaries]), hc)

    def dot_size(loss):
        return DOT_MIN + ((1.0 - np.clip(loss / RESIDUAL_VMAX, 0.0, 1.0)) ** 3) * (DOT_MAX - DOT_MIN)

    _scatter_3d([(f, s, s.loss_grid) for f, s in summaries], out_path, norm=norm,
                dot_size=dot_size, title=_hc_title(title, hc),
                colorbar_label=RESIDUAL_LABEL,
                footnote="dot size shrinks cubically as residual grows")


def plot_residual_scatter_3d_diff(summaries: List[Tuple[float, ResidualSummary]],
                                  out_path: Path, *,
                                  baseline_freq: Optional[float] = None) -> None:
    """The same 3D scatter, but of residual DIFFERENCES from a baseline frequency.

    The baseline defaults to the centre frequency. Bright yellow fits BETTER than the
    baseline, dark purple worse, mid-teal unchanged (the baseline layer is uniformly
    zero). Dot size grows with the deviation, so candidates that behave like the
    baseline stay see-through.
    """
    ordered = sorted(summaries, key=lambda t: t[0])
    freqs = [f for f, _ in ordered]
    if baseline_freq is None:
        baseline_freq = freqs[len(freqs) // 2]
    base_grid = next(s for f, s in ordered if f == baseline_freq).loss_grid

    layers = [(f, s, s.loss_grid - base_grid) for f, s in ordered]
    finite = np.concatenate([d[np.isfinite(d)].ravel() for _, _, d in layers])
    vlim = max(float(np.abs(finite).max()) if finite.size else 1e-6, 1e-12)
    baseline_ghz = baseline_freq / 1e9

    def dot_size(diff):
        return DOT_MIN + (np.abs(diff) / vlim) * (DOT_MAX - DOT_MIN)

    _scatter_3d(layers, out_path, norm=Normalize(vmin=-vlim, vmax=vlim),
                dot_size=dot_size,
                title=f"Candidate residual difference vs. {baseline_ghz:g} GHz baseline",
                colorbar_label=f"residual difference vs. {baseline_ghz:g} GHz "
                               "(yellow = better than baseline)",
                footnote="dot size grows with deviation from the baseline")


def plot_residual_freq_vs_joint(freq_hz: float, summary: ResidualSummary,
                                joint: ResidualSummary, out_path: Path, *,
                                hc: bool = False) -> None:
    """One frequency's residual component against the joint (mean) residual.

    Three panels: this frequency's component, the joint residual the solver
    actually minimized (shared scale), and their difference on a symmetric scale
    — bright yellow where this frequency fits BETTER than the joint, dark purple
    where worse. Cells where the frequency was invalid stay masked and
    NaN-propagate through the difference.
    """
    f_ghz = freq_hz / 1e9
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), layout="constrained")
    # one norm shared by both heatmap panels so they stay inter-comparable
    norm = _residual_norm(np.stack([summary.loss_grid, joint.loss_grid]), hc)

    mesh = None
    for ax, s, sub_title in ((axes[0], summary, f"{f_ghz:g} GHz"),
                             (axes[1], joint, "joint (mean across frequencies)")):
        mesh = _residual_mesh(ax, joint.x_values, joint.z_values, s.loss_grid, norm)
        ax.set_title(sub_title)
    fig.colorbar(mesh, ax=list(axes[:2]), label=RESIDUAL_LABEL, shrink=0.9)

    diff = summary.loss_grid - joint.loss_grid
    vlim = max(float(np.nanmax(np.abs(diff))) if np.isfinite(diff).any() else 1e-6, 1e-12)
    if hc:
        # symmetric log: two decades either side of a linear core, so small
        # deviations from the joint spread instead of washing out at teal
        dnorm = SymLogNorm(linthresh=vlim / 100.0, vmin=-vlim, vmax=vlim, base=10)
    else:
        dnorm = Normalize(vmin=-vlim, vmax=vlim)
    dmesh = _residual_mesh(axes[2], joint.x_values, joint.z_values, diff, dnorm)
    axes[2].set_title(f"difference ({f_ghz:g} GHz − joint)")
    fig.colorbar(dmesh, ax=axes[2],
                 label="residual difference (yellow = better than joint)", shrink=0.9)

    for ax in axes:
        _mark_true_tx(ax, joint, size=120)
        ax.set_xlabel("x_center (m)")
    axes[0].set_ylabel("z (m)")
    axes[0].legend(loc="upper right", framealpha=0.9)
    fig.suptitle(_hc_title(f"Residual: {f_ghz:g} GHz vs. joint", hc))
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
