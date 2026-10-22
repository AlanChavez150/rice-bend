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

The heatmap, scatter and surface (and their hc twins) take an optional `window`
(config.PlotWindowConfig): a VIEW of the grid. Cells whose centre lies outside it
are not drawn and do not set the colour limits; the heatmap shows exactly the window
at equal aspect. The markers (true TX, best candidate, top candidates) still come
from the whole grid and the analysis, and a marker outside the window is named as
such in the legend. window=None draws every plot exactly as before.
"""

import dataclasses
import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cm import ScalarMappable
from matplotlib.collections import LineCollection
from matplotlib.colors import LogNorm, Normalize, SymLogNorm
from matplotlib.patches import Rectangle
from matplotlib.path import Path as MplPath
from matplotlib.transforms import ScaledTranslation
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers the 3d projection)

from rice_bend.analysis import cell_edges
from rice_bend.config import AxisSweep, PlotWindowConfig
from rice_bend.grid_sweep import GridSearchRun
from rice_bend.plotting import add_wavelength_axis

@dataclass
class ResidualSummary:
    """The (z, x_center) grid of GS residuals, plus ground truth and the best point,
    plus the run's centre frequency for the λ axis, its receiver-noise block (for
    the SNR in plot titles) and its excess-power penalty gamma (the residual is
    then the penalized objective, which the titles say)."""
    z_values: np.ndarray
    x_values: np.ndarray
    loss_grid: np.ndarray              # shape (nz, nx); NaN where not run / skipped
    real_tx_z: float
    real_tx_x_center: float
    best: Optional[dict]               # {index, z, x_center, final_loss} or None
    ref_freq_hz: Optional[float] = None  # gs.ref_freq_hz (centre frequency); None on legacy schema-2 runs
    noise: Optional[dict] = None         # the manifest's noise block; None when noiseless or pre-noise
    power_penalty: Optional[float] = None  # gs.power_penalty (gamma); None when off or pre-penalty


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
    summary = _assemble_summary(run.grid_cfg.z.values(), run.grid_cfg.x_center.values(),
                                cand, run.real_tx_z, real_x_center)
    summary.ref_freq_hz = float(run.ref_freq)
    summary.noise = run.noise
    summary.power_penalty = run.power_penalty
    return summary


def summary_from_manifest(run_dir: Path) -> ResidualSummary:
    """Rebuild a ResidualSummary from a saved run's candidate_beams.json (no npz needed).

    Works on every schema: `final_loss` is the joint (mean-over-frequency)
    residual in a joint run (schema 3, or schema 4: the layout the removed
    refinement pass wrote, still read) and the single frequency's residual in a
    legacy schema-2 run — either way it is THE residual that ranks candidates. In
    a schema-4 run it is the loss of the candidate's STORED solve, so nothing here
    needs its extra keys."""
    with open(Path(run_dir) / "candidate_beams.json") as f:
        manifest = json.load(f)
    spec = manifest["grid_spec"]
    zs = AxisSweep(**spec["z"]).values()
    xs = AxisSweep(**spec["x_center"]).values()
    cand = [(c["index"], c["final_loss"]) for c in manifest["candidates"]]
    gt = manifest["ground_truth"]
    real_x_center = 0.5 * (gt["real_tx_x_min"] + gt["real_tx_x_max"])
    summary = _assemble_summary(zs, xs, cand, gt["real_tx_z"], real_x_center)
    summary.ref_freq_hz = (manifest.get("gs") or {}).get("ref_freq_hz")
    summary.noise = manifest.get("noise")
    summary.power_penalty = (manifest.get("gs") or {}).get("power_penalty")
    return summary


def freq_summaries_from_manifest(run_dir: Path) -> List[Tuple[float, ResidualSummary]]:
    """Per-frequency component summaries of a joint run (schema 3, or schema 4
    from the removed refinement pass: its extra keys change nothing this reads,
    and per_freq_losses belongs to the same stored solve as final_loss).

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
        summary = _assemble_summary(zs, xs, cand, gt["real_tx_z"], real_x_center)
        summary.noise = manifest.get("noise")
        out.append((float(entry["freq_hz"]), summary))
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
        summary = _assemble_summary(zs, xs, cand, run.real_tx_z, real_x_center)
        summary.noise = run.noise
        out.append((float(freq), summary))
    return out


@dataclass
class MaxItersMask:
    """Which candidates' solves stopped on the iteration cap instead of converging
    (stop_reason == "max_iters"), on the same (z, x_center) grid as the residual.

    Kept beside the ResidualSummary rather than in it: the summary is the residual
    grid every plot draws (and the surface crops), this is one plot's overlay."""
    capped: np.ndarray                 # shape (nz, nx) bool; False where converged or not run
    n_ran: int                         # candidates solved (the legend's "of M")
    limit: int                         # the sweep's effective max_iters
    resolve_limit: Optional[int] = None  # schema 4: the removed refinement pass's own cap


def _assemble_max_iters(zs: List[float], xs: List[float],
                        cand_index_stop: List[Tuple[int, str]], limit: int,
                        resolve_limit: Optional[int] = None) -> MaxItersMask:
    """Place each candidate's stop reason on the grid by its enumeration index, as
    _assemble_summary places its residual."""
    nz, nx = len(zs), len(xs)
    capped = np.zeros((nz, nx), dtype=bool)
    for index, stop_reason in cand_index_stop:
        z_idx, x_idx = divmod(index, nx)
        if 0 <= z_idx < nz and 0 <= x_idx < nx:
            capped[z_idx, x_idx] = stop_reason == "max_iters"
    return MaxItersMask(capped, len(cand_index_stop), int(limit), resolve_limit)


def max_iters_from_run(run: GridSearchRun) -> MaxItersMask:
    """Build a MaxItersMask from an in-memory grid run."""
    return _assemble_max_iters(run.grid_cfg.z.values(), run.grid_cfg.x_center.values(),
                               [(c.point.index, c.stop_reason) for c in run.candidates],
                               run.effective_max_iters)


def max_iters_from_manifest(run_dir: Path) -> Optional[MaxItersMask]:
    """Rebuild a MaxItersMask from a saved run's candidate_beams.json, or None when
    the manifest does not record one (a candidate without a stop_reason, or no
    gs.effective_max_iters): the caller skips the overlay rather than guess.

    A schema-4 manifest's stop_reason is its STORED solve's, and the removed
    refinement pass re-solved its near-best candidates under refine.max_iters, so
    that cap is carried too when it differs from the sweep's."""
    with open(Path(run_dir) / "candidate_beams.json") as f:
        manifest = json.load(f)
    gs = manifest.get("gs") or {}
    limit = gs.get("effective_max_iters")
    cands = manifest.get("candidates") or []
    if limit is None or any("stop_reason" not in c for c in cands):
        return None
    resolve_limit = (gs.get("refine") or {}).get("max_iters")
    if resolve_limit == limit or not any(c.get("refined") for c in cands):
        resolve_limit = None
    spec = manifest["grid_spec"]
    return _assemble_max_iters(AxisSweep(**spec["z"]).values(),
                               AxisSweep(**spec["x_center"]).values(),
                               [(c["index"], c["stop_reason"]) for c in cands],
                               limit, resolve_limit)


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
# residual_heatmap_maxiters.png: the hatch over the cells whose solve hit max_iters
CAPPED_SPACING = 0.08              # inches between the hatch lines
CAPPED_STROKES = (("black", 1.5), ("white", 0.6))   # (colour, width), drawn in order

# --- residual surface ---------------------------------------------------------
SURFACE_VIEW = (28, -55)           # elev, azim for the still; the orbit mp4 escapes occlusion
SURFACE_FLOOR_FRAC = 0.18          # flat-map plane sits this far below the surface's base
SURFACE_HEAD_FRAC = 0.28           # headroom above the tallest peak for the marker poles
SURFACE_LEVELS = 32                # contour bands in the floor projection
SURFACE_ORBIT_FRAMES = 72          # 5 degrees per frame
SURFACE_ORBIT_FPS = 15

# --- plot window --------------------------------------------------------------
def _window_rows_cols(summary: ResidualSummary,
                      window: PlotWindowConfig) -> Tuple[np.ndarray, np.ndarray]:
    """Boolean masks of the grid's z rows and x_center columns inside the window
    (edges included, as PlotWindowConfig.contains)."""
    tol = window.EDGE_TOL
    z, x = summary.z_values, summary.x_values
    rows = (z >= window.z_min - tol) & (z <= window.z_max + tol)
    cols = (x >= window.x_min - tol) & (x <= window.x_max + tol)
    return rows, cols


def _mask_to_window(summary: ResidualSummary, window: PlotWindowConfig) -> np.ndarray:
    """The loss grid with every cell outside the window set to NaN, which every plot
    treats as a cell that never ran: not drawn, and out of the colour limits."""
    rows, cols = _window_rows_cols(summary, window)
    grid = summary.loss_grid.copy()
    grid[~np.outer(rows, cols)] = np.nan
    return grid


def _crop_to_window(summary: ResidualSummary,
                    window: PlotWindowConfig) -> ResidualSummary:
    """The summary cut down to the window's rows and columns (the surface needs a
    grid, not a mask). Ground truth and best are the whole grid's, unchanged."""
    rows, cols = _window_rows_cols(summary, window)
    return dataclasses.replace(summary, z_values=summary.z_values[rows],
                               x_values=summary.x_values[cols],
                               loss_grid=summary.loss_grid[np.ix_(rows, cols)])


def _outside_note(window: Optional[PlotWindowConfig], x: float, z: float) -> str:
    """A marker's legend suffix: empty inside the window (or with none), else where
    the marker is, so a marker off the plot is named rather than silently lost."""
    if window is None or window.contains(x, z):
        return ""
    return f" — outside plot window, at x={x:+.3f} z={z:.3f} m"


def _hidden_note(summary: ResidualSummary, window: Optional[PlotWindowConfig]) -> str:
    """A title line counting the candidates the window leaves out; empty when it
    leaves none out (or there is no window)."""
    if window is None:
        return ""
    n_all = int(np.isfinite(summary.loss_grid).sum())
    n_in = int(np.isfinite(_mask_to_window(summary, window)).sum())
    return "" if n_in == n_all else f"\n{n_in} of {n_all} candidates inside the plot window"


def _window_figsize(window: PlotWindowConfig) -> Tuple[float, float]:
    """A figure size for an equal-aspect window plot: about 6 in of z, the width that
    makes, plus room for the colorbar, labels and the legend below."""
    ax_w = 6.0 * (window.x_max - window.x_min) / (window.z_max - window.z_min)
    return (min(max(ax_w + 2.8, 6.5), 16.0), 8.6)


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


def _mark_true_tx(ax, summary: "ResidualSummary", size: float = 170,
                  window: Optional[PlotWindowConfig] = None) -> None:
    label = "True TX location" + _outside_note(window, summary.real_tx_x_center,
                                               summary.real_tx_z)
    ax.scatter([summary.real_tx_x_center], [summary.real_tx_z], marker="X", s=size,
               c="red", edgecolor="white", linewidth=1.5, label=label, zorder=5)


def snr_suffix(noise: Optional[dict]) -> str:
    """The title suffix naming a noisy run's SNR (" — SNR 20 dB"); empty when
    noiseless, so a noiseless run's titles (and PNGs) are exactly what they were
    before noise existed."""
    if not noise or noise.get("snr_db") is None:
        return ""
    return f" — SNR {noise['snr_db']:g} dB"


def power_suffix(gamma: Optional[float]) -> str:
    """The title suffix of a run whose residual is the excess-power penalized
    objective (gerchberg_saxton.power_penalty = gamma); empty when the penalty is
    off, so such a run's titles (and PNGs) are exactly what they were before it
    existed."""
    if gamma is None:
        return ""
    return f" — × excess-power factor^{gamma:g}"


def _hc_title(title: str, hc: bool, noise: Optional[dict] = None,
              power_penalty: Optional[float] = None) -> str:
    title = title + snr_suffix(noise) + power_suffix(power_penalty)
    return title + HC_SUFFIX if hc else title


def _overlay_analysis(ax, summary: ResidualSummary, analysis: dict,
                      window: Optional[PlotWindowConfig] = None) -> None:
    """Draw the numeric analysis on a heatmap: solid red outlines on the
    adaptive top-candidates set (the argmin's 8-connected near-tie cluster).
    Pure axes-space geometry over the pcolormesh cells (cell_edges matches
    shading='nearest'), so it is identical on the linear and hc twins.
    Legend entry via a proxy handle, which with a window counts the stored top
    candidates outside it (their outlines fall off the axes)."""
    z_e = cell_edges(summary.z_values)
    x_e = cell_edges(summary.x_values)

    top = analysis.get("top_candidates") or []
    for c in top:
        i, j = c["z_idx"], c["x_idx"]
        ax.add_patch(Rectangle((x_e[j], z_e[i]), x_e[j + 1] - x_e[j],
                               z_e[i + 1] - z_e[i], fill=False, edgecolor="red",
                               linewidth=1.3, zorder=4))
    if top:
        n = analysis.get("n_top_candidates", len(top))
        factor = (analysis.get("top_candidates_criterion") or {}).get("loss_factor")
        crit = f", ≤{factor:g}× min" if factor else ""
        if window is not None:
            n_out = sum(1 for c in top if not window.contains(c["x_center"], c["z"]))
            if n_out:
                crit += f"; {n_out} outside plot window"
        ax.plot([], [], color="red", linewidth=1.3,
                label=f"top candidates (n={n}{crit})")


def _capped_region(summary: ResidualSummary, capped: np.ndarray) -> MplPath:
    """The capped cells as one compound path on the pcolormesh cell borders
    (cell_edges): a rectangle per run of adjacent capped cells along a z row, so a
    151x201 grid is a few thousand rectangles rather than one per cell."""
    z_e = cell_edges(summary.z_values)
    x_e = cell_edges(summary.x_values)
    polys = []
    for i, row in enumerate(capped):
        step = np.diff(np.concatenate([[0], row.astype(np.int8), [0]]))
        for j0, j1 in zip(np.flatnonzero(step == 1), np.flatnonzero(step == -1)):
            polys.append([(x_e[j0], z_e[i]), (x_e[j1], z_e[i]),
                          (x_e[j1], z_e[i + 1]), (x_e[j0], z_e[i + 1])])
    return MplPath.make_compound_path_from_polys(np.asarray(polys, float).reshape(-1, 4, 2))


def _overlay_max_iters(ax, summary: ResidualSummary, mask: MaxItersMask,
                       grid: np.ndarray,
                       window: Optional[PlotWindowConfig] = None) -> None:
    """Hatch the cells whose solve hit max_iters over the residual colours.

    The hatch is drawn by hand rather than with matplotlib's `hatch=`: 45-degree
    lines laid out in inches from the axes' corner (so the spacing is the same on the
    9x6 figure and on the equal-aspect window one), each a white line on a black one
    so it reads on the dark high-residual purple and the bright best-fit yellow
    alike, then clipped to the capped cells -- cell-exact, with the colour between
    the lines left as the hc heatmap draws it. A lone capped cell (3-4 px wide at
    151x201) shows a line only where one crosses it.

    Only cells the heatmap draws are hatched (`grid` is the windowed loss grid, NaN
    where nothing is drawn), so a skipped cell reads exactly as on the hc heatmap.
    The legend counts the whole grid, and with a window names the capped cells
    outside it, as _overlay_analysis does for the top candidates."""
    shown = mask.capped & np.isfinite(grid)
    if shown.any():
        region = _capped_region(summary, shown)
        to_axes = ax.figure.dpi_scale_trans + ScaledTranslation(0, 0, ax.transAxes)
        span = 30.0                      # inches: wider and taller than any figure here
        starts = np.arange(-span, span, CAPPED_SPACING * np.sqrt(2))
        segments = [[(c - 1.0, -1.0), (c + span, span)] for c in starts]
        for color, width in CAPPED_STROKES:
            lines = LineCollection(segments, colors=color, linewidths=width,
                                   transform=to_axes, zorder=3)
            ax.add_collection(lines, autolim=False)
            lines.set_clip_path(region, ax.transData)
            lines.set_clip_box(ax.bbox)
    n = int(mask.capped.sum())
    limit = f"limit {mask.limit}"
    if mask.resolve_limit is not None:
        limit += f", re-solved candidates {mask.resolve_limit}"
    n_out = 0
    if window is not None:
        rows, cols = _window_rows_cols(summary, window)
        n_out = int((mask.capped & ~np.outer(rows, cols)).sum())
    out = f"; {n_out:,} outside plot window" if n_out else ""
    # legend proxy, drawn even when nothing hit the cap; a zero-size Rectangle is
    # left out of autoscaling, so it cannot stretch the unwindowed axes to (0, 0)
    ax.add_patch(Rectangle((0, 0), 0, 0, facecolor=plt.get_cmap(CMAP)(0.8),
                           edgecolor=CAPPED_STROKES[-1][0], hatch="///", linewidth=0,
                           label=f"hit max_iters ({n:,} of {mask.n_ran:,} candidates, "
                                 f"{limit}{out})"))


def plot_residual_heatmap(summary: ResidualSummary, out_path: Path,
                          title: str = "Candidate residual over speculative TX locations",
                          hc: bool = False, analysis: Optional[dict] = None,
                          window: Optional[PlotWindowConfig] = None,
                          max_iters: Optional[MaxItersMask] = None) -> None:
    """The residual over (z, x_center), with the true TX and the best candidate marked.

    Lower residual = better fit to the measurement = more likely TX location, so this
    is the figure the whole search exists to produce. With `analysis` (the
    analysis.json dict) the top-K candidates are outlined in red.

    With `window` the axes are exactly the window at equal aspect (1 m of x = 1 m of
    z), the cells outside it are not drawn and do not set the colour limits, and the
    legend sits below the axes.

    With `max_iters` the cells whose solve hit the iteration cap are hatched and the
    title says so (residual_heatmap_maxiters.png); None draws the heatmap exactly as
    before.
    """
    if window is None:
        fig, ax = plt.subplots(figsize=(9, 6), layout="constrained")
    else:
        # "compressed": constrained layout for fixed-aspect axes, which keeps the
        # colorbar the height of the equal-aspect box rather than of its slot
        fig, ax = plt.subplots(figsize=_window_figsize(window), layout="compressed")
    grid = summary.loss_grid if window is None else _mask_to_window(summary, window)
    norm = _residual_norm(grid, hc)
    mesh = _residual_mesh(ax, summary.x_values, summary.z_values, grid, norm)
    fig.colorbar(mesh, ax=ax, label=RESIDUAL_LABEL)

    _mark_true_tx(ax, summary, window=window)
    if summary.best is not None:
        best = summary.best
        ax.scatter([best["x_center"]], [best["z"]], marker="*", s=260,
                   c="lime", edgecolor="black", linewidth=1.0, zorder=6,
                   label=f"Best candidate (residual {best['final_loss']:.4g})"
                         + _outside_note(window, best["x_center"], best["z"]))
    if analysis is not None:
        _overlay_analysis(ax, summary, analysis, window=window)
    if max_iters is not None:
        _overlay_max_iters(ax, summary, max_iters, grid, window=window)
    ax.set_xlabel("x_center (m)")
    ax.set_ylabel("z (m)")
    title = (_hc_title(title, hc, summary.noise, summary.power_penalty)
             + _hidden_note(summary, window))
    if max_iters is not None:
        title += "\nhatched: the solve hit max_iters before converging"
    if window is None:
        ax.set_title(title)
        ax.legend(loc="upper right", framealpha=0.9)
    else:
        # a figure title: the equal-aspect box is narrower than the hc title, which
        # as an axes title would run on under the colorbar
        fig.suptitle(title)
        ax.set_xlim(window.x_min, window.x_max)
        ax.set_ylim(window.z_min, window.z_max)
        ax.set_aspect("equal")
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.08), framealpha=0.9)
    if window is None:
        fig.savefig(out_path, dpi=120)
    else:
        # constrained layout sizes an equal-aspect axes' margins against its slot,
        # not the narrower box it is drawn in; the tight box keeps every label
        fig.savefig(out_path, dpi=120, bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)


def _surface_height(values: np.ndarray, norm: Normalize) -> np.ndarray:
    """Turn residuals into surface heights: low residual -> physically LOW, so
    the best candidates read as a basin the landscape drains into.

    Heights are <= 0 with the RESIDUAL_VMAX ceiling pinned at 0. The floor clip
    mirrors _residual_mesh's `np.maximum(grid, norm.vmin)` -- a LogNorm cannot
    take a zero. NaN survives np.clip, so cells that never ran stay NaN and
    plot_surface leaves them as holes.

    The two branches are the same scaling the 2D pair already draws: linear
    under Normalize, decades under LogNorm. That is what makes
    residual_surface.png the relief of residual_heatmap.png rather than a
    third, unrelated scaling.
    """
    lo = norm.vmin if isinstance(norm, LogNorm) else 0.0
    clipped = np.clip(values, lo, RESIDUAL_VMAX)
    if isinstance(norm, LogNorm):
        return np.log10(clipped / RESIDUAL_VMAX)   # decades below the 0.06 ceiling
    return clipped - RESIDUAL_VMAX                 # residual units, ceiling at 0


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
        return (np.log10(decades / RESIDUAL_VMAX),
                [f"$10^{{{int(e)}}}$" for e in exps])
    ticks = np.arange(0.0, RESIDUAL_VMAX + 1e-9, 0.01)
    return ticks - RESIDUAL_VMAX, [f"{t:g}" for t in ticks]


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


def _marker_pole(ax, x: float, z: float, floor: float, ceiling: float,
                 window: Optional[PlotWindowConfig], *, marker: str, color: str,
                 edgecolor: str, size: float, label: str) -> None:
    """_surface_pole (always, without a window), or with the marker outside the
    window only its legend entry, naming where it is: mplot3d does not clip to the
    axis limits, so a pole off the window would stand outside the box."""
    note = _outside_note(window, x, z)
    if not note:
        _surface_pole(ax, x, z, floor, ceiling, marker=marker, color=color,
                      edgecolor=edgecolor, size=size, label=label)
        return
    ax.scatter([], [], [], marker=marker, s=size, c=color, edgecolor=edgecolor,
               linewidth=1.0, depthshade=False, label=label + note)


def _draw_surface(ax, summary: ResidualSummary, hc: bool,
                  window: Optional[PlotWindowConfig] = None) -> Normalize:
    """Draw the residual surface, its floor projection and the marker poles onto `ax`.

    Returns the norm so the caller can hang a colorbar on it. Split out from
    plot_residual_surface because the orbit animation needs the identical scene.

    With `window` the surface, its colours and its depth come from the cells inside
    the window alone, the x / z limits are the window and the base of the box has
    its physical proportions.
    """
    if window is not None:
        summary = _crop_to_window(summary, window)
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

    # heights are <= 0 (basin orientation): depth = how far the best candidate
    # sinks below the 0 ceiling; the flat-map plane sits below the deepest basin
    depth = -float(np.nanmin(height))
    if not np.isfinite(depth) or depth <= 0.0:
        depth = 1.0            # every candidate at the ceiling: keep zlim sane
    floor = -depth * (1.0 + SURFACE_FLOOR_FRAC)
    ceiling = SURFACE_HEAD_FRAC * depth

    levels = (np.logspace(np.log10(norm.vmin), np.log10(RESIDUAL_VMAX), SURFACE_LEVELS)
              if isinstance(norm, LogNorm)
              else np.linspace(0.0, RESIDUAL_VMAX, SURFACE_LEVELS))
    ax.contourf(mesh_x, mesh_z, np.ma.masked_invalid(clipped), levels=levels,
                zdir="z", offset=floor, cmap=CMAP, norm=norm)

    _marker_pole(ax, summary.real_tx_x_center, summary.real_tx_z, floor, ceiling,
                 window, marker="X", color="red", edgecolor="white", size=130,
                 label="True TX location")
    if summary.best is not None:
        _marker_pole(ax, summary.best["x_center"], summary.best["z"], floor, ceiling,
                     window, marker="*", color="lime", edgecolor="black", size=260,
                     label=f"Best candidate (residual {summary.best['final_loss']:.4g})")
    if window is not None:
        x_span = window.x_max - window.x_min
        z_span = window.z_max - window.z_min
        ax.set_xlim(window.x_min, window.x_max)
        ax.set_ylim(window.z_min, window.z_max)
        # the default box is 4:4:3; keep its height at 3/4 of the longer side
        ax.set_box_aspect((x_span, z_span, 0.75 * max(x_span, z_span)))

    ax.set_zlim(floor, ceiling)
    positions, labels = _surface_zticks(norm)
    ax.set_zticks(positions)
    ax.set_zticklabels(labels)
    ax.set_xlabel("x_center (m)")
    ax.set_ylabel("z (m)")
    ax.set_zlabel("GS residual")
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
    fig.text(0.02, 0.015, "height = residual (lower = better fit, best candidates "
                          "form a basin); flat heatmap projected on the floor",
             fontsize=8, alpha=0.7)
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


def _surface_reason(summary: ResidualSummary,
                    window: Optional[PlotWindowConfig]) -> Optional[str]:
    """_surface_unplottable of the grid the surface is drawn from (the window's
    cells alone when there is one)."""
    if window is None:
        return _surface_unplottable(summary.loss_grid)
    reason = _surface_unplottable(_crop_to_window(summary, window).loss_grid)
    return None if reason is None else f"{reason} inside the plot window"


def plot_residual_surface(summary: ResidualSummary, out_path: Path,
                          title: str = "Candidate residual surface over speculative TX locations",
                          hc: bool = False,
                          window: Optional[PlotWindowConfig] = None) -> None:
    """The residual heatmap as relief: the lower the residual, the higher the surface.

    The flat heatmap saturates -- on a dense caustic sweep over half the cells sit
    within 1% of the maximum -- so the depth of the basin, which is the thing the
    search is actually measuring, is invisible there. Inverting it into height makes
    the best fit a peak standing over the TX.

    Same norms as plot_residual_heatmap, so the pair reads as one figure in two views:
    the linear variant shows the shape of the whole basin, the hc variant resolves the
    decades near the bottom of it into a single sharp summit.

    With `window` it is drawn from the cells inside the window alone (see
    _draw_surface).
    """
    title = (_hc_title(title, hc, summary.noise, summary.power_penalty)
             + _hidden_note(summary, window))
    reason = _surface_reason(summary, window)
    if reason is not None:
        _surface_placeholder(out_path, title, reason)
        return
    fig = plt.figure(figsize=(11, 8))
    ax = fig.add_subplot(projection="3d")
    norm = _draw_surface(ax, summary, hc, window=window)
    _finish_surface(fig, ax, norm, title)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def animate_residual_surface(summary: ResidualSummary, log, out_path: Path, *,
                             title: str = "Candidate residual surface over speculative TX locations",
                             hc: bool = False,
                             window: Optional[PlotWindowConfig] = None) -> Optional[Path]:
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

    if _surface_reason(summary, window) is not None:
        return None
    fig = plt.figure(figsize=(11, 8))
    ax = fig.add_subplot(projection="3d")
    norm = _draw_surface(ax, summary, hc, window=window)
    _finish_surface(fig, ax, norm,
                    _hc_title(title, hc, summary.noise, summary.power_penalty)
                    + _hidden_note(summary, window))
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
                          hc: bool = False,
                          window: Optional[PlotWindowConfig] = None) -> None:
    """Every candidate's residual against its distance to the true TX.

    A rising trend is what validates the search's premise: that a lower GS residual
    marks a candidate closer to the real transmitter.

    With `window` only the candidates inside it are plotted and set the limits.
    Its axes are a distance and a residual, not the (x, z) plane, so they have no
    physical aspect to hold.
    """
    grid = summary.loss_grid if window is None else _mask_to_window(summary, window)
    _, _, loss, dist = _candidate_points(summary.z_values, summary.x_values,
                                         grid, summary.real_tx_z,
                                         summary.real_tx_x_center)
    fig, ax = plt.subplots(figsize=(9, 6), layout="constrained")
    title = (_hc_title(title, hc, summary.noise, summary.power_penalty)
             + _hidden_note(summary, window))

    if loss.size == 0:
        ax.text(0.5, 0.5, "no candidates" if window is None
                else "no candidates inside the plot window", ha="center", va="center",
                transform=ax.transAxes)
        fig.suptitle(title)
        fig.savefig(out_path, dpi=120)
        plt.close(fig)
        return

    norm = _residual_norm(loss, hc)
    if hc:
        loss = np.maximum(loss, norm.vmin)   # keep sub-floor losses on the log axis
    ax.scatter(dist, loss, s=40, color="C0", edgecolor="black", linewidth=0.3, zorder=3)
    ax.set_xlabel("distance from candidate to true TX (m)")
    add_wavelength_axis(ax, summary.ref_freq_hz, axis="x", unit_m=1.0)
    ax.set_ylabel("GS residual")
    if hc:
        ax.set_yscale("log")
    ax.set_ylim(norm.vmin, RESIDUAL_VMAX)
    ax.set_title("Residual vs. distance to true TX")
    ax.grid(True, alpha=0.3)

    fig.suptitle(title)
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
                # every per-frequency summary carries the same run-level noise block
                dot_size=dot_size, title=_hc_title(title, hc, summaries[0][1].noise),
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
                title=(f"Candidate residual difference vs. {baseline_ghz:g} GHz baseline"
                       + snr_suffix(ordered[0][1].noise)),
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
    # the joint panel is the penalized objective when the run had the penalty on;
    # the per-frequency component is the plain loss either way
    fig.suptitle(_hc_title(f"Residual: {f_ghz:g} GHz vs. joint", hc, joint.noise,
                           joint.power_penalty))
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
