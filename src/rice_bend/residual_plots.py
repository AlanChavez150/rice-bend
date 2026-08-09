"""The candidate-residual view of a sweep: the (z, x_center) grid of GS residuals,
and the five plots drawn from it.

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
import warnings
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
    """Rebuild a ResidualSummary from a saved run's candidate_beams.json (no npz needed)."""
    with open(Path(run_dir) / "candidate_beams.json") as f:
        manifest = json.load(f)
    spec = manifest["grid_spec"]
    zs = AxisSweep(**spec["z"]).values()
    xs = AxisSweep(**spec["x_center"]).values()
    cand = [(c["index"], c["final_loss"]) for c in manifest["candidates"]]
    gt = manifest["ground_truth"]
    real_x_center = 0.5 * (gt["real_tx_x_min"] + gt["real_tx_x_max"])
    return _assemble_summary(zs, xs, cand, gt["real_tx_z"], real_x_center)


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


def average_summary(summaries: List[Tuple[float, ResidualSummary]]) -> ResidualSummary:
    """Average a multi-frequency run's residual grids into one ResidualSummary.

    Each grid cell becomes the mean residual over the frequencies where that
    candidate actually ran (NaN layers are ignored per cell); `best` is recomputed
    as the argmin of the averaged grid. The grid axes and ground truth are shared
    across frequencies, so they are taken from the first summary.
    """
    first = summaries[0][1]
    stack = np.stack([s.loss_grid for _, s in summaries])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)  # all-NaN cells
        avg = np.nanmean(stack, axis=0)
    best: Optional[dict] = None
    if np.isfinite(avg).any():
        flat = int(np.nanargmin(avg))
        z_idx, x_idx = divmod(flat, avg.shape[1])
        best = {"index": flat, "z": float(first.z_values[z_idx]),
                "x_center": float(first.x_values[x_idx]),
                "final_loss": float(avg[z_idx, x_idx])}
    return ResidualSummary(first.z_values, first.x_values, avg,
                           first.real_tx_z, first.real_tx_x_center, best)


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


def plot_residual_heatmap(summary: ResidualSummary, out_path: Path,
                          title: str = "Candidate residual over speculative TX locations",
                          norm: Optional[Normalize] = None) -> None:
    """Render the residual heatmap with the true TX location and best candidate marked.

    By default the residual colour scale is fixed to [0, 0.06] so the map is directly
    comparable across runs; pass `norm=_hc_norm(loss_grid)` for the high-contrast
    log-scale variant (data-driven floor: low residuals spread over the colormap,
    high residuals compress).
    """
    fig, ax = plt.subplots(figsize=(9, 6), layout="constrained")
    if norm is None:
        # Fixed residual scale [0, 0.06] + reversed colormap (low residual = bright
        # yellow), matching plot_residual_scatter's fixed residual axis.
        norm = Normalize(vmin=0.0, vmax=0.06)
    loss_grid = summary.loss_grid
    if isinstance(norm, LogNorm):
        # Log scale can't take zeros; clip up to the norm's floor (NaNs propagate, stay masked).
        loss_grid = np.maximum(loss_grid, norm.vmin)
    grid = np.ma.masked_invalid(loss_grid)
    mesh_x, mesh_z = np.meshgrid(summary.x_values, summary.z_values)

    mesh = ax.pcolormesh(mesh_x, mesh_z, grid, shading="nearest", cmap="viridis_r", norm=norm)
    fig.colorbar(mesh, ax=ax, label="GS residual (lower = better fit)")

    ax.scatter([summary.real_tx_x_center], [summary.real_tx_z], marker="X", s=170,
               c="red", edgecolor="white", linewidth=1.5, label="True TX location", zorder=5)
    if summary.best is not None:
        ax.scatter([summary.best["x_center"]], [summary.best["z"]], marker="*", s=260,
                   c="lime", edgecolor="black", linewidth=1.0, zorder=6,
                   label=f"Best candidate (loss {summary.best['final_loss']:.4g})")
    ax.set_xlabel("x_center (m)")
    ax.set_ylabel("z (m)")
    ax.set_title(title)
    ax.legend(loc="upper right", framealpha=0.9)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_residual_scatter(summary: ResidualSummary, out_path: Path, *,
                          title: str = "Candidate residual distribution",
                          log_scale: bool = False) -> None:
    """Render a residual-vs-distance scatter that complements the (z, x_center) heatmap.

    Every candidate's GS residual is plotted against its distance to the true TX. A
    rising trend validates the premise that a lower GS residual marks a candidate
    closer to the real transmitter. The residual axis is fixed to [0, 0.06] so the
    plot is directly comparable across runs; `log_scale=True` is the high-contrast
    variant — a log residual axis whose floor adapts to the data's own minimum
    (rounded down to a decade), spreading the lowest residuals apart.
    """
    _, _, loss, dist = _candidate_points(summary.z_values, summary.x_values,
                                         summary.loss_grid, summary.real_tx_z,
                                         summary.real_tx_x_center)
    fig, ax = plt.subplots(figsize=(9, 6), layout="constrained")

    if loss.size == 0:
        ax.text(0.5, 0.5, "no candidates", ha="center", va="center",
                transform=ax.transAxes)
        fig.suptitle(title)
        fig.savefig(out_path, dpi=120)
        plt.close(fig)
        return

    floor = _hc_norm(loss).vmin if log_scale else 0.0   # data-driven decade floor
    if log_scale:
        loss = np.maximum(loss, floor)   # keep sub-floor losses visible on the log axis
    ax.scatter(dist, loss, s=40, color="C0", edgecolor="black", linewidth=0.3, zorder=3)
    ax.set_xlabel("distance from candidate to true TX (m)")
    ax.set_ylabel("GS residual")
    if log_scale:
        ax.set_yscale("log")
        ax.set_ylim(floor, 0.06)     # floor adapts to the data's own minimum
    else:
        ax.set_ylim(0.0, 0.06)       # fixed residual range -> comparable across runs
    ax.set_title("Residual vs. distance to true TX")
    ax.grid(True, alpha=0.3)

    fig.suptitle(title)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_residual_scatter_3d(summaries: List[Tuple[float, ResidualSummary]], out_path: Path, *,
                             title: str = "Candidate residual across frequency",
                             norm: Optional[Normalize] = None) -> None:
    """Render every candidate as a 3D point at (x_center, z, frequency), coloured by residual.

    This is the multi-frequency view of the 2D residual heatmap: the spatial (x_center, z)
    plane lies flat on the bottom and frequency rises on the vertical axis, one layer of
    candidate dots per frequency. Points share the heatmap's fixed [0, 0.06] `viridis_r`
    colour scale by default (pass `norm=_hc_norm(losses)` for the high-contrast log-scale
    variant with a data-driven floor); low residual = bright yellow. Dot size falls off cubically as the residual
    grows, so only genuinely low-residual candidates stay large and the high-residual
    bulk shrinks to near-invisible dots the eye can see through. The true TX location is
    marked on every frequency layer and joined by a vertical guide line.
    """
    if norm is None:
        norm = Normalize(vmin=0.0, vmax=0.06)
    cmap = "viridis_r"
    s_min, s_max = 1.5, 60.0            # dot-size range; largest = lowest residual
    fig = plt.figure(figsize=(11, 8), layout="constrained")
    ax = fig.add_subplot(projection="3d")

    freqs_ghz = sorted(f / 1e9 for f, _ in summaries)
    real_x = real_z = None
    for freq_hz, summary in summaries:
        f_ghz = freq_hz / 1e9
        z, x, loss, _ = _candidate_points(summary.z_values, summary.x_values,
                                          summary.loss_grid, summary.real_tx_z,
                                          summary.real_tx_x_center)
        if loss.size:
            # Cubic falloff: size collapses quickly as the residual (MSE) rises, so
            # high-MSE dots are tiny and the layers stay see-through.
            loss_norm = np.clip(loss / 0.06, 0.0, 1.0)
            sizes = s_min + ((1.0 - loss_norm) ** 3) * (s_max - s_min)
            # Clip colours up to the norm's floor so a LogNorm never sees zero
            # (a no-op under the default linear norm, whose vmin is 0).
            ax.scatter(x, z, np.full_like(x, f_ghz), c=np.maximum(loss, norm.vmin),
                       cmap=cmap, norm=norm, s=sizes, depthshade=False, edgecolor="none")
        real_x, real_z = summary.real_tx_x_center, summary.real_tx_z
        ax.scatter([summary.real_tx_x_center], [summary.real_tx_z], [f_ghz], marker="X",
                   s=90, c="red", edgecolor="white", linewidth=1.0, depthshade=False,
                   zorder=6)

    # Vertical guide connecting the true-TX markers up the frequency axis.
    if real_x is not None and len(freqs_ghz) > 1:
        ax.plot([real_x, real_x], [real_z, real_z], [min(freqs_ghz), max(freqs_ghz)],
                color="red", linestyle="--", linewidth=1.0, alpha=0.6)

    ax.set_xlabel("x_center (m)")
    ax.set_ylabel("z (m)")
    ax.set_zlabel("frequency (GHz)")
    ax.set_zticks(freqs_ghz)
    ax.set_title(title)
    ax.text2D(0.02, 0.02, "dot size shrinks cubically as residual grows",
              transform=ax.transAxes, fontsize=8, alpha=0.7)
    ax.view_init(elev=22, azim=-60)

    sm = ScalarMappable(norm=norm, cmap=cmap)
    fig.colorbar(sm, ax=ax, shrink=0.6, pad=0.1,
                 label="GS residual (lower = better fit)")
    # A proxy handle so the legend documents the red X without duplicating it per layer.
    ax.scatter([], [], [], marker="X", s=90, c="red", edgecolor="white",
               linewidth=1.0, label="True TX location")
    ax.legend(loc="upper left")
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_residual_scatter_3d_diff(summaries: List[Tuple[float, ResidualSummary]],
                                  out_path: Path, *,
                                  baseline_freq: Optional[float] = None) -> None:
    """3D scatter of per-candidate residual DIFFERENCES from a baseline frequency.

    The baseline defaults to the center frequency (middle element of the sorted
    list). Each layer shows loss(f) − loss(baseline) on a symmetric `viridis_r`
    scale matching the other difference plots: bright yellow = fits BETTER than the
    baseline, dark purple = worse, mid-teal = unchanged (the baseline layer itself
    is uniformly zero). Dot size grows with the magnitude of the deviation, so
    candidates that behave like the baseline stay tiny and see-through.
    """
    ordered = sorted(summaries, key=lambda t: t[0])
    freqs = [f for f, _ in ordered]
    if baseline_freq is None:
        baseline_freq = freqs[len(freqs) // 2]
    base_grid = next(s for f, s in ordered if f == baseline_freq).loss_grid

    diffs = [(f, s, s.loss_grid - base_grid) for f, s in ordered]
    finite_all = np.concatenate([d[np.isfinite(d)].ravel() for _, _, d in diffs])
    vlim = float(np.abs(finite_all).max()) if finite_all.size else 1e-6
    vlim = max(vlim, 1e-12)
    norm = Normalize(vmin=-vlim, vmax=vlim)
    cmap = "viridis_r"
    s_min, s_max = 1.5, 60.0            # dot-size range; largest = biggest deviation

    fig = plt.figure(figsize=(11, 8), layout="constrained")
    ax = fig.add_subplot(projection="3d")
    freqs_ghz = [f / 1e9 for f in freqs]
    real_x = real_z = None
    for f, s, d in diffs:
        f_ghz = f / 1e9
        z, x, dval, _ = _candidate_points(s.z_values, s.x_values, d,
                                          s.real_tx_z, s.real_tx_x_center)
        if dval.size:
            sizes = s_min + (np.abs(dval) / vlim) * (s_max - s_min)
            ax.scatter(x, z, np.full_like(x, f_ghz), c=dval, cmap=cmap, norm=norm,
                       s=sizes, depthshade=False, edgecolor="none")
        real_x, real_z = s.real_tx_x_center, s.real_tx_z
        ax.scatter([s.real_tx_x_center], [s.real_tx_z], [f_ghz], marker="X", s=90,
                   c="red", edgecolor="white", linewidth=1.0, depthshade=False, zorder=6)

    if real_x is not None and len(freqs_ghz) > 1:
        ax.plot([real_x, real_x], [real_z, real_z], [min(freqs_ghz), max(freqs_ghz)],
                color="red", linestyle="--", linewidth=1.0, alpha=0.6)

    ax.set_xlabel("x_center (m)")
    ax.set_ylabel("z (m)")
    ax.set_zlabel("frequency (GHz)")
    ax.set_zticks(freqs_ghz)
    ax.set_title(f"Candidate residual difference vs. {baseline_freq / 1e9:g} GHz baseline")
    ax.text2D(0.02, 0.02, "dot size grows with deviation from the baseline",
              transform=ax.transAxes, fontsize=8, alpha=0.7)
    ax.view_init(elev=22, azim=-60)

    sm = ScalarMappable(norm=norm, cmap=cmap)
    fig.colorbar(sm, ax=ax, shrink=0.6, pad=0.1,
                 label=f"residual difference vs. {baseline_freq / 1e9:g} GHz "
                       "(yellow = better than baseline)")
    ax.scatter([], [], [], marker="X", s=90, c="red", edgecolor="white",
               linewidth=1.0, label="True TX location")
    ax.legend(loc="upper left")
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_residual_freq_vs_avg(freq_hz: float, summary: ResidualSummary,
                              avg: ResidualSummary, out_path: Path, *,
                              hc: bool = False) -> None:
    """Compare one frequency's residual heatmap against the all-frequency average.

    Three panels: this frequency's residuals and the frequency-averaged residuals
    (shared linear [0, 0.06] scale), then their difference (frequency − average) on a
    symmetric scale using the same `viridis_r` colormap as every other residual plot —
    bright yellow where this frequency fits BETTER than the average, dark purple where
    worse. The true TX is marked on each. With `hc=True` the two heatmaps use the
    data-driven log scale (shared floor across both grids) and the difference panel a
    symmetric log, spreading the smallest residuals/deviations apart.
    """
    f_ghz = freq_hz / 1e9
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), layout="constrained")
    mesh_x, mesh_z = np.meshgrid(avg.x_values, avg.z_values)
    if hc:
        # One data-driven log norm shared by both panels so they stay inter-comparable.
        norm = _hc_norm(np.stack([summary.loss_grid, avg.loss_grid]))
    else:
        norm = Normalize(vmin=0.0, vmax=0.06)

    mesh = None
    for ax, s, sub_title in ((axes[0], summary, f"{f_ghz:g} GHz"),
                             (axes[1], avg, "average across frequencies")):
        loss_grid = np.maximum(s.loss_grid, norm.vmin) if hc else s.loss_grid
        grid = np.ma.masked_invalid(loss_grid)
        mesh = ax.pcolormesh(mesh_x, mesh_z, grid, shading="nearest",
                             cmap="viridis_r", norm=norm)
        ax.set_title(sub_title)
    fig.colorbar(mesh, ax=list(axes[:2]), label="GS residual (lower = better fit)",
                 shrink=0.9)

    diff = summary.loss_grid - avg.loss_grid
    vlim = float(np.nanmax(np.abs(diff))) if np.isfinite(diff).any() else 1e-6
    vlim = max(vlim, 1e-12)
    if hc:
        # Symmetric log: two decades of log range each side of a linear core, so
        # small deviations from the average spread instead of washing out at teal.
        dnorm = SymLogNorm(linthresh=vlim / 100.0, vmin=-vlim, vmax=vlim, base=10)
    else:
        dnorm = Normalize(vmin=-vlim, vmax=vlim)
    dmesh = axes[2].pcolormesh(mesh_x, mesh_z, np.ma.masked_invalid(diff),
                               shading="nearest", cmap="viridis_r", norm=dnorm)
    axes[2].set_title(f"difference ({f_ghz:g} GHz − average)")
    fig.colorbar(dmesh, ax=axes[2],
                 label="residual difference (yellow = better than average)", shrink=0.9)

    for ax in axes:
        ax.scatter([avg.real_tx_x_center], [avg.real_tx_z], marker="X", s=120, c="red",
                   edgecolor="white", linewidth=1.2, zorder=5, label="True TX location")
        ax.set_xlabel("x_center (m)")
    axes[0].set_ylabel("z (m)")
    axes[0].legend(loc="upper right", framealpha=0.9)
    suffix = " (high contrast, log scale)" if hc else ""
    fig.suptitle(f"Residual: {f_ghz:g} GHz vs. frequency average{suffix}")
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
