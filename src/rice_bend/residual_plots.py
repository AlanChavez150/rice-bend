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
# --- shared plot vocabulary --------------------------------------------------
CMAP = "viridis_r"                 # low residual = bright yellow
RESIDUAL_VMAX = 0.06               # fixed residual ceiling -> comparable across runs
RESIDUAL_LABEL = "GS residual (lower = better fit)"
HC_SUFFIX = " (high contrast, log scale)"
DOT_MIN, DOT_MAX = 1.5, 60.0       # 3D dot-size range


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


def plot_residual_freq_vs_avg(freq_hz: float, summary: ResidualSummary,
                              avg: ResidualSummary, out_path: Path, *,
                              hc: bool = False) -> None:
    """One frequency's residual heatmap against the all-frequency average.

    Three panels: this frequency, the average (shared scale), and their difference on
    a symmetric scale — bright yellow where this frequency fits BETTER than the
    average, dark purple where worse.
    """
    f_ghz = freq_hz / 1e9
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), layout="constrained")
    # one norm shared by both heatmap panels so they stay inter-comparable
    norm = _residual_norm(np.stack([summary.loss_grid, avg.loss_grid]), hc)

    mesh = None
    for ax, s, sub_title in ((axes[0], summary, f"{f_ghz:g} GHz"),
                             (axes[1], avg, "average across frequencies")):
        mesh = _residual_mesh(ax, avg.x_values, avg.z_values, s.loss_grid, norm)
        ax.set_title(sub_title)
    fig.colorbar(mesh, ax=list(axes[:2]), label=RESIDUAL_LABEL, shrink=0.9)

    diff = summary.loss_grid - avg.loss_grid
    vlim = max(float(np.nanmax(np.abs(diff))) if np.isfinite(diff).any() else 1e-6, 1e-12)
    if hc:
        # symmetric log: two decades either side of a linear core, so small
        # deviations from the average spread instead of washing out at teal
        dnorm = SymLogNorm(linthresh=vlim / 100.0, vmin=-vlim, vmax=vlim, base=10)
    else:
        dnorm = Normalize(vmin=-vlim, vmax=vlim)
    dmesh = _residual_mesh(axes[2], avg.x_values, avg.z_values, diff, dnorm)
    axes[2].set_title(f"difference ({f_ghz:g} GHz − average)")
    fig.colorbar(dmesh, ax=axes[2],
                 label="residual difference (yellow = better than average)", shrink=0.9)

    for ax in axes:
        _mark_true_tx(ax, avg, size=120)
        ax.set_xlabel("x_center (m)")
    axes[0].set_ylabel("z (m)")
    axes[0].legend(loc="upper right", framealpha=0.9)
    fig.suptitle(_hc_title(f"Residual: {f_ghz:g} GHz vs. frequency average", hc))
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
