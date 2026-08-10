"""Numeric per-run analysis of a grid search: the top candidates, the region of
interest around the argmin, the localization error, and the energy the RX window
captures. Pure measurement — no matplotlib. The overlay drawing lives in
residual_plots, which imports cell_edges FROM here; analysis never imports the
plotting stack at runtime, so grid-sweep-side consumers stay matplotlib-free.

Written to analysis.json in every grid run directory, always RECOMPUTED (at save
time and at --replot time), never read back — a replot therefore doubles as the
reproducibility check.

Energy definition: the fraction of beam power arriving at the RX plane that
lands inside the RX window, per frequency and averaged. Numerator and
denominator both live on the scene x-axis, so power diffracted beyond the scene
extent is excluded from both — that is the metric's actual definition. The field
used is the PRE-interpolation RX-plane row synthesized by MGS.measure()
(FreqState.rx_plane_row, persisted as measurement.npz['rx_plane_row']): the
windowed measurement (rx_field) is zero-filled outside the window and cannot
supply the denominator.
"""

import json
import math
from pathlib import Path
from typing import List, Optional, Sequence, TYPE_CHECKING

import numpy as np
from scipy import ndimage

from rice_bend.data_store import write_json

if TYPE_CHECKING:
    from rice_bend.grid_sweep import GridSearchRun
    from rice_bend.residual_plots import ResidualSummary

ANALYSIS_NAME = "analysis.json"
ANALYSIS_SCHEMA_VERSION = 1
TOP_K = 10                 # candidates outlined solid red on the heatmaps
ROI_LOSS_FACTOR = 10.0     # ROI = connected cells with loss <= factor * min


def _f(v) -> Optional[float]:
    """JSON-safe float: finite -> float, else None. json.dump would happily emit
    a bare NaN literal, which is not valid JSON (same rule as _json_losses)."""
    v = float(v)
    return v if math.isfinite(v) else None


def cell_edges(values: np.ndarray) -> np.ndarray:
    """Cell edges for grid-center coordinates: midpoints between neighbours, the
    outer edges extrapolated by half the adjacent step — exactly pcolormesh
    shading='nearest' geometry, so overlays drawn on these edges sit on the same
    cell borders the heatmap shows. A length-1 axis gets zero-width edges (there
    is no step to infer a width from)."""
    v = np.asarray(values, dtype=float)
    if len(v) == 1:
        return np.array([v[0], v[0]])
    mid = 0.5 * (v[1:] + v[:-1])
    first = v[0] - (mid[0] - v[0])
    last = v[-1] + (v[-1] - mid[-1])
    return np.concatenate([[first], mid, [last]])


def _energy_dict(x_axis, rows, freqs, x_min, x_max) -> Optional[dict]:
    """Per-frequency + mean fraction of RX-plane power inside the RX window.
    None when any ingredient is unavailable (experimental captures, runs saved
    before rx_plane_row was persisted)."""
    if rows is None or x_axis is None or not freqs or x_min is None or x_max is None:
        return None
    x = np.asarray(x_axis, dtype=float)
    # the exact complement of measure()'s weighting zero-mask
    win = (x >= x_min) & (x <= x_max)
    per: List[dict] = []
    fracs: List[float] = []
    for f, u in zip(freqs, rows):
        p = np.abs(np.asarray(u)) ** 2
        tot = float(p.sum())
        frac = float(p[win].sum() / tot) if tot > 0 else None
        per.append({"freq_hz": float(f), "fraction_in_window": frac})
        if frac is not None:
            fracs.append(frac)
    return {"rx_window": {"x_min": float(x_min), "x_max": float(x_max)},
            "per_freq": per,
            "mean_fraction": float(np.mean(fracs)) if fracs else None}


def analyze_grid(z_values: Sequence[float], x_values: Sequence[float],
                 loss_grid: np.ndarray, real_tx_z: float, real_tx_x_center: float,
                 energy: Optional[dict]) -> dict:
    """The analysis dict for one (nz, nx) joint-residual grid.

    Cells are addressed as (z_idx, x_idx) with candidate index = z_idx*nx + x_idx
    (enumerate_grid's z-outer/x-inner order). NaN cells (skipped / not run) are
    excluded everywhere. `argmin.error_distance_m` is THE localization error the
    parameter studies put on their y-axis.
    """
    zs = np.asarray(z_values, dtype=float)
    xs = np.asarray(x_values, dtype=float)
    grid = np.asarray(loss_grid, dtype=float)
    nz, nx = grid.shape
    finite = np.isfinite(grid)
    n_finite = int(finite.sum())
    out = {
        "schema_version": ANALYSIS_SCHEMA_VERSION,
        "top_k": TOP_K,
        "roi_loss_factor": ROI_LOSS_FACTOR,
        "ground_truth": {"real_tx_z": float(real_tx_z),
                         "real_tx_x_center": float(real_tx_x_center)},
        "loss": {"n_finite_cells": n_finite,
                 "min": _f(np.nanmin(grid)) if n_finite else None,
                 "max": _f(np.nanmax(grid)) if n_finite else None},
        "argmin": None,
        "top_candidates": [],
        "roi": None,
        "energy": energy,
    }
    if n_finite == 0:
        return out

    def dist(i: int, j: int) -> float:
        return float(np.hypot(zs[i] - real_tx_z, xs[j] - real_tx_x_center))

    ai, aj = np.unravel_index(int(np.nanargmin(grid)), grid.shape)
    ai, aj = int(ai), int(aj)
    out["argmin"] = {
        "index": ai * nx + aj, "z_idx": ai, "x_idx": aj,
        "z": float(zs[ai]), "x_center": float(xs[aj]),
        "final_loss": _f(grid[ai, aj]),
        "error_z_m": float(zs[ai] - real_tx_z),
        "error_x_m": float(xs[aj] - real_tx_x_center),
        "error_distance_m": dist(ai, aj),
    }

    # top-K by (loss, flat index): stable, deterministic ties
    fi, fj = np.where(finite)
    order = sorted(range(len(fi)),
                   key=lambda t: (grid[fi[t], fj[t]], int(fi[t]) * nx + int(fj[t])))
    for rank, t in enumerate(order[:TOP_K], start=1):
        i, j = int(fi[t]), int(fj[t])
        out["top_candidates"].append({
            "rank": rank, "index": i * nx + j, "z_idx": i, "x_idx": j,
            "z": float(zs[i]), "x_center": float(xs[j]),
            "final_loss": _f(grid[i, j]), "dist_to_truth_m": dist(i, j),
        })

    # ROI: the 4-connected component containing the argmin, loss <= factor*min.
    # ndimage.label's default structure IS the 4-connected cross.
    loss_min = float(grid[ai, aj])
    cutoff = ROI_LOSS_FACTOR * loss_min
    mask = finite & (grid <= cutoff)
    labels, _ = ndimage.label(mask)
    roi_mask = labels == labels[ai, aj]
    cells = np.argwhere(roi_mask)              # row-major sorted (z_idx, x_idx)
    z_e, x_e = cell_edges(zs), cell_edges(xs)
    csz = float((zs[-1] - zs[0]) / (len(zs) - 1)) if len(zs) > 1 else None
    csx = float((xs[-1] - xs[0]) / (len(xs) - 1)) if len(xs) > 1 else None
    i_min, i_max = int(cells[:, 0].min()), int(cells[:, 0].max())
    j_min, j_max = int(cells[:, 1].min()), int(cells[:, 1].max())
    centroid_z = float(zs[cells[:, 0]].mean())
    centroid_x = float(xs[cells[:, 1]].mean())
    out["roi"] = {
        "loss_min": _f(loss_min), "loss_cutoff": _f(cutoff),
        "n_cells": int(roi_mask.sum()),
        "cells": [[int(i), int(j)] for i, j in cells],
        "cell_size_z_m": csz, "cell_size_x_m": csx,
        "area_m2": (int(roi_mask.sum()) * csz * csx
                    if csz is not None and csx is not None else None),
        "z_min_m": float(z_e[i_min]), "z_max_m": float(z_e[i_max + 1]),
        "z_extent_m": float(z_e[i_max + 1] - z_e[i_min]),
        "x_min_m": float(x_e[j_min]), "x_max_m": float(x_e[j_max + 1]),
        "x_extent_m": float(x_e[j_max + 1] - x_e[j_min]),
        "centroid": {"z": centroid_z, "x_center": centroid_x},
        "centroid_dist_to_truth_m": float(np.hypot(centroid_z - real_tx_z,
                                                   centroid_x - real_tx_x_center)),
    }
    return out


def analysis_from_run(run: "GridSearchRun", summary: "ResidualSummary") -> dict:
    """The analysis for an in-memory run (save-time path)."""
    energy = _energy_dict(run.scene_x_axis, run.rx_plane_rows, run.freqs,
                          run.rx_x_min, run.rx_x_max)
    return analyze_grid(summary.z_values, summary.x_values, summary.loss_grid,
                        summary.real_tx_z, summary.real_tx_x_center, energy)


def analysis_from_manifest(run_dir: Path, summary: "ResidualSummary") -> dict:
    """Rebuild the identical analysis from a saved run dir (--replot path).

    Legacy guards: schema-2 manifests lack 'frequencies'/'rx_aperture', and
    measurement.npz from before rx_plane_row existed lacks the key — any absence
    degrades energy to null, never an error.
    """
    run_dir = Path(run_dir)
    with open(run_dir / "candidate_beams.json") as f:
        m = json.load(f)
    energy = None
    freqs = m.get("frequencies")
    rx_win = m.get("rx_aperture")
    meas_path = run_dir / "measurement.npz"
    if freqs and rx_win and meas_path.exists():
        with np.load(meas_path) as z:
            if "rx_plane_row" in z.files:
                energy = _energy_dict(z["scene_x_axis"], z["rx_plane_row"],
                                      [e["freq_hz"] for e in freqs],
                                      rx_win["x_min"], rx_win["x_max"])
    return analyze_grid(summary.z_values, summary.x_values, summary.loss_grid,
                        summary.real_tx_z, summary.real_tx_x_center, energy)


def write_analysis(run_dir: Path, analysis: dict) -> Path:
    return write_json(Path(run_dir) / ANALYSIS_NAME, analysis)
