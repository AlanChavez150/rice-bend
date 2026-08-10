"""Numeric per-run analysis of a grid search: the top candidates, the
localization error, and the energy the RX window captures. Pure measurement —
no matplotlib. The overlay drawing lives in
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

from rice_bend import rs
from rice_bend.data_store import write_json
from rice_bend.interp import interp_amplitude, interp_real_imag

if TYPE_CHECKING:
    from rice_bend.grid_sweep import GridSearchRun
    from rice_bend.residual_plots import ResidualSummary

ANALYSIS_NAME = "analysis.json"
ANALYSIS_SCHEMA_VERSION = 4
TOP_LOSS_FACTOR = 1.5      # top candidates = the argmin's 8-connected cluster of
                           # cells with loss <= factor * min; the count is
                           # adaptive (calibrated on the observed ~1e-6 near-tie
                           # band at the lambda/20 loss floor, as a fraction)
TOP_MAX_STORED = 300       # cap on stored entries; stats always use the full set
N_DOF_EPS = 0.1            # the project-wide accuracy convention (see the docs)


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


def _n_dof_dict(scene_x_axis, tx_axis, tx_profiles, freqs, wavelengths,
                tx_z, rx_z, rx_x_min, rx_x_max,
                rx_elem_axes: Optional[Sequence] = None) -> Optional[dict]:
    """N_E(eps): the NOISELESS mode count of docs/information_metric.md.

    Per frequency, build the discrete radiation operator exactly as
    gs_reconstruct applies it — column j = rs_apply(e_j, h_fwd, dx) for a unit
    source at TX-support sample j, rows restricted to scene samples inside the
    RX window — then SVD and couple the actual beam:

        M = U S Vh;  c = Vh @ u0[support];  a_k = sigma_k * |c_k|
        N_E(eps) = #{ k : a_k >= eps * sigma_1 * ||u0[support]|| }

    The doc's threshold is sigma_n/eps — a receiver-noise floor. This variant is
    deliberately noise-free (per project direction: no SNR modeling): the
    absolute reference is the channel's own capacity, sigma_1*||u0|| — the
    largest received amplitude THIS geometry could deliver from THIS beam power.
    A beam walking off the window collapses every received mode amplitude a_k
    against that fixed reference, so the count falls exactly as captured energy
    falls, with no noise parameter anywhere.

    Multi-frequency: computed per frequency and SUMMED — the multi-arc template
    the doc cites (sufficiently separated observation arcs multiply the
    data-space dimension).

    Rows model the RECEIVER, not the window: when the per-frequency RX element
    axes are given, each row is the scene sample nearest an element position
    (deduplicated) — so a sparse array (lambda/2 spacing) has fewer rows and a
    genuinely smaller data space than the dense default (lambda/20, whose
    elements are denser than the scene grid and reduce to every window sample).
    Without element axes (legacy runs), all window samples are used.

    TX profiles are cast to complex64 first so the save-time and replot-time
    computations see bit-identical inputs (measurement.npz stores complex64).
    None when any ingredient is unavailable (experimental captures, legacy runs).
    """
    if (tx_profiles is None or tx_axis is None or scene_x_axis is None
            or not freqs or tx_z is None or rx_z is None
            or rx_x_min is None or rx_x_max is None):
        return None
    x = np.asarray(scene_x_axis, dtype=float)
    dx = x[1] - x[0]
    window = np.where((x >= rx_x_min) & (x <= rx_x_max))[0]
    if not len(window):
        return None
    per: List[dict] = []
    total = 0
    for i, (f, wl, profile) in enumerate(zip(freqs, wavelengths, tx_profiles)):
        elems = None
        if rx_elem_axes is not None and i < len(rx_elem_axes):
            elems = rx_elem_axes[i]
        if elems is not None and len(elems):
            e_pos = np.asarray(elems, dtype=float)
            rows = np.unique(np.argmin(np.abs(x[None, :] - e_pos[:, None]), axis=1))
        else:
            rows = window
        profile = np.asarray(profile, dtype=np.complex64)
        amp = interp_amplitude(tx_axis, profile, x)
        support = np.where(amp > 0)[0]
        if not len(support) or not len(rows):
            per.append({"freq_hz": float(f), "n_dof": None})
            continue
        h_fwd = rs.rs_kernel(x, float(wl), -1.0 * (rx_z - tx_z))
        cols = np.empty((len(rows), len(support)), dtype=np.complex64)
        e = np.zeros(len(x))
        for k, j in enumerate(support):
            e[j] = 1.0
            cols[:, k] = rs.rs_apply(e, h_fwd, dx)[rows]
            e[j] = 0.0
        sigma, vh = np.linalg.svd(cols, full_matrices=False)[1:]
        u0 = interp_real_imag(tx_axis, profile, x)[support]
        c = vh @ u0
        a = sigma * np.abs(c)
        threshold = N_DOF_EPS * float(sigma[0]) * float(np.linalg.norm(u0))
        n = int((a >= threshold).sum())
        total += n
        per.append({
            "freq_hz": float(f), "n_dof": n,
            "n_support": int(len(support)),
            "n_rows": int(len(rows)),
            "n_elements": int(len(elems)) if elems is not None else None,
            "sigma_max": _f(sigma[0]),
            "excitation_max": _f(a.max()) if len(a) else None,
            "threshold": _f(threshold),
        })
    return {
        "eps": N_DOF_EPS,
        "definition": ("noiseless N_E(eps): modes with sigma_k*|c_k| >= "
                       "eps*sigma_1*||u0[support]||, per frequency; total = sum "
                       "(see docs/information_metric.md)"),
        "tx_z_m": float(tx_z), "rx_z_m": float(rx_z),
        "per_freq": per,
        "total": total,
    }


def analyze_grid(z_values: Sequence[float], x_values: Sequence[float],
                 loss_grid: np.ndarray, real_tx_z: float, real_tx_x_center: float,
                 energy: Optional[dict], n_dof: Optional[dict] = None) -> dict:
    """The analysis dict for one (nz, nx) joint-residual grid.

    Cells are addressed as (z_idx, x_idx) with candidate index = z_idx*nx + x_idx
    (enumerate_grid's z-outer/x-inner order). NaN cells (skipped / not run) are
    excluded everywhere. `argmin.error_distance_m` and `top_mean_dist_to_truth_m`
    (unweighted mean over the adaptive top-candidates set, argmin included) are
    the two localization errors the parameter studies put on their y-axis.
    """
    zs = np.asarray(z_values, dtype=float)
    xs = np.asarray(x_values, dtype=float)
    grid = np.asarray(loss_grid, dtype=float)
    nz, nx = grid.shape
    finite = np.isfinite(grid)
    n_finite = int(finite.sum())
    out = {
        "schema_version": ANALYSIS_SCHEMA_VERSION,
        "top_candidates_criterion": {"loss_factor": TOP_LOSS_FACTOR,
                                     "connectivity": 8},
        "ground_truth": {"real_tx_z": float(real_tx_z),
                         "real_tx_x_center": float(real_tx_x_center)},
        "loss": {"n_finite_cells": n_finite,
                 "min": _f(np.nanmin(grid)) if n_finite else None,
                 "max": _f(np.nanmax(grid)) if n_finite else None},
        "argmin": None,
        "top_candidates": [],
        "n_top_candidates": 0,
        "n_qualifying_total": 0,
        "top_candidates_truncated": False,
        "top_mean_dist_to_truth_m": None,
        "energy": energy,
        "n_dof": n_dof,
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

    # top candidates: the 8-connected component (diagonals adjacent) containing
    # the argmin, among cells with loss <= TOP_LOSS_FACTOR * min. The count is
    # the size of the near-tie cluster, not a fixed K; qualifying cells that
    # are NOT connected to the argmin (deceptive far minima) are excluded from
    # the set but counted in n_qualifying_total.
    mask = finite & (grid <= TOP_LOSS_FACTOR * float(grid[ai, aj]))
    labels, _ = ndimage.label(mask, structure=np.ones((3, 3), dtype=bool))
    comp = labels == labels[ai, aj]
    ci, cj = np.where(comp)
    order = sorted(range(len(ci)),
                   key=lambda t: (grid[ci[t], cj[t]], int(ci[t]) * nx + int(cj[t])))
    out["n_qualifying_total"] = int(mask.sum())
    out["n_top_candidates"] = int(comp.sum())
    out["top_candidates_truncated"] = len(order) > TOP_MAX_STORED
    for rank, t in enumerate(order[:TOP_MAX_STORED], start=1):
        i, j = int(ci[t]), int(cj[t])
        out["top_candidates"].append({
            "rank": rank, "index": i * nx + j, "z_idx": i, "x_idx": j,
            "z": float(zs[i]), "x_center": float(xs[j]),
            "final_loss": _f(grid[i, j]), "dist_to_truth_m": dist(i, j),
        })
    out["top_mean_dist_to_truth_m"] = _f(float(
        np.mean([dist(int(i), int(j)) for i, j in zip(ci, cj)])))
    return out


def analysis_from_run(run: "GridSearchRun", summary: "ResidualSummary") -> dict:
    """The analysis for an in-memory run (save-time path)."""
    energy = _energy_dict(run.scene_x_axis, run.rx_plane_rows, run.freqs,
                          run.rx_x_min, run.rx_x_max)
    n_dof = _n_dof_dict(run.scene_x_axis, run.real_tx_aper_axis,
                        run.real_tx_aper_profiles, run.freqs, run.wavelengths,
                        run.real_tx_z, run.rx_z, run.rx_x_min, run.rx_x_max,
                        rx_elem_axes=run.rx_aper_axes)
    return analyze_grid(summary.z_values, summary.x_values, summary.loss_grid,
                        summary.real_tx_z, summary.real_tx_x_center, energy,
                        n_dof=n_dof)


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
    n_dof = None
    freqs = m.get("frequencies")
    rx_win = m.get("rx_aperture")
    meas_path = run_dir / "measurement.npz"
    if freqs and rx_win and meas_path.exists():
        freq_hz = [e["freq_hz"] for e in freqs]
        with np.load(meas_path) as z:
            if "rx_plane_row" in z.files:
                energy = _energy_dict(z["scene_x_axis"], z["rx_plane_row"],
                                      freq_hz, rx_win["x_min"], rx_win["x_max"])
            if ("real_tx_aper_profile" in z.files and rx_win.get("z") is not None
                    and np.asarray(z["real_tx_aper_profile"]).ndim == 2):
                elem_axes = None
                keys = [f"rx_aper_axis_{i:02d}" for i in range(len(freqs))]
                if all(k in z.files for k in keys):
                    elem_axes = [z[k] for k in keys]
                n_dof = _n_dof_dict(
                    z["scene_x_axis"], z["real_tx_aper_axis"],
                    z["real_tx_aper_profile"], freq_hz,
                    [e["wavelength_m"] for e in freqs],
                    m["ground_truth"]["real_tx_z"], rx_win["z"],
                    rx_win["x_min"], rx_win["x_max"],
                    rx_elem_axes=elem_axes)
    return analyze_grid(summary.z_values, summary.x_values, summary.loss_grid,
                        summary.real_tx_z, summary.real_tx_x_center, energy,
                        n_dof=n_dof)


def write_analysis(run_dir: Path, analysis: dict) -> Path:
    return write_json(Path(run_dir) / ANALYSIS_NAME, analysis)
