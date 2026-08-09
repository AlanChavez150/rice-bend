"""Speculative TX-location grid for `grid-search-mgs`.

Treats the TX location as unknown. The real TX location/trajectory (from config)
is used only to synthesize the single shared RX measurement; the search itself
sweeps a (z, x_center) grid of candidate TX planes. At each usable grid point a
fixed-width aperture (uniform assumed amplitude) is placed and MGS reconstructs
its phase against the measurement. Each reconstruction is persisted as a
"candidate beam" for the later ranking phase.
"""

import argparse
import json
import logging
import os
import shutil
import warnings
from collections import namedtuple
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import coloredlogs
import matplotlib.pyplot as plt
import numpy as np
import scipy.constants
from matplotlib.animation import FuncAnimation
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LogNorm, Normalize, SymLogNorm
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers the 3d projection)

from rice_bend import __version__, rs
from rice_bend.animate import _write_mp4
from rice_bend.config import AxisSweep, GridSearchConfig, SimConfig, SimSceneConfig, load_config
from rice_bend.data_store import _c64, _f64, _json_safe, check_run_dir, make_run_dir
from rice_bend.mgs import MGS, gs_params_from_cfg, gs_reconstruct, interp_complex_to_axis
from rice_bend.sim_scene import SimAperature

# Default config shipped in the repo's configs/ folder (repo_root/configs/caustic_config.yml)
DEFAULT_CONFIG = Path(__file__).resolve().parents[2] / "configs" / "caustic_config.yml"

# Tolerance for inclusive bounds checks, to absorb float round-off in the sweep
# endpoints (e.g. an aperture edge landing exactly on the scene boundary).
_BOUNDS_TOL = 1e-9


# --------------------------------------------------------------------------- #
# Grid enumeration
# --------------------------------------------------------------------------- #
@dataclass
class GridPoint:
    """One speculative TX location. `skip_reason` is None when the point is usable."""
    index: int
    z: float
    x_center: float
    x_min: float
    x_max: float
    dx: float
    skip_reason: Optional[str] = None

    @property
    def ok(self) -> bool:
        return self.skip_reason is None


def _scene_x_axis(scene_cfg: SimSceneConfig) -> np.ndarray:
    """Rebuild the scene's x sampling exactly as SimScene does (for the RS guard)."""
    nx = int((scene_cfg.x_max - scene_cfg.x_min) / scene_cfg.spacing)
    return np.linspace(scene_cfg.x_min, scene_cfg.x_max, nx)


def enumerate_grid(grid_cfg: GridSearchConfig, scene_cfg: SimSceneConfig,
                   wavelength: float) -> List[GridPoint]:
    """Enumerate the (z, x_center) grid, flagging out-of-bounds / undersampled points.

    A point is skipped (skip_reason set) when its assumed aperture window leaves
    the scene laterally, when z falls outside (z_min, z_max], or when the TX plane
    is so close to the RX plane (z_min) that rs() would reject the sampling.
    """
    half = grid_cfg.aperture.width / 2.0
    dx = grid_cfg.aperture.dx
    x_axis = _scene_x_axis(scene_cfg)
    rx_plane = np.array([scene_cfg.z_min])  # RX/origin plane the field propagates to

    points: List[GridPoint] = []
    index = 0
    for z in grid_cfg.z.values():
        for x_center in grid_cfg.x_center.values():
            x_min = x_center - half
            x_max = x_center + half
            reason: Optional[str] = None
            if x_min < scene_cfg.x_min - _BOUNDS_TOL or x_max > scene_cfg.x_max + _BOUNDS_TOL:
                reason = (f"aperture window [{x_min:.3f}, {x_max:.3f}] leaves scene x "
                          f"[{scene_cfg.x_min}, {scene_cfg.x_max}]")
            elif z <= scene_cfg.z_min or z > scene_cfg.z_max + _BOUNDS_TOL:
                reason = f"z {z:.3f} outside scene z ({scene_cfg.z_min}, {scene_cfg.z_max}]"
            else:
                quality = rs.sampling_quality(x_axis, rx_plane, wavelength,
                                              z_src=z, forward_dir=-1.0)
                if quality < 1.0:
                    reason = (f"z {z:.3f} too close to RX: RS undersampled "
                              f"(quality {quality:.3f} < 1)")
            points.append(GridPoint(index, float(z), float(x_center),
                                    float(x_min), float(x_max), float(dx), reason))
            index += 1
    return points


def grid_summary(points: List[GridPoint]) -> str:
    """One-line human summary of an enumerated grid."""
    usable = sum(1 for p in points if p.ok)
    return f"{len(points)} grid points ({usable} usable, {len(points) - usable} skipped)"


# --------------------------------------------------------------------------- #
# Candidate reconstruction (shared by the serial and multiprocess paths)
# --------------------------------------------------------------------------- #
# Read-only measurement + GS hyperparameters shared by every candidate. It is small
# (the measurement vectors, NOT the full 2D scene), so it is cheap to hand to workers.
SharedMeasurement = namedtuple(
    "SharedMeasurement", "x_axis rx_z rx_field error_weighting wavelength params seed")

# Per-worker copy of the shared measurement, set once by the pool initializer so it is
# not re-pickled for every one of the (potentially hundreds of) candidate tasks.
_WORKER_SHARED: Optional["SharedMeasurement"] = None


def _reconstruct_candidate(p: "GridPoint", shared: "SharedMeasurement") -> "CandidateResult":
    """Reconstruct one candidate beam at grid point `p`. Pure given `shared` — runs
    identically in the parent process or a worker."""
    x_axis = shared.x_axis
    hyp = SimAperature(x_min=p.x_min, x_max=p.x_max, z=p.z, dx=p.dx)
    # uniform assumed amplitude over the hypothesized window (support = window)
    assumed_amp = np.where((x_axis >= p.x_min) & (x_axis <= p.x_max), 1.0, 0.0)
    result = gs_reconstruct(
        tx_z=p.z, orig_aper_amp=assumed_amp, x_axis=x_axis, rx_z=shared.rx_z,
        rx_field=shared.rx_field, error_weighting=shared.error_weighting,
        wavelength=shared.wavelength, params=shared.params, seed=shared.seed,
        capture=False, log=None,
    )
    aper_profile = interp_complex_to_axis(x_axis, result.curr_aper_f, hyp.aper_axis)
    return CandidateResult(
        point=p,
        aper_axis=hyp.aper_axis.copy(),
        aper_profile=aper_profile,
        final_loss=result.final_loss,
        n_iters_run=int(result.n_iters_run),
        stop_reason=str(result.stop_reason),
        seed=int(result.seed),
        loss_full=np.asarray(result.loss_full, dtype=np.float32),
    )


def _init_worker(shared: "SharedMeasurement") -> None:
    """ProcessPoolExecutor initializer: stash the shared measurement once per worker and
    pin math-library threads to 1 so N processes don't oversubscribe the cores (the GS
    hot path is FFT + ufuncs, so this is mostly precautionary)."""
    global _WORKER_SHARED
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                "NUMEXPR_NUM_THREADS"):
        os.environ.setdefault(var, "1")
    _WORKER_SHARED = shared


def _worker_task(p: "GridPoint") -> "CandidateResult":
    """Worker entry point: reconstruct one candidate using the per-worker shared state."""
    return _reconstruct_candidate(p, _WORKER_SHARED)


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #
@dataclass
class CandidateResult:
    """A single reconstructed candidate beam at a hypothesized TX location."""
    point: GridPoint
    aper_axis: np.ndarray        # hypothesized aperture x axis
    aper_profile: np.ndarray     # complex reconstructed aperture
    final_loss: float
    n_iters_run: int
    stop_reason: str
    seed: int
    loss_full: np.ndarray        # per-iteration loss curve


@dataclass
class GridSearchRun:
    """Everything needed to persist + later rank a grid search."""
    freq: float
    wavelength: float
    seed: Optional[int]
    effective_max_iters: int
    grid_cfg: GridSearchConfig
    grid_points: List[GridPoint]
    candidates: List[CandidateResult]
    # shared measurement (one per run)
    scene_x_axis: np.ndarray
    rx_field: np.ndarray
    error_weighting: np.ndarray
    rx_aper_axis: np.ndarray
    rx_aper_profile: np.ndarray
    # ground truth, for later evaluation of how well candidates localize the TX
    real_tx_aper_axis: np.ndarray
    real_tx_aper_profile: np.ndarray
    real_tx_z: float
    real_tx_x_min: float
    real_tx_x_max: float
    scene_bounds: Tuple[float, float, float, float]  # x_min, x_max, z_min, z_max


def run_grid_search(config: SimConfig, freq: float, *, limit: Optional[int] = None,
                    jobs: int = 1, log: Optional[logging.Logger] = None) -> GridSearchRun:
    """Run MGS phase retrieval at every usable speculative TX location.

    Builds the real scene once (synthesizing the shared RX measurement), then
    reconstructs a fixed-width aperture (uniform assumed amplitude) at each hypothesized
    (z, x_center). Candidates are independent, so with `jobs > 1` they are distributed
    across worker processes; results are identical to the serial path (each candidate
    uses the same fixed seed) regardless of `jobs` or completion order.
    """
    grid_cfg = config.grid_search
    if grid_cfg is None:
        raise ValueError("config.grid_search is required for grid-search-mgs")
    log = log or logging.getLogger()

    # 1. real scene + the single shared RX measurement
    mgs = MGS(freq, config)
    mgs.run_sim(gs_rec=False, measure_rx=True)
    mgs.measure()

    # 2. apply grid-only GS overrides (cheaper sweep + one fixed seed for comparability)
    if grid_cfg.gs_overrides.max_iters is not None:
        mgs.gs_cfg.max_iters = grid_cfg.gs_overrides.max_iters
    if grid_cfg.seed is not None:
        mgs.gs_cfg.seed = grid_cfg.seed
    elif mgs.gs_cfg.seed is None:
        log.warning("grid_search.seed and gerchberg_saxton.seed are both null; candidates "
                    "will use independent random initial phases (residuals not comparable)")

    # 3. enumerate + report
    wavelength = mgs.wavelength
    scene_cfg = config.sim_scene
    points = enumerate_grid(grid_cfg, scene_cfg, wavelength)
    usable = [p for p in points if p.ok]
    log.info(f"Grid search: {grid_summary(points)}")
    for p in points:
        if not p.ok:
            log.warning(f"  skip #{p.index} z={p.z:.3f} x_center={p.x_center:+.3f}: {p.skip_reason}")
    if limit is not None:
        usable = usable[:limit]
        log.info(f"--limit: running the first {len(usable)} usable candidate(s)")

    # 4. reconstruct at each usable grid point. Bundle the small, read-only measurement
    #    (vectors + GS params) once; it is shared by every candidate.
    x_axis = mgs.scene.x_axis
    shared = SharedMeasurement(
        x_axis=np.asarray(x_axis).copy(),
        rx_z=float(mgs.scene.rx_ap.z),
        rx_field=np.asarray(mgs._rx_field).copy(),
        error_weighting=np.asarray(mgs._error_weighting).copy(),
        wavelength=float(wavelength),
        params=gs_params_from_cfg(mgs.gs_cfg),
        seed=mgs.gs_cfg.seed,
    )

    total = len(usable)
    n_jobs = max(1, int(jobs))
    candidates: List[CandidateResult] = []

    def _log_done(done: int, cand: CandidateResult) -> None:
        p = cand.point
        log.info(f"[{done}/{total}] #{p.index} z={p.z:.3f} x={p.x_center:+.3f} "
                 f"-> loss {cand.final_loss:.6g} ({cand.n_iters_run} iters, {cand.stop_reason})")

    if n_jobs == 1 or total <= 1:
        for n, p in enumerate(usable):
            cand = _reconstruct_candidate(p, shared)
            candidates.append(cand)
            _log_done(n + 1, cand)
    else:
        n_workers = min(n_jobs, total)
        log.info(f"Reconstructing {total} candidate(s) across {n_workers} worker process(es)")
        with ProcessPoolExecutor(max_workers=n_workers,
                                 initializer=_init_worker, initargs=(shared,)) as ex:
            futures = [ex.submit(_worker_task, p) for p in usable]
            for done, fut in enumerate(as_completed(futures), start=1):
                cand = fut.result()
                candidates.append(cand)
                _log_done(done, cand)
        # collected in completion order; restore enumeration order for a stable manifest
        candidates.sort(key=lambda c: c.point.index)

    real_tx = config.tx_aperture
    return GridSearchRun(
        freq=float(freq),
        wavelength=float(wavelength),
        seed=mgs.gs_cfg.seed,
        effective_max_iters=int(mgs.gs_cfg.max_iters),
        grid_cfg=grid_cfg,
        grid_points=points,
        candidates=candidates,
        scene_x_axis=np.asarray(x_axis).copy(),
        rx_field=np.asarray(mgs._rx_field).copy(),
        error_weighting=np.asarray(mgs._error_weighting).copy(),
        rx_aper_axis=mgs.scene.rx_ap.aper_axis.copy(),
        rx_aper_profile=mgs.scene.rx_ap.aper_profile.copy(),
        real_tx_aper_axis=mgs.scene.tx_ap.aper_axis.copy(),
        real_tx_aper_profile=mgs.scene.tx_ap.aper_profile.copy(),
        real_tx_z=float(real_tx.z),
        real_tx_x_min=float(real_tx.x_min),
        real_tx_x_max=float(real_tx.x_max),
        scene_bounds=(scene_cfg.x_min, scene_cfg.x_max, scene_cfg.z_min, scene_cfg.z_max),
    )


# --------------------------------------------------------------------------- #
# Persistence: per-candidate files + a manifest + the shared measurement
# --------------------------------------------------------------------------- #
def save_grid_run(run: GridSearchRun, run_dir: Path, config: SimConfig,
                  config_path: Path, args_dict: dict) -> None:
    """Write the grid run to disk under run_dir.

    Layout:
        run_dir/candidate_beams.json      manifest (real TX location, grid spec, candidate index)
        run_dir/measurement.npz           shared RX field + error weighting + apertures
        run_dir/candidates/cand_####.npz  reconstructed aperture + loss curve
        run_dir/candidates/cand_####.json per-candidate metadata
    """
    cand_dir = run_dir / "candidates"
    cand_dir.mkdir(parents=True, exist_ok=True)

    # shared measurement (saved once)
    np.savez_compressed(
        run_dir / "measurement.npz",
        scene_x_axis=_f64(run.scene_x_axis),
        rx_field=_c64(run.rx_field),
        error_weighting=_f64(run.error_weighting),
        rx_aper_axis=_f64(run.rx_aper_axis),
        rx_aper_profile=_c64(run.rx_aper_profile),
        real_tx_aper_axis=_f64(run.real_tx_aper_axis),
        real_tx_aper_profile=_c64(run.real_tx_aper_profile),
    )

    ground_truth = {
        "real_tx_z": run.real_tx_z,
        "real_tx_x_min": run.real_tx_x_min,
        "real_tx_x_max": run.real_tx_x_max,
    }

    cand_entries = []
    for cand in run.candidates:
        name = f"cand_{cand.point.index:04d}"
        np.savez_compressed(
            cand_dir / f"{name}.npz",
            aper_axis=_f64(cand.aper_axis),
            aper_profile=_c64(cand.aper_profile),
            loss_full=np.asarray(cand.loss_full, dtype=np.float32),
        )
        cmeta = {
            "index": cand.point.index,
            "location": {"z": cand.point.z, "x_center": cand.point.x_center,
                         "x_min": cand.point.x_min, "x_max": cand.point.x_max,
                         "dx": cand.point.dx},
            "gs_result": {"final_loss": cand.final_loss, "n_iters_run": cand.n_iters_run,
                          "stop_reason": cand.stop_reason, "seed": cand.seed},
            "freq_hz": run.freq,
            "wavelength_m": run.wavelength,
            "ground_truth": ground_truth,
            "npz": f"{name}.npz",
        }
        with open(cand_dir / f"{name}.json", "w") as f:
            json.dump(cmeta, f, indent=2, default=_json_safe)
        cand_entries.append({
            "index": cand.point.index, "z": cand.point.z, "x_center": cand.point.x_center,
            "x_min": cand.point.x_min, "x_max": cand.point.x_max,
            "final_loss": cand.final_loss, "n_iters_run": cand.n_iters_run,
            "stop_reason": cand.stop_reason,
            "npz": f"candidates/{name}.npz", "json": f"candidates/{name}.json",
        })

    skipped_entries = [{"index": p.index, "z": p.z, "x_center": p.x_center,
                        "skip_reason": p.skip_reason}
                       for p in run.grid_points if not p.ok]
    n_usable = sum(1 for p in run.grid_points if p.ok)

    manifest = {
        "schema_version": 1,
        "run_dir": str(run_dir),
        "freq_hz": run.freq,
        "wavelength_m": run.wavelength,
        "seed": run.seed,
        "grid_spec": {
            "z": {"min": run.grid_cfg.z.min, "max": run.grid_cfg.z.max, "num": run.grid_cfg.z.num},
            "x_center": {"min": run.grid_cfg.x_center.min, "max": run.grid_cfg.x_center.max,
                         "num": run.grid_cfg.x_center.num},
            "aperture": {"width": run.grid_cfg.aperture.width, "dx": run.grid_cfg.aperture.dx},
        },
        "scene_bounds": {"x_min": run.scene_bounds[0], "x_max": run.scene_bounds[1],
                         "z_min": run.scene_bounds[2], "z_max": run.scene_bounds[3]},
        "ground_truth": ground_truth,
        "gs": {"effective_max_iters": run.effective_max_iters},
        "provenance": {
            "cli_args": {k: (str(v) if isinstance(v, Path) else v)
                         for k, v in (args_dict or {}).items()},
            "package_version": __version__,
        },
        "counts": {"total": len(run.grid_points), "usable": n_usable,
                   "ran": len(run.candidates), "skipped": len(skipped_entries)},
        "candidates": cand_entries,
        "skipped": skipped_entries,
    }
    with open(run_dir / "candidate_beams.json", "w") as f:
        json.dump(manifest, f, indent=2, default=_json_safe)

    # config snapshots: effective (defaults filled) + raw source (preserves comments)
    with open(run_dir / "config_snapshot.json", "w") as f:
        json.dump(config.model_dump(mode="json"), f, indent=2, default=_json_safe)
    try:
        shutil.copyfile(config_path, run_dir / "config_source.yml")
    except (OSError, TypeError):
        pass

    logging.getLogger().info(f"Saved grid run to {run_dir}")


# --------------------------------------------------------------------------- #
# Residual heatmap summary (visualises candidate "likeliness" over the grid)
# --------------------------------------------------------------------------- #
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


# --------------------------------------------------------------------------- #
# Multi-frequency run layout
# --------------------------------------------------------------------------- #
FREQ_INDEX_NAME = "frequencies.json"


def _resolve_frequencies(config: SimConfig, cli_freqs: Optional[List[float]]) -> List[float]:
    """Resolve the frequency list: CLI override -> config.frequencies -> [150e9].

    Duplicates are dropped (order-preserving): repeated frequencies would map to the
    same freq_<GHz> subdir, clobbering the earlier sweep and double-counting the layer
    in the averaged/3D plots.
    """
    if cli_freqs:
        freqs = [float(f) for f in cli_freqs]
    elif config.frequencies:
        freqs = [float(f) for f in config.frequencies]
    else:
        return [150e9]
    return list(dict.fromkeys(freqs))


def _freq_dir_name(freq: float) -> str:
    """Per-frequency subdirectory name, e.g. 140e9 -> 'freq_140GHz'."""
    return f"freq_{freq / 1e9:g}GHz"


def write_frequencies_index(base_dir: Path, entries: List[dict],
                            real_tx_z: float, real_tx_x_center: float) -> Path:
    """Write the top-level index tying a multi-frequency run's per-frequency subdirs together.

    `entries` is a list of {freq_hz, wavelength_m, dir} dicts (one per frequency).
    """
    out = Path(base_dir) / FREQ_INDEX_NAME
    payload = {
        "schema_version": 1,
        "frequencies": entries,
        "ground_truth": {"real_tx_z": float(real_tx_z),
                         "real_tx_x_center": float(real_tx_x_center)},
    }
    with open(out, "w") as f:
        json.dump(payload, f, indent=2)
    return out


def load_frequencies_index(base_dir: Path) -> Optional[dict]:
    """Load a multi-frequency run's frequencies.json, or None if this is a flat single-freq run."""
    idx = Path(base_dir) / FREQ_INDEX_NAME
    if not idx.exists():
        return None
    with open(idx) as f:
        return json.load(f)


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


# --------------------------------------------------------------------------- #
# Residual scatter summary (distribution of candidate residuals)
# --------------------------------------------------------------------------- #
def _candidate_points(summary: ResidualSummary):
    """Flatten a ResidualSummary's loss grid into per-candidate arrays.

    Returns (z, x_center, loss, dist) for every grid cell that was actually run
    (finite residual). `dist` is the Euclidean distance in the (z, x_center) plane
    from the candidate to the true TX location; z and x_center share units (metres),
    so the distance is physically meaningful. enumerate_grid uses z-outer/x-inner
    ordering, which is exactly meshgrid's "ij" indexing, so cells map back to the
    correct (z, x_center) without float matching.
    """
    zz, xx = np.meshgrid(summary.z_values, summary.x_values, indexing="ij")
    finite = np.isfinite(summary.loss_grid)
    z = zz[finite]
    x = xx[finite]
    loss = summary.loss_grid[finite]
    dist = np.hypot(z - summary.real_tx_z, x - summary.real_tx_x_center)
    return z, x, loss, dist


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
    _, _, loss, dist = _candidate_points(summary)
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


# --------------------------------------------------------------------------- #
# 3D residual scatter (the 2D residual data, stacked along a frequency axis)
# --------------------------------------------------------------------------- #
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
        z, x, loss, _ = _candidate_points(summary)
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


# --------------------------------------------------------------------------- #
# Single "true" MGS run (baseline reconstruction at the KNOWN TX location)
# --------------------------------------------------------------------------- #
def _load_run_config(run_dir: Path, log: logging.Logger) -> Optional[SimConfig]:
    """Load a saved run's config: prefer the verbatim source .yml, fall back to the
    effective snapshot JSON. Returns None if neither is present."""
    src = run_dir / "config_source.yml"
    if src.exists():
        return load_config(src)
    snap = run_dir / "config_snapshot.json"
    if snap.exists():
        with open(snap) as f:
            return SimConfig.model_validate(json.load(f))
    log.warning(f"No config_source.yml/config_snapshot.json in {run_dir}; "
                "cannot build the true MGS run")
    return None


def make_true_mgs_plot(run_dir: Path, log: Optional[logging.Logger] = None) -> Optional[Path]:
    """Reconstruct + plot the single "true" MGS run for a saved grid run.

    The grid search treats the TX as unknown; this is the baseline plain-`mgs` result at
    the KNOWN (true) TX location. Using the run's saved config + frequency, it illuminates
    the real scene, runs MGS at the true TX plane, re-illuminates with the reconstruction,
    and saves the 4-panel scene plot (identical to what `mgs` produces). This recomputes a
    full MGS solve, so it is more expensive than the manifest-only residual plots.

    Output: <run_dir>/true_mgs_scene.png. Returns the path, or None if the config is
    missing from the run dir.
    """
    log = log or logging.getLogger()
    run_dir = Path(run_dir)
    with open(run_dir / "candidate_beams.json") as f:
        manifest = json.load(f)
    freq = float(manifest["freq_hz"])
    config = _load_run_config(run_dir, log)
    if config is None:
        return None

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
    log.info(f"Reconstructing the true MGS run (TX known) at {freq/1e9:.3g} GHz -> {out_path}")
    mgs = MGS(freq, config)
    mgs.run_sim(gs_rec=False)     # illuminate the real scene
    mgs.run_gerch_sax()           # MGS solve at the true TX plane
    mgs.run_sim(gs_rec=True)      # re-illuminate with the reconstructed aperture
    mgs.plot_scene(save_path=out_path, show=False)
    log.info(f"Wrote true MGS scene to {out_path}")
    return out_path


# --------------------------------------------------------------------------- #
# Candidate-beam scenes (re-illuminate the scene with each candidate aperture)
# --------------------------------------------------------------------------- #
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
    # scene (energy from outside the aperture); match MGS run_sim's fill_value=0.
    ap = (np.interp(x_axis, aper_axis, aper_profile.real, left=0.0, right=0.0)
          + 1j * np.interp(x_axis, aper_axis, aper_profile.imag, left=0.0, right=0.0))
    field = np.abs(rs.rs(x_axis, z_axis, ap, wavelength, z_src=tx_z, forward_dir=-1.0))
    field[z_axis > tx_z, :] = 0.0
    return field.astype(np.float32)


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


def scenes_from_run(run: GridSearchRun) -> SceneContext:
    """Adapter: a SceneContext from an in-memory run."""
    items = [CandidateScene(c.point.index, c.point.z, c.point.x_center, c.final_loss,
                            c.aper_axis, c.aper_profile, c.stop_reason)
             for c in run.candidates]
    return SceneContext(items, run.scene_bounds, run.wavelength, run.scene_x_axis,
                        run.real_tx_z, 0.5 * (run.real_tx_x_min + run.real_tx_x_max),
                        run.real_tx_aper_axis, run.real_tx_aper_profile, run.rx_aper_axis)


def scenes_from_manifest(run_dir: Path) -> SceneContext:
    """Adapter: a SceneContext from a saved run (reads each candidate npz)."""
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
    return SceneContext(items, scene_bounds, m["wavelength_m"], meas["scene_x_axis"],
                        gt["real_tx_z"], 0.5 * (gt["real_tx_x_min"] + gt["real_tx_x_max"]),
                        meas["real_tx_aper_axis"], meas["real_tx_aper_profile"],
                        meas["rx_aper_axis"])


def _draw_scene(fig, ax, field: np.ndarray, ctx: SceneContext, tx_axis: np.ndarray,
                tx_z: float, vmax: float, title: str) -> None:
    """imshow a re-illuminated |field|, marking the RX aperture (red) and TX aperture (blue)."""
    x_min, x_max, z_min, z_max = ctx.scene_bounds
    im = ax.imshow(field, extent=[x_min, x_max, z_min, z_max], origin="lower",
                   aspect="auto", cmap="inferno", vmin=0.0, vmax=vmax)
    fig.colorbar(im, ax=ax, label="|field| (V/m)")
    ax.scatter(ctx.rx_aper_axis, np.full(len(ctx.rx_aper_axis), z_min), s=6, c="red",
               label="RX aperture", zorder=5)
    ax.scatter(tx_axis, np.full(len(tx_axis), tx_z), s=6, c="blue",
               label="TX aperture", zorder=5)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("z (m)")
    ax.set_title(title)
    ax.legend(loc="upper right", framealpha=0.9, markerscale=2)


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
    _draw_scene(fig, ax_cand, cand_field, ctx, item.aper_axis, item.z, vmax,
                f"Candidate #{item.index}{tag} beam — MSE={item.final_loss:.4g}{note}")

    if real_field is not None:
        _draw_scene(fig, ax_real, real_field, ctx, ctx.real_tx_aper_axis, ctx.real_tx_z,
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

    ax_phase.plot(scene_x, phase, color="C0")
    ax_phase.set_title("Candidate TX phase")
    ax_phase.set_xlabel("x (m)"); ax_phase.set_ylabel("phase unwrapped [rad]")
    ax_phase.set_xlim(x_min, x_max); ax_phase.grid(True)

    ax_amp.plot(scene_x, amp, color="C3")
    ax_amp.set_title("Candidate TX amplitude")
    ax_amp.set_xlabel("x (m)"); ax_amp.set_ylabel("amplitude (V/m)")
    ax_amp.set_xlim(x_min, x_max); ax_amp.set_ylim(0.0, 1.1); ax_amp.grid(True)

    fig.savefig(out_path, dpi=120)
    plt.close(fig)


# Per-worker shared state for parallel scene rendering, set once by the pool initializer
# so the (read-only) scene geometry + real beam are not re-pickled for every chunk.
SceneShared = namedtuple("SceneShared", "ctx real_field x_axis z_axis best_index scenes_dir")
_SCENE_WORKER: Optional["SceneShared"] = None


def _init_scene_worker(shared: "SceneShared") -> None:
    """Scene-pool initializer: stash the shared scene state, force the headless matplotlib
    backend (each worker saves its own PNG), and pin math threads (precautionary)."""
    global _SCENE_WORKER
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                "NUMEXPR_NUM_THREADS"):
        os.environ.setdefault(var, "1")
    try:
        plt.switch_backend("Agg")
    except Exception:
        pass
    _SCENE_WORKER = shared


def _render_scene_chunk(chunk: "List[CandidateScene]"):
    """Render every candidate in `chunk` to its PNG; return (partial field-sum, count,
    skipped). One chunk per worker keeps the returned partial sum (for the averaged scene)
    small. Re-illumination + plotting are deterministic, so each PNG is identical to the
    serial render."""
    s = _SCENE_WORKER
    acc = None
    count = 0
    skipped: List[Tuple[int, str]] = []
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
    return acc, count, skipped


def make_candidate_scenes(ctx: SceneContext, out_dir: Path, *, z_planes: int = 200,
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
    n_jobs = max(1, int(jobs))
    acc = None
    count = 0
    if n_jobs == 1 or len(items) <= 1:
        log.info(f"Rendering {len(items)} candidate scene(s) over {z_planes} z-planes "
                 f"-> {scenes_dir}/")
        for n, it in enumerate(items):
            try:
                field = _reilluminate(x_axis, z_axis, it.aper_axis, it.aper_profile,
                                      ctx.wavelength, it.z)
            except RuntimeError as e:
                log.warning(f"  candidate #{it.index}: skipped ({e})")
                continue
            _plot_candidate_quad(field, real_field, it, ctx,
                                 scenes_dir / f"cand_{it.index:04d}.png",
                                 best=(it.index == best.index))
            acc = field.astype(np.float64) if acc is None else acc + field
            count += 1
            if (n + 1) % 10 == 0:
                log.info(f"  {n + 1}/{len(items)} rendered")
    else:
        n_workers = min(n_jobs, len(items))
        # round-robin chunks: each re-illumination costs the same, so striding the
        # candidates across workers keeps the load balanced, and one chunk per worker
        # keeps the returned partial sums small.
        chunks = [items[i::n_workers] for i in range(n_workers)]
        shared = SceneShared(ctx=ctx, real_field=real_field, x_axis=x_axis, z_axis=z_axis,
                             best_index=best.index, scenes_dir=scenes_dir)
        log.info(f"Rendering {len(items)} candidate scene(s) over {z_planes} z-planes "
                 f"across {n_workers} worker process(es) -> {scenes_dir}/")
        with ProcessPoolExecutor(max_workers=n_workers, initializer=_init_scene_worker,
                                 initargs=(shared,)) as ex:
            futures = [ex.submit(_render_scene_chunk, ch) for ch in chunks]
            for done, fut in enumerate(as_completed(futures), start=1):
                pacc, pcount, skipped = fut.result()
                if pacc is not None:
                    acc = pacc if acc is None else acc + pacc
                count += pcount
                for idx, msg in skipped:
                    log.warning(f"  candidate #{idx}: skipped ({msg})")
                log.info(f"  worker chunk {done}/{n_workers} done ({count} rendered so far)")

    log.info(f"Wrote {count} candidate scene(s) to {scenes_dir}/")
    if average and acc is not None:
        avg = (acc / count).astype(np.float32)
        fig, ax = plt.subplots(figsize=(8, 8), layout="constrained")
        _draw_scene(fig, ax, avg, ctx, ctx.real_tx_aper_axis, ctx.real_tx_z,
                    float(avg.max()) or 1.0,
                    f"Average of {count} candidate beams (TX shown = real)")
        avg_path = out_dir / "scene_average.png"
        fig.savefig(avg_path, dpi=120)
        plt.close(fig)
        log.info(f"Wrote averaged scene to {avg_path}")


# --------------------------------------------------------------------------- #
# Candidate-beam animation (one frame per candidate re-illuminating the scene)
# --------------------------------------------------------------------------- #
# Per-worker shared state for parallel candidate-beam frame re-illumination.
_AnimShared = namedtuple("_AnimShared", "x_axis z_axis wavelength")
_ANIM_WORKER: Optional["_AnimShared"] = None


def _init_anim_worker(shared: "_AnimShared") -> None:
    global _ANIM_WORKER
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ.setdefault(var, "1")
    _ANIM_WORKER = shared


def _anim_frame(item: "CandidateScene"):
    """Re-illuminate one candidate into a scene field (worker task); None if undersampled."""
    s = _ANIM_WORKER
    try:
        return _reilluminate(s.x_axis, s.z_axis, item.aper_axis, item.aper_profile,
                             s.wavelength, item.z)
    except RuntimeError:
        return None


def animate_candidate_beams(ctx: SceneContext, out_path: Path, *, z_planes: int = 200,
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
    n_jobs = max(1, int(jobs))
    frames, frame_items = [], []
    vmax = 0.0
    if n_jobs == 1 or len(items) <= 1:
        log.info(f"Propagating {len(items)} candidate frame(s) over {z_planes} z-planes")
        for n, it in enumerate(items):
            try:
                f = _reilluminate(x_axis, z_axis, it.aper_axis, it.aper_profile, ctx.wavelength, it.z)
            except RuntimeError as e:
                log.warning(f"  candidate #{it.index}: skipped ({e})")
                continue
            frames.append(f); frame_items.append(it); vmax = max(vmax, float(f.max()))
            if (n + 1) % 10 == 0:
                log.info(f"  {n + 1}/{len(items)} propagated")
    else:
        n_workers = min(n_jobs, len(items))
        log.info(f"Propagating {len(items)} candidate frame(s) over {z_planes} z-planes "
                 f"across {n_workers} worker process(es)")
        shared = _AnimShared(x_axis=x_axis, z_axis=z_axis, wavelength=ctx.wavelength)
        with ProcessPoolExecutor(max_workers=n_workers, initializer=_init_anim_worker,
                                 initargs=(shared,)) as ex:
            results = list(ex.map(_anim_frame, items))  # map preserves grid/best order
        for it, f in zip(items, results):
            if f is None:
                log.warning(f"  candidate #{it.index}: skipped (RS undersampled)")
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
    return _write_mp4(anim, fig, out_path, fps, dpi, show, log)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _emit_heatmaps(summary: ResidualSummary, out_dir: Path, log) -> None:
    """Write the residual heatmap plus its high-contrast (log-scale) twin."""
    heat_out = out_dir / "residual_heatmap.png"
    plot_residual_heatmap(summary, heat_out)
    plot_residual_heatmap(summary, out_dir / "residual_heatmap_hc.png",
                          title="Candidate residual (high contrast, log scale)",
                          norm=_hc_norm(summary.loss_grid))
    log.info(f"Wrote residual heatmap (+hc) to {heat_out}")


def _emit_scatters(summary: ResidualSummary, out_dir: Path, log) -> None:
    """Write the residual scatter plus its high-contrast (log-y) twin."""
    sc_out = out_dir / "residual_scatter.png"
    plot_residual_scatter(summary, sc_out)
    plot_residual_scatter(summary, out_dir / "residual_scatter_hc.png",
                          title="Candidate residual distribution (high contrast)",
                          log_scale=True)
    log.info(f"Wrote residual scatter (+hc) to {sc_out}")


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
        # Reuse the flattening helper on a summary carrying the difference grid.
        tmp = ResidualSummary(s.z_values, s.x_values, d,
                              s.real_tx_z, s.real_tx_x_center, None)
        z, x, dval, _ = _candidate_points(tmp)
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


def _emit_multifreq_plots(summaries: List[Tuple[float, ResidualSummary]],
                          base_dir: Path, log) -> None:
    """Top-level plots for a multi-frequency run: the frequency-averaged residual
    heatmap and the 3D residual scatter, each with a high-contrast (log-scale) twin."""
    avg = average_summary(summaries)
    avg_out = base_dir / "residual_heatmap_avg.png"
    plot_residual_heatmap(avg, avg_out, title="Average residual across frequencies")
    plot_residual_heatmap(avg, base_dir / "residual_heatmap_avg_hc.png",
                          title="Average residual across frequencies (high contrast, log scale)",
                          norm=_hc_norm(avg.loss_grid))
    log.info(f"Wrote frequency-averaged residual heatmap (+hc) to {avg_out}")
    out3d = base_dir / "residual_scatter_3d.png"
    plot_residual_scatter_3d(summaries, out3d)
    plot_residual_scatter_3d(summaries, base_dir / "residual_scatter_3d_hc.png",
                             title="Candidate residual across frequency (high contrast, log scale)",
                             norm=_hc_norm(np.stack([s.loss_grid for _, s in summaries])))
    log.info(f"Wrote 3D residual scatter (+hc) to {out3d}")
    if len(summaries) > 1:
        out3d_diff = base_dir / "residual_scatter_3d_diff.png"
        plot_residual_scatter_3d_diff(summaries, out3d_diff)
        log.info(f"Wrote 3D residual-difference scatter to {out3d_diff}")
    # Per-frequency vs-average comparison (freq | average | difference) in each freq dir.
    for freq_hz, s in summaries:
        sub = base_dir / _freq_dir_name(freq_hz)
        if sub.is_dir():
            plot_residual_freq_vs_avg(freq_hz, s, avg, sub / "residual_heatmap_vs_avg.png")
            plot_residual_freq_vs_avg(freq_hz, s, avg,
                                      sub / "residual_heatmap_vs_avg_hc.png", hc=True)
    log.info(f"Wrote {len(summaries)} per-frequency vs-average comparison(s) (+hc) under {base_dir}")


def _replot_one_dir(run_dir: Path, args, log) -> ResidualSummary:
    """Regenerate one saved run dir's plots from disk: residual heatmap + scatter with
    their high-contrast twins (cheap, manifest-only), the baseline true-MGS scene (one
    full solve; skipped with --skip-true-mgs), and — when --scenes/--anim are set — the
    per-candidate scenes/animation. Returns the ResidualSummary."""
    run_dir = Path(run_dir)
    summary = summary_from_manifest(run_dir)
    _emit_heatmaps(summary, run_dir, log)
    _emit_scatters(summary, run_dir, log)
    # Baseline single "true" MGS run at the known TX (recomputes one MGS solve).
    if not args.skip_true_mgs:
        make_true_mgs_plot(run_dir, log=log)
    if args.scenes or args.anim:
        ctx = scenes_from_manifest(run_dir)
        if args.scenes:
            make_candidate_scenes(ctx, run_dir, z_planes=args.scene_z_planes,
                                  top=args.scene_top, jobs=args.jobs, log=log)
        if args.anim:
            animate_candidate_beams(ctx, run_dir / "candidate_beams.mp4",
                                    z_planes=args.scene_z_planes, top=args.scene_top,
                                    fps=args.fps, jobs=args.jobs, log=log)
    return summary


def _run_one_frequency(config: SimConfig, freq: float, make_out_dir, args, log):
    """Run the whole sweep at one frequency, save it into make_out_dir(), emit its per-run
    2D plots (and scenes/anim when requested), and return (run, ResidualSummary).

    Used for both the single-frequency flat run and each layer of a multi-frequency run,
    so the two paths stay behaviourally identical per frequency. `make_out_dir` is a
    callable invoked only AFTER the sweep completes — creating (and clearing) the run
    directory is deferred so an interrupted sweep leaves prior saved results intact.
    """
    run = run_grid_search(config, freq, limit=args.limit, jobs=args.jobs, log=log)
    out_dir = make_out_dir()
    save_grid_run(run, out_dir, config, args.config, vars(args))
    log.info(f"Saved {len(run.candidates)} candidate beam(s) to {out_dir}")
    summary = summary_from_run(run)
    if args.summary:
        _emit_heatmaps(summary, out_dir, log)
    if args.scatter:
        _emit_scatters(summary, out_dir, log)
    if args.scenes or args.anim:
        ctx = scenes_from_run(run)
        if args.scenes:
            make_candidate_scenes(ctx, out_dir, z_planes=args.scene_z_planes,
                                  top=args.scene_top, jobs=args.jobs, log=log)
        if args.anim:
            animate_candidate_beams(ctx, out_dir / "candidate_beams.mp4",
                                    z_planes=args.scene_z_planes, top=args.scene_top,
                                    fps=args.fps, jobs=args.jobs, log=log)
    return run, summary


def main():
    parser = argparse.ArgumentParser(
        description="Grid search over speculative TX locations, running MGS at each "
                    "to emit candidate beams"
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG,
                        help="Path to a simulation config .yml with a grid_search block")
    parser.add_argument("--freq", "-f", type=float, nargs="+", default=None,
                        help="One or more frequencies in Hz (overrides config `frequencies`). "
                             "Multiple values run the whole sweep independently per frequency. "
                             "Default: config `frequencies`, else 150e9.")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Override output.output_dir (base dir for the run folder)")
    parser.add_argument("--run-name", type=str, default=None,
                        help="Run directory name under output_dir (default: grid_search)")
    parser.add_argument("--out", "-o", type=Path, default=None,
                        help="Run directory to write this run into, named outright "
                             "(overrides --output-dir + --run-name)")
    parser.add_argument("--seed", type=int, default=None,
                        help="Override grid_search.seed (fixed GS seed across candidates)")
    parser.add_argument("--max-iters", type=int, default=None,
                        help="Override grid_search.gs_overrides.max_iters for the sweep")
    parser.add_argument("--limit", type=int, default=None,
                        help="Only run the first N usable candidates (for quick tests)")
    parser.add_argument("--jobs", "-j", type=int, default=os.cpu_count(),
                        help="Worker processes for the candidate sweep and scene rendering "
                             "(default: all cores; 1 = serial). Per-candidate results are "
                             "identical regardless of value.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Enumerate the grid and print a summary without running MGS")
    parser.add_argument("--no-save", action="store_true", help="Disable persistence")
    parser.add_argument("--summary", action="store_true",
                        help="Write a residual-vs-(z, x_center) heatmap (residual_heatmap.png)")
    parser.add_argument("--scatter", action="store_true",
                        help="Write a residual-vs-distance-to-true-TX scatter plot (residual_scatter.png)")
    parser.add_argument("--replot", type=Path, default=None,
                        help="Re-plot the residual heatmap from a saved run dir, then exit")
    parser.add_argument("--skip-true-mgs", action="store_true",
                        help="[replot] skip recomputing the true-MGS baseline scene "
                             "(a full MGS solve per run dir); existing plots are left as-is")
    parser.add_argument("--scenes", action="store_true",
                        help="Render one PNG per candidate beam (scenes/) plus an averaged scene (scene_average.png)")
    parser.add_argument("--scene-top", type=int, default=None,
                        help="[scenes] only render the N lowest-residual candidates")
    parser.add_argument("--scene-z-planes", type=int, default=200,
                        help="[scenes/anim] z-planes per re-illumination (lower = faster/coarser)")
    parser.add_argument("--anim", action="store_true",
                        help="Animate the candidate beams to candidate_beams.mp4 (one frame per candidate, ffmpeg)")
    parser.add_argument("--fps", type=int, default=10, help="[anim] frames per second")
    parser.add_argument("--debug", action="store_true", help="Enable debug logs")
    args = parser.parse_args()

    level = "DEBUG" if args.debug else "INFO"
    coloredlogs.install(level=level, fmt="%(levelname)s: %(message)s")
    log = logging.getLogger()

    if args.replot is not None:
        base = Path(args.replot)
        freq_index = load_frequencies_index(base)
        if freq_index is not None:
            # Multi-frequency run: replot every per-frequency subdir, then the
            # top-level frequency-averaged heatmap + 3D scatter (each with hc twins).
            summaries = []
            for entry in freq_index["frequencies"]:
                sub = base / entry["dir"]
                summaries.append((float(entry["freq_hz"]), _replot_one_dir(sub, args, log)))
            _emit_multifreq_plots(summaries, base, log)
            return
        # Single-frequency (flat) run.
        _replot_one_dir(base, args, log)
        return

    config = load_config(args.config)
    if config.grid_search is None:
        log.error(f"Config {args.config} has no `grid_search` block")
        raise SystemExit(2)

    # CLI overrides onto the config
    if args.seed is not None:
        config.grid_search.seed = args.seed
    if args.max_iters is not None:
        config.grid_search.gs_overrides.max_iters = args.max_iters
    if args.output_dir is not None:
        config.output.output_dir = args.output_dir
    run_name = args.run_name or config.output.run_name or "grid_search"
    if args.out is not None:
        config.output.output_dir = Path(args.out).parent
        run_name = Path(args.out).name
    freqs = _resolve_frequencies(config, args.freq)

    if args.dry_run:
        # The grid enumeration only depends on frequency via the RS-undersampling check;
        # report at the first frequency (the others differ only in that check).
        if len(freqs) > 1:
            log.info(f"Dry run: enumerating grid at {freqs[0] / 1e9:g} GHz "
                     f"(of {len(freqs)} frequencies)")
        wavelength = scipy.constants.c / freqs[0]
        points = enumerate_grid(config.grid_search, config.sim_scene, wavelength)
        log.info(grid_summary(points))
        for p in points:
            if p.ok:
                log.info(f"  #{p.index:4d} z={p.z:.3f} x_center={p.x_center:+.3f} "
                         f"window [{p.x_min:.3f}, {p.x_max:.3f}]")
            else:
                log.warning(f"  #{p.index:4d} z={p.z:.3f} x_center={p.x_center:+.3f} "
                            f"SKIP: {p.skip_reason}")
        return

    persist = not (args.no_save or not config.output.save_run)
    if persist or len(freqs) > 1:
        # Fail fast on a run-directory collision: make_run_dir is deferred until after
        # the sweep, so without this the clash surfaces only once the compute is spent.
        check_run_dir(config.output.output_dir, run_name, kind="grid")

    if len(freqs) == 1:
        # Single frequency: original flat layout (results/<run_name>/...).
        freq = freqs[0]
        if not persist:
            log.info("Persistence disabled; not writing candidates")
            run = run_grid_search(config, freq, limit=args.limit, jobs=args.jobs, log=log)
            summary = summary_from_run(run)
            if args.summary:
                _emit_heatmaps(summary, Path("."), log)
            if args.scatter:
                _emit_scatters(summary, Path("."), log)
            if args.scenes or args.anim:
                ctx = scenes_from_run(run)
                if args.scenes:
                    make_candidate_scenes(ctx, Path("."), z_planes=args.scene_z_planes,
                                          top=args.scene_top, jobs=args.jobs, log=log)
                if args.anim:
                    animate_candidate_beams(ctx, Path("candidate_beams.mp4"),
                                            z_planes=args.scene_z_planes, top=args.scene_top,
                                            fps=args.fps, jobs=args.jobs, log=log)
            return
        _run_one_frequency(config, freq,
                           lambda: make_run_dir(config.output.output_dir, run_name,
                                                kind="grid"),
                           args, log)
        return

    # Multiple frequencies: one full sweep per frequency into results/<run_name>/freq_<GHz>/,
    # then a top-level 3D residual scatter aggregating them.
    if not persist:
        log.warning("Multi-frequency runs require saving; --no-save / save_run=false is ignored")
    base_path = Path(config.output.output_dir) / run_name
    log.info(f"Multi-frequency run: {len(freqs)} frequencies "
             f"({', '.join(f'{f / 1e9:g}' for f in freqs)} GHz) -> {base_path}")
    # The base dir is created (clearing any previous run) only when the FIRST sweep
    # finishes, and each freq subdir only after its own sweep — so an interrupted run
    # never destroys prior results without producing new ones.
    base: Optional[Path] = None

    def _get_base() -> Path:
        nonlocal base
        if base is None:
            base = make_run_dir(config.output.output_dir, run_name, kind="grid")
        return base

    entries, summaries = [], []
    real_x = real_z = 0.0
    for freq in freqs:
        log.info(f"=== frequency {freq / 1e9:g} GHz ===")
        run, summary = _run_one_frequency(
            config, freq,
            lambda f=freq: make_run_dir(_get_base(), _freq_dir_name(f), kind="grid"),
            args, log)
        entries.append({"freq_hz": float(freq), "wavelength_m": float(run.wavelength),
                        "dir": _freq_dir_name(freq)})
        summaries.append((freq, summary))
        real_x, real_z = summary.real_tx_x_center, summary.real_tx_z
    write_frequencies_index(_get_base(), entries, real_z, real_x)
    _emit_multifreq_plots(summaries, _get_base(), log)


if __name__ == "__main__":
    main()
