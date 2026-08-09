"""The speculative TX-location sweep: enumerate the grid, reconstruct a candidate
beam at every usable point, persist the result.

Deliberately free of matplotlib. Importing anything from the old grid_search
dragged in mpl_toolkits.mplot3d and FuncAnimation, so a worker process that only
ever solves MGS paid for the entire plotting stack.

The real TX location/trajectory from the config is used only to synthesize the one
shared RX measurement; the search itself treats the TX location as unknown. At each
usable grid point a fixed-width aperture (uniform assumed amplitude) is placed and
MGS reconstructs its phase against that measurement.
"""

import json
import logging
from collections import namedtuple
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from rice_bend import rs
from rice_bend.config import GridSearchConfig, SimConfig, SimSceneConfig
from rice_bend.data_store import (c64, f64, provenance, save_config_snapshot,
                                  write_json)
from rice_bend.mgs import MGS, gs_reconstruct
from rice_bend.parallel import map_workers, worker_shared
from rice_bend.sim_scene import sampled_axis

# Tolerance for inclusive bounds checks, to absorb float round-off in the sweep
# endpoints (e.g. an aperture edge landing exactly on the scene boundary).
_BOUNDS_TOL = 1e-9


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


def enumerate_grid(grid_cfg: GridSearchConfig, scene_cfg: SimSceneConfig,
                   wavelength: float) -> List[GridPoint]:
    """Enumerate the (z, x_center) grid, flagging out-of-bounds / undersampled points.

    A point is skipped (skip_reason set) when its assumed aperture window leaves
    the scene laterally, when z falls outside (z_min, z_max], or when the TX plane
    is so close to the RX plane (z_min) that rs() would reject the sampling.
    """
    half = grid_cfg.aperture.width / 2.0
    dx = grid_cfg.aperture.dx
    x_axis = sampled_axis(scene_cfg.x_min, scene_cfg.x_max, scene_cfg.spacing)
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


# Read-only measurement + GS hyperparameters shared by every candidate. It is small
# (the measurement vectors, NOT the full 2D scene), so it is cheap to hand to workers.
SharedMeasurement = namedtuple(
    "SharedMeasurement", "x_axis rx_z rx_field error_weighting wavelength params")


def _reconstruct_candidate(p: "GridPoint", shared: "SharedMeasurement") -> "CandidateResult":
    """Reconstruct one candidate beam at grid point `p`. Pure given `shared` — runs
    identically in the parent process or a worker."""
    x_axis = shared.x_axis
    # uniform assumed amplitude over the hypothesized window (support = window)
    support = (x_axis >= p.x_min) & (x_axis <= p.x_max)
    assumed_amp = np.where(support, 1.0, 0.0)
    result = gs_reconstruct(
        tx_z=p.z, orig_aper_amp=assumed_amp, x_axis=x_axis, rx_z=shared.rx_z,
        rx_field=shared.rx_field, error_weighting=shared.error_weighting,
        wavelength=shared.wavelength, params=shared.params,
        capture=False, log=None,
    )
    # Stored on the SCENE grid, sliced to the window. curr_aper_f already lives on
    # x_axis and is already zero outside the support, so resampling it down to a
    # separate aperture axis -- only so a CandidateResult.aper_axis existed -- was
    # pure loss, undone again by _reilluminate resampling it back up and a third
    # time by _on_scene_axis.
    #
    # The slice is mandatory, not tidiness: storing the full scene axis would paint
    # the blue TX marker across the entire scene in every candidate PNG and mp4
    # frame, and would collapse _on_scene_axis's NaN gaps.
    return CandidateResult(
        point=p,
        aper_axis=x_axis[support].copy(),
        aper_profile=result.curr_aper_f[support].copy(),
        final_loss=result.final_loss,
        n_iters_run=int(result.n_iters_run),
        stop_reason=str(result.stop_reason),
        seed=int(result.seed),
        loss_full=np.asarray(result.loss_full, dtype=np.float32),
    )


def _worker_task(p: "GridPoint") -> "CandidateResult":
    """Task entry point: reconstruct one candidate against the shared measurement.

    Deliberately a two-line wrapper rather than folding the global into
    _reconstruct_candidate: the shared state is transport, not a dependency of the
    physics, and _reconstruct_candidate's "pure given `shared`" stays true."""
    return _reconstruct_candidate(p, worker_shared())


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

    # 1. the single shared RX measurement. measure() synthesizes it by propagating
    #    to the RX plane alone -- the sweep never reads mgs.scene.data, so illuminating
    #    all 3400 planes here cost 1.71 s and +190 MB RSS per frequency, held in the
    #    parent for the whole sweep, to keep one row.
    mgs = MGS(freq, config)
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
        # snapshot, not an alias: the workers must not see a later mutation
        params=mgs.gs_cfg.model_copy(),
    )

    def _log_done(done: int, total: int, cand: CandidateResult) -> None:
        p = cand.point
        log.info(f"[{done}/{total}] #{p.index} z={p.z:.3f} x={p.x_center:+.3f} "
                 f"-> loss {cand.final_loss:.6g} ({cand.n_iters_run} iters, {cand.stop_reason})")

    # map_workers returns input order, so the manifest is stable without a re-sort.
    candidates: List[CandidateResult] = map_workers(
        _worker_task, usable, jobs=jobs, shared=shared, on_done=_log_done, log=log,
        desc=f"Reconstructing {len(usable)} candidate(s)")

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
        scene_x_axis=f64(run.scene_x_axis),
        rx_field=c64(run.rx_field),
        error_weighting=f64(run.error_weighting),
        rx_aper_axis=f64(run.rx_aper_axis),
        rx_aper_profile=c64(run.rx_aper_profile),
        real_tx_aper_axis=f64(run.real_tx_aper_axis),
        real_tx_aper_profile=c64(run.real_tx_aper_profile),
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
            aper_axis=f64(cand.aper_axis),
            aper_profile=c64(cand.aper_profile),
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
        write_json(cand_dir / f"{name}.json", cmeta)
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
        "schema_version": 2,
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
        "provenance": provenance(args_dict),
        "counts": {"total": len(run.grid_points), "usable": n_usable,
                   "ran": len(run.candidates), "skipped": len(skipped_entries)},
        "candidates": cand_entries,
        "skipped": skipped_entries,
    }
    write_json(run_dir / "candidate_beams.json", manifest)
    save_config_snapshot(run_dir, config, config_path)

    logging.getLogger().info(f"Saved grid run to {run_dir}")


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
    payload = {
        "schema_version": 2,
        "frequencies": entries,
        "ground_truth": {"real_tx_z": float(real_tx_z),
                         "real_tx_x_center": float(real_tx_x_center)},
    }
    return write_json(Path(base_dir) / FREQ_INDEX_NAME, payload)


def load_frequencies_index(base_dir: Path) -> Optional[dict]:
    """Load a multi-frequency run's frequencies.json, or None if this is a flat single-freq run."""
    idx = Path(base_dir) / FREQ_INDEX_NAME
    if not idx.exists():
        return None
    with open(idx) as f:
        return json.load(f)
