"""The speculative TX-location sweep: enumerate the grid, reconstruct a candidate
beam at every usable point, persist the result.

Deliberately free of matplotlib. Importing anything from the old grid_search
dragged in mpl_toolkits.mplot3d and FuncAnimation, so a worker process that only
ever solves MGS paid for the entire plotting stack.

The real TX location/trajectory from the config is used only to synthesize the
shared RX measurement set (one per frequency); the search itself treats the TX
location as unknown. At each usable grid point a fixed-width aperture (uniform
assumed amplitude) is placed and MGS reconstructs ONE phase for it jointly against
every frequency's measurement the point is valid for.
"""

import json
import logging
from collections import namedtuple
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from rice_bend import rs
from rice_bend.config import (GridSearchConfig, SimConfig, SimSceneConfig,
                              center_freq_index)
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
    """One speculative TX location. `skip_reason` is None when the point is usable
    — meaning valid at AT LEAST ONE of the requested frequencies. `freq_ok` (set
    for every geometrically-valid point) says which: freq_ok[i] is whether the RS
    sampling check passes at wavelengths[i], in enumerate_grid's wavelength order."""
    index: int
    z: float
    x_center: float
    x_min: float
    x_max: float
    dx: float
    skip_reason: Optional[str] = None
    freq_ok: Optional[List[bool]] = None

    @property
    def ok(self) -> bool:
        return self.skip_reason is None


def enumerate_grid(grid_cfg: GridSearchConfig, scene_cfg: SimSceneConfig,
                   wavelengths: List[float]) -> List[GridPoint]:
    """Enumerate the (z, x_center) grid, flagging out-of-bounds / undersampled points.

    A point is skipped (skip_reason set) when its assumed aperture window leaves
    the scene laterally, when z falls outside (z_min, z_max], or when the TX plane
    is so close to the RX plane (z_min) that rs() would reject the sampling at
    EVERY requested frequency. The RS check is per-frequency (shorter wavelengths
    are stricter, so high frequencies drop out first as z approaches the RX
    plane); a point valid at only a subset stays usable, with the subset recorded
    in `freq_ok` so the joint solve runs over exactly those frequencies.
    """
    half = grid_cfg.aperture.width / 2.0
    dx = grid_cfg.aperture.dx
    x_axis = sampled_axis(scene_cfg.x_min, scene_cfg.x_max, scene_cfg.spacing)
    rx_plane = np.array([scene_cfg.z_min])  # RX/origin plane the field propagates to

    points: List[GridPoint] = []
    index = 0
    for z in grid_cfg.z.values():
        # sampling quality depends only on (z, wavelength) given the shared scene
        # x-axis, so it is one row per z plane, not one call per grid point
        qualities = [rs.sampling_quality(x_axis, rx_plane, wl, z_src=z, forward_dir=-1.0)
                     for wl in wavelengths]
        freq_ok_row = [q >= 1.0 for q in qualities]
        for x_center in grid_cfg.x_center.values():
            x_min = x_center - half
            x_max = x_center + half
            reason: Optional[str] = None
            freq_ok: Optional[List[bool]] = None
            if x_min < scene_cfg.x_min - _BOUNDS_TOL or x_max > scene_cfg.x_max + _BOUNDS_TOL:
                reason = (f"aperture window [{x_min:.3f}, {x_max:.3f}] leaves scene x "
                          f"[{scene_cfg.x_min}, {scene_cfg.x_max}]")
            elif z <= scene_cfg.z_min or z > scene_cfg.z_max + _BOUNDS_TOL:
                reason = f"z {z:.3f} outside scene z ({scene_cfg.z_min}, {scene_cfg.z_max}]"
            else:
                freq_ok = list(freq_ok_row)
                if not any(freq_ok):
                    if len(wavelengths) == 1:
                        reason = (f"z {z:.3f} too close to RX: RS undersampled "
                                  f"(quality {qualities[0]:.3f} < 1)")
                    else:
                        reason = (f"z {z:.3f} too close to RX: RS undersampled at all "
                                  f"{len(wavelengths)} frequencies "
                                  f"(best quality {max(qualities):.3f} < 1)")
            points.append(GridPoint(index, float(z), float(x_center),
                                    float(x_min), float(x_max), float(dx), reason,
                                    freq_ok))
            index += 1
    return points


def grid_summary(points: List[GridPoint]) -> str:
    """One-line human summary of an enumerated grid."""
    usable = sum(1 for p in points if p.ok)
    return f"{len(points)} grid points ({usable} usable, {len(points) - usable} skipped)"


# Read-only measurement + GS hyperparameters shared by every candidate. It is small
# (the per-frequency measurement vectors, NOT the full 2D scene), so it is cheap to
# hand to workers. `channels` is the list of FreqChannel payloads from
# MGS.measurement_channels(), in config frequency order. `ref_freq` is the RUN-level
# reference frequency for the delay phase model: candidates solving on per-candidate
# valid SUBSETS must all share it so their psi profiles have the same units.
SharedMeasurement = namedtuple(
    "SharedMeasurement", "x_axis rx_z params channels ref_freq")


def _reconstruct_candidate(p: "GridPoint", shared: "SharedMeasurement") -> "CandidateResult":
    """Reconstruct one candidate beam at grid point `p`, jointly over the
    frequencies where `p` passes the RS sampling check. Pure given `shared` — runs
    identically in the parent process or a worker.

    The solver sees only the valid subset of channels; its per-frequency losses
    are scattered back into a full-length vector (NaN where invalid) so every
    candidate's `per_freq_losses` aligns with the run's frequency list. The joint
    `final_loss` is then the mean over the valid subset — which is exactly the
    per-cell nanmean the old per-frequency layout computed after the fact.
    """
    x_axis = shared.x_axis
    # uniform assumed amplitude over the hypothesized window (support = window)
    support = (x_axis >= p.x_min) & (x_axis <= p.x_max)
    assumed_amp = np.where(support, 1.0, 0.0)
    n_freq = len(shared.channels)
    freq_ok = p.freq_ok if p.freq_ok is not None else [True] * n_freq
    channels = [ch for ch, ok in zip(shared.channels, freq_ok) if ok]
    result = gs_reconstruct(
        tx_z=p.z, orig_aper_amp=assumed_amp, x_axis=x_axis, rx_z=shared.rx_z,
        channels=channels, params=shared.params, ref_freq=shared.ref_freq,
        capture=False, log=None,
    )
    per_freq_losses = np.full(n_freq, np.nan)
    per_freq_losses[[i for i, ok in enumerate(freq_ok) if ok]] = result.final_loss_per_freq
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
        per_freq_losses=per_freq_losses,
        freq_valid=[bool(ok) for ok in freq_ok],
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
    """A single reconstructed candidate beam at a hypothesized TX location.

    One aperture per candidate: the solved phase mask is shared across all
    frequencies (achromatic) and the assumed amplitude is the same uniform box at
    every frequency, so there is exactly one complex profile whatever F is.
    `final_loss` is the joint (mean-over-valid-frequencies) residual;
    `per_freq_losses` aligns with the run's frequency list, NaN where the
    candidate failed that frequency's RS sampling check (`freq_valid`)."""
    point: GridPoint
    aper_axis: np.ndarray        # hypothesized aperture x axis
    aper_profile: np.ndarray     # complex reconstructed aperture (shared across F)
    final_loss: float            # joint: mean over the valid frequencies
    per_freq_losses: np.ndarray  # (F,), NaN where the frequency was invalid
    freq_valid: List[bool]       # which frequencies contributed
    n_iters_run: int
    stop_reason: str
    seed: int
    loss_full: np.ndarray        # per-iteration JOINT loss curve


@dataclass
class GridSearchRun:
    """Everything needed to persist + later rank a grid search.

    Per-frequency arrays are stacked (F, ...) where the lengths agree (they are
    all on the scene x-axis or the shared TX axis) and kept as ragged lists only
    for the RX element arrays, whose counts genuinely differ per frequency
    (dx defaults to wavelength/20). `freqs` order is the alignment order."""
    freqs: List[float]
    wavelengths: List[float]
    seed: Optional[int]
    effective_max_iters: int
    phase_model: str                 # 'achromatic' | 'delay' (from gerchberg_saxton)
    ref_freq: float                  # run-level reference frequency for psi units
    grid_cfg: GridSearchConfig
    grid_points: List[GridPoint]
    candidates: List[CandidateResult]
    # shared measurement set (one per run; one row per frequency)
    scene_x_axis: np.ndarray
    rx_fields: np.ndarray            # (F, nx) on the scene x-axis
    error_weightings: np.ndarray     # (F, nx)
    rx_aper_axes: List[np.ndarray]   # ragged: per-frequency element axes
    rx_aper_profiles: List[np.ndarray]
    # ground truth, for later evaluation of how well candidates localize the TX
    real_tx_aper_axis: np.ndarray    # geometry-only, shared across frequencies
    real_tx_aper_profiles: np.ndarray  # (F, nt): beam phase ∝ k
    real_tx_z: float
    real_tx_x_min: float
    real_tx_x_max: float
    scene_bounds: Tuple[float, float, float, float]  # x_min, x_max, z_min, z_max


def run_grid_search(config: SimConfig, freqs: List[float], *, limit: Optional[int] = None,
                    jobs: int = 1, log: Optional[logging.Logger] = None) -> GridSearchRun:
    """Run MGS phase retrieval at every usable speculative TX location — ONE joint
    solve per candidate across all of `freqs` (a single frequency is the len-1
    case of the same path).

    Builds the real scene once (synthesizing the per-frequency RX measurements),
    then reconstructs a fixed-width aperture (uniform assumed amplitude) at each
    hypothesized (z, x_center) against every frequency it is valid for.
    Candidates are independent, so with `jobs > 1` they are distributed across
    worker processes; results are identical to the serial path (each candidate
    uses the same fixed seed) regardless of `jobs` or completion order.
    """
    grid_cfg = config.grid_search
    if grid_cfg is None:
        raise ValueError("config.grid_search is required for grid-search-mgs")
    log = log or logging.getLogger()

    # 1. the shared RX measurement set (one per frequency). measure() synthesizes
    #    each by propagating to the RX plane alone -- the sweep never reads
    #    mgs.scene.data, so illuminating all 3400 planes here cost 1.71 s and
    #    +190 MB RSS, held in the parent for the whole sweep, to keep one row.
    mgs = MGS(freqs, config)
    mgs.measure()

    # 2. Apply the grid-only GS overrides (cheaper sweep + one fixed seed so residuals
    #    are comparable). Written to config.gerchberg_saxton DIRECTLY and not through
    #    mgs.gs_cfg, which is an alias for the same object -- so these overrides always
    #    did mutate the caller's SimConfig, and that mutated object is what
    #    save_grid_run dumps to config_snapshot.json. That is why the snapshot records
    #    effective sweep values while config_source.yml records the file's.
    #
    #    The mutation is intentional (the snapshot SHOULD record what actually ran);
    #    only the action-at-a-distance was not. Threading a model_copy through instead
    #    would patch max_iters and forget seed, and since load_run_config falls back to
    #    the snapshot when config_source.yml is absent, reverting the seed would change
    #    true_mgs_scene.png's numerics.
    gs_cfg = config.gerchberg_saxton
    if grid_cfg.gs_overrides.max_iters is not None:
        gs_cfg.max_iters = grid_cfg.gs_overrides.max_iters
    if grid_cfg.seed is not None:
        gs_cfg.seed = grid_cfg.seed
    elif gs_cfg.seed is None:
        log.warning("grid_search.seed and gerchberg_saxton.seed are both null; candidates "
                    "will use independent random initial phases (residuals not comparable)")

    # 3. enumerate + report. Validity is per (z, frequency); a point is usable when
    #    it passes at >= 1 frequency, so `usable` (and therefore --limit's slice) is
    #    one joint set — the same candidates are attempted at every frequency by
    #    construction, which the old per-frequency sweep did not guarantee.
    wavelengths = [fs.wavelength for fs in mgs.freq_states]
    scene_cfg = config.sim_scene
    points = enumerate_grid(grid_cfg, scene_cfg, wavelengths)
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
    channels = mgs.measurement_channels()
    ref_freq = mgs.freqs[center_freq_index(mgs.freqs)]
    shared = SharedMeasurement(
        x_axis=np.asarray(x_axis).copy(),
        rx_z=float(mgs.scene.rx_ap.z),
        channels=channels,
        # snapshot, not an alias: the workers must not see a later mutation
        params=mgs.gs_cfg.model_copy(),
        ref_freq=float(ref_freq),
    )

    def _log_done(done: int, total: int, cand: CandidateResult) -> None:
        p = cand.point
        n_valid, n_f = sum(cand.freq_valid), len(cand.freq_valid)
        subset = f", {n_valid}/{n_f} freqs" if n_valid < n_f else ""
        log.info(f"[{done}/{total}] #{p.index} z={p.z:.3f} x={p.x_center:+.3f} "
                 f"-> loss {cand.final_loss:.6g} ({cand.n_iters_run} iters, "
                 f"{cand.stop_reason}{subset})")

    # map_workers returns input order, so the manifest is stable without a re-sort.
    candidates: List[CandidateResult] = map_workers(
        _worker_task, usable, jobs=jobs, shared=shared, on_done=_log_done, log=log,
        desc=f"Reconstructing {len(usable)} candidate(s)")

    real_tx = config.tx_aperture
    return GridSearchRun(
        freqs=[float(f) for f in mgs.freqs],
        wavelengths=[float(w) for w in wavelengths],
        seed=mgs.gs_cfg.seed,
        effective_max_iters=int(mgs.gs_cfg.max_iters),
        phase_model=str(mgs.gs_cfg.phase_model),
        ref_freq=float(ref_freq),
        grid_cfg=grid_cfg,
        grid_points=points,
        candidates=candidates,
        scene_x_axis=np.asarray(x_axis).copy(),
        rx_fields=np.stack([np.asarray(ch.rx_field) for ch in channels]),
        error_weightings=np.stack([np.asarray(ch.error_weighting) for ch in channels]),
        rx_aper_axes=[fs.rx_ap.aper_axis.copy() for fs in mgs.freq_states],
        rx_aper_profiles=[fs.rx_ap.aper_profile.copy() for fs in mgs.freq_states],
        real_tx_aper_axis=mgs.scene.tx_ap.aper_axis.copy(),
        real_tx_aper_profiles=np.stack([fs.tx_ap.aper_profile.copy()
                                        for fs in mgs.freq_states]),
        real_tx_z=float(real_tx.z),
        real_tx_x_min=float(real_tx.x_min),
        real_tx_x_max=float(real_tx.x_max),
        scene_bounds=(scene_cfg.x_min, scene_cfg.x_max, scene_cfg.z_min, scene_cfg.z_max),
    )


def _json_losses(per_freq_losses: np.ndarray) -> List[Optional[float]]:
    """Per-frequency losses as a JSON-safe list: NaN (frequency invalid) -> None.

    Bare NaN is not valid JSON; every reader maps None back to np.nan.
    """
    return [float(v) if np.isfinite(v) else None for v in per_freq_losses]


def save_grid_run(run: GridSearchRun, run_dir: Path, config: SimConfig,
                  config_path: Path, args_dict: dict) -> None:
    """Write the grid run to disk under run_dir — one flat directory whatever the
    frequency count (the old freq_<GHz>/ per-frequency layout is retired).

    Layout:
        run_dir/candidate_beams.json      manifest (real TX location, grid spec, frequency
                                          list, candidate index with joint + per-freq losses)
        run_dir/measurement.npz           per-frequency RX fields/weightings + apertures
        run_dir/candidates/cand_####.npz  reconstructed aperture + joint loss curve
        run_dir/candidates/cand_####.json per-candidate metadata

    The manifest's `frequencies` list order is THE alignment order for every
    per-frequency value in the run (per_freq_losses, freq_valid, the (F, ...) npz
    stacks and the rx_aper_*_NN indexed keys).
    """
    cand_dir = run_dir / "candidates"
    cand_dir.mkdir(parents=True, exist_ok=True)

    # shared measurement set (saved once; one row / indexed key per frequency)
    meas = {
        "scene_x_axis": f64(run.scene_x_axis),
        "freq_hz": f64(run.freqs),
        "wavelength_m": f64(run.wavelengths),
        "rx_field": c64(run.rx_fields),                    # (F, nx) on the scene axis
        "error_weighting": f64(run.error_weightings),      # (F, nx)
        "real_tx_aper_axis": f64(run.real_tx_aper_axis),
        "real_tx_aper_profile": c64(run.real_tx_aper_profiles),  # (F, nt)
    }
    # RX element arrays are the one genuinely ragged set (dx = wavelength/20), so
    # they get indexed keys instead of a stack — object arrays would break the
    # content hash and need allow_pickle.
    for i, (ax, prof) in enumerate(zip(run.rx_aper_axes, run.rx_aper_profiles)):
        meas[f"rx_aper_axis_{i:02d}"] = f64(ax)
        meas[f"rx_aper_profile_{i:02d}"] = c64(prof)
    np.savez_compressed(run_dir / "measurement.npz", **meas)

    ground_truth = {
        "real_tx_z": run.real_tx_z,
        "real_tx_x_min": run.real_tx_x_min,
        "real_tx_x_max": run.real_tx_x_max,
    }
    freq_entries = [{"freq_hz": float(f), "wavelength_m": float(w)}
                    for f, w in zip(run.freqs, run.wavelengths)]

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
            "gs_result": {"final_loss": cand.final_loss,
                          "per_freq_losses": _json_losses(cand.per_freq_losses),
                          "freq_valid": cand.freq_valid,
                          "n_iters_run": cand.n_iters_run,
                          "stop_reason": cand.stop_reason, "seed": cand.seed},
            "frequencies": freq_entries,
            "ground_truth": ground_truth,
            "npz": f"{name}.npz",
        }
        write_json(cand_dir / f"{name}.json", cmeta)
        cand_entries.append({
            "index": cand.point.index, "z": cand.point.z, "x_center": cand.point.x_center,
            "x_min": cand.point.x_min, "x_max": cand.point.x_max,
            "final_loss": cand.final_loss,
            "per_freq_losses": _json_losses(cand.per_freq_losses),
            "freq_valid": cand.freq_valid,
            "n_iters_run": cand.n_iters_run,
            "stop_reason": cand.stop_reason,
            "npz": f"candidates/{name}.npz", "json": f"candidates/{name}.json",
        })

    skipped_entries = [{"index": p.index, "z": p.z, "x_center": p.x_center,
                        "skip_reason": p.skip_reason}
                       for p in run.grid_points if not p.ok]
    n_usable = sum(1 for p in run.grid_points if p.ok)
    ran_per_freq = [sum(1 for c in run.candidates if c.freq_valid[i])
                    for i in range(len(run.freqs))]

    manifest = {
        "schema_version": 3,
        "run_dir": str(run_dir),
        "frequencies": freq_entries,
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
        "gs": {"effective_max_iters": run.effective_max_iters, "loss_combine": "mean",
               "phase_model": run.phase_model, "ref_freq_hz": run.ref_freq},
        "provenance": provenance(args_dict),
        "counts": {"total": len(run.grid_points), "usable": n_usable,
                   "ran": len(run.candidates), "skipped": len(skipped_entries),
                   "ran_per_freq": ran_per_freq},
        "candidates": cand_entries,
        "skipped": skipped_entries,
    }
    write_json(run_dir / "candidate_beams.json", manifest)
    save_config_snapshot(run_dir, config, config_path)

    logging.getLogger().info(f"Saved grid run to {run_dir}")


# Retired layout's index file. New runs never write it; it survives only as the
# marker by which --replot (and run-dir ownership checks) recognize a pre-joint
# per-frequency results directory.
FREQ_INDEX_NAME = "frequencies.json"


def load_frequencies_index(base_dir: Path) -> Optional[dict]:
    """Load a LEGACY multi-frequency run's frequencies.json, or None if absent.

    The per-frequency freq_<GHz>/ layout was retired when the solver went joint;
    this reader remains so --replot can regenerate the per-frequency subdir plots
    of old results directories."""
    idx = Path(base_dir) / FREQ_INDEX_NAME
    if not idx.exists():
        return None
    with open(idx) as f:
        return json.load(f)
