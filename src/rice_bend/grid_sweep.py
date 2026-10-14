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

The flow is the sweep and then, when the config asks for it (grid_search.refine),
a second pass over the candidates the sweep left near its minimum:

  1. the sweep: every usable point solved once, from the one start the config
     names, capped at grid_search.gs_overrides.max_iters. With a refine block,
     a solve the flatness test stopped is solved again from the OTHER start,
     under the same cap, and the candidate keeps the lower result
     (_reconstruct_candidate): the band of step 2 is taken on the sweep's
     losses, and a candidate whose one start stalled far above what the
     location can reach would never enter it. "Again from the other start" is
     one more descent for a single tone. For several tones it is one descent
     from EVERY offset that start's scan cannot tell apart
     (mgs.gs_reconstruct_multistart: the best, and any within
     mgs.OFFSET_NEAR_TIE_RTOL of it, mgs.OFFSET_NEAR_TIE_MAX at most), the
     lowest of them kept;
  2. the refinement (_refine_candidates): each candidate whose sweep loss is
     within `refine.within` x the sweep's minimum is solved again — under the
     refinement's own, higher cap and tighter stop threshold, from two starts
     (mgs.gs_reconstruct_multistart), each of them, for several tones, again
     from every near-tied offset — and takes that result when it ends below its
     sweep loss.

The sweep's limits suit the thousands of candidates far from the TX; the ranking
is decided by the few hundred near it, which is where the iteration cap cuts the
descent short, where an absolute stop threshold stops it above the floor, and
where the start a candidate was solved from still shows in its loss. A candidate
solved under the refinement's limits costs 4.9-14x what it cost in the sweep
(measured per candidate); the second pass pays that only where it changes the
answer. Without a refine block the sweep's result is the run's result, and what
is written is what was written before the pass existed.

One combination is refused (run_grid_search): a sweep from the data-driven start
(gerchberg_saxton.init: backprop) WITHOUT a refine block. That start is the one
whose solves the flatness test stops early, and the block is what puts a second
start behind them.
"""

import json
import logging
import time
from collections import namedtuple
from dataclasses import dataclass, replace
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from rice_bend import rs
from rice_bend.config import (GridRefineConfig, GridSearchConfig, SimConfig,
                              SimSceneConfig, center_freq_index)
from rice_bend.data_store import (c64, f64, provenance, save_config_snapshot,
                                  write_json)
from rice_bend.mgs import (MGS, MULTISTART_STARTS, gs_reconstruct,
                           gs_reconstruct_multistart, rust_core)
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


def _candidate_problem(p: "GridPoint", shared: "SharedMeasurement"):
    """What a solve at grid point `p` is handed: (support, assumed_amp, freq_ok,
    channels).

    One definition for the sweep's solve and the refinement's re-solve. The second
    pass compares a candidate's two losses and keeps the lower, which means
    something only if both solved the same problem — the same window, the same
    uniform amplitude, the same channel subset.
    """
    x_axis = shared.x_axis
    # uniform assumed amplitude over the hypothesized window (support = window)
    support = (x_axis >= p.x_min) & (x_axis <= p.x_max)
    assumed_amp = np.where(support, 1.0, 0.0)
    freq_ok = p.freq_ok if p.freq_ok is not None else [True] * len(shared.channels)
    # the solver sees only the frequencies `p` passes the RS sampling check at
    channels = [ch for ch, ok in zip(shared.channels, freq_ok) if ok]
    return support, assumed_amp, freq_ok, channels


def _candidate_result(p: "GridPoint", shared: "SharedMeasurement", support: np.ndarray,
                      freq_ok: List[bool], result, **refine_fields) -> "CandidateResult":
    """Pack one solve's GSResult as the CandidateResult of grid point `p`. EVERY
    solve-result field comes from the one `result`, so `final_loss` is the mean of
    `per_freq_losses` over the valid frequencies for the sweep's solve and for a
    re-solve alike (scripts/characterize.py requires it). `refine_fields` are the
    fields only a run with a refine block fills (start, second_start_tried,
    refined); a run without one passes none.

    The solver's per-frequency losses are scattered back into a full-length vector
    (NaN where invalid) so every candidate's `per_freq_losses` aligns with the
    run's frequency list. The joint `final_loss` is then the mean over the valid
    subset — which is exactly the per-cell nanmean the old per-frequency layout
    computed after the fact.
    """
    per_freq_losses = np.full(len(shared.channels), np.nan)
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
        aper_axis=shared.x_axis[support].copy(),
        aper_profile=result.curr_aper_f[support].copy(),
        final_loss=result.final_loss,
        per_freq_losses=per_freq_losses,
        freq_valid=[bool(ok) for ok in freq_ok],
        n_iters_run=int(result.n_iters_run),
        stop_reason=str(result.stop_reason),
        seed=int(result.seed),
        loss_full=np.asarray(result.loss_full, dtype=np.float32),
        **refine_fields,
    )


def _other_start(init: str) -> str:
    """The start the sweep falls back on when a solve from `init` (the config's
    gerchberg_saxton.init) is stopped by the flatness test: the random one for the
    data-driven start, the data-driven one for a random or a legacy warm start.
    A name gs_reconstruct_multistart knows."""
    return "random" if init == "backprop" else "backprop"


def _reconstruct_candidate(p: "GridPoint", shared: "SharedMeasurement",
                           second_start: bool = False) -> "CandidateResult":
    """Reconstruct one candidate beam at grid point `p`, jointly over the
    frequencies where `p` passes the RS sampling check. Pure given `shared` — runs
    identically in the parent process or a worker.

    This is the SWEEP's solve: one start (shared.params.init), the sweep's cap.

    `second_start` (a run with a refine block) adds the one case where the sweep
    does not trust that solve. A solve that stops "converged" was stopped by the
    flatness test — the loss had stopped moving — and from a single start that
    says the START has stalled, not that the location is poor: next to the true
    TX of the caustic scenario (30 dB, single tone) the data-driven start stops
    after 47 to 69 iterations at 20-29x the sweep's minimum on four candidates
    which the random start takes to 1.4-1.8x of it. The refinement cannot repair
    that: it selects by the sweep's loss, and those four sit outside any band
    that would be affordable. So such a candidate is solved again, from the
    other start (_other_start), under the SAME cap and threshold, and keeps the
    lower loss — the whole result of that one solve, as in the refinement. For
    a single tone that is one more descent. For several tones the other start
    goes through gs_reconstruct_multistart, which descends from every offset
    its scan cannot tell apart (up to mgs.OFFSET_NEAR_TIE_MAX, usually one) and
    hands back the lowest; when the other start is the random one, its
    reference solve comes in front of those. A solve that ran into the cap is
    left alone: it was still descending, and solving every candidate twice is
    the 2x the second pass exists to avoid.

    What this does NOT give is a sweep loss the band is certain to cover. A
    candidate's sweep loss is only an upper bound on what its location reaches,
    and the refinement sees the candidate only if that bound falls inside
    `refine.within` x the sweep's minimum. Measured, the default band is wide:
    across 18 reference tests (the selected set against a full re-solve of every
    candidate) the band needed to capture every member of the final 1.5x set was
    at most 2.4x, against the 10x used. But it is a margin and not a guarantee:
    on the caustic scenario at ±10 % one candidate stalled at 16.0x the sweep's
    minimum, the second start at the sweep's cap did not help, and fully
    re-solved it ends at 2.17x — outside the 1.5x set, so nothing was lost
    there, but outside the band too: a candidate the second pass never saw.

    The second solve is pinned to the seed the first one recorded, so with a null
    grid seed a candidate's two solves share its one draw. With `second_start`
    the result also names the start it came from (`start`) and says whether the
    second solve was run (`second_start_tried`); without it both stay unset and
    the result is what it has always been.
    """
    support, assumed_amp, freq_ok, channels = _candidate_problem(p, shared)
    result = gs_reconstruct(
        tx_z=p.z, orig_aper_amp=assumed_amp, x_axis=shared.x_axis, rx_z=shared.rx_z,
        channels=channels, params=shared.params, ref_freq=shared.ref_freq,
        capture=False, log=None,
    )
    if not second_start:
        return _candidate_result(p, shared, support, freq_ok, result)

    start, tried = str(shared.params.init), False
    if result.stop_reason == "converged":
        tried = True
        other = _other_start(start)
        second, _ = gs_reconstruct_multistart(
            tx_z=p.z, orig_aper_amp=assumed_amp, x_axis=shared.x_axis, rx_z=shared.rx_z,
            channels=channels,
            params=shared.params.model_copy(update={"seed": int(result.seed)}),
            ref_freq=shared.ref_freq, starts=(other,), log=None,
        )
        # strictly lower replaces: a tie keeps the config's own start
        if second.final_loss < result.final_loss:
            result, start = second, other
    return _candidate_result(p, shared, support, freq_ok, result,
                             start=start, second_start_tried=tried)


def _worker_task(p: "GridPoint") -> "CandidateResult":
    """Task entry point: reconstruct one candidate against the shared measurement.

    Deliberately a two-line wrapper rather than folding the global into
    _reconstruct_candidate: the shared state is transport, not a dependency of the
    physics, and _reconstruct_candidate's "pure given `shared`" stays true."""
    return _reconstruct_candidate(p, worker_shared())


def _worker_task_second_start(p: "GridPoint") -> "CandidateResult":
    """_worker_task for a run with a refine block: the sweep's solve with the
    second start behind it (see _reconstruct_candidate). A task of its own, not a
    flag in the shared payload: which sweep runs is the parent's choice of task,
    and the run without a refine block keeps the task it has always had."""
    return _reconstruct_candidate(p, worker_shared(), second_start=True)


def _refine_candidate(p: "GridPoint", seed: int,
                      shared: "SharedMeasurement") -> "CandidateResult":
    """Re-solve grid point `p` for the refinement pass: the same problem the sweep
    solved there (_candidate_problem), from both of gs_reconstruct_multistart's
    starts, under `shared.params` — the sweep's params with the refinement's
    iteration cap and stop threshold. Returns the lower-loss start's result,
    carrying that start's name. Pure given its arguments, like the sweep's solve.

    `seed` is the one recorded on the candidate's SWEEP result, written over
    whatever the params hold. With grid_search.seed set that is the same number
    for every candidate; with a null seed each candidate drew its own, and its
    re-solve starts again from that draw and not from a second one — so the
    "random" start here and the sweep's random start are the same phase, and the
    candidate's two losses differ by the cap, the threshold and the start, never
    by the luck of a new draw.

    Whether this result REPLACES the sweep's is the parent's decision
    (_refine_candidates): it holds the sweep loss, and a worker does not. The
    result is therefore marked `refined` and named for its start, and nothing
    else: `sweep_loss` and `second_start_tried` are the sweep's to give.
    """
    support, assumed_amp, freq_ok, channels = _candidate_problem(p, shared)
    result, start = gs_reconstruct_multistart(
        tx_z=p.z, orig_aper_amp=assumed_amp, x_axis=shared.x_axis, rx_z=shared.rx_z,
        channels=channels, params=shared.params.model_copy(update={"seed": int(seed)}),
        ref_freq=shared.ref_freq, log=None,
    )
    return _candidate_result(p, shared, support, freq_ok, result,
                             refined=True, start=start)


def _refine_task(item: Tuple["GridPoint", int]) -> "CandidateResult":
    """Task entry point of the refinement pass: `item` is (grid point, the seed of
    its sweep result). The same two-line shape as _worker_task, for its reason."""
    p, seed = item
    return _refine_candidate(p, seed, worker_shared())


@dataclass
class CandidateResult:
    """A single reconstructed candidate beam at a hypothesized TX location.

    One aperture per candidate: the solved profile is one delay plate shared
    across all frequencies (stored as its reference-frequency field) and the
    assumed amplitude is the same uniform box at every frequency, so there is
    exactly one complex profile whatever F is.
    `final_loss` is the joint (mean-over-valid-frequencies) residual;
    `per_freq_losses` aligns with the run's frequency list, NaN where the
    candidate failed that frequency's RS sampling check (`freq_valid`).

    Every solve-result field (the profile, the losses, `n_iters_run`,
    `stop_reason`, `seed`, `loss_full`) describes ONE solve, and in a run with a
    refine block `start` names the start that solve began from. A candidate can
    have been solved from up to four starts — the sweep's own, the sweep's second
    start if the first was stopped by the flatness test, and the refinement's two
    — and, for several tones, each start after the sweep's own from every offset
    its scan could not tell apart (mgs.OFFSET_NEAR_TIE_MAX at most), so the
    number of descents can be higher than four. What is stored is whichever
    descent ended lowest, whole, never a mix:

    - after the sweep, `start` is the config's init ('backprop' | 'random' |
      'warm_start'), or the other start's name where the second solve won;
    - a candidate the refinement pass re-solved is `refined`, and `sweep_loss`
      keeps what the sweep had found. If the re-solve ended strictly lower the
      result is the winning start's and `start` is its name; if not, the sweep's
      result stands and `start` stays the sweep's. So `final_loss <= sweep_loss`
      for every refined candidate, and the two are equal exactly when the
      sweep's result was kept — which is how the two cases are told apart, since
      a start's name can be the same in both.

    In a run without a refine block the last four fields keep their defaults."""
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
    # a run with a refine block only (None / False without one)
    start: Optional[str] = None        # the stored solve's start: 'backprop' |
    #                                    'random' | 'warm_start'
    # the sweep solved this candidate a second time, from the other start (its
    # first solve was stopped by the flatness test). In memory only: the run
    # reports the counts (refine["sweep_second_start"]), not the flag
    second_start_tried: bool = False
    # the refinement pass (both at their defaults for a candidate it did not
    # re-solve: a skipped pass, or a loss outside `within`)
    refined: bool = False              # re-solved by the refinement pass
    sweep_loss: Optional[float] = None  # the sweep's final_loss for this candidate


@dataclass
class GridSearchRun:
    """Everything needed to persist + later rank a grid search.

    Per-frequency arrays are stacked (F, ...) where the lengths agree (they are
    all on the scene x-axis or the shared TX axis) and kept as ragged lists only
    for the RX element arrays, whose counts genuinely differ per frequency
    (dx defaults to wavelength/20). `freqs` order is the alignment order.

    `candidates` are the run's FINAL results: the sweep's, with the refinement
    pass's replacements already made where it ran (see CandidateResult).
    `effective_max_iters` and `init` describe the sweep whatever the refinement
    did; the refinement's own settings and outcome are in `refine`."""
    freqs: List[float]
    wavelengths: List[float]
    seed: Optional[int]
    effective_max_iters: int         # the SWEEP's cap (the re-solve's: refine["max_iters"])
    # the SWEEP's start: 'random' | 'warm_start' | 'backprop' (from gerchberg_saxton).
    # The start each candidate's stored solve began from is CandidateResult.start
    init: str
    ref_freq: float                  # run-level reference frequency for psi units
    grid_cfg: GridSearchConfig
    grid_points: List[GridPoint]
    candidates: List[CandidateResult]
    # shared measurement set (one per run; one row per frequency)
    scene_x_axis: np.ndarray
    rx_fields: np.ndarray            # (F, nx) on the scene x-axis
    error_weightings: np.ndarray     # (F, nx)
    rx_plane_rows: Optional[np.ndarray]  # (F, nx) c64 PRE-interp RX-plane fields
    #   (None when not synthesized) — the energy metric's numerator+denominator
    rx_x_min: float                  # the RX window (frequency-independent)
    rx_x_max: float
    rx_z: float                      # the RX/measurement plane
    rx_aper_axes: List[np.ndarray]   # ragged: per-frequency element axes
    rx_aper_profiles: List[np.ndarray]
    # ground truth, for later evaluation of how well candidates localize the TX
    real_tx_aper_axis: np.ndarray    # geometry-only, shared across frequencies
    real_tx_aper_profiles: np.ndarray  # (F, nt): beam phase ∝ k
    real_tx_z: float
    real_tx_x_min: float
    real_tx_x_max: float
    scene_bounds: Tuple[float, float, float, float]  # x_min, x_max, z_min, z_max
    # receiver noise (both None when the measurement is noiseless). With noise on,
    # rx_fields / error_weightings / rx_aper_profiles are the NOISY measurement the
    # solver saw and these are the noiseless element profiles beside it.
    rx_aper_profiles_clean: Optional[List[np.ndarray]] = None
    noise: Optional[dict] = None     # MGS.noise_report: the JSON-safe noise block
    # the refinement pass's JSON-safe block (settings, status, outcome: see
    # _refine_candidates); None when grid_search.refine is not configured — which
    # is also what keeps such a run's files as they were before the pass existed
    refine: Optional[dict] = None


def _second_start_counts(candidates: List["CandidateResult"], init: str) -> dict:
    """The refine block's `sweep_second_start`: over ALL of the sweep's candidates,
    how many first solves were stopped by the flatness test and solved again from
    the other start (`n_tried`), and how many of those second solves ended lower
    and became the candidate's sweep result (`n_won`: its start is then not the
    config's `init`). Taken from the sweep's results, before the refinement
    replaces any of them."""
    tried = [c for c in candidates if c.second_start_tried]
    return {"n_tried": len(tried),
            "n_won": sum(1 for c in tried if c.start != init)}


def _refine_block(refine_cfg: GridRefineConfig, threshold: float, status: str,
                  second_start: dict, sweep_min: Optional[float], n_selected: int,
                  resolved: Optional[List["CandidateResult"]] = None) -> dict:
    """The run's `refine` block (manifest gs.refine): the settings the pass ran
    under, what the sweep's second start did (`second_start`, see
    _second_start_counts), whether the pass ran, and what it changed. `resolved`
    are the selected candidates AFTER replacement (None when nothing was
    re-solved).

    `wins` says where each selected candidate's stored result came from: the
    re-solve's 'backprop' or 'random' start, or 'kept' — the re-solve did not get
    below the sweep's result, which stands (whatever start THAT came from: see
    CandidateResult.start).

    Nothing in it depends on wall-clock time or on the worker count: a serial run
    and a parallel one must write the same manifest.
    """
    wins = {name: 0 for name in tuple(MULTISTART_STARTS) + ("kept",)}
    gain_max, gain_max_index = None, None
    if resolved is not None:
        # 1.0 when no candidate improved: the largest sweep_loss / final_loss is
        # then the ratio of a loss to itself
        gain_max = 1.0
        for cand in resolved:
            # the replacement is strict, so equal losses ARE the sweep's result
            kept = cand.final_loss == cand.sweep_loss
            wins["kept" if kept else cand.start] += 1
            # A kept sweep result has ratio 1 and an improved candidate > 1, so the
            # strict test never names a candidate that did not improve, and keeps
            # the first of equal gains (candidate order). A loss of exactly 0 has
            # no ratio to report.
            if cand.final_loss > 0.0:
                gain = cand.sweep_loss / cand.final_loss
                if gain > gain_max:
                    gain_max, gain_max_index = float(gain), int(cand.point.index)
    return {
        "within": float(refine_cfg.within),
        "max_iters": int(refine_cfg.max_iters),
        # the EFFECTIVE one: the block's own, or gerchberg_saxton's when it is null
        "convergence_threshold": float(threshold),
        "max_candidates": int(refine_cfg.max_candidates),
        "status": status,                    # 'done' | 'skipped'
        "sweep_second_start": second_start,  # {n_tried, n_won}, whatever the status
        "sweep_min": sweep_min,              # smallest finite sweep loss (null: none)
        "n_selected": int(n_selected),       # within `within` x sweep_min
        "n_improved": sum(n for name, n in wins.items() if name != "kept"),
        "wins": wins,                        # per selected candidate
        "gain_max": gain_max,                # null when skipped
        "gain_max_index": gain_max_index,    # that candidate's grid index
    }


def _refine_candidates(candidates: List["CandidateResult"], shared: "SharedMeasurement",
                       refine_cfg: GridRefineConfig, *, jobs: int,
                       log: logging.Logger) -> Tuple[List["CandidateResult"], dict]:
    """The refinement pass: re-solve the candidates the sweep left near its
    minimum and keep, per candidate, the lower of its two losses. Returns
    (the candidates with the replacements made, the run's refine block);
    `candidates` itself is not modified.

    SELECTION is by loss alone: every candidate with a finite sweep loss
    <= within x the smallest finite sweep loss, in candidate order. The stop
    reason plays no part — a candidate that "converged" in the sweep did so under
    the sweep's threshold and cap, and the tighter threshold or the longer
    descent may still lower it. The sweep losses are the ones the sweep's second
    start has already been through (_reconstruct_candidate), and a skipped pass
    leaves those results as they are.

    More than `max_candidates` selected: NOTHING is re-solved (status 'skipped').
    Re-solving the best N would lower the ones kept and leave the rest at their
    sweep loss, forcing the answer into the N (see config.GridRefineConfig).
    No candidate with a finite sweep loss (or no candidate at all) is 'skipped'
    too, with a null sweep_min: there is no minimum to select against.

    REPLACEMENT is all-or-nothing per candidate: a re-solve that ends strictly
    below the sweep loss replaces the candidate's whole solve result (the profile
    with the losses that describe it); otherwise the sweep's result stands
    untouched. So a refined candidate never ends above its sweep loss, and
    final_loss stays the mean of per_freq_losses. Both outcomes are marked
    `refined` and carry `sweep_loss`, the loss the sweep had found; `start` is
    the winning start of the re-solve in the first case and stays the sweep's in
    the second.

    The re-solve params are the sweep's (`shared.params`) with the refinement's
    cap and threshold — a copy: config.gerchberg_saxton keeps the sweep's cap,
    which is what the config snapshot records and what mgs-study's resume check
    compares. Each candidate is re-solved from the seed of its own sweep result
    (_refine_candidate). Dispatch is map_workers', as for the sweep, so the
    result does not depend on `jobs`.
    """
    n_total = len(candidates)
    threshold = (refine_cfg.convergence_threshold
                 if refine_cfg.convergence_threshold is not None
                 else shared.params.convergence_threshold)
    second_start = _second_start_counts(candidates, str(shared.params.init))
    finite = [k for k, c in enumerate(candidates) if np.isfinite(c.final_loss)]
    if not finite:
        # nothing to rank, so no minimum to select against
        log.info(f"Refinement skipped: none of the {n_total} candidate(s) has a "
                 "finite sweep loss")
        return candidates, _refine_block(refine_cfg, threshold, "skipped",
                                         second_start, None, 0)
    sweep_min = float(min(candidates[k].final_loss for k in finite))
    selected = [k for k in finite
                if candidates[k].final_loss <= refine_cfg.within * sweep_min]
    if len(selected) > refine_cfg.max_candidates:
        log.info(f"Refinement skipped: {len(selected)} of {n_total} candidate(s) lie "
                 f"within {refine_cfg.within:g}x the sweep minimum {sweep_min:.6g} and "
                 f"exceed max_candidates ({refine_cfg.max_candidates}); the sweep's "
                 "result stands for every candidate")
        return candidates, _refine_block(refine_cfg, threshold, "skipped",
                                         second_start, sweep_min, len(selected))
    log.info(f"Refinement: re-solving {len(selected)} of {n_total} candidate(s) within "
             f"{refine_cfg.within:g}x the sweep minimum {sweep_min:.6g} "
             f"(starts {' + '.join(MULTISTART_STARTS)}, up to {refine_cfg.max_iters} "
             f"iters, threshold {threshold:g})")

    refine_shared = shared._replace(params=shared.params.model_copy(update={
        "max_iters": refine_cfg.max_iters, "convergence_threshold": threshold}))
    swept = {candidates[k].point.index: candidates[k] for k in selected}

    def _log_refined(done: int, total: int, cand: CandidateResult) -> None:
        p = cand.point
        before = swept[p.index]
        solve = f"{cand.n_iters_run} iters, {cand.stop_reason}"
        # the same test as the replacement below
        outcome = (f"{cand.final_loss:.6g} ({cand.start}, {solve})"
                   if cand.final_loss < before.final_loss else
                   f"{before.final_loss:.6g} (kept, {before.start}: {cand.start} "
                   f"reached {cand.final_loss:.6g}, {solve})")
        log.info(f"[refine {done}/{total}] #{p.index} z={p.z:.3f} x={p.x_center:+.3f} "
                 f"loss {before.final_loss:.6g} -> {outcome}")

    t0 = time.perf_counter()
    resolved: List[CandidateResult] = map_workers(
        _refine_task, [(candidates[k].point, candidates[k].seed) for k in selected],
        jobs=jobs, shared=refine_shared, on_done=_log_refined, log=log,
        desc=f"Refining {len(selected)} candidate(s)")
    elapsed = time.perf_counter() - t0

    out = list(candidates)
    for k, new in zip(selected, resolved):
        old = candidates[k]
        # strictly below: a re-solve that only ties changes nothing. The winner
        # brings its own start; a kept sweep result keeps the one it had
        out[k] = (replace(new, sweep_loss=old.final_loss,
                          second_start_tried=old.second_start_tried)
                  if new.final_loss < old.final_loss else
                  replace(old, refined=True, sweep_loss=old.final_loss))
    block = _refine_block(refine_cfg, threshold, "done", second_start, sweep_min,
                          len(selected), [out[k] for k in selected])
    wins = ", ".join(f"{name} {n}" for name, n in block["wins"].items())
    best = ("" if block["gain_max_index"] is None
            else f" at #{block['gain_max_index']}")
    log.info(f"Refinement: {block['n_improved']} of {len(selected)} candidate(s) "
             f"improved (stored result from: {wins}); largest gain "
             f"{block['gain_max']:.4g}x{best} [{elapsed:.1f} s]")
    return out, block


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

    Raises ValueError, before any solve, for gerchberg_saxton.init == "backprop"
    without a grid_search.refine block (the comment at the check says why).

    With a grid_search.refine block two things are added. In the sweep, a solve
    stopped by the flatness test is solved again from the other start (for
    several tones, from every near-tied offset of that start) and the candidate
    keeps the lowest result (_reconstruct_candidate). After it comes
    the refinement pass (_refine_candidates): the candidates near the sweep's
    minimum are solved again, under the block's cap and stop threshold and from
    two starts, and the returned run's candidates carry the lowest of their
    results. Both run inside per-candidate tasks and are independent of `jobs` in
    the same way. Without the block the sweep's results are returned as they
    are, from the one solve each.
    """
    grid_cfg = config.grid_search
    if grid_cfg is None:
        raise ValueError("config.grid_search is required for grid-search-mgs")
    # Refused before anything is measured or solved. The data-driven start is the
    # one whose descent the flatness test stops early at locations it does not
    # suit, and only a run with a refine block puts a second start behind those
    # solves (_reconstruct_candidate) and re-solves the near-best candidates. On
    # its own it leaves them where they stalled: on the caustic single-tone patch
    # round the true TX, 11 candidates next to it end at 12-30x the minimum.
    # Single-shot solves (`mgs`, --true-mgs) do not come through here and are not
    # affected; neither is a sweep from the legacy starts.
    if config.gerchberg_saxton.init == "backprop" and grid_cfg.refine is None:
        raise ValueError(
            "gerchberg_saxton.init is 'backprop' but grid_search.refine is not set: "
            "a sweep from the data-driven start needs the refinement block, which "
            "gives the solves the flatness test stops early a second start and "
            "re-solves the near-best candidates. Add `refine: {}` under grid_search "
            "(its defaults), or sweep with init: warm_start or random")
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
    if gs_cfg.engine == "rust":
        rust_core()   # fail here, not in every worker after the setup
    log.info(f"Solver engine: {gs_cfg.engine}")

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
        # a run with a refine block: the start the stored solve came from, and
        # whether it took the second start to get it
        start = ""
        if cand.start is not None:
            start = f" [{cand.start}" + ("; second start tried]"
                                         if cand.second_start_tried else "]")
        log.info(f"[{done}/{total}] #{p.index} z={p.z:.3f} x={p.x_center:+.3f} "
                 f"-> loss {cand.final_loss:.6g} ({cand.n_iters_run} iters, "
                 f"{cand.stop_reason}{subset}){start}")

    # map_workers returns input order, so the manifest is stable without a re-sort.
    # The task decides which sweep this is: with a refine block, the one that puts
    # a second start behind every solve the flatness test stopped.
    refining = grid_cfg.refine is not None
    t0 = time.perf_counter()
    candidates: List[CandidateResult] = map_workers(
        _worker_task_second_start if refining else _worker_task, usable, jobs=jobs,
        shared=shared, on_done=_log_done, log=log,
        desc=f"Reconstructing {len(usable)} candidate(s)")
    # elapsed times are logged, never persisted: the manifest of a run must not
    # depend on the machine or on `jobs`
    log.info(f"Sweep: {len(candidates)} candidate(s) solved in "
             f"{time.perf_counter() - t0:.1f} s")
    if refining:
        init = str(shared.params.init)
        counts = _second_start_counts(candidates, init)
        log.info(f"Sweep: {counts['n_tried']} of {len(candidates)} solve(s) were "
                 f"stopped by the flatness test and solved again from the "
                 f"{_other_start(init)} start; {counts['n_won']} of those second "
                 "solves ended lower and were kept")

    # 5. the refinement pass, when configured: re-solve the near-best candidates
    #    under its own cap and threshold and from two starts. It works from
    #    `shared` (the sweep's params, copied) and returns new candidates, so
    #    config.gerchberg_saxton still holds the sweep's cap — and
    #    effective_max_iters below still reports it.
    refine = None
    if refining:
        candidates, refine = _refine_candidates(candidates, shared, grid_cfg.refine,
                                                jobs=jobs, log=log)

    real_tx = config.tx_aperture
    return GridSearchRun(
        freqs=[float(f) for f in mgs.freqs],
        wavelengths=[float(w) for w in wavelengths],
        seed=mgs.gs_cfg.seed,
        effective_max_iters=int(mgs.gs_cfg.max_iters),
        init=str(mgs.gs_cfg.init),
        ref_freq=float(ref_freq),
        grid_cfg=grid_cfg,
        grid_points=points,
        candidates=candidates,
        scene_x_axis=np.asarray(x_axis).copy(),
        rx_fields=np.stack([np.asarray(ch.rx_field) for ch in channels]),
        error_weightings=np.stack([np.asarray(ch.error_weighting) for ch in channels]),
        rx_plane_rows=(np.stack([fs.rx_plane_row for fs in mgs.freq_states])
                       if all(fs.rx_plane_row is not None for fs in mgs.freq_states)
                       else None),
        rx_x_min=float(mgs.scene.rx_ap.x_min),
        rx_x_max=float(mgs.scene.rx_ap.x_max),
        rx_z=float(mgs.scene.rx_ap.z),
        rx_aper_axes=[fs.rx_ap.aper_axis.copy() for fs in mgs.freq_states],
        rx_aper_profiles=[fs.rx_ap.aper_profile.copy() for fs in mgs.freq_states],
        real_tx_aper_axis=mgs.scene.tx_ap.aper_axis.copy(),
        real_tx_aper_profiles=np.stack([fs.tx_ap.aper_profile.copy()
                                        for fs in mgs.freq_states]),
        real_tx_z=float(real_tx.z),
        real_tx_x_min=float(real_tx.x_min),
        real_tx_x_max=float(real_tx.x_max),
        scene_bounds=(scene_cfg.x_min, scene_cfg.x_max, scene_cfg.z_min, scene_cfg.z_max),
        rx_aper_profiles_clean=(None if mgs.noise_report is None
                                else [fs.rx_aper_profile_clean.copy()
                                      for fs in mgs.freq_states]),
        noise=mgs.noise_report,
        refine=refine,
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

    With receiver noise on, the manifest's `noise` block records it (null when off)
    and measurement.npz additionally carries rx_aper_profile_clean_NN; a noiseless
    run's key set is unchanged.

    The manifest is schema 3 for a run with no refinement configured
    (`run.refine is None`), and that run's files are exactly what they were before
    the refinement pass existed: no key is added, anywhere. A run WITH a refine
    block is schema 4, which adds — and changes nothing else:
        gs.refine                         the refine block (settings, status
                                          'done' | 'skipped', the sweep's second-
                                          start counts, what the pass changed)
        candidates[].refined/start/sweep_loss, and the same three under gs_result
                                          in each cand_####.json. `start`, for
                                          EVERY candidate: the start its stored
                                          solve began from ('backprop' | 'random'
                                          | 'warm_start'). `refined` and
                                          `sweep_loss`: whether the pass re-solved
                                          the candidate and the sweep's loss;
                                          false / null where it did not
    The solve-result fields beside them (final_loss, per_freq_losses, n_iters_run,
    stop_reason, seed, and the npz's profile and loss curve) are the STORED
    solve's — the lowest of the sweep's (one or two starts) and the re-solve's —
    so a schema-3 reader ranks a schema-4 run correctly without knowing about
    either; gs.effective_max_iters stays the sweep's cap. A refined candidate
    whose final_loss equals its sweep_loss kept the sweep's result. The npz key
    sets are the same under both schemas.
    """
    t0 = time.perf_counter()
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
    # With noise on, rx_field / error_weighting / rx_aper_profile_NN above are the
    # NOISY measurement the solver saw; the clean element profiles go beside them.
    # rx_plane_row below stays noiseless: it feeds the energy metric, which scores
    # the beam's physics, not the measurement.
    if run.rx_aper_profiles_clean is not None:
        for i, prof in enumerate(run.rx_aper_profiles_clean):
            meas[f"rx_aper_profile_clean_{i:02d}"] = c64(prof)
    if run.rx_plane_rows is not None:
        # pre-interp RX-plane fields: lets --replot recompute the energy metric
        # without re-propagating (rs already produced complex64 — lossless)
        meas["rx_plane_row"] = c64(run.rx_plane_rows)
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
        # schema 4 only. An EMPTY dict for a run with no refine block, not three
        # nulls: its json files must not gain a key. (second_start_tried is not
        # written per candidate: gs.refine carries the counts.)
        refine_fields = ({} if run.refine is None else
                         {"refined": cand.refined, "start": cand.start,
                          "sweep_loss": cand.sweep_loss})
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
                          "stop_reason": cand.stop_reason, "seed": cand.seed,
                          **refine_fields},
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
            **refine_fields,
            "npz": f"candidates/{name}.npz", "json": f"candidates/{name}.json",
        })

    skipped_entries = [{"index": p.index, "z": p.z, "x_center": p.x_center,
                        "skip_reason": p.skip_reason}
                       for p in run.grid_points if not p.ok]
    n_usable = sum(1 for p in run.grid_points if p.ok)
    ran_per_freq = [sum(1 for c in run.candidates if c.freq_valid[i])
                    for i in range(len(run.freqs))]

    manifest = {
        # 4 = 3 + the refinement's keys (see the docstring); a run without a refine
        # block stays 3 because its manifest IS a schema-3 manifest, key for key
        "schema_version": 3 if run.refine is None else 4,
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
        "rx_aperture": {"x_min": run.rx_x_min, "x_max": run.rx_x_max, "z": run.rx_z},
        "gs": {"effective_max_iters": run.effective_max_iters, "loss_combine": "mean",
               "init": run.init, "ref_freq_hz": run.ref_freq,
               **({} if run.refine is None else {"refine": run.refine})},
        "provenance": provenance(args_dict),
        "noise": run.noise,          # receiver-noise block; null when noiseless
        "counts": {"total": len(run.grid_points), "usable": n_usable,
                   "ran": len(run.candidates), "skipped": len(skipped_entries),
                   "ran_per_freq": ran_per_freq},
        "candidates": cand_entries,
        "skipped": skipped_entries,
    }
    write_json(run_dir / "candidate_beams.json", manifest)
    save_config_snapshot(run_dir, config, config_path)

    # logged, not persisted (a 49,377-candidate run writes ~10^5 files here)
    logging.getLogger().info(f"Saved grid run to {run_dir} in "
                             f"{time.perf_counter() - t0:.1f} s")


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
