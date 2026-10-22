"""The speculative TX-location sweep: enumerate the grid, reconstruct a candidate
beam at every usable point, persist the result.

Deliberately free of matplotlib. Importing anything from the old grid_search
dragged in mpl_toolkits.mplot3d and FuncAnimation, so a worker process that only
ever solves MGS paid for the entire plotting stack.

The real TX location/trajectory from the config is used only to synthesize the
shared RX measurement set (one per frequency); the search itself treats the TX
location as unknown. A `measured:` config replaces the synthesis with a measured
THz-TDS row (MGS.from_measured): the TX's position is then the bench's, and its
beam, the plate's profile, is not known at all. At each usable grid point a
fixed-width aperture (uniform assumed amplitude) is placed and MGS reconstructs ONE
phase for it jointly against every frequency's measurement the point is valid for.

Every usable point is solved ONCE, from the start the config names
(gerchberg_saxton.init), under gerchberg_saxton's settings with
grid_search.gs_overrides applied: every candidate gets the same treatment, so the
ranking compares locations under one solver setting.
"""

import json
import logging
import time
from collections import namedtuple
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from rice_bend import rs
from rice_bend.config import (GerchbergSaxtonConfig, GridSearchConfig, SimConfig,
                              SimSceneConfig, center_freq_index)
from rice_bend.data_store import (c64, f64, provenance, save_config_snapshot,
                                  write_json)
from rice_bend.mgs import (MGS, FreqChannel, check_amplitude_mode, gs_reconstruct,
                           rust_core)
from rice_bend.parallel import map_workers, worker_shared
from rice_bend.sim_scene import scene_axis

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


def window_samples(x_axis: np.ndarray, x_center: float, width: float) -> Tuple[int, int]:
    """The candidate window as scene sample indices (i_lo, i_hi), both inclusive:
    the sample nearest `x_center` plus round(width / (2 dx)) samples on each side,
    so every window is the same 2*half + 1 samples and is centred on a grid sample.
    The indices can fall outside the axis (a window that leaves the scene); the
    caller decides what that means.

    Defined by index rather than by comparing the axis against x_center +- width/2
    because that comparison is decided by float noise: a linspace x_center of
    -0.011999999999999983 puts the left edge a few ulp above the sample at -0.062
    and drops it, while the next x_center keeps both edges. On the measured 2 mm
    grid half the candidates got a 400-sample plate 0.25 mm narrower and 0.125 mm
    off-centre, the other half the 401-sample one, and ill-conditioned off-bearing
    solves moved by up to 63% between two sweeps that should have been identical
    (2026-10-22)."""
    dx = float(x_axis[1] - x_axis[0])
    i_c = int(round((float(x_center) - float(x_axis[0])) / dx))
    half = int(round(float(width) / (2.0 * dx)))
    return i_c - half, i_c + half


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

    The window is a run of scene samples chosen by index (window_samples): the
    point's x_min / x_max are the exact axis values of its first and last sample,
    so a float comparison of the axis against them downstream recovers exactly
    those samples; x_center stays the grid's nominal coordinate.
    """
    dx = grid_cfg.aperture.dx
    # the scene's own axis (SimScene builds it with the same call)
    x_axis = scene_axis(scene_cfg.x_min, scene_cfg.x_max, scene_cfg.spacing,
                        scene_cfg.exact_step)
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
            i_lo, i_hi = window_samples(x_axis, x_center, grid_cfg.aperture.width)
            inside = 0 <= i_lo and i_hi < len(x_axis)
            # the exact sample values at the window's ends (the nominal edges only
            # in the skip message, where there is no sample to name)
            x_min = float(x_axis[i_lo]) if inside else x_center - grid_cfg.aperture.width / 2.0
            x_max = float(x_axis[i_hi]) if inside else x_center + grid_cfg.aperture.width / 2.0
            reason: Optional[str] = None
            freq_ok: Optional[List[bool]] = None
            if not inside:
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
    identically in the parent process or a worker. One solve, from the start
    shared.params names.

    The solver sees only the valid subset of channels; its per-frequency losses
    are scattered back into a full-length vector (NaN where invalid) so every
    candidate's `per_freq_losses` aligns with the run's frequency list. The joint
    `final_loss` is then the mean over the valid subset — which is exactly the
    per-cell nanmean the old per-frequency layout computed after the fact.
    """
    x_axis = shared.x_axis
    # uniform assumed amplitude over the hypothesized window (support = window).
    # With amplitude descend or level it is the support alone: the solve fits the
    # amplitude. p.x_min / p.x_max are the exact axis values of the window's end
    # samples (window_samples), and the half-sample slack makes the comparison
    # immune to the last-bit noise that used to drop an edge sample (see there)
    slack = 0.5 * float(x_axis[1] - x_axis[0])
    support = (x_axis >= p.x_min - slack) & (x_axis <= p.x_max + slack)
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
        final_mse=float(result.final_mse),
        power_ratio=None if result.power_ratio is None else float(result.power_ratio),
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

    One aperture per candidate: the solved profile is one delay plate shared
    across all frequencies (stored as its reference-frequency field) and the
    assumed amplitude is the same uniform box at every frequency, so there is
    exactly one complex profile whatever F is. With amplitude: descend the
    amplitude is the solve's own, still one array for every frequency, and
    `aper_profile` carries it with the phase; with amplitude: level it is the
    solve's one uniform level.
    `final_loss` is the joint (mean-over-valid-frequencies) residual;
    `per_freq_losses` aligns with the run's frequency list, NaN where the
    candidate failed that frequency's RS sampling check (`freq_valid`).
    With gerchberg_saxton.power_penalty on, `final_loss` (and `loss_full`) is
    the penalized objective J = L * power_ratio**gamma the solve minimized and
    that ranks the candidates, `final_mse` the plain joint residual L and
    `per_freq_losses` its plain components; off, `final_mse` equals
    `final_loss` and `power_ratio` is None (mgs.GSResult)."""
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
    # the plain joint residual L and R = max(1, P_plate/P_rx) of the final iterate
    # (gerchberg_saxton.power_penalty); NaN / None when not set by a solve
    final_mse: float = float("nan")
    power_ratio: Optional[float] = None


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
    effective_max_iters: int         # every solve's cap (gs_overrides applied)
    init: str                        # gerchberg_saxton.init: every solve's start
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
    # ground truth, for later evaluation of how well candidates localize the TX.
    # The plate itself (axis + profiles) is None on a measured run, where the beam
    # is the unknown: its truth is the position below alone.
    real_tx_aper_axis: Optional[np.ndarray]    # geometry-only, shared across frequencies
    real_tx_aper_profiles: Optional[np.ndarray]  # (F, nt): beam phase ∝ k
    real_tx_z: float
    real_tx_x_min: float
    real_tx_x_max: float
    scene_bounds: Tuple[float, float, float, float]  # x_min, x_max, z_min, z_max
    # receiver noise (both None when the measurement is noiseless). With noise on,
    # rx_fields / error_weightings / rx_aper_profiles are the NOISY measurement the
    # solver saw and these are the noiseless element profiles beside it.
    rx_aper_profiles_clean: Optional[List[np.ndarray]] = None
    noise: Optional[dict] = None     # MGS.noise_report: the JSON-safe noise block
    # MGS.measured_report on a measured run (the manifest's `measured` block), None
    # on a simulated one
    measured: Optional[dict] = None
    # gerchberg_saxton.amplitude / amp_max of every solve ('fixed' / None: the
    # uniform box held). Recorded in the manifest's gs block only when descended
    amplitude: str = "fixed"
    amp_max: Optional[float] = None
    # gerchberg_saxton.power_penalty (gamma) of every solve; None when off. Recorded
    # in the manifest's gs block only when set
    power_penalty: Optional[float] = None
    # MGS.tx_amplitude_report: tx_aperture.amplitude as applied to the simulated
    # plate ({setting, scale, clean_rx_peak_unscaled}); None at the default 1.0
    # and on a measured run. The manifest's top-level tx_amplitude, only when set
    tx_amplitude: Optional[dict] = None


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
    uses the same fixed seed) regardless of `jobs` or completion order. Each
    candidate's result is its one solve.

    With config.measured set, the measurement is the measured RX row instead
    (MGS.from_measured, which applies the measured geometry to `config`), and
    everything after it is the simulated on-grid path unchanged. The ground
    truth is then the measured TX POSITION alone: the run's real_tx_aper_axis /
    real_tx_aper_profiles are None, and `measured` holds the provenance.
    """
    grid_cfg = config.grid_search
    if grid_cfg is None:
        raise ValueError("config.grid_search is required for grid-search-mgs")
    # an amplitude mode the engine cannot run (or level on several tones under
    # init: warm_start) is refused before the measurement is built, not by the
    # first candidate's solve after it (no capture: a candidate solve never captures)
    check_amplitude_mode(config.gerchberg_saxton, n_tones=len(freqs))
    log = log or logging.getLogger()

    # 1. the shared RX measurement set (one per frequency). measure() synthesizes
    #    each by propagating to the RX plane alone -- the sweep never reads
    #    mgs.scene.data, so illuminating all 3400 planes here cost 1.71 s and
    #    +190 MB RSS, held in the parent for the whole sweep, to keep one row.
    #    A measured run reads it from the data file instead, and measure() only puts
    #    it on the grid.
    measured = config.measured is not None
    mgs = MGS.from_measured(freqs, config) if measured else MGS(freqs, config)
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
        log.info(f"[{done}/{total}] #{p.index} z={p.z:.3f} x={p.x_center:+.3f} "
                 f"-> loss {cand.final_loss:.6g} ({cand.n_iters_run} iters, "
                 f"{cand.stop_reason}{subset})")

    # map_workers returns input order, so the manifest is stable without a re-sort.
    t0 = time.perf_counter()
    candidates: List[CandidateResult] = map_workers(
        _worker_task, usable, jobs=jobs, shared=shared, on_done=_log_done, log=log,
        desc=f"Reconstructing {len(usable)} candidate(s)")
    # elapsed times are logged, never persisted: the manifest of a run must not
    # depend on the machine or on `jobs`
    log.info(f"Sweep: {len(candidates)} candidate(s) solved in "
             f"{time.perf_counter() - t0:.1f} s")

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
        real_tx_aper_axis=None if measured else mgs.scene.tx_ap.aper_axis.copy(),
        real_tx_aper_profiles=(None if measured else
                               np.stack([fs.tx_ap.aper_profile.copy()
                                         for fs in mgs.freq_states])),
        # config.tx_aperture: on a measured run, the window apply_measured derived
        real_tx_z=float(real_tx.z),
        real_tx_x_min=float(real_tx.x_min),
        real_tx_x_max=float(real_tx.x_max),
        scene_bounds=(scene_cfg.x_min, scene_cfg.x_max, scene_cfg.z_min, scene_cfg.z_max),
        rx_aper_profiles_clean=(None if mgs.noise_report is None
                                else [fs.rx_aper_profile_clean.copy()
                                      for fs in mgs.freq_states]),
        noise=mgs.noise_report,
        measured=mgs.measured_report if measured else None,
        amplitude=str(mgs.gs_cfg.amplitude),
        amp_max=mgs.gs_cfg.amp_max,
        power_penalty=mgs.gs_cfg.power_penalty,
        tx_amplitude=mgs.tx_amplitude_report,
    )


def solve_at_measured_tx(run: GridSearchRun, params: GerchbergSaxtonConfig
                         ) -> Tuple[Optional[CandidateResult], dict]:
    """--true-mgs as a candidate: ONE solve at the TX position the run was made
    from, made exactly the way every candidate is -- that TX window (run.real_tx_z,
    real_tx_x_min..real_tx_x_max: the measured window on a measured run, the
    configured tx_aperture on a simulated one) with the uniform assumed amplitude
    (or, with amplitude descend or level, the fitted one the candidates get),
    against the run's own measurement, under the sweep's solver settings
    (`params`: the effective gerchberg_saxton, i.e. gs_overrides and the grid seed
    applied, and its engine) and the run's reference frequency, through
    _reconstruct_candidate itself. Its loss therefore compares directly with the
    candidates': does the TX position fit the data as well as the best candidate
    does, and if not, by how much.

    Two runs make it. A measured run, whose plate is unknown: the plain `mgs`
    baseline (candidate_scenes.make_true_mgs_plot, the solve under the real
    plate's own amplitude with its scene and LANDING block) needs the plate. And a
    simulated run with amplitude descend or level: that baseline is a solve that
    captures GS history, which runs only at a fixed amplitude
    (mgs.check_amplitude_mode), so the solve at the true TX is this one, as the
    candidates fit the amplitude. A simulated fixed-amplitude run keeps
    make_true_mgs_plot. The name says "measured" for the run it was written for;
    the report's `kind` says which TX it was.

    Returns (result, report): `result` is the CandidateResult (None when the TX
    plane fails the RS sampling check at every frequency, as enumerate_grid would
    skip it) and `report` the JSON-safe record (true_mgs.json), which sets the
    result's loss beside the best candidate's. The keys are the same on both runs.
    """
    x_axis = run.scene_x_axis
    tx_z = float(run.real_tx_z)
    # the measured (or true) window as a run of scene samples, as every candidate's
    # window is (window_samples): the configured edges are not axis values to the
    # last bit, and the solve must get the same plate a candidate there would
    nom_min, nom_max = float(run.real_tx_x_min), float(run.real_tx_x_max)
    i_lo, i_hi = window_samples(np.asarray(x_axis), 0.5 * (nom_min + nom_max),
                                nom_max - nom_min)
    i_lo, i_hi = max(i_lo, 0), min(i_hi, len(x_axis) - 1)
    x_min, x_max = float(x_axis[i_lo]), float(x_axis[i_hi])
    # enumerate_grid's per-frequency check, at the TX plane
    qualities = [rs.sampling_quality(x_axis, np.array([run.rx_z]), wl, z_src=tx_z,
                                     forward_dir=-1.0)
                 for wl in run.wavelengths]
    freq_ok = [q >= 1.0 for q in qualities]
    point = GridPoint(index=-1, z=tx_z, x_center=0.5 * (x_min + x_max),
                      x_min=x_min, x_max=x_max, dx=float(run.grid_cfg.aperture.dx),
                      skip_reason=None if any(freq_ok) else
                      f"RS undersampled at every frequency (best quality "
                      f"{max(qualities):.3f} < 1)",
                      freq_ok=freq_ok)

    finite = [c for c in run.candidates if np.isfinite(c.final_loss)]
    best = min(finite, key=lambda c: c.final_loss) if finite else None
    which = "measured" if run.measured is not None else "true"
    report = {
        "kind": f"solve at the {which} TX position (grid-search-mgs --true-mgs)",
        "tx": {"z": point.z, "x_center": point.x_center, "x_min": x_min, "x_max": x_max,
               "width": x_max - x_min},
        "amplitude": _measured_tx_amplitude_text(params),
        "gs": {"engine": str(params.engine), "init": str(params.init),
               "effective_max_iters": int(params.max_iters), "seed": params.seed,
               "ref_freq_hz": float(run.ref_freq)},
        "skip_reason": point.skip_reason,
        "solve": None,
        "best_candidate": None if best is None else {
            "index": best.point.index, "z": best.point.z,
            "x_center": best.point.x_center, "final_loss": float(best.final_loss),
            "dist_to_tx_m": float(np.hypot(best.point.z - point.z,
                                           best.point.x_center - point.x_center))},
        "loss_ratio_to_best": None,
    }
    if params.amplitude != "fixed":
        # only on a fitted-amplitude run, so a fixed run's report is what it always
        # was; amp_max only where there is a cap ('descend')
        report["gs"]["amplitude"] = str(params.amplitude)
        if params.amp_max is not None:
            report["gs"]["amp_max"] = float(params.amp_max)
        # likewise only when the penalty is on (it needs a fitted amplitude)
        if params.power_penalty is not None:
            report["gs"]["power_penalty"] = float(params.power_penalty)
    if not point.ok:
        return None, report

    shared = SharedMeasurement(
        x_axis=np.asarray(x_axis).copy(),
        rx_z=float(run.rx_z),
        # rebuilt from the run's own arrays: the same values the candidates got
        channels=[FreqChannel(freq=f, wavelength=wl, rx_field=y, error_weighting=w)
                  for f, wl, y, w in zip(run.freqs, run.wavelengths, run.rx_fields,
                                         run.error_weightings)],
        params=params.model_copy(),
        ref_freq=float(run.ref_freq),
    )
    result = _reconstruct_candidate(point, shared)
    report["solve"] = {"final_loss": float(result.final_loss),
                       "final_mse": float(result.final_mse),
                       "power_ratio": result.power_ratio,
                       "per_freq_losses": _json_losses(result.per_freq_losses),
                       "freq_valid": result.freq_valid,
                       "n_iters_run": result.n_iters_run,
                       "stop_reason": result.stop_reason, "seed": result.seed}
    if best is not None and best.final_loss > 0:
        report["loss_ratio_to_best"] = float(result.final_loss / best.final_loss)
    return result, report


def _measured_tx_amplitude_text(params: GerchbergSaxtonConfig) -> str:
    """true_mgs.json's `amplitude`: what the solve at the measured (or, simulated,
    true) TX did with the plate's amplitude, which is what every candidate did.
    'fixed' reaches here on a measured run only: a simulated fixed run's --true-mgs
    is make_true_mgs_plot. With power_penalty on, the text says so: the loss the
    report compares is then the penalized objective."""
    if params.amplitude == "fixed":
        return ("uniform over the TX window, the candidates' assumption (the "
                "plate's own amplitude and beam are unknown)")
    penalty = ("" if params.power_penalty is None else
               "; loss = plain loss x max(1, P_plate/P_rx)^"
               f"{params.power_penalty:g} (excess-power penalty)")
    if params.amplitude == "level":
        return ("uniform level descended from the power-matched start, over the TX "
                "window, as for every candidate (one level shared by every tone, "
                f"fitted with the phase; no cap{penalty})")
    return ("descended from the back-propagated magnitude, capped at amp_max = "
            f"{params.amp_max:g}, over the TX window, as for every candidate (one "
            f"amplitude shared by every tone, fitted with the phase{penalty})")


def save_measured_tx_solve(run_dir: Path, result: Optional[CandidateResult],
                           report: dict) -> Path:
    """Write solve_at_measured_tx's output: true_mgs.json (the report) and, when
    the solve ran, true_mgs.npz (aperture + loss curve, a candidate npz's keys)."""
    run_dir = Path(run_dir)
    if result is not None:
        np.savez_compressed(run_dir / "true_mgs.npz",
                            aper_axis=f64(result.aper_axis),
                            aper_profile=c64(result.aper_profile),
                            loss_full=np.asarray(result.loss_full, dtype=np.float32))
    return write_json(run_dir / "true_mgs.json", report)


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

    With tx_aperture.amplitude set (anything but the default 1.0) the manifest
    gains a top-level `tx_amplitude` block (MGS.tx_amplitude_report): the setting,
    the factor the simulated plate carried and, for rx_peak, the clean RX peak it
    was set from. A default run's key set is unchanged.

    Every candidate entry, and its cand_####.json gs_result, carries `final_mse`
    (the plain joint residual) and `power_ratio` (null without the penalty).
    With gerchberg_saxton.power_penalty on, the manifest's gs block gains
    `power_penalty` and each `final_loss` is the penalized objective
    final_mse * power_ratio**power_penalty that ranks the candidates.

    A measured run (run.measured set) writes the same layout with two differences:
    measurement.npz has no real_tx_aper_axis / real_tx_aper_profile (the plate's
    profile is the unknown; ground_truth is its measured position, as on every run),
    and the manifest gains a `measured` block, the provenance of the measurement.
    Its config_snapshot.json records tx_aperture.beam as null (_save_measured_snapshot).

    The manifest is schema 3. Schema 4 (gs.refine, and refined / start /
    sweep_loss per candidate) is what the removed refinement pass wrote; no run
    writes it now, and the readers still read it (the extra keys are ignored).
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
    }
    if run.real_tx_aper_profiles is not None:   # None: a measured run's unknown plate
        meas["real_tx_aper_axis"] = f64(run.real_tx_aper_axis)
        meas["real_tx_aper_profile"] = c64(run.real_tx_aper_profiles)  # (F, nt)
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
                          "final_mse": cand.final_mse,
                          "power_ratio": cand.power_ratio,
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
            "final_mse": cand.final_mse,
            "power_ratio": cand.power_ratio,
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
        "rx_aperture": {"x_min": run.rx_x_min, "x_max": run.rx_x_max, "z": run.rx_z},
        "gs": {"effective_max_iters": run.effective_max_iters, "loss_combine": "mean",
               "init": run.init, "ref_freq_hz": run.ref_freq},
        "provenance": provenance(args_dict),
        "noise": run.noise,          # receiver-noise block; null when noiseless
    }
    if run.amplitude != "fixed":
        # only on a fitted-amplitude run, so a fixed manifest is what it always was;
        # amp_max only where there is a cap ('descend')
        manifest["gs"]["amplitude"] = run.amplitude
        if run.amp_max is not None:
            manifest["gs"]["amp_max"] = run.amp_max
        # likewise only when the penalty is on (it needs a fitted amplitude): the
        # candidates' final_loss is then J, and their final_mse the plain residual
        if run.power_penalty is not None:
            manifest["gs"]["power_penalty"] = run.power_penalty
    if run.measured is not None:
        # only on a measured run, so a simulated manifest is what it always was
        manifest["measured"] = run.measured
    if run.tx_amplitude is not None:
        # only when tx_aperture.amplitude is not the default 1.0, likewise: the
        # setting, the factor s the simulated plate carried, and (rx_peak) the
        # clean RX peak element of the unscaled plate that s was set from
        manifest["tx_amplitude"] = run.tx_amplitude
    if config.plot_window is not None:
        # only when set, likewise: the window the plots were drawn with, which
        # --replot uses again unless --plot-window overrides it
        manifest["plot_window"] = config.plot_window.model_dump()
    manifest.update({
        "counts": {"total": len(run.grid_points), "usable": n_usable,
                   "ran": len(run.candidates), "skipped": len(skipped_entries),
                   "ran_per_freq": ran_per_freq},
        "candidates": cand_entries,
        "skipped": skipped_entries,
    })
    write_json(run_dir / "candidate_beams.json", manifest)
    if run.measured is None:
        save_config_snapshot(run_dir, config, config_path)
    else:
        _save_measured_snapshot(run_dir, config, config_path)

    # logged, not persisted (a 49,377-candidate run writes ~10^5 files here)
    logging.getLogger().info(f"Saved grid run to {run_dir} in "
                             f"{time.perf_counter() - t0:.1f} s")


def _save_measured_snapshot(run_dir: Path, config: SimConfig, config_path: Path) -> None:
    """save_config_snapshot for a measured run, with tx_aperture.beam written as
    null. The config holds a stand-in beam there only so the block validates
    (config.py's _default_tx_aperture); it was never used, and a snapshot that
    named it would read as the TX's beam, which is the one thing a measured run
    does not know. The derived window (x_min, x_max, z) stays: it is where the
    plate sat. Neither file loads back (data_store.load_run_config refuses a
    measured run), so the snapshot is a record only."""
    save_config_snapshot(run_dir, config, config_path)
    snap = config.model_dump(mode="json")
    snap["tx_aperture"]["beam"] = None
    write_json(Path(run_dir) / "config_snapshot.json", snap)


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
