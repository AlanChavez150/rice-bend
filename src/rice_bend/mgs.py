import logging
import argparse
from collections import namedtuple
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np
from pydantic import ValidationError

from rice_bend import beam_aperatures, rs
from rice_bend.cli import setup_logging
from rice_bend.config import (DEFAULT_CONFIG, DEFAULT_FREQ_HZ, GerchbergSaxtonConfig,
                              SimConfig, apply_noise_overrides, center_freq_index,
                              load_config, parse_snr_db_arg, resolve_frequencies)
from rice_bend.interp import interp_amplitude, interp_real_imag
from rice_bend.landing import (beam_target, field_match, landing_metrics,
                               propagator_offset)
from rice_bend.noise import add_awgn
from rice_bend.plotting import draw_line_panel, draw_scene
from rice_bend.data_store import GSHistory, check_run_dir, make_run_dir, save_run
from rice_bend.exp_data import parse_oscope_heatmap_data, parse_oscope_rx_data
from rice_bend.sim_scene import SimAperature, SimScene


# One frequency's contribution to a joint solve: its measurement (on the scene
# x-axis, so every channel has the same length) and the weighting derived from it.
# Picklable, so a worker payload can carry a list of these.
FreqChannel = namedtuple("FreqChannel", "freq wavelength rx_field error_weighting")


# Warm-start stage 3: scan step for the absolute-offset search — fine enough to land
# well inside the joint basin; the joint descent polishes the remainder. The scan is
# elementwise work only (see _warm_start_phase), so a fine step costs nothing.
WARM_START_SCAN_STEP = 2 * np.pi / 64

# Backstop on offset samples for pathological near-duplicate frequency lists, whose
# synthetic period (2π·f_ref/Δf_min) explodes as the frequency gaps shrink.
WARM_START_MAX_SAMPLES = 100_000


def _warm_start_phase(tx_z, orig_aper_amp, x_axis, rx_z, channels, params, seed,
                      rho, ref_freq, h_fwd, support, curr_aper_amp, dx,
                      log=None) -> np.ndarray:
    """Multi-wavelength warm start: an initial psi inside the true basin.

    The joint delay solve must recover the ABSOLUTE profile — the 2π degeneracy
    that makes single-frequency retrieval easy is exactly what it breaks — and
    plain gradient descent from a random start lands in a wrong basin it cannot
    leave (measured: stuck ~100x above the reachable floor). The standard fix,
    per docs/delay_model_warm_start.md:

      1. solve the channel nearest the reference frequency ALONE — its degenerate
         landscape is easy, and it recovers the profile's SHAPE;
      2. unwrap that phase over the support — the per-point 2π ambiguities resolve
         relative to each other, collapsing the unknown to ONE scalar, the
         absolute offset delta;
      3. scan delta over one synthetic-wavelength period of the frequency comb
         against the joint loss — the comb makes the absolute offset observable
         (multi-wavelength interferometry), unique within one period. Propagation
         is linear, so each channel's field is propagated ONCE and every delta
         sample costs elementwise work only.

    The unwrap assumes a CONNECTED support and a smooth profile — true for every
    shipped scene (box candidate windows; the contiguous plate aperture).
    Returns the starting psi (zero off support).
    """
    # 1. reference solve. Pinning the PARENT's resolved seed keeps `seed: null`
    # runs reproducible end-to-end (the parent draws once; the child must not draw
    # its own); init='random' terminates the recursion.
    c_idx = int(np.argmin(np.abs(rho - 1.0)))
    ch_ref = channels[c_idx]
    sub_params = params.model_copy(update={"init": "random", "seed": int(seed)})
    stage1 = gs_reconstruct(tx_z=tx_z, orig_aper_amp=orig_aper_amp, x_axis=x_axis,
                            rx_z=rx_z, channels=[ch_ref], params=sub_params,
                            ref_freq=ch_ref.freq, capture=False, log=None)

    # 2. unwrap over the support and rescale into psi units (theta_c = rho_c * psi)
    psi0 = np.zeros(len(x_axis))
    idx = np.where(support)[0]
    psi0[idx] = np.unwrap(np.angle(stage1.curr_aper_f[idx])) / rho[c_idx]

    # 3. the absolute-offset scan. Adding delta multiplies channel f's field by
    # exp(1j*rho_f*delta); the joint loss is periodic in delta with the comb's
    # synthetic wavelength 2π·f_ref/Δf_min (the 2π fallback is defensive only —
    # the warm start requires F > 1 and frequencies are validated unique).
    freqs = np.sort(np.array([ch.freq for ch in channels]))
    gaps = np.diff(freqs)
    gaps = gaps[gaps > 0]
    period = 2 * np.pi * ref_freq / gaps.min() if gaps.size else 2 * np.pi
    n_samples = int(np.ceil(period / WARM_START_SCAN_STEP))
    if n_samples > WARM_START_MAX_SAMPLES:
        if log is not None:
            log.warning(f"warm start: synthetic period {period:.3g} rad needs "
                        f"{n_samples} offset samples; capping at {WARM_START_MAX_SAMPLES} "
                        "(nearly-duplicate frequencies?)")
        n_samples = WARM_START_MAX_SAMPLES
    deltas = np.linspace(0.0, period, n_samples, endpoint=False)

    base_props = [rs.rs_apply(curr_aper_amp * np.exp(1j * (rho[f] * psi0)),
                              h_fwd[f], dx)
                  for f in range(len(channels))]
    best_delta, best_loss = 0.0, np.inf
    for delta in deltas:
        loss = 0.0
        for f, ch in enumerate(channels):
            r = ch.error_weighting * (np.exp(1j * rho[f] * delta) * base_props[f]
                                      - ch.rx_field)
            loss += 0.5 * np.mean(np.abs(r) ** 2)
        loss /= len(channels)
        if loss < best_loss:
            best_loss, best_delta = loss, float(delta)

    if log is not None:
        log.info(f"warm start: stage-1 loss {stage1.final_loss:.4g} at "
                 f"{ch_ref.freq / 1e9:g} GHz; offset scan over {period:.1f} rad "
                 f"({n_samples} samples) -> delta {best_delta:.3f} "
                 f"(joint loss there {best_loss:.4g})")
    psi = psi0 + best_delta
    psi[~support] = 0.0
    return psi


# Result of one solver run. `curr_aper_f` is the reconstructed complex aperture on the
# math (scene) x-axis — the field at `ref_freq`; `history` is a GSHistory when
# capture=True, else None. `final_loss`/`loss_full` are the joint
# (mean-over-frequency) numbers; the `*_per_freq` twins carry each frequency's
# component, aligned with `frequencies` (the input channel order).
#
# `psi` is the solved unknown itself — the delay profile as phase at `ref_freq`, on
# the scene x-axis, zero off the support — and it is NOT recoverable from
# `curr_aper_f`: that is exp(1j*psi), which keeps psi only modulo 2π, while tone f's
# phase is (f/ref_freq)·psi and needs the absolute value. Set only when
# capture=True on the python engine (the single-shot path, where
# MGS.landing_report rebuilds every tone's field from it); None on the grid path
# and the rust engine, where nothing reads it and a sweep would carry one extra
# scene-length float array per candidate.
GSResult = namedtuple(
    "GSResult",
    "curr_aper_f final_loss final_loss_per_freq n_iters_run stop_reason seed "
    "loss_full loss_full_per_freq frequencies ref_freq history psi",
    defaults=(None,),
)


def rust_core():
    """The optional rust engine (rust/, module rice_bend_core), or a RuntimeError
    naming the build command. Never falls back to python silently: a run that
    asked for rust and got python would misattribute its timings and results."""
    try:
        import rice_bend_core
    except ImportError as e:
        raise RuntimeError(
            "gerchberg_saxton.engine is 'rust' but the rice_bend_core extension is "
            "not installed; build it with: VIRTUAL_ENV=$PWD/.venv .venv/bin/maturin "
            "develop --release -m rust/Cargo.toml") from e
    return rice_bend_core


def _gs_solve_rust(channels, params, ref_freq, seed, initial_phase, curr_aper_amp,
                   h_fwd, dx, log=None) -> "GSResult":
    """The warm start + descent of gs_reconstruct, run by rice_bend_core.gs_solve
    from the SAME seed draw, kernels and measurements the python path would use.
    Parity with the python engine: rust/python/parity.py, rust/README.md."""
    (curr_aper_f, final_loss, final_loss_pf, n_iters_run, stop_reason, loss_full,
     loss_full_pf) = rust_core().gs_solve(
        initial_phase, np.asarray(curr_aper_amp, dtype=np.float64),
        np.array([ch.freq for ch in channels], dtype=np.float64), float(ref_freq),
        np.stack([np.asarray(ch.rx_field, dtype=np.complex128) for ch in channels]),
        np.stack([np.asarray(ch.error_weighting, dtype=np.float64) for ch in channels]),
        np.stack(h_fwd).astype(np.complex128, copy=False), float(dx),
        params.init, int(params.max_iters), int(params.convergence_count),
        float(params.convergence_threshold), float(params.lr0),
        float(params.bt_shrink), int(params.bt_tries))
    if log is not None:
        if stop_reason == "converged":
            log.info(f"MGS has converged after {n_iters_run} iterations")
        else:
            log.error(f"MGS did not converge after {params.max_iters} iterations")
    return GSResult(curr_aper_f=curr_aper_f, final_loss=float(final_loss),
                    final_loss_per_freq=final_loss_pf,
                    n_iters_run=int(n_iters_run), stop_reason=str(stop_reason),
                    seed=seed, loss_full=loss_full, loss_full_per_freq=loss_full_pf,
                    frequencies=tuple(float(ch.freq) for ch in channels),
                    ref_freq=float(ref_freq), history=None)


def gs_reconstruct(tx_z: float, orig_aper_amp: np.ndarray, x_axis: np.ndarray,
                   rx_z: float, channels: Sequence[FreqChannel],
                   params: GerchbergSaxtonConfig, ref_freq: Optional[float] = None,
                   capture: bool = False, log=None) -> "GSResult":
    """Modified Gerchberg-Saxton solver core (no MGS/SimScene instance required).

    Solves for ONE aperture profile psi(x) at plane `tx_z` that best reproduces
    every channel's measured RX field at once, holding the amplitude fixed at
    `orig_aper_amp` (defined on `x_axis`; its nonzero region is the support).
    psi is a PLATE's shape — the optical-path/delay profile expressed as phase at
    `ref_freq` — and each channel's phase is (f/ref_freq)·psi, the way a physical
    plate's phase scales with wavenumber. This matches every simulated beam type
    (each generator bakes the wavenumber into the phase).

    The loss is the UNWEIGHTED MEAN of the per-frequency losses. Each channel's
    error_weighting is normalized to its own measurement's peak (see MGS.measure),
    so the mean weights frequencies equally regardless of absolute RX power; do
    not "fix" it into a power-weighted sum. A single frequency is simply the
    len(channels) == 1 case of the same code path (rho = 1.0 exactly, so psi IS
    that frequency's phase).

    `ref_freq` defaults to the centre-by-value of `channels`. Callers that solve
    different candidates on different channel SUBSETS (the grid sweep) must pass
    the run-level reference instead, so psi keeps the same units for every
    candidate.

    Pure given its arguments — arrays, scalars and a picklable config model — so
    this runs unchanged in a worker process.

    With `capture=True` a GSHistory is built and per-iteration state recorded, and
    the result carries the final psi (the single-shot `mgs` path, used by animation
    and the landing report); with `capture=False` only the dense loss curves + final
    aperture are produced (the grid-search path). The math is identical
    either way, so results do not depend on `capture`. `log` (optional) receives the
    per-iteration progress lines; workers pass None to stay quiet.

    `params.engine == "rust"` hands the warm start and descent to the rust engine
    after the shared setup below (seed, initial draw, kernels, sampling checks).
    GS history exists only on the python engine, so a capture=True solve runs
    wholly on python — including its warm-start stage-1 recursion.
    """
    if capture and params.engine == "rust":
        if log is not None:
            log.info("engine 'rust' does not capture GS history; this solve runs on python")
        params = params.model_copy(update={"engine": "python"})
    size = len(x_axis)
    dx = x_axis[1] - x_axis[0]
    n_freq = len(channels)
    assert n_freq >= 1, "gs_reconstruct needs at least one FreqChannel"
    if ref_freq is None:
        ref_freq = channels[center_freq_index([ch.freq for ch in channels])].freq
    # rho_f = f/f_ref = k_f/k_ref exactly: each channel's phase is rho_f * psi, and
    # the same factor chain-rules into its gradient contribution
    rho = np.array([ch.freq / ref_freq for ch in channels], dtype=np.float64)

    # The geometry is fixed for the whole solve, so there are exactly TWO kernels
    # PER FREQUENCY here -- the forward TX->RX one and its adjoint -- and rs() would
    # rebuild both from scipy.special.hankel1 over the full x axis on all three of
    # its calls per iteration, for up to 10,000 iterations. Building them once is
    # 71% of the numerical core: measured 1.225 s -> 0.351 s (3.49x) on a real
    # candidate, with final_loss, the aperture and the whole 800-point loss curve
    # bit-identical. (One build is ~0.375 ms, so the per-solve cost of F builds is
    # noise; it is the per-iteration rebuild that must never come back.)
    #
    # The adjoint is bit-exactly conj(forward): rs()'s two calls here differ only in
    # the sign of the propagation distance, and kernel_rs_inverse is defined as the
    # conjugate of kernel_rs. Verified for both propagation directions.
    #
    # sampling_quality depends on the propagation distance only through |prop|, which
    # is the same either way, so rs()'s per-call guard collapses to one check per
    # frequency here. The check raises on ANY undersampled channel (the highest
    # frequency binds); solving over a valid subset is the caller's decision, made
    # before building the channel list.
    rx_plane = np.array([rx_z])   # measurement (RX) plane target
    for ch in channels:
        rs.check_sampling(x_axis, rx_plane, ch.wavelength, z_src=tx_z, forward_dir=-1.0)
    h_fwd = [rs.rs_kernel(x_axis, ch.wavelength, -1.0 * (rx_z - tx_z)) for ch in channels]
    h_adj = [np.conjugate(h) for h in h_fwd]

    max_iters = params.max_iters
    cvrg_count = params.convergence_count
    lr0 = params.lr0
    bt_shrink = params.bt_shrink
    bt_tries = params.bt_tries
    conv_threshold = params.convergence_threshold

    # seeded initial phase for reproducibility; draw + record a seed if none given
    seed = params.seed
    if seed is None:
        seed = int(np.random.SeedSequence().entropy % (2**32))
    rng = np.random.default_rng(seed)

    # track aperature amplitude and phase seperatly. curr_aper_phase is the shared
    # unknown psi: the plate's delay profile expressed as phase at ref_freq.
    curr_aper_amp = np.abs(orig_aper_amp.copy())
    curr_aper_phase = 2 * np.pi * rng.random(size)
    initial_phase = curr_aper_phase.copy()

    support = np.abs(orig_aper_amp) > 0  # aperature mask

    if params.engine == "rust":
        return _gs_solve_rust(channels, params, ref_freq, seed, curr_aper_phase,
                              curr_aper_amp, h_fwd, dx, log=log)

    # Multi-wavelength warm start: replace the random start with a point inside the
    # true basin (reference-frequency solve -> unwrap -> absolute-offset scan). A
    # strict no-op at a single frequency — the random path above runs untouched —
    # which is what preserves the F=1 bit-identity guarantees under any config.
    if params.init == "warm_start" and n_freq > 1:
        curr_aper_phase = _warm_start_phase(
            tx_z=tx_z, orig_aper_amp=orig_aper_amp, x_axis=x_axis, rx_z=rx_z,
            channels=channels, params=params, seed=seed, rho=rho, ref_freq=ref_freq,
            h_fwd=h_fwd, support=support, curr_aper_amp=curr_aper_amp, dx=dx, log=log)
        initial_phase = curr_aper_phase.copy()

    history = None
    if capture:
        history = GSHistory(
            x_axis=x_axis,
            frequencies=np.array([ch.freq for ch in channels], dtype=np.float64),
            orig_prop_f=np.stack([np.asarray(ch.rx_field) for ch in channels]),
            orig_aper_amp=orig_aper_amp,
            error_weighting=np.stack([np.asarray(ch.error_weighting) for ch in channels]),
            support=support,
            initial_phase=initial_phase,
            rx_z=float(rx_z),
            seed=seed,
            history_stride=params.history_stride,
        )

    stop_reason = "max_iters"
    n_iters_run = max_iters
    hist_error = np.zeros(max_iters, np.float32)
    hist_error_pf = np.zeros((max_iters, n_freq), np.float32)
    for iter_idx in range(max_iters):
        # propogate aperature guess to measurement plane, once per frequency. A
        # python loop over channels, NOT a batched 2D fftconvolve: rs() benchmarked
        # the batched form slower at every size tried, and going through the same
        # rs_apply path per frequency keeps the load-bearing complex64 downcast --
        # which is what makes n_freq == 1 bit-identical to the single-frequency
        # solver this generalizes. Each channel builds its own field from the one
        # shared plate profile: phase = rho_f * psi.
        loss_pf = np.empty(n_freq, dtype=np.float64)
        grad_theta = np.zeros(size, dtype=np.float64)
        prop_fields = [] if history is not None else None
        for f, ch in enumerate(channels):
            u0 = curr_aper_amp * np.exp(1j * (rho[f] * curr_aper_phase))
            curr_prop_f = rs.rs_apply(u0, h_fwd[f], dx)
            r_cx = ch.error_weighting * (curr_prop_f - ch.rx_field)
            loss_pf[f] = 0.5 * np.mean(np.abs(r_cx)**2)
            if prop_fields is not None:
                prop_fields.append(curr_prop_f)
            # back propogate the residual from the RX plane to the TX (aperture) plane
            g_u0 = rs.rs_apply(r_cx, h_adj[f], dx)
            # chain rule: d(theta_f)/d(psi) = rho_f
            grad_theta += rho[f] * (2.0 * np.imag(g_u0 * np.conj(u0)))
        # gradient of the MEAN loss: the divisor is n_freq, not 1. Summing instead
        # would silently rescale the effective lr0 by n_freq and change backtracking
        # accept rates, breaking step-size comparability with single-frequency runs.
        grad_theta /= float(n_freq)
        grad_theta[~support] = 0.0

        loss = loss_pf.mean()
        hist_error[iter_idx] = loss
        hist_error_pf[iter_idx] = loss_pf
        if history is not None:
            history.record_iter(iter_idx, loss, loss_pf, curr_aper_phase, prop_fields)

        step = lr0
        for _ in range(bt_tries):
            theta_trial = curr_aper_phase - step * grad_theta
            trial_pf = np.empty(n_freq, dtype=np.float64)
            for f, ch in enumerate(channels):
                u0_trial = curr_aper_amp * np.exp(1j * (rho[f] * theta_trial))
                um_trial = rs.rs_apply(u0_trial, h_fwd[f], dx)
                r_trial = ch.error_weighting * (um_trial - ch.rx_field)
                trial_pf[f] = 0.5 * np.mean(np.abs(r_trial)**2)
            loss_trial = trial_pf.mean()
            if loss_trial < loss:  # sufficient decrease of the JOINT loss
                curr_aper_phase = theta_trial.copy()
                curr_aper_phase[~support] = 0.0
                loss = loss_trial
                loss_pf = trial_pf
                break
            step *= bt_shrink
        curr_aper_f = curr_aper_amp * np.exp(1j * curr_aper_phase)

        if log is not None and iter_idx % 1000 == 0:
            log.info(f"{iter_idx} loss {loss}")

        # if the last x iterations did not improve the error, GS has "converged"
        if iter_idx > cvrg_count:
            recent_err = hist_error[iter_idx-cvrg_count: iter_idx]
            recent_err_flatness = np.mean(np.diff(recent_err))
            if recent_err_flatness > conv_threshold:
                if log is not None:
                    log.info(f"MGS has converged after {iter_idx+1} iterations")
                stop_reason = "converged"
                n_iters_run = iter_idx + 1
                break

    if stop_reason == "max_iters" and log is not None:
        log.error(f"MGS did not converge after {max_iters} iterations")

    if history is not None:
        # the final captured prop fields are the PRE-step fields of the last
        # iteration (as they always were); loss/phase are post-accept.
        history.capture_final(iter_idx, loss, curr_aper_phase, prop_fields)
        history.finalize(stop_reason, n_iters_run, loss, loss_pf)

    loss_full = hist_error[:n_iters_run].copy()
    loss_full_per_freq = hist_error_pf[:n_iters_run].copy()
    return GSResult(curr_aper_f=curr_aper_f, final_loss=float(loss),
                    final_loss_per_freq=np.asarray(loss_pf, dtype=np.float64).copy(),
                    n_iters_run=n_iters_run, stop_reason=stop_reason, seed=seed,
                    loss_full=loss_full, loss_full_per_freq=loss_full_per_freq,
                    frequencies=tuple(float(ch.freq) for ch in channels),
                    ref_freq=float(ref_freq),
                    history=history,
                    psi=curr_aper_phase.copy() if capture else None)


@dataclass
class FreqState:
    """Everything MGS holds per frequency for a joint solve.

    The geometry of the apertures is frequency-independent (their axes come from
    config bounds + dx alone), but the RX element spacing is wavelength/20 when
    the config leaves dx null, and every beam constructor bakes the wavenumber
    into the emitted phase — so each frequency gets its own aperture objects.
    """
    freq: float
    wavelength: float
    rx_ap: SimAperature            # per-freq: dx = wavelength/20 when cfg dx is null
    tx_ap: SimAperature            # per-freq beam profile (every beam's phase ∝ k)
    rx_field: Optional[np.ndarray] = None        # set by measure(), on scene x_axis
    error_weighting: Optional[np.ndarray] = None  # set by measure(), on scene x_axis
    rx_plane_row: Optional[np.ndarray] = None    # full scene-axis field at the RX
    #   plane, BEFORE the RX-element interpolation (set by _synthesize_rx; None on
    #   the experimental path). analysis.py's energy metric needs the un-windowed
    #   field — rx_field is zero-filled outside the window.
    rx_aper_profile_clean: Optional[np.ndarray] = None  # the NOISELESS RX element
    #   profile when measure() added noise (rx_ap.aper_profile then holds the noisy
    #   one); None when noiseless or on the experimental path. rx_plane_row stays
    #   noiseless either way — noise is a receiver property, added at the elements.


def _db_or_none(num, den) -> Optional[float]:
    """10*log10(num/den) as a plain float, or None when it is not finite (a
    JSON-safe noise report: json.dump would write a bare NaN/Infinity)."""
    with np.errstate(divide="ignore", invalid="ignore"):
        v = float(10 * np.log10(np.float64(num) / np.float64(den)))
    return v if np.isfinite(v) else None


def _fmt_db(v: Optional[float]) -> str:
    return "n/a" if v is None else f"{v:.1f} dB"


def _fmt_mm(v: Optional[float]) -> str:
    """A position in metres as signed millimetres: +3.13."""
    return "n/a" if v is None else f"{v * 1e3:+.2f}"


def _fmt_num(v: Optional[float], spec: str) -> str:
    return "n/a" if v is None else format(v, spec)


def landing_lines(report: dict) -> List[str]:
    """MGS.landing_report() as the LANDING log block, one string per line.

    The layout is fixed, so the block reads the same in every log that carries it
    (`mgs`, the --true-mgs extras of a grid run): positions in mm with sign and two
    decimals, fractions with four, losses as %.3e, and a `<f> GHz` tag on every
    per-tone line (a true / reconstructed / match triple per tone, in the run's
    frequency order).
    """
    lines = []
    target = report["target"]
    if target is None:
        lines.append("LANDING target: none")
    else:
        lines.append(f"LANDING target: {target['kind']} x={_fmt_mm(target['x_m'])} mm "
                     f"({target['note']})")
    for rec in report["per_freq"]:
        ghz = f"{rec['freq_hz'] / 1e9:g} GHz"
        for label in ("true", "reconstructed"):
            m = rec[label]
            if m is None:
                lines.append(f"LANDING {label:<13} {ghz}: no power at the RX plane")
                continue
            lines.append(
                f"LANDING {label:<13} {ghz}: peak {_fmt_mm(m['peak_x_m'])} mm, "
                f"centroid {_fmt_mm(m['centroid_x_m'])} mm, "
                f"5-95% [{_fmt_mm(m['e05_x_m'])}, {_fmt_mm(m['e95_x_m'])}] mm, "
                f"in-window {m['frac_in_window']:.4f} | "
                f"in-window peak {_fmt_mm(m['peak_x_in_window_m'])} mm, "
                f"centroid {_fmt_mm(m['centroid_x_in_window_m'])} mm")
        vs_true, vs_target = rec["match_true"], rec["match_target"]
        lines.append(
            f"LANDING {'match':<13} {ghz}: corr {_fmt_num(vs_true['corr'], '.4f')}, "
            f"rel-L2 vs true {_fmt_num(vs_true['rel_l2'], '.3f')}, "
            f"power ratio {_fmt_num(vs_true['power_ratio'], '.3f')}, "
            f"rel-L2 vs target {_fmt_num(vs_target['rel_l2'], '.3f')}")
    loss = report["loss"]
    lines.append(
        f"LANDING loss solver={loss['solver']:.3e} true_plate={loss['true_plate']:.3e} "
        f"zero_model={loss['zero_model']:.3e} ratio={_fmt_num(loss['ratio'], '.3f')} "
        f"iters={loss['n_iters']} stop={loss['stop_reason']}")
    bias, dx = report["dx_bias_m"], report["scene_dx_m"]
    if abs(bias - 0.5 * dx) <= 1e-6 * dx:
        offset = f"+dx/2 lateral offset ({bias * 1e3:+.3f} mm)"
    else:
        # not one of the shipped scene axes: see landing.propagator_offset
        offset = (f"lateral offset ({bias * 1e3:+.3f} mm; not +dx/2: the scene x-axis "
                  "is not an even-length axis centred on x = 0)")
    lines.append(f"LANDING note: positions carry the propagator's {offset}; outside "
                 "the RX window the reconstruction is unconstrained")
    return lines


class MGS():
    def __init__(self, freqs, config: SimConfig):
        self._init_common(freqs, config)

        # Convention: the RX aperture sits at the origin (z=0), the bottom of the
        # image. The TX aperture sits above it and projects toward -Z, so its beam
        # travels *down* to the RX. Move the TX by changing tx.z (height) and its
        # x_min/x_max (lateral).
        scene_cfg = config.sim_scene
        self.tx_cfg = config.tx_aperture
        for freq in self.freqs:
            wl = rs.wavelength(freq)
            self.freq_states.append(FreqState(
                freq=freq, wavelength=wl,
                rx_ap=self._build_rx(config.rx_aperture, wl),
                tx_ap=self._build_tx(config.tx_aperture, scene_cfg.z_min, freq),
            ))
        # the scene holds the PRIMARY frequency's apertures as its views; geometry
        # is identical across frequencies, only profiles/dx differ
        primary = self.freq_states[0]
        tx = primary.tx_ap
        self.gs_tx = SimAperature(x_min=tx.x_min, x_max=tx.x_max, z=tx.z, dx=tx.dx)
        self.scene = SimScene(
            x_min=scene_cfg.x_min, x_max=scene_cfg.x_max,
            z_min=scene_cfg.z_min, z_max=scene_cfg.z_max,
            spacing=scene_cfg.spacing, rx_ap=primary.rx_ap, tx_ap=tx,
        )
        self._log_geometry()

    def _init_common(self, freqs, config: SimConfig) -> None:
        """State every MGS has, whatever built its scene.

        `freqs` is a float or a sequence of floats (a joint solve holds one
        FreqState per frequency). `self.freq`/`self.wavelength` alias the primary
        (first) frequency for the inherently single-frequency consumers: geometry
        logging, scene (re-)illumination, plot_scene.
        """
        self.log = logging.getLogger()
        if np.isscalar(freqs):
            freqs = [freqs]
        self.freqs = [float(f) for f in freqs]
        assert len(self.freqs) >= 1, "MGS needs at least one frequency"
        self.freq = self.freqs[0]
        self.wavelength = rs.wavelength(self.freq)
        self.plot_path = config.plot_path
        self.gs_cfg = config.gerchberg_saxton
        # an ALIAS, like gs_cfg: a drawn noise seed written back here lands in the
        # config that save_run later snapshots
        self.noise_cfg = config.noise
        self.noise_report = None       # set by measure() when noise is on
        self.output_cfg = config.output
        self.tx_cfg = None             # the simulated constructor sets it: the beam's
        #   geometric landing target (landing_report) is read from the TX config, and
        #   an experimental capture has no configured beam to aim
        self.gs_history = None
        self.gs_result = None
        self.freq_states: List[FreqState] = []   # filled by the constructor path
        self.gs_rec_data = None        # scene re-illuminated by the reconstruction

    @classmethod
    def from_experiment(cls, rx_path: Path, heatmap_path: Path, freq: float,
                        config: SimConfig) -> "MGS":
        """Build an MGS from experimental .mat captures instead of a simulated scene.

        Geometry comes from the capture and the bench rather than config.sim_scene:
        the RX aperture IS the measurement, and the TX aperture is what the
        experiment is trying to recover (has_real_aper is False, so no "Real TX"
        overlay is drawn and measure() does not synthesize an RX field).

        Everything the rig contributes -- the down-conversion chain, the aperture
        edges, the coordinate origins, the amplitude normalisation, the scene
        margins -- lives in config.experimental rather than as thirteen magic
        numbers spread across two modules.
        """
        exp = config.experimental
        # alternate constructor: __init__ builds a scene from config, and here the
        # scene comes from data, so there is nothing for it to do
        self = cls.__new__(cls)
        self._init_common(freq, config)
        self.log.info(f"Carrier Frequency: {freq*1e-9: 0.2f} GHz")

        self.log.info(f"Reading file as RX data: {rx_path}")
        rx = parse_oscope_rx_data(rx_path, freq, exp)
        rx.z = exp.rx_z_origin - rx.z
        dx = rx.dx

        # TX aperture edges in rig coordinates, mirrored into scene coordinates
        tx_x_min = exp.rig_x_origin - exp.tx_left_edge
        tx_x_max = exp.rig_x_origin - exp.tx_right_edge
        # recentre everything so the TX aperture straddles x = 0
        x_offset = -((tx_x_max - tx_x_min) / 2.0 + tx_x_min)

        rx_centred = SimAperature(x_min=rx.x_min + x_offset, x_max=rx.x_max + x_offset,
                                  z=rx.z, dx=dx)
        # the capture can be one sample longer than the recentred aperture's own
        # sampling, so it is truncated rather than resampled
        rx_centred.aper_profile = rx.aper_profile[0:len(rx_centred.aper_profile)]
        rx_centred.aper_profile = exp.rx_amplitude_scale * (
            rx_centred.aper_profile / np.max(rx_centred.aper_profile))
        rx = rx_centred

        tx = SimAperature(x_min=tx_x_min + x_offset, x_max=tx_x_max + x_offset,
                          z=0, dx=dx)
        # broadside plane wave: the solver's seed
        tx.aper_profile = beam_aperatures.steered_profile(freq, tx.aper_axis, theta_deg=0)
        self.gs_tx = SimAperature(x_min=tx.x_min, x_max=tx.x_max, z=tx.z, dx=dx)
        # a single capture is inherently one frequency: the length-1 joint path
        self.freq_states = [FreqState(freq=self.freq, wavelength=self.wavelength,
                                      rx_ap=rx, tx_ap=tx)]

        spacing_ratio = round(dx / self.wavelength, 3)
        self.log.info(f"RX measurements are spread {spacing_ratio:0.3f} wavelengths apart")

        self.log.info(f"Reading file with heatmap data: {heatmap_path}")
        base_scene = SimScene(
            x_min=tx.x_min - exp.scene_x_margin, x_max=tx.x_max + exp.scene_x_margin,
            z_min=exp.scene_z_min, z_max=exp.scene_z_max,
            spacing=dx, rx_ap=rx, tx_ap=tx,
        )
        self.scene = parse_oscope_heatmap_data(heatmap_path, base_scene, x_offset,
                                               freq, exp)

        self.has_real_aper = False   # the experimental TX aperture is unknown
        self.beam_type = "directional"
        self._log_geometry()
        return self

    def _build_rx(self, rx_cfg, wavelength: float) -> SimAperature:
        """The receive aperture at `wavelength`: a window of `width` centred at
        `x_center`, at the scene origin (z=0). `dx` defaults to wavelength/20 when
        left null -- so the element count is per-frequency -- and is what sets the
        receiver element count, the independent variable in the
        scenario_caustic_hit_lambda2/lambda4 experiments."""
        # written as a multiply by 1/20, not a divide by 20: 0.05 is not exactly 1/20
        # in binary, so the two disagree by an ulp (measured on 28 config x frequency
        # combinations, though none of them moved num_points)
        rx_dx = rx_cfg.dx if rx_cfg.dx is not None else wavelength * (1 / 20)
        return SimAperature(
            x_min=rx_cfg.x_center - rx_cfg.width / 2.0,
            x_max=rx_cfg.x_center + rx_cfg.width / 2.0,
            z=0,
            dx=rx_dx,
        )

    def _build_tx(self, tx_cfg, z_min: float, freq: float) -> SimAperature:
        """The transmit aperture and the beam it emits at `freq`, defined
        independently of the scene grid. Sets self.beam_type and self.has_real_aper.

        The profile comes from beam_aperatures.py, whose generators all bake the
        wavenumber into the PHASE only, so the aperture amplitude (the solver's
        fixed constraint) is frequency-independent."""
        tx = SimAperature(x_min=tx_cfg.x_min, x_max=tx_cfg.x_max, z=tx_cfg.z, dx=tx_cfg.dx)
        assert tx.z > z_min, f"tx_aperture.z ({tx.z}) must be above the scene floor z_min ({z_min})"

        beam = tx_cfg.beam
        self.beam_type = beam.type
        # a known real aperture exists (every simulated beam); false for the
        # experimental path, where the TX aperture is what we are trying to recover
        self.has_real_aper = True
        tx.aper_profile = beam_aperatures.beam_profile(beam, freq, tx.aper_axis, tx.z - z_min)
        return tx

    def _log_geometry(self) -> None:
        scene = self.scene
        self.log.info(f"Simulation scene:")
        self.log.info(f" - X axis: {scene.x_min:0.3f} - {scene.x_max:0.3f}")
        self.log.info(f" - Z axis: {scene.z_min:0.3f} - {scene.z_max:0.3f}")
        self.log.info(f"RX aperature:")
        rx_wl_ratio = scene.rx_ap.dx / self.wavelength
        rx_width = scene.rx_ap.x_max - scene.rx_ap.x_min
        self.log.info(f" - X axis: {scene.rx_ap.x_min:0.3f} {scene.rx_ap.x_max:0.3f} (width {rx_width:0.3f} m, dx {rx_wl_ratio:0.3f} wavelength)")
        self.log.info(f" - Z: {scene.rx_ap.z:0.3f}")
        self.log.info(f"TX aperature ({self.beam_type} beam)")
        tx_wl_ratio = scene.tx_ap.dx / self.wavelength
        self.log.info(f" - X axis: {scene.tx_ap.x_min:0.3f} {scene.tx_ap.x_max:0.3f} (dx {tx_wl_ratio:0.3f} wavelength)")
        self.log.info(f" - Z: {scene.tx_ap.z:0.3f}")

    # ----------------------------------------------------------------- scene ---
    def _propagate(self, tx_ap: SimAperature) -> np.ndarray:
        """Illuminate the whole scene from `tx_ap`: resample its profile onto the
        scene x-axis, then RS-propagate down through every z plane."""
        u0 = interp_real_imag(tx_ap.aper_axis, tx_ap.aper_profile, self.scene.x_axis)
        self.log.info("Computing wave propogation across scene")
        return rs.illuminate(self.scene.x_axis, self.scene.z_axis, u0,
                             self.wavelength, tx_ap.z)

    def illuminate_real(self) -> None:
        """Fill scene.data with the field radiated by the real TX aperture.

        Only the plots need this. The RX measurement does NOT come from here -- see
        _synthesize_rx, which propagates to one plane instead of all 3400 of them.
        """
        self.log.info("Running scene simulation with real aperature")
        self.scene.data = self._propagate(self.scene.tx_ap)

    def illuminate_reconstructed(self) -> None:
        """Fill gs_rec_data with the field radiated by the MGS-reconstructed aperture.

        A re-illumination is inherently monochromatic; this (like _propagate and
        plot_scene) runs at the PRIMARY frequency. The reconstruction itself is one
        delay plate fitted jointly across all configured frequencies, stored as its
        reference-frequency field."""
        self.log.info("Running scene simulation with MGS reconstructed aperature")
        self.gs_rec_data = self._propagate(self.gs_tx)

    # ----------------------------------------------------------- measurement ---
    def _synthesize_rx(self, fs: FreqState) -> np.ndarray:
        """Propagate the real TX aperture to the RX measurement plane at this
        frequency and sample it onto the RX element axis. Returns the complex RX
        aperture profile.

        Propagates directly to rx_ap.z -- the plane gs_reconstruct actually models.
        It used to snap to the nearest scene z-plane, with the comparison inverted so
        it landed on the FARTHER one (index 1, never 0, for all 12 shipped configs):
        the measurement was synthesized 0.25 mm, lambda/8 at 150 GHz, away from where
        it was then fitted. Snapping is not merely inverted but wrong in principle --
        config.py permits z_min > 0, where rx.z sits below z_axis[0] and no scene
        plane is the right answer.

        Propagates to ONE plane, not all of them. run_grid_search used to call
        run_sim purely to get this, paying a full 3400 x 2400 scene (1.71 s and
        +190 MB RSS per frequency, held in the parent for the whole sweep) to keep a
        single row. Verified bit-identical: rs.rs(x, [z_k], ...)[0] equals
        rs.rs(x, z_axis, ...)[k], since each row is computed independently from u0.

        The round trip through rx_ap.aper_axis is kept deliberately -- it models the
        receiver element spacing (lambda/2 vs lambda/20), which is the whole point of
        scenario_caustic_hit_lambda2.yml.
        """
        scene = self.scene
        tx_ap = fs.tx_ap
        u0 = interp_real_imag(tx_ap.aper_axis, tx_ap.aper_profile, scene.x_axis)
        row = rs.rs(scene.x_axis, np.array([fs.rx_ap.z]), u0, fs.wavelength,
                    z_src=tx_ap.z, forward_dir=-1.0)[0]
        fs.rx_plane_row = row   # kept for the energy metric (analysis.py)
        return interp_real_imag(scene.x_axis, row, fs.rx_ap.aper_axis)

    def measure(self) -> None:
        """Sample the measured RX field onto the scene grid and build the error
        weighting used by phase retrieval, once per frequency. Idempotent.

        On the simulated path each frequency's measurement is synthesized here, by
        propagating that frequency's known TX aperture to the RX plane. On the
        experimental path there is no known TX aperture (has_real_aper is False)
        and rx_ap.aper_profile already holds the measured data. The results
        (FreqState.rx_field / .error_weighting) are the single measurement set
        shared across every hypothesized TX location.

        Each frequency's weighting is normalized to its OWN measurement's peak, so
        a joint solve's mean loss weights frequencies equally regardless of their
        absolute RX power.

        With noise.snr_db set, the simulated path adds complex AWGN to every tone's
        RX ELEMENT profile before it is interpolated onto the scene grid — where a
        real receiver adds it (rice_bend/noise.py has the model). The floor is
        referenced to the peak over every tone, so all clean profiles are
        synthesized first, then noised, then interpolated. The weighting is built
        from the NOISY measurement, as it would be from a capture. The clean profile
        is kept on FreqState.rx_aper_profile_clean and the realized SNRs in
        self.noise_report. Idempotency holds with noise too: the clean field is
        re-synthesized from the TX and each tone's stream rebuilt from (seed,
        frequency), and a null seed is drawn once and written back to the config.
        On the experimental path noise.snr_db is ignored (with a warning): the
        capture carries its own receiver noise.
        """
        # Fail fast, naming EVERY undersampled frequency at once — rs() would raise
        # on only the first it meets, and the highest frequency binds, so "which
        # frequencies are the problem" is exactly the question the error must answer.
        if self.has_real_aper:
            rx_plane = np.array([self.scene.rx_ap.z])
            bad = []
            for fs in self.freq_states:
                q = rs.sampling_quality(self.scene.x_axis, rx_plane, fs.wavelength,
                                        z_src=self.scene.tx_ap.z, forward_dir=-1.0)
                if q < 1:
                    bad.append(f"{fs.freq / 1e9:g} GHz (quality {q:.3f})")
            if bad:
                raise RuntimeError(
                    f"RS undersampled for {len(bad)} of {len(self.freq_states)} "
                    f"frequencies at the TX plane: {', '.join(bad)}. Use a finer "
                    f"sim_scene.spacing or drop the highest frequencies.")

        x_axis = self.scene.x_axis
        snr_db = self.noise_cfg.snr_db
        noise_on = self.has_real_aper and snr_db is not None
        if self.has_real_aper:
            # every clean profile first: the noise floor is referenced to the peak
            # over ALL tones, so no tone can be noised before the last is synthesized
            clean = [self._synthesize_rx(fs) for fs in self.freq_states]
            if noise_on:
                if self.noise_cfg.seed is None:
                    # same idiom as the solver's seed; written back so the run is
                    # replayable from its config snapshot, and so a second measure()
                    # reuses it (idempotency)
                    self.noise_cfg.seed = int(np.random.SeedSequence().entropy % (2**32))
                seed = self.noise_cfg.seed
                noisy, peak_power, sigma2 = add_awgn(
                    clean, [fs.freq for fs in self.freq_states], snr_db, seed)
                for fs, c, y in zip(self.freq_states, clean, noisy):
                    fs.rx_ap.aper_profile = y
                    fs.rx_aper_profile_clean = c
            else:
                for fs, c in zip(self.freq_states, clean):
                    fs.rx_ap.aper_profile = c
                    fs.rx_aper_profile_clean = None
        elif snr_db is not None:
            self.log.warning(f"noise.snr_db ({snr_db:g} dB) is ignored on the experimental "
                             "path: the capture carries its own receiver noise")

        per_freq = []
        for fs in self.freq_states:
            rx_ap = fs.rx_ap
            # A FIELD, so cartesian: interpolating its phase through the ±π branch cut
            # corrupted the measurement by 20.8% rel L2 on scenario_caustic_hit and 55%
            # on the lambda/2-spaced variant, and the solver then fitted the corruption
            # at full weight.
            orig_prop_f = interp_real_imag(rx_ap.aper_axis, rx_ap.aper_profile, x_axis)
            # error computations are weighted to favor higher amplitude data, and ignore things outside the recieve aperature
            error_weighting = (np.abs(orig_prop_f) / np.abs(orig_prop_f).max()) + 0.25
            error_weighting[(x_axis < rx_ap.x_min) | (x_axis > rx_ap.x_max)] = 0.0
            fs.rx_field = orig_prop_f
            fs.error_weighting = error_weighting
            if noise_on:
                # what the solver actually sees: the interpolation onto the scene
                # grid averages the element noise, so this sits above the per-antenna
                # figure. Measured inside the RX window only (the weighting zeroes
                # the rest).
                clean_interp = interp_real_imag(rx_ap.aper_axis, fs.rx_aper_profile_clean,
                                                x_axis)
                in_win = (x_axis >= rx_ap.x_min) & (x_axis <= rx_ap.x_max)
                per_freq.append({
                    "freq_hz": float(fs.freq),
                    "n_elements": int(len(fs.rx_aper_profile_clean)),
                    "mean_snr_db": _db_or_none(
                        np.mean(np.abs(fs.rx_aper_profile_clean) ** 2), sigma2),
                    "effective_snr_db": _db_or_none(
                        np.sum(np.abs(clean_interp[in_win]) ** 2),
                        np.sum(np.abs(orig_prop_f[in_win] - clean_interp[in_win]) ** 2)),
                })

        if not noise_on:
            self.noise_report = None
            return
        self.noise_report = {
            "snr_db": float(snr_db), "seed": int(seed),
            "reference": "peak_rx_elements",
            "peak_power": float(peak_power), "sigma2": float(sigma2),
            "numpy_version": np.__version__,
            "per_freq": per_freq,
        }
        self.log.info(f"RX noise: SNR {snr_db:g} dB re peak RX element, seed {seed}, "
                      f"peak power {peak_power:.4g}, sigma^2 {sigma2:.4g}")
        for rec in per_freq:
            self.log.info(f" - {rec['freq_hz'] / 1e9:g} GHz: {rec['n_elements']} elements, "
                          f"mean per-antenna SNR {_fmt_db(rec['mean_snr_db'])}, "
                          f"effective solver-grid SNR {_fmt_db(rec['effective_snr_db'])}")

    def measurement_channels(self) -> List[FreqChannel]:
        """FreqChannel payloads for gs_reconstruct, one per frequency in self.freqs
        order. measure() must have been called."""
        assert all(fs.rx_field is not None for fs in self.freq_states), \
            "measure() must be called before measurement_channels()"
        return [FreqChannel(freq=fs.freq, wavelength=fs.wavelength,
                            rx_field=fs.rx_field, error_weighting=fs.error_weighting)
                for fs in self.freq_states]

    def run_gerch_sax(self):
        """Reconstruct the aperture phase at the *real* TX plane (single-shot path).

        Thin wrapper over measure() + reconstruct_at() that preserves the original
        behaviour: solve at scene.tx_ap.z using the real aperture's amplitude as the
        fixed support, writing the result into self.gs_tx. One solve, jointly over
        every configured frequency.
        """
        self.log.info(f"Running modified Gerchberg-saxton algorithm")
        self.measure()
        # amplitude-only ON PURPOSE: this is the solver's fixed amplitude constraint
        # and its support mask, not a field. Forcing it cartesian (|interp_real_imag|)
        # is a different array: the straight line between two samples of different
        # phase cuts inside the arc, so it dips wherever the plate's phase turns
        # between aperture samples -- by an amount that depends on the beam and on
        # tx_aperture.dx, with no error raised and no visibly broken plot.
        #
        # The PRIMARY frequency's profile serves as THE constraint for a joint
        # solve: every beam constructor emits unit amplitude (only the phase depends
        # on the wavenumber), so the amplitude is frequency-independent by
        # construction and any freq_state's profile gives the same array.
        tx_ap = self.scene.tx_ap
        orig_aper_amp = interp_amplitude(tx_ap.aper_axis, tx_ap.aper_profile,
                                         self.scene.x_axis)
        result = self.reconstruct_at(
            tx_z=self.scene.tx_ap.z,
            orig_aper_amp=orig_aper_amp,
            out_aper=self.gs_tx,
        )
        self.gs_result = result
        self.gs_history = result.history

    def reconstruct_at(self, tx_z: float, orig_aper_amp: np.ndarray,
                       out_aper: SimAperature) -> GSResult:
        """Run modified Gerchberg-Saxton for a hypothesized TX plane.

        Solves for the one aperture phase at plane `tx_z` that best reproduces
        every frequency's measured RX field (FreqState.rx_field), holding the
        amplitude fixed at `orig_aper_amp` (defined on scene.x_axis; its nonzero
        region is the support). Writes the reconstructed complex aperture into
        out_aper.aper_profile and returns the run's full GSResult. measure() must
        have been called first.
        """
        # Run the solver core (capture=True so the single-shot path keeps its full
        # GSHistory for animation/persistence), then interp the result onto out_aper.
        result = gs_reconstruct(
            tx_z=tx_z,
            orig_aper_amp=orig_aper_amp,
            x_axis=self.scene.x_axis.copy(),
            rx_z=self.scene.rx_ap.z,
            channels=self.measurement_channels(),
            params=self.gs_cfg,
            ref_freq=self.freqs[center_freq_index(self.freqs)],
            capture=True,
            log=self.log,
        )
        # the converged aperture is a FIELD, and it is heavily wrapped -- cartesian
        out_aper.aper_profile = interp_real_imag(
            self.scene.x_axis, result.curr_aper_f, out_aper.aper_axis)
        return result

    # --------------------------------------------------------------- landing ---
    def _joint_loss(self, rx_rows: Sequence[np.ndarray]) -> float:
        """The solver's joint loss for one RX-plane field per tone (self.freqs
        order): the mean over tones of 0.5*mean(|w*(row - rx_field)|^2), the
        expression gs_reconstruct descends, so the result compares directly with
        gs_result.final_loss."""
        return float(np.mean([
            0.5 * np.mean(np.abs(fs.error_weighting * (row - fs.rx_field)) ** 2)
            for fs, row in zip(self.freq_states, rx_rows)]))

    def landing_report(self) -> dict:
        """Where the energy lands on the RX plane — the real beam against the
        solver's reconstruction, per tone — and the final loss in context.
        Simulated path only; run_gerch_sax() must have run.

            {"target":    landing.beam_target(...): the beam's geometric aim, or None
             "rx_window": {"x_min", "x_max"}
             "dx_bias_m": the propagator's lateral offset; half the scene x step
                          on every shipped scene (see the last paragraph)
             "scene_dx_m": the scene x step
             "per_freq":  [{"freq_hz",
                            "true":          landing_metrics(true row)
                            "reconstructed": landing_metrics(reconstructed row)
                            "match_true":    field_match(true row, reconstructed row)
                            "match_target":  field_match(rx_field, reconstructed row)},
                           ...]                                  self.freqs order
             "loss": {"solver", "true_plate", "zero_model", "ratio", "n_iters",
                      "stop_reason"}}

        The true row is FreqState.rx_plane_row: the clean field of the real plate at
        the RX plane, before the RX elements sample it and before any noise. The
        reconstructed row is the solver's OWN model propagated the solver's own way
        — amp * exp(1j * (f/ref_freq) * psi) on the scene axis, through the same
        kernel — so the joint loss of these rows IS gs_result.final_loss. It is
        deliberately not gs_rec_data[0]: that is gs_tx re-illuminated, which has
        been resampled scene -> aperture -> scene on the way, and on a multi-tone
        run it is the reference-frequency field propagated at the first-listed
        frequency.

        match_true answers the question being asked (does the energy land where
        the real beam's does); match_target is what the solver was actually given
        to fit: the measurement as the RX elements sampled it, noise included. The
        two differ by the receiver, not by the solver — a sparse array's
        interpolated target carries less power than the field it samples.

        `loss` sets the solver's final loss against two references evaluated with
        the same expression: `true_plate`, the loss of the REAL plate's phase under
        the solver's model (the solver's amplitude constraint times
        exp(1j*angle(true profile on the scene axis)), per tone), and `zero_model`,
        the loss of radiating nothing. `ratio` = solver / true_plate. The true plate
        does not score zero: the target is the field as the RX elements sampled it
        (and noised it), which the true plate does not reproduce — so with a sparse
        or noisy receiver the solver gets BELOW the true plate, and a ratio under 1
        is the ordinary case, not a suspicious one. A ratio well above 1 means the
        solve stopped short of the true plate: cut off by max_iters (stop_reason
        says so), or converged in another basin, which a multi-tone solve does when
        the warm start's phase unwrap slips by 2π.

        Positions are read off the scene grid and carry the propagator's lateral
        offset, reported as `dx_bias_m`: landing.propagator_offset, which is +dx/2
        on the even-length, origin-centred x-axis of every shipped scene (and is
        reported as whatever it is on any other). Outside the RX window the
        weighting is zero and the reconstruction is unconstrained, so its full-row
        numbers (peak, centroid, 5-95 % span) say where the fitted plate happens to
        throw its energy; only the in-window ones are held to the measurement.
        """
        result, hist = self.gs_result, self.gs_history
        if not self.has_real_aper:
            raise RuntimeError("landing_report needs the real TX aperture; the "
                               "experimental path has none to compare against")
        if result is None or hist is None or result.psi is None:
            raise RuntimeError("landing_report needs a solved run; call "
                               "run_gerch_sax() first")
        scene = self.scene
        x_axis = scene.x_axis
        dx = x_axis[1] - x_axis[0]
        rx_lo, rx_hi = scene.rx_ap.x_min, scene.rx_ap.x_max
        # the fixed amplitude constraint exactly as the solver held it
        amp = np.abs(hist.orig_aper_amp)

        per_freq, rec_rows, true_plate_rows = [], [], []
        for fs in self.freq_states:
            # gs_reconstruct's forward kernel and field construction, term for term
            h_fwd = rs.rs_kernel(x_axis, fs.wavelength,
                                 -1.0 * (scene.rx_ap.z - scene.tx_ap.z))
            rho = fs.freq / result.ref_freq
            rec_row = rs.rs_apply(amp * np.exp(1j * (rho * result.psi)), h_fwd, dx)
            rec_rows.append(rec_row)
            # the real plate as the solver could represent it: its own phase on the
            # scene axis, under the solver's amplitude
            true_u0 = interp_real_imag(fs.tx_ap.aper_axis, fs.tx_ap.aper_profile, x_axis)
            true_plate_rows.append(
                rs.rs_apply(amp * np.exp(1j * np.angle(true_u0)), h_fwd, dx))
            per_freq.append({
                "freq_hz": float(fs.freq),
                "true": landing_metrics(x_axis, fs.rx_plane_row, rx_lo, rx_hi),
                "reconstructed": landing_metrics(x_axis, rec_row, rx_lo, rx_hi),
                "match_true": field_match(fs.rx_plane_row, rec_row, x_axis, rx_lo, rx_hi),
                "match_target": field_match(fs.rx_field, rec_row, x_axis, rx_lo, rx_hi),
            })

        solver_loss = float(result.final_loss)
        true_plate_loss = self._joint_loss(true_plate_rows)
        dark = np.zeros(len(x_axis), dtype=np.complex64)
        return {
            "target": beam_target(self.tx_cfg, rx_z=scene.rx_ap.z),
            "rx_window": {"x_min": float(rx_lo), "x_max": float(rx_hi)},
            "dx_bias_m": propagator_offset(x_axis),
            "scene_dx_m": float(dx),
            "per_freq": per_freq,
            "loss": {
                "solver": solver_loss,
                "true_plate": true_plate_loss,
                "zero_model": self._joint_loss([dark] * len(self.freq_states)),
                "ratio": solver_loss / true_plate_loss if true_plate_loss > 0 else None,
                "n_iters": int(result.n_iters_run),
                "stop_reason": str(result.stop_reason),
            },
        }

    def log_landing(self) -> dict:
        """Log landing_report() as the LANDING block (landing_lines has the layout)
        and return the report."""
        report = self.landing_report()
        for line in landing_lines(report):
            self.log.info(line)
        return report

    def plot_scene(self, save_path=None, show=True):
        """Save the 4-panel scene plot (real vs MGS-reconstructed scene + TX aperture
        phase/amplitude). Writes to `save_path` if given, else self.plot_path; only
        opens an interactive window when `show` is True (set False for headless/batch)."""
        # imported here, not at module scope: pyplot costs ~137 ms and drags in
        # mpl_toolkits.mplot3d, which the grid sweep imports mgs without ever wanting
        import matplotlib.pyplot as plt

        self.log.info("Plotting scene")

        scene, gs_tx = self.scene, self.gs_tx
        bounds = (scene.x_axis.min(), scene.x_axis.max(),
                  scene.z_axis.min(), scene.z_axis.max())
        panel = dict(bounds=bounds, rx_axis=scene.rx_ap.aper_axis, rx_z=scene.rx_ap.z,
                     tx_axis=scene.tx_ap.aper_axis, tx_z=scene.tx_ap.z,
                     colorbar_label="EMW (V/m)")

        if scene.data is None or self.gs_rec_data is None:
            raise RuntimeError(
                "plot_scene needs both scenes illuminated; call illuminate_real() and "
                "illuminate_reconstructed() first")

        # the 4-panel figure renders at single-column width in the report, where it
        # shrinks ~3x, so the fonts are set large enough to survive that scaling
        with plt.rc_context({"font.size": 19, "axes.titlesize": 20,
                             "axes.labelsize": 19, "xtick.labelsize": 16,
                             "ytick.labelsize": 16, "legend.fontsize": 15}):
            fig = plt.figure(figsize=(10.5, 8), layout="constrained")
            (ax_real, ax_rec), (ax_phase, ax_amp) = fig.subplots(2, 2)

            v_max = np.nanmax([np.nanmax(np.abs(scene.data)),
                               np.nanmax(np.abs(self.gs_rec_data))])
            draw_scene(fig, ax_real, np.abs(scene.data), title="Scene Amplitude",
                       vmax=v_max, **panel)
            # Panel 2 autoscales. v_max above was computed to make the two panels
            # comparable and then commented out at the call; the reconstruction renders
            # dimmer than the real scene, so an independent scale is easier to read.
            # Pass vmax=v_max here instead to put them on one scale.
            draw_scene(fig, ax_rec, np.abs(self.gs_rec_data),
                       title="MGS Reconstruction Amplitude", vmax=None, **panel)

            # both aperture panels are drawn against the full scene x extent
            xlim = (scene.x_min, scene.x_max)
            tx_interp = interp_real_imag(scene.tx_ap.aper_axis, scene.tx_ap.aper_profile,
                                         scene.x_axis)
            gs_amp = interp_amplitude(gs_tx.aper_axis, gs_tx.aper_profile, scene.x_axis)

            phase_series, amp_series = [], []
            if self.has_real_aper:
                phase_series.append(("Real TX", scene.tx_ap.aper_axis,
                                     np.unwrap(np.angle(scene.tx_ap.aper_profile))))
                amp_series.append(("Real TX", scene.x_axis, np.abs(tx_interp)))
            phase_series.append(("MGS Reconstructed TX", gs_tx.aper_axis,
                                 np.unwrap(np.angle(gs_tx.aper_profile))))
            amp_series.append(("MGS Reconstructed TX", scene.x_axis, gs_amp))

            draw_line_panel(ax_phase, phase_series, title="TX Aperture Phase",
                            xlabel="x (m)", ylabel="Phase [rad]", xlim=xlim)
            draw_line_panel(ax_amp, amp_series, title="TX Aperture Amplitude",
                            xlabel="x (m)", ylabel="Amplitude EMF (V/m)", xlim=xlim)

            out_path = save_path if save_path is not None else self.plot_path
            self.log.info(f"Saving scene plot to {out_path}")
            fig.savefig(out_path)
            if show:
                plt.show()
            plt.close(fig)


def resolve_run_frequencies(config: SimConfig, cli_freqs: Optional[List[float]],
                            experimental: bool, log=None) -> List[float]:
    """The frequencies one `mgs` invocation runs at.

    The simulated path is config.resolve_frequencies as it stands: --freq, else
    the config's `frequencies`, else DEFAULT_FREQ_HZ. The experimental path shares
    the first two and NOT the fallback. A capture's carrier is a property of the
    bench, not of the project default, and exp_data reads the carrier out of the
    FFT bin at freq - lo_freq*trx_n: a carrier the rig did not transmit points
    that read at a bin the signal is not in. So with neither --freq nor
    `frequencies`, a capture is read at the carrier the receive chain's multiplied
    LO sits on, experimental.lo_freq * experimental.trx_n (150 GHz for the default
    bench constants, 25 GHz x 6).
    """
    if experimental and not cli_freqs and not config.frequencies:
        exp = config.experimental
        freq = float(exp.lo_freq * exp.trx_n)
        if log is not None:
            log.info(f"Experimental path with no --freq and no config frequencies: "
                     f"using the bench carrier experimental.lo_freq * "
                     f"experimental.trx_n = {freq / 1e9:g} GHz")
        return [freq]
    return resolve_frequencies(config, cli_freqs)


def main():
    parser = argparse.ArgumentParser(
        description="Program to simulate phase retrieval using a modified gerchberg-saxton algorithm"
    )
    parser.add_argument(
        "--debug",
        help="Enables debug logs",
        action="store_true",
        default=False
    )
    parser.add_argument(
        "--freq", "-f",
        help="One or more frequencies in Hz (overrides config `frequencies`). "
             "Multiple values run ONE joint solve for a single phase mask across "
             "all of them. Default: config `frequencies`, else "
             f"{DEFAULT_FREQ_HZ / 1e9:g}e9 (with --rx-path/--heatmap-path: else "
             "experimental.lo_freq * experimental.trx_n).",
        type=float,
        nargs="+",
        default=None
    )
    parser.add_argument(
        "--rx-path",
        type=Path,
        help="Path to .mat with experimentation data. Only a slice of z axis measurements",
        default=None
    )
    parser.add_argument(
        "--heatmap-path",
        type=Path,
        help="Path to .mat experimentation data. Expects full heatmap measurements",
        default=None
    )
    parser.add_argument(
        "--config",
        type=Path,
        help="Path to a simulation config .yml (sim_scene bounds and plot_path)",
        default=DEFAULT_CONFIG
    )
    parser.add_argument(
        "--out", "-o",
        type=Path,
        help="Run directory to write this run into, named outright "
             "(default: <output.output_dir>/<output.run_name>)",
        default=None
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="Override gerchberg_saxton.seed for a reproducible initial phase",
        default=None
    )
    parser.add_argument(
        "--no-save",
        help="Disable run persistence for this invocation",
        action="store_true",
        default=False
    )
    parser.add_argument(
        "--snr-db",
        type=parse_snr_db_arg,
        metavar="DB|off",
        help="Peak-referenced SNR in dB of the complex AWGN added to the simulated RX "
             "measurement (overrides noise.snr_db); 'off' forces a noiseless run",
        default=None
    )
    parser.add_argument(
        "--noise-seed",
        type=int,
        help="Override noise.seed for a reproducible noise draw",
        default=None
    )
    args = parser.parse_args()
    log = setup_logging(args.debug)

    config = load_config(args.config)

    # apply CLI overrides onto the config
    if args.out is not None:
        config.output.output_dir = Path(args.out).parent
        config.output.run_name = Path(args.out).name
    if args.seed is not None: config.gerchberg_saxton.seed = args.seed
    if args.no_save: config.output.save_run = False
    try:
        apply_noise_overrides(config, args.snr_db, args.noise_seed)
    except ValidationError as e:
        parser.error("invalid noise override: " + "; ".join(
            f"{'.'.join(str(p) for p in err['loc'])}: {err['msg']}" for err in e.errors()))

    # Fail fast on a run-directory collision, before spending the solve.
    if config.output.save_run:
        check_run_dir(config.output.output_dir, config.output.run_name, kind="mgs")

    is_exp = args.rx_path is not None and not args.heatmap_path is None
    freqs = resolve_run_frequencies(config, args.freq, experimental=is_exp, log=log)

    if is_exp:
        if len(freqs) != 1:
            parser.error("the experimental path (--rx-path/--heatmap-path) is a "
                         "single-frequency capture; pass exactly one --freq")
        rx_path = Path(args.rx_path)
        heatmap_path = Path(args.heatmap_path)
        mgs = MGS.from_experiment(rx_path, heatmap_path, freqs[0], config)
        mgs.run_gerch_sax()
        mgs.illuminate_reconstructed()
        mgs.plot_scene()
    else:
        mgs = MGS(freqs, config)
        mgs.illuminate_real()
        mgs.run_gerch_sax()
        # Logged HERE, ahead of the re-illumination and the plot: where the energy
        # landed is the answer a single run exists to give, and plot_scene can block
        # on an interactive window (or fail) with the numbers still unprinted.
        mgs.log_landing()
        mgs.illuminate_reconstructed()
        mgs.plot_scene()

    # Deferred to here on purpose: make_run_dir clears the target, so a crashed (or
    # plt.show()-blocked) solve must not have destroyed the previous run already.
    if config.output.save_run:
        run_dir = make_run_dir(config.output.output_dir, config.output.run_name, kind="mgs")
        save_run(mgs, run_dir, config, args.config, vars(args), is_exp=is_exp)

if __name__ == "__main__":
    main()

