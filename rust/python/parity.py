#!/usr/bin/env python
"""End-to-end parity + timing: python gs_reconstruct vs the rust engine.

The M5 acceptance check (see rust/README.md). Read-only w.r.t. src/rice_bend —
it builds candidate inputs exactly the way the sweep does and calls both
engines side by side. Requires the extension:

    VIRTUAL_ENV=$PWD/.venv .venv/bin/maturin develop --release -m rust/Cargo.toml

Run from the repo root:

    .venv/bin/python rust/python/parity.py

Sixteen slices. The first seven: the two noiseless ones (tiny_check joint F=2 at 280/300 GHz; the
characterize caustic_hit_sparse F=1 slice at 300 GHz), the same two with receiver
noise on the measurement (noise.snr_db 20 / 10 dB), one on the ON-GRID receiver
with the LEGACY start (caustic_hit_1f_grid2l, F=1 at 300 GHz,
rx_aperture.grid_stride: 2) under that config's own 30 dB noise block, and two on
the same config and noise with the DATA-DRIVEN start: F=1 at 300 GHz and F=3 at
270/300/330 GHz (±10 %). The noiseless slices force the noise OFF (SNR_OFF, what
the CLI's `--snr-db off` passes): the scenario configs carry a 30 dB noise block
of their own, which a slice would otherwise inherit — as the on-grid slices
deliberately do. Noise is added in python by MGS.measure() before the engine
boundary, so both engines get the identical noisy arrays — the noisy slices check
the solver on a measurement that is no longer an exact field.

The start of every on-grid slice is set IN MEMORY, never taken from the file. The
caustic base config says `init: backprop` (it is a campaign config), so the legacy
slice overrides it to 'warm_start' — at one tone, the seeded random draw, which
crosses the engine boundary as the draw plus a mode. Left to the file it would be a
third backprop slice, and no slice would put a legacy start on a comb-shaped
measurement. The two backprop slices pass 'backprop' explicitly for the same
reason: what a slice tests must not change when a config's start does. The first
four slices take the legacy start from their configs (tiny_check.yml and
scenario_caustic_hit_sparse.yml set no init, so warm_start), which run_slice
prints.

The on-grid slice covers the other shape a measurement can take. The interpolated
receiver (rx_aperture.dx) hands the engines a target and a weighting that are dense
across the RX window; the on-grid one hands them a comb — the element samples at
every grid_stride-th scene sample, the weighting zero at every sample between and
sqrt(grid_stride) higher at the elements. That is built in MGS.measure() too, so the
engine boundary and the bar are the same.

The backprop slices move the engine boundary itself. A legacy start ('random',
'warm_start') crosses it as the seeded random draw plus a mode, and rust runs its
own twin of the warm start. The backprop start is computed in PYTHON for both
engines — the back-propagated phase and, for several tones, the offset scan over
the comb's true period (mgs._backprop_phase) — and rust descends from it as handed
over. So these two slices compare gs_reconstruct on the python engine with
gs_reconstruct on the rust engine, the path a sweep with `init: backprop` takes,
and what they check is the descent from an identical, structured start: one tone,
and three tones with the offset scan in front.

The DESCENDED-AMPLITUDE slices (gerchberg_saxton.amplitude: descend, amp_max set in
memory like the start) move it once more: the start amplitude a0
(mgs._backprop_amplitude) is computed in python and crosses as the amplitude, with
the mode, amp_max and the support (|orig_aper_amp| > 0, which a0 > 0 is not when a0
has zeros in the window). They also go through gs_reconstruct on both engines.
One tone from the backprop start; three tones from the LEGACY warm start, where
rust's own twin runs the reference-tone solve phase-only at a0 on a0's support;
those three tones again with a0 zeroed at a fifth of the window and over one run
(a wrapper around mgs._backprop_amplitude, in memory), which is the case where the
two supports differ; and, when its data file is on this machine, the measured
wide-RX row (configs/measured/measured_rx177_wide_amp.yml, the config as written:
backprop, descend, amp_max 10, 6000 iterations) at six candidates spread over the
grid.

The UNIFORM-LEVEL slices (gerchberg_saxton.amplitude: level, set in memory) cross
the same way: the start level A0 (mgs._power_matched_level, from the backprop
phase) is computed in python and crosses as A0 on the support, with the mode and
the support. One tone and three tones from the backprop start (several tones under
the legacy warm start are refused: level needs init: backprop there), and, when
its data file is on this machine, the measured whole-row level config
(configs/measured/rxsize/rx177_level_w200_coarse.yml as written: backprop, level,
6000 iterations) at six candidates spread over its 4 mm grid.

The EXCESS-POWER PENALTY slices (gerchberg_saxton.power_penalty, gamma 1, set in
memory) cross with two scalars more: gamma, and P_rx (mgs._received_power),
computed in python and handed over, so both engines charge against one number.
The level: one tone from the backprop start on the on-grid config (amplitude level
set in memory too). The descended amplitude: the measured 120 mm RX on row 177
(configs/measured/rxsize/rx177_w120.yml as written: backprop, descend, amp_max 10,
6000 iterations) at four candidates near the beam -- (z, x_center) = (0.242,
-0.020), (0.164, -0.010), (0.080, 0.000), (0.320, -0.014) m, where the ratio sits
above its floor in ~98 % of the 6000 iterations and ends at R 1.19-1.69 -- when
its data file is on this machine. Not candidates spread over the grid, and not the
on-grid config: at six spread candidates of either, the raw back-propagated start
radiates less than P_rx, the ratio never leaves its floor at 1 (0 iterations
above it, measured) and the slice would be the unpenalized one to the bit.
final_loss is then the objective J = L * R, which is what the bar compares.

The RELATIVE STOP RULE slice (gerchberg_saxton.convergence_rel 1e-6, set in
memory; it replaces the absolute convergence_threshold) crosses as one scalar
more. The measured row's simulated twin
(configs/twin/twin_caustic_rx120_level_pp1.yml as written: backprop, level,
power_penalty 1, 6000 iterations, 30 dB noise) at four candidates near its true TX
-- (z, x_center) = (0.242, -0.020), (0.230, -0.020), (0.206, -0.012), (0.280,
-0.016) m: the true cell, two neighbours, and a cell on the valley wall whose loss
is ~50x the true cell's, so the one tolerance is tried at two loss scales.

Pass criteria (the settled parity bar — python stays the golden reference), per slice:
  - every candidate's final_loss within rtol 1e-3;
  - identical argmin index over the slice;
  - n_iters_run / stop_reason reported but NOT gated (ulp drift near the
    convergence threshold legitimately moves the stop point).
"""
import contextlib
import sys
import time
from pathlib import Path

import numpy as np

try:
    import rice_bend_core
except ImportError:
    sys.exit("rice_bend_core is not importable — build it first:\n"
             "  VIRTUAL_ENV=$PWD/.venv .venv/bin/maturin develop --release -m rust/Cargo.toml")

from rice_bend import rs
from rice_bend import mgs as mgs_module
from rice_bend.config import (SNR_OFF, GerchbergSaxtonConfig, apply_noise_overrides,
                              center_freq_index, load_config, resolve_frequencies)
from rice_bend.grid_sweep import enumerate_grid
from rice_bend.measured import apply_measured
from rice_bend.mgs import MGS, gs_reconstruct

REPO = Path(__file__).resolve().parents[2]
RTOL = 1e-3


def rust_solve(tx_z, amp, x_axis, rx_z, channels, params, ref_freq):
    """One candidate on the rust engine, as rice_bend_core.gs_solve's result tuple.

    For the legacy starts (init 'random' / 'warm_start') this marshals the
    candidate for gs_solve by hand, mirroring what the python engine derives
    internally (gs_reconstruct's shared setup): the seeded initial phase, the
    per-channel forward kernels, and the flattened params — an independent second
    copy of the marshaling, so a slip in mgs._gs_solve_rust shows up here.

    `init: backprop` is not a mode rust knows. That start is built in python and
    handed to rust as the phase to descend from, and the code that builds it lives
    in mgs.py alone; re-deriving it here would be a third copy that proves nothing
    about the first. So for backprop this IS gs_reconstruct with the engine forced
    to rust — the path a sweep takes — repacked into gs_solve's tuple shape, which
    keeps the signature bench.py and run_slice rely on. `amplitude: descend` and
    `level` take the same path for the same reason: their start amplitude (a0, or
    the level A0) is built in mgs.py; so does the excess-power penalty, which
    needs one of them. gs_solve's tuple has grown two (final_mse, power_ratio)
    since; the hand-marshaled path is cut back to the seven both callers read.
    The relative stop rule (convergence_rel) takes it too: the hand-marshaled call
    predates it and would run the absolute rule."""
    if (params.init == "backprop" or params.amplitude != "fixed"
            or params.convergence_rel is not None):
        r = gs_reconstruct(tx_z=tx_z, orig_aper_amp=amp, x_axis=x_axis, rx_z=rx_z,
                           channels=channels,
                           params=params.model_copy(update={"engine": "rust"}),
                           ref_freq=ref_freq, capture=False, log=None)
        return (r.curr_aper_f, r.final_loss, r.final_loss_per_freq, r.n_iters_run,
                r.stop_reason, r.loss_full, r.loss_full_per_freq)
    dx = float(x_axis[1] - x_axis[0])
    assert params.seed is not None, "parity requires a pinned seed"
    initial_phase = 2 * np.pi * np.random.default_rng(int(params.seed)).random(len(x_axis))
    freqs = np.array([ch.freq for ch in channels], dtype=np.float64)
    rx = np.stack([np.asarray(ch.rx_field, dtype=np.complex128) for ch in channels])
    w = np.stack([np.asarray(ch.error_weighting, dtype=np.float64) for ch in channels])
    h_fwd = np.stack([rs.rs_kernel(x_axis, ch.wavelength, -1.0 * (rx_z - tx_z))
                      for ch in channels]).astype(np.complex128)
    return tuple(rice_bend_core.gs_solve(
        initial_phase, np.abs(amp).astype(np.float64), freqs, float(ref_freq),
        rx, w, h_fwd, dx, params.init, int(params.max_iters),
        int(params.convergence_count), float(params.convergence_threshold),
        float(params.lr0), float(params.bt_shrink), int(params.bt_tries)))[:7]


def sweep_setup(cfg_path, freqs, *, snr_db=None, noise_seed=0, init=None,
                amplitude=None, amp_max=None, power_penalty=None, convergence_rel=None):
    """Steps 1-4 of grid_sweep.run_grid_search (the measurement, the grid-only GS
    overrides, the enumeration, the shared payload), without persistence: a slice
    calls the solver directly, one solve per candidate.
    snr_db follows the CLI's --snr-db: a number of dB, SNR_OFF to force a noiseless
    measurement, or None to keep the config's noise block — which is 30 dB in every
    scenario config (tiny_check.yml has none), so a slice meant to be noiseless must
    pass SNR_OFF. `init` overrides gerchberg_saxton.init in memory (None keeps the
    config's): the start is a property of the solve, not of the measurement, so a
    slice can put another start on a shipped config without a copy of the file.
    `amplitude` / `amp_max` override gerchberg_saxton's the same way (both or
    neither), and `power_penalty` and `convergence_rel` likewise (None keeps the
    config's). A measured config (a `measured` block) is built the way
    grid-search-mgs builds it — apply_measured, then MGS.from_measured — with
    `freqs` None for the config's own; it takes no noise override."""
    cfg = load_config(cfg_path)
    if cfg.measured is not None:
        freqs = resolve_frequencies(cfg, freqs)
        apply_measured(cfg, freqs)
        m = MGS.from_measured(freqs, cfg)
    else:
        apply_noise_overrides(cfg, snr_db, noise_seed)
        m = MGS(freqs, cfg)
    m.measure()
    gs_cfg = cfg.gerchberg_saxton
    update = {}
    if init is not None:
        update["init"] = init
    if amplitude is not None:
        update.update(amplitude=amplitude, amp_max=amp_max)
    if power_penalty is not None:
        update["power_penalty"] = power_penalty
    if convergence_rel is not None:
        update["convergence_rel"] = convergence_rel
    if update:
        # rebuilt through the constructor so the mode is validated (attribute
        # assignment is not: the models have no validate_assignment)
        gs_cfg = GerchbergSaxtonConfig(**{**gs_cfg.model_dump(), **update})
    if cfg.grid_search.gs_overrides.max_iters is not None:
        gs_cfg.max_iters = cfg.grid_search.gs_overrides.max_iters
    if cfg.grid_search.seed is not None:
        gs_cfg.seed = cfg.grid_search.seed
    wavelengths = [fs.wavelength for fs in m.freq_states]
    points = enumerate_grid(cfg.grid_search, cfg.sim_scene, wavelengths)
    x_axis = np.asarray(m.scene.x_axis).copy()
    rx_z = float(m.scene.rx_ap.z)
    ref_freq = float(m.freqs[center_freq_index(m.freqs)])
    return x_axis, rx_z, m.measurement_channels(), gs_cfg.model_copy(), ref_freq, points


def run_slice(tag, cfg_path, freqs, limit, *, snr_db=None, noise_seed=0, init=None,
              amplitude=None, amp_max=None, power_penalty=None, convergence_rel=None,
              stride=1, spread=None, near=None):
    """`stride` spreads the candidates: every stride-th usable point, `limit` of them.
    `spread` (instead) picks `spread` candidates evenly over all the usable ones.
    `near` (instead of both) is a list of (z, x_center) in m: for each, the usable
    candidate nearest to it, for a slice that must sit where something happens."""
    x_axis, rx_z, channels, params, ref_freq, points = sweep_setup(
        cfg_path, freqs, snr_db=snr_db, noise_seed=noise_seed, init=init,
        amplitude=amplitude, amp_max=amp_max, power_penalty=power_penalty,
        convergence_rel=convergence_rel)
    # the reference side is the PYTHON engine whatever the config's `engine` says:
    # a config that named rust would otherwise be compared with itself
    py_params = params.model_copy(update={"engine": "python"})
    usable = [p for p in points if p.ok]
    if spread is not None:
        stride = max(1, len(usable) // spread)
    if near is not None:
        usable = [min(usable, key=lambda p, z=z, xc=xc: (p.z - z) ** 2 + (p.x_center - xc) ** 2)
                  for z, xc in near]
    else:
        usable = usable[::stride][:limit]
    if snr_db == SNR_OFF:
        noise = "off"
    elif snr_db is None:
        noise = "as configured"
    else:
        noise = f"SNR {snr_db:g} dB seed {noise_seed}"
    amp_note = ("" if params.amplitude == "fixed" else
                f", amplitude {params.amplitude}" + (
                    "" if params.amp_max is None else f" (amp_max {params.amp_max:g})"))
    if params.power_penalty is not None:
        amp_note += f", power_penalty {params.power_penalty:g}"
    if params.convergence_rel is not None:
        amp_note += f", convergence_rel {params.convergence_rel:g}"
    print(f"\n== {tag}: {len(usable)} candidates, F={len(channels)}, "
          f"init={params.init}, max_iters={params.max_iters}, seed={params.seed}, "
          f"noise {noise}{amp_note}")

    py_losses, rust_losses = [], []
    t_py = t_rust = 0.0
    all_ok = True
    for p in usable:
        support = (x_axis >= p.x_min) & (x_axis <= p.x_max)
        amp = np.where(support, 1.0, 0.0)
        freq_ok = p.freq_ok if p.freq_ok is not None else [True] * len(channels)
        chs = [ch for ch, ok in zip(channels, freq_ok) if ok]

        t0 = time.perf_counter()
        py = gs_reconstruct(tx_z=p.z, orig_aper_amp=amp, x_axis=x_axis, rx_z=rx_z,
                            channels=chs, params=py_params, ref_freq=ref_freq,
                            capture=False, log=None)
        t_py += time.perf_counter() - t0

        t0 = time.perf_counter()
        (_aper, r_loss, _pf, r_iters, r_stop, _lf, _lpf) = rust_solve(
            p.z, amp, x_axis, rx_z, chs, params, ref_freq)
        t_rust += time.perf_counter() - t0

        rel = abs(r_loss - py.final_loss) / max(abs(py.final_loss), 1e-300)
        ok = rel <= RTOL
        all_ok &= ok
        py_losses.append(py.final_loss)
        rust_losses.append(r_loss)
        print(f"  #{p.index:3d} z={p.z:6.3f} x={p.x_center:+6.3f}  "
              f"loss py {py.final_loss:.6e} rust {r_loss:.6e} rel {rel:.2e} "
              f"[{'ok' if ok else 'FAIL'}]  iters {py.n_iters_run}/{r_iters} "
              f"stop {py.stop_reason}/{r_stop}")

    if len(usable) > 1:
        a_py, a_rust = int(np.argmin(py_losses)), int(np.argmin(rust_losses))
        argmin_ok = a_py == a_rust
        all_ok &= argmin_ok
        print(f"  argmin: py #{usable[a_py].index} rust #{usable[a_rust].index} "
              f"[{'ok' if argmin_ok else 'FAIL'}]")
    print(f"  time: python {t_py:.2f}s, rust {t_rust:.2f}s "
          f"({t_py / max(t_rust, 1e-9):.1f}x)")
    return all_ok


@contextlib.contextmanager
def _holed_start_amplitude():
    """mgs._backprop_amplitude with zeros inside the window (every 5th support
    sample, and a run of 25 a third of the way in), for both engines alike: the
    start amplitude is built in python and handed to rust."""
    real = mgs_module._backprop_amplitude

    def holed(*args, **kwargs):
        amp = real(*args, **kwargs)
        idx = np.where(kwargs["support"])[0]
        amp[idx[::5]] = 0.0
        amp[idx[len(idx) // 3: len(idx) // 3 + 25]] = 0.0
        return amp

    mgs_module._backprop_amplitude = holed
    try:
        yield
    finally:
        mgs_module._backprop_amplitude = real


def main():
    ok = True
    # the joint multi-frequency path on the tiny config (fast, F=2, warm start).
    # NOISELESS, explicitly: tiny_check.yml has no noise block today, but the slice
    # must not start depending on that
    ok &= run_slice("tiny_check joint", REPO / "configs" / "tiny_check.yml",
                    [280e9, 300e9], limit=3, snr_db=SNR_OFF)
    # the characterize check-3 geometry (full-resolution scene, F=1), NOISELESS:
    # SNR_OFF overrides the config's 30 dB, which check 3 itself runs with
    ok &= run_slice("caustic_hit_sparse limit-6",
                    REPO / "configs" / "scenario_caustic_hit_sparse.yml",
                    [300e9], limit=6, snr_db=SNR_OFF)
    # the same two slices on a NOISY measurement: identical arrays reach both
    # engines, so the bar does not move
    ok &= run_slice("tiny_check joint, 20 dB", REPO / "configs" / "tiny_check.yml",
                    [280e9, 300e9], limit=3, snr_db=20.0, noise_seed=0)
    ok &= run_slice("caustic_hit_sparse limit-6, 10 dB",
                    REPO / "configs" / "scenario_caustic_hit_sparse.yml",
                    [300e9], limit=6, snr_db=10.0, noise_seed=0)
    # the ON-GRID receiver (rx_aperture.grid_stride: 2): the target and the weighting
    # are non-zero at the 80 element samples only, not across the whole RX window.
    # snr_db=None keeps the config's OWN noise block (30 dB, seed 0), so this is the
    # measurement the campaign base config actually solves against. The LEGACY start,
    # forced in memory: the config itself says backprop (see the module docstring).
    # At one tone warm_start is the seeded random draw, hand-marshaled by rust_solve
    ok &= run_slice("caustic_hit_1f_grid2l limit-6, on-grid RX, legacy start",
                    REPO / "configs" / "scenario_caustic_hit_1f_grid2l.yml",
                    [300e9], limit=6, init="warm_start")
    # the DATA-DRIVEN start (init: backprop, passed explicitly) on that same config
    # and noise. The start is built in python for both engines and rust descends
    # from it as handed over, so these check the descent from one identical
    # structured start: a single tone first (the back-propagated phase alone)...
    ok &= run_slice("caustic_hit_1f_grid2l limit-6, backprop start, 1 tone",
                    REPO / "configs" / "scenario_caustic_hit_1f_grid2l.yml",
                    [300e9], limit=6, init="backprop")
    # ...then three tones at ±10 %, where the absolute-offset scan over the comb's
    # true period (10 x 2π here) runs in front of the joint descent
    ok &= run_slice("caustic_hit_1f_grid2l limit-6, backprop start, 3 tones ±10 %",
                    REPO / "configs" / "scenario_caustic_hit_1f_grid2l.yml",
                    [270e9, 300e9, 330e9], limit=6, init="backprop")
    # the DESCENDED amplitude (amplitude: descend, set in memory with its cap): one
    # tone from the backprop start...
    ok &= run_slice("caustic_hit_1f_grid2l limit-6, descend, backprop start, 1 tone",
                    REPO / "configs" / "scenario_caustic_hit_1f_grid2l.yml",
                    [300e9], limit=6, init="backprop", amplitude="descend", amp_max=2.0)
    # ...three tones from the legacy warm start: rust's own twin, its reference-tone
    # solve phase-only at the start amplitude a0
    ok &= run_slice("caustic_hit_1f_grid2l limit-6, descend, warm start, 3 tones ±10 %",
                    REPO / "configs" / "scenario_caustic_hit_1f_grid2l.yml",
                    [270e9, 300e9, 330e9], limit=6, init="warm_start",
                    amplitude="descend", amp_max=2.0)
    # ...and again with ZEROS in a0 inside the window, where the solve's support
    # (the window) and a0 > 0 part: the descent must still move those samples, and
    # the warm start's reference solve runs on a0 > 0 (its python child's support)
    with _holed_start_amplitude():
        ok &= run_slice("caustic_hit_1f_grid2l limit-6, descend, warm start, 3 tones, "
                        "a0 zero in the window",
                        REPO / "configs" / "scenario_caustic_hit_1f_grid2l.yml",
                        [270e9, 300e9, 330e9], limit=6, init="warm_start",
                        amplitude="descend", amp_max=2.0)
    # the measured wide-RX row as configured (backprop, descend, amp_max 10, 6000
    # iterations), six candidates spread over the 30 200 usable ones — when this
    # machine has its data file
    measured_cfg = REPO / "configs" / "measured" / "measured_rx177_wide_amp.yml"
    data_file = Path(load_config(measured_cfg).measured.data_file).expanduser()
    if data_file.exists():
        ok &= run_slice("measured_rx177_wide_amp, descend as configured, 6 spread",
                        measured_cfg, None, limit=6, stride=5033)
    else:
        print(f"\n== measured_rx177_wide_amp: SKIPPED ({data_file} not found)")
    # the UNIFORM LEVEL (amplitude: level, no cap): one tone and three tones from the
    # backprop start, the start level A0 matched to it in python
    ok &= run_slice("caustic_hit_1f_grid2l, level, backprop start, 1 tone, 6 spread",
                    REPO / "configs" / "scenario_caustic_hit_1f_grid2l.yml",
                    [300e9], limit=6, spread=6, init="backprop", amplitude="level")
    ok &= run_slice("caustic_hit_1f_grid2l, level, backprop start, 3 tones ±10 %, 6 spread",
                    REPO / "configs" / "scenario_caustic_hit_1f_grid2l.yml",
                    [270e9, 300e9, 330e9], limit=6, spread=6, init="backprop",
                    amplitude="level")
    # the measured whole-row level config as written (backprop, level, 6000 iterations)
    level_cfg = REPO / "configs" / "measured" / "rxsize" / "rx177_level_w200_coarse.yml"
    data_file = Path(load_config(level_cfg).measured.data_file).expanduser()
    if data_file.exists():
        ok &= run_slice("rx177_level_w200_coarse, level as configured, 6 spread",
                        level_cfg, None, limit=6, spread=6)
    else:
        print(f"\n== rx177_level_w200_coarse: SKIPPED ({data_file} not found)")
    # the EXCESS-POWER PENALTY (power_penalty, gamma 1, set in memory): P_rx is
    # computed in python and handed to rust, so both engines charge against one
    # number. One tone from the backprop start, under the level...
    ok &= run_slice("caustic_hit_1f_grid2l, level, power_penalty 1, backprop start, "
                    "1 tone, 6 spread",
                    REPO / "configs" / "scenario_caustic_hit_1f_grid2l.yml",
                    [300e9], limit=6, spread=6, init="backprop", amplitude="level",
                    power_penalty=1.0)
    # ...and under the capped per-sample amplitude, on the measured 120 mm RX of row
    # 177 as configured (backprop, descend, amp_max 10, 6000 iterations) at four
    # candidates near the beam, where the ratio stays above its floor (at spread
    # candidates, and on the on-grid config, it never leaves it) -- when this
    # machine has its data file
    penalty_cfg = REPO / "configs" / "measured" / "rxsize" / "rx177_w120.yml"
    data_file = Path(load_config(penalty_cfg).measured.data_file).expanduser()
    if data_file.exists():
        ok &= run_slice("rx177_w120, descend as configured, power_penalty 1, 4 near the beam",
                        penalty_cfg, None, limit=4, power_penalty=1.0,
                        near=[(0.242, -0.020), (0.164, -0.010), (0.080, 0.000),
                              (0.320, -0.014)])
    else:
        print(f"\n== rx177_w120 power_penalty: SKIPPED ({data_file} not found)")
    # the RELATIVE STOP RULE (convergence_rel 1e-6, set in memory), on the measured
    # row's simulated twin as configured (backprop, level, power_penalty 1, 6000
    # iterations): the true cell, two neighbours and a valley-wall cell (~50x the
    # loss), the one tolerance at two loss scales
    ok &= run_slice("twin_caustic_rx120_level_pp1, level + power_penalty as configured, "
                    "convergence_rel 1e-6, 4 near the true TX",
                    REPO / "configs" / "twin" / "twin_caustic_rx120_level_pp1.yml",
                    [300e9], limit=4, convergence_rel=1e-6,
                    near=[(0.242, -0.020), (0.230, -0.020), (0.206, -0.012),
                          (0.280, -0.016)])
    print(f"\nPARITY: {'PASS' if ok else 'FAIL'} (rtol {RTOL}, argmin exact)")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
