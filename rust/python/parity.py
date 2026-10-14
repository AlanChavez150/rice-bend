#!/usr/bin/env python
"""End-to-end parity + timing: python gs_reconstruct vs the rust engine.

The M5 acceptance check (see rust/README.md). Read-only w.r.t. src/rice_bend —
it builds candidate inputs exactly the way the sweep does and calls both
engines side by side. Requires the extension:

    VIRTUAL_ENV=$PWD/.venv .venv/bin/maturin develop --release -m rust/Cargo.toml

Run from the repo root:

    .venv/bin/python rust/python/parity.py

Seven slices: the two noiseless ones (tiny_check joint F=2 at 280/300 GHz; the
characterize caustic_hit_sparse F=1 slice at 300 GHz), the same two with receiver
noise on the measurement (noise.snr_db 20 / 10 dB), one on the ON-GRID receiver
(caustic_hit_1f_grid2l, F=1 at 300 GHz, rx_aperture.grid_stride: 2) under that
config's own 30 dB noise block, and two on the same config and noise with the
DATA-DRIVEN start (gerchberg_saxton.init overridden to 'backprop' in memory): F=1
at 300 GHz and F=3 at 270/300/330 GHz (±10 %). The noiseless slices force the
noise OFF (SNR_OFF, what the CLI's `--snr-db off` passes): the scenario configs
carry a 30 dB noise block of their own, which a slice would otherwise inherit — as
the on-grid slices deliberately do. Noise is added in python by MGS.measure()
before the engine boundary, so both engines get the identical noisy arrays — the
noisy slices check the solver on a measurement that is no longer an exact field.

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

Pass criteria (the settled parity bar — python stays the golden reference), per slice:
  - every candidate's final_loss within rtol 1e-3;
  - identical argmin index over the slice;
  - n_iters_run / stop_reason reported but NOT gated (ulp drift near the
    convergence threshold legitimately moves the stop point).
"""
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
from rice_bend.config import (SNR_OFF, GerchbergSaxtonConfig, apply_noise_overrides,
                              center_freq_index, load_config)
from rice_bend.grid_sweep import enumerate_grid
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
    keeps the signature bench.py and run_slice rely on."""
    if params.init == "backprop":
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
    return rice_bend_core.gs_solve(
        initial_phase, np.abs(amp).astype(np.float64), freqs, float(ref_freq),
        rx, w, h_fwd, dx, params.init, int(params.max_iters),
        int(params.convergence_count), float(params.convergence_threshold),
        float(params.lr0), float(params.bt_shrink), int(params.bt_tries))


def sweep_setup(cfg_path, freqs, *, snr_db=None, noise_seed=0, init=None):
    """Steps 1-4 of run_grid_search (grid_sweep.py:243-317), without persistence.
    snr_db follows the CLI's --snr-db: a number of dB, SNR_OFF to force a noiseless
    measurement, or None to keep the config's noise block — which is 30 dB in every
    scenario config (tiny_check.yml has none), so a slice meant to be noiseless must
    pass SNR_OFF. `init` overrides gerchberg_saxton.init in memory (None keeps the
    config's): the start is a property of the solve, not of the measurement, so a
    slice can put another start on a shipped config without a copy of the file."""
    cfg = load_config(cfg_path)
    apply_noise_overrides(cfg, snr_db, noise_seed)
    m = MGS(freqs, cfg)
    m.measure()
    gs_cfg = cfg.gerchberg_saxton
    if init is not None:
        # rebuilt through the constructor so the mode is validated (attribute
        # assignment is not: the models have no validate_assignment)
        gs_cfg = GerchbergSaxtonConfig(**{**gs_cfg.model_dump(), "init": init})
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


def run_slice(tag, cfg_path, freqs, limit, *, snr_db=None, noise_seed=0, init=None):
    x_axis, rx_z, channels, params, ref_freq, points = sweep_setup(
        cfg_path, freqs, snr_db=snr_db, noise_seed=noise_seed, init=init)
    # the reference side is the PYTHON engine whatever the config's `engine` says:
    # a config that named rust would otherwise be compared with itself
    py_params = params.model_copy(update={"engine": "python"})
    usable = [p for p in points if p.ok][:limit]
    if snr_db == SNR_OFF:
        noise = "off"
    elif snr_db is None:
        noise = "as configured"
    else:
        noise = f"SNR {snr_db:g} dB seed {noise_seed}"
    print(f"\n== {tag}: {len(usable)} candidates, F={len(channels)}, "
          f"init={params.init}, max_iters={params.max_iters}, seed={params.seed}, "
          f"noise {noise}")

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
    # measurement the campaign base config actually solves against
    ok &= run_slice("caustic_hit_1f_grid2l limit-6, on-grid RX",
                    REPO / "configs" / "scenario_caustic_hit_1f_grid2l.yml",
                    [300e9], limit=6)
    # the DATA-DRIVEN start (init: backprop) on that same config and noise. The
    # start is built in python for both engines and rust descends from it as handed
    # over, so these check the descent from one identical structured start: a
    # single tone first (the back-propagated phase alone)...
    ok &= run_slice("caustic_hit_1f_grid2l limit-6, backprop start, 1 tone",
                    REPO / "configs" / "scenario_caustic_hit_1f_grid2l.yml",
                    [300e9], limit=6, init="backprop")
    # ...then three tones at ±10 %, where the absolute-offset scan over the comb's
    # true period (10 x 2π here) runs in front of the joint descent
    ok &= run_slice("caustic_hit_1f_grid2l limit-6, backprop start, 3 tones ±10 %",
                    REPO / "configs" / "scenario_caustic_hit_1f_grid2l.yml",
                    [270e9, 300e9, 330e9], limit=6, init="backprop")
    print(f"\nPARITY: {'PASS' if ok else 'FAIL'} (rtol {RTOL}, argmin exact)")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
