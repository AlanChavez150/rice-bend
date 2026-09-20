#!/usr/bin/env python
"""End-to-end parity + timing: python gs_reconstruct vs the rust engine.

The M5 acceptance check (see rust/README.md). Read-only w.r.t. src/rice_bend —
it builds candidate inputs exactly the way the sweep does and calls both
engines side by side. Requires the extension:

    VIRTUAL_ENV=$PWD/.venv .venv/bin/maturin develop --release -m rust/Cargo.toml

Run from the repo root:

    .venv/bin/python rust/python/parity.py

Pass criteria (the settled parity bar — python stays the golden reference):
  - every candidate's final_loss within rtol 1e-3;
  - identical argmin index over the 6-candidate slice;
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
from rice_bend.config import center_freq_index, load_config
from rice_bend.grid_sweep import enumerate_grid
from rice_bend.mgs import MGS, gs_reconstruct

REPO = Path(__file__).resolve().parents[2]
RTOL = 1e-3


def rust_solve(tx_z, amp, x_axis, rx_z, channels, params, ref_freq):
    """Marshal one candidate for rice_bend_core.gs_solve, mirroring what the
    python engine derives internally (mgs.py:194-216): the seeded initial
    phase, the per-channel forward kernels, and the flattened params."""
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


def sweep_setup(cfg_path, freqs):
    """Steps 1-4 of run_grid_search (grid_sweep.py:243-317), without persistence."""
    cfg = load_config(cfg_path)
    m = MGS(freqs, cfg)
    m.measure()
    gs_cfg = cfg.gerchberg_saxton
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


def run_slice(tag, cfg_path, freqs, limit):
    x_axis, rx_z, channels, params, ref_freq, points = sweep_setup(cfg_path, freqs)
    usable = [p for p in points if p.ok][:limit]
    print(f"\n== {tag}: {len(usable)} candidates, F={len(channels)}, "
          f"max_iters={params.max_iters}, seed={params.seed}")

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
                            channels=chs, params=params, ref_freq=ref_freq,
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
    # the joint multi-frequency path on the tiny config (fast, F=2, warm start)
    ok &= run_slice("tiny_check joint", REPO / "configs" / "tiny_check.yml",
                    [140e9, 150e9], limit=3)
    # the characterize check-3 slice: full-resolution scene, F=1
    ok &= run_slice("caustic_hit_sparse limit-6",
                    REPO / "configs" / "scenario_caustic_hit_sparse.yml",
                    [150e9], limit=6)
    print(f"\nPARITY: {'PASS' if ok else 'FAIL'} (rtol {RTOL}, argmin exact)")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
