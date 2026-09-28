#!/usr/bin/env python
"""Speed experiment: python gs_reconstruct vs the rust engine on a real study point.

Single process, single thread — both engines are single-threaded per candidate
and a sweep parallelizes over candidates, so the per-candidate speedup carries
over to sweep wall time directly. Both engines get identical inputs, built the
way the sweep builds them (parity.py's sweep_setup / rust_solve); the rust
timing includes the python-side kernel build and marshaling.

    .venv/bin/python rust/python/bench.py [--candidates 16]

Workloads:
  A  configs/scenario_caustic_hit_pm5.yml — the study's default: 3-tone comb
     (142.5/150/157.5 GHz), warm start, N=2400, max_iters 800, 164x76 grid
  B  the same candidates at 150 GHz alone (the 1-frequency anchor, random init)
"""
import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import sys
import time
import timeit
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from parity import REPO, rust_solve, sweep_setup  # noqa: E402

from rice_bend import rs  # noqa: E402
from rice_bend.mgs import gs_reconstruct  # noqa: E402

PM5 = REPO / "configs" / "scenario_caustic_hit_pm5.yml"


def run_workload(tag, freqs, n_cand):
    x_axis, rx_z, channels, params, ref_freq, points = sweep_setup(PM5, freqs)
    usable = [p for p in points if p.ok]
    picks = [usable[i] for i in np.linspace(0, len(usable) - 1, n_cand).astype(int)]

    def inputs(p):
        amp = np.where((x_axis >= p.x_min) & (x_axis <= p.x_max), 1.0, 0.0)
        ok = p.freq_ok if p.freq_ok is not None else [True] * len(channels)
        return amp, [ch for ch, keep in zip(channels, ok) if keep]

    # warm-up both engines (first-call effects) on a short solve
    amp, chs = inputs(picks[0])
    short = params.model_copy(update={"max_iters": 5})
    gs_reconstruct(tx_z=picks[0].z, orig_aper_amp=amp, x_axis=x_axis, rx_z=rx_z,
                   channels=chs, params=short, ref_freq=ref_freq)
    rust_solve(picks[0].z, amp, x_axis, rx_z, chs, short, ref_freq)

    rows = []
    print(f"\n== {tag}: {len(picks)} of {len(usable)} usable candidates, "
          f"F={len(channels)}, init={params.init}, max_iters={params.max_iters}")
    for p in picks:
        amp, chs = inputs(p)
        t0 = time.perf_counter()
        py = gs_reconstruct(tx_z=p.z, orig_aper_amp=amp, x_axis=x_axis, rx_z=rx_z,
                            channels=chs, params=params, ref_freq=ref_freq)
        t_py = time.perf_counter() - t0
        t0 = time.perf_counter()
        (_a, r_loss, _pf, r_iters, _stop, _lf, _lpf) = rust_solve(
            p.z, amp, x_axis, rx_z, chs, params, ref_freq)
        t_rs = time.perf_counter() - t0
        rel = abs(r_loss - py.final_loss) / abs(py.final_loss)
        rows.append((p, t_py, t_rs, py.final_loss, r_loss, rel, py.n_iters_run, r_iters))
        print(f"  #{p.index:5d} z={p.z:.3f} x={p.x_center:+.3f} F={len(chs)}  "
              f"py {t_py:6.3f}s  rust {t_rs:6.3f}s  ({t_py / t_rs:5.1f}x)  "
              f"iters {py.n_iters_run}/{r_iters}  rel {rel:.1e}")

    t_py = np.array([r[1] for r in rows])
    t_rs = np.array([r[2] for r in rows])
    py_losses = [r[3] for r in rows]
    rs_losses = [r[4] for r in rows]
    iters = np.array([r[6] for r in rows])
    return {
        "tag": tag,
        "n": len(rows),
        "n_grid": len(points),
        "py_mean": t_py.mean(),
        "rs_mean": t_rs.mean(),
        "speedup": t_py.sum() / t_rs.sum(),
        "speedup_min": (t_py / t_rs).min(),
        "speedup_max": (t_py / t_rs).max(),
        "iters_mean": iters.mean(),
        "max_rel": max(r[5] for r in rows),
        "same_iters": sum(r[6] == r[7] for r in rows),
        "argmin_ok": int(np.argmin(py_losses)) == int(np.argmin(rs_losses)),
    }


def primitive_bench():
    n = 2400
    x_axis = np.linspace(-0.3, 0.3, n)
    h = rs.rs_kernel(x_axis, rs.wavelength(150e9), 0.3)
    rng = np.random.default_rng(0)
    u0 = np.where(np.abs(x_axis) <= 0.05, 1.0, 0.0) * np.exp(1j * 2 * np.pi * rng.random(n))
    dx = float(x_axis[1] - x_axis[0])
    reps = 2000
    t = min(timeit.repeat(lambda: rs.rs_apply(u0, h, dx), number=reps, repeat=5)) / reps
    return t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", type=int, default=16)
    args = ap.parse_args()

    results = [
        run_workload("A: pm5 study point (3-tone, warm start)", [142.5e9, 150e9, 157.5e9],
                     args.candidates),
        run_workload("B: 1-frequency anchor (150 GHz)", [150e9], args.candidates),
    ]
    t_prim = primitive_bench()

    print("\n## Results\n")
    print("| Workload | Python s/cand | Rust s/cand | Speedup (range) | "
          "mean iters | max rel Δloss | same n_iters | argmin |")
    print("|---|---|---|---|---|---|---|---|")
    for r in results:
        print(f"| {r['tag']} | {r['py_mean']:.3f} | {r['rs_mean']:.3f} | "
              f"**{r['speedup']:.1f}×** ({r['speedup_min']:.1f}–{r['speedup_max']:.1f}) | "
              f"{r['iters_mean']:.0f} | {r['max_rel']:.1e} | {r['same_iters']}/{r['n']} | "
              f"{'same' if r['argmin_ok'] else 'DIFFERENT'} |")
    print("\n### Projected cost of one full study point (single core)\n")
    print("| Workload | Grid | Python core-h | Rust core-h | 10-point study, 16 cores |")
    print("|---|---|---|---|---|")
    for r in results:
        py_h = r["n_grid"] * r["py_mean"] / 3600
        rs_h = r["n_grid"] * r["rs_mean"] / 3600
        print(f"| {r['tag']} | {r['n_grid']} | {py_h:.2f} | {rs_h:.2f} | "
              f"{10 * py_h / 16 * 60:.0f} min → {10 * rs_h / 16 * 60:.0f} min |")
    print(f"\nPrimitive: scipy fftconvolve rs_apply at N=2400 = {t_prim * 1e6:.1f} µs/call "
          f"(rust rs_apply_into: see `cargo bench`)")


if __name__ == "__main__":
    main()
