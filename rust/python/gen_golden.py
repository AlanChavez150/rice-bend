#!/usr/bin/env python
"""Generate the golden test vectors under rust/tests/data/.

Read-only with respect to src/rice_bend (imports only); rerunnable. The Python
engine is the reference — these vectors ARE its behaviour, captured. Regenerate
after any (approved, separately-validated) change to the reference solver, and
note the commit recorded in tests/data/meta.json.

    .venv/bin/python rust/python/gen_golden.py

Cases produced (names are pinned by rust/tests/conv_parity.rs / solve_parity.rs):
  conv_n{7,8,2399,2401}   random complex pairs — odd/even "same"-slice centering
  conv_n2400              a real RS kernel + box-aperture field at the flagship size
  solve_single            one F=1 candidate solve, tiny_check geometry (N=400)
  solve_joint             the same candidate at F=2 with the full warm start,
                          including the stage-2/3 intermediates (psi0, delta)
"""
import json
import subprocess
import sys
from datetime import date
from pathlib import Path

import numpy as np
import scipy.fft
import scipy.signal

from rice_bend import rs
from rice_bend.config import center_freq_index, load_config
from rice_bend.grid_sweep import enumerate_grid
from rice_bend.mgs import (MGS, WARM_START_MAX_SAMPLES, WARM_START_SCAN_STEP,
                           gs_reconstruct)

REPO = Path(__file__).resolve().parents[2]
DATA = Path(__file__).resolve().parents[1] / "tests" / "data"

_DTYPE_TAGS = {
    np.dtype(np.complex128): "c128",
    np.dtype(np.complex64): "c64",
    np.dtype(np.float64): "f64",
    np.dtype(np.float32): "f32",
}


def save_case(name, arrays, scalars):
    d = DATA / name
    d.mkdir(parents=True, exist_ok=True)
    manifest = {"arrays": {}, "scalars": scalars}
    for key, arr in arrays.items():
        arr = np.ascontiguousarray(arr)
        tag = _DTYPE_TAGS[arr.dtype]
        fname = f"{key}.bin"
        (d / fname).write_bytes(arr.tobytes())
        manifest["arrays"][key] = {"file": fname, "dtype": tag, "shape": list(arr.shape)}
    (d / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
    n_bytes = sum((d / v["file"]).stat().st_size for v in manifest["arrays"].values())
    print(f"  {name}: {len(arrays)} arrays, {n_bytes / 1024:.0f} KiB")


def conv_case(name, u0, h, dx):
    u0 = np.asarray(u0, dtype=np.complex128)
    h = np.asarray(h, dtype=np.complex128)
    n = len(u0)
    # what scipy.signal.fftconvolve pads to for complex inputs (11-smooth)
    nfft = scipy.fft.next_fast_len(2 * n - 1, real=False)
    expected_f64 = scipy.signal.fftconvolve(u0, h, mode="same") * dx
    expected_c64 = rs.rs_apply(u0, h, dx)
    assert expected_c64.dtype == np.complex64
    save_case(name,
              {"u0": u0, "h": h,
               "expected_f64": expected_f64.astype(np.complex128),
               "expected_c64": expected_c64},
              {"n": n, "nfft": int(nfft), "dx": float(dx)})


def gen_conv_cases():
    for n in (7, 8, 2399, 2401):
        rng = np.random.default_rng(1000 + n)
        u0 = rng.standard_normal(n) + 1j * rng.standard_normal(n)
        h = rng.standard_normal(n) + 1j * rng.standard_normal(n)
        conv_case(f"conv_n{n}", u0, h, 0.00025)

    # the flagship size, with physically-shaped data: a real forward RS kernel
    # (TX 0.3 m above the RX plane at 150 GHz) and a box aperture random-phase field
    n = 2400
    x_axis = np.linspace(-0.3, 0.3, n)
    h = rs.rs_kernel(x_axis, rs.wavelength(150e9), 0.3)
    rng = np.random.default_rng(2400)
    amp = np.where(np.abs(x_axis) <= 0.05, 1.0, 0.0)
    u0 = amp * np.exp(1j * 2 * np.pi * rng.random(n))
    conv_case("conv_n2400", u0, h, float(x_axis[1] - x_axis[0]))


def _candidate_setup(freqs):
    """The sweep's own construction of one usable candidate on tiny_check:
    MGS + measure + run_grid_search's gs_overrides/seed mutation
    (grid_sweep.py:280-287) + enumerate_grid to pick a point valid at every
    frequency + _reconstruct_candidate's box amplitude (grid_sweep.py:137-143)."""
    cfg = load_config(REPO / "configs" / "tiny_check.yml")
    m = MGS(freqs, cfg)
    m.measure()

    gs_cfg = cfg.gerchberg_saxton
    if cfg.grid_search.gs_overrides.max_iters is not None:
        gs_cfg.max_iters = cfg.grid_search.gs_overrides.max_iters
    if cfg.grid_search.seed is not None:
        gs_cfg.seed = cfg.grid_search.seed
    params = gs_cfg.model_copy()

    wavelengths = [fs.wavelength for fs in m.freq_states]
    points = enumerate_grid(cfg.grid_search, cfg.sim_scene, wavelengths)
    point = next(p for p in points if p.ok and all(p.freq_ok))

    x_axis = np.asarray(m.scene.x_axis).copy()
    support = (x_axis >= point.x_min) & (x_axis <= point.x_max)
    amp = np.where(support, 1.0, 0.0)
    channels = m.measurement_channels()
    ref_freq = float(m.freqs[center_freq_index(m.freqs)])
    rx_z = float(m.scene.rx_ap.z)
    return x_axis, point, support, amp, channels, params, ref_freq, rx_z


def _solve_inputs(x_axis, tx_z, rx_z, amp, channels, params, ref_freq):
    """The arrays that cross the Rust boundary, built exactly as mgs.py does."""
    dx = float(x_axis[1] - x_axis[0])
    seed = int(params.seed)
    initial_phase = 2 * np.pi * np.random.default_rng(seed).random(len(x_axis))
    h_fwd = np.stack([rs.rs_kernel(x_axis, ch.wavelength, -1.0 * (rx_z - tx_z))
                      for ch in channels])
    arrays = {
        "initial_phase": initial_phase,
        "aper_amp": np.abs(amp).astype(np.float64),
        "freqs": np.array([ch.freq for ch in channels], dtype=np.float64),
        "rx_field": np.stack([np.asarray(ch.rx_field, dtype=np.complex128)
                              for ch in channels]),
        "error_weighting": np.stack([np.asarray(ch.error_weighting, dtype=np.float64)
                                     for ch in channels]),
        "h_fwd": h_fwd.astype(np.complex128),
    }
    scalars = {
        "ref_freq": ref_freq, "dx": dx, "seed": seed, "tx_z": float(tx_z),
        "rx_z": float(rx_z), "init_mode": params.init,
        "max_iters": int(params.max_iters),
        "convergence_count": int(params.convergence_count),
        "convergence_threshold": float(params.convergence_threshold),
        "lr0": float(params.lr0), "bt_shrink": float(params.bt_shrink),
        "bt_tries": int(params.bt_tries),
    }
    return arrays, scalars


def _result_outputs(result):
    arrays = {
        "curr_aper_f": np.asarray(result.curr_aper_f, dtype=np.complex128),
        "final_loss_per_freq": np.asarray(result.final_loss_per_freq, dtype=np.float64),
        "loss_full": np.asarray(result.loss_full, dtype=np.float32),
        "loss_full_per_freq": np.asarray(result.loss_full_per_freq, dtype=np.float32),
    }
    scalars = {
        "final_loss": float(result.final_loss),
        "n_iters_run": int(result.n_iters_run),
        "stop_reason": str(result.stop_reason),
    }
    return arrays, scalars


def gen_solve_single():
    x_axis, point, _support, amp, channels, params, _ref, rx_z = _candidate_setup([150e9])
    ref_freq = 150e9
    result = gs_reconstruct(tx_z=point.z, orig_aper_amp=amp, x_axis=x_axis, rx_z=rx_z,
                            channels=channels, params=params, ref_freq=ref_freq,
                            capture=False, log=None)
    arrays, scalars = _solve_inputs(x_axis, point.z, rx_z, amp, channels, params, ref_freq)
    out_a, out_s = _result_outputs(result)
    arrays.update(out_a)
    scalars.update(out_s)
    save_case("solve_single", arrays, scalars)


def gen_solve_joint():
    freqs = [140e9, 150e9]
    x_axis, point, support, amp, channels, params, ref_freq, rx_z = _candidate_setup(freqs)
    assert params.init == "warm_start", "solve_joint must exercise the warm start"

    result = gs_reconstruct(tx_z=point.z, orig_aper_amp=amp, x_axis=x_axis, rx_z=rx_z,
                            channels=channels, params=params, ref_freq=ref_freq,
                            capture=False, log=None)

    # --- warm-start intermediates, mirroring _warm_start_phase (mgs.py:37-116)
    # op for op. gs_reconstruct does not expose them, so they are recomputed here
    # with the same primitives; keep this block in sync with mgs.py.
    dx = float(x_axis[1] - x_axis[0])
    seed = int(params.seed)
    rho = np.array([ch.freq / ref_freq for ch in channels], dtype=np.float64)
    curr_aper_amp = np.abs(amp.copy())
    c_idx = int(np.argmin(np.abs(rho - 1.0)))
    ch_ref = channels[c_idx]
    sub_params = params.model_copy(update={"init": "random", "seed": seed})
    stage1 = gs_reconstruct(tx_z=point.z, orig_aper_amp=amp, x_axis=x_axis, rx_z=rx_z,
                            channels=[ch_ref], params=sub_params, ref_freq=ch_ref.freq,
                            capture=False, log=None)
    psi0 = np.zeros(len(x_axis))
    idx = np.where(support)[0]
    psi0[idx] = np.unwrap(np.angle(stage1.curr_aper_f[idx])) / rho[c_idx]

    sorted_freqs = np.sort(np.array([ch.freq for ch in channels]))
    gaps = np.diff(sorted_freqs)
    gaps = gaps[gaps > 0]
    period = 2 * np.pi * ref_freq / gaps.min() if gaps.size else 2 * np.pi
    n_samples = int(np.ceil(period / WARM_START_SCAN_STEP))
    n_samples = min(n_samples, WARM_START_MAX_SAMPLES)
    deltas = np.linspace(0.0, period, n_samples, endpoint=False)

    h_fwd = [rs.rs_kernel(x_axis, ch.wavelength, -1.0 * (rx_z - point.z))
             for ch in channels]
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
    warm_psi = psi0 + best_delta
    warm_psi[~support] = 0.0

    arrays, scalars = _solve_inputs(x_axis, point.z, rx_z, amp, channels, params, ref_freq)
    out_a, out_s = _result_outputs(result)
    arrays.update(out_a)
    scalars.update(out_s)
    arrays["psi0"] = psi0
    arrays["warm_psi"] = warm_psi
    scalars.update({
        "warm_best_delta": float(best_delta),
        "warm_best_loss": float(best_loss),
        "warm_period": float(period),
        "warm_n_samples": int(n_samples),
        "warm_ref_channel": c_idx,
    })
    save_case("solve_joint", arrays, scalars)


def main():
    assert sys.byteorder == "little", "golden vectors are little-endian raw bytes"
    DATA.mkdir(parents=True, exist_ok=True)
    print(f"writing golden vectors to {DATA}")
    gen_conv_cases()
    gen_solve_single()
    gen_solve_joint()
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, check=True,
                            capture_output=True, text=True).stdout.strip()
    meta = {
        "generated": date.today().isoformat(),
        "commit": commit,
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "scipy": scipy.__version__,
    }
    (DATA / "meta.json").write_text(json.dumps(meta, indent=1) + "\n")
    print(f"done (reference commit {commit[:12]})")


if __name__ == "__main__":
    main()
