#!/usr/bin/env python
"""Generate the golden test vectors under rust/tests/data/.

Read-only with respect to src/rice_bend (imports only); rerunnable. The Python
engine is the reference — these vectors ARE its behaviour, captured. Regenerate
after any (approved, separately-validated) change to the reference solver, and
note the commit recorded in tests/data/meta.json.

    .venv/bin/python rust/python/gen_golden.py            # every case
    .venv/bin/python rust/python/gen_golden.py solve_descend solve_descend_joint
    .venv/bin/python rust/python/gen_golden.py solve_level solve_level_joint
    .venv/bin/python rust/python/gen_golden.py solve_level_penalty solve_descend_penalty

Naming cases regenerates those alone and leaves meta.json as it is; each such case
records its own provenance (generated / commit) in its manifest's scalars.

Cases produced (names are pinned by rust/tests/conv_parity.rs / solve_parity.rs):
  conv_n{7,8,2399,2401}   random complex pairs — odd/even "same"-slice centering
  conv_n2400              a real RS kernel + box-aperture field at the flagship size
  solve_single            one F=1 candidate solve at 300 GHz, tiny_check geometry
                          (N=800)
  solve_joint             the same candidate at F=2 (280/300 GHz) with the full warm
                          start, including the stage-2/3 intermediates (psi0, delta)
  numerics                np.add.reduce (f64/f32), the f32 convergence-window
                          mean, and np.unwrap — gates src/npmath.rs bit-exactly
  solve_descend           amplitude: descend, F=1 at 300 GHz from the backprop
                          start, tiny_check geometry; the start amplitude a0 is
                          zeroed inside the window (DESCEND_HOLES), so the support
                          handed over is not a0 > 0
  solve_descend_joint     the same at F=2 (280/300 GHz) from the legacy warm start
  solve_level             amplitude: level, F=1 at 300 GHz from the backprop start,
                          tiny_check geometry: the start level A0 crosses as A0 on
                          the support (aper_amp), with the support
  solve_level_joint       the same at F=2 (280/300 GHz), backprop start (several
                          tones under the legacy warm start are refused for level)
  solve_level_penalty     solve_level with the excess-power penalty on
                          (power_penalty 1.0): the manifest adds power_penalty,
                          rx_power (P_rx as python hands it to rust), final_mse
                          and power_ratio
  solve_descend_penalty   solve_descend's holed a0 with the penalty on
                          (power_penalty 1.0), the same added scalars, but the
                          start scaled: a0 x DESCEND_PENALTY_START_SCALE (150, the
                          holes staying 0) and amp_max 2 x 150 = 300, recorded as
                          start_scale. Unscaled, the raw back-propagated a0
                          radiates 1e-4 of P_rx on tiny_check: the ratio never
                          lifts off its floor at 1, the solve is solve_descend's
                          to the bit and the case would pin nothing of the
                          penalty. Scaled, the start plate radiates ~2x P_rx

The frequencies and sizes above are what a regeneration produces NOW, from the
retargeted configs (300 GHz carrier, tiny_check.yml at 0.75 mm scene spacing). The
vectors committed under tests/data/ predate the retarget — 150 GHz, 140/150 GHz and
N=400, at the commit meta.json records — and stay valid: they pin the solver's
arithmetic, which does not care which carrier produced its inputs.
"""
import json
import subprocess
import sys
from datetime import date
from pathlib import Path

import numpy as np
import scipy.fft
import scipy.signal

from rice_bend import mgs as mgs_module
from rice_bend import rs
from rice_bend.config import GerchbergSaxtonConfig, center_freq_index, load_config
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
    # (TX 0.3 m above the RX plane at 300 GHz) and a box aperture random-phase field
    n = 2400
    x_axis = np.linspace(-0.3, 0.3, n)
    h = rs.rs_kernel(x_axis, rs.wavelength(300e9), 0.3)
    rng = np.random.default_rng(2400)
    amp = np.where(np.abs(x_axis) <= 0.05, 1.0, 0.0)
    u0 = amp * np.exp(1j * 2 * np.pi * rng.random(n))
    conv_case("conv_n2400", u0, h, float(x_axis[1] - x_axis[0]))


def _candidate_setup(freqs):
    """The sweep's own construction of one usable candidate on tiny_check:
    MGS + measure + run_grid_search's gs_overrides/seed mutation (step 2 of
    grid_sweep.run_grid_search) + enumerate_grid to pick a point valid at every
    frequency + the box amplitude every candidate solve is handed
    (grid_sweep._reconstruct_candidate)."""
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
    x_axis, point, _support, amp, channels, params, _ref, rx_z = _candidate_setup([300e9])
    ref_freq = 300e9
    result = gs_reconstruct(tx_z=point.z, orig_aper_amp=amp, x_axis=x_axis, rx_z=rx_z,
                            channels=channels, params=params, ref_freq=ref_freq,
                            capture=False, log=None)
    arrays, scalars = _solve_inputs(x_axis, point.z, rx_z, amp, channels, params, ref_freq)
    out_a, out_s = _result_outputs(result)
    arrays.update(out_a)
    scalars.update(out_s)
    save_case("solve_single", arrays, scalars)


def gen_solve_joint():
    freqs = [280e9, 300e9]
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


# amp_max of the descend cases, in the fixed solver's amplitude units
DESCEND_AMP_MAX = 2.0


def _holed_backprop_amplitude(*args, **kwargs):
    """mgs._backprop_amplitude with zeros INSIDE the window — every 5th support
    sample and a run of 25 a third of the way in — so that the support the solve
    keeps (|orig_aper_amp| > 0) and a0 > 0 differ, the case the engine must not
    get wrong. Installed for the reference solve and the capture alike."""
    amp = _REAL_BACKPROP_AMPLITUDE(*args, **kwargs)
    idx = np.where(kwargs["support"])[0]
    amp[idx[::5]] = 0.0
    amp[idx[len(idx) // 3: len(idx) // 3 + 25]] = 0.0
    return amp


_REAL_BACKPROP_AMPLITUDE = mgs_module._backprop_amplitude


class _Captured(Exception):
    pass


def _rust_boundary_inputs(solve_kwargs, params):
    """What gs_reconstruct hands the rust engine — mgs._gs_solve_rust's arguments,
    captured in memory without running rust. The start phase (backprop) and the
    start amplitude (descend) are built in mgs.py, so they are taken from there
    rather than re-derived here; so is P_rx (mgs._received_power), which crosses
    as `rx_power` beside the penalty's gamma."""
    seen = {}
    real = mgs_module._gs_solve_rust

    def spy(channels, params, ref_freq, seed, initial_phase, curr_aper_amp, h_fwd, dx,
            log=None, init_mode=None, support=None, rx_power=None):
        seen.update(initial_phase=np.asarray(initial_phase, dtype=np.float64).copy(),
                    aper_amp=np.asarray(curr_aper_amp, dtype=np.float64).copy(),
                    support=np.asarray(support, dtype=bool).copy(),
                    init_mode=init_mode if init_mode is not None else params.init,
                    power_penalty=params.power_penalty, rx_power=rx_power)
        raise _Captured

    mgs_module._gs_solve_rust = spy
    try:
        gs_reconstruct(params=params.model_copy(update={"engine": "rust"}), **solve_kwargs)
    except _Captured:
        pass
    finally:
        mgs_module._gs_solve_rust = real
    assert seen, "gs_reconstruct did not reach the rust dispatch"
    return seen


def _penalty_scalars(crossing, result, power_penalty):
    """The excess-power penalty's manifest scalars: gamma and P_rx as they cross
    the rust boundary, and the reference's final plain loss and ratio (beside
    final_loss, which is then the objective). Empty with the penalty off, so the
    plain cases' manifests are what they were."""
    if power_penalty is None:
        return {}
    assert crossing["power_penalty"] == power_penalty
    assert crossing["rx_power"] is not None and crossing["rx_power"] > 0
    assert result.power_ratio is not None
    return {"power_penalty": float(crossing["power_penalty"]),
            "rx_power": float(crossing["rx_power"]),
            "final_mse": float(result.final_mse),
            "power_ratio": float(result.power_ratio)}


def _gen_descend(name, freqs, init, power_penalty=None, start_scale=None):
    """`start_scale` (None: off) multiplies the holed start amplitude, and amp_max
    with it, so the start plate's power can be set against P_rx (the penalty
    case); the holes stay 0."""
    x_axis, point, support, amp, channels, params, ref_freq, rx_z = _candidate_setup(freqs)
    if len(freqs) == 1:
        ref_freq = float(freqs[0])
    amp_max = DESCEND_AMP_MAX if start_scale is None else DESCEND_AMP_MAX * start_scale
    params = GerchbergSaxtonConfig(**{**params.model_dump(), "init": init,
                                      "amplitude": "descend",
                                      "amp_max": amp_max,
                                      "power_penalty": power_penalty})
    solve_kwargs = dict(tx_z=point.z, orig_aper_amp=amp, x_axis=x_axis, rx_z=rx_z,
                        channels=channels, ref_freq=ref_freq, capture=False, log=None)
    start_amplitude = _holed_backprop_amplitude
    if start_scale is not None:
        def start_amplitude(*args, **kwargs):
            return start_scale * _holed_backprop_amplitude(*args, **kwargs)
    mgs_module._backprop_amplitude = start_amplitude
    try:
        result = gs_reconstruct(params=params.model_copy(update={"engine": "python"}),
                                **solve_kwargs)
        crossing = _rust_boundary_inputs(solve_kwargs, params)
    finally:
        mgs_module._backprop_amplitude = _REAL_BACKPROP_AMPLITUDE
    assert np.array_equal(crossing["support"], support)
    assert not np.array_equal(crossing["aper_amp"] > 0, support), "a0 has no holes"

    arrays, scalars = _solve_inputs(x_axis, point.z, rx_z, amp, channels, params, ref_freq)
    arrays["initial_phase"] = crossing["initial_phase"]   # the draw, or the backprop start
    arrays["aper_amp"] = crossing["aper_amp"]             # the start amplitude a0
    arrays["support"] = crossing["support"].astype(np.float64)   # 0/1 (the loader has no bool)
    scalars["init_mode"] = crossing["init_mode"]           # what rust is told
    scalars["amp_max"] = float(params.amp_max)
    if start_scale is not None:
        scalars["start_scale"] = float(start_scale)
    out_a, out_s = _result_outputs(result)
    arrays.update(out_a)
    scalars.update(out_s)
    scalars.update(_penalty_scalars(crossing, result, power_penalty))
    scalars.update(_provenance())
    save_case(name, arrays, scalars)


def _gen_level(name, freqs, power_penalty=None):
    x_axis, point, support, amp, channels, params, ref_freq, rx_z = _candidate_setup(freqs)
    if len(freqs) == 1:
        ref_freq = float(freqs[0])
    params = GerchbergSaxtonConfig(**{**params.model_dump(), "init": "backprop",
                                      "amplitude": "level", "amp_max": None,
                                      "power_penalty": power_penalty})
    solve_kwargs = dict(tx_z=point.z, orig_aper_amp=amp, x_axis=x_axis, rx_z=rx_z,
                        channels=channels, ref_freq=ref_freq, capture=False, log=None)
    result = gs_reconstruct(params=params.model_copy(update={"engine": "python"}),
                            **solve_kwargs)
    crossing = _rust_boundary_inputs(solve_kwargs, params)
    assert np.array_equal(crossing["support"], support)
    a0 = crossing["aper_amp"]
    assert np.all(a0[support] == a0[support][0]) and np.all(a0[~support] == 0), \
        "level's start plate is A0 on the support"

    arrays, scalars = _solve_inputs(x_axis, point.z, rx_z, amp, channels, params, ref_freq)
    arrays["initial_phase"] = crossing["initial_phase"]   # the backprop start
    arrays["aper_amp"] = a0                               # A0 * 1_support
    arrays["support"] = crossing["support"].astype(np.float64)
    scalars["init_mode"] = crossing["init_mode"]
    scalars["level_start"] = float(a0[support][0])
    out_a, out_s = _result_outputs(result)
    arrays.update(out_a)
    scalars.update(out_s)
    scalars.update(_penalty_scalars(crossing, result, power_penalty))
    scalars.update(_provenance())
    save_case(name, arrays, scalars)


def gen_solve_level():
    _gen_level("solve_level", [300e9])


def gen_solve_level_joint():
    _gen_level("solve_level_joint", [280e9, 300e9])


def gen_solve_descend():
    _gen_descend("solve_descend", [300e9], "backprop")


def gen_solve_descend_joint():
    _gen_descend("solve_descend_joint", [280e9, 300e9], "warm_start")


# gamma of the penalty cases: the plain product L * R, the simplest J that charges
PENALTY_GAMMA = 1.0
# solve_descend_penalty's start: the holed a0 (and amp_max) times this. The raw
# back-propagated a0 on tiny_check radiates 9.9e-5 of P_rx, so unscaled the ratio
# never leaves its floor and the case is solve_descend's; 150^2 * 9.9e-5 ~ 2.2
DESCEND_PENALTY_START_SCALE = 150.0


def gen_solve_level_penalty():
    _gen_level("solve_level_penalty", [300e9], power_penalty=PENALTY_GAMMA)


def gen_solve_descend_penalty():
    _gen_descend("solve_descend_penalty", [300e9], "backprop",
                 power_penalty=PENALTY_GAMMA, start_scale=DESCEND_PENALTY_START_SCALE)


def _provenance():
    commit = subprocess.run(["git", "describe", "--always", "--dirty", "--abbrev=40"],
                            cwd=REPO, check=True, capture_output=True,
                            text=True).stdout.strip()
    return {"generated": date.today().isoformat(), "commit": commit,
            "numpy": np.__version__, "scipy": scipy.__version__}


def gen_numerics():
    """Ragged batches stored flat: `<name>_data` concatenated, `<name>_lens` as
    f64 (exact small integers), `<name>_expected` one value per batch."""
    rng = np.random.default_rng(7)

    def pack(arrays):
        return (np.concatenate(arrays),
                np.array([len(a) for a in arrays], dtype=np.float64))

    # f64 sums: every size class of the pairwise tree, incl. the 2400 scene size
    lens64 = [1, 2, 7, 8, 9, 15, 16, 17, 127, 128, 129, 130, 255, 256, 400, 1000,
              2399, 2400, 2401] + [int(v) for v in rng.integers(1, 3000, 40)]
    arr64 = [rng.standard_normal(n) * 10.0 ** rng.integers(-6, 6) for n in lens64]
    sum64_data, sum64_lens = pack(arr64)
    sum64_expected = np.array([np.add.reduce(a) for a in arr64], dtype=np.float64)

    # f32 sums, including the loss-history sizes the convergence window sees
    lens32 = [1, 7, 8, 9, 10, 16, 17, 100] + [int(v) for v in rng.integers(1, 200, 30)]
    arr32 = [(rng.standard_normal(n) * 10.0 ** rng.integers(-6, 3)).astype(np.float32)
             for n in lens32]
    sum32_data, sum32_lens = pack(arr32)
    sum32_expected = np.array([np.add.reduce(a) for a in arr32], dtype=np.float32)

    # the convergence test verbatim: mean(diff(10 f32 losses)) (mgs.py:309-310)
    win = (1e-3 * np.exp(-rng.uniform(0, 3, (60, 10)).cumsum(axis=1))).astype(np.float32)
    win_expected = np.array([np.mean(np.diff(w)) for w in win], dtype=np.float32)

    # unwrap: smooth ramps, noisy walks, and exact +/-pi steps (the ambiguous boundary)
    unw = [np.angle(np.exp(1j * np.cumsum(rng.normal(0, s, n))))
           for s, n in ((0.3, 400), (1.5, 400), (2.9, 200), (0.05, 2400))]
    unw.append(np.array([0.0, np.pi, 0.0, -np.pi, 0.0, np.pi, 2 * np.pi, -np.pi]))
    unw.append(np.array([3.0, -3.0, 3.1, -3.1, 1e-3]))
    unwrap_in, unwrap_lens = pack(unw)
    unwrap_out = np.concatenate([np.unwrap(a) for a in unw])

    save_case("numerics",
              {"sum64_data": sum64_data, "sum64_lens": sum64_lens,
               "sum64_expected": sum64_expected,
               "sum32_data": sum32_data, "sum32_lens": sum32_lens,
               "sum32_expected": sum32_expected,
               "win_data": win.ravel(), "win_expected": win_expected,
               "unwrap_in": unwrap_in, "unwrap_lens": unwrap_lens,
               "unwrap_out": unwrap_out},
              {})


CASES = {
    "conv": gen_conv_cases,
    "solve_single": gen_solve_single,
    "solve_joint": gen_solve_joint,
    "numerics": gen_numerics,
    "solve_descend": gen_solve_descend,
    "solve_descend_joint": gen_solve_descend_joint,
    "solve_level": gen_solve_level,
    "solve_level_joint": gen_solve_level_joint,
    "solve_level_penalty": gen_solve_level_penalty,
    "solve_descend_penalty": gen_solve_descend_penalty,
}


def main():
    assert sys.byteorder == "little", "golden vectors are little-endian raw bytes"
    DATA.mkdir(parents=True, exist_ok=True)
    names = sys.argv[1:]
    unknown = [n for n in names if n not in CASES]
    if unknown:
        sys.exit(f"unknown case(s) {unknown}; known: {sorted(CASES)}")
    print(f"writing golden vectors to {DATA}")
    if names:
        for n in names:
            CASES[n]()
        print("meta.json left as it is (the named cases carry their own provenance)")
        return
    for gen in CASES.values():
        gen()
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
