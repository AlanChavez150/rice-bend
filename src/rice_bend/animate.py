"""Animate the TX aperture estimate as it evolves over Gerchberg-Saxton iterations.

Reads a saved run (run.npz + run.json, written by rice_bend.data_store) and renders
the reconstructed TX aperture phase plate frame-by-frame, alongside the convergence
loss curve. The TX estimate at captured frame k is:

    estimate = gs_fixed_aper_amp * exp(1j * gs_phase_captured[k])

Only the phase is optimized (amplitude is held fixed), so the phase is what animates.
"""

import argparse
import json
import logging
import os
from collections import namedtuple
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import coloredlogs
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter, writers

from rice_bend import rs


# Per-worker shared state for parallel scene-frame re-illumination (set by the pool
# initializer so the read-only arrays are not re-pickled for every frame).
_SceneAnimShared = namedtuple("_SceneAnimShared", "x amp phases z_axis wavelength tx_z")
_SCENE_ANIM = None


def _init_scene_anim_worker(shared: "_SceneAnimShared") -> None:
    global _SCENE_ANIM
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ.setdefault(var, "1")
    _SCENE_ANIM = shared


def _reilluminate_frame(k: int) -> np.ndarray:
    """Re-illuminate the scene with the TX estimate at captured iteration k (worker task)."""
    s = _SCENE_ANIM
    u0 = s.amp * np.exp(1j * s.phases[k])
    return np.abs(rs.illuminate(s.x, s.z_axis, u0, s.wavelength, s.tx_z))


def find_latest_run(base: Path) -> Path:
    """Return the most recent run directory under `base` that contains a run.npz."""
    base = Path(base)
    candidates = sorted(p.parent for p in base.glob("*/run.npz"))
    if not candidates:
        raise FileNotFoundError(f"No run.npz found under {base}/")
    return candidates[-1]


def resolve_run_dir(path: Path) -> Path:
    """Accept a run directory, a run.npz path, or a results/ base; return the run dir."""
    path = Path(path)
    if path.is_file() and path.suffix == ".npz":
        return path.parent
    if (path / "run.npz").exists():
        return path
    # treat as a base dir holding run subdirectories
    return find_latest_run(path)


def _unwrap_masked(phase: np.ndarray, support: np.ndarray) -> np.ndarray:
    """Unwrap phase across the aperture support only; NaN elsewhere (np.unwrap
    can't span the meaningless out-of-aperture region)."""
    out = np.full(phase.shape, np.nan, dtype=float)
    idx = np.where(support)[0]
    if idx.size:
        out[idx] = np.unwrap(phase[idx])
    return out


def _aligned_reference_phase(x_axis, support, amp, final_phase, tx_axis, tx_profile):
    """Interpolate the real TX onto x_axis and rotate by a global phase so its
    convention matches the recovered estimate (phase retrieval has a global-phase
    ambiguity). Returns unwrapped phase masked outside the aperture, or None."""
    # np.interp (not interp.interp_real_imag): the two disagree at ULP level, and this
    # one's edge clamp is inert anyway -- `real` is only ever read at [support], and
    # support lies inside the TX aperture extent, so the clamped tail is never used.
    real = np.interp(x_axis, tx_axis, tx_profile.real) + 1j * np.interp(x_axis, tx_axis, tx_profile.imag)
    est_final = amp * np.exp(1j * final_phase)
    overlap = np.sum(np.conj(est_final[support]) * real[support])
    if overlap == 0:
        return None
    global_phase = np.angle(overlap)
    real_aligned = real * np.exp(-1j * global_phase)
    return _unwrap_masked(np.angle(real_aligned), support)


def animate_tx_estimate(run_dir: Path, out_path: Path, fps: int = 15,
                        show: bool = False) -> Path:
    log = logging.getLogger()
    run_dir = Path(run_dir)
    z = np.load(run_dir / "run.npz")
    with open(run_dir / "run.json") as f:
        meta = json.load(f)

    if "gs_phase_captured" not in z.files:
        raise KeyError(
            "run.npz has no 'gs_phase_captured' — this run was saved without GS history. "
            "Re-run with output.save_gs_history: true (and a small history_stride for more frames)."
        )

    x = z["x_axis"]
    amp = z["gs_fixed_aper_amp"]
    support = z["gs_support"].astype(bool)
    phases = z["gs_phase_captured"]          # (K, size)
    iters = z["gs_iter_indices"]             # (K,)
    loss_full = z["gs_loss_full"]            # (n_iters,)
    loss_cap = z["gs_loss_captured"]         # (K,)
    n_frames = phases.shape[0]
    log.info(f"Animating {n_frames} captured frames from {run_dir}")

    # estimate phase per frame, unwrapped over the aperture support
    def frame_phase(k):
        ph = np.angle(amp * np.exp(1j * phases[k]))
        return _unwrap_masked(ph, support)

    # precompute all frames so the phase axis can be scaled once (stable across frames)
    all_phase = np.array([frame_phase(k) for k in range(n_frames)])

    # optional real TX reference (sim runs only)
    ref_phase = None
    if not meta.get("is_experimental", False) and "tx_real_aper_axis" in z.files:
        ref_phase = _aligned_reference_phase(
            x, support, amp, phases[-1],
            z["tx_real_aper_axis"], z["tx_real_aper_profile"],
        )

    # phase-axis limits from all data (unwrapped phase is unbounded)
    finite = all_phase[np.isfinite(all_phase)]
    if ref_phase is not None:
        finite = np.concatenate([finite, ref_phase[np.isfinite(ref_phase)]])
    y_lo, y_hi = float(finite.min()), float(finite.max())
    pad = 0.1 * (y_hi - y_lo) if y_hi > y_lo else 1.0

    # 1920x1080 output: figsize(inches) * dpi = pixels
    dpi = 100
    fig, (ax_ph, ax_loss) = plt.subplots(2, 1, figsize=(19.2, 10.8), dpi=dpi, layout="constrained")

    # panel 1: aperture phase estimate
    ax_ph.set_title("TX aperture phase estimate")
    ax_ph.set_xlabel("x (m)")
    ax_ph.set_ylabel("phase unwrapped [rad]")
    ax_ph.set_xlim(np.nanmin(x), np.nanmax(x))
    ax_ph.set_ylim(y_lo - pad, y_hi + pad)
    ax_ph.grid(True)
    if ref_phase is not None:
        ax_ph.plot(x, ref_phase, "--", color="0.6", linewidth=2, label="Real TX (global-phase aligned)")
    (est_line,) = ax_ph.plot([], [], color="C0", linewidth=2, label="MGS estimate")
    ax_ph.legend(loc="upper right")

    # panel 2: convergence loss with a moving marker
    ax_loss.set_title("Convergence")
    ax_loss.set_xlabel("iteration")
    ax_loss.set_ylabel("loss")
    ax_loss.grid(True)
    ax_loss.plot(np.arange(len(loss_full)), loss_full, color="0.6", linewidth=1)
    (loss_dot,) = ax_loss.plot([], [], "o", color="C3", markersize=8)

    def init():
        est_line.set_data([], [])
        loss_dot.set_data([], [])
        return est_line, loss_dot

    def update(k):
        est_line.set_data(x, all_phase[k])
        loss_dot.set_data([iters[k]], [loss_cap[k]])
        ax_ph.set_title(f"TX aperture phase estimate — iteration {int(iters[k])}, loss {loss_cap[k]:.3e}")
        return est_line, loss_dot

    anim = FuncAnimation(fig, update, init_func=init, frames=n_frames, blit=False)
    return _write_mp4(anim, fig, out_path, fps, dpi, show, log)


def animate_scene_reillumination(run_dir: Path, out_path: Path, fps: int = 15,
                                 frame_stride: int = None, z_stride: int = None,
                                 jobs: int = 1, show: bool = False) -> Path:
    """Animate the 2D scene field re-illuminated by the TX aperture estimate at each
    captured iteration. The scene is NOT stored per iteration, so it is recomputed here
    via Rayleigh-Sommerfeld propagation of estimate = gs_fixed_aper_amp * exp(1j*phase[k])."""
    log = logging.getLogger()
    run_dir = Path(run_dir)
    z = np.load(run_dir / "run.npz")
    with open(run_dir / "run.json") as f:
        meta = json.load(f)

    if "gs_phase_captured" not in z.files:
        raise KeyError(
            "run.npz has no 'gs_phase_captured' — re-run with output.save_gs_history: true."
        )
    if "scene_z_axis" not in z.files:
        raise KeyError(
            "run.npz has no 'scene_z_axis' — re-run mgs to enable scene re-illumination "
            "(older runs did not save the scene axis)."
        )

    x = z["x_axis"]
    amp = z["gs_fixed_aper_amp"]
    phases = z["gs_phase_captured"]
    iters = z["gs_iter_indices"]
    loss_cap = z["gs_loss_captured"]
    z_axis_full = z["scene_z_axis"]
    wavelength = meta["wavelength_m"]
    sc = meta["scene"]
    extent = [sc["x_min"], sc["x_max"], sc["z_min"], sc["z_max"]]
    # source (TX) plane; energy flows toward -Z. Fall back to z_max (the TX plane)
    # for runs saved before tx_z was persisted.
    tx_z = sc.get("tx_z", sc["z_max"])

    # subsample captured iterations (a full-scene propagation per frame is expensive)
    K = phases.shape[0]
    auto_stride = frame_stride is None
    if auto_stride:
        frame_stride = max(1, K // 60)
    fsel = list(range(0, K, frame_stride))
    if fsel[-1] != K - 1:
        fsel.append(K - 1)
    if frame_stride > 1:
        msg = (f"Using {len(fsel)} of {K} captured iterations (frame_stride={frame_stride}); "
               f"pass --frame-stride 1 to animate every captured iteration.")
        log.warning(msg + " [auto-capped to keep render time reasonable]" if auto_stride else msg)

    # subsample output z-planes to bound compute/memory (x stays full-res: rs needs it)
    nz = len(z_axis_full)
    if z_stride is None:
        z_stride = max(1, nz // 300)
    z_axis = z_axis_full[::z_stride]

    log.info(f"Re-illuminating scene for {len(fsel)} frames over {len(z_axis)} z-planes "
             f"(frame_stride={frame_stride}, z_stride={z_stride})")
    n_jobs = max(1, int(jobs))
    if n_jobs == 1 or len(fsel) <= 1:
        frames = []
        for j, k in enumerate(fsel):
            u0 = amp * np.exp(1j * phases[k])
            frames.append(np.abs(rs.illuminate(x, z_axis, u0, wavelength, tx_z)))
            if j % 10 == 0:
                log.info(f"  propagated frame {j + 1}/{len(fsel)} (iteration {int(iters[k])})")
    else:
        # each frame's full-scene RS propagation is independent -> render across workers
        n_workers = min(n_jobs, len(fsel))
        log.info(f"  propagating {len(fsel)} frames across {n_workers} worker process(es)")
        shared = _SceneAnimShared(x=x, amp=amp, phases=phases, z_axis=z_axis,
                                  wavelength=wavelength, tx_z=tx_z)
        with ProcessPoolExecutor(max_workers=n_workers, initializer=_init_scene_anim_worker,
                                 initargs=(shared,)) as ex:
            # map preserves input order, so frames line up with fsel
            frames = list(ex.map(_reilluminate_frame, fsel))
    vmax = max(float(f.max()) for f in frames)

    dpi = 100
    fig, ax = plt.subplots(figsize=(19.2, 10.8), dpi=dpi, layout="constrained")  # 1920x1080
    ax.set_xlabel("x (m)")
    ax.set_ylabel("z (m)")
    im = ax.imshow(frames[0], extent=extent, cmap="inferno", vmin=0.0, vmax=vmax,
                   aspect="auto", origin="lower")
    fig.colorbar(im, ax=ax, label="|field| (V/m)")

    def update(j):
        im.set_data(frames[j])
        ax.set_title(f"Scene re-illuminated by TX estimate — iteration "
                     f"{int(iters[fsel[j]])}, loss {loss_cap[fsel[j]]:.3e}")
        return [im]

    anim = FuncAnimation(fig, update, frames=len(fsel), blit=False)
    return _write_mp4(anim, fig, out_path, fps, dpi, show, log)


def _write_mp4(anim, fig, out_path: Path, fps: int, dpi: int, show: bool, log) -> Path:
    if not writers.is_available("ffmpeg"):
        raise RuntimeError(
            "ffmpeg is not available — install it (e.g. `apt install ffmpeg`) to render .mp4 animations."
        )
    out_path = Path(out_path).with_suffix(".mp4")
    writer = FFMpegWriter(fps=fps)
    log.info(f"Writing animation to {out_path} ({fps} fps)")
    anim.save(out_path, writer=writer, dpi=dpi)

    if show:
        plt.show()
    plt.close(fig)
    return out_path


def main():
    parser = argparse.ArgumentParser(
        description="Animate Gerchberg-Saxton progress from a saved run (.mp4 via ffmpeg)"
    )
    parser.add_argument(
        "run",
        nargs="?",
        type=Path,
        default=Path("results"),
        help="Run directory, a run.npz, or a results/ base dir (default: latest under results/)",
    )
    parser.add_argument(
        "--mode",
        choices=["phase", "scene"],
        default="phase",
        help="'phase': TX aperture phase estimate; 'scene': 2D scene re-illuminated by the TX estimate",
    )
    parser.add_argument(
        "--out", "-o",
        type=Path,
        default=None,
        help="Output .mp4 path. Default: tx_estimate.mp4 / scene_reillum.mp4 in the run dir",
    )
    parser.add_argument("--fps", type=int, default=15, help="Frames per second")
    parser.add_argument("--frame-stride", type=int, default=None,
                        help="[scene] use every Nth captured iteration (default: ~60 frames total)")
    parser.add_argument("--jobs", "-j", type=int, default=1,
                        help="[scene] worker processes for frame re-illumination (1 = serial)")
    parser.add_argument("--z-stride", type=int, default=None,
                        help="[scene] subsample output z-planes (default: ~300 planes)")
    parser.add_argument("--show", action="store_true", default=False, help="Also display interactively")
    parser.add_argument("--debug", action="store_true", default=False)
    args = parser.parse_args()

    coloredlogs.install(level="DEBUG" if args.debug else "INFO", fmt="%(levelname)s: %(message)s")

    run_dir = resolve_run_dir(args.run)
    if args.mode == "scene":
        out_path = args.out if args.out is not None else run_dir / "scene_reillum.mp4"
        animate_scene_reillumination(run_dir, out_path, fps=args.fps,
                                     frame_stride=args.frame_stride, z_stride=args.z_stride,
                                     jobs=args.jobs, show=args.show)
    else:
        out_path = args.out if args.out is not None else run_dir / "tx_estimate.mp4"
        animate_tx_estimate(run_dir, out_path, fps=args.fps, show=args.show)


if __name__ == "__main__":
    main()
