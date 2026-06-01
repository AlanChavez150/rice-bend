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
from pathlib import Path

import coloredlogs
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter, writers


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
        description="Animate the TX aperture estimate across Gerchberg-Saxton iterations"
    )
    parser.add_argument(
        "run",
        nargs="?",
        type=Path,
        default=Path("results"),
        help="Run directory, a run.npz, or a results/ base dir (default: latest under results/)",
    )
    parser.add_argument(
        "--out", "-o",
        type=Path,
        default=None,
        help="Output .mp4 path. Default: tx_estimate.mp4 in the run dir",
    )
    parser.add_argument("--fps", type=int, default=15, help="Frames per second")
    parser.add_argument("--show", action="store_true", default=False, help="Also display interactively")
    parser.add_argument("--debug", action="store_true", default=False)
    args = parser.parse_args()

    coloredlogs.install(level="DEBUG" if args.debug else "INFO", fmt="%(levelname)s: %(message)s")

    run_dir = resolve_run_dir(args.run)
    out_path = args.out if args.out is not None else run_dir / "tx_estimate.mp4"
    animate_tx_estimate(run_dir, out_path, fps=args.fps, show=args.show)


if __name__ == "__main__":
    main()
