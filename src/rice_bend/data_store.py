"""Run persistence: the per-run directory, its numeric arrays (.npz), its scalar
metadata (.json), the config snapshots, and the small I/O helpers both entry points
share.

Storage split: all numeric arrays (including complex) live in the .npz; only
JSON-safe scalars (no complex, no numpy types) live in the .json.

The helpers below are public because grid_search needs them too. They used to be
underscore-private here, reached across a command boundary -- a reliable signal that
the layering was wrong rather than that the names should stay private.
"""

import json
import logging
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import numpy as np

from rice_bend import __version__
from rice_bend.config import SimConfig, load_config


@dataclass
class GSHistory:
    """Accumulates gradient-descent state during run_gerch_sax.

    Per-iteration phase/field are captured on a stride (and always on the final
    iteration). Growing lists are used during the loop so early convergence
    doesn't waste a preallocated (max_iters, size) block; finalize() converts
    them to arrays.

    A joint multi-frequency solve shares ONE phase/support/amplitude across all
    frequencies, so the phase-side arrays are unchanged; the measurement-side
    arrays carry a leading frequency axis (F, ...) aligned with `frequencies`,
    and losses come in a joint (mean) form plus per-frequency components.
    """
    # one-time fixed arrays (set at construction, before the loop)
    x_axis: np.ndarray
    frequencies: np.ndarray        # (F,) Hz — alignment order for every (F, ...) array
    orig_prop_f: np.ndarray        # (F, size) target RX fields
    orig_aper_amp: np.ndarray      # fixed aperture amplitude (shared across F)
    error_weighting: np.ndarray    # (F, size)
    support: np.ndarray            # bool mask (shared across F)
    initial_phase: np.ndarray      # seeded RNG draw (shared across F)
    rx_z: float
    seed: int
    history_stride: int

    # dense per-iteration loss (cheap, full length); joint + per-frequency rows
    loss_full: List[float] = field(default_factory=list)
    loss_full_per_freq: List[np.ndarray] = field(default_factory=list)   # rows (F,)
    # strided captures
    iter_indices: List[int] = field(default_factory=list)
    loss_captured: List[float] = field(default_factory=list)             # joint
    phase_captured: List[np.ndarray] = field(default_factory=list)
    prop_field_captured: List[np.ndarray] = field(default_factory=list)  # each (F, size)

    # start/stop conditions (filled at finalize)
    stop_reason: Optional[str] = None
    n_iters_run: int = 0
    final_loss: float = float("nan")
    final_loss_per_freq: Optional[np.ndarray] = None                     # (F,)

    def _capture(self, iter_idx: int, loss: float, phase: np.ndarray,
                 prop_fields: List[np.ndarray]) -> None:
        # avoid duplicate capture (e.g. final iter also landing on the stride)
        if self.iter_indices and self.iter_indices[-1] == iter_idx:
            return
        self.iter_indices.append(int(iter_idx))
        self.loss_captured.append(float(loss))
        self.phase_captured.append(phase.copy())
        # np.stack copies, so the captured (F, size) block is decoupled from the loop
        self.prop_field_captured.append(np.stack([np.asarray(p) for p in prop_fields]))

    def record_iter(self, iter_idx: int, loss: float, loss_per_freq: np.ndarray,
                    phase: np.ndarray, prop_fields: List[np.ndarray]) -> None:
        """Call once per iteration. Always records dense losses; captures
        phase/fields on the stride."""
        self.loss_full.append(float(loss))
        self.loss_full_per_freq.append(np.asarray(loss_per_freq, dtype=np.float64).copy())
        if iter_idx % self.history_stride == 0:
            self._capture(iter_idx, loss, phase, prop_fields)

    def capture_final(self, iter_idx: int, loss: float, phase: np.ndarray,
                      prop_fields: List[np.ndarray]) -> None:
        """Force-capture the last iteration's strided state (no dense loss append)."""
        self._capture(iter_idx, loss, phase, prop_fields)

    def finalize(self, stop_reason: str, n_iters_run: int, final_loss: float,
                 final_loss_per_freq: np.ndarray) -> None:
        self.stop_reason = stop_reason
        self.n_iters_run = int(n_iters_run)
        self.final_loss = float(final_loss)
        self.final_loss_per_freq = np.asarray(final_loss_per_freq,
                                              dtype=np.float64).copy()


# Marker files identifying which entry point owns a run directory. `mgs` and
# `grid-search-mgs` resolve output.run_name to the SAME path, so without this a
# scenario config run through one tool silently deletes the other's saved results.
# `frequencies.json` stays in the grid tuple even though new (joint) runs never
# write it: it marks the RETIRED per-frequency layout, and keeping it lets a new
# joint run clear an old results directory it legitimately owns.
RUN_DIR_MARKERS = {
    "mgs": ("run.json",),
    "grid": ("candidate_beams.json", "frequencies.json"),
}


def check_run_dir(output_base: Path, run_name: Optional[str], kind: str) -> Path:
    """Resolve <output_base>/<run_name>/ and exit if clearing it would destroy data
    this run does not own. Creates and deletes nothing.

    Callers run this up front, because make_run_dir is deliberately deferred until
    after the solve — without a pre-flight the sweep would discover the collision
    only after burning hours of compute.
    """
    if kind not in RUN_DIR_MARKERS:
        raise ValueError(f"unknown run kind {kind!r}; expected one of {sorted(RUN_DIR_MARKERS)}")
    markers = RUN_DIR_MARKERS[kind]
    run_dir = Path(output_base) / (run_name or "data_dump")
    if not run_dir.exists():
        return run_dir

    def refuse(why: str) -> None:
        logging.getLogger().error(
            f"Refusing to clear {run_dir}: {why}. Pass -o/--out to write somewhere else.")
        raise SystemExit(2)

    if not run_dir.is_dir():
        refuse("it exists and is not a directory")
    if any(run_dir.iterdir()) and not any((run_dir / m).exists() for m in markers):
        refuse(f"it is not empty and holds none of {list(markers)}, so it was not "
               f"written by this tool (kind={kind!r})")
    return run_dir


def make_run_dir(output_base: Path, run_name: Optional[str], kind: str) -> Path:
    """Create and return <output_base>/<run_name>/ (run_name defaults to 'data_dump'),
    cleared of any previous run. Refuses to clear a directory this run does not own
    (see check_run_dir).

    Ordering matters and is the caller's responsibility: invoke this only once the
    expensive work has finished, so an interrupted run never destroys prior results
    without producing new ones.
    """
    run_dir = check_run_dir(output_base, run_name, kind)
    if run_dir.exists():
        if any(run_dir.iterdir()):
            logging.getLogger().warning(
                f"Run directory {run_dir} already exists — clearing its contents")
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def json_safe(obj):
    """default= hook for json.dump. Casts numpy scalars to python; rejects complex."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, complex):
        raise TypeError("complex values must be stored in the .npz, not run.json")
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def write_json(path: Path, obj) -> Path:
    """Write `obj` as indented JSON through the json_safe hook.

    Six sites spelled this line out, and write_frequencies_index silently omitted
    `default=` -- it worked only because its caller float()-cast everything first.
    """
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=json_safe)
    return Path(path)


def provenance(args_dict: dict, **extra) -> dict:
    """The provenance block every run manifest carries: the CLI invocation that
    produced it and the package version."""
    out = {
        "cli_args": {k: (str(v) if isinstance(v, Path) else v)
                     for k, v in (args_dict or {}).items()},
        "package_version": __version__,
    }
    out.update(extra)
    return out


def save_config_snapshot(run_dir: Path, config, config_path: Path) -> None:
    """Write both config snapshots: the effective one (defaults filled in) and a
    verbatim copy of the source .yml, which preserves comments."""
    write_json(Path(run_dir) / "config_snapshot.json", config.model_dump(mode="json"))
    try:
        shutil.copyfile(config_path, Path(run_dir) / "config_source.yml")
    except (OSError, TypeError):
        pass


def load_run_config(run_dir: Path, log: logging.Logger) -> Optional[SimConfig]:
    """Read a saved run's config back: prefer the verbatim source .yml, fall back to
    the effective snapshot JSON. None if neither is present.

    This is the reader for the format save_config_snapshot writes; the two halves
    used to live in different modules.
    """
    run_dir = Path(run_dir)
    src = run_dir / "config_source.yml"
    if src.exists():
        return load_config(src)
    snap = run_dir / "config_snapshot.json"
    if snap.exists():
        with open(snap) as f:
            return SimConfig.model_validate(json.load(f))
    log.warning(f"No config_source.yml/config_snapshot.json in {run_dir}; "
                "cannot build the true MGS run")
    return None


def write_mp4(anim, fig, out_path: Path, fps: int, dpi: int, show: bool, log) -> Path:
    """Render a matplotlib animation to .mp4 via ffmpeg, then close the figure.

    matplotlib is imported inside the function on purpose: this module is also the
    persistence layer for the numeric sweep, which must stay importable without
    dragging in FuncAnimation and mpl_toolkits.
    """
    import matplotlib.pyplot as plt
    from matplotlib.animation import FFMpegWriter, writers

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


def c64(a: np.ndarray) -> np.ndarray:
    return np.asarray(a, dtype=np.complex64)


def f64(a: np.ndarray) -> np.ndarray:
    return np.asarray(a, dtype=np.float64)


def _collect_arrays(mgs) -> dict:
    """Build the dict of arrays for run.npz. Guards every optional array.

    Per-frequency arrays carry a leading (F,) axis aligned with `frequencies_hz`
    — even at F == 1, so the key set and shapes never branch on the frequency
    count. The one exception is the RX aperture element arrays, whose lengths are
    genuinely ragged across frequencies (dx defaults to wavelength/20): those get
    indexed keys rx_aper_axis_00, rx_aper_profile_00, ... instead (no object
    arrays — they would break content hashing and require allow_pickle).
    """
    out = {}
    scene = mgs.scene
    gs_tx = mgs.gs_tx

    # frequency axis (always) — the alignment order for every (F, ...) array
    out["frequencies_hz"] = f64([fs.freq for fs in mgs.freq_states])
    out["wavelengths_m"] = f64([fs.wavelength for fs in mgs.freq_states])

    # apertures / axes (always). TX aperture AXES are geometry-only and identical
    # across frequencies; the profiles differ (beam phase ∝ k) and stack to (F, n).
    out["tx_real_aper_axis"] = f64(scene.tx_ap.aper_axis)
    out["tx_real_aper_profile"] = c64(
        np.stack([fs.tx_ap.aper_profile for fs in mgs.freq_states]))
    for i, fs in enumerate(mgs.freq_states):
        out[f"rx_aper_axis_{i:02d}"] = f64(fs.rx_ap.aper_axis)
        out[f"rx_aper_profile_{i:02d}"] = c64(fs.rx_ap.aper_profile)
    out["gs_tx_aper_axis"] = f64(gs_tx.aper_axis)
    out["gs_tx_aper_profile"] = c64(gs_tx.aper_profile)

    # gradient-descent history
    hist = mgs.gs_history
    if hist is not None:
        out["x_axis"] = f64(hist.x_axis)
        out["gs_target_rx_field"] = c64(hist.orig_prop_f)            # (F, size)
        out["gs_fixed_aper_amp"] = f64(hist.orig_aper_amp)
        out["gs_error_weighting"] = f64(hist.error_weighting)        # (F, size)
        out["gs_support"] = np.asarray(hist.support, dtype=bool)
        out["gs_initial_phase"] = f64(hist.initial_phase)
        out["gs_loss_full"] = np.asarray(hist.loss_full, dtype=np.float32)
        out["gs_loss_full_per_freq"] = np.asarray(hist.loss_full_per_freq,
                                                  dtype=np.float32)  # (n_iters, F)
        if mgs.output_cfg.save_gs_history and len(hist.iter_indices) > 0:
            out["gs_iter_indices"] = np.asarray(hist.iter_indices, dtype=np.int64)
            out["gs_loss_captured"] = np.asarray(hist.loss_captured, dtype=np.float32)
            out["gs_phase_captured"] = np.asarray(hist.phase_captured, dtype=np.float32)
            out["gs_prop_field_captured"] = c64(np.asarray(hist.prop_field_captured))  # (K, F, size)

    # scene axes are always saved (cheap, 1D) so a run can be re-illuminated later
    out["scene_x_axis"] = f64(scene.x_axis)
    out["scene_z_axis"] = f64(scene.z_axis)

    # large 2D scene fields (opt-in). Both are lazily filled, so a run that saved
    # before illuminating would otherwise fail with an opaque TypeError on None.
    if mgs.output_cfg.save_scene_fields:
        if scene.data is None or mgs.gs_rec_data is None:
            raise RuntimeError(
                "output.save_scene_fields is set but the scene has not been "
                "illuminated; call illuminate_real() and illuminate_reconstructed()")
        out["scene_data"] = c64(scene.data)
        out["gs_rec_scene_data"] = c64(mgs.gs_rec_data)

    return out


def _collect_metadata(mgs, run_dir: Path,
                      args_dict: dict, is_exp: bool, npz_keys) -> dict:
    """Build the JSON-safe metadata dict (scalars only).

    `freq_hz`/`wavelength_m` remain, as the PRIMARY (first) frequency, so older
    tooling keeps working; the full lists live in `frequencies_hz`/`wavelengths_m`
    in the same order as every per-frequency array in run.npz.
    """
    scene = mgs.scene
    hist = mgs.gs_history
    gs_cfg = mgs.gs_cfg
    out_cfg = mgs.output_cfg

    meta = {
        "schema_version": 3,
        "run_dir": str(run_dir),
        "is_experimental": bool(is_exp),
        "freq_hz": float(mgs.freqs[0]),
        "wavelength_m": float(mgs.freq_states[0].wavelength),
        "frequencies_hz": [float(f) for f in mgs.freqs],
        "wavelengths_m": [float(fs.wavelength) for fs in mgs.freq_states],
        "n_frequencies": len(mgs.freqs),
        "provenance": provenance(args_dict, plot_filename=Path(mgs.plot_path).name),
        "gerchberg_saxton": {
            "init": gs_cfg.init,
            "max_iters": gs_cfg.max_iters,
            "convergence_count": gs_cfg.convergence_count,
            "convergence_threshold": gs_cfg.convergence_threshold,
            "lr0": gs_cfg.lr0,
            "bt_shrink": gs_cfg.bt_shrink,
            "bt_tries": gs_cfg.bt_tries,
            "seed": (int(hist.seed) if hist is not None else gs_cfg.seed),
            "history_stride": gs_cfg.history_stride,
        },
        "scene": {
            "x_min": scene.x_min, "x_max": scene.x_max,
            "z_min": scene.z_min, "z_max": scene.z_max,
            "spacing": scene.spacing,
            "nx": int(len(scene.x_axis)), "nz": int(len(scene.z_axis)),
            # source plane for re-illumination (TX projects toward -Z from here)
            "tx_z": float(scene.tx_ap.z),
            "has_real_tx": (not is_exp),
        },
        "outputs": {
            "saved_gs_history": bool(out_cfg.save_gs_history and hist is not None
                                     and len(hist.iter_indices) > 0),
            "saved_scene_fields": bool(out_cfg.save_scene_fields),
            "npz_keys": sorted(npz_keys),
        },
    }
    if hist is not None:
        gs_result = getattr(mgs, "gs_result", None)
        meta["gs_result"] = {
            "stop_reason": hist.stop_reason,
            "n_iters_run": hist.n_iters_run,
            "n_iters_captured": int(len(hist.iter_indices)),
            "final_loss": hist.final_loss,
            "final_loss_per_freq": [float(v) for v in hist.final_loss_per_freq],
            # psi's units: the delay profile is expressed as phase at this frequency
            "ref_freq_hz": (float(gs_result.ref_freq) if gs_result is not None else None),
            "rx_z_m": hist.rx_z,
        }
    else:
        meta["gs_result"] = None
    return meta


def save_run(mgs, run_dir: Path, config, config_path: Path,
             args_dict: dict, is_exp: bool) -> None:
    """Persist a completed MGS run into run_dir. Frequencies come from the MGS
    instance itself (mgs.freqs / mgs.freq_states)."""
    arrays = _collect_arrays(mgs)
    np.savez_compressed(run_dir / "run.npz", **arrays)

    meta = _collect_metadata(mgs, run_dir, args_dict, is_exp,
                             list(arrays.keys()))
    write_json(run_dir / "run.json", meta)
    save_config_snapshot(run_dir, config, config_path)

    # copy the scene plot in if it was written
    plot_path = Path(mgs.plot_path)
    if plot_path.exists():
        shutil.copyfile(plot_path, run_dir / plot_path.name)

    mgs.log.info(f"Saved run to {run_dir}")
