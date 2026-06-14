"""Persistence of MGS runs: per-run directory with numeric arrays (.npz),
scalar metadata (.json), config snapshots, and a copy of the scene plot.

Storage split: all numeric arrays (including complex) live in run.npz; only
JSON-safe scalars (no complex, no numpy types) live in run.json.
"""

import json
import logging
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import numpy as np

from rice_bend import __version__


@dataclass
class GSHistory:
    """Accumulates gradient-descent state during run_gerch_sax.

    Per-iteration phase/field are captured on a stride (and always on the final
    iteration). Growing lists are used during the loop so early convergence
    doesn't waste a preallocated (max_iters, size) block; finalize() converts
    them to arrays.
    """
    # one-time fixed arrays (set at construction, before the loop)
    x_axis: np.ndarray
    orig_prop_f: np.ndarray        # target RX field
    orig_aper_amp: np.ndarray      # fixed aperture amplitude
    error_weighting: np.ndarray
    support: np.ndarray            # bool mask
    initial_phase: np.ndarray      # seeded RNG draw
    rx_z: float
    seed: int
    history_stride: int

    # dense per-iteration loss (cheap, full length)
    loss_full: List[float] = field(default_factory=list)
    # strided captures
    iter_indices: List[int] = field(default_factory=list)
    loss_captured: List[float] = field(default_factory=list)
    phase_captured: List[np.ndarray] = field(default_factory=list)
    prop_field_captured: List[np.ndarray] = field(default_factory=list)

    # start/stop conditions (filled at finalize)
    stop_reason: Optional[str] = None
    n_iters_run: int = 0
    final_loss: float = float("nan")

    def _capture(self, iter_idx: int, loss: float, phase: np.ndarray,
                 prop_field: np.ndarray) -> None:
        # avoid duplicate capture (e.g. final iter also landing on the stride)
        if self.iter_indices and self.iter_indices[-1] == iter_idx:
            return
        self.iter_indices.append(int(iter_idx))
        self.loss_captured.append(float(loss))
        self.phase_captured.append(phase.copy())
        self.prop_field_captured.append(prop_field.copy())

    def record_iter(self, iter_idx: int, loss: float, phase: np.ndarray,
                    prop_field: np.ndarray) -> None:
        """Call once per iteration. Always records dense loss; captures
        phase/field on the stride."""
        self.loss_full.append(float(loss))
        if iter_idx % self.history_stride == 0:
            self._capture(iter_idx, loss, phase, prop_field)

    def capture_final(self, iter_idx: int, loss: float, phase: np.ndarray,
                      prop_field: np.ndarray) -> None:
        """Force-capture the last iteration's strided state (no dense loss append)."""
        self._capture(iter_idx, loss, phase, prop_field)

    def finalize(self, stop_reason: str, n_iters_run: int, final_loss: float) -> None:
        self.stop_reason = stop_reason
        self.n_iters_run = int(n_iters_run)
        self.final_loss = float(final_loss)


def make_run_dir(output_base: Path, run_name: Optional[str]) -> Path:
    """Create and return <output_base>/<run_name>/ (run_name defaults to 'data_dump').

    If the directory already exists its contents are cleared (with a warning), so each
    run starts from a clean directory.
    """
    name = run_name or "data_dump"
    run_dir = Path(output_base) / name
    if run_dir.exists():
        logging.getLogger().warning(f"Run directory {run_dir} already exists — clearing its contents")
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _json_safe(obj):
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


def _c64(a: np.ndarray) -> np.ndarray:
    return np.asarray(a, dtype=np.complex64)


def _f64(a: np.ndarray) -> np.ndarray:
    return np.asarray(a, dtype=np.float64)


def _traj_array(traj) -> np.ndarray:
    """[a, b, c] -> float64 (3,); empty list -> float64 (0,)."""
    if traj is None or len(traj) == 0:
        return np.empty(0, dtype=np.float64)
    return np.asarray(traj, dtype=np.float64)


def _collect_arrays(mgs, is_exp: bool) -> dict:
    """Build the dict of arrays for run.npz. Guards every optional array."""
    out = {}
    scene = mgs.scene
    gs_tx = mgs.gs_tx

    # apertures / axes (always)
    out["tx_real_aper_axis"] = _f64(scene.tx_ap.aper_axis)
    out["tx_real_aper_profile"] = _c64(scene.tx_ap.aper_profile)
    out["rx_aper_axis"] = _f64(scene.rx_ap.aper_axis)
    out["rx_aper_profile"] = _c64(scene.rx_ap.aper_profile)
    out["gs_tx_aper_axis"] = _f64(gs_tx.aper_axis)
    out["gs_tx_aper_profile"] = _c64(gs_tx.aper_profile)

    # trajectories (may be empty)
    out["real_traj"] = _traj_array(getattr(mgs, "real_traj", []))
    out["rec_traj"] = _traj_array(getattr(mgs, "rec_traj", []))

    # gradient-descent history
    hist = mgs.gs_history
    if hist is not None:
        out["x_axis"] = _f64(hist.x_axis)
        out["gs_target_rx_field"] = _c64(hist.orig_prop_f)
        out["gs_fixed_aper_amp"] = _f64(hist.orig_aper_amp)
        out["gs_error_weighting"] = _f64(hist.error_weighting)
        out["gs_support"] = np.asarray(hist.support, dtype=bool)
        out["gs_initial_phase"] = _f64(hist.initial_phase)
        out["gs_loss_full"] = np.asarray(hist.loss_full, dtype=np.float32)
        if mgs.output_cfg.save_gs_history and len(hist.iter_indices) > 0:
            out["gs_iter_indices"] = np.asarray(hist.iter_indices, dtype=np.int64)
            out["gs_loss_captured"] = np.asarray(hist.loss_captured, dtype=np.float32)
            out["gs_phase_captured"] = np.asarray(hist.phase_captured, dtype=np.float32)
            out["gs_prop_field_captured"] = _c64(np.asarray(hist.prop_field_captured))

    # scene axes are always saved (cheap, 1D) so a run can be re-illuminated later
    out["scene_x_axis"] = _f64(scene.x_axis)
    out["scene_z_axis"] = _f64(scene.z_axis)

    # large 2D scene fields (opt-in)
    if mgs.output_cfg.save_scene_fields:
        out["scene_data"] = _c64(scene.data)
        out["gs_rec_scene_data"] = _c64(mgs.gs_rec_scene.data)

    return out


def _collect_metadata(mgs, run_dir: Path, config, freq: float,
                      args_dict: dict, is_exp: bool, npz_keys) -> dict:
    """Build the JSON-safe metadata dict (scalars only)."""
    scene = mgs.scene
    hist = mgs.gs_history
    gs_cfg = mgs.gs_cfg
    out_cfg = mgs.output_cfg

    real_traj = list(getattr(mgs, "real_traj", []))
    rec_traj = list(getattr(mgs, "rec_traj", []))

    meta = {
        "schema_version": 1,
        "run_dir": str(run_dir),
        "is_experimental": bool(is_exp),
        "freq_hz": float(freq),
        "wavelength_m": float(getattr(mgs, "wavelength", float("nan"))),
        "provenance": {
            "cli_args": {k: (str(v) if isinstance(v, Path) else v)
                         for k, v in (args_dict or {}).items()},
            "package_version": __version__,
            "plot_filename": Path(mgs.plot_path).name,
        },
        "gerchberg_saxton": {
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
        },
        "trajectory": {
            "has_real_traj": len(real_traj) > 0,
            "real_traj": real_traj if len(real_traj) > 0 else None,
            "rec_traj": rec_traj if len(rec_traj) > 0 else None,
            "rec_traj_computed": len(rec_traj) > 0,
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
        meta["gs_result"] = {
            "stop_reason": hist.stop_reason,
            "n_iters_run": hist.n_iters_run,
            "n_iters_captured": int(len(hist.iter_indices)),
            "final_loss": hist.final_loss,
            "rx_z_m": hist.rx_z,
        }
    else:
        meta["gs_result"] = None
    return meta


def save_run(mgs, run_dir: Path, config, config_path: Path, freq: float,
             args_dict: dict, is_exp: bool) -> None:
    """Persist a completed MGS run into run_dir."""
    arrays = _collect_arrays(mgs, is_exp)
    np.savez_compressed(run_dir / "run.npz", **arrays)

    meta = _collect_metadata(mgs, run_dir, config, freq, args_dict, is_exp,
                             list(arrays.keys()))
    with open(run_dir / "run.json", "w") as f:
        json.dump(meta, f, indent=2, default=_json_safe)

    # config snapshots: effective (defaults filled) + raw source (preserves comments)
    with open(run_dir / "config_snapshot.json", "w") as f:
        json.dump(config.model_dump(mode="json"), f, indent=2, default=_json_safe)
    try:
        shutil.copyfile(config_path, run_dir / "config_source.yml")
    except (OSError, TypeError):
        pass

    # copy the scene plot in if it was written
    plot_path = Path(mgs.plot_path)
    if plot_path.exists():
        shutil.copyfile(plot_path, run_dir / plot_path.name)

    mgs.log.info(f"Saved run to {run_dir}")
