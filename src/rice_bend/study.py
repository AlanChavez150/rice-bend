"""Parameter studies (`mgs-study`): run a code-defined series of grid searches
varying one parameter, measure each point's localization error via analysis.json,
and plot error against the study's x-axis.

The base config still says what the underlying experiment IS (scene, beam, RX,
grid, GS); a study is a code-defined SERIES over it — each point a programmatic
derivation (a comb formula; a rigid TX translation touching three coupled
fields), with the x-axis semantics and plotting in code. The series tables below
are versioned next to the code that interprets them rather than in a
one-consumer yaml DSL.

Layout: <study root>/study.json (the ownership marker, written FIRST and updated
after every point, so an interrupted study resumes), configs/point_*.yml (the
materialized effective config per point — passing it to persistence makes each
point dir's config_source.yml reconstruct the derived experiment, not the base),
one ordinary grid run dir per point, and the study plot. A point whose
candidate_beams.json already exists skips its solve on re-run — so a root whose
saved points were solved with different noise is refused up front, not reused.
"""

import argparse
import json
import math
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import numpy as np
import yaml
from pydantic import ValidationError

from rice_bend.analysis import analysis_from_manifest, write_analysis
from rice_bend.cli import setup_logging
from rice_bend.config import (NoiseConfig, SimConfig, apply_noise_overrides,
                              load_config, parse_snr_db_arg, resolve_frequencies)
from rice_bend.data_store import check_run_dir, make_run_dir, write_json
from rice_bend.grid_search import persist_and_plot
from rice_bend.grid_sweep import run_grid_search
from rice_bend.plotting import add_wavelength_axis
from rice_bend.residual_plots import snr_suffix, summary_from_manifest

STUDY_SCHEMA_VERSION = 6   # v4: points[] gained ref_freq_hz (nullable)
#                            v5: top-level noise {snr_db, seed} (null when noiseless);
#                                points[] gained snr_db (nullable)
#                            v6: noise gained swept (true: the series sets each point's
#                                SNR, so the top-level snr_db is null; see points[])
STUDY_NAME_FILE = "study.json"
STUDY_DEFAULT_CONFIG = (Path(__file__).resolve().parents[2] / "configs"
                        / "scenario_caustic_hit_pm5.yml")

FREQ_CENTER_HZ = 150e9
# bandwidth half-widths (+/- % of the center) — dense at the small end where the
# error curve moves fastest; 0 is the no-diversity anchor (a single tone)
BANDWIDTH_PCTS = (0.0, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 13.0, 16.0, 20.0)
# the fixed comb the tx_shift study solves with (the pm5 comparison comb)
PM5_COMB_HZ = (142.5e9, 150.0e9, 157.5e9)
# rigid TX translations (m): dense at small s where the captured energy falls
# fastest; 0.115 reproduces configs/scenario_caustic_miss.yml exactly
TX_SHIFTS_M = (0.0, 0.010, 0.020, 0.030, 0.045, 0.065, 0.090, 0.115, 0.130)
# peak-referenced receiver SNRs (dB): every 5 dB from 0 through 40
SNR_DBS = (0.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0)
# the fine series: every 1 dB from 0 through 30 (a separate table, so roots of
# the 5 dB series still replot against the values they were solved with)
SNR_DBS_FINE = tuple(float(s) for s in range(0, 31))


def _apply_frequency(cfg: SimConfig, pct: float) -> SimConfig:
    c = cfg.model_copy(deep=True)
    if pct == 0:
        # single tone, not three copies of the center (the config layer
        # rejects duplicate frequencies): the 1f anchor of the sweep
        c.frequencies = [FREQ_CENTER_HZ]
    else:
        c.frequencies = [FREQ_CENTER_HZ * (1 - pct / 100.0), FREQ_CENTER_HZ,
                         FREQ_CENTER_HZ * (1 + pct / 100.0)]
    return c


def _apply_tx_shift(cfg: SimConfig, s: float) -> SimConfig:
    """Rigid TX translation: the window AND the caustic trajectory's lateral
    offset move together (a plate is a physical object — the miss scenario is
    exactly this with s = +0.115). Shifting only the window would slide it off
    the plate: the shipped window is exactly the caustic's native support, so
    the window would cut off one end of the plate and sample dark (zero-filled)
    plate at the other."""
    c = cfg.model_copy(deep=True)
    c.frequencies = list(PM5_COMB_HZ)   # fixed comb: energy is the only variable
    c.tx_aperture.x_min += s
    c.tx_aperture.x_max += s
    traj = list(c.tx_aperture.beam.trajectory)
    traj[2] += s
    c.tx_aperture.beam.trajectory = traj
    return c


def _apply_snr(cfg: SimConfig, snr_db: float) -> SimConfig:
    """Receiver SNR only: the frequencies are whatever the base config lists and
    the TX stays put. Rebuilt through NoiseConfig (not attribute assignment) so
    validation runs; the seed is the base config's, which _resolve_noise_seed has
    already fixed, so every point draws from the same seed."""
    c = cfg.model_copy(deep=True)
    c.noise = NoiseConfig(snr_db=snr_db, seed=cfg.noise.seed)
    return c


@dataclass(frozen=True)
class StudyDef:
    name: str
    param_name: str
    param_unit: str
    x_axis: str                # 'param_value' | 'energy_pct'
    x_label: str
    values: Tuple[float, ...]
    apply: Callable[[SimConfig, float], SimConfig]
    tag: Callable[[float], str]
    plot_name: str
    plot_argmin: bool = True   # include the argmin series on the study plot
    # apply() sets each point's noise: the base config's snr_db is overridden,
    # so there is no single study-wide SNR to guard, suffix or put in the root name
    sweeps_noise: bool = False
    ticks_at_values: bool = False   # x ticks exactly on the swept values


STUDIES = {
    "frequency": StudyDef(
        name="frequency", param_name="bandwidth_pct", param_unit="%",
        x_axis="param_value", x_label="bandwidth (± % of 150 GHz)",
        values=BANDWIDTH_PCTS, apply=_apply_frequency,
        tag=lambda p: f"bw{int(round(p)):02d}pct",
        plot_name="study_error_vs_bandwidth.png"),
    "tx_shift": StudyDef(
        name="tx_shift", param_name="tx_shift_m", param_unit="m",
        x_axis="energy_pct", x_label="energy received in RX window (%, mean over comb)",
        values=TX_SHIFTS_M, apply=_apply_tx_shift,
        tag=lambda s: f"s{int(round(s * 1000)):03d}mm",
        plot_name="study_error_vs_energy.png"),
    "snr": StudyDef(
        name="snr", param_name="snr_db", param_unit="dB",
        x_axis="param_value", x_label="SNR (dB, re peak RX-element power)",
        values=SNR_DBS, apply=_apply_snr,
        tag=lambda s: f"snr{int(round(s)):02d}dB",
        plot_name="study_error_vs_snr.png",
        sweeps_noise=True, ticks_at_values=True),
    "snr_fine": StudyDef(
        name="snr_fine", param_name="snr_db", param_unit="dB",
        x_axis="param_value", x_label="SNR (dB, re peak RX-element power)",
        values=SNR_DBS_FINE, apply=_apply_snr,
        tag=lambda s: f"snr{int(round(s)):02d}dB",
        plot_name="study_error_vs_snr.png",
        sweeps_noise=True),   # 31 values: default ticks, not one per point
}


def _record(study: StudyDef, value: float, point_name: str, analysis: dict,
            resumed: bool, ref_freq_hz: Optional[float],
            noise: Optional[dict]) -> dict:
    """One study.json points[] entry, from the point's analysis dict."""
    argmin = analysis.get("argmin")
    err_m = argmin["error_distance_m"] if argmin else None
    energy = analysis.get("energy") or {}
    mean_frac = energy.get("mean_fraction")
    top_mean = analysis.get("top_mean_dist_to_truth_m")
    n_dof = analysis.get("n_dof") or {}
    rec = {
        "point": point_name,
        "param_value": float(value),
        "run_dir": point_name,
        "resumed": bool(resumed),
        "ref_freq_hz": float(ref_freq_hz) if ref_freq_hz is not None else None,
        "snr_db": noise.get("snr_db") if noise else None,
        "error_distance_m": err_m,
        "error_mm": 1000.0 * err_m if err_m is not None else None,
        "argmin": ({"z": argmin["z"], "x_center": argmin["x_center"],
                    "final_loss": argmin["final_loss"]} if argmin else None),
        "top_mean_dist_m": top_mean,
        "top_mean_dist_mm": 1000.0 * top_mean if top_mean is not None else None,
        "n_top_candidates": analysis.get("n_top_candidates"),
        "energy_pct": 100.0 * mean_frac if mean_frac is not None else None,
        "n_dof_total": n_dof.get("total"),
        "n_dof_per_freq": ([p.get("n_dof") for p in n_dof.get("per_freq", [])]
                           if n_dof else None),
    }
    rec["x_value"] = rec["energy_pct"] if study.x_axis == "energy_pct" else rec["param_value"]
    return rec


def _recompute_analysis(point_dir: Path,
                        log) -> Tuple[dict, Optional[float], Optional[dict]]:
    """Always recompute (cheap, and the run dir's analysis.json stays fresh).
    Also hands back the manifest's centre frequency for the record's λ axis, and
    its noise block (None when noiseless) for the record's snr_db."""
    summary = summary_from_manifest(point_dir)
    analysis = analysis_from_manifest(point_dir, summary)
    write_analysis(point_dir, analysis)
    return analysis, summary.ref_freq_hz, summary.noise


def _warn_if_stale(study: StudyDef, base_cfg: SimConfig, value: float,
                   point_dir: Path, log) -> None:
    """A resumed point must match the study table's current value — warn when a
    table edit between runs would silently pair an old run with a new label."""
    try:
        expected = study.apply(base_cfg, value)
        with open(point_dir / "candidate_beams.json") as f:
            m = json.load(f)
        got = [e["freq_hz"] for e in m.get("frequencies", [])]
        want = [float(fq) for fq in expected.frequencies]
        if len(got) != len(want) or not np.allclose(got, want):
            log.warning(f"{point_dir.name}: resumed run's frequencies {got} != the "
                        f"study table's {want} — was the series edited?")
        snap = point_dir / "config_snapshot.json"
        if snap.exists():
            with open(snap) as f:
                s = json.load(f)
            gx = s.get("tx_aperture", {}).get("x_min")
            if gx is not None and abs(gx - expected.tx_aperture.x_min) > 1e-9:
                log.warning(f"{point_dir.name}: resumed run's tx_aperture.x_min {gx} "
                            f"!= expected {expected.tx_aperture.x_min} — was the "
                            "series edited?")
    except Exception as e:                                  # cross-check only
        log.warning(f"{point_dir.name}: could not cross-check the resumed run ({e})")


def _point_noise(cfg: SimConfig) -> Optional[dict]:
    """The noise one point is solved with, in the manifest's {snr_db, seed} shape."""
    if cfg.noise.snr_db is None:
        return None
    return {"snr_db": cfg.noise.snr_db, "seed": cfg.noise.seed}


def _noise_block(study: StudyDef, cfg: SimConfig) -> Optional[dict]:
    """study.json's top-level noise record. A non-sweeping series solves every
    point with the base config's noise; a sweeping one shares only the seed (the
    per-point SNRs live in points[])."""
    if study.sweeps_noise:
        return {"snr_db": None, "seed": cfg.noise.seed, "swept": True}
    nz = _point_noise(cfg)
    if nz is not None:
        nz["swept"] = False
    return nz


def _describe_noise(noise: Optional[dict]) -> str:
    if not noise or noise.get("snr_db") is None:
        return "no noise"
    return f"SNR {noise['snr_db']:g} dB, seed {noise.get('seed')}"


def _same_snr(a: Optional[float], b: Optional[float]) -> bool:
    if a is None or b is None:
        return a is None and b is None
    return math.isclose(a, b, rel_tol=0.0, abs_tol=1e-9)


def _resolve_noise_seed(study: StudyDef, base_cfg: SimConfig, root: Path, log) -> None:
    """Fix a null noise seed ONCE for the whole study. Every point deep-copies
    base_cfg, so left null each point would draw its own seed, and the
    configs/point_*.yml written before the solve would record null. A resume
    adopts the seed the root's study.json recorded for the same SNR (for a
    series that sweeps the SNR: for the same series) — otherwise the pre-flight
    guard would refuse every re-run of a null-seed study."""
    nz = base_cfg.noise
    # a sweeping series is noisy at every point even when the base snr_db is null
    if nz.seed is not None or (nz.snr_db is None and not study.sweeps_noise):
        return
    seed = None
    marker = root / STUDY_NAME_FILE
    if marker.exists():
        try:
            with open(marker) as f:
                data = json.load(f)
            saved = data.get("noise") if data.get("study") == study.name else None
        except (OSError, ValueError, AttributeError):
            saved = None
        same = (bool(saved) and (study.sweeps_noise
                                 or _same_snr(saved.get("snr_db"), nz.snr_db)))
        if same and saved.get("seed") is not None:
            seed = int(saved["seed"])
            log.info(f"noise.seed is null: adopting seed {seed} from {marker}")
    if seed is None:
        seed = int(np.random.SeedSequence().entropy % (2**32))
        log.info(f"noise.seed is null: drew seed {seed} for every point of the study")
    base_cfg.noise = NoiseConfig(snr_db=nz.snr_db, seed=seed)


def _check_resume_noise(study: StudyDef, base_cfg: SimConfig, root: Path, log) -> None:
    """Pre-flight: refuse a root whose saved points were solved with different
    noise. The resume cache key is only the point dir name, so without this a
    20 dB re-run over a 10 dB root would silently mix the two. Runs BEFORE
    study.json is rewritten and before any solve, so a refusal modifies nothing
    (and stays out of _warn_if_stale, whose broad except would demote it).
    The expectation is per point — what apply() derives — since a sweeping
    series gives every point its own SNR."""
    for i, value in enumerate(study.values):
        name = f"point_{i:02d}_{study.tag(value)}"
        manifest = root / name / "candidate_beams.json"
        if not manifest.exists():
            continue
        want = _point_noise(study.apply(base_cfg, value))
        with open(manifest) as f:
            got = json.load(f).get("noise")         # absent (pre-noise) or null: off
        if want is None:
            ok = not got or got.get("snr_db") is None
        else:
            ok = (bool(got) and _same_snr(got.get("snr_db"), want["snr_db"])
                  and got.get("seed") == want["seed"])
        if not ok:
            log.error(f"{root / name} was solved with {_describe_noise(got)}, but this "
                      f"invocation asks for {_describe_noise(want)}; resuming would mix "
                      "the two. Pass a different --out (or matching "
                      + ("--noise-seed)." if study.sweeps_noise
                         else "--snr-db/--noise-seed)."))
            raise SystemExit(2)


def _repair_partial(point_dir: Path, log) -> None:
    """Clear an interrupted persist: a dir with sweep artifacts but no manifest.

    The fingerprint is deliberately narrow — measurement.npz / candidates/ are
    written ONLY by save_grid_run — and anything carrying another tool's marker
    (run.json, frequencies.json, candidate_beams.json) is left for
    check_run_dir's ownership refusal rather than deleted here."""
    ours = ((point_dir / "measurement.npz").exists()
            or (point_dir / "candidates").is_dir())
    foreign = any((point_dir / m).exists()
                  for m in ("run.json", "frequencies.json", "candidate_beams.json"))
    if ours and not foreign:
        log.warning(f"{point_dir.name}: interrupted persist (no manifest) — "
                    "clearing for a re-run")
        shutil.rmtree(point_dir)


def _warn_bounds(cfg: SimConfig, name: str, log) -> None:
    gs = cfg.grid_search
    cx = 0.5 * (cfg.tx_aperture.x_min + cfg.tx_aperture.x_max)
    if gs is not None and not (gs.x_center.min <= cx <= gs.x_center.max):
        log.warning(f"{name}: true TX x_center {cx:+.3f} lies outside the candidate "
                    f"grid [{gs.x_center.min}, {gs.x_center.max}] — the argmin error "
                    "is floor-bounded by the grid edge")
    sc = cfg.sim_scene
    if cfg.tx_aperture.x_min < sc.x_min or cfg.tx_aperture.x_max > sc.x_max:
        log.warning(f"{name}: TX window [{cfg.tx_aperture.x_min:.3f}, "
                    f"{cfg.tx_aperture.x_max:.3f}] leaves scene x "
                    f"[{sc.x_min}, {sc.x_max}]")


def _plot_namespace(cfg_path: Path, args) -> argparse.Namespace:
    """The argparse surface persist_and_plot expects. Scene/anim extras stay off:
    a study's per-point dirs keep the cheap plot set."""
    return argparse.Namespace(
        config=cfg_path, out=None, freq=None, limit=args.limit, jobs=args.jobs,
        dry_run=False, replot=None, true_mgs=False, scenes=False, scene_top=None,
        scene_freq=None, anim=False, surface_anim=False, debug=args.debug,
        study=getattr(args, "study", None), snr_db=getattr(args, "snr_db", None),
        noise_seed=getattr(args, "noise_seed", None))


def _run_point(study: StudyDef, i: int, value: float, base_cfg: SimConfig,
               root: Path, args, log) -> dict:
    n = len(study.values)
    name = f"point_{i:02d}_{study.tag(value)}"
    point_dir = root / name

    if (point_dir / "candidate_beams.json").exists():
        log.info(f"[{i + 1}/{n}] {name}: candidate_beams.json present — "
                 "skipping the solve")
        _warn_if_stale(study, base_cfg, value, point_dir, log)
        analysis, ref, noise = _recompute_analysis(point_dir, log)
        return _record(study, value, name, analysis, resumed=True, ref_freq_hz=ref,
                       noise=noise)
    if point_dir.exists():
        _repair_partial(point_dir, log)

    cfg = study.apply(base_cfg, value)     # deep copy inside apply — mandatory,
    #   run_grid_search mutates cfg.gerchberg_saxton (sweep overrides)
    cfg.output.output_dir = root
    cfg.output.run_name = name
    _warn_bounds(cfg, name, log)
    check_run_dir(root, name, kind="grid")

    cfg_path = root / "configs" / f"{name}.yml"
    with open(cfg_path, "w") as f:
        yaml.safe_dump(cfg.model_dump(mode="json"), f, sort_keys=False)

    freqs = resolve_frequencies(cfg, None)
    log.info(f"[{i + 1}/{n}] {name}: {study.param_name}={value:g}{study.param_unit} "
             f"({', '.join(f'{fq / 1e9:g}' for fq in freqs)} GHz)")
    run = run_grid_search(cfg, freqs, limit=args.limit, jobs=args.jobs, log=log)
    out_dir = make_run_dir(root, name, kind="grid")
    analysis = persist_and_plot(run, out_dir, cfg, _plot_namespace(cfg_path, args), log)
    return _record(study, value, name, analysis, resumed=False,
                   ref_freq_hz=run.ref_freq, noise=run.noise)


def _write_study_json(root: Path, study: StudyDef, base_config, records: List[dict],
                      status: str, noise: Optional[dict]) -> None:
    write_json(root / STUDY_NAME_FILE, {
        "schema_version": STUDY_SCHEMA_VERSION,
        "study": study.name,
        "status": status,
        "base_config": str(base_config),
        "noise": noise,
        "param_name": study.param_name,
        "param_unit": study.param_unit,
        "x_axis": study.x_axis,
        "x_label": study.x_label,
        "values": [float(v) for v in study.values],
        "points": records,
    })


# the two plotted error series: record key, legend label, color, marker, then the
# draw weight — line width, line-plot marker size (pt), scatter area (pt²), zorder.
# The top-K mean is a slightly heavier UNDERLAY: where the top set has a single
# member (n=1) its mean IS the argmin, and drawn at equal weight on top it hid the
# argmin entirely. Beneath it, a shared point reads as a disc centred in a larger
# square and a shared segment as a line with an edge on both sides.
_ERROR_SERIES = (
    ("error_mm", "argmin to true TX", "C0", "o", 1.5, 6.0, 45, 3),
    ("top_mean_dist_mm", "top candidates mean to true TX", "C1", "s", 3.0, 9.0, 100, 2),
)


def _records_ref_freq(records: List[dict]) -> Optional[float]:
    """The one centre frequency shared by every point, or None (no λ axis).
    Today's study tables agree bit-exactly (150 GHz everywhere); a series that
    varied the centre would have no single wavelength to offer."""
    refs = {r.get("ref_freq_hz") for r in records} - {None}
    return refs.pop() if len(refs) == 1 else None


def _records_snr(records: List[dict]) -> Optional[float]:
    """The one SNR shared by every point, or None (noiseless, or a mixed root —
    which the resume guard refuses, so only a hand-assembled study gets here)."""
    snrs = {r.get("snr_db") for r in records}
    return snrs.pop() if len(snrs) == 1 else None


def _plot_study(study: StudyDef, records: List[dict], out_path: Path, log) -> None:
    """Argmin error and top-K mean distance vs the study's x-axis. Points
    connect in SERIES order (= table order), which matters for the energy axis:
    x there is a measured quantity."""
    import matplotlib.pyplot as plt
    from matplotlib import patheffects

    plotted = [r for r in records if r.get("x_value") is not None]
    fig, ax = plt.subplots(figsize=(9, 6), layout="constrained")
    drew = False
    for key, label, color, marker, lw, ms, _area, zorder in _ERROR_SERIES:
        if key == "error_mm" and not study.plot_argmin:
            continue
        pts = [(r["x_value"], r[key], r) for r in plotted if r.get(key) is not None]
        if not pts:
            continue
        drew = True
        ax.plot([p[0] for p in pts], [p[1] for p in pts], marker=marker,
                color=color, label=label, lw=lw, ms=ms, zorder=zorder)
        if key == "top_mean_dist_mm":
            # the adaptive set size is part of the signal — annotate it, offset
            # clear of the underlay's larger square and haloed: a line through
            # the label (its own heavier orange one included) would swallow it
            halo = [patheffects.withStroke(linewidth=3, foreground="white")]
            for x, y, r in pts:
                if r.get("n_top_candidates"):
                    ax.annotate(f"n={r['n_top_candidates']}", (x, y),
                                textcoords="offset points", xytext=(6, -12),
                                fontsize=7, alpha=0.8, color="C1",
                                path_effects=halo)
    if drew:
        if study.x_axis == "energy_pct":
            # x is measured, not the swept parameter — label each point with it
            for r in plotted:
                if r.get("error_mm") is None:
                    continue
                ax.annotate(f"{r['param_value'] * 1000:.0f} mm",
                            (r["x_value"], r["error_mm"]),
                            textcoords="offset points", xytext=(6, 6), fontsize=8,
                            alpha=0.8)
        # outside, below the x label: with both series drawn no corner of the
        # axes is reliably empty, and an in-axes legend covered data
        fig.legend(*ax.get_legend_handles_labels(), loc="outside lower center",
                   ncol=2, framealpha=0.9)
    else:
        ax.text(0.5, 0.5, "no measurable points", ha="center", va="center",
                transform=ax.transAxes)
    # room right of the last point for its n= label: up to "n=1393" needs ~36 pt
    # (offset + text + halo); the default 5 % margin leaves ~26 pt
    ax.margins(x=0.09)
    if study.ticks_at_values:
        ax.set_xticks([float(v) for v in study.values])
    ax.set_xlabel(study.x_label)
    ax.set_ylabel("distance to true TX (mm)")
    add_wavelength_axis(ax, _records_ref_freq(records), axis="y", unit_m=1e-3)
    ax.set_title(f"{study.name} study" + snr_suffix({"snr_db": _records_snr(records)}))
    ax.grid(True, alpha=0.3)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    log.info(f"Wrote study plot to {out_path}")


def _plot_ndof(study: StudyDef, records: List[dict], out_path: Path, log) -> None:
    """N_E(eps) on the x-axis against the two localization errors (argmin and
    top-K mean distance). This is the information-metric view (see
    docs/information_metric.md): whatever knob the study turned, points with the
    same mode count should behave alike."""
    import matplotlib.pyplot as plt

    pts = [r for r in records if r.get("n_dof_total") is not None]
    fig, ax = plt.subplots(figsize=(9, 6), layout="constrained")
    drew = False
    for key, label, color, marker, _lw, _ms, area, zorder in _ERROR_SERIES:
        keep = [(r["n_dof_total"], r[key], r) for r in pts if r.get(key) is not None]
        if not keep:
            continue
        drew = True
        ax.scatter([k[0] for k in keep], [k[1] for k in keep],
                   s=area, color=color, marker=marker, zorder=zorder, label=label)
        if key == "error_mm":
            # one annotation per point (the series share their x positions)
            for x, y, r in keep:
                ax.annotate(f"{r['param_value']:g}", (x, y),
                            textcoords="offset points", xytext=(6, 6),
                            fontsize=8, alpha=0.8)
    if drew:
        # outside, below the x label, as on the study plot
        fig.legend(*ax.get_legend_handles_labels(), loc="outside lower center",
                   ncol=2, framealpha=0.9)
    else:
        ax.text(0.5, 0.5, "no n_dof-measurable points", ha="center", va="center",
                transform=ax.transAxes)
    ax.set_xlabel("N_E(eps=0.1), summed over the comb")
    ax.set_ylabel("distance to true TX (mm)")
    add_wavelength_axis(ax, _records_ref_freq(records), axis="y", unit_m=1e-3)
    ax.grid(True, alpha=0.3)
    fig.suptitle(f"{study.name} study — information metric view "
                 f"(points labelled by {study.param_name})"
                 + snr_suffix({"snr_db": _records_snr(records)}))
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    log.info(f"Wrote n_dof plot to {out_path}")


def _replot_study(root: Path, args, log) -> None:
    """Rebuild study.json + the study plot from the per-point run dirs (each
    point's analysis is recomputed; nothing is re-solved)."""
    with open(root / STUDY_NAME_FILE) as f:
        data = json.load(f)
    study = STUDIES[data["study"]]
    records: List[dict] = []
    complete = True
    for i, value in enumerate(study.values):
        name = f"point_{i:02d}_{study.tag(value)}"
        pdir = root / name
        if not (pdir / "candidate_beams.json").exists():
            log.warning(f"{name}: missing — skipped")
            complete = False
            continue
        analysis, ref, noise = _recompute_analysis(pdir, log)
        records.append(_record(study, value, name, analysis,
                               resumed=True, ref_freq_hz=ref, noise=noise))
    # no config here: carry the saved noise block through, or a replot drops it
    _write_study_json(root, study, data.get("base_config", "?"), records,
                      status="complete" if complete else "partial",
                      noise=data.get("noise"))
    _plot_study(study, records, root / study.plot_name, log)
    _plot_ndof(study, records, root / "study_metrics_vs_ndof.png", log)


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Run a parameter study: a series of grid searches varying one "
                    "parameter, measuring localization error per point")
    parser.add_argument("--study", choices=sorted(STUDIES), default=None,
                        help="Which study series to run (required unless --replot)")
    parser.add_argument("--config", type=Path, default=STUDY_DEFAULT_CONFIG,
                        help="Base simulation config the series derives from "
                             f"(default: {STUDY_DEFAULT_CONFIG.name})")
    parser.add_argument("--out", type=Path, default=None,
                        help="Study root directory (default: "
                             "<output.output_dir>/study_<study>, plus _snr<DB>dB "
                             "with noise on, except for --study snr)")
    parser.add_argument("--jobs", "-j", type=int, default=None,
                        help="Worker processes per point (default: all cores)")
    parser.add_argument("--limit", type=int, default=None,
                        help="Per-point candidate cap (smoke tests)")
    parser.add_argument("--engine", choices=["python", "rust"], default=None,
                        help="Solver engine (overrides gerchberg_saxton.engine; default: "
                             "the config's, python unless set). 'rust' needs the "
                             "rice_bend_core extension (rust/README.md).")
    parser.add_argument("--snr-db", type=parse_snr_db_arg, default=None, metavar="DB|off",
                        help="Peak-referenced SNR in dB of the complex AWGN added to the "
                             "simulated RX measurement at every point (overrides "
                             "noise.snr_db); 'off' forces a noiseless study. Not "
                             "with --study snr, which sets every point's SNR")
    parser.add_argument("--noise-seed", type=int, default=None,
                        help="Override noise.seed for a reproducible noise draw (one "
                             "seed for every point)")
    parser.add_argument("--replot", type=Path, default=None,
                        help="Rebuild study.json + the study plot from a saved "
                             "study dir (recomputes each point's analysis; no solving)")
    parser.add_argument("--debug", action="store_true", help="Enable debug logs")
    args = parser.parse_args()
    if args.replot is None and args.study is None:
        parser.error("--study is required (or pass --replot STUDY_DIR)")
    if args.jobs is None:
        import os
        args.jobs = os.cpu_count()
    return args


def main():
    args = _parse_args()
    log = setup_logging(args.debug)

    if args.replot is not None:
        if args.snr_db is not None or args.noise_seed is not None:
            log.warning("--snr-db/--noise-seed are ignored with --replot: the replot "
                        "rebuilds from the saved runs and the noise they recorded")
        _replot_study(Path(args.replot), args, log)
        return

    study = STUDIES[args.study]
    if study.sweeps_noise and args.snr_db is not None:
        # refused before anything is loaded or written: the flag would be
        # silently overridden at every point
        log.error(f"--snr-db cannot be combined with --study {study.name}: the "
                  f"series sets each point's SNR ({', '.join(f'{v:g}' for v in study.values)} "
                  "dB). --noise-seed is allowed and applies to every point.")
        raise SystemExit(2)
    base_cfg = load_config(args.config)
    if args.engine is not None:
        base_cfg.gerchberg_saxton.engine = args.engine
    if study.sweeps_noise and base_cfg.noise.snr_db is not None:
        log.info(f"{args.config} sets noise.snr_db={base_cfg.noise.snr_db:g}; the "
                 f"{study.name} series overrides it at every point")
    try:
        apply_noise_overrides(base_cfg, args.snr_db, args.noise_seed)
    except ValidationError as e:
        log.error("invalid noise override: " + "; ".join(
            f"{'.'.join(str(p) for p in err['loc'])}: {err['msg']}" for err in e.errors()))
        raise SystemExit(2)
    snr = None if study.sweeps_noise else base_cfg.noise.snr_db
    # noisy and noiseless studies of the same series must not share a default root
    # (a series sweeping the SNR has no one SNR to name it by)
    root = (Path(args.out) if args.out is not None
            else Path(base_cfg.output.output_dir)
            / (f"study_{study.name}" + (f"_snr{snr:g}dB" if snr is not None else "")))

    # marker-first initialization: study.json lands before any other content, so
    # an interrupt at any point leaves a resumable root, never a refused one. The
    # seed and the resume guard come first: both read the root as it was left.
    check_run_dir(root.parent, root.name, kind="study")
    _resolve_noise_seed(study, base_cfg, root, log)
    _check_resume_noise(study, base_cfg, root, log)
    noise = _noise_block(study, base_cfg)
    if study.sweeps_noise:
        log.info(f"Each point is solved at its own SNR "
                 f"({', '.join(f'{v:g}' for v in study.values)} dB), seed {noise['seed']}")
    elif noise is not None:
        log.info(f"Every point is solved at {_describe_noise(noise)}")
    root.mkdir(parents=True, exist_ok=True)
    _write_study_json(root, study, args.config, [], status="running", noise=noise)
    (root / "configs").mkdir(exist_ok=True)

    records: List[dict] = []
    for i, value in enumerate(study.values):
        records.append(_run_point(study, i, value, base_cfg, root, args, log))
        _write_study_json(root, study, args.config, records, status="running",
                          noise=noise)

    _write_study_json(root, study, args.config, records, status="complete",
                      noise=noise)
    _plot_study(study, records, root / study.plot_name, log)
    _plot_ndof(study, records, root / "study_metrics_vs_ndof.png", log)
    n_ok = sum(1 for r in records if r.get("error_mm") is not None)
    log.info(f"Study '{study.name}' complete: {n_ok}/{len(records)} measurable "
             f"point(s) -> {root}")


if __name__ == "__main__":
    main()
