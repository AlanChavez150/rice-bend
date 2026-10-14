#!/usr/bin/env python
"""Which points of a bandwidth study are SOLVER-limited: one single MGS solve at the
KNOWN TX location per comb of the `frequency` study, its final loss set against the
loss of the true plate.

A grid search can only rank the true location first if the solve AT the true location
gets down to what the true plate scores. The multi-tone joint solve does not always.
It has to recover the plate's ABSOLUTE delay profile, and whether it does depends on
where it starts (gerchberg_saxton.init, which this script takes from the config as it
stands). The legacy `warm_start` unwraps a random-start solve of the reference
frequency and scans the absolute offset over 2*pi*f_ref/df_min: that range is not a
period of the joint loss unless f_ref/df is an integer, and the reference solve is not
unique where part of the beam misses the RX window, so at some bandwidths the descent
starts outside the true plate's basin and converges well above the true-plate loss.
`backprop` starts from the back-propagated measurement and scans the true period
(rice_bend.mgs.comb_period). (A single tone cannot fail this way -- profiles that
differ by 2*pi anywhere are the same field to it.) A study point where the solve stops
short measures the solver, not the physics, and this table says which ones they are.

For each pct in rice_bend.study.BANDWIDTH_PCTS the tones are the study's own comb
about the config's centre frequency (study.comb, study.base_center_freq), solved the
way `mgs` solves them (MGS.run_gerch_sax on a copy of the config; no scene
illumination, no plot). The losses are MGS.landing_report()["loss"].

The ratio is a basin test only when the true plate's loss is a floor the solver can be
held to. On a NOISELESS run with an on-grid receiver (rx_aperture.grid_stride) it is
not: the target is then the true field at the elements, the true plate scores almost
zero, and every solve sits far above it however good it is (see _ratio_is_basin_test).
Such a run gets its ratios written and `n/a` in the flag column.

The solve runs under the config's gerchberg_saxton settings as they stand (start,
iteration cap, stop threshold). For a config with no grid_search.gs_overrides — the
three scenario_*_hit_1f_grid2l campaign configs have none — those are also the solver
settings of the sweep's one solve per candidate, so the table says how far a solve
under the sweep's settings gets at the true location (which is itself not a grid
point).

Usage:
    python scripts/true_tx_table.py CONFIG.yml --out FILE.md \
        [--snr-db DB|off] [--noise-seed N] [--max-iters N]

Writes the Markdown table to FILE.md, the same rows as a JSON list to FILE.json beside
it, and prints the table. A null gerchberg_saxton.seed is pinned (to grid_search.seed,
else 0) and a null noise seed is drawn once, so every row shares both.
"""
import argparse
import time
from pathlib import Path

import numpy as np
from pydantic import ValidationError

from rice_bend import study
from rice_bend.cli import setup_logging
from rice_bend.config import apply_noise_overrides, load_config, parse_snr_db_arg
from rice_bend.data_store import write_json
from rice_bend.mgs import MGS

# a solve this far above the true plate's loss did not reach the true plate's basin
SOLVER_LIMITED_RATIO = 2.0
SOLVER_LIMITED_FLAG = "solver-limited"
# the flag column of a run whose ratio is not a basin test (_ratio_is_basin_test)
NOT_APPLICABLE_FLAG = "n/a"

COLUMNS = ("bandwidth (± %)", "tones (GHz)", "solver loss", "true-plate loss", "ratio",
           "zero-model loss", "iterations", "stop reason", "flag")
ALIGN = ("---:", ":---", "---:", "---:", "---:", "---:", "---:", ":---", ":---")


def _ratio_is_basin_test(cfg) -> bool:
    """Whether solver loss / true-plate loss says anything about the basin.

    The test rests on the true plate NOT scoring zero: with an interpolated
    receiver its loss is the element sampling it cannot reproduce, with noise the
    noise it cannot fit, and a solve in the right basin gets down to about that. A
    NOISELESS run with an on-grid receiver (rx_aperture.grid_stride) has neither.
    Its target IS the true field at the elements, so the true plate is left with
    the gap between the solver's amplitude constraint and the magnitude of the
    plate's resampling — 4.6e-8 to 6.6e-8 on the three scenario_*_hit_1f_grid2l
    configs — while the descent stops far above it, on the flatness test or the
    iteration cap. Under the -1e-8 stop threshold those configs had before, that
    was at 4e-6 to 1e-5: ratios of 90 to 170 on solves that matched the true field
    to 1-2 % rel-L2. Under their current 3000 cap and -1e-10 threshold the ratios
    run from 1.6 to 1934 (MGS.landing_report has the measurements). A threshold on
    that ratio would flag rows whatever their basin, so there the flag is not
    computed at all.
    """
    return not (cfg.noise.snr_db is None and cfg.rx_aperture.grid_stride is not None)


def _solve_row(cfg, pct: float, center_hz: float) -> dict:
    """One bandwidth point: the joint solve at the true TX and its loss block."""
    freqs = study.comb(center_hz, pct)
    mgs = MGS(freqs, cfg.model_copy(deep=True))
    mgs.run_gerch_sax()
    loss = mgs.landing_report()["loss"]
    ratio = loss["ratio"]
    limited = ratio is not None and ratio > SOLVER_LIMITED_RATIO
    if _ratio_is_basin_test(cfg):
        flag = SOLVER_LIMITED_FLAG if limited else ""
    else:
        flag = NOT_APPLICABLE_FLAG     # the ratio is still written
    return {
        "bandwidth_pct": float(pct),
        "freqs_hz": [float(f) for f in freqs],
        "solver_loss": loss["solver"],
        "true_plate_loss": loss["true_plate"],
        "ratio": ratio,
        "zero_model_loss": loss["zero_model"],
        "n_iters": loss["n_iters"],
        "stop_reason": loss["stop_reason"],
        "flag": flag,
        # realized mean per-RX-element SNR of each tone (null when noiseless)
        "mean_snr_db": (None if mgs.noise_report is None else
                        [rec["mean_snr_db"] for rec in mgs.noise_report["per_freq"]]),
    }


def _fmt_ratio(v) -> str:
    return "n/a" if v is None else f"{v:.3f}"


def _table(rows) -> str:
    lines = ["| " + " | ".join(COLUMNS) + " |", "| " + " | ".join(ALIGN) + " |"]
    for r in rows:
        lines.append("| " + " | ".join([
            f"{r['bandwidth_pct']:g}",
            ", ".join(f"{f / 1e9:g}" for f in r["freqs_hz"]),
            f"{r['solver_loss']:.3e}", f"{r['true_plate_loss']:.3e}",
            _fmt_ratio(r["ratio"]),
            f"{r['zero_model_loss']:.3e}", str(r["n_iters"]), r["stop_reason"],
            r["flag"]]) + " |")
    return "\n".join(lines)


def _noise_line(cfg, rows) -> str:
    if cfg.noise.snr_db is None:
        return "none (noiseless measurement)"
    snrs = [v for r in rows for v in (r["mean_snr_db"] or []) if v is not None]
    realized = (f"realized mean per-element SNR {min(snrs):.1f} to {max(snrs):.1f} dB "
                "over the table's tones" if snrs else "realized mean per-element SNR n/a")
    return (f"complex AWGN at {cfg.noise.snr_db:g} dB re the peak RX element, seed "
            f"{cfg.noise.seed}; {realized}")


def _header(config_path: Path, cfg, center_hz: float, rows, seed_note: str) -> str:
    beam = cfg.tx_aperture.beam
    params = ", ".join(f"{k}={v}" for k, v in beam.model_dump(mode="json").items()
                       if k != "type" and v is not None)
    gs = cfg.gerchberg_saxton
    if _ratio_is_basin_test(cfg):
        flag_line = (
            f"- flag: `{SOLVER_LIMITED_FLAG}` where ratio > {SOLVER_LIMITED_RATIO:g}. "
            "There the joint solve stopped well short of the true plate, so a "
            "localization error at that bandwidth reflects the solver, not the physics. "
            "A flagged row that stopped on `max_iters` ran out of iterations (raise "
            "--max-iters to see where it was heading); one that `converged` settled "
            "somewhere other than the true plate.")
    else:
        flag_line = (
            f"- flag: `{NOT_APPLICABLE_FLAG}` on every row; no row is tested for "
            f"`{SOLVER_LIMITED_FLAG}`. This run is noiseless with an on-grid receiver "
            f"(rx_aperture.grid_stride: {cfg.rx_aperture.grid_stride}): the target is "
            "the true field at the elements themselves, so the true plate scores "
            "almost zero, and the descent stops far above that, on the flatness "
            "test or the iteration cap, even when it sits in the true plate's "
            "basin. The ratio is written "
            "as measured, but here it is not a basin test — a large value does not "
            "mean the solve stopped short of the true plate. Run with receiver noise "
            "(--snr-db) for a ratio that tests the basin.")
    return "\n".join([
        "# Single MGS solve at the true TX location, per bandwidth",
        "",
        f"- config: `{config_path}`",
        f"- beam: {beam.type} ({params}), TX at z = {cfg.tx_aperture.z:g} m, "
        f"x in [{cfg.tx_aperture.x_min:g}, {cfg.tx_aperture.x_max:g}] m",
        f"- combs: ± % about {center_hz / 1e9:g} GHz (both edges and the centre; "
        "0 is the single tone)",
        f"- noise: {_noise_line(cfg, rows)}",
        f"- solve: ONE solve from `init: {gs.init}`, iteration cap {gs.max_iters}, "
        f"stop threshold {gs.convergence_threshold:g}, solver seed {gs.seed}"
        f"{seed_note} — what `mgs` runs at this location.",
        "- ratio = solver loss / true-plate loss, both the joint (mean over tones) "
        "loss at the true TX location. The true-plate loss is what the real plate's "
        "phase scores under the solver's own model, so a ratio at or below about 1 "
        "means the solve reached the true plate's basin (below 1 is ordinary: the "
        "solver also fits the receiver's element sampling and noise, which the true "
        "plate does not reproduce). The zero-model loss is what radiating nothing "
        "scores.",
        flag_line,
    ])


def main() -> int:
    p = argparse.ArgumentParser(
        description="Single MGS solve at the true TX location for every comb of the "
                    "bandwidth study; flags the solver-limited ones (not on a "
                    "noiseless run with an on-grid receiver, where the ratio is not "
                    "a basin test).")
    p.add_argument("config", type=Path)
    p.add_argument("--out", type=Path, required=True,
                   help="Markdown file to write; the rows also go to the .json beside it")
    p.add_argument("--snr-db", type=parse_snr_db_arg, metavar="DB|off", default=None,
                   help="Peak-referenced SNR in dB of the receiver noise (overrides "
                        "noise.snr_db); 'off' forces a noiseless run")
    p.add_argument("--noise-seed", type=int, default=None,
                   help="Override noise.seed")
    p.add_argument("--max-iters", type=int, default=None,
                   help="Override gerchberg_saxton.max_iters, the iteration cap")
    args = p.parse_args()
    if args.max_iters is not None and args.max_iters <= 0:
        p.error("--max-iters must be positive")
    if args.out.suffix.lower() == ".json":
        p.error("--out names the Markdown file; the JSON is written beside it")
    log = setup_logging(False)

    cfg = load_config(args.config)
    try:
        apply_noise_overrides(cfg, args.snr_db, args.noise_seed)
    except ValidationError as e:
        p.error("invalid noise override: " + "; ".join(
            f"{'.'.join(str(q) for q in err['loc'])}: {err['msg']}" for err in e.errors()))
    if args.max_iters is not None:
        cfg.gerchberg_saxton.max_iters = args.max_iters

    # every row deep-copies cfg, so a null seed left null would be drawn per row:
    # each solve from its own random start, each measurement with its own noise
    seed_note = ""
    if cfg.gerchberg_saxton.seed is None:
        grid_seed = cfg.grid_search.seed if cfg.grid_search is not None else None
        cfg.gerchberg_saxton.seed = 0 if grid_seed is None else int(grid_seed)
        seed_note = "; the config's was null"
    if cfg.noise.snr_db is not None and cfg.noise.seed is None:
        cfg.noise.seed = int(np.random.SeedSequence().entropy % (2**32))

    center_hz = study.base_center_freq(cfg)
    rows = []
    t_start = time.time()
    for pct in study.BANDWIDTH_PCTS:
        log.info(f"--- ±{pct:g} % about {center_hz / 1e9:g} GHz")
        rows.append(_solve_row(cfg, pct, center_hz))
    log.info(f"Solved {len(rows)} bandwidth points in {time.time() - t_start:.1f} s")

    table = _table(rows)
    out_md = args.out
    out_json = out_md.with_suffix(".json")
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text(_header(args.config, cfg, center_hz, rows, seed_note)
                      + "\n\n" + table + "\n", encoding="utf-8")
    write_json(out_json, rows)
    log.info(f"Wrote {out_md} and {out_json}")
    print(table)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
