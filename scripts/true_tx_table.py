#!/usr/bin/env python
"""Which points of a bandwidth study are SOLVER-limited: one single MGS solve at the
KNOWN TX location per comb of the `frequency` study, its final loss set against the
loss of the true plate.

A grid search can only rank the true location first if the solve AT the true location
gets down to what the true plate scores. The multi-tone joint solve does not always.
It has to recover the plate's ABSOLUTE delay profile, and its warm start gets there by
unwrapping the reference-frequency phase: when that unwrap slips by 2*pi the descent
starts outside the true plate's basin and converges well above the true-plate loss.
(A single tone cannot fail this way -- profiles that differ by 2*pi anywhere are the
same field to it.) A study point where that happens measures the solver, not the
physics, and this table says which ones they are.

For each pct in rice_bend.study.BANDWIDTH_PCTS the tones are the study's own comb
about the config's centre frequency (study.comb, study.base_center_freq), solved the
way `mgs` solves them (MGS.run_gerch_sax on a copy of the config; no scene
illumination, no plot). The losses are MGS.landing_report()["loss"].

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

COLUMNS = ("bandwidth (± %)", "tones (GHz)", "solver loss", "true-plate loss", "ratio",
           "zero-model loss", "iterations", "stop reason", "flag")
ALIGN = ("---:", ":---", "---:", "---:", "---:", "---:", "---:", ":---", ":---")


def _solve_row(cfg, pct: float, center_hz: float) -> dict:
    """One bandwidth point: the joint solve at the true TX and its loss block."""
    freqs = study.comb(center_hz, pct)
    mgs = MGS(freqs, cfg.model_copy(deep=True))
    mgs.run_gerch_sax()
    loss = mgs.landing_report()["loss"]
    ratio = loss["ratio"]
    limited = ratio is not None and ratio > SOLVER_LIMITED_RATIO
    return {
        "bandwidth_pct": float(pct),
        "freqs_hz": [float(f) for f in freqs],
        "solver_loss": loss["solver"],
        "true_plate_loss": loss["true_plate"],
        "ratio": ratio,
        "zero_model_loss": loss["zero_model"],
        "n_iters": loss["n_iters"],
        "stop_reason": loss["stop_reason"],
        "flag": SOLVER_LIMITED_FLAG if limited else "",
        # realized mean per-RX-element SNR of each tone (null when noiseless)
        "mean_snr_db": (None if mgs.noise_report is None else
                        [rec["mean_snr_db"] for rec in mgs.noise_report["per_freq"]]),
    }


def _table(rows) -> str:
    lines = ["| " + " | ".join(COLUMNS) + " |", "| " + " | ".join(ALIGN) + " |"]
    for r in rows:
        ratio = "n/a" if r["ratio"] is None else f"{r['ratio']:.3f}"
        lines.append("| " + " | ".join([
            f"{r['bandwidth_pct']:g}",
            ", ".join(f"{f / 1e9:g}" for f in r["freqs_hz"]),
            f"{r['solver_loss']:.3e}", f"{r['true_plate_loss']:.3e}", ratio,
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
    return "\n".join([
        "# Single MGS solve at the true TX location, per bandwidth",
        "",
        f"- config: `{config_path}`",
        f"- beam: {beam.type} ({params}), TX at z = {cfg.tx_aperture.z:g} m, "
        f"x in [{cfg.tx_aperture.x_min:g}, {cfg.tx_aperture.x_max:g}] m",
        f"- combs: ± % about {center_hz / 1e9:g} GHz (both edges and the centre; "
        "0 is the single tone)",
        f"- noise: {_noise_line(cfg, rows)}",
        f"- iteration cap: {gs.max_iters} (init {gs.init}, solver seed {gs.seed}"
        f"{seed_note})",
        "- ratio = solver loss / true-plate loss, both the joint (mean over tones) "
        "loss at the true TX location. The true-plate loss is what the real plate's "
        "phase scores under the solver's own model, so a ratio at or below about 1 "
        "means the solve reached the true plate's basin (below 1 is ordinary: the "
        "solver also fits the receiver's element sampling and noise, which the true "
        "plate does not reproduce). The zero-model loss is what radiating nothing "
        "scores.",
        f"- flag: `{SOLVER_LIMITED_FLAG}` where ratio > {SOLVER_LIMITED_RATIO:g}. "
        "There the joint solve stopped well short of the true plate, so a "
        "localization error at that bandwidth reflects the solver, not the physics. "
        "A flagged row that stopped on `max_iters` ran out of iterations (raise "
        "--max-iters to see where it was heading); one that `converged` settled "
        "somewhere other than the true plate.",
    ])


def main() -> int:
    p = argparse.ArgumentParser(
        description="Single MGS solve at the true TX location for every comb of the "
                    "bandwidth study; flags the solver-limited ones.")
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
