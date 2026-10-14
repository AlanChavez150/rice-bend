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

When the config has a grid_search.refine block, each bandwidth gets a SECOND solve at
the true TX: what the sweep's refinement pass would give this location if it were a
near-best candidate — rice_bend.mgs.gs_reconstruct_multistart (both starts, each from
every offset its scan cannot tell apart) under the block's iteration cap and stop
threshold, from the seed the sweep uses (grid_search.seed). Its loss, its ratio to the
true plate's loss and the start it came from go beside the single-start columns, and
the `solver-limited` flag is then read off the REFINED ratio: that is the loss a
sweep's ranking is made from near the truth, and the single start alone can sit well
above it (a comb that nearly repeats inside its period leaves two offsets the scan
cannot call, and the single start takes one of them). The single-start columns stay,
for comparison. Without a refine block the output is what it always was.

Usage:
    python scripts/true_tx_table.py CONFIG.yml --out FILE.md \
        [--snr-db DB|off] [--noise-seed N] [--max-iters N] [--engine python|rust]

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
from rice_bend.config import (apply_noise_overrides, center_freq_index, load_config,
                              parse_snr_db_arg)
from rice_bend.data_store import write_json
from rice_bend.mgs import MGS, gs_reconstruct_multistart

# a solve this far above the true plate's loss did not reach the true plate's basin
SOLVER_LIMITED_RATIO = 2.0
SOLVER_LIMITED_FLAG = "solver-limited"
# the flag column of a run whose ratio is not a basin test (_ratio_is_basin_test)
NOT_APPLICABLE_FLAG = "n/a"

COLUMNS = ("bandwidth (± %)", "tones (GHz)", "solver loss", "true-plate loss", "ratio",
           "zero-model loss", "iterations", "stop reason", "flag")
ALIGN = ("---:", ":---", "---:", "---:", "---:", "---:", "---:", ":---", ":---")
# the layout of a run with a grid_search.refine block: the same single-start columns,
# named as such, and the refined solve's beside them
COLUMNS_REFINED = ("bandwidth (± %)", "tones (GHz)", "single-start loss",
                   "true-plate loss", "single-start ratio", "zero-model loss",
                   "single-start iterations", "single-start stop", "refined loss",
                   "refined ratio", "refined start", "flag")
ALIGN_REFINED = ("---:", ":---", "---:", "---:", "---:", "---:", "---:", ":---", "---:",
                 "---:", ":---", ":---")


def _ratio_is_basin_test(cfg) -> bool:
    """Whether solver loss / true-plate loss says anything about the basin.

    The test rests on the true plate NOT scoring zero: with an interpolated
    receiver its loss is the element sampling it cannot reproduce, with noise the
    noise it cannot fit, and a solve in the right basin gets down to about that. A
    NOISELESS run with an on-grid receiver (rx_aperture.grid_stride) has neither.
    Its target IS the true field at the elements, so the true plate is left with
    the gap between the solver's amplitude constraint and the magnitude of the
    plate's resampling — 4.6e-8 to 6.6e-8 on the three scenario_*_hit_1f_grid2l
    configs — while the descent's flatness test stops at 4e-6 to 1e-5: ratios of 90
    to 170 on solves that match the true field to 1-2 % rel-L2
    (MGS.landing_report has the measurements). A threshold on that ratio would
    flag every row, right basin or not, so there the flag is not computed at all.
    """
    return not (cfg.noise.snr_db is None and cfg.rx_aperture.grid_stride is not None)


def _refine_cfg(cfg):
    """The config's grid_search.refine block, or None: whether the table carries the
    refined solve at all."""
    return None if cfg.grid_search is None else cfg.grid_search.refine


def _refined_params(cfg):
    """The solver settings of the refined solve: gerchberg_saxton's, under the refine
    block's iteration cap and stop threshold (null -> gerchberg_saxton's own), from
    the seed the sweep runs every candidate from (grid_search.seed, else
    gerchberg_saxton.seed) — what grid_sweep._refine_candidates hands
    gs_reconstruct_multistart. --max-iters does not reach it: that option is the
    single-start solve's cap."""
    refine, gs = _refine_cfg(cfg), cfg.gerchberg_saxton
    threshold = (refine.convergence_threshold if refine.convergence_threshold is not None
                 else gs.convergence_threshold)
    seed = cfg.grid_search.seed if cfg.grid_search.seed is not None else gs.seed
    return gs.model_copy(update={"max_iters": refine.max_iters,
                                 "convergence_threshold": threshold, "seed": int(seed)})


def _solve_row(cfg, pct: float, center_hz: float) -> dict:
    """One bandwidth point: the joint solve at the true TX and its loss block, and —
    with a grid_search.refine block — the refined solve of the same problem."""
    freqs = study.comb(center_hz, pct)
    mgs = MGS(freqs, cfg.model_copy(deep=True))
    mgs.run_gerch_sax()
    loss = mgs.landing_report()["loss"]
    ratio = loss["ratio"]
    row = {
        "bandwidth_pct": float(pct),
        "freqs_hz": [float(f) for f in freqs],
        "solver_loss": loss["solver"],
        "true_plate_loss": loss["true_plate"],
        "ratio": ratio,
        "zero_model_loss": loss["zero_model"],
        "n_iters": loss["n_iters"],
        "stop_reason": loss["stop_reason"],
    }
    # the ratio the flag is read off: the single start's, unless a refined solve
    # replaces it below
    flag_ratio = ratio
    refined = {}
    if _refine_cfg(cfg) is not None:
        # The SAME problem the single start just solved: its amplitude constraint
        # (the array run_gerch_sax handed the solver, which is also the one the
        # true plate's loss was scored under), its channels, its reference tone.
        scene = mgs.scene
        result, start = gs_reconstruct_multistart(
            tx_z=scene.tx_ap.z, orig_aper_amp=mgs.gs_history.orig_aper_amp,
            x_axis=scene.x_axis.copy(), rx_z=scene.rx_ap.z,
            channels=mgs.measurement_channels(), params=_refined_params(cfg),
            ref_freq=mgs.freqs[center_freq_index(mgs.freqs)], log=mgs.log)
        true_plate = loss["true_plate"]
        flag_ratio = result.final_loss / true_plate if true_plate > 0 else None
        refined = {
            "refined_loss": float(result.final_loss),
            "refined_ratio": flag_ratio,
            "refined_start": start,
            "refined_n_iters": int(result.n_iters_run),
            "refined_stop_reason": str(result.stop_reason),
        }
    limited = flag_ratio is not None and flag_ratio > SOLVER_LIMITED_RATIO
    if _ratio_is_basin_test(cfg):
        flag = SOLVER_LIMITED_FLAG if limited else ""
    else:
        flag = NOT_APPLICABLE_FLAG     # the ratios are still written
    row["flag"] = flag
    # realized mean per-RX-element SNR of each tone (null when noiseless)
    row["mean_snr_db"] = (None if mgs.noise_report is None else
                          [rec["mean_snr_db"] for rec in mgs.noise_report["per_freq"]])
    # only a run with a refine block has these keys (and says which ratio the flag
    # was read off); a run without one writes the rows it always wrote
    if refined:
        row.update(refined)
        row["flag_basis"] = "refined_ratio"
    return row


def _fmt_ratio(v) -> str:
    return "n/a" if v is None else f"{v:.3f}"


def _table(rows) -> str:
    refined = bool(rows) and "refined_loss" in rows[0]
    columns, align = (COLUMNS_REFINED, ALIGN_REFINED) if refined else (COLUMNS, ALIGN)
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(align) + " |"]
    for r in rows:
        cells = [
            f"{r['bandwidth_pct']:g}",
            ", ".join(f"{f / 1e9:g}" for f in r["freqs_hz"]),
            f"{r['solver_loss']:.3e}", f"{r['true_plate_loss']:.3e}",
            _fmt_ratio(r["ratio"]),
            f"{r['zero_model_loss']:.3e}", str(r["n_iters"]), r["stop_reason"]]
        if refined:
            cells += [f"{r['refined_loss']:.3e}", _fmt_ratio(r["refined_ratio"]),
                      r["refined_start"]]
        lines.append("| " + " | ".join(cells + [r["flag"]]) + " |")
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
    refined = _refine_cfg(cfg) is not None
    if _ratio_is_basin_test(cfg) and not refined:
        flag_line = (
            f"- flag: `{SOLVER_LIMITED_FLAG}` where ratio > {SOLVER_LIMITED_RATIO:g}. "
            "There the joint solve stopped well short of the true plate, so a "
            "localization error at that bandwidth reflects the solver, not the physics. "
            "A flagged row that stopped on `max_iters` ran out of iterations (raise "
            "--max-iters to see where it was heading); one that `converged` settled "
            "somewhere other than the true plate.")
    elif _ratio_is_basin_test(cfg):
        flag_line = (
            f"- flag: `{SOLVER_LIMITED_FLAG}` where the REFINED ratio > "
            f"{SOLVER_LIMITED_RATIO:g}: even the refined solve stopped well short of "
            "the true plate, so a localization error at that bandwidth reflects the "
            "solver, not the physics. The flag is read off the refined ratio because "
            "that is the loss the sweep's near-best candidates are ranked by; a "
            "single-start ratio above the threshold beside an unflagged refined one "
            "is a start the refinement repairs (a comb that nearly repeats inside "
            "its period leaves the single start a near-tie between two offsets, and "
            "it can take the wrong one). The refined solve's own iteration count and "
            "stop reason are in the JSON (`refined_n_iters`, `refined_stop_reason`): "
            "a flagged row that stopped on `max_iters` ran out of the refinement's "
            "iterations, one that `converged` settled somewhere other than the true "
            "plate.")
    else:
        flag_line = (
            f"- flag: `{NOT_APPLICABLE_FLAG}` on every row; no row is tested for "
            f"`{SOLVER_LIMITED_FLAG}`. This run is noiseless with an on-grid receiver "
            f"(rx_aperture.grid_stride: {cfg.rx_aperture.grid_stride}): the target is "
            "the true field at the elements themselves, so the true plate scores "
            "almost zero, and the descent's flatness test stops a solve far above "
            "that even when it sits in the true plate's basin. The ratio is written "
            "as measured, but here it is not a basin test — a large value does not "
            "mean the solve stopped short of the true plate. Run with receiver noise "
            "(--snr-db) for a ratio that tests the basin.")
    if not refined:
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
            flag_line,
        ])
    rp = _refined_params(cfg)
    return "\n".join([
        "# MGS solves at the true TX location, per bandwidth: single start and refined",
        "",
        f"- config: `{config_path}`",
        f"- beam: {beam.type} ({params}), TX at z = {cfg.tx_aperture.z:g} m, "
        f"x in [{cfg.tx_aperture.x_min:g}, {cfg.tx_aperture.x_max:g}] m",
        f"- combs: ± % about {center_hz / 1e9:g} GHz (both edges and the centre; "
        "0 is the single tone)",
        f"- noise: {_noise_line(cfg, rows)}",
        f"- single-start columns: ONE solve from `init: {gs.init}`, iteration cap "
        f"{gs.max_iters}, stop threshold {gs.convergence_threshold:g}, solver seed "
        f"{gs.seed}{seed_note} — what `mgs` runs at this location.",
        f"- refined columns: the lowest final loss over both starts "
        f"(rice_bend.mgs.gs_reconstruct_multistart: the data-driven and the random "
        f"one, each descended from every offset its scan cannot tell apart), "
        f"iteration cap {rp.max_iters}, stop threshold {rp.convergence_threshold:g} "
        f"(grid_search.refine), solver seed {rp.seed} (the sweep's), engine "
        f"{rp.engine} — what the sweep's refinement pass gives a near-best candidate. "
        "`refined start` is the start that loss came from.",
        "- ratio = loss / true-plate loss, for either solve; all are the joint (mean "
        "over tones) loss at the true TX location. The true-plate loss is what the "
        "real plate's phase scores under the solver's own model, so a ratio at or "
        "below about 1 means the solve reached the true plate's basin (below 1 is "
        "ordinary: the solver also fits the receiver's element sampling and noise, "
        "which the true plate does not reproduce). The zero-model loss is what "
        "radiating nothing scores.",
        flag_line,
    ])


def main() -> int:
    p = argparse.ArgumentParser(
        description="Single MGS solve at the true TX location for every comb of the "
                    "bandwidth study; flags the solver-limited ones (not on a "
                    "noiseless run with an on-grid receiver, where the ratio is not "
                    "a basin test). With a grid_search.refine block in the config, "
                    "also the refined solve the sweep's second pass would run "
                    "there, and the flag is read off that one.")
    p.add_argument("config", type=Path)
    p.add_argument("--out", type=Path, required=True,
                   help="Markdown file to write; the rows also go to the .json beside it")
    p.add_argument("--snr-db", type=parse_snr_db_arg, metavar="DB|off", default=None,
                   help="Peak-referenced SNR in dB of the receiver noise (overrides "
                        "noise.snr_db); 'off' forces a noiseless run")
    p.add_argument("--noise-seed", type=int, default=None,
                   help="Override noise.seed")
    p.add_argument("--max-iters", type=int, default=None,
                   help="Override gerchberg_saxton.max_iters, the iteration cap "
                        "(of the single-start solve; the refined solve's is "
                        "grid_search.refine.max_iters)")
    p.add_argument("--engine", choices=("python", "rust"), default=None,
                   help="Override gerchberg_saxton.engine. It decides the REFINED "
                        "solve's engine; the single-start solve captures history and "
                        "runs on python whatever this says")
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
    if args.engine is not None:
        cfg.gerchberg_saxton.engine = args.engine

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
