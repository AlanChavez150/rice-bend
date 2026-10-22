"""Entry point for `grid-search-mgs`: orchestration and CLI.

The work lives in three modules this one drives, and which never import each other:
  grid_sweep       -- enumerate the grid, solve, persist   (no matplotlib)
  residual_plots   -- the residual grid and its six plots
  candidate_scenes -- re-illuminated candidate beams, and the true-MGS baseline
"""

import argparse
import json
import os
from pathlib import Path
from typing import Callable, List, Optional, Tuple

from pydantic import ValidationError

from rice_bend import rs
from rice_bend.analysis import analysis_from_manifest, analysis_from_run, write_analysis
from rice_bend.candidate_scenes import (ANIM_FPS, ANIM_TOP_DEFAULT, ANIM_WARN_FRAMES,
                                        SCENE_Z_PLANES, animate_candidate_beams,
                                        make_candidate_scenes, make_true_mgs_plot,
                                        scenes_from_manifest, scenes_from_run)
from rice_bend.cli import setup_logging
from rice_bend.config import (DEFAULT_CONFIG, DEFAULT_FREQ_HZ, PlotWindowConfig,
                              SimConfig, apply_noise_overrides, load_config,
                              parse_snr_db_arg, resolve_frequencies)
from rice_bend.data_store import check_run_dir, make_run_dir
from rice_bend.grid_sweep import (GridSearchRun,
                                  enumerate_grid, grid_summary, load_frequencies_index,
                                  run_grid_search, save_grid_run,
                                  save_measured_tx_solve, solve_at_measured_tx)
from rice_bend.measured import apply_measured
from rice_bend.mgs import check_amplitude_mode
from rice_bend.residual_plots import (MaxItersMask, ResidualSummary,
                                      animate_residual_surface,
                                      freq_summaries_from_manifest,
                                      freq_summaries_from_run,
                                      max_iters_from_manifest, max_iters_from_run,
                                      plot_residual_freq_vs_joint,
                                      plot_residual_heatmap, plot_residual_scatter,
                                      plot_residual_scatter_3d,
                                      plot_residual_scatter_3d_diff,
                                      plot_residual_surface, summary_from_manifest,
                                      summary_from_run)
from rice_bend.tds_data import MeasuredDataError

def _emit_pair(plot, out_path: Path, *args, **kwargs) -> Path:
    """Write a residual plot and its high-contrast twin (<name>_hc.png).

    Every plot function takes `hc: bool` and builds its own norm and title suffix,
    so that flag is the entire difference between the two -- which is why this is
    four lines and not five near-copies. `hc` used to be spelled four different ways
    across the plot functions (norm=, log_scale=, norm=_hc_norm(np.stack(...)), hc=).
    """
    plot(*args, out_path=out_path, hc=False, **kwargs)
    plot(*args, out_path=out_path.with_name(f"{out_path.stem}_hc{out_path.suffix}"),
         hc=True, **kwargs)
    return out_path


def _emit_heatmaps(summary: ResidualSummary, out_dir: Path, log,
                   analysis: Optional[dict] = None,
                   window: Optional[PlotWindowConfig] = None,
                   max_iters: Optional[MaxItersMask] = None) -> None:
    """Write the residual heatmap plus its high-contrast (log-scale) twin, with
    the top-candidate outlines when the analysis dict is given, and the hc heatmap
    again with the candidates that hit max_iters hatched (one file, no linear twin).
    `max_iters` is None only on a replot of a manifest without stop reasons."""
    out = _emit_pair(plot_residual_heatmap, out_dir / "residual_heatmap.png", summary,
                     analysis=analysis, window=window)
    log.info(f"Wrote residual heatmap (+hc) to {out}")
    if max_iters is None:
        log.info(f"No residual_heatmap_maxiters.png for {out_dir}: its manifest "
                 "records no per-candidate stop_reason / gs.effective_max_iters")
        return
    out = out_dir / "residual_heatmap_maxiters.png"
    plot_residual_heatmap(summary, out, hc=True, analysis=analysis, window=window,
                          max_iters=max_iters)
    log.info(f"Wrote max_iters overlay ({int(max_iters.capped.sum()):,} of "
             f"{max_iters.n_ran:,} candidates hit max_iters {max_iters.limit}) to {out}")


def _emit_scatters(summary: ResidualSummary, out_dir: Path, log,
                   window: Optional[PlotWindowConfig] = None) -> None:
    """Write the residual scatter plus its high-contrast (log-y) twin."""
    out = _emit_pair(plot_residual_scatter, out_dir / "residual_scatter.png", summary,
                     window=window)
    log.info(f"Wrote residual scatter (+hc) to {out}")


def _emit_surfaces(summary: ResidualSummary, out_dir: Path, args, log,
                   window: Optional[PlotWindowConfig] = None) -> None:
    """Write the residual surface plus its high-contrast twin, and optionally the orbit.

    The stills are manifest-only and as cheap as the heatmaps, so they always run. The
    orbit mp4 is opt-in for two reasons: write_mp4 raises when ffmpeg is missing, so an
    unconditional video would make the whole command unrunnable on a box without it;
    and at ~12 s each (measured, 48x48 grid), one per variant per frequency directory
    is ~4 min added to a 10-frequency run -- and to every --replot of it, which
    otherwise finishes in seconds.
    """
    out = _emit_pair(plot_residual_surface, out_dir / "residual_surface.png", summary,
                     window=window)
    log.info(f"Wrote residual surface (+hc) to {out}")
    if args.surface_anim:
        orbit = _emit_pair(animate_residual_surface,
                           out_dir / "residual_surface_orbit.mp4", summary, log,
                           window=window)
        # a degenerate grid has no surface to rotate, so animate_ returns without
        # writing; say that rather than claiming a file that is not there.
        if orbit.exists():
            log.info(f"Wrote residual-surface orbit (+hc) to {orbit}")
        else:
            log.info(f"No residual-surface orbit for {out_dir}: grid is "
                     f"{summary.loss_grid.shape[0]}x{summary.loss_grid.shape[1]}")


def _freq_tag(freq_hz: float) -> str:
    """Filename tag for one frequency, e.g. 140e9 -> '140GHz'."""
    return f"{freq_hz / 1e9:g}GHz"


def _emit_multifreq_extras(summary: ResidualSummary,
                           freq_summaries: List[Tuple[float, ResidualSummary]],
                           out_dir: Path, log) -> None:
    """The F>1 plot extras, all in the one run directory: the 3D per-frequency
    scatters and a vs-joint comparison per frequency.

    These draw the per-frequency COMPONENTS of the joint solve — the residual each
    frequency contributes to the mean the solver minimized — not independent
    per-frequency solves (those died with the freq_<GHz>/ layout)."""
    out3d = _emit_pair(plot_residual_scatter_3d,
                       out_dir / "residual_scatter_3d.png", freq_summaries)
    log.info(f"Wrote 3D residual scatter (+hc) to {out3d}")
    out3d_diff = out_dir / "residual_scatter_3d_diff.png"
    plot_residual_scatter_3d_diff(freq_summaries, out3d_diff)
    log.info(f"Wrote 3D residual-difference scatter to {out3d_diff}")
    for freq_hz, s in freq_summaries:
        _emit_pair(plot_residual_freq_vs_joint,
                   out_dir / f"residual_heatmap_vs_joint_{_freq_tag(freq_hz)}.png",
                   freq_hz, s, summary)
    log.info(f"Wrote {len(freq_summaries)} per-frequency vs-joint comparison(s) "
             f"(+hc) to {out_dir}")


def _emit_scenes_and_anim(make_ctx, out_dir: Path, args, log,
                          window: Optional[PlotWindowConfig] = None) -> None:
    """Per-candidate scenes and/or the candidate animation, if asked for.

    `make_ctx` is a THUNK, not a SceneContext: scenes_from_manifest reads every
    candidate npz (2304 of them on scenario_caustic_hit), so constructing one
    eagerly only to discover that both flags are off would be a real regression.

    The plot window applies to the scenes; the animation always shows the whole
    scene.
    """
    if not (args.scenes or args.anim):
        return
    ctx = make_ctx()
    if args.scenes:
        make_candidate_scenes(ctx, out_dir, z_planes=SCENE_Z_PLANES,
                              top=args.scene_top, jobs=args.jobs, log=log,
                              window=window)
    if args.anim:
        animate_candidate_beams(ctx, out_dir / "candidate_beams.mp4",
                                z_planes=SCENE_Z_PLANES, top=_anim_top(args, log),
                                fps=ANIM_FPS, jobs=args.jobs, log=log)


def _anim_top(args, log) -> Optional[int]:
    """How many frames the animation may hold.

    animate_candidate_beams must keep every frame resident to fix a shared colour
    scale -- 2304 x 200 x 2400 float32 is 4.4 GB on scenario_caustic_hit -- so
    --anim caps itself unless --scene-top says otherwise.
    """
    if args.scene_top is None:
        log.info(f"--anim: animating the {ANIM_TOP_DEFAULT} lowest-residual candidates. "
                 f"Every frame is held in memory at once to fix a shared colour scale; "
                 f"pass --scene-top to choose a different cap.")
        return ANIM_TOP_DEFAULT
    if args.scene_top > ANIM_WARN_FRAMES:
        log.warning(f"--anim with --scene-top {args.scene_top}: that is "
                    f"{args.scene_top} full scene frames held in memory at once "
                    f"(roughly {args.scene_top * SCENE_Z_PLANES * 2400 * 4 / 1e9:.1f} GB "
                    f"at the default scene width).")
    return args.scene_top


def _emit_run_plots(summary: ResidualSummary,
                    freq_summaries: List[Tuple[float, ResidualSummary]],
                    out_dir: Path, args, log, make_ctx,
                    analysis: Optional[dict] = None,
                    true_mgs: Optional[Callable[[], None]] = None,
                    window: Optional[PlotWindowConfig] = None,
                    max_iters: Optional[MaxItersMask] = None) -> None:
    """Everything one run directory gets.

    One rule: the cheap manifest-only plots always run; anything that re-solves MGS
    or renders per-candidate PNGs is opt-in. A fresh run and a --replot of it now
    produce the same set of files, which was not true when --summary/--scatter
    gated the fresh path and --replot ignored them.

    `summary` is the joint residual (what the solver minimized); `freq_summaries`
    its per-frequency components, driving the F>1 extras. A single-frequency run
    (or a legacy schema-2 replot, whose component list is empty) keeps exactly the
    flat plot set.

    `true_mgs` replaces the --true-mgs baseline when given: the candidate-style
    solve at the TX (_measured_true_mgs) of a measured run, which has no plate to
    rebuild a scene from, or of a simulated run whose candidates fit the amplitude
    (descend / level), which the baseline's capturing solve cannot
    (_true_mgs_as_candidate).

    `window` (the plot window, None for none) crops the residual heatmap, scatter
    and surface and the candidate scenes; the multi-frequency extras, the true-MGS
    baseline and the animation always show the whole grid / scene.

    `max_iters` (which candidates hit the iteration cap) drives
    residual_heatmap_maxiters.png; a replot of a manifest without stop reasons
    passes None and that one plot is skipped with a log line.
    """
    if window is not None:
        log.info(f"Plot window: x {window.x_min:g}..{window.x_max:g} m, "
                 f"z {window.z_min:g}..{window.z_max:g} m (solver frame)")
    _emit_heatmaps(summary, out_dir, log, analysis=analysis, window=window,
                   max_iters=max_iters)
    _emit_scatters(summary, out_dir, log, window=window)
    _emit_surfaces(summary, out_dir, args, log, window=window)
    if len(freq_summaries) > 1:
        _emit_multifreq_extras(summary, freq_summaries, out_dir, log)
    if args.true_mgs:
        if true_mgs is not None:
            true_mgs()
        else:
            # baseline MGS run at the KNOWN TX location: one full solve per run dir
            make_true_mgs_plot(out_dir, log=log, scene_freq=args.scene_freq)
    _emit_scenes_and_anim(make_ctx, out_dir, args, log, window=window)


def _true_mgs_as_candidate(run: GridSearchRun) -> bool:
    """True when this run's --true-mgs is the candidate-style solve at the TX
    (_measured_true_mgs) rather than the plain-`mgs` baseline
    (make_true_mgs_plot): on a measured run, which has no plate to rebuild that
    baseline from, and on a simulated run whose candidates fit the amplitude
    (descend or level), which the baseline's capturing solve cannot
    (check_amplitude_mode). A simulated fixed-amplitude run keeps the baseline."""
    return run.measured is not None or run.amplitude != "fixed"


def _measured_true_mgs(run: GridSearchRun, config: SimConfig, out_dir: Path,
                       log) -> None:
    """--true-mgs as a candidate (_true_mgs_as_candidate): the solve at the TX
    position (grid_sweep.solve_at_measured_tx) -- the measured TX on a measured
    run, the true (configured) TX on a simulated one -- saved to true_mgs.json /
    true_mgs.npz and logged beside the best candidate's loss.

    config.gerchberg_saxton is the sweep's effective setting here: run_grid_search
    wrote the gs_overrides and the grid seed into it, and handed the candidates a
    copy of that same object."""
    which = "measured" if run.measured is not None else "true"
    tx_x = 0.5 * (run.real_tx_x_min + run.real_tx_x_max)
    gs = config.gerchberg_saxton
    amplitude = {"fixed": "uniform assumed amplitude",
                 "level": "uniform level descended from the power-matched start"
                 }.get(gs.amplitude) or f"amplitude descended, cap {gs.amp_max:g}"
    if gs.power_penalty is not None:
        amplitude += f", excess-power penalty gamma {gs.power_penalty:g}"
    log.info(f"Solving at the {which} TX (z={run.real_tx_z:.4f} x={tx_x:+.4f} m, "
             f"{amplitude}, the candidates' solver settings)")
    result, report = solve_at_measured_tx(run, config.gerchberg_saxton)
    path = save_measured_tx_solve(out_dir, result, report)
    best = report["best_candidate"]
    best_text = ("no candidate to compare with" if best is None else
                 f"best candidate #{best['index']} (z={best['z']:.4f} "
                 f"x={best['x_center']:+.4f} m, {best['dist_to_tx_m'] * 1e3:.1f} mm "
                 f"from the {which} TX) loss {best['final_loss']:.6g}")
    if result is None:
        log.warning(f"No solve at the {which} TX: {report['skip_reason']}; "
                    f"{best_text}. Wrote {path}")
        return
    ratio = report["loss_ratio_to_best"]
    log.info(f"{which.capitalize()} TX loss {result.final_loss:.6g} "
             f"({result.n_iters_run} iters, {result.stop_reason}) vs {best_text}; "
             f"{which}/best = {'n/a' if ratio is None else format(ratio, '.4g')}. "
             f"Wrote {path}")


def _cli_plot_window(args) -> Optional[PlotWindowConfig]:
    """--plot-window XMIN XMAX ZMIN ZMAX as a validated PlotWindowConfig (None when
    not given). Raises pydantic.ValidationError on a window that is not one.
    getattr: mgs-study hands persist_and_plot a namespace without the flag."""
    values = getattr(args, "plot_window", None)
    if values is None:
        return None
    x_min, x_max, z_min, z_max = values
    return PlotWindowConfig(x_min=x_min, x_max=x_max, z_min=z_min, z_max=z_max)


def _recorded_plot_window(run_dir: Path) -> Optional[PlotWindowConfig]:
    """The plot window a saved run recorded in candidate_beams.json (a run saved
    with one set), or None."""
    with open(Path(run_dir) / "candidate_beams.json") as f:
        recorded = json.load(f).get("plot_window")
    return None if recorded is None else PlotWindowConfig(**recorded)


def _replot_one_dir(run_dir: Path, args, log,
                    cli_window: Optional[PlotWindowConfig] = None) -> ResidualSummary:
    """Regenerate one saved run directory's plots + analysis.json from disk.

    The analysis is always RECOMPUTED (never read back), so a replot doubles as
    the reproducibility check against the save-time analysis.json. A schema-4
    manifest, the layout the removed refinement pass wrote, replots like a
    schema-3 one: its extra keys are ignored.

    The plots are windowed by `cli_window` (--plot-window) when given, else by the
    window the run recorded, else not at all. A --plot-window is used for this
    replot only: the manifest is a record of the run and is not rewritten."""
    run_dir = Path(run_dir)
    _refuse_replot_extras(run_dir, args, log)
    window = cli_window if cli_window is not None else _recorded_plot_window(run_dir)
    summary = summary_from_manifest(run_dir)
    analysis = analysis_from_manifest(run_dir, summary)
    write_analysis(run_dir, analysis)
    freq_summaries = freq_summaries_from_manifest(run_dir)
    _emit_run_plots(summary, freq_summaries, run_dir, args, log,
                    lambda: scenes_from_manifest(run_dir, scene_freq=args.scene_freq),
                    analysis=analysis, window=window,
                    max_iters=max_iters_from_manifest(run_dir))
    return summary


def persist_and_plot(run: GridSearchRun, out_dir: Path, config: SimConfig,
                     args, log) -> dict:
    """Save a finished sweep into out_dir, emit its plots and analysis.json.

    Public because the study driver (mgs-study) persists its per-point runs
    through the exact same path. Returns the analysis dict (the per-run numbers
    a study aggregates).

    The plots are windowed by config.plot_window (main writes --plot-window into
    it), which save_grid_run records in the manifest."""
    # --plot-window is left out of provenance.cli_args while it is not given, so a
    # run without it saves exactly the manifest it did before the flag existed
    cli_args = {k: v for k, v in vars(args).items()
                if not (k == "plot_window" and v is None)}
    save_grid_run(run, out_dir, config, args.config, cli_args)
    log.info(f"Saved {len(run.candidates)} candidate beam(s) to {out_dir}")
    summary = summary_from_run(run)
    analysis = analysis_from_run(run, summary)
    write_analysis(out_dir, analysis)
    freq_summaries = freq_summaries_from_run(run)
    true_mgs = None
    if _true_mgs_as_candidate(run):
        true_mgs = lambda: _measured_true_mgs(run, config, out_dir, log)  # noqa: E731
    _emit_run_plots(summary, freq_summaries, out_dir, args, log,
                    lambda: scenes_from_run(run, scene_freq=args.scene_freq),
                    analysis=analysis, true_mgs=true_mgs, window=config.plot_window,
                    max_iters=max_iters_from_run(run))
    return analysis


# The flags a measured config refuses, each with its reason as the error states
# it: --anim has no measured-run variant (--scenes does: no real-beam panel, the
# measured TX window outlined), and --snr-db / --noise-seed would add noise to data
# that carries its own.
_MEASURED_REFUSED = {
    "anim": "--anim (the candidate animation has no measured-run variant; "
            "--scenes --scene-top N renders the candidates, the measured TX marked)",
    "snr_db": "--snr-db (the measured traces carry their own receiver noise; none "
              "is added)",
    "noise_seed": "--noise-seed (no noise is drawn on measured data)",
}
# The value flags (default None) count as given whenever they are not None: 0, 0.0
# and 'off' are values, and `0 == False` would let a zero through a falsy test.
# The switches (store_true) count when set.
_MEASURED_VALUE_FLAGS = ("snr_db", "noise_seed")


def _refuse_measured_flags(args, config_path: Path, log) -> None:
    """Exit 2, naming every flag a measured config cannot take, before anything is
    derived, read or written."""
    refused = [why for key, why in _MEASURED_REFUSED.items()
               if (getattr(args, key) is not None if key in _MEASURED_VALUE_FLAGS
                   else getattr(args, key))]
    if refused:
        log.error(f"{config_path} is a measured-data config; drop "
                  + "; ".join(refused))
        raise SystemExit(2)


def _refuse_replot_extras(run_dir: Path, args, log) -> None:
    """A --replot of a measured run redraws its residual plots and analysis like
    any other, and --scenes renders its candidates from the saved npz (no real-beam
    panel, the measured TX window outlined); --true-mgs and --anim are refused
    (exit 2) before any of it. The measured --true-mgs solve runs on the in-memory
    measurement at sweep time and is not replayed: measurement.npz holds it as
    complex64, so a replay would solve a rounded copy, and the saved config does
    not load back. --anim has no measured-run variant.

    A simulated run whose candidates fit the amplitude (the manifest's
    gs.amplitude descend or level) refuses --true-mgs the same way: its solve at
    the true TX is the candidate-style one, made by the sweep (true_mgs.json) for
    the reason above, and the plain-`mgs` baseline a replot would otherwise rebuild
    (make_true_mgs_plot) captures GS history, which runs at a fixed amplitude only.
    Everything else replots as on any simulated run."""
    with open(run_dir / "candidate_beams.json") as f:
        manifest = json.load(f)
    if manifest.get("measured") is None:
        fitted = (manifest.get("gs") or {}).get("amplitude", "fixed")
        if args.true_mgs and fitted != "fixed":
            log.error(f"{run_dir} is a simulated run with gerchberg_saxton.amplitude "
                      f"'{fitted}'; drop --true-mgs: its solve at the true TX is a "
                      "candidate's, made by the sweep itself (true_mgs.json) and not "
                      "replayed, and the plain-`mgs` baseline runs at a fixed amplitude "
                      "only. Re-run grid-search-mgs --config <its config> --true-mgs "
                      "for one")
            raise SystemExit(2)
        return
    refused = []
    if args.true_mgs:
        refused.append("--true-mgs (the solve at the measured TX is made by the sweep "
                       "itself, true_mgs.json, and is not replayed: re-run "
                       "grid-search-mgs --config <the measured config> --true-mgs "
                       "for one)")
    if args.anim:
        refused.append("--anim (the candidate animation has no measured-run variant; "
                       "--scenes --scene-top N renders the candidates, the measured TX "
                       "marked)")
    if refused:
        log.error(f"{run_dir} is a measured-data run; drop " + "; ".join(refused))
        raise SystemExit(2)


def _dry_run(config: SimConfig, freqs: List[float], log) -> None:
    """Enumerate the grid and print it, without running any MGS.

    The enumeration depends on frequency only through the per-frequency
    RS-undersampling check, so a multi-frequency dry run reports which points
    are only partially valid and how many points each frequency keeps.
    """
    n_f = len(freqs)
    points = enumerate_grid(config.grid_search, config.sim_scene,
                            [rs.wavelength(f) for f in freqs])
    log.info(grid_summary(points))
    for p in points:
        if p.ok:
            partial = ("" if p.freq_ok is None or all(p.freq_ok)
                       else f"  [valid at {sum(p.freq_ok)}/{n_f} frequencies]")
            log.info(f"  #{p.index:4d} z={p.z:.3f} x_center={p.x_center:+.3f} "
                     f"window [{p.x_min:.3f}, {p.x_max:.3f}]{partial}")
        else:
            log.warning(f"  #{p.index:4d} z={p.z:.3f} x_center={p.x_center:+.3f} "
                        f"SKIP: {p.skip_reason}")
    if n_f > 1:
        for i, f in enumerate(freqs):
            n_ok = sum(1 for p in points if p.freq_ok is not None and p.freq_ok[i])
            log.info(f"  {f / 1e9:g} GHz: usable at {n_ok}/{len(points)} grid point(s)")


def _parse_args():
    """The CLI. Rule: the config says what the experiment IS; the CLI says what to do
    with this invocation.

    provenance.cli_args across 90 saved runs shows only --config, --run-name, --jobs,
    --summary and --scatter were ever set, so the surface is 12 flags rather than 20:
    --output-dir/--run-name merged into -o/--out; --seed and --max-iters dropped (every
    shipped config sets them); --scene-z-planes and --fps demoted to module constants;
    --summary/--scatter made unconditional; --no-save deleted; --skip-true-mgs inverted
    to an opt-in --true-mgs.

    --snr-db/--noise-seed are a deliberate exception to dropping --seed: SNR is the
    knob that gets swept from the shell. Like --engine and --freq they override the
    config, and the effective values land in the manifest's noise block and the
    config snapshot, so the run still records what the experiment was.
    """
    parser = argparse.ArgumentParser(
        description="Grid search over speculative TX locations, running MGS at each "
                    "to emit candidate beams")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG,
                        help="Path to a simulation config .yml with a grid_search block")
    parser.add_argument("--out", "-o", type=Path, default=None,
                        help="Run directory to write into, named outright "
                             "(default: <output.output_dir>/<output.run_name>)")
    parser.add_argument("--freq", "-f", type=float, nargs="+", default=None,
                        help="One or more frequencies in Hz (overrides config `frequencies`). "
                             "Multiple values run ONE joint sweep: each candidate is solved "
                             "once, for a single phase mask, against all of them. "
                             "Default: config `frequencies`, else "
                             f"{DEFAULT_FREQ_HZ / 1e9:g}e9.")
    parser.add_argument("--limit", type=int, default=None,
                        help="Only run the first N usable candidates (for quick tests)")
    parser.add_argument("--jobs", "-j", type=int, default=os.cpu_count(),
                        help="Worker processes for the candidate sweep and scene rendering "
                             "(default: all cores; 1 = serial). Per-candidate results are "
                             "identical regardless of value.")
    parser.add_argument("--engine", choices=["python", "rust"], default=None,
                        help="Solver engine (overrides gerchberg_saxton.engine; default: "
                             "the config's, python unless set). 'rust' needs the "
                             "rice_bend_core extension (rust/README.md).")
    parser.add_argument("--snr-db", type=parse_snr_db_arg, default=None, metavar="DB|off",
                        help="Peak-referenced SNR in dB of the complex AWGN added to the "
                             "simulated RX measurement (overrides noise.snr_db); 'off' "
                             "forces a noiseless run")
    parser.add_argument("--noise-seed", type=int, default=None,
                        help="Override noise.seed for a reproducible noise draw")
    parser.add_argument("--dry-run", action="store_true",
                        help="Enumerate the grid and print a summary without running MGS")
    parser.add_argument("--replot", type=Path, default=None,
                        help="Re-plot a saved run dir (single- or multi-frequency), then exit")
    parser.add_argument("--true-mgs", action="store_true",
                        help="Also recompute the baseline MGS run at the KNOWN TX location "
                             "(true_mgs_scene.png) -- one full MGS solve per run dir. "
                             "On a measured config, or a simulated one with "
                             "gerchberg_saxton.amplitude descend or level: one solve at "
                             "the measured / true TX position, as a candidate "
                             "(true_mgs.json/.npz)")
    parser.add_argument("--scenes", action="store_true",
                        help="Render one PNG per candidate beam (scenes/) plus an averaged "
                             "scene. On a measured run: no real-beam panel, the measured "
                             "TX window outlined")
    parser.add_argument("--scene-top", type=int, default=None,
                        help="[scenes/anim] only render the N lowest-residual candidates "
                             f"(--anim defaults to {ANIM_TOP_DEFAULT})")
    parser.add_argument("--scene-freq", type=float, default=None,
                        help="[scenes/anim/true-mgs] which frequency's view to render, in "
                             "Hz (must be one of the run's frequencies). Scene "
                             "re-illumination is monochromatic, so a multi-frequency run "
                             "needs one picked; default: the centre frequency.")
    parser.add_argument("--anim", action="store_true",
                        help="Animate the candidate beams to candidate_beams.mp4 (ffmpeg)")
    parser.add_argument("--surface-anim", action="store_true",
                        help="Also orbit each residual surface into "
                             "residual_surface_orbit.mp4 (+hc) -- the way past a "
                             "viewing angle that hides the peak (ffmpeg)")
    parser.add_argument("--plot-window", type=float, nargs=4, default=None,
                        metavar=("XMIN", "XMAX", "ZMIN", "ZMAX"),
                        help="Show only this (x, z) rectangle, in metres in the solver "
                             "frame (RX centre at 0, 0; TX at +z), in the residual "
                             "heatmap/scatter/surface and the --scenes PNGs (equal "
                             "aspect). Overrides the config's plot_window, and the "
                             "window a run recorded on --replot (the only way to "
                             "re-window a saved run)")
    parser.add_argument("--debug", action="store_true", help="Enable debug logs")
    return parser.parse_args()


def main():
    args = _parse_args()
    log = setup_logging(args.debug)
    try:
        cli_window = _cli_plot_window(args)
    except ValidationError as e:
        problems = []
        for err in e.errors():
            msg = err["msg"]
            if msg.startswith("Value error, "):     # str.removeprefix is 3.9+
                msg = msg[len("Value error, "):]
            where = ".".join(str(p) for p in err["loc"])
            problems.append(f"{where}: {msg}" if where else msg)
        log.error("invalid --plot-window XMIN XMAX ZMIN ZMAX "
                  f"{' '.join(f'{v:g}' for v in args.plot_window)}: "
                  + "; ".join(problems))
        raise SystemExit(2)

    if args.replot is not None:
        if args.snr_db is not None or args.noise_seed is not None:
            # a replot re-reads a finished run; --true-mgs replays it with the
            # noise its manifest recorded, so an override here would be a lie
            log.warning("--snr-db/--noise-seed are ignored with --replot: a replot "
                        "(including --true-mgs) uses the noise recorded in the saved run")
        base = Path(args.replot)
        freq_index = load_frequencies_index(base)
        if freq_index is None:
            _replot_one_dir(base, args, log, cli_window)          # joint (flat) run
            return
        # LEGACY per-frequency layout (pre-joint solver): each freq_<GHz>/ subdir
        # is a self-contained flat run, so their plots regenerate fine. The retired
        # top-level averaged/3D plots are NOT regenerated — they would present
        # independent per-frequency solves as if they were one joint solve.
        for e in freq_index["frequencies"]:
            _replot_one_dir(base / e["dir"], args, log, cli_window)
        log.info(f"{base} uses the retired per-frequency layout; regenerated plots "
                 f"in {len(freq_index['frequencies'])} freq_<GHz>/ subdir(s). For the "
                 "joint residual plots, re-run grid-search-mgs: the solver now fits "
                 "one phase mask across all frequencies in a single run.")
        return

    config = load_config(args.config)
    if cli_window is not None:
        # into the config, so the plots, the manifest's plot_window and the config
        # snapshot all name the window the run was drawn with
        config.plot_window = cli_window
    # resolved first: a measured config's geometry is checked against the data
    # file's columns at these frequencies (a pure function of the config and --freq)
    freqs = resolve_frequencies(config, args.freq)
    measured = config.measured is not None
    if measured:
        # before the noise overrides (an --snr-db assigned onto a measured config
        # would bypass the validator that refuses it) and before the grid or the
        # scene is used, the dry run included: the TX / RX blocks and the grid's
        # aperture width are derived here
        _refuse_measured_flags(args, args.config, log)
        try:
            apply_measured(config, freqs)
        except MeasuredDataError as e:
            log.error(f"{args.config}: {e}")
            raise SystemExit(2)
    if args.engine is not None:
        config.gerchberg_saxton.engine = args.engine
    try:
        apply_noise_overrides(config, args.snr_db, args.noise_seed)
    except ValidationError as e:
        log.error("invalid noise override: " + "; ".join(
            f"{'.'.join(str(p) for p in err['loc'])}: {err['msg']}" for err in e.errors()))
        raise SystemExit(2)
    if config.grid_search is None:
        log.error(f"Config {args.config} has no `grid_search` block")
        raise SystemExit(2)
    try:
        # a solver setting this run cannot honour, refused before the sweep rather
        # than after it: level on several tones under init: warm_start. No solve of
        # this run captures GS history at a fitted amplitude: with amplitude descend
        # or level, --true-mgs is the candidate-style solve at the TX on a simulated
        # config as on a measured one (_true_mgs_as_candidate), and the capturing
        # `mgs` baseline runs only on a fixed-amplitude simulated config
        check_amplitude_mode(config.gerchberg_saxton, n_tones=len(freqs))
    except ValueError as e:
        log.error(f"{args.config}: {e}")
        raise SystemExit(2)
    if (not measured and config.gerchberg_saxton.amplitude == "fixed"
            and config.tx_aperture.amplitude != 1.0):
        # allowed, but said once, up front: a fixed-amplitude candidate is the unit
        # box, so on a simulated plate scaled by s != 1 no candidate can match the
        # data's power, the true position's included. The measured path has no
        # simulated plate to scale (its amplitude_scale is the data's own)
        s = config.tx_aperture.amplitude
        log.warning(
            f"tx_aperture.amplitude {s if isinstance(s, str) else f'{s:g}'} with "
            "gerchberg_saxton.amplitude 'fixed': the candidates hold the unit-box "
            "amplitude, which cannot match the data's scale s, so this comparison is a "
            "deliberate wrong-power test (amplitude descend or level fit the scale)")

    run_name =config.output.run_name or "grid_search"
    if args.out is not None:
        config.output.output_dir = Path(args.out).parent
        run_name = Path(args.out).name

    if args.dry_run:
        _dry_run(config, freqs, log)
        return

    # Fail fast on a run-directory collision: the directory is created only once the
    # sweep has finished, so without this the clash surfaces after hours of compute.
    check_run_dir(config.output.output_dir, run_name, kind="grid")

    if len(freqs) > 1:
        log.info(f"Joint run across {len(freqs)} frequencies "
                 f"({', '.join(f'{f / 1e9:g}' for f in freqs)} GHz): one solve per "
                 "candidate, one shared phase mask")
    try:
        run = run_grid_search(config, freqs, limit=args.limit, jobs=args.jobs, log=log)
    except MeasuredDataError as e:
        # a measured row that cannot be used (a non-finite cell, a data file
        # rewritten since the geometry was derived): raised while reading the RX
        # values, before any candidate is solved
        log.error(f"{args.config}: {e}")
        raise SystemExit(2)
    # created only once the sweep has finished, so an interrupted run never
    # destroys prior results without producing new ones
    out_dir = make_run_dir(config.output.output_dir, run_name, kind="grid")
    persist_and_plot(run, out_dir, config, args, log)


if __name__ == "__main__":
    main()
