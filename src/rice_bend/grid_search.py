"""Entry point for `grid-search-mgs`: orchestration and CLI.

The work lives in three modules this one drives, and which never import each other:
  grid_sweep       -- enumerate the grid, solve, persist   (no matplotlib)
  residual_plots   -- the residual grid and its six plots
  candidate_scenes -- re-illuminated candidate beams, and the true-MGS baseline
"""

import argparse
import os
from pathlib import Path
from typing import List, Optional, Tuple

from rice_bend import rs
from rice_bend.candidate_scenes import (ANIM_FPS, ANIM_TOP_DEFAULT, ANIM_WARN_FRAMES,
                                        SCENE_Z_PLANES, animate_candidate_beams,
                                        make_candidate_scenes, make_true_mgs_plot,
                                        scenes_from_manifest, scenes_from_run)
from rice_bend.cli import setup_logging
from rice_bend.config import (DEFAULT_CONFIG, SimConfig, load_config,
                              resolve_frequencies)
from rice_bend.data_store import check_run_dir, make_run_dir
from rice_bend.grid_sweep import (GridSearchRun,
                                  enumerate_grid, grid_summary, load_frequencies_index,
                                  run_grid_search, save_grid_run)
from rice_bend.residual_plots import (ResidualSummary, animate_residual_surface,
                                      freq_summaries_from_manifest,
                                      freq_summaries_from_run,
                                      plot_residual_freq_vs_joint,
                                      plot_residual_heatmap, plot_residual_scatter,
                                      plot_residual_scatter_3d,
                                      plot_residual_scatter_3d_diff,
                                      plot_residual_surface, summary_from_manifest,
                                      summary_from_run)

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


def _emit_heatmaps(summary: ResidualSummary, out_dir: Path, log) -> None:
    """Write the residual heatmap plus its high-contrast (log-scale) twin."""
    out = _emit_pair(plot_residual_heatmap, out_dir / "residual_heatmap.png", summary)
    log.info(f"Wrote residual heatmap (+hc) to {out}")


def _emit_scatters(summary: ResidualSummary, out_dir: Path, log) -> None:
    """Write the residual scatter plus its high-contrast (log-y) twin."""
    out = _emit_pair(plot_residual_scatter, out_dir / "residual_scatter.png", summary)
    log.info(f"Wrote residual scatter (+hc) to {out}")


def _emit_surfaces(summary: ResidualSummary, out_dir: Path, args, log) -> None:
    """Write the residual surface plus its high-contrast twin, and optionally the orbit.

    The stills are manifest-only and as cheap as the heatmaps, so they always run. The
    orbit mp4 is opt-in for two reasons: write_mp4 raises when ffmpeg is missing, so an
    unconditional video would make the whole command unrunnable on a box without it;
    and at ~12 s each (measured, 48x48 grid), one per variant per frequency directory
    is ~4 min added to a 10-frequency run -- and to every --replot of it, which
    otherwise finishes in seconds.
    """
    out = _emit_pair(plot_residual_surface, out_dir / "residual_surface.png", summary)
    log.info(f"Wrote residual surface (+hc) to {out}")
    if args.surface_anim:
        orbit = _emit_pair(animate_residual_surface,
                           out_dir / "residual_surface_orbit.mp4", summary, log)
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


def _emit_scenes_and_anim(make_ctx, out_dir: Path, args, log) -> None:
    """Per-candidate scenes and/or the candidate animation, if asked for.

    `make_ctx` is a THUNK, not a SceneContext: scenes_from_manifest reads every
    candidate npz (2304 of them on scenario_caustic_hit), so constructing one
    eagerly only to discover that both flags are off would be a real regression.
    """
    if not (args.scenes or args.anim):
        return
    ctx = make_ctx()
    if args.scenes:
        make_candidate_scenes(ctx, out_dir, z_planes=SCENE_Z_PLANES,
                              top=args.scene_top, jobs=args.jobs, log=log)
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
                    out_dir: Path, args, log, make_ctx) -> None:
    """Everything one run directory gets.

    One rule: the cheap manifest-only plots always run; anything that re-solves MGS
    or renders per-candidate PNGs is opt-in. A fresh run and a --replot of it now
    produce the same set of files, which was not true when --summary/--scatter
    gated the fresh path and --replot ignored them.

    `summary` is the joint residual (what the solver minimized); `freq_summaries`
    its per-frequency components, driving the F>1 extras. A single-frequency run
    (or a legacy schema-2 replot, whose component list is empty) keeps exactly the
    flat plot set.
    """
    _emit_heatmaps(summary, out_dir, log)
    _emit_scatters(summary, out_dir, log)
    _emit_surfaces(summary, out_dir, args, log)
    if len(freq_summaries) > 1:
        _emit_multifreq_extras(summary, freq_summaries, out_dir, log)
    if args.true_mgs:
        # baseline MGS run at the KNOWN TX location: one full solve per run dir
        make_true_mgs_plot(out_dir, log=log)
    _emit_scenes_and_anim(make_ctx, out_dir, args, log)


def _replot_one_dir(run_dir: Path, args, log) -> ResidualSummary:
    """Regenerate one saved run directory's plots from disk."""
    run_dir = Path(run_dir)
    summary = summary_from_manifest(run_dir)
    freq_summaries = freq_summaries_from_manifest(run_dir)
    _emit_run_plots(summary, freq_summaries, run_dir, args, log,
                    lambda: scenes_from_manifest(run_dir))
    return summary


def _persist_and_plot(run: GridSearchRun, out_dir: Path, config: SimConfig,
                      args, log) -> ResidualSummary:
    """Save a finished sweep into out_dir and emit its plots."""
    save_grid_run(run, out_dir, config, args.config, vars(args))
    log.info(f"Saved {len(run.candidates)} candidate beam(s) to {out_dir}")
    summary = summary_from_run(run)
    freq_summaries = freq_summaries_from_run(run)
    _emit_run_plots(summary, freq_summaries, out_dir, args, log,
                    lambda: scenes_from_run(run))
    return summary


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
                             "Multiple values run the whole sweep independently per frequency. "
                             "Default: config `frequencies`, else 150e9.")
    parser.add_argument("--limit", type=int, default=None,
                        help="Only run the first N usable candidates (for quick tests)")
    parser.add_argument("--jobs", "-j", type=int, default=os.cpu_count(),
                        help="Worker processes for the candidate sweep and scene rendering "
                             "(default: all cores; 1 = serial). Per-candidate results are "
                             "identical regardless of value.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Enumerate the grid and print a summary without running MGS")
    parser.add_argument("--replot", type=Path, default=None,
                        help="Re-plot a saved run dir (single- or multi-frequency), then exit")
    parser.add_argument("--true-mgs", action="store_true",
                        help="Also recompute the baseline MGS run at the KNOWN TX location "
                             "(true_mgs_scene.png) -- one full MGS solve per run dir")
    parser.add_argument("--scenes", action="store_true",
                        help="Render one PNG per candidate beam (scenes/) plus an averaged scene")
    parser.add_argument("--scene-top", type=int, default=None,
                        help="[scenes/anim] only render the N lowest-residual candidates "
                             f"(--anim defaults to {ANIM_TOP_DEFAULT})")
    parser.add_argument("--anim", action="store_true",
                        help="Animate the candidate beams to candidate_beams.mp4 (ffmpeg)")
    parser.add_argument("--surface-anim", action="store_true",
                        help="Also orbit each residual surface into "
                             "residual_surface_orbit.mp4 (+hc) -- the way past a "
                             "viewing angle that hides the peak (ffmpeg)")
    parser.add_argument("--debug", action="store_true", help="Enable debug logs")
    return parser.parse_args()


def main():
    args = _parse_args()
    log = setup_logging(args.debug)

    if args.replot is not None:
        base = Path(args.replot)
        freq_index = load_frequencies_index(base)
        if freq_index is None:
            _replot_one_dir(base, args, log)          # joint (flat) run
            return
        # LEGACY per-frequency layout (pre-joint solver): each freq_<GHz>/ subdir
        # is a self-contained flat run, so their plots regenerate fine. The retired
        # top-level averaged/3D plots are NOT regenerated — they would present
        # independent per-frequency solves as if they were one joint solve.
        for e in freq_index["frequencies"]:
            _replot_one_dir(base / e["dir"], args, log)
        log.info(f"{base} uses the retired per-frequency layout; regenerated plots "
                 f"in {len(freq_index['frequencies'])} freq_<GHz>/ subdir(s). For the "
                 "joint residual plots, re-run grid-search-mgs: the solver now fits "
                 "one phase mask across all frequencies in a single run.")
        return

    config = load_config(args.config)
    if config.grid_search is None:
        log.error(f"Config {args.config} has no `grid_search` block")
        raise SystemExit(2)

    run_name = config.output.run_name or "grid_search"
    if args.out is not None:
        config.output.output_dir = Path(args.out).parent
        run_name = Path(args.out).name
    freqs = resolve_frequencies(config, args.freq)

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
    run = run_grid_search(config, freqs, limit=args.limit, jobs=args.jobs, log=log)
    # created only once the sweep has finished, so an interrupted run never
    # destroys prior results without producing new ones
    out_dir = make_run_dir(config.output.output_dir, run_name, kind="grid")
    _persist_and_plot(run, out_dir, config, args, log)


if __name__ == "__main__":
    main()
