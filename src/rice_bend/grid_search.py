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
from rice_bend.config import DEFAULT_CONFIG, SimConfig, load_config
from rice_bend.data_store import check_run_dir, make_run_dir
from rice_bend.grid_sweep import (GridSearchRun, _freq_dir_name, _resolve_frequencies,
                                  enumerate_grid, grid_summary, load_frequencies_index,
                                  run_grid_search, save_grid_run, write_frequencies_index)
from rice_bend.residual_plots import (ResidualSummary, animate_residual_surface,
                                      average_summary, plot_residual_freq_vs_avg,
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


def _emit_multifreq_plots(summaries: List[Tuple[float, ResidualSummary]],
                          base_dir: Path, args, log) -> None:
    """Top-level plots for a multi-frequency run: the frequency-averaged residual
    heatmap and surface and the 3D residual scatter, each with a high-contrast
    (log-scale) twin."""
    avg = average_summary(summaries)
    avg_out = _emit_pair(plot_residual_heatmap, base_dir / "residual_heatmap_avg.png",
                         avg, title="Average residual across frequencies")
    log.info(f"Wrote frequency-averaged residual heatmap (+hc) to {avg_out}")
    avg_surf = _emit_pair(plot_residual_surface, base_dir / "residual_surface_avg.png",
                          avg, title="Average residual surface across frequencies")
    log.info(f"Wrote frequency-averaged residual surface (+hc) to {avg_surf}")
    if args.surface_anim:
        avg_orbit = _emit_pair(animate_residual_surface,
                               base_dir / "residual_surface_avg_orbit.mp4", avg, log,
                               title="Average residual surface across frequencies")
        log.info(f"Wrote frequency-averaged residual-surface orbit (+hc) to {avg_orbit}")
    out3d = _emit_pair(plot_residual_scatter_3d,
                       base_dir / "residual_scatter_3d.png", summaries)
    log.info(f"Wrote 3D residual scatter (+hc) to {out3d}")
    if len(summaries) > 1:
        out3d_diff = base_dir / "residual_scatter_3d_diff.png"
        plot_residual_scatter_3d_diff(summaries, out3d_diff)
        log.info(f"Wrote 3D residual-difference scatter to {out3d_diff}")
    # Per-frequency vs-average comparison (freq | average | difference) in each freq dir.
    for freq_hz, s in summaries:
        sub = base_dir / _freq_dir_name(freq_hz)
        if sub.is_dir():
            _emit_pair(plot_residual_freq_vs_avg,
                       sub / "residual_heatmap_vs_avg.png", freq_hz, s, avg)
    log.info(f"Wrote {len(summaries)} per-frequency vs-average comparison(s) (+hc) under {base_dir}")


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


def _emit_run_plots(summary: ResidualSummary, out_dir: Path, args, log, make_ctx) -> None:
    """Everything one run directory gets.

    One rule: the cheap manifest-only plots always run; anything that re-solves MGS
    or renders per-candidate PNGs is opt-in. A fresh run and a --replot of it now
    produce the same set of files, which was not true when --summary/--scatter
    gated the fresh path and --replot ignored them.
    """
    _emit_heatmaps(summary, out_dir, log)
    _emit_scatters(summary, out_dir, log)
    _emit_surfaces(summary, out_dir, args, log)
    if args.true_mgs:
        # baseline MGS run at the KNOWN TX location: one full solve per run dir
        make_true_mgs_plot(out_dir, log=log)
    _emit_scenes_and_anim(make_ctx, out_dir, args, log)


def _replot_one_dir(run_dir: Path, args, log) -> ResidualSummary:
    """Regenerate one saved run directory's plots from disk."""
    run_dir = Path(run_dir)
    summary = summary_from_manifest(run_dir)
    _emit_run_plots(summary, run_dir, args, log, lambda: scenes_from_manifest(run_dir))
    return summary


def _persist_and_plot(run: GridSearchRun, out_dir: Path, config: SimConfig,
                      args, log) -> ResidualSummary:
    """Save a finished sweep into out_dir and emit its plots."""
    save_grid_run(run, out_dir, config, args.config, vars(args))
    log.info(f"Saved {len(run.candidates)} candidate beam(s) to {out_dir}")
    summary = summary_from_run(run)
    _emit_run_plots(summary, out_dir, args, log, lambda: scenes_from_run(run))
    return summary


def _dry_run(config: SimConfig, freqs: List[float], log) -> None:
    """Enumerate the grid and print it, without running any MGS."""
    # the enumeration depends on frequency only through the RS-undersampling check
    if len(freqs) > 1:
        log.info(f"Dry run: enumerating grid at {freqs[0] / 1e9:g} GHz "
                 f"(of {len(freqs)} frequencies)")
    points = enumerate_grid(config.grid_search, config.sim_scene, rs.wavelength(freqs[0]))
    log.info(grid_summary(points))
    for p in points:
        if p.ok:
            log.info(f"  #{p.index:4d} z={p.z:.3f} x_center={p.x_center:+.3f} "
                     f"window [{p.x_min:.3f}, {p.x_max:.3f}]")
        else:
            log.warning(f"  #{p.index:4d} z={p.z:.3f} x_center={p.x_center:+.3f} "
                        f"SKIP: {p.skip_reason}")


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
            _replot_one_dir(base, args, log)          # single-frequency (flat) run
            return
        # multi-frequency: every per-frequency subdir, then the top-level plots
        summaries = [(float(e["freq_hz"]), _replot_one_dir(base / e["dir"], args, log))
                     for e in freq_index["frequencies"]]
        _emit_multifreq_plots(summaries, base, args, log)
        return

    config = load_config(args.config)
    if config.grid_search is None:
        log.error(f"Config {args.config} has no `grid_search` block")
        raise SystemExit(2)

    run_name = config.output.run_name or "grid_search"
    if args.out is not None:
        config.output.output_dir = Path(args.out).parent
        run_name = Path(args.out).name
    freqs = _resolve_frequencies(config, args.freq)

    if args.dry_run:
        _dry_run(config, freqs, log)
        return

    # Fail fast on a run-directory collision: the directory is created only once the
    # sweep has finished, so without this the clash surfaces after hours of compute.
    check_run_dir(config.output.output_dir, run_name, kind="grid")

    if len(freqs) == 1:
        # single frequency keeps the flat layout: results/<run_name>/...
        run = run_grid_search(config, freqs[0], limit=args.limit, jobs=args.jobs, log=log)
        out_dir = make_run_dir(config.output.output_dir, run_name, kind="grid")
        _persist_and_plot(run, out_dir, config, args, log)
        return

    base_path = Path(config.output.output_dir) / run_name
    log.info(f"Multi-frequency run: {len(freqs)} frequencies "
             f"({', '.join(f'{f / 1e9:g}' for f in freqs)} GHz) -> {base_path}")

    base: Optional[Path] = None

    def _get_base() -> Path:
        """The multi-frequency base directory, created (and cleared) exactly once.

        Both halves are load-bearing. Deferring the creation to the first COMPLETED
        sweep means an interrupted run never destroys prior results without
        producing new ones. Memoizing it means the second frequency does not rmtree
        the freq_140GHz directory just written -- and clearing it at all is what
        stops a re-run with a shorter frequency list from leaving a stale
        freq_<GHz> dir beside an average computed from a different sweep.
        """
        nonlocal base
        if base is None:
            base = make_run_dir(config.output.output_dir, run_name, kind="grid")
        return base

    entries, summaries = [], []
    real_x = real_z = 0.0
    for freq in freqs:
        log.info(f"=== frequency {freq / 1e9:g} GHz ===")
        run = run_grid_search(config, freq, limit=args.limit, jobs=args.jobs, log=log)
        out_dir = make_run_dir(_get_base(), _freq_dir_name(freq), kind="grid")
        summary = _persist_and_plot(run, out_dir, config, args, log)
        entries.append({"freq_hz": float(freq), "wavelength_m": float(run.wavelength),
                        "dir": _freq_dir_name(freq)})
        summaries.append((freq, summary))
        real_x, real_z = summary.real_tx_x_center, summary.real_tx_z
    write_frequencies_index(_get_base(), entries, real_z, real_x)
    _emit_multifreq_plots(summaries, _get_base(), args, log)


if __name__ == "__main__":
    main()
