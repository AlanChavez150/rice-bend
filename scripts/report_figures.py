"""Figures and rank-of-truth numbers for the final report (docs/final_report_new/).

Reads the four frequency-study summaries (coarse 64x48 and fine 164x76 grids,
lambda/20 and lambda/2 receivers) and, for the fine-grid runs, each point's
candidate manifest. Writes into docs/final_report_new/:

  error_vs_bw_pitch.png    both receivers on the fine grid, argmin + top-mean
  error_coarse_vs_fine.png per-receiver coarse-vs-fine top-mean comparison
  rank_of_truth.json       rank of the truth-nearest cell by loss, per point

The shipped study plots omit the argmin series for frequency studies (Stage
13e), and the deck's pitch-comparison overlay had no generator; both report
figures live here instead.
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from rice_bend.plotting import add_wavelength_axis

REPO = Path(__file__).resolve().parent.parent
REF_FREQ_HZ = 150e9

STUDIES = {
    "lambda20_coarse": REPO / "results/study_frequency",
    "lambda20_fine": REPO / "results/study_frequency_grid2l",
    "lambda2_coarse": REPO / "results/study_frequency_lambda2",
    "lambda2_fine": REPO / "results/study_frequency_lambda2_grid2l",
}

X_LABEL = "bandwidth (± % of 150 GHz); 0 = single tone"
Y_LABEL = "top candidates mean distance to true TX (mm)"


def load_points(root: Path) -> list:
    study = json.loads((root / "study.json").read_text())
    return [p for p in study["points"] if p.get("x_value") is not None]


def bw_pct(point: dict) -> float:
    return float(point["param_value"])  # already in ± percent of 150 GHz


def annotate_n(ax, xs, ys, points, color) -> None:
    for x, y, p in zip(xs, ys, points):
        if p.get("n_top_candidates"):
            ax.annotate(f"n={p['n_top_candidates']}", (x, y),
                        textcoords="offset points", xytext=(6, -11),
                        fontsize=7, alpha=0.8, color=color)


def plot_pitch_comparison(out_path: Path) -> None:
    """Fine-grid overlay of the two receiver pitches, top candidates mean."""
    fig, ax = plt.subplots(figsize=(9, 6), layout="constrained")
    for key, color, pitch in (("lambda20_fine", "C0", "λ/20"),
                              ("lambda2_fine", "C3", "λ/2")):
        points = load_points(STUDIES[key])
        xs = [bw_pct(p) for p in points]
        top = [p["top_mean_dist_mm"] for p in points]
        ax.plot(xs, top, marker="s", color=color, label=f"{pitch} receiver")
        annotate_n(ax, xs, top, points, color)
    ax.set_ylim(0, 60)
    ax.set_xlabel(X_LABEL)
    ax.set_ylabel(Y_LABEL)
    add_wavelength_axis(ax, REF_FREQ_HZ, axis="y", unit_m=1e-3)
    ax.set_title("Frequency study: receiver element pitch comparison, 164x76 grid")
    ax.grid(True, alpha=0.3)
    ax.legend(framealpha=0.9)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"wrote {out_path}")


def plot_coarse_vs_fine(out_path: Path) -> None:
    """Stacked per-pitch panels: 64x48 vs 164x76 top-candidates mean."""
    fig, axes = plt.subplots(2, 1, figsize=(9, 8), layout="constrained",
                             sharex=True)
    panels = (
        (axes[0], "λ/20 receiver", "lambda20_coarse", "lambda20_fine", "C0"),
        (axes[1], "λ/2 receiver", "lambda2_coarse", "lambda2_fine", "C3"),
    )
    for ax, title, coarse_key, fine_key, color in panels:
        for key, style in ((coarse_key, dict(linestyle="--", marker="o",
                                             markerfacecolor="none", alpha=0.7,
                                             label="64x48 grid")),
                           (fine_key, dict(linestyle="-", marker="s",
                                           label="164x76 grid"))):
            points = load_points(STUDIES[key])
            xs = [bw_pct(p) for p in points]
            ys = [p["top_mean_dist_mm"] for p in points]
            ax.plot(xs, ys, color=color, **style)
            if key == fine_key:
                annotate_n(ax, xs, ys, points, color)
        ax.set_ylim(0, 60)
        ax.set_ylabel(Y_LABEL)
        add_wavelength_axis(ax, REF_FREQ_HZ, axis="y", unit_m=1e-3)
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.legend(framealpha=0.9)
    axes[1].set_xlabel(X_LABEL)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"wrote {out_path}")


def rank_of_truth(root: Path) -> list:
    """Per point: where the truth-nearest cell ranks among all candidates by
    final loss. Not stored in any analysis.json — recomputed from manifests."""
    out = []
    for point in load_points(root):
        run_dir = root / point["run_dir"]
        manifest = json.loads((run_dir / "candidate_beams.json").read_text())
        truth = manifest["ground_truth"]
        tz = truth["real_tx_z"]
        tx = 0.5 * (truth["real_tx_x_min"] + truth["real_tx_x_max"])
        cands = [c for c in manifest["candidates"]
                 if c.get("final_loss") is not None]
        nearest = min(cands, key=lambda c: (c["z"] - tz) ** 2
                      + (c["x_center"] - tx) ** 2)
        rank = 1 + sum(1 for c in cands if c["final_loss"] < nearest["final_loss"])
        analysis = json.loads((run_dir / "analysis.json").read_text())
        n_finite = analysis["loss"]["n_finite_cells"]
        if n_finite != len(cands):
            print(f"WARNING {run_dir.name}: {len(cands)} finite candidates "
                  f"vs analysis n_finite_cells={n_finite}")
        dist_mm = 1e3 * ((nearest["z"] - tz) ** 2
                         + (nearest["x_center"] - tx) ** 2) ** 0.5
        out.append({
            "bandwidth_pct": bw_pct(point),
            "rank": rank,
            "n_candidates": len(cands),
            "percentile": 100.0 * rank / len(cands),
            "nearest_cell": {"z": nearest["z"], "x_center": nearest["x_center"],
                             "dist_to_truth_mm": dist_mm},
            "argmin_dist_mm": point["error_mm"],
            "argmin_loss": point["argmin"]["final_loss"],
        })
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path,
                        default=REPO / "docs/final_report_new",
                        help="Output directory (default: docs/final_report_new)")
    parser.add_argument("--skip-ranks", action="store_true",
                        help="Skip the rank-of-truth pass (reads 20 manifests)")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    plot_pitch_comparison(args.out / "error_vs_bw_pitch.png")
    plot_coarse_vs_fine(args.out / "error_coarse_vs_fine.png")

    if args.skip_ranks:
        return
    ranks = {key: rank_of_truth(STUDIES[key])
             for key in ("lambda20_coarse", "lambda20_fine",
                         "lambda2_coarse", "lambda2_fine")}
    rank_path = args.out / "rank_of_truth.json"
    rank_path.write_text(json.dumps(ranks, indent=2) + "\n")
    print(f"wrote {rank_path}")
    for key, rows in ranks.items():
        print(f"\n{key}: truth-cell rank by loss")
        for r in rows:
            print(f"  bw {r['bandwidth_pct']:4.0f}%  rank {r['rank']:4d}"
                  f"/{r['n_candidates']}  ({r['percentile']:.2f}%)"
                  f"  nearest {r['nearest_cell']['dist_to_truth_mm']:.2f} mm"
                  f"  argmin {r['argmin_dist_mm']:.2f} mm")


if __name__ == "__main__":
    main()
