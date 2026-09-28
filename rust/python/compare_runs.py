#!/usr/bin/env python
"""Compare two saved grid-search runs of the same config — typically the python
engine vs the rust engine — on everything a study consumes.

    .venv/bin/python rust/python/compare_runs.py <python_run_dir> <rust_run_dir>

Exits 0 when the runs agree under the parity bar (rust/README.md): every
candidate's final_loss within rtol 1e-3, the same argmin cell, and the same
top-candidate set. n_iters/stop_reason and the localization metrics are reported.
"""
import json
import sys
from pathlib import Path

import numpy as np

RTOL = 1e-3


def load(run_dir: Path):
    with open(run_dir / "candidate_beams.json") as f:
        manifest = json.load(f)
    with open(run_dir / "analysis.json") as f:
        analysis = json.load(f)
    engine = "?"
    snap = run_dir / "config_snapshot.json"
    if snap.exists():
        with open(snap) as f:
            engine = json.load(f).get("gerchberg_saxton", {}).get("engine", "python")
    cands = {c["index"]: c for c in manifest["candidates"]}
    return cands, analysis, engine


def main():
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    a_dir, b_dir = Path(sys.argv[1]), Path(sys.argv[2])
    a, a_an, a_eng = load(a_dir)
    b, b_an, b_eng = load(b_dir)
    print(f"A: {a_dir}  (engine {a_eng}, {len(a)} candidates)")
    print(f"B: {b_dir}  (engine {b_eng}, {len(b)} candidates)")
    ok = True

    if set(a) != set(b):
        print(f"FAIL candidate index sets differ ({len(set(a) ^ set(b))} mismatched)")
        sys.exit(1)

    rel, same_iters, same_stop = [], 0, 0
    for i in a:
        la, lb = a[i]["final_loss"], b[i]["final_loss"]
        rel.append(abs(la - lb) / abs(la) if la else abs(lb))
        same_iters += a[i]["n_iters_run"] == b[i]["n_iters_run"]
        same_stop += a[i]["stop_reason"] == b[i]["stop_reason"]
    rel = np.array(rel)
    worst = max(a, key=lambda i: abs(a[i]["final_loss"] - b[i]["final_loss"]) / abs(a[i]["final_loss"]))
    n_bad = int((rel > RTOL).sum())
    ok &= n_bad == 0
    print(f"\nper-candidate final_loss: max rel diff {rel.max():.2e} (#{worst}), "
          f"median {np.median(rel):.2e}, bit-identical {int((rel == 0).sum())}/{len(rel)}, "
          f"over rtol {RTOL}: {n_bad}  [{'ok' if n_bad == 0 else 'FAIL'}]")
    print(f"n_iters identical {same_iters}/{len(a)}, stop_reason identical "
          f"{same_stop}/{len(a)}  (report-only)")

    am_a, am_b = a_an["argmin"], b_an["argmin"]
    same_argmin = am_a["index"] == am_b["index"]
    ok &= same_argmin
    print(f"\nargmin: #{am_a['index']} (z={am_a['z']:.4f}, x={am_a['x_center']:+.4f}) vs "
          f"#{am_b['index']} (z={am_b['z']:.4f}, x={am_b['x_center']:+.4f})  "
          f"[{'ok' if same_argmin else 'FAIL'}]")

    top_a = sorted(c["index"] for c in a_an["top_candidates"])
    top_b = sorted(c["index"] for c in b_an["top_candidates"])
    same_top = top_a == top_b
    ok &= same_top
    print(f"top candidates: {top_a} vs {top_b}  [{'ok' if same_top else 'FAIL'}]")

    for label, key in (("argmin error to truth", ("argmin", "error_distance_m")),):
        va, vb = a_an[key[0]][key[1]], b_an[key[0]][key[1]]
        print(f"{label}: {va * 1e3:.2f} mm vs {vb * 1e3:.2f} mm")
    ta, tb = a_an.get("top_mean_dist_to_truth_m"), b_an.get("top_mean_dist_to_truth_m")
    if ta is not None and tb is not None:
        print(f"top-cluster mean distance to truth: {ta * 1e3:.2f} mm vs {tb * 1e3:.2f} mm")

    print(f"\nCOMPARE: {'PASS' if ok else 'FAIL'}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
