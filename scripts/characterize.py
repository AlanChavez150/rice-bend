#!/usr/bin/env python
"""Extract a numeric digest from a characterization scratch directory and diff it
against scripts/characterize_expected.json.

Driven by scripts/characterize.sh, which runs the fixed command list that produces
the scratch directory. Kept separate from the shell script so the command list stays
readable as a command list.

The expected values are GENERATED (`--bless`), never transcribed by hand: hand-typed
float literals rot the moment someone stops re-blessing them.

    python scripts/characterize.py SCRATCH_DIR              # diff, exit 1 on any change
    python scripts/characterize.py SCRATCH_DIR --bless      # overwrite the expected file
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

EXPECTED = Path(__file__).resolve().parent / "characterize_expected.json"


# --------------------------------------------------------------------------- #
# Digest primitives
# --------------------------------------------------------------------------- #
def npz_keys(path: Path):
    """Sorted array names in an .npz."""
    with np.load(path) as z:
        return sorted(z.files)


def npz_content_hash(path: Path) -> str:
    """Stable hash of an .npz's CONTENT (names, dtypes, shapes, raw bytes).

    Not a hash of the file: np.savez_compressed stamps the local wall clock into
    each zip entry, so two byte-identical runs produce different files. This hashes
    what the arrays actually are, which is the invariant stages 1-6 must hold.
    """
    h = hashlib.sha256()
    with np.load(path) as z:
        for name in sorted(z.files):
            a = np.ascontiguousarray(z[name])
            h.update(name.encode())
            h.update(str(a.dtype).encode())
            h.update(str(a.shape).encode())
            h.update(a.tobytes())
    return h.hexdigest()


def file_list(run_dir: Path, skip_dirs=("candidates",)):
    """Sorted relative paths under run_dir, with the bulky candidates/ subtree
    replaced by a count (the names there are mechanical; the count is the fact)."""
    files, skipped = [], 0
    for p in sorted(run_dir.rglob("*")):
        if not p.is_file():
            continue
        rel = p.relative_to(run_dir)
        if any(part in skip_dirs for part in rel.parts):
            skipped += 1
            continue
        files.append(str(rel))
    return files, skipped


def read_json(path: Path):
    with open(path) as f:
        return json.load(f)


def manifest_without_run_identity(manifest: dict) -> dict:
    """A manifest stripped of the two fields that legitimately differ between two
    runs of the same sweep: the absolute run directory and the CLI that produced it."""
    m = json.loads(json.dumps(manifest))
    m.pop("run_dir", None)
    m.get("provenance", {}).pop("cli_args", None)
    return m


# --------------------------------------------------------------------------- #
# The five checks
# --------------------------------------------------------------------------- #
def check1_mgs_tiny(d: Path) -> dict:
    """`mgs` on tiny_check: the solver's stop state, the captured-history indices and
    the exact set of arrays a run persists."""
    meta = read_json(d / "run.json")
    with np.load(d / "run.npz") as z:
        iter_indices = [int(v) for v in z["gs_iter_indices"]]
    files, _ = file_list(d)
    return {
        "final_loss": float(meta["gs_result"]["final_loss"]),
        "n_iters_run": int(meta["gs_result"]["n_iters_run"]),
        "stop_reason": meta["gs_result"]["stop_reason"],
        "seed": meta["gerchberg_saxton"]["seed"],
        "gs_iter_indices": iter_indices,
        "npz_keys": npz_keys(d / "run.npz"),
        "npz_content_sha256": npz_content_hash(d / "run.npz"),
        "files": files,
    }


def check2_mgs_caustic_hit(d: Path) -> dict:
    """`mgs` at full scene resolution: the only full-size check, and the one that
    covers the caustic ground-truth branch end to end."""
    meta = read_json(d / "run.json")
    with np.load(d / "run.npz") as z:
        aper_norm = float(np.linalg.norm(z["gs_tx_aper_profile"]))
    files, _ = file_list(d)
    return {
        "final_loss": float(meta["gs_result"]["final_loss"]),
        "n_iters_run": int(meta["gs_result"]["n_iters_run"]),
        "stop_reason": meta["gs_result"]["stop_reason"],
        "gs_tx_aper_profile_norm": aper_norm,
        "npz_keys": npz_keys(d / "run.npz"),
        "npz_content_sha256": npz_content_hash(d / "run.npz"),
        "files": files,
    }


def _grid_digest(d: Path) -> dict:
    """Digest of one joint (schema-3) grid run — the SAME shape whatever the
    frequency count, which is itself the invariant: a multi-frequency run is one
    flat directory now, not freq_<GHz>/ subdirs.

    `joint_is_mean_of_valid` pins the loss-combination rule: every candidate's
    joint residual must equal the mean of its per-frequency components over the
    frequencies that actually contributed (all of them, at F=1 — which is also
    the N=1-reduces-through-the-same-path guarantee).
    """
    m = read_json(d / "candidate_beams.json")
    files, n_cand_files = file_list(d)
    joint = [c["final_loss"] for c in m["candidates"]]
    per_freq = [c["per_freq_losses"] for c in m["candidates"]]
    joint_is_mean = all(
        j == float(np.mean([v for v in row if v is not None]))
        for j, row in zip(joint, per_freq)
    )
    return {
        "counts": m["counts"],
        "ground_truth": m["ground_truth"],
        "seed": m["seed"],
        "init": m["gs"].get("init"),
        "frequencies": [f["freq_hz"] for f in m["frequencies"]],
        "residuals": joint,
        "per_freq_losses": per_freq,
        "freq_valid": [c["freq_valid"] for c in m["candidates"]],
        "argmin": int(np.argmin(joint)) if joint else None,
        "joint_is_mean_of_valid": bool(joint_is_mean),
        "n_iters_run": [c["n_iters_run"] for c in m["candidates"]],
        "grid_map": [[c["index"], c["z"], c["x_center"]] for c in m["candidates"]],
        "cand_0003_sha256": npz_content_hash(d / "candidates" / "cand_0003.npz"),
        "measurement_sha256": npz_content_hash(d / "measurement.npz"),
        "files": files,
        "n_candidate_files": n_cand_files,
    }


def check3_grid_serial(d: Path) -> dict:
    """The sweep run serially: counts, ground truth and every residual."""
    return _grid_digest(d)


def check4_grid_parallel(serial_dir: Path, parallel_dir: Path) -> dict:
    """The same sweep across 4 workers. Covers seeding, worker ordering and dtype at
    once — Stage 3e (one map_workers replacing four pool copies) is the riskiest
    change in the refactor and this is what catches it."""
    ser = manifest_without_run_identity(read_json(serial_dir / "candidate_beams.json"))
    par = manifest_without_run_identity(read_json(parallel_dir / "candidate_beams.json"))
    return {
        "manifest_matches_serial": ser == par,
        "cand_0003_matches_serial": (npz_content_hash(serial_dir / "candidates" / "cand_0003.npz")
                                     == npz_content_hash(parallel_dir / "candidates" / "cand_0003.npz")),
        "digest": _grid_digest(parallel_dir),
    }


def check5_multifreq(d: Path) -> dict:
    """The joint multi-frequency run. Its digest is deliberately IDENTICAL in shape
    to the flat checks — no frequencies.json, no freq_<GHz>/ subdirs is precisely
    what the joint solver changed — with the 2-frequency loss decomposition pinned
    through per_freq_losses/freq_valid and the mean rule through
    joint_is_mean_of_valid. `grid_map` still pins enumerate_grid's z-outer/x-inner
    index -> (z, x_center) mapping, which nothing else in the digest does and which
    a module split could silently transpose."""
    return _grid_digest(d)


def build_digest(scratch: Path) -> dict:
    return {
        "check1_mgs_tiny": check1_mgs_tiny(scratch / "c1_mgs_tiny"),
        "check2_mgs_caustic_hit": check2_mgs_caustic_hit(scratch / "c2_mgs_caustic_hit"),
        "check3_grid_serial": check3_grid_serial(scratch / "c3_grid_j1"),
        "check4_grid_parallel": check4_grid_parallel(scratch / "c3_grid_j1",
                                                     scratch / "c4_grid_j4"),
        "check5_multifreq": check5_multifreq(scratch / "c5_multifreq"),
    }


# --------------------------------------------------------------------------- #
# Diff
# --------------------------------------------------------------------------- #
def diff(expected, actual, path="") -> list:
    """Recursive structural diff, returning one human-readable line per difference."""
    if type(expected) is not type(actual) and not (
            isinstance(expected, (int, float)) and isinstance(actual, (int, float))):
        return [f"{path or '<root>'}: type {type(expected).__name__} -> {type(actual).__name__}"]
    if isinstance(expected, dict):
        out = []
        for k in sorted(set(expected) | set(actual)):
            sub = f"{path}.{k}" if path else k
            if k not in expected:
                out.append(f"{sub}: ADDED ({actual[k]!r})")
            elif k not in actual:
                out.append(f"{sub}: REMOVED (was {expected[k]!r})")
            else:
                out += diff(expected[k], actual[k], sub)
        return out
    if isinstance(expected, list):
        if len(expected) != len(actual):
            return [f"{path}: length {len(expected)} -> {len(actual)}"]
        out = []
        for i, (e, a) in enumerate(zip(expected, actual)):
            out += diff(e, a, f"{path}[{i}]")
        return out
    if expected != actual:
        return [f"{path}: {expected!r} -> {actual!r}"]
    return []


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("scratch", type=Path, help="Scratch dir produced by characterize.sh")
    ap.add_argument("--bless", action="store_true",
                    help="Overwrite scripts/characterize_expected.json with this run")
    args = ap.parse_args()

    actual = build_digest(args.scratch)

    if args.bless:
        with open(EXPECTED, "w") as f:
            json.dump(actual, f, indent=2, sort_keys=True)
            f.write("\n")
        print(f"blessed -> {EXPECTED}")
        return 0

    if not EXPECTED.exists():
        print(f"No {EXPECTED}; run with --bless to record the baseline.", file=sys.stderr)
        return 1

    differences = diff(read_json(EXPECTED), actual)
    if not differences:
        print("characterize: OK (digest matches scripts/characterize_expected.json)")
        return 0
    print(f"characterize: {len(differences)} DIFFERENCE(S)\n", file=sys.stderr)
    for line in differences:
        print(f"  {line}", file=sys.stderr)
    print("\nIf the change is intended, re-run with --bless.", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
