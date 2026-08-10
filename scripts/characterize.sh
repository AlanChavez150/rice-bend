#!/usr/bin/env bash
#
# Characterization harness — the repo has no tests, so this is the net.
#
# Runs a fixed command list into a scratch directory, then hands the result to
# scripts/characterize.py, which extracts a numeric digest and diffs it against
# scripts/characterize_expected.json.
#
#   scripts/characterize.sh           # run + diff (non-zero exit on any difference)
#   scripts/characterize.sh --bless   # run + regenerate the expected digest
#
# Env:
#   PYTHON=...              interpreter to use (default: ./.venv/bin/python, else python3)
#   CHARACTERIZE_SCRATCH=…  keep output in a directory you choose (default: mktemp -d, removed on success)
#
# Each command runs with the scratch dir as its cwd, so the config's `plot_path`
# figure lands there instead of in the repo root.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

BLESS=""
if [ "${1:-}" = "--bless" ]; then
    BLESS="--bless"
    shift
fi

if [ -z "${PYTHON:-}" ]; then
    if [ -x "$REPO_ROOT/.venv/bin/python" ]; then
        PYTHON="$REPO_ROOT/.venv/bin/python"
    else
        PYTHON="$(command -v python3)"
    fi
fi

KEEP_SCRATCH=1
if [ -z "${CHARACTERIZE_SCRATCH:-}" ]; then
    CHARACTERIZE_SCRATCH="$(mktemp -d -t rice-bend-characterize-XXXXXX)"
    KEEP_SCRATCH=0
fi
S="$CHARACTERIZE_SCRATCH"
mkdir -p "$S"

export MPLBACKEND=Agg

MGS="$PYTHON -m rice_bend.mgs"
GRID="$PYTHON -m rice_bend.grid_search"
C="$REPO_ROOT/configs"

echo "characterize: scratch $S"
echo "characterize: python  $PYTHON"

run() {
    echo "--- $*"
    ( cd "$S" && "$@" ) > "$S/$(echo "$1_$2" | tr -c 'A-Za-z0-9' '_').log" 2>&1 \
        || { echo "COMMAND FAILED: $*"; tail -40 "$S"/*.log; exit 1; }
}

# 1. mgs on the tiny config: solver stop state, captured-history indices, npz key set.
#    Deliberately UNPINNED: tiny_check.yml lists two frequencies, so this exercises
#    the joint multi-frequency solve through the single-run entry point.
run $MGS --config "$C/tiny_check.yml" -o "$S/c1_mgs_tiny"

# 2. mgs at full scene resolution — the only full-size check; covers the caustic branch.
#    PINNED to one frequency: the config lists ten, and an unpinned run would be a
#    10x full-resolution joint solve — minutes added for no extra coverage (check 1
#    already pins the joint path).
run $MGS --config "$C/scenario_caustic_hit.yml" --freq 150e9 -o "$S/c2_mgs_caustic_hit"

# 3. the sweep, serial.
run $GRID --config "$C/scenario_caustic_hit_sparse.yml" --freq 150e9 \
    --limit 6 --jobs 1 -o "$S/c3_grid_j1"

# 4. the same sweep across 4 workers — pins seeding, worker ordering and dtype at once.
run $GRID --config "$C/scenario_caustic_hit_sparse.yml" --freq 150e9 \
    --limit 6 --jobs 4 -o "$S/c4_grid_j4"

# 5. multi-frequency JOINT run: one flat layout (no frequencies.json, no freq_<GHz>/
#    subdirs), the joint + per-frequency loss decomposition, the mean-combination rule,
#    and the index -> (z, x_center) map that pins the z-outer/x-inner ordering.
run $GRID --config "$C/tiny_check.yml" --freq 140e9 150e9 -o "$S/c5_multifreq"

echo "---"
set +e
"$PYTHON" "$REPO_ROOT/scripts/characterize.py" "$S" $BLESS
status=$?
set -e

if [ "$status" -eq 0 ] && [ "$KEEP_SCRATCH" -eq 0 ]; then
    rm -rf "$S"
else
    echo "characterize: scratch kept at $S"
fi
exit "$status"
