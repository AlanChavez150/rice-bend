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

# Noise: checks 1 and 5 run tiny_check.yml, which has no noise block, so they pin the
# NOISELESS path (and the runtime lambda/20 RX sampling). Checks 2-4 and 6 run scenario
# configs, which carry noise.snr_db 30 / seed 0, so they pin the receiver-noise path
# instead: their residuals and hashes depend on the seeded draw, i.e. on numpy's
# Generator stream (rice_bend/noise.py) as well as on the solver.
#
# Receiver and start, per check (all of 2-4 and 6 use one 80-element lambda/2 array):
#   1, 5   interpolated, lambda/20 at runtime (rx_aperture.dx: null)   legacy start
#   2      ON the solver grid (rx_aperture.grid_stride: 2)             legacy start
#   3, 4   interpolated, fixed pitch (rx_aperture.dx: 0.000499654)     legacy start
#   6      ON the solver grid (rx_aperture.grid_stride: 2)             init: backprop,
#                                                                      grid_search.refine
# scenario_caustic_hit_sparse.yml (checks 3 and 4) stays on the interpolated receiver on
# purpose: it is the only noisy run of that path here (see its header).

# 1. mgs on the tiny config: solver stop state, captured-history indices, npz key set.
#    Deliberately UNPINNED: tiny_check.yml lists two frequencies (280 and 300 GHz), so
#    this exercises the joint multi-frequency solve through the single-run entry point.
#    Noiseless; stops on max_iters after 40 iterations.
run $MGS --config "$C/tiny_check.yml" -o "$S/c1_mgs_tiny"

# 2. mgs at full scene resolution — the only full-size single solve; covers the caustic
#    branch. PINNED to one frequency: the config lists ten, and an unpinned run would
#    be a 10x full-resolution joint solve — minutes added for no extra coverage (check 1
#    already pins the joint path). Noisy (30 dB, seed 0), 80-element lambda/2 RX array
#    ON the solver grid, legacy start (at one tone, the seeded random draw); stops on
#    the config's 800-iteration cap (under a 5000 cap the same solve converges after
#    1488).
run $MGS --config "$C/scenario_caustic_hit.yml" --freq 300e9 -o "$S/c2_mgs_caustic_hit"

# 3. the sweep, serial. Noisy (30 dB, seed 0): one draw, shared by every candidate.
#    Interpolated receiver, legacy start, no refinement: a schema-3 manifest.
run $GRID --config "$C/scenario_caustic_hit_sparse.yml" --freq 300e9 \
    --limit 6 --jobs 1 -o "$S/c3_grid_j1"

# 4. the same sweep across 4 workers — pins seeding, worker ordering and dtype at once
#    (and that the workers all see the parent's one noise draw).
run $GRID --config "$C/scenario_caustic_hit_sparse.yml" --freq 300e9 \
    --limit 6 --jobs 4 -o "$S/c4_grid_j4"

# 5. multi-frequency JOINT run under the DEFAULT (delay) phase model: one flat layout
#    (no frequencies.json, no freq_<GHz>/ subdirs), the joint + per-frequency loss
#    decomposition, the mean-combination rule, and the index -> (z, x_center) map that
#    pins the z-outer/x-inner ordering. Noiseless (tiny_check.yml).
run $GRID --config "$C/tiny_check.yml" --freq 280e9 300e9 -o "$S/c5_multifreq"

# 6. the caustic campaign base config, as it stands (one tone, 300 GHz, 30 dB, seed 0,
#    on-grid receiver), serial and then across 4 workers, the pattern of checks 3 and
#    4. It is the only check whose config sets gerchberg_saxton.init: backprop and
#    grid_search.refine, so it is what puts the data-driven start, the sweep's second
#    start and the refinement pass in the net, and the schema-4 manifest they write.
#    On these six candidates (the first six of the grid, far from the TX) one sweep
#    solve is stopped by the flatness test and solved again from the random start, and
#    all six fall within the refinement's band and are re-solved from both starts under
#    its 6000 cap. That re-solve is most of this check's time (about half a minute
#    serial). The parallel run pins that the refinement, like the sweep, does not
#    depend on the worker count.
run $GRID --config "$C/scenario_caustic_hit_1f_grid2l.yml" \
    --limit 6 --jobs 1 -o "$S/c6_refine_j1"
run $GRID --config "$C/scenario_caustic_hit_1f_grid2l.yml" \
    --limit 6 --jobs 4 -o "$S/c6_refine_j4"

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
