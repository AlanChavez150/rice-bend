import logging
import argparse
import math
from collections import namedtuple
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
from pydantic import ValidationError

from rice_bend import beam_aperatures, rs
from rice_bend.cli import setup_logging
from rice_bend.config import (DEFAULT_CONFIG, DEFAULT_FREQ_HZ, GerchbergSaxtonConfig,
                              SimConfig, apply_noise_overrides, center_freq_index,
                              load_config, parse_snr_db_arg, resolve_frequencies)
from rice_bend.interp import interp_amplitude, interp_real_imag
from rice_bend.landing import (beam_target, field_match, landing_metrics,
                               propagator_offset)
from rice_bend.noise import add_awgn
from rice_bend.plotting import draw_line_panel, draw_scene
from rice_bend.data_store import GSHistory, check_run_dir, make_run_dir, save_run
from rice_bend.exp_data import parse_oscope_heatmap_data, parse_oscope_rx_data
from rice_bend.sim_scene import SimAperature, SimScene, sampled_axis


# One frequency's contribution to a joint solve: its measurement (on the scene
# x-axis, so every channel has the same length) and the weighting derived from it.
# Picklable, so a worker payload can carry a list of these.
FreqChannel = namedtuple("FreqChannel", "freq wavelength rx_field error_weighting")


# Warm-start stage 3: scan step for the absolute-offset search — fine enough to land
# well inside the joint basin; the joint descent polishes the remainder. The scan is
# elementwise work only (see _warm_start_phase), so a fine step costs nothing.
# The LEGACY scan's step, frozen with it; the corrected scan has its own
# (OFFSET_SCAN_STEP).
WARM_START_SCAN_STEP = 2 * np.pi / 64

# Backstop on offset samples for pathological near-duplicate frequency lists, whose
# synthetic period (2π·f_ref/Δf_min) explodes as the frequency gaps shrink. The
# LEGACY scan's cap, frozen with it; the corrected scan has its own
# (_offset_scan_max_samples), sized to its own step.
WARM_START_MAX_SAMPLES = 100_000

# The largest denominator comb_period looks for when it turns a tone ratio into a
# fraction, and so the length in cycles of the range it falls back on when the
# tones have no period that short. One number, because two things hang on it: the
# fallback range itself, and the corrected scan's sample cap, which is sized so
# that this many cycles always fit (_offset_scan_max_samples).
COMB_PERIOD_MAX_DEN = 1024

# Grid step of the corrected absolute-offset scan (_offset_scan), in rad of psi. The
# grid's job is to FIND the peaks of the score; which peak wins is decided after each
# has been polished to its own top, so the choice no longer rests on this number. It
# did before the polish, and the case says why it must not. A ±13 % comb (87:100:113)
# repeats exactly after 100 cycles and ALMOST after 77 (0.87 * 77 = 66.99,
# 1.13 * 77 = 87.01), which leaves two offsets that differ by 0.063 rad on each edge
# tone and by nothing on the centre one. At the true TX of the caustic scenario
# (30 dB) their peaks score 719.405 and 719.297: 1.5e-4 apart. A grid sample sits up
# to half a step off its peak. At the legacy 2π/64 that costs up to 1.2e-3 of the
# score, the plain grid maximum is the WRONG peak (sampled 718.557 against 718.604),
# the descent cannot cross from one to the other, and the solve ends at 2.28x the
# true plate's loss instead of 1.09. At 0.02 rad it costs up to 5e-5 and the grid
# maximum happens to be the right peak — on a third of the gap, which is not a
# margin to rest on. With the polish the same offset comes back from either grid, to
# 1e-13 rad, for every comb of the study and all three beams. What the step still
# has to do is put every peak on the grid as a local maximum well inside its own
# basin: peaks are about one cycle apart, so 314 samples each, and a 100-cycle period
# is 31,416 samples of F phasors — still free.
#
# (Those two scores are noise seed 0's. WHICH of the two peaks is the higher is the
# noise draw's choice — the other one in 22 of 48 seeds — and no grid or polish
# settles that: see OFFSET_NEAR_TIE_RTOL.)
OFFSET_SCAN_STEP = 0.02

# The polish that follows the grid scan (_offset_scan): how many of the grid's local
# maxima are carried to their own peak before the best is chosen, and how many
# Newton steps each gets. Thirty-two is far more than ever compete: over the study's
# combs and the three beams at most two peaks come within 1.2e-3 of the best (what
# the legacy step's misalignment can cost), and where a period holds more than 32
# peaks the 32nd is 29 % or more below it. Three steps from within half a grid step
# of a peak land on it to rounding: Newton converges quadratically, and from 0.01
# rad that is the offset to 1e-13 rad (the same from a 2π/64 grid, 0.049 rad off).
OFFSET_POLISH_CANDIDATES = 32
OFFSET_POLISH_STEPS = 3

# Near-tied offsets (_offset_candidates): a polished peak within this fraction of
# the best score is too close to the best for the scan to call, and
# gs_reconstruct_multistart descends from it as well — from at most
# OFFSET_NEAR_TIE_MAX offsets per start, the best included.
#
# Both numbers are measured (30 dB, on-grid receiver, back-propagated start unless
# said). What makes a near-tie is a comb that almost repeats inside its period: the
# runner-up is then the best offset's twin (77 or 23 cycles away at ±13 %, 43 at
# ±7 %, 33 at ±3 %, the adjacent cycle at ±1 %), 0.063 rad off on each edge tone,
# and the descent from the wrong one of the pair stays about 2x above the other.
#   - At the true TX of the caustic scenario at ±13 % the twin scores HIGHER than
#     the offset in the true plate's basin in 22 of 48 noise seeds, by up to 2.5e-4.
#     (±7 %: never in 48, closest 1.8e-4; ±3 %, ±1 % and the other two beams:
#     never, closest 4.8e-4.)
#   - At wrong locations the gap is wider. Over the 97-114 candidates the refinement
#     re-solves on a 21x21 patch round that TX at ±13 % (noise seeds 0, 1, 2), the
#     runner-up is the better basin for 22-25 of them — a final loss 1.07x to 1.93x
#     lower — at gaps from 2e-5 up to 1.3e-3.
#   - 1e-3 reaches 19-21 of those; the 3-5 it leaves end at 2.6x the run's minimum
#     or more either way. 5e-4 reaches 11-12 and 3e-4 only 6-7, and both leave out
#     members of the final 1.5x set (candidates at 1.25x to 1.48x the minimum).
#   - The price at 1e-3: a second descent that does not help for 40-42 of those
#     candidates (1.6 descents per start on average at ±13 %); at ±1 %, where the
#     runner-up was the better basin for none of 114, a second offset still comes
#     within 1e-3 for 91 of the patch's 441 candidates — 1.2 descents per start,
#     for nothing. At 2e-3 every candidate would get two and most three, at every
#     bandwidth. So 1e-3: the widest tolerance that does not charge the combs
#     without a close twin.
#   - What it bought on that patch: the candidate nearest the TX took the twin's
#     offset in all three seeds and ended at 1.62-1.77x the run's minimum; with
#     both offsets descended it ends at 1.42-1.51x (1.51x is where the right
#     offset stands when it runs into the refinement's 6000-iteration cap; it
#     reaches 1.42x by 40,000).
# OFFSET_NEAR_TIE_MAX is a backstop, not a tuning: at 1e-3 a third peak came within
# tolerance in none of these measurements (45 true-TX cases x 2 starts, six patch
# surveys of 441 candidates). It bounds the cost for a comb with several
# near-repeats.
OFFSET_NEAR_TIE_RTOL = 1e-3
OFFSET_NEAR_TIE_MAX = 3

# The data-driven start (_backprop_start). The back-propagated reference tone is
# "lit" where its amplitude is at least this fraction of its peak over the support;
# the phase is read between the first and the last lit sample and continued outside.
# 0 turns the threshold off (everything is lit: a plain unwrap over the whole
# support). What it buys and what it costs is measured in _backprop_start's
# docstring — a trade, not a clear win, so treat the value as a knob.
BACKPROP_LIT_FRACTION = 0.1

# ...and how the phase is continued outside the lit span: at the constant slope of a
# least-squares line through this many lit samples nearest the edge (fewer if the
# span is shorter). One sample pair would hand the edge's own noise to the whole
# dark stretch; this many average it out while staying local enough to follow a
# curved profile's slope AT the edge.
BACKPROP_SLOPE_FIT_SAMPLES = 24


def _warm_start_phase(tx_z, orig_aper_amp, x_axis, rx_z, channels, params, seed,
                      rho, ref_freq, h_fwd, support, curr_aper_amp, dx,
                      log=None) -> np.ndarray:
    """Multi-wavelength warm start: an initial psi inside the true basin.

    The joint delay solve must recover the ABSOLUTE profile — the 2π degeneracy
    that makes single-frequency retrieval easy is exactly what it breaks — and
    plain gradient descent from a random start lands in a wrong basin it cannot
    leave (measured: stuck ~100x above the reachable floor). The standard fix,
    per docs/delay_model_warm_start.md:

      1. solve the channel nearest the reference frequency ALONE — its degenerate
         landscape is easy, and it recovers the profile's SHAPE;
      2. unwrap that phase over the support — the per-point 2π ambiguities resolve
         relative to each other, collapsing the unknown to ONE scalar, the
         absolute offset delta;
      3. scan delta over one synthetic-wavelength period of the frequency comb
         against the joint loss — the comb makes the absolute offset observable
         (multi-wavelength interferometry), unique within one period. Propagation
         is linear, so each channel's field is propagated ONCE and every delta
         sample costs elementwise work only.

    The unwrap assumes a CONNECTED support and a smooth profile — true for every
    shipped scene (box candidate windows; the contiguous plate aperture).
    Returns the starting psi (zero off support).
    """
    # 1. reference solve. Pinning the PARENT's resolved seed keeps `seed: null`
    # runs reproducible end-to-end (the parent draws once; the child must not draw
    # its own); init='random' terminates the recursion.
    c_idx = int(np.argmin(np.abs(rho - 1.0)))
    ch_ref = channels[c_idx]
    sub_params = params.model_copy(update={"init": "random", "seed": int(seed)})
    stage1 = gs_reconstruct(tx_z=tx_z, orig_aper_amp=orig_aper_amp, x_axis=x_axis,
                            rx_z=rx_z, channels=[ch_ref], params=sub_params,
                            ref_freq=ch_ref.freq, capture=False, log=None)

    # 2. unwrap over the support and rescale into psi units (theta_c = rho_c * psi)
    psi0 = np.zeros(len(x_axis))
    idx = np.where(support)[0]
    psi0[idx] = np.unwrap(np.angle(stage1.curr_aper_f[idx])) / rho[c_idx]

    # 3. the absolute-offset scan. Adding delta multiplies channel f's field by
    # exp(1j*rho_f*delta); the joint loss is periodic in delta with the comb's
    # synthetic wavelength 2π·f_ref/Δf_min (the 2π fallback is defensive only —
    # the warm start requires F > 1 and frequencies are validated unique).
    freqs = np.sort(np.array([ch.freq for ch in channels]))
    gaps = np.diff(freqs)
    gaps = gaps[gaps > 0]
    period = 2 * np.pi * ref_freq / gaps.min() if gaps.size else 2 * np.pi
    n_samples = int(np.ceil(period / WARM_START_SCAN_STEP))
    if n_samples > WARM_START_MAX_SAMPLES:
        if log is not None:
            log.warning(f"warm start: synthetic period {period:.3g} rad needs "
                        f"{n_samples} offset samples; capping at {WARM_START_MAX_SAMPLES} "
                        "(nearly-duplicate frequencies?)")
        n_samples = WARM_START_MAX_SAMPLES
    deltas = np.linspace(0.0, period, n_samples, endpoint=False)

    base_props = [rs.rs_apply(curr_aper_amp * np.exp(1j * (rho[f] * psi0)),
                              h_fwd[f], dx)
                  for f in range(len(channels))]
    best_delta, best_loss = 0.0, np.inf
    for delta in deltas:
        loss = 0.0
        for f, ch in enumerate(channels):
            r = ch.error_weighting * (np.exp(1j * rho[f] * delta) * base_props[f]
                                      - ch.rx_field)
            loss += 0.5 * np.mean(np.abs(r) ** 2)
        loss /= len(channels)
        if loss < best_loss:
            best_loss, best_delta = loss, float(delta)

    if log is not None:
        log.info(f"warm start: stage-1 loss {stage1.final_loss:.4g} at "
                 f"{ch_ref.freq / 1e9:g} GHz; offset scan over {period:.1f} rad "
                 f"({n_samples} samples) -> delta {best_delta:.3f} "
                 f"(joint loss there {best_loss:.4g})")
    psi = psi0 + best_delta
    psi[~support] = 0.0
    return psi


# ---------------------------------------------------------------------------------
# The data-driven start (`init: backprop`) and the comb's TRUE offset period.
#
# _warm_start_phase above is the LEGACY start and is kept exactly as it was, its
# short scan included: `init: warm_start` has to reproduce every result already on
# disk bit for bit, on both engines (rust/src/warm_start.rs is its twin and is frozen
# with it). It has two limits, both measured at the true TX with an on-grid receiver
# at 30 dB as final loss / loss of the true plate's own phase, over the frequency
# study's three-tone combs (±1 % to ±20 %):
#
#   1. The scan range. 2π·f_ref/Δf_min is a period of the joint loss only when
#      f_ref/Δf is an integer. For tones (100-p):100:(100+p) the period is
#      2π·100/gcd(100, p), so the range is the period at ±1, 2, 5, 10 and 20 % and
#      a fraction of it at ±3, 7, 13 and 16 % — the four bandwidths at which the
#      focus beam fails (2.00, 5.02, 31.5, 16.8 against 1.01 to 1.04 at the others):
#      the offset that lines the tones up lies outside what was scanned.
#   2. The reference solve. Started from a random phase, the single-tone solve is
#      not unique where part of the beam misses the RX window: nothing measured
#      constrains the part of the plate that feeds the missing energy. The scaled
#      copy of what the solve returns there (rho_f * psi) misses the other tones,
#      and the joint descent stalls (bessel: 1.60 at ±1 % rising to 165 at ±20 %).
#
# The start below takes the phase from the DATA instead (the reference tone's
# measurement back-propagated to the candidate plane) and scans the offset over the
# true period; the ordinary descent follows, with no reference solve. Measured the
# same way — on the configs' noise draw, seed 0 — it ends at 0.95-1.11 (caustic),
# 0.93-1.00 (bessel) and 0.98-1.01 (focus) of the true plate's loss at all ten
# bandwidths, after 804-1714, 375-570 and 117-223 iterations (the legacy route's
# reference solve alone takes 1488, 1765 and 1424).
#
# Other noise draws (seeds 1-8, same single start, cap 5000, stop -1e-8): bessel
# and focus stay at or below 1.022 at every bandwidth. The caustic goes above 1.2
# at ±13 % for four of the eight seeds (2.115, 2.310, 1.225, 1.224) and at ±20 % for
# two (1.203, 1.270). The two near 2.2 are not a slow descent: the ±13 % comb
# nearly repeats after 77 cycles, the scan's two best offsets are a near-tie that
# the noise draw decides, and the single start took the one outside the true
# plate's basin (OFFSET_NEAR_TIE_RTOL). A single start cannot do better than
# that, and the single-shot paths (`mgs`, --true-mgs) and the sweep's first solve
# ARE a single start. gs_reconstruct_multistart, which the refinement runs on the
# sweep's near-best candidates, descends from every near-tied offset: under the
# refinement's settings (cap 6000, stop -1e-10) the caustic at ±13 % then ends at
# 0.92-1.13 over noise seeds 0-8, where the single start under the same settings
# ends at 0.93-2.21.
#
# The data-driven start is also not the better one at every WRONG location, which
# is the other reason gs_reconstruct_multistart exists: it runs this start against
# the random-start route (with the scan corrected) and keeps the lower loss.
# ---------------------------------------------------------------------------------
def comb_period(freqs: Sequence[float], ref_freq: float,
                max_den: int = COMB_PERIOD_MAX_DEN,
                rtol: float = 1e-9) -> Tuple[float, bool]:
    """The true period of the joint loss in the absolute offset: the smallest
    delta > 0 with rho_f * delta = 0 (mod 2π) for every tone, rho_f = f / ref_freq.
    Returns (period in rad, exact).

    Adding delta to psi multiplies tone f's field by exp(1j * rho_f * delta), so
    the joint loss repeats when — and only when — every one of those factors is
    back at 1 together. Each rho_f is taken as a fraction n_f / d_f in lowest
    terms. With Q = lcm(d_f) the ratios are m_f / Q for the integers
    m_f = n_f * Q / d_f, the factors are exp(1j * m_f * delta / Q), and the first
    delta that returns them all is 2π * Q / gcd(m_f).

    For the study's combs (100-p):100:(100+p) about the reference this is
    2π * 100 / gcd(100, p): 100 cycles of 2π at ±1, 3, 7 and 13 %, 50 at ±2 %, 25
    at ±16 %, 20 at ±5 %, 10 at ±10 % and 5 at ±20 %. The legacy scan range
    2π·f_ref/Δf_min = 2π * 100 / p agrees only where p divides 100; where it does
    not, it is not a period at all and the range it covers can miss the best offset.
    A single tone at the reference has period 2π, and a single tone off it (a
    candidate that solves on a subset of the run's tones) 2π / rho.

    The fractions come from Fraction.limit_denominator(max_den): tones are floats
    (338999999999.99994 Hz for +13 % of 300 GHz), so the ratio is matched to the
    nearest fraction with a small denominator, not read off the bits. A ratio that
    is not within `rtol` (relative) of that fraction has no period this short —
    then (2π * max_den, False) is returned: a range to scan, but not a period, and
    the caller should say so.
    """
    ratios = []
    for f in freqs:
        rho = float(f) / float(ref_freq)
        frac = Fraction(rho).limit_denominator(max_den)
        if frac <= 0 or abs(float(frac) - rho) > rtol * abs(rho):
            return 2 * np.pi * max_den, False
        ratios.append(frac)
    # Q = lcm of the denominators, folded pairwise: math.lcm needs python 3.9 and
    # math.gcd takes two arguments before it
    q = 1
    for frac in ratios:
        q = q * frac.denominator // math.gcd(q, frac.denominator)
    # gcd of the integers m_f = n_f * Q / d_f (gcd(0, m) == m starts the fold)
    g = 0
    for frac in ratios:
        g = math.gcd(g, frac.numerator * (q // frac.denominator))
    return 2 * np.pi * q / g, True


def _offset_scan_max_samples() -> int:
    """The corrected scan's cap on grid samples: what COMB_PERIOD_MAX_DEN cycles
    take at OFFSET_SCAN_STEP (321,700 at 0.02 rad), read at call time like the
    step itself.

    Sized that way because that is the longest range comb_period can ask for
    without the tones being near-duplicates: its fallback for ratios that are
    not small fractions is exactly COMB_PERIOD_MAX_DEN cycles, and a comb whose
    ratios ARE fractions with a common denominator up to it has a true period no
    longer. Under the legacy cap (WARM_START_MAX_SAMPLES, 100,000 — sized for a
    step five times coarser) the fallback was cut to its first third, and on
    the sweep's path, where workers have no logger, cut silently; and a scan at
    a finer step lost the end of an ordinary 100-cycle period (at 0.005 rad:
    125,664 samples). What still exceeds the cap is a true period LONGER than
    COMB_PERIOD_MAX_DEN cycles — tones whose ratios need several different large
    denominators, i.e. nearly duplicate frequencies — and that is truncated to
    the start of the range, with a WARNING where there is a logger
    (_scan_comb_offset)."""
    return int(np.ceil(2 * np.pi * COMB_PERIOD_MAX_DEN / OFFSET_SCAN_STEP))


def _offset_peaks(psi0: np.ndarray, channels: Sequence[FreqChannel], rho: np.ndarray,
                  h_fwd: Sequence[np.ndarray], curr_aper_amp: np.ndarray, dx: float,
                  period: float, periodic: bool = True
                  ) -> Tuple[np.ndarray, np.ndarray]:
    """The peaks of the absolute-offset score of psi0 over [0, period): a grid scan
    at OFFSET_SCAN_STEP to find them, a Newton polish of each to its own top.
    Returns (offsets, scores), best first — the highest score, and among equal
    scores the smallest offset. `period` is the range to cover (comb_period's);
    `periodic` says whether it is a true period of the loss (comb_period's
    `exact`), i.e. whether the range wraps. _offset_scan takes the first of
    these, _offset_candidates the first few when they are too close to call.

    Adding delta multiplies tone f's field by exp(1j * rho_f * delta) and
    propagation is linear, so with b_f the field of psi0 at the RX plane the loss
    of tone f is 0.5 * mean(|w_f * (exp(1j * rho_f * delta) * b_f - y_f)|^2).
    Expanding the square, only the cross term depends on delta:

        joint loss(delta) = const - S(delta) / (N * F),
        S(delta) = sum_f Re(c_f * exp(1j * rho_f * delta)),
        c_f = sum(conj(w_f^2 * y_f) * b_f)

    So each tone is propagated once, its whole contribution is the one complex
    number c_f, and the score S — whose maximum is the loss's minimum — is a sum
    of F phasors per sample. The comb's full period, up to 13 times the legacy
    range on the study's combs, costs nothing; the legacy loop rebuilds every
    tone's residual array per sample. The same closed form gives the derivatives
    the polish needs:

        S'(delta)  =  sum_f Re(1j * rho_f * c_f * exp(1j * rho_f * delta))
        S''(delta) = -sum_f Re(rho_f^2 * c_f * exp(1j * rho_f * delta))

    NO BLAS. All three are written as explicit sums of elementwise products: no
    `@`, np.dot, np.vdot or einsum. This runs once per start of every candidate,
    inside the sweep's worker processes, and a BLAS call there is not one thread:
    each worker inherits the parent's OpenBLAS, sized to every core, and nothing a
    worker can set afterwards shrinks it (parallel._init_worker). Written as
    (len(d), F) phasors `e` against the coefficients, `e @ c`, the score put all
    32 threads of every worker onto a matrix of three columns: 2.75 ms a call
    against 0.16 ms on one thread, and a three-tone 441-candidate patch at 4
    workers (rust engine) took 9 min 01 s of user and 10 min 57 s of system
    time, against 5 min 39 s and 9 s once the matmul was gone. F is
    at most ten, so a python loop over the tones costs nothing. np.vdot is out
    for the same reason, though it was not what was measured: at a scene of 2,400
    samples it stays on one thread here, at 24,000 it takes all 32, and where
    that line falls is the BLAS build's business.

    WHY THE POLISH. The grid's best sample is not the best offset: it sits up to
    half a step off its peak, and by a different amount at every peak. That
    error must not be what chooses between two peaks, and on its own it would
    be: a comb can repeat ALMOST exactly well inside its true period, leaving
    two offsets whose scores differ by less than the grid's misalignment costs
    (see OFFSET_SCAN_STEP). So the grid is only asked to FIND the peaks. The
    OFFSET_POLISH_CANDIDATES highest samples that are local maxima of the grid
    score are each moved to the top of their own peak by OFFSET_POLISH_STEPS
    Newton steps on S, and the offsets are compared there, where the grid no
    longer enters. A Newton step is clamped to one grid step (a candidate stays
    on the peak it was found on) and skipped where S'' >= 0 (not a maximum's
    neighbourhood: the step would head for a minimum); a candidate whose polish
    did not raise S keeps its grid sample, so the result is never worse than
    the plain grid maximum.

    The ends. With a true period the range wraps: sample 0 and the last sample
    are neighbours, a peak that straddles the wrap is one candidate, and the
    polished offset is returned in [0, period). Without one (`periodic` false,
    or the sample count capped at _offset_scan_max_samples for near-duplicate
    tones, when the grid covers the start of the range only) the ends are ends:
    an end sample is a candidate if it beats its one neighbour, and the polish
    is held inside the scanned range.

    Deterministic: no draw, and every choice among equals is by index.
    """
    rho = np.asarray(rho, dtype=np.float64)
    c = np.array([
        np.sum(np.conj(ch.error_weighting ** 2 * ch.rx_field)
               * rs.rs_apply(curr_aper_amp * np.exp(1j * (rho[f] * psi0)), h_fwd[f], dx))
        for f, ch in enumerate(channels)])

    def score_at(d):
        """S, S' and S'' at the offsets `d`, accumulated one tone at a time (see
        NO BLAS above). With t_f = c_f * exp(1j * rho_f * d):
        Re(1j * rho_f * t_f) = -rho_f * Im(t_f) and Re(rho_f^2 * t_f) =
        rho_f^2 * Re(t_f), so one phasor array per tone gives all three."""
        s0, s1, s2 = np.zeros(len(d)), np.zeros(len(d)), np.zeros(len(d))
        for f in range(len(c)):
            t = c[f] * np.exp(1j * (rho[f] * d))
            s0 += t.real
            s1 -= rho[f] * t.imag
            s2 -= rho[f] ** 2 * t.real
        return s0, s1, s2

    # 1. the grid. The step is read HERE, at call time, so a test can scan the
    # same score on another grid (the gate that the choice does not depend on it)
    step = OFFSET_SCAN_STEP
    n_full = int(np.ceil(period / step))
    n_samples = min(n_full, _offset_scan_max_samples())
    wraps = periodic and n_samples == n_full
    deltas = step * np.arange(n_samples)
    score = score_at(deltas)[0]

    # 2. the candidates: grid local maxima, the highest first. Strict on one side
    # only, so a flat pair of samples is one candidate and not two; a tie in
    # height keeps the smaller offset first (stable sort on an ascending index)
    left, right = np.roll(score, 1), np.roll(score, -1)
    if not wraps:
        left[0] = right[-1] = -np.inf
    peaks = np.flatnonzero((score > left) & (score >= right))
    if peaks.size == 0:
        # a constant score (every c_f zero: a dark candidate) has no peak; any
        # offset is as good, and the first is the one a plain argmax would return
        peaks = np.array([int(np.argmax(score))])
    cand = peaks[np.argsort(-score[peaks], kind="stable")[:OFFSET_POLISH_CANDIDATES]]

    # 3. the polish: Newton on S from each candidate's grid sample
    d = deltas[cand].copy()
    for _ in range(OFFSET_POLISH_STEPS):
        _, s1, s2 = score_at(d)
        concave = s2 < 0
        move = np.zeros_like(d)
        move[concave] = np.clip(-s1[concave] / s2[concave], -step, step)
        d = d + move
    if not wraps:
        # an end is an end: the polish stays inside the range that was scanned
        d = np.clip(d, 0.0, deltas[-1])
    polished = score_at(d)[0]
    # a polish that did not raise S is dropped for the sample it started from, so
    # no candidate ends below the grid — nor the result below the grid maximum
    lower = polished < score[cand]
    d[lower], polished[lower] = deltas[cand][lower], score[cand][lower]
    if wraps:
        # a peak across the wrap comes back into [0, period) (np.mod can round a
        # tiny negative offset up to `period` itself, which is offset 0)
        d = np.mod(d, period)
        d[d >= period] = 0.0

    # 4. best first: the highest polished score; among equals, the smallest offset
    order = np.lexsort((d, -polished))
    return d[order], polished[order]


def _offset_scan(psi0: np.ndarray, channels: Sequence[FreqChannel], rho: np.ndarray,
                 h_fwd: Sequence[np.ndarray], curr_aper_amp: np.ndarray, dx: float,
                 period: float, periodic: bool = True) -> float:
    """The absolute offset delta that minimises the joint loss of psi0 + delta over
    [0, period): the best of _offset_peaks (the closed form, the grid, the polish
    and the ends are documented there) — the polished offset with the highest
    score, a tie going to the smallest offset. The ONE offset the sweep's solve
    and the single-shot paths start from (_backprop_phase)."""
    offsets, _ = _offset_peaks(psi0, channels, rho, h_fwd, curr_aper_amp, dx, period,
                               periodic=periodic)
    return float(offsets[0])


def _offset_candidates(psi0: np.ndarray, channels: Sequence[FreqChannel],
                       rho: np.ndarray, h_fwd: Sequence[np.ndarray],
                       curr_aper_amp: np.ndarray, dx: float, period: float,
                       periodic: bool = True) -> List[float]:
    """The offsets worth descending from: _offset_scan's best, followed by the
    peaks that score within OFFSET_NEAR_TIE_RTOL (relative to the best score) of
    it, best first, OFFSET_NEAR_TIE_MAX at most. One offset when nothing comes
    that close. The constants' comment has the measurements behind both numbers.

    WHY MORE THAN ONE. The score ranks offsets by the loss of the START phase,
    and near a tie that is not the order of what the descent reaches from them.
    A comb that nearly repeats inside its true period (see OFFSET_SCAN_STEP)
    leaves two offsets that differ by a few hundredths of a radian on the edge
    tones: their scores can be closer than 1e-4, the noise draw then decides
    which is higher, and only one of them is in the true plate's basin — the
    descent from the other stays about 2x above it at any iteration cap. No
    amount of care in the scan settles that from the start phase alone. The
    descent does: so where the scan cannot call it, gs_reconstruct_multistart
    descends from each of these and keeps the lowest final loss.
    """
    offsets, scores = _offset_peaks(psi0, channels, rho, h_fwd, curr_aper_amp, dx,
                                    period, periodic=periodic)
    near = scores[0] - scores <= OFFSET_NEAR_TIE_RTOL * abs(scores[0])
    near[0] = True     # the best is a candidate whatever its score is
    # `scores` is descending, so the near-tied ones are a leading run
    return [float(v) for v in offsets[near][:OFFSET_NEAR_TIE_MAX]]


def _backprop_field(ch_ref: FreqChannel, h_fwd_ref: np.ndarray, dx: float,
                    idx: np.ndarray) -> np.ndarray:
    """One tone's weighted measurement back-propagated to the candidate plane, at
    the support samples `idx`: the solver's own adjoint kernel applied to
    w^2 * y, the field whose phase _backprop_start reads. complex128: rs_apply
    returns complex64, and the unwrap that follows accumulates in float64."""
    g = rs.rs_apply(ch_ref.error_weighting ** 2 * ch_ref.rx_field,
                    np.conjugate(h_fwd_ref), dx)[idx]
    return g.astype(np.complex128)


def _lit_span(amp: np.ndarray) -> Optional[Tuple[int, int]]:
    """First and last sample at which a back-propagated amplitude `amp` reaches
    BACKPROP_LIT_FRACTION of its peak — the stretch of the plate that illuminates
    the RX window, as far as the data can tell. None when there is nothing to
    read: an empty support, or a back-propagation that is dark throughout."""
    if amp.size == 0 or not amp.max() > 0:
        return None
    lit = np.flatnonzero(amp >= BACKPROP_LIT_FRACTION * amp.max())
    return int(lit[0]), int(lit[-1])


def _line_fit(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """The least-squares line through (x, y) as [slope, intercept] — what
    np.polyfit(x, y, 1) returns, and np.polyval evaluates — from the closed form:
    slope = sum((x - mean x) * (y - mean y)) / sum((x - mean x)^2), and the line
    passes through the two means. Needs two or more samples at distinct x.

    Not np.polyfit, which solves the same two unknowns through LAPACK's
    least-squares driver: this is called twice per start of every candidate, in
    the sweep's worker processes, where a LAPACK call wakes the worker's whole
    inherited thread pool (see _offset_scan, NO BLAS). On 24 samples polyfit ran
    at 2.4 cores of CPU per core of wall time, a third of it system time.
    """
    x_mean, y_mean = np.mean(x), np.mean(y)
    xc = x - x_mean
    slope = np.sum(xc * (y - y_mean)) / np.sum(xc * xc)
    return np.array([slope, y_mean - slope * x_mean])


def _backprop_start(ch_ref: FreqChannel, h_fwd_ref: np.ndarray, dx: float,
                    support: np.ndarray, rho_ref: float) -> np.ndarray:
    """The data-driven starting psi: the unwrapped phase of the reference tone's
    measurement back-propagated to the candidate plane, in psi units (divided by
    `rho_ref`). float64, zero off the support.

    Why this is a start for EVERY tone and not only the one it was read from: a
    delay plate steers achromatically. Its phase at tone f is rho_f * psi, the
    wavenumber scales by the same rho_f, and the ray angle — phase slope over
    wavenumber — does not change with frequency. A smooth, ray-like profile that
    sends the reference tone's energy where it was measured therefore sends the
    other tones' energy to the same place. The back-propagated phase is such a
    profile, and its unwrap is the one continuous choice among its 2π-shifted
    copies — which a single tone cannot tell apart and several can. The legacy
    warm start gets its profile from a random-start solve of the reference tone
    instead, which is not unique where part of the beam misses the RX window.

    The part of the plate that does not illuminate the RX window carries no
    information: the back-propagated field is dark there, and its phase is noise
    and sidelobe leakage. So the phase is read only between the first and the
    last sample whose amplitude reaches BACKPROP_LIT_FRACTION of the peak
    (_lit_span), and CONTINUED outside: at the constant slope of a least-squares
    line through the nearest BACKPROP_SLOPE_FIT_SAMPLES lit samples, joined
    continuously at the edge — the plate keeps steering the way it does where it
    can be seen, rather than taking a phase from noise.

    What the threshold is worth, measured (30 dB, on-grid receiver, against a
    plain unwrap over the whole support, BACKPROP_LIT_FRACTION = 0): a trade.
    It only acts where the plate is partly dark. At the true TX that is the
    caustic alone (the last 10 of its 288 samples; the bessel and focus plates
    are lit throughout and the two starts are the same array), and there it
    COSTS a little: 0.95-1.11 of the true plate's loss over the ten bandwidths
    (noise seed 0), against 0.94-1.04 for the plain unwrap, in up to twice the
    iterations. Over
    22 candidate locations on and around the TX (single tone, ±10 %, ±20 %, at
    the re-solve's cap and stop threshold) it removed the plain unwrap's one
    early stall that matters — caustic, single tone, a candidate within 10x of
    the minimum: stopped after 67 iterations at 5.2x the thresholded start's
    loss — and caused three of its own, all at one far candidate whose loss is
    over 100x the minimum either way (1.6x to 1.8x the plain unwrap's);
    everywhere else the two end within 9 % of each other. It is NOT what keeps the
    multi-tone solve out of a wrong basin: a comparison that had the plain
    unwrap at 2.25x at ±13 % on the caustic had also changed the offset scan's
    grid, and the grid was the cause (see OFFSET_SCAN_STEP — with the offset
    chosen right the plain unwrap ends at 1.02 there, the thresholded start at
    1.09).

    An all-dark back-propagation (the measurement gives this candidate nothing to
    go on) returns zeros: a flat plate, from which the descent starts as it would
    from any other phase. Like the legacy unwrap this assumes a CONNECTED support
    (box candidate windows; the contiguous plate aperture).

    The absolute offset is not settled here: the other tones see rho_f times
    any common offset, which is what _offset_scan resolves over comb_period.
    """
    psi = np.zeros(len(support))
    idx = np.where(support)[0]
    g = _backprop_field(ch_ref, h_fwd_ref, dx, idx)
    span = _lit_span(np.abs(g))
    if span is None:
        return psi
    lo, hi = span
    ph = np.unwrap(np.angle(g[lo:hi + 1]))
    out = np.empty(len(idx))
    out[lo:hi + 1] = ph
    k = np.arange(len(idx), dtype=float)
    n_fit = min(BACKPROP_SLOPE_FIT_SAMPLES, hi - lo + 1)
    if n_fit >= 2:
        # each side: the line through the n_fit lit samples at that edge, shifted so
        # it passes THROUGH the edge sample (the fit's own value there is off by its
        # residual, and a step at the edge is a discontinuity the plate does not have)
        left = _line_fit(k[lo:lo + n_fit], ph[:n_fit])
        out[:lo] = np.polyval(left, k[:lo]) + (ph[0] - np.polyval(left, k[lo]))
        right = _line_fit(k[hi - n_fit + 1:hi + 1], ph[-n_fit:])
        out[hi + 1:] = np.polyval(right, k[hi + 1:]) + (ph[-1] - np.polyval(right, k[hi]))
    else:
        # a single lit sample has no slope to continue: hold its phase
        out[:lo] = ph[0]
        out[hi + 1:] = ph[-1]
    psi[idx] = out / rho_ref
    return psi


def _scan_comb_offset(psi0: np.ndarray, channels: Sequence[FreqChannel],
                      rho: np.ndarray, ref_freq: float, h_fwd: Sequence[np.ndarray],
                      curr_aper_amp: np.ndarray, dx: float, log=None,
                      near_ties: bool = False) -> Tuple[List[float], str]:
    """The offset scan over comb_period, for a multi-tone start: the offset(s) to
    add to psi0 (on the support), best first, and a phrase describing the scan
    for the caller's log line. One offset — _offset_scan's — unless `near_ties`,
    which adds the peaks too close to it to call (_offset_candidates).

    Warns — once, when `log` is given — if the range scanned is not the whole
    period: the tones have no period short enough to find, or it needs more
    samples than _offset_scan_max_samples."""
    period, exact = comb_period([ch.freq for ch in channels], ref_freq)
    if log is not None:
        n_needed = int(np.ceil(period / OFFSET_SCAN_STEP))
        n_max = _offset_scan_max_samples()
        if not exact:
            log.warning(
                "offset scan: the tones' ratios to the reference frequency "
                f"({ref_freq / 1e9:g} GHz) are not ratios of small integers, so the "
                "joint loss has no period short enough to scan in full; scanning "
                f"{period / (2 * np.pi):g} x 2π, which is NOT a period — the best "
                "offset may lie outside it")
        elif n_needed > n_max:
            log.warning(
                f"offset scan: the comb's period ({period / (2 * np.pi):g} x 2π) needs "
                f"{n_needed} offset samples; capping at {n_max}, so the best offset "
                "may lie outside the range scanned (nearly-duplicate frequencies?)")
    if near_ties:
        offsets = _offset_candidates(psi0, channels, rho, h_fwd, curr_aper_amp, dx,
                                     period, periodic=exact)
    else:
        offsets = [_offset_scan(psi0, channels, rho, h_fwd, curr_aper_amp, dx, period,
                                periodic=exact)]
    note = (f"offset scan over {period / (2 * np.pi):g} x 2π "
            f"({'exact period' if exact else 'NOT a period'}) -> delta "
            f"{offsets[0]:.3f} rad")
    if len(offsets) > 1:
        note += (f", with {len(offsets) - 1} near-tied offset(s) at "
                 + ", ".join(f"{d:.3f}" for d in offsets[1:]) + " rad")
    return offsets, note


def _backprop_phase(channels: Sequence[FreqChannel], rho: np.ndarray, ref_freq: float,
                    h_fwd: Sequence[np.ndarray], support: np.ndarray,
                    curr_aper_amp: np.ndarray, dx: float, log=None) -> np.ndarray:
    """The whole `init: backprop` start, the counterpart of _warm_start_phase:
    _backprop_start on the reference channel, then — for more than one channel —
    the absolute offset from _offset_scan over the comb's true period. No solve.
    Returns the starting psi (zero off the support).

    The reference channel is the one nearest rho == 1, as the legacy warm start
    picks it, and NOT "the channel at ref_freq": a grid candidate solves on the
    subset of the run's tones that pass its sampling check, against the run-level
    reference, so the reference tone itself may be missing. _backprop_start
    divides by that channel's own rho, so psi comes out in the run's units either
    way. A single channel gets no scan: with one tone a common offset is a global
    phase, which the descent corrects on its own, and there is no second tone to
    line it up with.
    """
    c_idx = int(np.argmin(np.abs(rho - 1.0)))
    ch_ref = channels[c_idx]
    psi = _backprop_start(ch_ref, h_fwd[c_idx], dx, support, rho[c_idx])
    scan_note = ""
    if len(channels) > 1:
        offsets, scan_note = _scan_comb_offset(psi, channels, rho, ref_freq, h_fwd,
                                               curr_aper_amp, dx, log=log)
        # on the support only: off it the amplitude is zero, the offset changes no
        # field, and psi stays zero there as everywhere else in the solver
        psi[support] += offsets[0]
        scan_note = "; " + scan_note
    if log is not None:
        # re-derived here rather than returned by _backprop_start: only a logged
        # (single-shot) solve wants it, and it is one convolution
        idx = np.where(support)[0]
        span = _lit_span(np.abs(_backprop_field(ch_ref, h_fwd[c_idx], dx, idx)))
        n_lit = 0 if span is None else span[1] - span[0] + 1
        rest = "" if n_lit == len(idx) else ", phase continued over the rest"
        log.info(f"backprop start: {ch_ref.freq / 1e9:g} GHz back-propagated, lit over "
                 f"{100 * n_lit / max(len(idx), 1):.1f} % of the support{rest}{scan_note}")
    return psi


# Result of one solver run. `curr_aper_f` is the reconstructed complex aperture on the
# math (scene) x-axis — the field at `ref_freq`; `history` is a GSHistory when
# capture=True, else None. `final_loss`/`loss_full` are the joint
# (mean-over-frequency) numbers; the `*_per_freq` twins carry each frequency's
# component, aligned with `frequencies` (the input channel order).
#
# `psi` is the solved unknown itself — the delay profile as phase at `ref_freq`, on
# the scene x-axis, zero off the support — and it is NOT recoverable from
# `curr_aper_f`: that is exp(1j*psi), which keeps psi only modulo 2π, while tone f's
# phase is (f/ref_freq)·psi and needs the absolute value. Set only when
# capture=True on the python engine (the single-shot path, where
# MGS.landing_report rebuilds every tone's field from it); None on the grid path
# and the rust engine, where nothing reads it and a sweep would carry one extra
# scene-length float array per candidate.
GSResult = namedtuple(
    "GSResult",
    "curr_aper_f final_loss final_loss_per_freq n_iters_run stop_reason seed "
    "loss_full loss_full_per_freq frequencies ref_freq history psi",
    defaults=(None,),
)


def rust_core():
    """The optional rust engine (rust/, module rice_bend_core), or a RuntimeError
    naming the build command. Never falls back to python silently: a run that
    asked for rust and got python would misattribute its timings and results."""
    try:
        import rice_bend_core
    except ImportError as e:
        raise RuntimeError(
            "gerchberg_saxton.engine is 'rust' but the rice_bend_core extension is "
            "not installed; build it with: VIRTUAL_ENV=$PWD/.venv .venv/bin/maturin "
            "develop --release -m rust/Cargo.toml") from e
    return rice_bend_core


def _gs_solve_rust(channels, params, ref_freq, seed, initial_phase, curr_aper_amp,
                   h_fwd, dx, log=None, init_mode: Optional[str] = None) -> "GSResult":
    """The warm start + descent of gs_reconstruct, run by rice_bend_core.gs_solve
    from the SAME seed draw, kernels and measurements the python path would use.
    Parity with the python engine: rust/python/parity.py, rust/README.md.

    `init_mode` is what rust is told to do with `initial_phase`, and it is passed
    explicitly rather than read off params.init: the rust side knows two modes
    only. 'warm_start' runs its own twin of _warm_start_phase from the phase
    (the seeded draw); 'random' descends from the phase as handed over, unchanged
    — which is how a start computed in python (a caller's initial_phase, the
    `init: backprop` start) reaches it. None means params.init, the legacy
    behaviour, valid for 'random' and 'warm_start' alone: 'backprop' forwarded
    as it stands is a mode rust rejects."""
    if init_mode is None:
        init_mode = params.init
    (curr_aper_f, final_loss, final_loss_pf, n_iters_run, stop_reason, loss_full,
     loss_full_pf) = rust_core().gs_solve(
        initial_phase, np.asarray(curr_aper_amp, dtype=np.float64),
        np.array([ch.freq for ch in channels], dtype=np.float64), float(ref_freq),
        np.stack([np.asarray(ch.rx_field, dtype=np.complex128) for ch in channels]),
        np.stack([np.asarray(ch.error_weighting, dtype=np.float64) for ch in channels]),
        np.stack(h_fwd).astype(np.complex128, copy=False), float(dx),
        init_mode, int(params.max_iters), int(params.convergence_count),
        float(params.convergence_threshold), float(params.lr0),
        float(params.bt_shrink), int(params.bt_tries))
    if log is not None:
        if stop_reason == "converged":
            log.info(f"MGS has converged after {n_iters_run} iterations")
        else:
            log.error(f"MGS did not converge after {params.max_iters} iterations")
    return GSResult(curr_aper_f=curr_aper_f, final_loss=float(final_loss),
                    final_loss_per_freq=final_loss_pf,
                    n_iters_run=int(n_iters_run), stop_reason=str(stop_reason),
                    seed=seed, loss_full=loss_full, loss_full_per_freq=loss_full_pf,
                    frequencies=tuple(float(ch.freq) for ch in channels),
                    ref_freq=float(ref_freq), history=None)


def gs_reconstruct(tx_z: float, orig_aper_amp: np.ndarray, x_axis: np.ndarray,
                   rx_z: float, channels: Sequence[FreqChannel],
                   params: GerchbergSaxtonConfig, ref_freq: Optional[float] = None,
                   capture: bool = False, log=None,
                   initial_phase: Optional[np.ndarray] = None) -> "GSResult":
    """Modified Gerchberg-Saxton solver core (no MGS/SimScene instance required).

    Solves for ONE aperture profile psi(x) at plane `tx_z` that best reproduces
    every channel's measured RX field at once, holding the amplitude fixed at
    `orig_aper_amp` (defined on `x_axis`; its nonzero region is the support).
    psi is a PLATE's shape — the optical-path/delay profile expressed as phase at
    `ref_freq` — and each channel's phase is (f/ref_freq)·psi, the way a physical
    plate's phase scales with wavenumber. This matches every simulated beam type
    (each generator bakes the wavenumber into the phase).

    The loss is the UNWEIGHTED MEAN of the per-frequency losses. Each channel's
    error_weighting is normalized to its own measurement's peak (see MGS.measure),
    so the mean weights frequencies equally regardless of absolute RX power; do
    not "fix" it into a power-weighted sum. A single frequency is simply the
    len(channels) == 1 case of the same code path (rho = 1.0 exactly, so psi IS
    that frequency's phase).

    `ref_freq` defaults to the centre-by-value of `channels`. Callers that solve
    different candidates on different channel SUBSETS (the grid sweep) must pass
    the run-level reference instead, so psi keeps the same units for every
    candidate.

    Pure given its arguments — arrays, scalars and a picklable config model — so
    this runs unchanged in a worker process.

    With `capture=True` a GSHistory is built and per-iteration state recorded, and
    the result carries the final psi (the single-shot `mgs` path, used by animation
    and the landing report); with `capture=False` only the dense loss curves + final
    aperture are produced (the grid-search path). The math is identical
    either way, so results do not depend on `capture`. `log` (optional) receives the
    per-iteration progress lines; workers pass None to stay quiet.

    Where the descent starts, in order of precedence:

    - `initial_phase` (an array on `x_axis`, in psi units): the plain descent from
      it, cast to float64 and zeroed off the support, with no warm start of any
      kind whatever `params.init` says. For a caller that has built its own start
      (gs_reconstruct_multistart).
    - `params.init == "backprop"`: the data-driven start (_backprop_phase) — the
      reference channel's measurement back-propagated and unwrapped, plus, for
      more than one channel, the absolute offset scanned over the comb's true
      period — then the plain descent. No reference-frequency solve.
    - `params.init == "warm_start"` (the default) and `"random"`: the LEGACY
      starts, bit-identical to what they have always produced. 'warm_start' is
      _warm_start_phase for more than one channel and the random draw for one;
      'random' is the seeded random draw.

    GSHistory.initial_phase (capture=True) records the phase the descent actually
    started from, whichever of these it was.

    `params.engine == "rust"` hands the descent to the rust engine after the
    shared setup below (seed, initial draw, kernels, sampling checks). The legacy
    starts run there too (rust has its own twin of the warm start); a given
    `initial_phase` and the backprop start are computed HERE, on python, and rust
    descends from them as handed over (_gs_solve_rust's init_mode). GS history
    exists only on the python engine, so a capture=True solve runs wholly on
    python — including its warm-start stage-1 recursion.
    """
    if capture and params.engine == "rust":
        if log is not None:
            log.info("engine 'rust' does not capture GS history; this solve runs on python")
        params = params.model_copy(update={"engine": "python"})
    size = len(x_axis)
    dx = x_axis[1] - x_axis[0]
    n_freq = len(channels)
    assert n_freq >= 1, "gs_reconstruct needs at least one FreqChannel"
    if ref_freq is None:
        ref_freq = channels[center_freq_index([ch.freq for ch in channels])].freq
    # rho_f = f/f_ref = k_f/k_ref exactly: each channel's phase is rho_f * psi, and
    # the same factor chain-rules into its gradient contribution
    rho = np.array([ch.freq / ref_freq for ch in channels], dtype=np.float64)

    # The geometry is fixed for the whole solve, so there are exactly TWO kernels
    # PER FREQUENCY here -- the forward TX->RX one and its adjoint -- and rs() would
    # rebuild both from scipy.special.hankel1 over the full x axis on all three of
    # its calls per iteration, for up to 10,000 iterations. Building them once is
    # 71% of the numerical core: measured 1.225 s -> 0.351 s (3.49x) on a real
    # candidate, with final_loss, the aperture and the whole 800-point loss curve
    # bit-identical. (One build is ~0.375 ms, so the per-solve cost of F builds is
    # noise; it is the per-iteration rebuild that must never come back.)
    #
    # The adjoint is bit-exactly conj(forward): rs()'s two calls here differ only in
    # the sign of the propagation distance, and kernel_rs_inverse is defined as the
    # conjugate of kernel_rs. Verified for both propagation directions.
    #
    # sampling_quality depends on the propagation distance only through |prop|, which
    # is the same either way, so rs()'s per-call guard collapses to one check per
    # frequency here. The check raises on ANY undersampled channel (the highest
    # frequency binds); solving over a valid subset is the caller's decision, made
    # before building the channel list.
    rx_plane = np.array([rx_z])   # measurement (RX) plane target
    for ch in channels:
        rs.check_sampling(x_axis, rx_plane, ch.wavelength, z_src=tx_z, forward_dir=-1.0)
    h_fwd = [rs.rs_kernel(x_axis, ch.wavelength, -1.0 * (rx_z - tx_z)) for ch in channels]
    h_adj = [np.conjugate(h) for h in h_fwd]

    max_iters = params.max_iters
    cvrg_count = params.convergence_count
    lr0 = params.lr0
    bt_shrink = params.bt_shrink
    bt_tries = params.bt_tries
    conv_threshold = params.convergence_threshold

    # seeded initial phase for reproducibility; draw + record a seed if none given
    seed = params.seed
    if seed is None:
        seed = int(np.random.SeedSequence().entropy % (2**32))
    rng = np.random.default_rng(seed)

    # track aperature amplitude and phase seperatly. curr_aper_phase is the shared
    # unknown psi: the plate's delay profile expressed as phase at ref_freq.
    #
    # The draw is UNCONDITIONAL, made even when a given or back-propagated start
    # replaces it a few lines down. Every legacy start begins from it, on both
    # engines, and those must stay bit-identical; one unused size-N draw is the
    # price of never having a branch between the seed and the draw to get wrong.
    curr_aper_amp = np.abs(orig_aper_amp.copy())
    curr_aper_phase = 2 * np.pi * rng.random(size)

    support = np.abs(orig_aper_amp) > 0  # aperature mask

    # The two starts that are built HERE, for either engine (see the docstring for
    # the order of precedence). Anything else is a legacy start and leaves the draw
    # in place: _warm_start_phase below on python, its twin inside rust.
    legacy_start = initial_phase is None and params.init != "backprop"
    if initial_phase is not None:
        # a copy, so the caller's array is not the one the descent overwrites
        curr_aper_phase = np.array(initial_phase, dtype=np.float64)
        if curr_aper_phase.shape != (size,):
            raise ValueError(f"initial_phase has shape {curr_aper_phase.shape}, "
                             f"expected ({size},): one phase per x_axis sample")
        curr_aper_phase[~support] = 0.0
    elif params.init == "backprop":
        curr_aper_phase = _backprop_phase(
            channels=channels, rho=rho, ref_freq=ref_freq, h_fwd=h_fwd,
            support=support, curr_aper_amp=curr_aper_amp, dx=dx, log=log)

    if params.engine == "rust":
        # a start built above goes over as the phase to descend from ('random':
        # used unchanged); a legacy start goes over as the draw plus its own mode
        return _gs_solve_rust(channels, params, ref_freq, seed, curr_aper_phase,
                              curr_aper_amp, h_fwd, dx, log=log,
                              init_mode=None if legacy_start else "random")

    # Multi-wavelength warm start: replace the random start with a point inside the
    # true basin (reference-frequency solve -> unwrap -> absolute-offset scan). A
    # strict no-op at a single frequency — the random path above runs untouched —
    # which is what preserves the F=1 bit-identity guarantees under any config.
    if legacy_start and params.init == "warm_start" and n_freq > 1:
        curr_aper_phase = _warm_start_phase(
            tx_z=tx_z, orig_aper_amp=orig_aper_amp, x_axis=x_axis, rx_z=rx_z,
            channels=channels, params=params, seed=seed, rho=rho, ref_freq=ref_freq,
            h_fwd=h_fwd, support=support, curr_aper_amp=curr_aper_amp, dx=dx, log=log)
    # the phase the descent starts from, whichever start produced it
    start_phase = curr_aper_phase.copy()

    history = None
    if capture:
        history = GSHistory(
            x_axis=x_axis,
            frequencies=np.array([ch.freq for ch in channels], dtype=np.float64),
            orig_prop_f=np.stack([np.asarray(ch.rx_field) for ch in channels]),
            orig_aper_amp=orig_aper_amp,
            error_weighting=np.stack([np.asarray(ch.error_weighting) for ch in channels]),
            support=support,
            initial_phase=start_phase,
            rx_z=float(rx_z),
            seed=seed,
            history_stride=params.history_stride,
        )

    stop_reason = "max_iters"
    n_iters_run = max_iters
    hist_error = np.zeros(max_iters, np.float32)
    hist_error_pf = np.zeros((max_iters, n_freq), np.float32)
    for iter_idx in range(max_iters):
        # propogate aperature guess to measurement plane, once per frequency. A
        # python loop over channels, NOT a batched 2D fftconvolve: rs() benchmarked
        # the batched form slower at every size tried, and going through the same
        # rs_apply path per frequency keeps the load-bearing complex64 downcast --
        # which is what makes n_freq == 1 bit-identical to the single-frequency
        # solver this generalizes. Each channel builds its own field from the one
        # shared plate profile: phase = rho_f * psi.
        loss_pf = np.empty(n_freq, dtype=np.float64)
        grad_theta = np.zeros(size, dtype=np.float64)
        prop_fields = [] if history is not None else None
        for f, ch in enumerate(channels):
            u0 = curr_aper_amp * np.exp(1j * (rho[f] * curr_aper_phase))
            curr_prop_f = rs.rs_apply(u0, h_fwd[f], dx)
            r_cx = ch.error_weighting * (curr_prop_f - ch.rx_field)
            loss_pf[f] = 0.5 * np.mean(np.abs(r_cx)**2)
            if prop_fields is not None:
                prop_fields.append(curr_prop_f)
            # back propogate the residual from the RX plane to the TX (aperture) plane
            g_u0 = rs.rs_apply(r_cx, h_adj[f], dx)
            # chain rule: d(theta_f)/d(psi) = rho_f
            grad_theta += rho[f] * (2.0 * np.imag(g_u0 * np.conj(u0)))
        # gradient of the MEAN loss: the divisor is n_freq, not 1. Summing instead
        # would silently rescale the effective lr0 by n_freq and change backtracking
        # accept rates, breaking step-size comparability with single-frequency runs.
        grad_theta /= float(n_freq)
        grad_theta[~support] = 0.0

        loss = loss_pf.mean()
        hist_error[iter_idx] = loss
        hist_error_pf[iter_idx] = loss_pf
        if history is not None:
            history.record_iter(iter_idx, loss, loss_pf, curr_aper_phase, prop_fields)

        step = lr0
        for _ in range(bt_tries):
            theta_trial = curr_aper_phase - step * grad_theta
            trial_pf = np.empty(n_freq, dtype=np.float64)
            for f, ch in enumerate(channels):
                u0_trial = curr_aper_amp * np.exp(1j * (rho[f] * theta_trial))
                um_trial = rs.rs_apply(u0_trial, h_fwd[f], dx)
                r_trial = ch.error_weighting * (um_trial - ch.rx_field)
                trial_pf[f] = 0.5 * np.mean(np.abs(r_trial)**2)
            loss_trial = trial_pf.mean()
            if loss_trial < loss:  # sufficient decrease of the JOINT loss
                curr_aper_phase = theta_trial.copy()
                curr_aper_phase[~support] = 0.0
                loss = loss_trial
                loss_pf = trial_pf
                break
            step *= bt_shrink
        curr_aper_f = curr_aper_amp * np.exp(1j * curr_aper_phase)

        if log is not None and iter_idx % 1000 == 0:
            log.info(f"{iter_idx} loss {loss}")

        # if the last x iterations did not improve the error, GS has "converged"
        if iter_idx > cvrg_count:
            recent_err = hist_error[iter_idx-cvrg_count: iter_idx]
            recent_err_flatness = np.mean(np.diff(recent_err))
            if recent_err_flatness > conv_threshold:
                if log is not None:
                    log.info(f"MGS has converged after {iter_idx+1} iterations")
                stop_reason = "converged"
                n_iters_run = iter_idx + 1
                break

    if stop_reason == "max_iters" and log is not None:
        log.error(f"MGS did not converge after {max_iters} iterations")

    if history is not None:
        # the final captured prop fields are the PRE-step fields of the last
        # iteration (as they always were); loss/phase are post-accept.
        history.capture_final(iter_idx, loss, curr_aper_phase, prop_fields)
        history.finalize(stop_reason, n_iters_run, loss, loss_pf)

    loss_full = hist_error[:n_iters_run].copy()
    loss_full_per_freq = hist_error_pf[:n_iters_run].copy()
    return GSResult(curr_aper_f=curr_aper_f, final_loss=float(loss),
                    final_loss_per_freq=np.asarray(loss_pf, dtype=np.float64).copy(),
                    n_iters_run=n_iters_run, stop_reason=stop_reason, seed=seed,
                    loss_full=loss_full, loss_full_per_freq=loss_full_per_freq,
                    frequencies=tuple(float(ch.freq) for ch in channels),
                    ref_freq=float(ref_freq),
                    history=history,
                    psi=curr_aper_phase.copy() if capture else None)


def _multistart_setup(tx_z: float, orig_aper_amp: np.ndarray, x_axis: np.ndarray,
                      rx_z: float, channels: Sequence[FreqChannel], ref_freq: float):
    """What gs_reconstruct_multistart needs to build a multi-tone start itself:
    (dx, rho, h_fwd, curr_aper_amp, support) — gs_reconstruct's own setup,
    written the same way, so an offset is scanned against the fields the descent
    will then compute. Runs gs_reconstruct's sampling guard too, BEFORE any
    solve, so an undersampled channel fails here and not after a wasted
    reference solve."""
    dx = x_axis[1] - x_axis[0]
    rho = np.array([ch.freq / ref_freq for ch in channels], dtype=np.float64)
    rx_plane = np.array([rx_z])
    for ch in channels:
        rs.check_sampling(x_axis, rx_plane, ch.wavelength, z_src=tx_z, forward_dir=-1.0)
    h_fwd = [rs.rs_kernel(x_axis, ch.wavelength, -1.0 * (rx_z - tx_z)) for ch in channels]
    curr_aper_amp = np.abs(orig_aper_amp)
    support = curr_aper_amp > 0
    return dx, rho, h_fwd, curr_aper_amp, support


def _random_route_phase(tx_z: float, orig_aper_amp: np.ndarray, x_axis: np.ndarray,
                        rx_z: float, ch_ref: FreqChannel, rho_ref: float,
                        params: GerchbergSaxtonConfig,
                        support: np.ndarray) -> Tuple[np.ndarray, "GSResult"]:
    """The multi-tone start phase of gs_reconstruct_multistart's "random" start,
    BEFORE its absolute offset: the legacy warm start's first two stages — solve
    the channel nearest the reference alone from the seeded random draw, unwrap
    it over the support. Returns (psi0, the reference solve's result); psi0 is
    zero off the support.

    _warm_start_phase is not touched to get this (it is frozen with the results on
    disk), so those two stages are repeated here, term for term. What follows
    them is the difference: the offset is scanned over the comb's true period
    (comb_period) instead of 2π·f_ref/Δf_min, and it is the difference that
    matters where the legacy range is not a period: at the true TX of the focus
    scenario at ±13 % the legacy start ends at 31.5x the true plate's loss and
    this route at 1.00. It does nothing for the reference solve's other limit
    (not unique where the beam partly misses the window): on the bessel scenario
    at ±20 % the two end at 165x and 164x.
    """
    # 1. reference solve, from the random draw of the pinned seed (init='random':
    # a single channel under 'backprop' would start from the data instead)
    stage1 = gs_reconstruct(tx_z=tx_z, orig_aper_amp=orig_aper_amp, x_axis=x_axis,
                            rx_z=rx_z, channels=[ch_ref],
                            params=params.model_copy(update={"init": "random"}),
                            ref_freq=ch_ref.freq, capture=False, log=None)

    # 2. unwrap over the support and rescale into psi units (theta_c = rho_c * psi)
    psi0 = np.zeros(len(x_axis))
    idx = np.where(support)[0]
    psi0[idx] = np.unwrap(np.angle(stage1.curr_aper_f[idx])) / rho_ref
    return psi0, stage1


# The starts gs_reconstruct_multistart knows, in its default order (a tie goes to
# the earlier one).
MULTISTART_STARTS = ("backprop", "random")


def gs_reconstruct_multistart(tx_z: float, orig_aper_amp: np.ndarray,
                              x_axis: np.ndarray, rx_z: float,
                              channels: Sequence[FreqChannel],
                              params: GerchbergSaxtonConfig,
                              ref_freq: Optional[float] = None,
                              starts: Sequence[str] = MULTISTART_STARTS,
                              log=None) -> Tuple["GSResult", str]:
    """Solve one candidate from each of `starts` — for several tones, from every
    offset a start's scan cannot tell apart — and keep the lowest final loss.
    Returns (that solve's GSResult, the name of its start).

    No single start wins everywhere. On the configs' noise draw the data-driven
    start reaches the true plate's loss at the true TX at every bandwidth, but at
    a WRONG location neither it nor the random one is reliably the lower: over 22
    candidates on and around the TX (30 dB, single tone, ±10 % and ±20 %) the
    data start ended up to 2.8x above the random route on the focus beam, and the
    random route up to 219x above the data start on the bessel beam. A candidate
    ranking is a comparison of losses across locations, so a location solved from
    its worse start is ranked by the start and not by the location. Taking the
    minimum over the starts removes that from the comparison, at the cost of one
    solve per start; it is meant for the few candidates near the sweep's minimum,
    not for the sweep.

    The starts (`params.init` is overridden per start, never read):

    - "backprop": the back-propagated reference tone (_backprop_start).
    - "random": the reference channel solved alone from the seeded random draw
      and unwrapped (_random_route_phase) — the legacy warm start's route, but
      with the offset scanned over the comb's true period. This is NOT
      init='warm_start', whose scan range is kept as it was for the results on
      disk; it is what that start does once the range is right.

    For ONE channel each start is a plain gs_reconstruct with that init. For
    several, each start's phase gets its absolute offset from the scan, and that
    is where one descent per start stops being enough: a comb that nearly
    repeats inside its period leaves two offsets that can score within 1e-4 of
    each other, only one of them in the basin the other tones agree with, and
    which scores higher is decided by the noise draw (at the true TX of the
    caustic scenario at ±13 %, the wrong one in 22 of 48 seeds). So each start
    is descended from EVERY offset _offset_candidates returns — the best, and
    any other within OFFSET_NEAR_TIE_RTOL of it, OFFSET_NEAR_TIE_MAX at most —
    and keeps the lowest final loss; the descent from the first is the very
    solve init='backprop' runs, so a start never does worse here than alone.
    Measured at that true TX under the refinement's settings (cap 6000, stop
    -1e-10), noise seeds 0-8: 0.92-1.13 of the true plate's loss, against
    0.93-2.21 from the single data-driven start. Most starts have ONE such
    offset (see the constants for the counts), so the cost is paid where the
    comb has a close twin: about 1.6 descents per start at ±13 %, 1.2 at ±1 %.

    A tie goes to the earlier name in `starts`, and within a start to the
    higher-scoring offset. The returned name is the START's ("backprop" /
    "random"), whichever of its offsets won. The result's n_iters_run counts the
    winning joint descent alone, as it does for the legacy warm start (the
    "random" route's reference solve and the descents that lost are not in it).
    With a logger, each start logs its scan — the near-tied offsets by value —
    and how many it tried.

    `params.seed` must be an int: the two starts and every candidate they are
    compared across have to share one draw, so the caller pins it (the sweep's
    grid_search.seed) rather than letting each solve draw its own. capture=False
    only — the single-shot paths that capture history (`mgs`, --true-mgs) go
    through gs_reconstruct with the one start the config names. Follows
    `params.engine` (the start phases and the scan are built on python either
    way). `ref_freq` defaults as in gs_reconstruct, and a caller solving on a
    subset of the run's tones passes the run-level reference just the same.
    """
    if params.seed is None:
        raise ValueError("gs_reconstruct_multistart needs gerchberg_saxton.seed pinned "
                         "to an int: its starts must share one random draw")
    if len(starts) == 0:
        raise ValueError("gs_reconstruct_multistart needs at least one start")
    unknown = [name for name in starts if name not in MULTISTART_STARTS]
    if unknown:
        raise ValueError(f"unknown start(s) {unknown}; expected any of "
                         f"{list(MULTISTART_STARTS)}")
    if ref_freq is None:
        # resolved once HERE, so every start solves in the same psi units
        ref_freq = channels[center_freq_index([ch.freq for ch in channels])].freq
    common = dict(tx_z=tx_z, orig_aper_amp=orig_aper_amp, x_axis=x_axis, rx_z=rx_z,
                  channels=channels, ref_freq=ref_freq, capture=False, log=log)
    multi_tone = len(channels) > 1
    if multi_tone:
        dx, rho, h_fwd, curr_aper_amp, support = _multistart_setup(
            tx_z, orig_aper_amp, x_axis, rx_z, channels, ref_freq)
        # the reference channel: nearest rho == 1, as every start picks it
        c_idx = int(np.argmin(np.abs(rho - 1.0)))

    best, best_name = None, None
    for name in starts:
        if not multi_tone:
            # one channel: the plain solve from this start (both names are inits)
            result = gs_reconstruct(params=params.model_copy(update={"init": name}),
                                    **common)
            tried = ""
        else:
            # the start phase WITHOUT its offset, then one descent per offset the
            # scan could not tell apart (one, when nothing comes close to the best)
            if name == "backprop":
                psi0 = _backprop_start(channels[c_idx], h_fwd[c_idx], dx, support,
                                       rho[c_idx])
                how = "back-propagated"
            else:
                psi0, stage1 = _random_route_phase(
                    tx_z, orig_aper_amp, x_axis, rx_z, channels[c_idx], rho[c_idx],
                    params, support)
                how = (f"reference solve loss {stage1.final_loss:.4g} after "
                       f"{stage1.n_iters_run} iterations ({stage1.stop_reason})")
            offsets, scan_note = _scan_comb_offset(
                psi0, channels, rho, ref_freq, h_fwd, curr_aper_amp, dx, log=log,
                near_ties=True)
            if log is not None:
                log.info(f"{name} start: {channels[c_idx].freq / 1e9:g} GHz {how}; "
                         f"{scan_note}")
            result, won = None, 0
            for k, delta in enumerate(offsets):
                psi = psi0.copy()
                psi[support] += delta
                # initial_phase overrides params.init, whatever it says. From the
                # FIRST offset this is the very solve init='backprop' runs (for the
                # "backprop" start): same phase, same descent
                r = gs_reconstruct(params=params, initial_phase=psi, **common)
                # strictly lower replaces: a tie keeps the higher-scoring offset
                if result is None or r.final_loss < result.final_loss:
                    result, won = r, k
            tried = (f" (best of {len(offsets)} near-tied offsets: the one at "
                     f"{offsets[won]:.3f} rad)" if len(offsets) > 1 else "")
        if log is not None:
            log.info(f"multistart: {name} start -> loss {result.final_loss:.6g} after "
                     f"{result.n_iters_run} iterations ({result.stop_reason}){tried}")
        # strictly lower replaces: a tie keeps the earlier start
        if best is None or result.final_loss < best.final_loss:
            best, best_name = result, name
    if log is not None and len(starts) > 1:
        log.info(f"multistart: keeping the {best_name} start (loss {best.final_loss:.6g})")
    return best, best_name


@dataclass
class FreqState:
    """Everything MGS holds per frequency for a joint solve.

    The geometry of the apertures is frequency-independent (their axes come from
    config bounds + dx alone), but the RX element spacing is wavelength/20 when
    the config leaves both dx and grid_stride null, and every beam constructor
    bakes the wavenumber into the emitted phase — so each frequency gets its own
    aperture objects. A fixed dx and an on-grid receiver (grid_stride) are one
    physical array: every frequency's rx_ap then has the same element axis, and
    only the profiles differ.
    """
    freq: float
    wavelength: float
    rx_ap: SimAperature            # per-freq: dx = wavelength/20 when cfg dx and
    #   grid_stride are null. On-grid (rx_ap.grid_idx is not None) the elements are
    #   scene x samples, the same ones for every frequency.
    tx_ap: SimAperature            # per-freq beam profile (every beam's phase ∝ k)
    rx_field: Optional[np.ndarray] = None        # set by measure(), on scene x_axis:
    #   the element samples interpolated onto it, or, on-grid, placed at grid_idx
    #   with zero at every other sample
    error_weighting: Optional[np.ndarray] = None  # set by measure(), on scene x_axis:
    #   zero outside the RX window and, on-grid, at every sample that is not an element
    rx_plane_row: Optional[np.ndarray] = None    # full scene-axis field at the RX
    #   plane, BEFORE the RX elements sample it — by interpolation, or on-grid by
    #   indexing (set by _synthesize_rx; None on the experimental path). analysis.py's
    #   energy metric needs the un-windowed field — rx_field is zero-filled outside
    #   the window.
    rx_aper_profile_clean: Optional[np.ndarray] = None  # the NOISELESS RX element
    #   profile when measure() added noise (rx_ap.aper_profile then holds the noisy
    #   one); None when noiseless or on the experimental path. rx_plane_row stays
    #   noiseless either way — noise is a receiver property, added at the elements.


def _place_on_grid(profile: np.ndarray, grid_idx: np.ndarray, size: int) -> np.ndarray:
    """An on-grid receiver's element `profile` as a field on the scene x-axis: each
    sample at its own grid point (`grid_idx`), zero at every other one of the `size`
    samples. Nothing is interpolated, so the values at `grid_idx` are the element
    samples to the last bit. The zeros between elements are NOT a measurement of
    zero field: measure() gives those samples zero weight, so the solver never
    reads them."""
    out = np.zeros(size, dtype=np.complex128)
    out[grid_idx] = profile
    return out


def _db_or_none(num, den) -> Optional[float]:
    """10*log10(num/den) as a plain float, or None when it is not finite (a
    JSON-safe noise report: json.dump would write a bare NaN/Infinity)."""
    with np.errstate(divide="ignore", invalid="ignore"):
        v = float(10 * np.log10(np.float64(num) / np.float64(den)))
    return v if np.isfinite(v) else None


def _fmt_db(v: Optional[float]) -> str:
    return "n/a" if v is None else f"{v:.1f} dB"


def _fmt_mm(v: Optional[float]) -> str:
    """A position in metres as signed millimetres: +3.13."""
    return "n/a" if v is None else f"{v * 1e3:+.2f}"


def _fmt_num(v: Optional[float], spec: str) -> str:
    return "n/a" if v is None else format(v, spec)


def landing_lines(report: dict) -> List[str]:
    """MGS.landing_report() as the LANDING log block, one string per line.

    The layout is fixed, so the block reads the same in every log that carries it
    (`mgs`, the --true-mgs extras of a grid run): positions in mm with sign and two
    decimals, fractions with four, losses as %.3e, and a `<f> GHz` tag on every
    per-tone line (a true / reconstructed / match triple per tone, in the run's
    frequency order).
    """
    lines = []
    target = report["target"]
    if target is None:
        lines.append("LANDING target: none")
    else:
        lines.append(f"LANDING target: {target['kind']} x={_fmt_mm(target['x_m'])} mm "
                     f"({target['note']})")
    for rec in report["per_freq"]:
        ghz = f"{rec['freq_hz'] / 1e9:g} GHz"
        for label in ("true", "reconstructed"):
            m = rec[label]
            if m is None:
                lines.append(f"LANDING {label:<13} {ghz}: no power at the RX plane")
                continue
            lines.append(
                f"LANDING {label:<13} {ghz}: peak {_fmt_mm(m['peak_x_m'])} mm, "
                f"centroid {_fmt_mm(m['centroid_x_m'])} mm, "
                f"5-95% [{_fmt_mm(m['e05_x_m'])}, {_fmt_mm(m['e95_x_m'])}] mm, "
                f"in-window {m['frac_in_window']:.4f} | "
                f"in-window peak {_fmt_mm(m['peak_x_in_window_m'])} mm, "
                f"centroid {_fmt_mm(m['centroid_x_in_window_m'])} mm")
        vs_true, vs_target = rec["match_true"], rec["match_target"]
        lines.append(
            f"LANDING {'match':<13} {ghz}: corr {_fmt_num(vs_true['corr'], '.4f')}, "
            f"rel-L2 vs true {_fmt_num(vs_true['rel_l2'], '.3f')}, "
            f"power ratio {_fmt_num(vs_true['power_ratio'], '.3f')}, "
            f"rel-L2 vs target {_fmt_num(vs_target['rel_l2'], '.3f')}")
    loss = report["loss"]
    lines.append(
        f"LANDING loss solver={loss['solver']:.3e} true_plate={loss['true_plate']:.3e} "
        f"zero_model={loss['zero_model']:.3e} ratio={_fmt_num(loss['ratio'], '.3f')} "
        f"iters={loss['n_iters']} stop={loss['stop_reason']}")
    bias, dx = report["dx_bias_m"], report["scene_dx_m"]
    if abs(bias - 0.5 * dx) <= 1e-6 * dx:
        offset = f"+dx/2 lateral offset ({bias * 1e3:+.3f} mm)"
    else:
        # not one of the shipped scene axes: see landing.propagator_offset
        offset = (f"lateral offset ({bias * 1e3:+.3f} mm; not +dx/2: the scene x-axis "
                  "is not an even-length axis centred on x = 0)")
    lines.append(f"LANDING note: positions carry the propagator's {offset}; outside "
                 "the RX window the reconstruction is unconstrained")
    return lines


class MGS():
    def __init__(self, freqs, config: SimConfig):
        self._init_common(freqs, config)

        # Convention: the RX aperture sits at the origin (z=0), the bottom of the
        # image. The TX aperture sits above it and projects toward -Z, so its beam
        # travels *down* to the RX. Move the TX by changing tx.z (height) and its
        # x_min/x_max (lateral).
        scene_cfg = config.sim_scene
        self.tx_cfg = config.tx_aperture
        # The apertures are built before the SimScene, which takes them as arguments,
        # but an on-grid receiver (rx_aperture.grid_stride) takes its elements FROM
        # the scene x-axis. So the axis is laid out here first, with the exact call
        # SimScene makes; the assert below holds the two to the same floats.
        scene_x_axis = sampled_axis(scene_cfg.x_min, scene_cfg.x_max, scene_cfg.spacing)
        for freq in self.freqs:
            wl = rs.wavelength(freq)
            self.freq_states.append(FreqState(
                freq=freq, wavelength=wl,
                rx_ap=self._build_rx(config.rx_aperture, wl, scene_x_axis),
                tx_ap=self._build_tx(config.tx_aperture, scene_cfg.z_min, freq),
            ))
        # the scene holds the PRIMARY frequency's apertures as its views; geometry
        # is identical across frequencies, only profiles/dx differ
        primary = self.freq_states[0]
        tx = primary.tx_ap
        self.gs_tx = SimAperature(x_min=tx.x_min, x_max=tx.x_max, z=tx.z, dx=tx.dx)
        self.scene = SimScene(
            x_min=scene_cfg.x_min, x_max=scene_cfg.x_max,
            z_min=scene_cfg.z_min, z_max=scene_cfg.z_max,
            spacing=scene_cfg.spacing, rx_ap=primary.rx_ap, tx_ap=tx,
        )
        # an on-grid receiver's elements must BE samples of the axis the solver
        # runs on, to the last bit: measure() reads and scores the field by index,
        # and an index into any other axis would be a silently shifted array
        for fs in self.freq_states:
            rx = fs.rx_ap
            if rx.grid_idx is not None:
                assert np.array_equal(self.scene.x_axis[rx.grid_idx], rx.aper_axis), \
                    "on-grid RX elements are not samples of the scene x-axis"
        self._log_geometry()

    def _init_common(self, freqs, config: SimConfig) -> None:
        """State every MGS has, whatever built its scene.

        `freqs` is a float or a sequence of floats (a joint solve holds one
        FreqState per frequency). `self.freq`/`self.wavelength` alias the primary
        (first) frequency for the inherently single-frequency consumers: geometry
        logging, scene (re-)illumination, plot_scene.
        """
        self.log = logging.getLogger()
        if np.isscalar(freqs):
            freqs = [freqs]
        self.freqs = [float(f) for f in freqs]
        assert len(self.freqs) >= 1, "MGS needs at least one frequency"
        self.freq = self.freqs[0]
        self.wavelength = rs.wavelength(self.freq)
        self.plot_path = config.plot_path
        self.gs_cfg = config.gerchberg_saxton
        # an ALIAS, like gs_cfg: a drawn noise seed written back here lands in the
        # config that save_run later snapshots
        self.noise_cfg = config.noise
        self.noise_report = None       # set by measure() when noise is on
        self.output_cfg = config.output
        self.tx_cfg = None             # the simulated constructor sets it: the beam's
        #   geometric landing target (landing_report) is read from the TX config, and
        #   an experimental capture has no configured beam to aim
        self.gs_history = None
        self.gs_result = None
        self.freq_states: List[FreqState] = []   # filled by the constructor path
        self.gs_rec_data = None        # scene re-illuminated by the reconstruction

    @classmethod
    def from_experiment(cls, rx_path: Path, heatmap_path: Path, freq: float,
                        config: SimConfig) -> "MGS":
        """Build an MGS from experimental .mat captures instead of a simulated scene.

        Geometry comes from the capture and the bench rather than config.sim_scene:
        the RX aperture IS the measurement, and the TX aperture is what the
        experiment is trying to recover (has_real_aper is False, so no "Real TX"
        overlay is drawn and measure() does not synthesize an RX field).

        Everything the rig contributes -- the down-conversion chain, the aperture
        edges, the coordinate origins, the amplitude normalisation, the scene
        margins -- lives in config.experimental rather than as thirteen magic
        numbers spread across two modules.
        """
        exp = config.experimental
        # alternate constructor: __init__ builds a scene from config, and here the
        # scene comes from data, so there is nothing for it to do
        self = cls.__new__(cls)
        self._init_common(freq, config)
        self.log.info(f"Carrier Frequency: {freq*1e-9: 0.2f} GHz")

        self.log.info(f"Reading file as RX data: {rx_path}")
        rx = parse_oscope_rx_data(rx_path, freq, exp)
        rx.z = exp.rx_z_origin - rx.z
        dx = rx.dx

        # TX aperture edges in rig coordinates, mirrored into scene coordinates
        tx_x_min = exp.rig_x_origin - exp.tx_left_edge
        tx_x_max = exp.rig_x_origin - exp.tx_right_edge
        # recentre everything so the TX aperture straddles x = 0
        x_offset = -((tx_x_max - tx_x_min) / 2.0 + tx_x_min)

        rx_centred = SimAperature(x_min=rx.x_min + x_offset, x_max=rx.x_max + x_offset,
                                  z=rx.z, dx=dx)
        # the capture can be one sample longer than the recentred aperture's own
        # sampling, so it is truncated rather than resampled
        rx_centred.aper_profile = rx.aper_profile[0:len(rx_centred.aper_profile)]
        rx_centred.aper_profile = exp.rx_amplitude_scale * (
            rx_centred.aper_profile / np.max(rx_centred.aper_profile))
        rx = rx_centred

        tx = SimAperature(x_min=tx_x_min + x_offset, x_max=tx_x_max + x_offset,
                          z=0, dx=dx)
        # broadside plane wave: the solver's seed
        tx.aper_profile = beam_aperatures.steered_profile(freq, tx.aper_axis, theta_deg=0)
        self.gs_tx = SimAperature(x_min=tx.x_min, x_max=tx.x_max, z=tx.z, dx=dx)
        # a single capture is inherently one frequency: the length-1 joint path
        self.freq_states = [FreqState(freq=self.freq, wavelength=self.wavelength,
                                      rx_ap=rx, tx_ap=tx)]

        spacing_ratio = round(dx / self.wavelength, 3)
        self.log.info(f"RX measurements are spread {spacing_ratio:0.3f} wavelengths apart")

        self.log.info(f"Reading file with heatmap data: {heatmap_path}")
        base_scene = SimScene(
            x_min=tx.x_min - exp.scene_x_margin, x_max=tx.x_max + exp.scene_x_margin,
            z_min=exp.scene_z_min, z_max=exp.scene_z_max,
            spacing=dx, rx_ap=rx, tx_ap=tx,
        )
        self.scene = parse_oscope_heatmap_data(heatmap_path, base_scene, x_offset,
                                               freq, exp)

        self.has_real_aper = False   # the experimental TX aperture is unknown
        self.beam_type = "directional"
        self._log_geometry()
        return self

    def _build_rx(self, rx_cfg, wavelength: float,
                  scene_x_axis: np.ndarray) -> SimAperature:
        """The receive aperture at `wavelength`: a window of `width` centred at
        `x_center`, at the scene origin (z=0).

        With `grid_stride` set the elements sit ON the solver grid: every
        grid_stride-th sample of `scene_x_axis` inside the window
        (SimAperature.on_grid). The wavelength plays no part, so every frequency
        gets the same element set -- one physical array -- and measure() reads and
        scores the field at those samples without interpolating.

        Otherwise the elements sit on an axis of their own at pitch `dx`, which
        defaults to wavelength/20 when left null -- so the element count is
        per-frequency -- and is what sets the receiver element count, the
        independent variable of the archived pitch experiments
        (configs/archive/scenario_caustic_hit_lambda2.yml and
        configs/archive/scenario_caustic_hit_lambda4.yml). `scene_x_axis` is not
        used on this path."""
        if rx_cfg.grid_stride is not None:
            return SimAperature.on_grid(
                x_min=rx_cfg.x_center - rx_cfg.width / 2.0,
                x_max=rx_cfg.x_center + rx_cfg.width / 2.0,
                z=0,
                scene_x_axis=scene_x_axis,
                stride=rx_cfg.grid_stride,
            )
        # written as a multiply by 1/20, not a divide by 20: 0.05 is not exactly 1/20
        # in binary, so the two disagree by an ulp (measured on 28 config x frequency
        # combinations, though none of them moved num_points)
        rx_dx = rx_cfg.dx if rx_cfg.dx is not None else wavelength * (1 / 20)
        return SimAperature(
            x_min=rx_cfg.x_center - rx_cfg.width / 2.0,
            x_max=rx_cfg.x_center + rx_cfg.width / 2.0,
            z=0,
            dx=rx_dx,
        )

    def _build_tx(self, tx_cfg, z_min: float, freq: float) -> SimAperature:
        """The transmit aperture and the beam it emits at `freq`, defined
        independently of the scene grid. Sets self.beam_type and self.has_real_aper.

        The profile comes from beam_aperatures.py, whose generators all bake the
        wavenumber into the PHASE only, so the aperture amplitude (the solver's
        fixed constraint) is frequency-independent."""
        tx = SimAperature(x_min=tx_cfg.x_min, x_max=tx_cfg.x_max, z=tx_cfg.z, dx=tx_cfg.dx)
        assert tx.z > z_min, f"tx_aperture.z ({tx.z}) must be above the scene floor z_min ({z_min})"

        beam = tx_cfg.beam
        self.beam_type = beam.type
        # a known real aperture exists (every simulated beam); false for the
        # experimental path, where the TX aperture is what we are trying to recover
        self.has_real_aper = True
        tx.aper_profile = beam_aperatures.beam_profile(beam, freq, tx.aper_axis, tx.z - z_min)
        return tx

    def _log_geometry(self) -> None:
        scene = self.scene
        self.log.info(f"Simulation scene:")
        self.log.info(f" - X axis: {scene.x_min:0.3f} - {scene.x_max:0.3f}")
        self.log.info(f" - Z axis: {scene.z_min:0.3f} - {scene.z_max:0.3f}")
        self.log.info(f"RX aperature:")
        rx_wl_ratio = scene.rx_ap.dx / self.wavelength
        rx_width = scene.rx_ap.x_max - scene.rx_ap.x_min
        self.log.info(f" - X axis: {scene.rx_ap.x_min:0.3f} {scene.rx_ap.x_max:0.3f} (width {rx_width:0.3f} m, dx {rx_wl_ratio:0.3f} wavelength)")
        if scene.rx_ap.grid_idx is not None:
            self.log.info(f" - Elements: {scene.rx_ap.num_points} on the solver grid "
                          f"(grid_stride {scene.rx_ap.grid_stride}, pitch "
                          f"{scene.rx_ap.dx * 1e3:0.4f} mm), read and scored without "
                          "interpolation")
        self.log.info(f" - Z: {scene.rx_ap.z:0.3f}")
        self.log.info(f"TX aperature ({self.beam_type} beam)")
        tx_wl_ratio = scene.tx_ap.dx / self.wavelength
        self.log.info(f" - X axis: {scene.tx_ap.x_min:0.3f} {scene.tx_ap.x_max:0.3f} (dx {tx_wl_ratio:0.3f} wavelength)")
        self.log.info(f" - Z: {scene.tx_ap.z:0.3f}")

    # ----------------------------------------------------------------- scene ---
    def _propagate(self, tx_ap: SimAperature) -> np.ndarray:
        """Illuminate the whole scene from `tx_ap`: resample its profile onto the
        scene x-axis, then RS-propagate down through every z plane."""
        u0 = interp_real_imag(tx_ap.aper_axis, tx_ap.aper_profile, self.scene.x_axis)
        self.log.info("Computing wave propogation across scene")
        return rs.illuminate(self.scene.x_axis, self.scene.z_axis, u0,
                             self.wavelength, tx_ap.z)

    def illuminate_real(self) -> None:
        """Fill scene.data with the field radiated by the real TX aperture.

        Only the plots need this. The RX measurement does NOT come from here -- see
        _synthesize_rx, which propagates to one plane instead of all 3400 of them.
        """
        self.log.info("Running scene simulation with real aperature")
        self.scene.data = self._propagate(self.scene.tx_ap)

    def illuminate_reconstructed(self) -> None:
        """Fill gs_rec_data with the field radiated by the MGS-reconstructed aperture.

        A re-illumination is inherently monochromatic; this (like _propagate and
        plot_scene) runs at the PRIMARY frequency. The reconstruction itself is one
        delay plate fitted jointly across all configured frequencies, stored as its
        reference-frequency field."""
        self.log.info("Running scene simulation with MGS reconstructed aperature")
        self.gs_rec_data = self._propagate(self.gs_tx)

    # ----------------------------------------------------------- measurement ---
    def _synthesize_rx(self, fs: FreqState) -> np.ndarray:
        """Propagate the real TX aperture to the RX measurement plane at this
        frequency and sample it at the RX elements. Returns the complex RX
        aperture profile.

        Propagates directly to rx_ap.z -- the plane gs_reconstruct actually models.
        It used to snap to the nearest scene z-plane, with the comparison inverted so
        it landed on the FARTHER one (index 1, never 0, for all 12 shipped configs):
        the measurement was synthesized 0.25 mm, lambda/8 at 150 GHz, away from where
        it was then fitted. Snapping is not merely inverted but wrong in principle --
        config.py permits z_min > 0, where rx.z sits below z_axis[0] and no scene
        plane is the right answer.

        Propagates to ONE plane, not all of them. run_grid_search used to call
        run_sim purely to get this, paying a full 3400 x 2400 scene (1.71 s and
        +190 MB RSS per frequency, held in the parent for the whole sweep) to keep a
        single row. Verified bit-identical: rs.rs(x, [z_k], ...)[0] equals
        rs.rs(x, z_axis, ...)[k], since each row is computed independently from u0.

        The round trip through rx_ap.aper_axis is kept deliberately -- it models the
        receiver element spacing (lambda/2 vs lambda/20), which was the whole point of
        configs/archive/scenario_caustic_hit_lambda2.yml. It is also a LINEAR
        interpolation of a field whose phase turns between scene samples: the
        elements of an array on its own axis fall between grid points, so what they
        are handed is the chord, not the field (about -31 dB off it for the
        80-element lambda/2 array at 300 GHz).

        An on-grid receiver (rx_ap.grid_idx, from rx_aperture.grid_stride) has no
        round trip to make: its elements are scene x samples, so the row is read at
        them by index and the profile is the propagated field itself, bit for bit
        (upcast from the propagator's complex64 to the complex128 every element
        profile is held in).
        """
        scene = self.scene
        tx_ap = fs.tx_ap
        u0 = interp_real_imag(tx_ap.aper_axis, tx_ap.aper_profile, scene.x_axis)
        row = rs.rs(scene.x_axis, np.array([fs.rx_ap.z]), u0, fs.wavelength,
                    z_src=tx_ap.z, forward_dir=-1.0)[0]
        fs.rx_plane_row = row   # kept for the energy metric (analysis.py)
        if fs.rx_ap.grid_idx is not None:
            return np.asarray(row[fs.rx_ap.grid_idx], dtype=np.complex128)
        return interp_real_imag(scene.x_axis, row, fs.rx_ap.aper_axis)

    def measure(self) -> None:
        """Put the measured RX field on the scene grid and build the error
        weighting used by phase retrieval, once per frequency. Idempotent.

        On the simulated path each frequency's measurement is synthesized here, by
        propagating that frequency's known TX aperture to the RX plane. On the
        experimental path there is no known TX aperture (has_real_aper is False)
        and rx_ap.aper_profile already holds the measured data. The results
        (FreqState.rx_field / .error_weighting) are the single measurement set
        shared across every hypothesized TX location.

        Each frequency's weighting is normalized to its OWN measurement's peak, so
        a joint solve's mean loss weights frequencies equally regardless of their
        absolute RX power.

        How the element samples reach the scene grid depends on the receiver:

        - Interpolated (rx_aperture.dx, and the experimental path): the elements
          sit on an axis of their own, so their samples are linearly interpolated
          onto the scene grid and the loss is scored at EVERY grid point inside the
          RX window. Harmless when the elements are denser than the grid
          (wavelength/20). With a sparser array the fill between elements is a
          straight line through a field whose phase keeps turning — about 60
          degrees per element at lambda/2 — so it dips: the target the solver is
          handed sits about -18 dB off the true field at 300 GHz and carries about
          20 % less power, and most of the points it is scored at are that fill,
          not measurements.
        - On-grid (rx_aperture.grid_stride): the elements ARE scene samples. Their
          profile is placed at rx_ap.grid_idx and the weighting is non-zero at
          those samples only, so the solver is held to what was measured and to
          nothing else. The weighting carries a sqrt(grid_stride) factor that keeps
          the loss on the dense receiver's scale (see the comment at the branch).

        With noise.snr_db set, the simulated path adds complex AWGN to every tone's
        RX ELEMENT profile before it is put on the scene grid — where a real
        receiver adds it (rice_bend/noise.py has the model). The floor is
        referenced to the peak over every tone, so all clean profiles are
        synthesized first, then noised, then interpolated or placed. The weighting
        is built from the NOISY measurement, as it would be from a capture. The
        clean profile is kept on FreqState.rx_aper_profile_clean and the realized
        SNRs in self.noise_report. Idempotency holds with noise too: the clean
        field is re-synthesized from the TX and each tone's stream rebuilt from
        (seed, frequency), and a null seed is drawn once and written back to the
        config. On the experimental path noise.snr_db is ignored (with a warning):
        the capture carries its own receiver noise.
        """
        # Fail fast, naming EVERY undersampled frequency at once — rs() would raise
        # on only the first it meets, and the highest frequency binds, so "which
        # frequencies are the problem" is exactly the question the error must answer.
        if self.has_real_aper:
            rx_plane = np.array([self.scene.rx_ap.z])
            bad = []
            for fs in self.freq_states:
                q = rs.sampling_quality(self.scene.x_axis, rx_plane, fs.wavelength,
                                        z_src=self.scene.tx_ap.z, forward_dir=-1.0)
                if q < 1:
                    bad.append(f"{fs.freq / 1e9:g} GHz (quality {q:.3f})")
            if bad:
                raise RuntimeError(
                    f"RS undersampled for {len(bad)} of {len(self.freq_states)} "
                    f"frequencies at the TX plane: {', '.join(bad)}. Use a finer "
                    f"sim_scene.spacing or drop the highest frequencies.")

        x_axis = self.scene.x_axis
        snr_db = self.noise_cfg.snr_db
        noise_on = self.has_real_aper and snr_db is not None
        if self.has_real_aper:
            # every clean profile first: the noise floor is referenced to the peak
            # over ALL tones, so no tone can be noised before the last is synthesized
            clean = [self._synthesize_rx(fs) for fs in self.freq_states]
            if noise_on:
                if self.noise_cfg.seed is None:
                    # same idiom as the solver's seed; written back so the run is
                    # replayable from its config snapshot, and so a second measure()
                    # reuses it (idempotency)
                    self.noise_cfg.seed = int(np.random.SeedSequence().entropy % (2**32))
                seed = self.noise_cfg.seed
                noisy, peak_power, sigma2 = add_awgn(
                    clean, [fs.freq for fs in self.freq_states], snr_db, seed)
                for fs, c, y in zip(self.freq_states, clean, noisy):
                    fs.rx_ap.aper_profile = y
                    fs.rx_aper_profile_clean = c
            else:
                for fs, c in zip(self.freq_states, clean):
                    fs.rx_ap.aper_profile = c
                    fs.rx_aper_profile_clean = None
        elif snr_db is not None:
            self.log.warning(f"noise.snr_db ({snr_db:g} dB) is ignored on the experimental "
                             "path: the capture carries its own receiver noise")

        per_freq = []
        for fs in self.freq_states:
            rx_ap = fs.rx_ap
            on_grid = rx_ap.grid_idx is not None
            if on_grid:
                # The elements ARE scene samples: place the (possibly noisy) profile
                # at them and score the solver there and nowhere else. Every other
                # sample gets zero weight — outside the window as before, and also
                # INSIDE it wherever no element sits, where the receiver measured
                # nothing and the interpolated path would have scored its own fill.
                orig_prop_f = _place_on_grid(rx_ap.aper_profile, rx_ap.grid_idx,
                                             len(x_axis))
                # Same amplitude weighting as below, normalized to the peak ELEMENT,
                # times sqrt(grid_stride). The factor is deliberate. The loss is
                # 0.5 * mean over the WHOLE scene axis of |w * r|^2, and each
                # element stands for grid_stride grid samples, so squaring the
                # weight puts grid_stride back: the sum over the elements stays the
                # Riemann sum over the window that a dense receiver's every-sample
                # sum is. Without it the same fit would score 1/grid_stride of the
                # loss, residuals would not compare across receivers, and the
                # absolute convergence_threshold would mean a different thing for
                # every stride.
                elem_amp = np.abs(rx_ap.aper_profile)
                error_weighting = np.zeros(len(x_axis), dtype=np.float64)
                error_weighting[rx_ap.grid_idx] = np.sqrt(rx_ap.grid_stride) * (
                    (elem_amp / elem_amp.max()) + 0.25)
            else:
                # A FIELD, so cartesian: interpolating its phase through the ±π branch cut
                # corrupted the measurement by 20.8% rel L2 on scenario_caustic_hit and 55%
                # on the lambda/2-spaced variant, and the solver then fitted the corruption
                # at full weight.
                orig_prop_f = interp_real_imag(rx_ap.aper_axis, rx_ap.aper_profile, x_axis)
                # error computations are weighted to favor higher amplitude data, and ignore things outside the recieve aperature
                error_weighting = (np.abs(orig_prop_f) / np.abs(orig_prop_f).max()) + 0.25
                error_weighting[(x_axis < rx_ap.x_min) | (x_axis > rx_ap.x_max)] = 0.0
            fs.rx_field = orig_prop_f
            fs.error_weighting = error_weighting
            if noise_on:
                # what the solver actually sees: the interpolation onto the scene
                # grid averages the element noise, so this sits above the per-antenna
                # figure. Measured inside the RX window only (the weighting zeroes
                # the rest). On-grid nothing is interpolated, so nothing is averaged:
                # the clean profile is placed the way the noisy one was, the two
                # differ at the elements alone, and the ratio is the REALIZED
                # per-element SNR of this draw (mean_snr_db is the same quantity
                # against the nominal sigma^2).
                if on_grid:
                    clean_on_scene = _place_on_grid(fs.rx_aper_profile_clean,
                                                    rx_ap.grid_idx, len(x_axis))
                else:
                    clean_on_scene = interp_real_imag(rx_ap.aper_axis,
                                                      fs.rx_aper_profile_clean, x_axis)
                in_win = (x_axis >= rx_ap.x_min) & (x_axis <= rx_ap.x_max)
                per_freq.append({
                    "freq_hz": float(fs.freq),
                    "n_elements": int(len(fs.rx_aper_profile_clean)),
                    "mean_snr_db": _db_or_none(
                        np.mean(np.abs(fs.rx_aper_profile_clean) ** 2), sigma2),
                    "effective_snr_db": _db_or_none(
                        np.sum(np.abs(clean_on_scene[in_win]) ** 2),
                        np.sum(np.abs(orig_prop_f[in_win] - clean_on_scene[in_win]) ** 2)),
                })

        if not noise_on:
            self.noise_report = None
            return
        self.noise_report = {
            "snr_db": float(snr_db), "seed": int(seed),
            "reference": "peak_rx_elements",
            "peak_power": float(peak_power), "sigma2": float(sigma2),
            "numpy_version": np.__version__,
            "per_freq": per_freq,
        }
        self.log.info(f"RX noise: SNR {snr_db:g} dB re peak RX element, seed {seed}, "
                      f"peak power {peak_power:.4g}, sigma^2 {sigma2:.4g}")
        for rec in per_freq:
            self.log.info(f" - {rec['freq_hz'] / 1e9:g} GHz: {rec['n_elements']} elements, "
                          f"mean per-antenna SNR {_fmt_db(rec['mean_snr_db'])}, "
                          f"effective solver-grid SNR {_fmt_db(rec['effective_snr_db'])}")

    def measurement_channels(self) -> List[FreqChannel]:
        """FreqChannel payloads for gs_reconstruct, one per frequency in self.freqs
        order. measure() must have been called."""
        assert all(fs.rx_field is not None for fs in self.freq_states), \
            "measure() must be called before measurement_channels()"
        return [FreqChannel(freq=fs.freq, wavelength=fs.wavelength,
                            rx_field=fs.rx_field, error_weighting=fs.error_weighting)
                for fs in self.freq_states]

    def run_gerch_sax(self):
        """Reconstruct the aperture phase at the *real* TX plane (single-shot path).

        Thin wrapper over measure() + reconstruct_at() that preserves the original
        behaviour: solve at scene.tx_ap.z using the real aperture's amplitude as the
        fixed support, writing the result into self.gs_tx. One solve, jointly over
        every configured frequency.
        """
        self.log.info(f"Running modified Gerchberg-saxton algorithm")
        self.measure()
        # amplitude-only ON PURPOSE: this is the solver's fixed amplitude constraint
        # and its support mask, not a field. Forcing it cartesian (|interp_real_imag|)
        # is a different array: the straight line between two samples of different
        # phase cuts inside the arc, so it dips wherever the plate's phase turns
        # between aperture samples -- by an amount that depends on the beam and on
        # tx_aperture.dx, with no error raised and no visibly broken plot.
        #
        # The PRIMARY frequency's profile serves as THE constraint for a joint
        # solve: every beam constructor emits unit amplitude (only the phase depends
        # on the wavenumber), so the amplitude is frequency-independent by
        # construction and any freq_state's profile gives the same array.
        tx_ap = self.scene.tx_ap
        orig_aper_amp = interp_amplitude(tx_ap.aper_axis, tx_ap.aper_profile,
                                         self.scene.x_axis)
        result = self.reconstruct_at(
            tx_z=self.scene.tx_ap.z,
            orig_aper_amp=orig_aper_amp,
            out_aper=self.gs_tx,
        )
        self.gs_result = result
        self.gs_history = result.history

    def reconstruct_at(self, tx_z: float, orig_aper_amp: np.ndarray,
                       out_aper: SimAperature) -> GSResult:
        """Run modified Gerchberg-Saxton for a hypothesized TX plane.

        Solves for the one aperture phase at plane `tx_z` that best reproduces
        every frequency's measured RX field (FreqState.rx_field), holding the
        amplitude fixed at `orig_aper_amp` (defined on scene.x_axis; its nonzero
        region is the support). Writes the reconstructed complex aperture into
        out_aper.aper_profile and returns the run's full GSResult. measure() must
        have been called first.
        """
        # Run the solver core (capture=True so the single-shot path keeps its full
        # GSHistory for animation/persistence), then interp the result onto out_aper.
        result = gs_reconstruct(
            tx_z=tx_z,
            orig_aper_amp=orig_aper_amp,
            x_axis=self.scene.x_axis.copy(),
            rx_z=self.scene.rx_ap.z,
            channels=self.measurement_channels(),
            params=self.gs_cfg,
            ref_freq=self.freqs[center_freq_index(self.freqs)],
            capture=True,
            log=self.log,
        )
        # the converged aperture is a FIELD, and it is heavily wrapped -- cartesian
        out_aper.aper_profile = interp_real_imag(
            self.scene.x_axis, result.curr_aper_f, out_aper.aper_axis)
        return result

    # --------------------------------------------------------------- landing ---
    def _joint_loss(self, rx_rows: Sequence[np.ndarray]) -> float:
        """The solver's joint loss for one RX-plane field per tone (self.freqs
        order): the mean over tones of 0.5*mean(|w*(row - rx_field)|^2), the
        expression gs_reconstruct descends, so the result compares directly with
        gs_result.final_loss."""
        return float(np.mean([
            0.5 * np.mean(np.abs(fs.error_weighting * (row - fs.rx_field)) ** 2)
            for fs, row in zip(self.freq_states, rx_rows)]))

    def landing_report(self) -> dict:
        """Where the energy lands on the RX plane — the real beam against the
        solver's reconstruction, per tone — and the final loss in context.
        Simulated path only; run_gerch_sax() must have run.

            {"target":    landing.beam_target(...): the beam's geometric aim, or None
             "rx_window": {"x_min", "x_max"}
             "dx_bias_m": the propagator's lateral offset; half the scene x step
                          on every shipped scene (see the last paragraph)
             "scene_dx_m": the scene x step
             "per_freq":  [{"freq_hz",
                            "true":          landing_metrics(true row)
                            "reconstructed": landing_metrics(reconstructed row)
                            "match_true":    field_match(true row, reconstructed row)
                            "match_target":  field_match(rx_field, reconstructed row),
                                             at the element samples alone when the
                                             receiver is on-grid},
                           ...]                                  self.freqs order
             "loss": {"solver", "true_plate", "zero_model", "ratio", "n_iters",
                      "stop_reason"}}

        The true row is FreqState.rx_plane_row: the clean field of the real plate at
        the RX plane, before the RX elements sample it and before any noise. The
        reconstructed row is the solver's OWN model propagated the solver's own way
        — amp * exp(1j * (f/ref_freq) * psi) on the scene axis, through the same
        kernel — so the joint loss of these rows IS gs_result.final_loss. It is
        deliberately not gs_rec_data[0]: that is gs_tx re-illuminated, which has
        been resampled scene -> aperture -> scene on the way, and on a multi-tone
        run it is the reference-frequency field propagated at the first-listed
        frequency.

        match_true answers the question being asked (does the energy land where
        the real beam's does); match_target is what the solver was actually given
        to fit: the measurement as the RX elements sampled it, noise included. The
        two differ by the receiver, not by the solver — a sparse array's
        interpolated target carries less power than the field it samples. With an
        on-grid receiver (rx_aperture.grid_stride) match_target is taken at the
        ELEMENT samples only: rx_field is zero at the in-window samples that carry
        no element, which is the absence of a measurement and not a target, and
        comparing there would charge the reconstruction for the field it rightly
        puts between elements. match_true stays over every window sample either
        way, so it also says how well the fit holds BETWEEN the elements, where
        nothing constrained it.

        `loss` sets the solver's final loss against two references evaluated with
        the same expression: `true_plate`, the loss of the REAL plate's phase under
        the solver's model (the solver's amplitude constraint times
        exp(1j*angle(true profile on the scene axis)), per tone), and `zero_model`,
        the loss of radiating nothing. `ratio` = solver / true_plate. The true plate
        does not score zero: the target is the field as the RX elements sampled it
        (and noised it), which the true plate does not reproduce — so with a sparse
        or noisy receiver the solver gets BELOW the true plate, and a ratio under 1
        is the ordinary case, not a suspicious one. A ratio well above 1 means the
        solve stopped short of the true plate: cut off by max_iters (stop_reason
        says so), or converged in another basin, which a multi-tone solve from the
        legacy warm start does at some bandwidths (its offset scan does not cover
        the comb's true period, and its reference solve is not unique where part
        of the beam misses the RX window; see the notes above comb_period).

        An on-grid receiver changes what the ratio is read against. Its target IS
        the true field at the elements, so the sampling term is gone and, noiseless,
        the true plate scores almost nothing: all that is left is the gap between
        the solver's amplitude constraint (interp_amplitude) and the magnitude of
        the plate's cartesian resampling, 4.6e-8 to 6.6e-8 on the three
        scenario_*_hit_1f_grid2l configs (and exactly 0 with that magnitude as the
        amplitude). The flatness test stops the descent well above that, at 4e-6 to
        1e-5 there, so the ratio comes out at 90 to 170 on solves that match the
        true field to 1-2 % rel-L2: stopped short, but in the right basin. With
        receiver noise the true plate's loss is the noise it cannot fit. A SINGLE
        tone reaches that floor and the ratio is back at about 1 (0.99 to 1.04 at
        30 dB on those configs, which solve one tone). Several tones get there or
        not depending on the start. Over the frequency study's three-tone combs
        (±1 % to ±20 %) at 30 dB, `init: warm_start` comes out at 1.09 to 1.66 at
        the true TX of the caustic config, at 1.01 to 31.5 on the focus one and at
        1.60 to 165 on the bessel one; `init: backprop` at 0.96 to 1.11 on all
        three. Those are the configs' noise draw (seed 0). On other draws the
        single data-driven start of this solve can still land about 2.1-2.3x up
        on the caustic at ±13 %: that comb nearly repeats after 77 cycles, and
        the offset scan's two best are a near-tie the noise decides (see the
        notes above comb_period; the sweep's refinement descends from both).

        Positions are read off the scene grid and carry the propagator's lateral
        offset, reported as `dx_bias_m`: landing.propagator_offset, which is +dx/2
        on the even-length, origin-centred x-axis of every shipped scene (and is
        reported as whatever it is on any other). Outside the RX window the
        weighting is zero and the reconstruction is unconstrained, so its full-row
        numbers (peak, centroid, 5-95 % span) say where the fitted plate happens to
        throw its energy; only the in-window ones are held to the measurement.
        """
        result, hist = self.gs_result, self.gs_history
        if not self.has_real_aper:
            raise RuntimeError("landing_report needs the real TX aperture; the "
                               "experimental path has none to compare against")
        if result is None or hist is None or result.psi is None:
            raise RuntimeError("landing_report needs a solved run; call "
                               "run_gerch_sax() first")
        scene = self.scene
        x_axis = scene.x_axis
        dx = x_axis[1] - x_axis[0]
        rx_lo, rx_hi = scene.rx_ap.x_min, scene.rx_ap.x_max
        # the fixed amplitude constraint exactly as the solver held it
        amp = np.abs(hist.orig_aper_amp)

        per_freq, rec_rows, true_plate_rows = [], [], []
        for fs in self.freq_states:
            # gs_reconstruct's forward kernel and field construction, term for term
            h_fwd = rs.rs_kernel(x_axis, fs.wavelength,
                                 -1.0 * (scene.rx_ap.z - scene.tx_ap.z))
            rho = fs.freq / result.ref_freq
            rec_row = rs.rs_apply(amp * np.exp(1j * (rho * result.psi)), h_fwd, dx)
            rec_rows.append(rec_row)
            # the real plate as the solver could represent it: its own phase on the
            # scene axis, under the solver's amplitude
            true_u0 = interp_real_imag(fs.tx_ap.aper_axis, fs.tx_ap.aper_profile, x_axis)
            true_plate_rows.append(
                rs.rs_apply(amp * np.exp(1j * np.angle(true_u0)), h_fwd, dx))
            # the target against the reconstruction where the target EXISTS: every
            # window sample for an interpolated receiver, the element samples alone
            # for an on-grid one (rx_field is zero between its elements, and that
            # zero is "not measured", not a field to match)
            idx = fs.rx_ap.grid_idx
            if idx is None:
                match_target = field_match(fs.rx_field, rec_row, x_axis, rx_lo, rx_hi)
            else:
                match_target = field_match(fs.rx_field[idx], rec_row[idx], x_axis[idx],
                                           rx_lo, rx_hi)
            per_freq.append({
                "freq_hz": float(fs.freq),
                "true": landing_metrics(x_axis, fs.rx_plane_row, rx_lo, rx_hi),
                "reconstructed": landing_metrics(x_axis, rec_row, rx_lo, rx_hi),
                "match_true": field_match(fs.rx_plane_row, rec_row, x_axis, rx_lo, rx_hi),
                "match_target": match_target,
            })

        solver_loss = float(result.final_loss)
        true_plate_loss = self._joint_loss(true_plate_rows)
        dark = np.zeros(len(x_axis), dtype=np.complex64)
        return {
            "target": beam_target(self.tx_cfg, rx_z=scene.rx_ap.z),
            "rx_window": {"x_min": float(rx_lo), "x_max": float(rx_hi)},
            "dx_bias_m": propagator_offset(x_axis),
            "scene_dx_m": float(dx),
            "per_freq": per_freq,
            "loss": {
                "solver": solver_loss,
                "true_plate": true_plate_loss,
                "zero_model": self._joint_loss([dark] * len(self.freq_states)),
                "ratio": solver_loss / true_plate_loss if true_plate_loss > 0 else None,
                "n_iters": int(result.n_iters_run),
                "stop_reason": str(result.stop_reason),
            },
        }

    def log_landing(self) -> dict:
        """Log landing_report() as the LANDING block (landing_lines has the layout)
        and return the report."""
        report = self.landing_report()
        for line in landing_lines(report):
            self.log.info(line)
        return report

    def plot_scene(self, save_path=None, show=True):
        """Save the 4-panel scene plot (real vs MGS-reconstructed scene + TX aperture
        phase/amplitude). Writes to `save_path` if given, else self.plot_path; only
        opens an interactive window when `show` is True (set False for headless/batch)."""
        # imported here, not at module scope: pyplot costs ~137 ms and drags in
        # mpl_toolkits.mplot3d, which the grid sweep imports mgs without ever wanting
        import matplotlib.pyplot as plt

        self.log.info("Plotting scene")

        scene, gs_tx = self.scene, self.gs_tx
        bounds = (scene.x_axis.min(), scene.x_axis.max(),
                  scene.z_axis.min(), scene.z_axis.max())
        panel = dict(bounds=bounds, rx_axis=scene.rx_ap.aper_axis, rx_z=scene.rx_ap.z,
                     tx_axis=scene.tx_ap.aper_axis, tx_z=scene.tx_ap.z,
                     colorbar_label="EMW (V/m)")

        if scene.data is None or self.gs_rec_data is None:
            raise RuntimeError(
                "plot_scene needs both scenes illuminated; call illuminate_real() and "
                "illuminate_reconstructed() first")

        # the 4-panel figure renders at single-column width in the report, where it
        # shrinks ~3x, so the fonts are set large enough to survive that scaling
        with plt.rc_context({"font.size": 19, "axes.titlesize": 20,
                             "axes.labelsize": 19, "xtick.labelsize": 16,
                             "ytick.labelsize": 16, "legend.fontsize": 15}):
            fig = plt.figure(figsize=(10.5, 8), layout="constrained")
            (ax_real, ax_rec), (ax_phase, ax_amp) = fig.subplots(2, 2)

            v_max = np.nanmax([np.nanmax(np.abs(scene.data)),
                               np.nanmax(np.abs(self.gs_rec_data))])
            draw_scene(fig, ax_real, np.abs(scene.data), title="Scene Amplitude",
                       vmax=v_max, **panel)
            # Panel 2 autoscales. v_max above was computed to make the two panels
            # comparable and then commented out at the call; the reconstruction renders
            # dimmer than the real scene, so an independent scale is easier to read.
            # Pass vmax=v_max here instead to put them on one scale.
            draw_scene(fig, ax_rec, np.abs(self.gs_rec_data),
                       title="MGS Reconstruction Amplitude", vmax=None, **panel)

            # both aperture panels are drawn against the full scene x extent
            xlim = (scene.x_min, scene.x_max)
            tx_interp = interp_real_imag(scene.tx_ap.aper_axis, scene.tx_ap.aper_profile,
                                         scene.x_axis)
            gs_amp = interp_amplitude(gs_tx.aper_axis, gs_tx.aper_profile, scene.x_axis)

            phase_series, amp_series = [], []
            if self.has_real_aper:
                phase_series.append(("Real TX", scene.tx_ap.aper_axis,
                                     np.unwrap(np.angle(scene.tx_ap.aper_profile))))
                amp_series.append(("Real TX", scene.x_axis, np.abs(tx_interp)))
            phase_series.append(("MGS Reconstructed TX", gs_tx.aper_axis,
                                 np.unwrap(np.angle(gs_tx.aper_profile))))
            amp_series.append(("MGS Reconstructed TX", scene.x_axis, gs_amp))

            draw_line_panel(ax_phase, phase_series, title="TX Aperture Phase",
                            xlabel="x (m)", ylabel="Phase [rad]", xlim=xlim)
            draw_line_panel(ax_amp, amp_series, title="TX Aperture Amplitude",
                            xlabel="x (m)", ylabel="Amplitude EMF (V/m)", xlim=xlim)

            out_path = save_path if save_path is not None else self.plot_path
            self.log.info(f"Saving scene plot to {out_path}")
            fig.savefig(out_path)
            if show:
                plt.show()
            plt.close(fig)


def resolve_run_frequencies(config: SimConfig, cli_freqs: Optional[List[float]],
                            experimental: bool, log=None) -> List[float]:
    """The frequencies one `mgs` invocation runs at.

    The simulated path is config.resolve_frequencies as it stands: --freq, else
    the config's `frequencies`, else DEFAULT_FREQ_HZ. The experimental path shares
    the first two and NOT the fallback. A capture's carrier is a property of the
    bench, not of the project default, and exp_data reads the carrier out of the
    FFT bin at freq - lo_freq*trx_n: a carrier the rig did not transmit points
    that read at a bin the signal is not in. So with neither --freq nor
    `frequencies`, a capture is read at the carrier the receive chain's multiplied
    LO sits on, experimental.lo_freq * experimental.trx_n (150 GHz for the default
    bench constants, 25 GHz x 6).
    """
    if experimental and not cli_freqs and not config.frequencies:
        exp = config.experimental
        freq = float(exp.lo_freq * exp.trx_n)
        if log is not None:
            log.info(f"Experimental path with no --freq and no config frequencies: "
                     f"using the bench carrier experimental.lo_freq * "
                     f"experimental.trx_n = {freq / 1e9:g} GHz")
        return [freq]
    return resolve_frequencies(config, cli_freqs)


def main():
    parser = argparse.ArgumentParser(
        description="Program to simulate phase retrieval using a modified gerchberg-saxton algorithm"
    )
    parser.add_argument(
        "--debug",
        help="Enables debug logs",
        action="store_true",
        default=False
    )
    parser.add_argument(
        "--freq", "-f",
        help="One or more frequencies in Hz (overrides config `frequencies`). "
             "Multiple values run ONE joint solve for a single phase mask across "
             "all of them. Default: config `frequencies`, else "
             f"{DEFAULT_FREQ_HZ / 1e9:g}e9 (with --rx-path/--heatmap-path: else "
             "experimental.lo_freq * experimental.trx_n).",
        type=float,
        nargs="+",
        default=None
    )
    parser.add_argument(
        "--rx-path",
        type=Path,
        help="Path to .mat with experimentation data. Only a slice of z axis measurements",
        default=None
    )
    parser.add_argument(
        "--heatmap-path",
        type=Path,
        help="Path to .mat experimentation data. Expects full heatmap measurements",
        default=None
    )
    parser.add_argument(
        "--config",
        type=Path,
        help="Path to a simulation config .yml (sim_scene bounds and plot_path)",
        default=DEFAULT_CONFIG
    )
    parser.add_argument(
        "--out", "-o",
        type=Path,
        help="Run directory to write this run into, named outright "
             "(default: <output.output_dir>/<output.run_name>)",
        default=None
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="Override gerchberg_saxton.seed for a reproducible initial phase",
        default=None
    )
    parser.add_argument(
        "--no-save",
        help="Disable run persistence for this invocation",
        action="store_true",
        default=False
    )
    parser.add_argument(
        "--snr-db",
        type=parse_snr_db_arg,
        metavar="DB|off",
        help="Peak-referenced SNR in dB of the complex AWGN added to the simulated RX "
             "measurement (overrides noise.snr_db); 'off' forces a noiseless run",
        default=None
    )
    parser.add_argument(
        "--noise-seed",
        type=int,
        help="Override noise.seed for a reproducible noise draw",
        default=None
    )
    args = parser.parse_args()
    log = setup_logging(args.debug)

    config = load_config(args.config)

    # apply CLI overrides onto the config
    if args.out is not None:
        config.output.output_dir = Path(args.out).parent
        config.output.run_name = Path(args.out).name
    if args.seed is not None: config.gerchberg_saxton.seed = args.seed
    if args.no_save: config.output.save_run = False
    try:
        apply_noise_overrides(config, args.snr_db, args.noise_seed)
    except ValidationError as e:
        parser.error("invalid noise override: " + "; ".join(
            f"{'.'.join(str(p) for p in err['loc'])}: {err['msg']}" for err in e.errors()))

    # Fail fast on a run-directory collision, before spending the solve.
    if config.output.save_run:
        check_run_dir(config.output.output_dir, config.output.run_name, kind="mgs")

    is_exp = args.rx_path is not None and not args.heatmap_path is None
    freqs = resolve_run_frequencies(config, args.freq, experimental=is_exp, log=log)

    if is_exp:
        if len(freqs) != 1:
            parser.error("the experimental path (--rx-path/--heatmap-path) is a "
                         "single-frequency capture; pass exactly one --freq")
        rx_path = Path(args.rx_path)
        heatmap_path = Path(args.heatmap_path)
        mgs = MGS.from_experiment(rx_path, heatmap_path, freqs[0], config)
        mgs.run_gerch_sax()
        mgs.illuminate_reconstructed()
        mgs.plot_scene()
    else:
        mgs = MGS(freqs, config)
        mgs.illuminate_real()
        mgs.run_gerch_sax()
        # Logged HERE, ahead of the re-illumination and the plot: where the energy
        # landed is the answer a single run exists to give, and plot_scene can block
        # on an interactive window (or fail) with the numbers still unprinted.
        mgs.log_landing()
        mgs.illuminate_reconstructed()
        mgs.plot_scene()

    # Deferred to here on purpose: make_run_dir clears the target, so a crashed (or
    # plt.show()-blocked) solve must not have destroyed the previous run already.
    if config.output.save_run:
        run_dir = make_run_dir(config.output.output_dir, config.output.run_name, kind="mgs")
        save_run(mgs, run_dir, config, args.config, vars(args), is_exp=is_exp)

if __name__ == "__main__":
    main()

