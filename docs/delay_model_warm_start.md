# The delay model's optimization problem, and the warm-start fix

Status: **implemented** (`gerchberg_saxton.init: warm_start`, the default; see the
Results section at the end for the measured outcome). Originally written as the
design note for that implementation.
Date: 2026-08-09. Branch: `freq-integration`.

## Background

The joint MGS solver fits ONE aperture profile against every configured frequency's
measurement at once (`gs_reconstruct`, loss = mean of the per-frequency losses).
`gerchberg_saxton.phase_model` sets what that one profile is:

- **`achromatic`** — a mask: the identical phase θ(x) at every frequency.
- **`delay`** (default) — a plate: the unknown ψ(x) is the delay/optical-path profile
  expressed as phase at a reference frequency (the centre of the run's frequency
  list), and each frequency's phase is `(f / f_ref) · ψ(x)` — the way a physical
  plate's phase scales with wavenumber. Both simulated beam types (caustic and
  steered) are exactly this kind of source.

The delay model was added because the achromatic model is physically mismatched for
plate-like sources: the bandwidth experiment on `scenario_caustic_hit` showed that at
F > 1 no single phase profile can reproduce every frequency's measurement even at the
exactly-correct TX location, so the loss floor rose from ~5e-6 (single frequency) to
~1e-2 and the residual minimum drifted off the truth (9 mm error at 150 GHz alone →
121 mm at 150 GHz ± 10%).

## The issue: the delay model is correct, but gradient descent cannot exploit it

Verification established two facts (probe: one full-resolution solve at the known TX
under `scenario_caustic_hit_pm10_delay.yml`, 135/150/165 GHz):

1. **The delay-consistent solution exists and fits.** Evaluating the joint loss at
   the plate's true profile — taken from the caustic construction *with its absolute
   value*, not from wrapped phase — gives ~2e-4 at every frequency (the residual is
   test-side interpolation noise; the true floor is lower).
2. **The solver never finds it.** From the standard random initial phase, descent
   drops fast to ~0.025 within 100 iterations and then crawls (~-8e-7/iteration,
   every backtracking step accepted), finishing at 0.023 — two orders of magnitude
   above the reachable floor, and no better than the mismatched achromatic model.

The mechanism: **the delay model's extra information and its optimization hardness
are the same thing.** Single-frequency phase retrieval only ever observes phase
modulo 2π, so its solution set is hugely degenerate — profiles differing by
per-point 2π wraps or a global offset are all exact minima. That degeneracy is what
makes the landscape benign: a random start falls into *some* acceptable minimum
(hence the 5e-6 single-frequency result). The delay model breaks the degeneracy —
the profile must be right in an *absolute* sense, because a 2π·n error at the
reference frequency scales to `(f/f_ref) · 2πn`, a large, inconsistent phase error
at every other frequency. All the formerly-equivalent solutions become distinct
basins, almost all wrong. A random start lands in one of them, and escaping would
require moving whole stretches of ψ by ~2π through high-loss territory — which
per-point gradient descent essentially never does.

Empirical confirmation (dense 48×48 sweeps, all six runs in `results/`):

| Case | Model | Joint loss (argmin) | Localization error | Truth-cell rank |
| --- | --- | --- | --- | --- |
| 150 GHz | either (bit-identical) | 4.6e-6 | **8.7 mm** | 10/2304 |
| ±5%  | achromatic | 0.0100 | 33 mm  | 156/2304 |
| ±5%  | delay      | 0.0094 | 69 mm  | 111/2304 |
| ±10% | achromatic | 0.0125 | 121 mm | 184/2304 |
| ±10% | delay      | 0.0122 | 121 mm | 154/2304 |

Every multi-frequency candidate — true location or not — stalls at the same ~1e-2
level, so the ranking the localization depends on is noise from stuck
optimizations. (At a single frequency the two models are the same solve — the
frequency ratio is exactly 1 — which the 1f runs confirm bit-for-bit.)

## The warm start: initialize inside the true basin

The standard multi-wavelength recipe, as a solver initialization stage. Four steps,
each doing one specific job:

1. **Solve the reference frequency alone** (the existing F=1 path). Its degenerate
   landscape is now a *feature*: an easy solve that recovers the profile's SHAPE,
   correct up to per-point 2π wraps and one global constant.
2. **Unwrap** the stage-1 phase along x over the aperture support (`np.unwrap`).
   Valid because the plate profile is smooth and well-sampled (~0.34 rad/sample for
   the caustic scene, far under the π limit). This resolves all per-point wraps
   *relative to each other*, collapsing the remaining unknown to a single scalar:
   the absolute offset δ, with ψ_true ≈ ψ_unwrapped + δ.
3. **Scan δ against the joint loss.** Adding δ multiplies each frequency's field by
   `exp(i·(f/f_ref)·δ)`, so the joint loss is periodic in δ with the comb's
   **synthetic wavelength**, `2π·f_ref/Δf` (±10% case: 2π·150/15 ≈ 63 rad; ±5%:
   ≈ 126 rad). Evaluate the joint loss on a grid of δ over one period — each sample
   is F forward propagations, well under a second total — and keep the minimizer.
   This is the multi-wavelength interferometry step: one frequency can never
   observe the absolute delay; a comb converts it into an observable, unique within
   one synthetic period (residual whole-period ambiguity shifts every frequency by
   full 2π turns simultaneously and is genuinely equivalent).
4. **Joint gradient descent from ψ_unwrapped + δ\*** — the existing delay solve,
   now started inside (or beside) the true basin, polishing interpolation noise and
   stray wrap defects down toward the ~1e-5–1e-4 floor instead of stalling at 1e-2.

**Why this restores localization, not just the fit at the truth:** the scan helps
the true location *differentially*. At the true TX a δ exists that makes all
frequencies consistent, so the loss collapses; at a wrong candidate no
profile-plus-offset can reconcile three frequencies of wrong geometry, so its loss
stays high. That re-creates the basin contrast the stuck sweeps lack, and bandwidth
becomes a constraint that sharpens the minimum rather than noise that flattens it.

**Cost:** per candidate, one extra single-frequency solve (~one channel's share of
the joint solve) + the sub-second δ scan + the joint solve as today — roughly
1.5–2× the current per-candidate cost.

**Where it would live:** inside `gs_reconstruct` as an opt-in initialization
(e.g. `gerchberg_saxton.init: warm_start`), preserving the solver's purity (same
arrays-in/arrays-out contract, runs unchanged in sweep workers).

**Caveats:** stage 2 trusts that the stage-1 solution is smooth over a connected
support — true for the shipped scenes; a fragmented support or genuinely
discontinuous plate would need a more careful unwrapping step. Stage 1's solution
may also carry isolated interior wrap defects that survive unwrapping; the final
descent is expected to fix point defects, but this is the residual risk to check
when the stage is implemented.

## Results (measured after implementation, same date)

The collapse gate passed: the warm-started pm10 delay solve at the true TX reaches a
joint loss of **1.8e-5** (was 0.023 stuck; the hand-verified floor was ~2e-4, which
the solver polished past), all three per-frequency components balanced, converged in
403 iterations, bit-reproducible across runs, and still seed-sensitive.

The rerun dense sweeps (`results/scenario_caustic_hit_*`, 48×48, seed 0):

| Case | Model / init | Best candidate | Joint loss | Error vs truth | Truth-cell rank |
| --- | --- | --- | --- | --- | --- |
| 150 GHz | either (identical) | (0.302, −0.093) | 4.6e-6 | 8.7 mm | 10/2304 |
| ±5%  | achromatic / random | (0.274, −0.080) | 0.0100 | 33 mm | 156/2304 |
| ±5%  | delay / warm_start  | (0.302, −0.099) | 1.6e-5 | **3.0 mm** | **1/2304** |
| ±10% | achromatic / random | (0.413, −0.144) | 0.0125 | 121 mm | 184/2304 |
| ±10% | delay / warm_start  | (0.302, −0.105) | 2.4e-5 | **4.8 mm** | 3/2304 |

Bandwidth now helps instead of hurting: the warm-started delay runs localize ~2–3×
better than the single-frequency baseline, the nearest-to-truth grid cell ranks
first (±5%) / third (±10%), and the basin is dramatically sharper — the median
candidate's joint loss sits ~2000–3000× above the minimum (the stuck runs managed
~5×), with only ~35 of 2304 cells within a decade of it. The differential mechanism
worked as predicted: at the true TX the warm-started descent drives every frequency
down together; at wrong locations no profile-plus-offset reconciles three
frequencies of wrong geometry, and the loss stays pinned high.
