# The solver's physical model and the warm-start initialization

Status: **current behavior** (both are the solver's only mode: the delay plate model
and, at multiple frequencies, the `warm_start` initialization by default).
Date: 2026-08-09. Branch: `freq-integration`.

## What the solver reconstructs

The joint MGS solver fits ONE aperture profile against every configured frequency's
measurement at once (`gs_reconstruct`, loss = mean of the per-frequency losses). The
unknown ψ(x) is a *plate's shape*: the aperture's delay/optical-path profile,
expressed as phase at a reference frequency, with each frequency's phase derived as

```
θ_f(x) = (f / f_ref) · ψ(x)
```

— the way a physical plate's phase scales with wavenumber. Both simulated beam types
are exactly this kind of source: the caustic construction integrates a phase whose
gradient is proportional to the wavenumber, and the steered beam is `−k·x·sinθ`. At
a single frequency the ratio is exactly 1, so ψ is simply that frequency's phase and
the solve is ordinary single-frequency phase retrieval.

The reference frequency is the centre-by-value of the run's frequency list (the same
convention as the scene views and the 3D-diff baseline), recorded as `ref_freq_hz`
in the sweep manifest's `gs` block and `run.json`'s `gs_result`. Every candidate in
a sweep shares the run-level reference — including candidates solved on a
valid-subset of frequencies — so all profiles carry the same units. The stored
reconstruction (`aper_profile`, `gs_tx_aper_profile`) is the aperture field at the
reference frequency; `mgs-animate --freq-index` rescales it to the phase the plate
presents at any selected frequency.

## Why multi-frequency solves need initialization care

Because the per-frequency phases are coupled through one shape, the profile must be
recovered in an **absolute** sense. That coupling is the information gain — only the
true delay is consistent with every frequency at once, which resolves the
single-frequency 2π ambiguity exactly as multi-wavelength interferometry does — but
it is also an optimization hazard: single-frequency retrieval is easy *because* it
is degenerate (profiles differing by per-point 2π wraps or a global offset are all
exact minima, so a random start falls into an acceptable one), and the
multi-frequency constraint turns all those formerly-equivalent solutions into
distinct basins, almost all wrong. Measured on `scenario_caustic_hit` at ±10%
bandwidth: plain gradient descent from a random start stalls ~100× above the
reachable floor (0.023 vs a verified ~2e-4), because escaping a wrong basin would
require moving whole stretches of ψ by ~2π through high-loss territory, which
per-point descent essentially never does.

## The warm start (`gerchberg_saxton.init: warm_start`, the default)

Four stages, inside every multi-frequency solve (a strict no-op at one frequency):

1. **Reference solve.** The channel nearest the reference frequency is solved alone
   — the forgiving single-frequency landscape recovers the profile's SHAPE. (Loss
   ~1e-5 at the true TX.)
2. **Unwrap.** The result's phase is unwrapped over the aperture support, resolving
   the per-point 2π ambiguities relative to each other; the remaining unknown
   collapses to ONE scalar, the absolute offset δ. (Assumes a connected support and
   a smooth profile — true for the shipped scenes: box candidate windows and the
   contiguous plate aperture.)
3. **Offset scan.** Adding δ multiplies channel f's field by `exp(i·(f/f_ref)·δ)`,
   so the joint loss is periodic in δ with the comb's synthetic wavelength,
   `2π·f_ref/Δf_min` (±10% case: ~63 rad; ±5%: ~126 rad). Each channel's field is
   propagated once and every δ sample costs elementwise work only (propagation is
   linear), so scanning one period at 2π/64 steps takes milliseconds. The best δ
   pins the absolute offset — the quantity no single frequency can observe.
4. **Joint descent.** The existing solver loop starts from ψ₀ + δ* and polishes
   into the true basin.

Cost: the reference solve adds roughly 1/F to the runtime; the scan is free.
`init: random` restores the plain seeded start (it reproduces the stuck behavior at
multiple frequencies and is otherwise identical at one).

The initialization helps the true TX *differentially*: there, a consistent δ exists
and the loss collapses to the floor; at wrong candidate locations no
profile-plus-offset reconciles several frequencies of wrong geometry, so their
losses stay pinned high. That contrast is the localization signal.

## Measured results

`scenario_caustic_hit`, dense 48×48 grid, seed 0, warm start
(`results/scenario_caustic_hit_{1f,pm5,pm10}`):

| Case | Frequencies | Best candidate | Joint loss | Error vs truth | Truth-cell rank |
| --- | --- | --- | --- | --- | --- |
| 1f   | 150 GHz | (0.302, −0.093) | 4.6e-6 | 8.7 mm | 10/2304 |
| pm5  | 150 GHz ± 5% | (0.302, −0.099) | 1.6e-5 | **3.0 mm** | **1/2304** |
| pm10 | 150 GHz ± 10% | (0.302, −0.105) | 2.4e-5 | 4.8 mm | 3/2304 |

Bandwidth helps: the ±5% joint solve localizes ~3× better than the single-frequency
baseline, and the residual basin is far sharper — the median candidate's joint loss
sits ~2000–3000× above the minimum, with only ~35 of 2304 cells within a decade of
it. At the true TX the warm-started solve converges to ~1e-5 with all per-frequency
components balanced.
