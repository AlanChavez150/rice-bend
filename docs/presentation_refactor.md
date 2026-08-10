# Deck refactor plan — "Localization of Curving Beams" (MGS Summer 26)

Slide-by-slide edit plan for the existing 22-slide deck. Ground rules from planning:

- **N_E(ε) stays out entirely** — no metric, no `study_metrics_vs_ndof` figures, no mode-count language.
- The multi-frequency work is called **"integrated multi-frequency support for MGS"** throughout.
- **Success criterion:** localization within **16 mm** of the true TX aperture center (~1.5 grid cells, under ⅓ of the beam's ~50 mm extent).
- **Beam width:** "≈50 mm wide (central 90% of energy; FWHM 30 mm)".
- Closing argument: **frequency diversity is not needed in this noiseless test case** — chiefly because of infinite SNR.

**Numbers policy.** Every number below is verified against the fresh Stage-13 rerun
(`results/study_frequency/study.json`, 64×48 = 3,072-candidate grid, cells ≈ 10.3 × 6.4 mm,
seed 0, warm start, ~83% of beam energy captured at every point). Do **not** reuse the older
8.7 mm → 3.0 mm table in `docs/delay_model_warm_start.md` — those runs used the retired
48×48 grid and their result directories were deleted on 2026-08-10.

---

## Edits at a glance

| # | Current slide | Action |
| --- | --- | --- |
| 1 | Title | Keep |
| 2 | Problem Statement | Keep (typo fix) |
| 3 | Jamming & Communications | Keep |
| 4 | Phase Retrieval Algorithms | Edit one sub-bullet |
| 5 | MGS example | Keep (refresh figure when rerun lands) |
| 6 | Proposed Solution | Tighten wording |
| 7 | Grid Search Implementation | Keep + add scale (typo fix) |
| 8 | Grid Search Example | Keep |
| — | **NEW 9: Integrated multi-frequency support for MGS** | Add |
| — | **NEW 10: Why a warm start is necessary** | Add |
| 9 | Ranking Candidate Beams | Rewrite → **11: Ranking & success criterion** |
| 10 | Simulation Results: Best Case | Replace → **12: Results — single frequency** |
| 11 | Simulation Results: Partial hit | Replace → **13: Results — multi-frequency** |
| — | **NEW 14: Does bandwidth help here? No — and here's why** | Add |
| 12 | Realistic antenna spacing (λ/2) | Keep as **15**, update numbers when λ/2 rerun finishes |
| 13 | Simulation Conclusions | Rewrite → **16** |
| 14 | Next Steps | Rewrite → **17** |
| 15 | Questions | Keep → **18** |
| 16–22 | Supplemental | Keep; typo fixes; add 2 slides |

Net: 15 → 18 main slides.

---

## Slide-by-slide content

### 1–3. Title / Problem Statement / Jamming & Communications — keep

Only fix on slide 2: "estimate **a** TX aperture parameters" → "estimate TX aperture parameters".

### 4. Phase Retrieval Algorithms — one edit

Change sub-bullet "Use of multiple frequencies" → **"Integrated multi-frequency support"**.
Keep the bold problem line — it sets up everything that follows.

> **Notes:** The two problems named here (no unique solution; TX geometry must be known) are
> answered later by the warm start + grid search respectively — call that out as a roadmap.

### 5. MGS example — keep

Refresh the figure with the regenerated `true_mgs_scene.png` when the queued
`grid-search-mgs --true-mgs` runs finish (see Figure tracker below). Current figure is fine
in the interim.

### 6. Proposed Solution — tighten

- MGS paired with a grid search generates a set of candidate beams
- Candidates are **ranked by how well their reconstruction fits the RX measurements** (residual error)

> **Notes:** Drop "various quality metrics" — the deck now commits to one ranking signal.

### 7. Grid Search Implementation — keep + add scale

Add one bullet:

- Current scale: **64 × 48 = 3,072 candidate TX locations** over a 650 × 300 mm search
  region (grid cells ≈ 10.3 × 6.4 mm); one fixed seed shared by every candidate so
  residuals are directly comparable

Typo: "Move onto to next" → "Move on to the next".

### 8. Grid Search Example — keep

Existing figure still illustrates the concept. Optional refresh later via
`grid-search-mgs --scenes` (not part of the queued rerun).

### NEW 9. Integrated multi-frequency support for MGS

**What it means:**

- MGS now solves **all frequencies in a single run** — one solve fits every frequency's RX
  measurement simultaneously (not one solve per frequency, averaged afterwards)
- The unknown is **one aperture phase profile**, defined at a reference frequency; each
  frequency sees that profile scaled by **f / f_ref** — the physically correct frequency
  scaling for a fixed aperture (a fixed path-length profile)
- Loss = mean of the per-frequency residuals; each frequency keeps its own measurement and
  its own RX element spacing
- A single frequency is just the length-1 case of the same solver

**Why do it:** several frequencies constrain the *absolute* phase profile — the
single-frequency 2π ambiguity cannot survive a comb (the multi-wavelength
interferometry principle).

> **Notes:** Emphasize "integrated": the frequencies constrain one solve, versus the earlier
> approach of independent per-frequency solves whose residuals were averaged after the fact —
> not the same measurement. At one frequency the new solver is bit-identical to the old one.

### NEW 10. Why a warm start is necessary

- Single-frequency retrieval is easy **because** it is degenerate: profiles differing by 2π
  wraps or a global offset all fit perfectly, so any basin a random start falls into is fine
- The multi-frequency constraint turns those formerly-equivalent solutions into **distinct
  basins — almost all wrong**
- Measured: plain gradient descent from a random start stalls **~100× above the reachable
  floor** (loss 0.023 vs ~2e-4, ±10% comb)
- Warm start (default): ① solve the center frequency alone → recovers the profile's *shape*;
  ② unwrap its phase → the remaining ambiguity collapses to **one scalar offset**; ③ scan
  that offset over one **synthetic-wavelength period** (2π·f_ref/Δf_min) — nearly free;
  ④ refine jointly. Extra cost ≈ 1/F of runtime
- Established practice in multi-wavelength optics [1–3]

**References (verified):**
1. Y.-Y. Cheng, J. C. Wyant, "Two-wavelength phase shifting interferometry," *Applied Optics* 23(24):4539 (1984)
2. J. Gass, A. Dakoff, M. K. Kim, "Phase imaging without 2π ambiguity by multiwavelength digital holography," *Optics Letters* 28(13):1141 (2003)
3. P. Bao, F. Zhang, G. Pedrini, W. Osten, "Phase retrieval using multiple illumination wavelengths," *Optics Letters* 33(4):309 (2008)

> **Notes:** The synthetic wavelength is the same trick two-wavelength interferometry has
> used since 1984: two nearby wavelengths beat together into a much longer effective
> wavelength that removes the 2π ambiguity. Solving one channel first and using a second to
> disambiguate is exactly the multiwavelength-holography recipe [2]; [3] brings it to phase
> retrieval proper. If pressed on cost: the scan is cheap because propagation is linear — the
> offset only rotates each frequency's already-propagated field.

### 11. Ranking & success criterion (rewrite of old slide 9)

- Candidates ranked by residual error; report the **argmin** location and the **top-10 mean
  distance** to the truth
- The beam is **≈50 mm wide** at the RX plane (central 90% of energy; FWHM 30 mm)
- **Success: argmin within 16 mm of the true TX center** — about 1.5 grid cells, and under
  ⅓ of the beam's extent
- Floor on any result: the grid itself — the best possible cell sits 5.2 mm from the true
  center, and cells are 10.3 × 6.4 mm

> **Notes:** Drop the old "region of interest" and "numerically indistinguishable" framing.
> The 16 mm bar is grid-aware: errors below ~1.5 cells are quantization, not physics.

### 12. Results — single frequency (replaces "Best Case")

- 150 GHz alone, λ/20 RX element spacing, no noise
- Argmin error **15.2 mm** (~1.5 grid cells) → **success**
- True TX cell ranked **7 of 3,072** — the best truth rank of *any* configuration tested
- Residual contrast: median candidate ~**15,000×** above the best fit

**Figure:** `results/study_frequency/point_00_bw00pct/residual_heatmap_hc.png`
(or `residual_surface_hc.png` for the relief view)

> **Notes:** This slide quietly plants the conclusion: one frequency already localizes to
> within the success bar with enormous ranking contrast. ~83% of beam energy is captured.

### 13. Results — multi-frequency (replaces "Partial hit")

- Same scene, 3-tone comb 150 GHz ± 5% (142.5 / 150 / 157.5 GHz), one integrated solve per
  candidate
- Argmin error **5.9 mm** → success; truth ranked 11 of 3,072
- The warm start works as designed — the joint solve converges at every candidate
- But across the full bandwidth sweep, most combs land at **15.8 mm** — the *same grid cell*
  as single-frequency

**Figures:** `results/study_frequency/point_04_bw05pct/residual_heatmap_hc.png`; optionally
`point_04_bw05pct/residual_scatter_3d.png` (one integrated solve, per-frequency components).

> **Notes:** Honest framing: the machinery is integrated and functioning (this is the
> engineering contribution), and ±5% happens to land closer — but the next slide shows that's
> not a trend.

### NEW 14. Does bandwidth help here? No — and here's why

**Figure:** `results/study_frequency/study_error_vs_bandwidth.png` (now includes the
bandwidth-0 anchor).

- Errors bounce **5.9–15.8 mm non-monotonically** from ±1% to ±20%; five bandwidths land on
  the *identical* grid cell; every difference is at or below the grid quantum (10.3 × 6.4 mm)
- Single frequency already succeeds: 15.2 mm, truth ranked 7/3,072 — the best rank tested
- **Why: infinite SNR.** With noiseless measurements the ranking can exploit residual
  contrast of ~1 part in 15,000. Frequency diversity adds discriminating constraint for when
  noise compresses that dynamic range — here there is nothing for it to rescue
- Single-frequency even fits *deeper* (best loss 3.3e-6 vs 1.2–2.8e-5 for the combs): the
  degenerate problem fits best, so residual ranking has more headroom, not less

> **Notes:** Scope the claim carefully — "not needed **in this test case**," not "not
> useful." Two caveats to volunteer before the audience does: (1) with receiver noise the
> four decades of contrast collapse toward the noise floor, and the multi-frequency
> constraint is the expected rescue — that's exactly the over-the-air regime (sets up Next
> Steps); (2) one frequency can never recover the *absolute* aperture profile (2π ambiguity)
> — irrelevant for localization-by-residual, essential if the aperture itself is the target.

### 15. Realistic antenna spacing (λ/2) — keep, update numbers

- λ/2 RX element spacing (vs λ/20 elsewhere); same captured energy
- Localization degrades: **[TBD — λ/2 rerun in progress, fills in today]** vs λ/20 at the
  same bandwidths
- The loss is measurement fidelity, not captured power

**Figure:** `results/study_frequency_lambda2/study_error_vs_bandwidth.png` once the study
completes.

> **Notes:** Pairs with the infinite-SNR argument: even noiseless, hardware realism (element
> pitch) already costs accuracy. Fill in the exact numbers from
> `results/study_frequency_lambda2/study.json` when the rerun finishes.

### 16. Conclusions (rewrite)

- A 1D RX aperture is enough: the residual-ranked grid search localizes the curving-beam TX
  to within ~1.5 grid cells (≤16 mm, ⅓ of beam width) in **every** configuration tested
- Integrated multi-frequency support works (single solve across a comb, warm-started), but
  **is not required in this noiseless test case** — single frequency matched or beat the
  combs on rank and fit depth
- The current error floor is set by grid quantization and the assumed candidate aperture
  (100 mm vs the true 132 mm), not by lack of information

### 17. Next Steps (rewrite)

- **Add receiver noise** — the regime where frequency diversity should earn its place; find
  the SNR where single-frequency ranking breaks
- Attack the assumed-aperture-width mismatch (100 mm candidate vs 132 mm truth)
- Validate over the air — the experimental `.mat` capture path already feeds the same solver

### 18. Questions — keep

---

## Supplemental slides (16–22 → 19–25)

- **17 (GS flowchart), 18 (closest analog), 19 (gradient descent), 20 (phase measurements):** keep as-is
- **21 (Phase Retrieval Using GS):** change sub-bullet "Use of multiple frequencies" →
  "Integrated multi-frequency support"
- **22 (MGS with Rayleigh-Sommerfeld):** fix "The propagation is replaced is replaced with" →
  "The propagation is replaced with"; "Its Computationally" → "It's computationally"
- **ADD: Warm start details** — the four stages with the synthetic-wavelength formula
  (period 2π·f_ref/Δf_min ≈ 126 rad at ±5%, scanned at 2π/64 steps), and the full reference
  list from slide 10
- **ADD: Full bandwidth sweep table** — the 10-row table below, for the inevitable "what
  about other bandwidths?" question

| ±% of 150 GHz | argmin err (mm) | top-10 mean (mm) | truth rank |
| --- | --- | --- | --- |
| 0 (single freq) | 15.2 | 17.9 | 7 / 3072 |
| 1 | 9.7 | 11.3 | 11 |
| 2 | 9.7 | 11.4 | — |
| 3 | 15.8 | 15.5 | — |
| 5 | 5.9 | 11.4 | 11 |
| 7 | 15.8 | 15.0 | — |
| 10 | 15.8 | 12.3 | 14 |
| 13 | 15.8 | 23.7 | — |
| 16 | 15.8 | 26.2 | — |
| 20 | 6.4 | 13.2 | 12 |

---

## Figure tracker

**Ready now** (all under `results/study_frequency/`):
`study_error_vs_bandwidth.png`; per-point `residual_heatmap{,_hc}.png`,
`residual_surface{,_hc}.png`, `residual_scatter{,_hc}.png`; multi-frequency points also have
`residual_scatter_3d{,_hc,_diff}.png`.

**Lands later today** (rerun batch in progress, queued in this order):
1. `results/study_frequency_lambda2/` study-level plots → slide 15 numbers + figure
2. `results/study_tx_shift/`, `results/study_tx_shift_lambda2/` (not currently used by the deck)
3. Fresh `results/scenario_caustic_hit_{1f,pm5,pm10}/` with `true_mgs_scene.png` → slide 5
   figure refresh

**Do not use:** anything referencing the deleted pre-Stage-13 runs, the old
8.7 mm / 3.0 mm / rank-1 table, or any `study_metrics_vs_ndof.png` (N_E is out of this deck).
