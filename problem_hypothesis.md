# Why measured performance dropped once TX power became unknown

Investigation of 2026-10-21 on measured row 177 (TX at 242 mm), 300 GHz, single RX, W = 100 mm known.

## The question

> It seems that relatively wide RX aperture is now required. I suspect this is because my phase plate is not working well: some of the EM energy travels through my TX aperture without being modulated correctly. There may be other issues happening that I would like to understand. There has been in general a huge nosedive in performance since the TX aperture power is no longer known. I understand why this might happen, but I also don't think it fully justifies the drop in performance.

## Short answer

You're partly right about the plate, but it isn't what forces the wide RX. Not knowing the TX power does that, for any plate. Part of the drop is still avoidable, though: the plate's edge leakage, some solver behaviour and time-window cuts in the data add error that physics doesn't require.

Method: a multi-agent workflow (`wf_ffb3c5e5-c49`) of six investigation lanes and three skeptic verifiers. The repo was not touched.

## 1. Your plate hypothesis: largely confirmed

The other raster rows were used only as a diagnostic, never in the localization method.

- At 300 GHz only 22–30% of the TX power leaves at angles that can reach the main lobe. An ideal curving (caustic) plate would send 76% that way.
- About 53% goes nearly straight ahead, against 9% for an ideal plate. On far rows about 65% of the power sits under the plate.
- It isn't a separate leak: nothing arrives earlier than the main pulse above −37 dB. It is the plate's own efficiency, and it depends strongly on frequency:
  - there is little bending below about 275 GHz;
  - 300 GHz sits on the rising edge;
  - 310–345 GHz bends about 25–30% more;
  - 520–600 GHz bends best.
- The part of the plate that actually radiates is about x 89–189 mm, not 85–185 mm. Its right edge is bright, at about 2× the median amplitude.

## 2. Why the RX must be wide: mostly physics, with power unknown

- **Narrow RX:** with the power free, a plate placed just above the RX can reproduce whatever an RX narrower than the plate sees, and send the extra energy outside the RX. The RX therefore has to be wider than the 100 mm plate (by about 10 mm), and the beam at the RX has to be wider than 100 mm.
- **The plate isn't the cause:** an ideal plate making your measured beam shape ("plate M") gives the same width thresholds you measured, with no defect at all.
- **A perfect plate would do worse:** a concentrated caustic carries no range information without the power. The simulated twin only looked fine because of how the solver initialises the plate phase (backprop start reads the phase over the lit span only), which disadvantages candidates near the RX.
- **The misrouted light actually helps:** it makes the beam broad, and a broad beam is what gives range information when the power is unknown.
- **What hurts is energy outside the assumed 100 mm window.** With the power unknown, the plate width is the only range ruler. Putting about 3% of the TX power outside the best 100 mm window turned an exact noiseless localization into −20 mm at 6k iterations and −62 mm at 30k.

## 3. Why the drop is bigger than it needs to be

| Issue | Effect on the result |
|---|---|
| Iteration cap acts as a hidden regularizer | Measured 120 mm descend: 24 / 20 / 28 / 38 mm at 6k / 20k / 30k / 100k iterations. Known power stays at 0 mm on the twin. |
| Solver depends on the data scale (phase step ∝ s², amplitude step ∝ s, fixed lr0 and stop threshold) | The arbitrary `amplitude_scale` alone gives 12 / 24 / 20 mm at s = 0.5 / 1 / 2.24. |
| False "converged" stops (the step can be perpendicular to the true gradient; the level step can have the wrong sign) | 38–50% of measured level-mode candidates stall, so level maps are unreliable. |
| Absolute stop threshold (−1e-10) | Causes the twin's 26.7 mm error; without it the twin gives 2 mm. |
| 400 vs 401 samples in the plate window | 10–30% loss swings at the bright right edge. |
| 200 ps time frame starts in 50 ps steps per row and cuts 300 GHz main-lobe energy still arriving 100+ ps later | Rows agree with each other only to −16 dB (noise floor −25 dB). Adding that mismatch to the ideal twin moves descend from 0–8 mm to 28–68 mm over 4 noise draws; known power stays at 0 mm. |

The late main-lobe content cannot come from an ideal plate. Gating to the RX alone did not reliably help.

## 4. Corrections to earlier claims

- The "9–24× worse fit at the true TX" is mostly a ~4 mm x offset. With the plate centred at about x 139, the true-range loss is 1.0–1.45× the best, so it isn't evidence of a bad model.
- The twin "reproducing" the measured ~25 mm error didn't prove the error is intrinsic. That number came from the stop threshold.
- The known-power 19 mm result had help: its scale came from the whole 200 mm row. Its error is also a cross-range offset that grows with range (+13 / +19 / +25 mm), which points to a beam-direction mismatch, not a power error.

## What would help, most useful first

1. **Fix the solver's dependence on scale and budget:** steps that don't depend on the data scale, a relative stop threshold, real stall detection, and an exact-width window. These change the MGS algorithm, so they need sign-off first.
2. **Recapture with time frames aligned to the arrival time** and long enough to hold the late main-lobe energy and the second pulse.
3. **Put an absorbing mask exactly 100 mm wide on the plate**, since the width is the range ruler once power is unknown.
4. **Try 310–345 GHz, averaged over the band**, where the plate bends more.
5. **Change the decision rule only after the data mismatch is fixed.** On ideal broad-beam twins, "farthest candidate whose loss is at or below the noise floor" gives 0–16 mm where the lowest-loss candidate gives 44–242 mm. On the measured data it fails until the mismatch is reduced. A minimum-range prior did not help.

## Caveats

- Most twin thresholds rest on one noise draw.
- Some candidate sets were centred on the truth, which is optimistic.
- Several argmins are near-ties.
- A separate study earlier the same day (range identifiability) reached the same conclusions on the early stopping and on row 177 not fitting any model (−14 dB row consistency).

## Where the raw results are

These are in the session scratchpad, which is temporary:

- `/tmp/claude-1000/-home-alanchavez-Desktop-rice-bend-git/88b5682b-4f96-40aa-86d0-98ce6db77d9d/scratchpad/investigate/wf_result.json`: full workflow result.
- `/tmp/claude-1000/-home-alanchavez-Desktop-rice-bend-git/88b5682b-4f96-40aa-86d0-98ce6db77d9d/scratchpad/investigate/lane_findings.txt`: lane findings.
- Per-lane directories in the same folder: `A_tx_forensics` (including `tx_field_300GHz.npz`), `B_ideal_twin`, `C_defect_twins`, `D_forward_model`, `E_solver_audit`, `F_estimator_theory`. Verifier logs are in `verify_AC`, `verify_BF` and `verify_DE`.

The agents were blocked from writing their own report files, so the structured results and logs are the full record.
