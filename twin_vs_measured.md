# Why the digital twin localizes and the measured row does not

Investigation of 2026-10-22 on measured row 177 (TX plate at 242 mm) against its digital twin
(`configs/twin/twin_caustic_rx120.yml`), 300 GHz, single RX of 120 mm, uniform-level plate with
the excess-power penalty inside the descent (`gerchberg_saxton.amplitude: level`,
`power_penalty: 1`), 6000 iterations, rust engine. Method: five evidence lanes (fit gap against
the noise floor, row-to-row consistency, beam and wavefront geometry, twin ablations, manifest
forensics), every claim attacked by three skeptics (physics, method, statistics), a synthesis,
a completeness critic, then the decisive cells re-solved by hand. 73 agents, about 70 minutes.

## The question

> Tell me why the digital twin performed so much better than the experimental data.
> Are there any other reasons why I might not be able to find the location when using
> experimental data?

## Short answer

The twin is an inverse crime twice over: its candidate window *is* its plate (uniform amplitude,
support exactly the 100 mm window, a smooth caustic phase), and the solver's back-propagated start
at the true cell is that plate's phase up to noise. The measured row has neither. Of the measured
"failure", most of the 23.7x handicap of the true cell is a 4 mm misregistration of the truth
window (the plate radiates from stage 89..189, not 85..185), and the 36.9 mm range error is not a
point error at all: the uniform-level objective has no range valley on this field, so the argmin
is wherever a 100 mm flat floor happens to bottom. A twin that carries the documented plate defects
(53% unmodulated pass-through plus a 2x bright right edge) reproduces the measured failure. Noise,
iteration budget and the penalty exponent are ruled out by measurement. Beyond the plate, there
are several experimental effects the pipeline does not model and has not yet tested; section 8
lists them with the test for each.

## 1. What was compared

All runs: 2 mm grid of 30,200 candidate TX positions (z 0..0.40 m, x_center -0.15..0.15 m), a
100 mm candidate plate, 300 GHz, backprop start, seed 0, 6000 iterations. The bench TX is at
z 0.242 m, x_center -0.020 m in the solver frame (z_sim = (177 - z) mm, x_sim = (x - 155) mm).

| Run | Config | Argmin | Error | True cell / best | R at best | 1.5x set |
|---|---|---|---|---|---|---|
| Twin, level + penalty | `twin/twin_caustic_rx120_level_pp1.yml` | z 242, x -20 | 0.0 mm | 1.00x (rank 1) | 1.03 | 28 cells, z 224..244 |
| Measured, level + penalty | `measured/rxsize/rx177_level_w120_pp1.yml` | z 206, x -12 | 36.9 mm | 23.7x | 1.98 | 97 cells, z 186..286 |
| Measured, level, no penalty | `measured/rxsize/rx177_level_w120.yml` | z 164, x -10 | 78.6 mm | 21.6x | - | 132 cells, z 122..274 |
| Measured, free amplitude + penalty | `measured/rxsize/rx177_descend_w120_pp1.yml` | z 220, x -16 | 22.4 mm | 16.2x | 1.26 | 192 cells, z 136..272 |

The twin's best cell sits at its noise floor (J 2.61e-5 against a floor of 2.92e-5). The
measured runs' winning plates all radiate about twice the power the receiver collects.

## 2. The twin is an inverse crime, in geometry and in initialization

**Geometry.** The twin's plate is a uniform amplitude on exactly the candidate window with a
smooth caustic phase, propagated by the same 1D scalar kernel the solver inverts, and its
back-propagated field at the true range returns that plate with 2% amplitude ripple. The real
plate, back-propagated from row 177 to z 242, has 44% rms amplitude ripple, a right edge 1.9x
the median, a dark left 5 mm, and its radiating region is stage 89..189. Half of the measured
RX energy (50.7%) sits on the far side of the row, under the plate, against 5.8% for the twin:
that is the unmodulated pass-through `problem_hypothesis.md` estimated at 53%.

**Initialization.** Started from a random phase instead of the back-propagated data, the twin's
own search lands at z 202 (seed 0) or 206 (seed 1) with the true cell 2.4x / 1.8x above the
best, which is the measured failure. The twin's valley is real and deep (on the noiseless twin
the true cell beats z 246 by 369x and z 278 by 4519x), but at 6000 iterations the solver finds it
only from the backprop start, and at the twin's true cell that start is the generator plate's
phase up to noise. The measured ranking does not depend on the start (random seeds give the
argmin at z 194 and 210, on the same flat ridge).

## 3. Cause 1: the measured "truth" window is misregistered by 4 mm

The truth marker uses the nominal plate at stage 85..185. The plate radiates from 89..189. On the
measured penalized run's z = 242 row, J relative to the best cell of the whole grid:

| x_center (mm) | -24 | -22 | -20 (85..185) | -18 | -16 (89..189) | -14 | -12 | -10 | -8 |
|---|---|---|---|---|---|---|---|---|---|
| J / best | 45.2 | 38.4 | **23.7** | 7.2 | 2.2 | 1.9 | 1.5 | **1.41** | 1.6 |

The row minimum is 1.41x the global best; the nominal window is 16.9x the row minimum; 16.9 x 1.41
= 23.7. In log terms the registration is 89% of the handicap. A 56-window edge sweep puts 99% of
the variance of ln J on the right edge and none on the left: the bright edge at stage 189 is what
the nominal window cuts off. At the true range the measured field *is* explained to within its
noise once the window is right: fixed-amplitude fit at 89..189 L 9.6e-6 (1.4x the floor,
converged); uniform level at 89..189, 20,000 iterations, L 1.04e-5 (1.5x the floor), J 2.55e-5
against the converged argmin's 1.99e-5. On the twin this term is exactly 1 by construction.

## 4. Cause 2: the measured objective carries no range information

J along the valley ridge (the minimum over x in each z row), relative to the best cell:

| z (mm) | 194 | 206 | 218 | 230 | 242 | 254 | 266 | 278 | 290 |
|---|---|---|---|---|---|---|---|---|---|
| Measured, 6000 it | 1.44 | 1.00 | 1.38 | 1.60 | 1.41 | 1.23 | 1.18 | 1.41 | 1.87 |
| Measured, converged (12,000 it) | 1.25 | 1.05 | 1.17 | 1.48 | 1.22 | 1.28 | 1.18 | 1.42 | 2.12 |
| Twin, 6000 it | 4.52 | 2.70 | 1.68 | 1.27 | 1.00 | 5.90 | 12.1 | 18.2 | 23.6 |

The measured ridge is flat within 1.7x over 100 mm, four times the 23 mm depth of focus, with
local minima at 206, 242, 254 and 266. Converged, the argmin moves from 206 to 210..218, so about
13 of the 36.9 mm is budget jitter on a flat floor and the rest is the absence of a minimum. The
free-amplitude run has the same shape (within 1.25x from z 194 to 278).

Where the twin's valley comes from: the back-traced footprint of its ray fan is 34 / 57 / 77 /
110 mm at z 164 / 206 / 242 / 300, so it fills the 100 mm plate only near z 242, and the twin's
argmin follows the candidate width (60 / 100 / 140 mm windows give z 206 / 242 / 260). The
measured beam is nearly collimated (5..95% of its angular spectrum within sin theta -0.07..+0.21
against -0.33..+0.37 for the twin), its footprint is 65..76 mm at every z, always inside the
window, and the same width test gives z 286 / 206 / 300. The measured range is set by an
arbitrary model width, not by the data.

## 5. Cause 3: the real plate is not the twin's plate

| | Twin | Measured row 177 |
|---|---|---|
| RX energy on the far side (x < 0) | 5.8% | 50.7% |
| Needle -3 dB width | 6.2 mm | 2.7 mm (lambda z / D = 2.4 mm) |
| Back-propagated plate, std / mean | 0.02 | 0.44 |
| Right edge / median | 1.05 | 1.94 |
| R of the 1.5x-set plates | 1.03..1.15 | 1.77..2.27 |

Twin ablations, a 1071-cell coarse grid (z 0.10..0.40 step 6 mm, x -0.06..0.02 step 4 mm) of the
same penalized level search, with the twin's plate modified and the rx_peak scale re-derived:

| Twin variant | Argmin error | R at best | True cell / best | 1.5x set |
|---|---|---|---|---|
| as configured | 0 mm | 1.03 | 1.00 | 4 cells, z 224..242 |
| + 37 dB noise (the measured floor) | 0 mm | 1.04 | 1.00 | 2 cells |
| + 53% flat-phase pass-through | 6 mm short | 1.64 | 1.00 (rank 4) | 23 cells, z 116..242 |
| + bright right edge (1 -> 2x) | 6 mm short | 1.41 | 1.30 | 6 cells |
| + pass-through and bright edge | **42 mm short** | **2.37** | 1.17 (rank 15) | 39 cells, z 122..248 |
| plate shifted +4 mm, marker not | 0 mm in z, +4 in x | 1.17 | 4.25 | 3 cells |

The two documented defects together reproduce the measured signature: a flat set tens of
millimetres short in range, winning plates at R 2.4, the nominal truth beaten. Freeing the
amplitude does not undo it on the measured row either (22 mm short, true cell 16x), because the
pass-through is not a plate feature the window can hold.

## 6. Ruled out

| Suspect | Measurement |
|---|---|
| Receiver noise | Measured floor L 6.9e-6 (37.6 dB re the RX peak, from the 800..1000 GHz bins) against the twin's 2.9e-5 (30.1 dB): the data are four times cleaner than the twin. |
| Iteration budget | Nominal true cell J 5.72e-4 at 6000 iterations, 5.67e-4 at 21,000 (converged), 5.65e-4 at one million. The argmin cell goes 2.41e-5 to 1.99e-5. The gap widens. |
| Penalty exponent | Post hoc J = L R^gamma for gamma 0..20: no exponent ranks the nominal truth first (best rank 1548 at gamma 2), because 51 cells have both lower L and lower R than the truth. |

## 7. The 200 ps TDS frame

The 300 GHz bin is one 5 GHz slice of a 200 ps rectangular frame over a response longer than
180 ps. Of the -14 dB row-to-row inconsistency with the 1D model, 60..80% is a frame-end
artefact: the residual correlates 0.91 with the tail mismatch and 0.09 with distance, and
frame-matched rows agree at -25..-27 dB (2..2.5x the two-row noise floor). 38 of the 101 rows,
including 150 and 165, are cut at the frame start. Row 177's lobe leaves 5 / 11 / 20% of its
energy in the last 10 / 20 / 50 ps. For a single-row search this costs little (the 89..189 fit
still reaches 1.4x the floor); it will dominate any scheme that uses a second row for range.

## 8. Other reasons localization can fail on experimental data (not yet tested)

The twin comparison could only test what the twin could vary. These are the remaining
experimental effects, roughly in order of suspicion, each with the test that would settle it.

1. **Timing and phase drift across the raster.** Each point is a separate pulse acquisition over
   hours. A slow delay-line or laser drift along the row is a phase ramp or curvature across x;
   a ramp shifts the apparent source in x, a curvature changes the apparent range directly, since
   range from one row is read from the wavefront curvature. A few degrees over 120 mm matter at
   300 GHz. Test: a repeat scan in the opposite direction, odd against even points, or the phase
   of a fixed reference reflection in the traces.
2. **Late arrivals inside the frame.** The raw traces show second arrivals about 25 and 47 ps
   after the main peak at 15..50% of its envelope, consistent with internal reflections in a few
   millimetres of plate material (n about 1.5). At 300 GHz they add coherently as sources at other
   effective ranges, and the solver has one source plane. Test: gate each trace to the first
   arrival before the DFT and rerun the ridge.
3. **Absolute geometry.** The 242 mm is plate face to nominal detector position. A silicon-lens
   coupled antenna has an effective phase centre that can sit 5..15 mm from where the stage says
   the detector is, and a scan axis tilted by one degree walks 2 mm in z across the row. Nothing
   in the pipeline checks this. Test: scan a point-like source or a pinhole at a known distance
   with the same detector and run the localization on it.
4. **Detector angular response and aperture.** The bent lobe arrives at about 17 degrees, the
   pass-through at normal incidence. A directional detector reweights them against each other
   and low-passes the field over its aperture. This commutes with propagation, so the row-to-row
   check cannot see it, but it distorts the apparent plate and the wavefront. Test: pre-whiten
   the row with a cos^n directivity for n 0..8 and see whether the ridge moves.
5. **The 1D kernel.** The solver propagates line sources with a 1 / sqrt(r) amplitude law. The
   plate has a finite height, the detector sits at one y, the beam spreads in y. Across the row
   r runs 242..285 mm, so the 1 / r against 1 / sqrt(r) difference is about 8% in amplitude, the
   order of the level fit's residual. Test: a finite-height kernel built by quadrature for
   H = 10..160 mm, applied to the stored true-cell plate and compared on the row.
6. **Bin content and dispersion.** One 5 GHz bin of a rectangular frame, on a plate whose
   bending efficiency rises steeply through 300 GHz. Test: the same ridge at 295, 305, 310 and
   330 GHz. If the argmin wanders by more than 20 mm between adjacent bins, the 36.9 mm is one
   draw of a bin lottery and must be reported as a spread.
7. **Two source planes.** The unmodulated light originates at the illumination optics behind the
   plate, not at the plate, and the illumination itself is not uniform across it. The row is a
   superposition of sources at different ranges with different amplitude profiles, which a
   single uniform plate cannot represent, so the fit lands at a compromise range. This is the
   modelling side of the leak-plus-edge ablation.
8. **Polarization and plate birefringence.** The detector is polarization sensitive and a
   3D-printed or layered plate may be birefringent; the scalar model ignores both. Low on the
   list, and only a measurement with the detector rotated would show it.
9. **The intrinsic one.** A nearly collimated beam carries little curvature information in a
   single row, whatever the model. The twin had range leverage because its ray fan was wide. For
   the real beam the honest range constraint from one row is the flat 100 mm set, and the fix is
   a second row (prevented today by the frame-start cuts) or several frequencies solved jointly
   (supported by the solver, not yet used on the measured data).

Also unknown and only answerable from the lab notebook: the plate height and the y position of
the scan line, the detector module (lens, aperture, polarization), whether the plate is a
cylindrical (1D) profile, and its thickness and material.

## 9. What to do, most useful first

1. Re-register the measured ground truth to where the plate radiates (stage 89..189, x_center
   -0.016, or the power centroid at -0.0115) and re-score the existing penalized manifest. No new
   solves. Expected: the true cell's handicap falls from 23.7x to about 2x at 6000 iterations
   and 1.3x converged, and the headline becomes a pure range problem on a 100 mm flat set.
2. Bin stability and time gating. Both need no new data and about an afternoon. Gating also
   answers item 2 of section 8 and is the prerequisite for any second-row scheme.
3. A pinhole or point-source calibration scan at a known distance. Needs lab time, but it is
   the only way to settle the absolute geometry and the detector response together.
4. A replica twin: back-propagate row 177 to z 0.242, mask to 89..189, forward-propagate with
   37 dB noise, and run the grid search. One skeptic showed this reproduces the measured beam
   (88.5% against 88.6% of the energy in the RX, peak at stage 209.5). If its ridge is flat, the
   plate profile is the whole story.
5. A two-component source: a level plate on the window plus a free unmodulated pass-through
   term, keeping the MGS descent and the power penalty. The goal is R near 1 at the true range so
   the penalty regains its gradient. Validate on the leak twin first.
6. Measure the plate's exit amplitude from the near-plate rows and run the measured grid in
   fixed-amplitude mode with it; at 89..189 the fixed fit already reaches 1.4x the floor.

## Caveats

The twin's results rest on one noise draw (seed 0). The width test and the random-start test are
one seed each at 6000 iterations with most cells at the cap. The measured noise floor assumes the
TDS noise is white in time, so the 300 GHz bin carries the same per-bin variance as the
800..1000 GHz tail; the tail is flat and zero-mean, which supports but does not prove it. The
"-25 dB frame-matched" number rests on rows 174 and 180. The twin ablations change the plate, not
the beam's angular content: no variant reproduces the measured 2.7 mm needle (all 5.5..6.5 mm),
so a twin with the leak and the edge is still not the experiment.

## Where the evidence is

- Runs: `results/measured/rxsize/rx177_level_w120_pp1/`, `rx177_level_w120/`,
  `rx177_descend_w120_pp1/`; `results/twin/twin_caustic_rx120_level_pp1/`; figures in
  `results/descent_penalty/`. Earlier rounds (old windows; clamp without the exact level
  gradient) under `results/measured/rxsize/pre_window_fix/` and `pre_w2_gradient/`.
- Investigation scripts, logs and small data: `results/why_twin/` (lanes `fit_gap`,
  `row_consistency`, `beam_geometry`, `twin_ablations`, `manifest_forensics`, the `refute/`
  folders per claim and lens, `completeness_critic/`, and `workflow_result.json` with every
  claim, verdict and the synthesis). Large residual dumps were left out.
- The twin ablation table: `python results/why_twin/twin_ablations/table.py`.
- The random-start and converged ridges: `results/why_twin/refute/manifest_forensics_C1_physics/analyze_resolve.log`.
