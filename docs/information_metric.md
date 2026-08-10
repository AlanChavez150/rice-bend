# Decision: N(eps) as the information metric for RX aperture capture

**Status:** adopted (2026-08-09).
**Decision:** quantify the information the RX aperture captures with a single number,
`N(eps)` — the number of modes of the TX->RX radiation operator that the transmitted
beam excites above the receiver noise floor, at a stated retrieval accuracy
`eps = 0.1`. All definitions are drawn from the electromagnetic inverse-source
literature (NDF / dimension of the data space); no statistical effective-DOF
constructs are used.

## Definition

Build the discrete radiation operator exactly as `gs_reconstruct` applies it:
column j = `rs.rs_apply(e_j, h_fwd, dx)` for a unit source at TX-support sample j,
rows restricted to scene samples inside the RX window. Then:

```
M = U S Vh            SVD; sigma_k = S[k] are the mode gains
c = Vh @ u0[support]  coupling of the actual TX beam into mode k
N(eps) = #{ k : sigma_k * |c_k| >= sigma_n / eps }
```

- `sigma_k` carries the geometry (TX width, RX width, standoff z, wavelength)
- `|c_k|` carries the beam (where it points, what lands in the window)
- `sigma_n` carries the receiver noise (SNR)
- `eps` is the accuracy convention: a mode below the threshold cannot be
  retrieved to relative accuracy `eps` by any inversion (TSVD truncation logic)

The metric is reported as e.g. `N(0.1) = 8`, never as a bare integer: the
accuracy convention travels with the number.

## Why this metric fits this project

**It responds to both observed failure axes with one number.** Empirically
(this repo, 150 GHz, 30 dB peak SNR unless stated):

| condition                | NDF (rank at 1e-2) | energy in window | N(0.1) |
| ------------------------ | ------------------ | ---------------- | ------ |
| `scenario_caustic_hit`   | 12                 | 83.3%            | 8      |
| `scenario_caustic_miss`  | 13                 | 0.5%             | 1      |
| hit, SNR 10 dB           | 12                 | 83.3%            | 1      |
| hit, SNR 20 dB           | 12                 | 83.3%            | 6      |
| hit, SNR 40 dB           | 12                 | 83.3%            | 10     |

A beam walking off the window collapses the `|c_k|`; lowering SNR raises
`sigma_n`; both drive the same inequality. The bare NDF cannot distinguish hit
from miss (12 vs 13) because the channel geometry barely changes — the failure
is in what the beam excites, not in what the geometry supports.

**Its high-SNR limit is the classical NDF.** As `sigma_n -> 0`, `N(eps)`
saturates at the number of relevant singular values of the operator — the
degrees of freedom of the field [1], equal to the Shannon-number estimate
`D_tx * D_rx / (lambda * z)` (~8.8 here; off-axis corrected ~7.4; measured knee
~12 including the transition band). Nothing classical is lost; the ceiling
appears as saturation in an SNR sweep.

**Every ingredient is standard EM practice.**
- The step-shaped singular spectrum of a finite-aperture propagation operator,
  and counting its relevant values, is the NDF framework [1], [7].
- The SVD mode picture (transmit pattern v_k -> receive sigma_k * u_k) is
  Miller's communication modes between volumes [2].
- Making the count depend on a stated accuracy is the *definition* used by
  Pierri & Moretta: the dimension of the data space is "the number of
  independent functions that allow representing the data with a given degree
  of accuracy" [3], [4], [5]. `N(eps)` makes that accuracy explicit instead of
  implicit.
- Robustness: the error weighting applied in `measure()` does not disturb the
  count — Pierri & Moretta prove a weight in the adjoint changes the singular
  values' dynamics but not the number of relevant ones [5]; verified
  numerically on this repo's operator.

**It predicts solver behavior, not just data fit.** The quadratic-inversion
literature shows local minima (trap points) vanish when the dimension of the
data space is sufficiently large relative to the unknowns, and proliferate when
it is small ([3] Sec. 1 and references therein). Low `N(eps)` therefore
predicts exactly the observed pathology: flat, multi-modal residual heatmaps in
miss scenarios, and reconstructions that fit the data while being wrong.

## Rejected alternatives

- **Energy fraction in the RX window** (`verify_scenario.py` already computes
  it): necessary but not sufficient — blind to mode structure (a focused spot
  can carry ~all energy but ~1 mode) and to SNR. It survives as a sanity check,
  not the metric.
- **Bare NDF / operator rank:** geometry-only; measured 12 vs 13 for hit vs
  miss — fails to discriminate the primary phenomenon of interest.
- **GS final residual:** measures data fit, not information; a miss produces
  deceptively low residuals by fitting near-zero data, and the value depends on
  the algorithm and its stopping rule.

## Measurement protocol

Prerequisite: complex AWGN injection into the synthesized RX field (the
simulated path is currently noiseless), seeded, with `sigma_n` referenced to
the peak field on the RX plane so hit and miss face the same absolute noise.
On the experimental path, estimate `sigma_n` from off-beam samples or repeated
captures.

1. **Beam-location sweep** at fixed SNR (hit -> partial -> miss): report
   `N(0.1)` per condition beside that condition's residual heatmap.
2. **SNR sweep** at fixed hit geometry (10-40 dB): `N(0.1)` should climb and
   then saturate at the NDF; heatmap contrast should sharpen and saturate at
   the same point.
3. Reduce each heatmap to one scalar (localization error of the best candidate,
   or basin contrast = median residual / best residual) and scatter it against
   `N(0.1)` across all conditions from both sweeps. The claim under test: both
   sweeps fall on one monotone curve — beam position and SNR are
   interchangeable through `N(eps)`.

## Known limitations

- It is a count: it moves in unit steps, so it suits comparison tables and
  heatmaps, not gradient-based design optimization (the EM literature operates
  the same way — sweep and tabulate).
- In the spectrum's transition band the count depends on `eps`; the convention
  is therefore fixed at `eps = 0.1` project-wide and printed with every value.
- `N(eps)` counts modes retrievable *individually* to accuracy eps; it does not
  weight how far above threshold each mode sits.

## References

[1] O. M. Bucci, G. Franceschetti, "On the degrees of freedom of scattered
    fields," IEEE Trans. Antennas Propag. 37(7):918-926, 1989.
[2] D. A. B. Miller, "Communicating with waves between volumes: evaluating
    orthogonal spatial channels and limits on coupling strengths," Appl. Opt.
    39(11):1681-1699, 2000.
[3] R. Pierri, R. Moretta, "An evaluation of the data space dimension in phase
    retrieval: results in Fresnel zone," URSI GASS 2021; arXiv:2202.02809.
[4] R. Pierri, R. Moretta, "An SVD approach for estimating the dimension of
    phaseless data on multiple arcs in Fresnel zone," Electronics 10(5):606,
    2021.
[5] R. Pierri, R. Moretta, "The dimension of phaseless near-field data by
    asymptotic investigation of the lifting operator," Electronics
    10(14):1658, 2021.
[6] G. Toraldo di Francia, "Degrees of freedom of an image," J. Opt. Soc. Am.
    59(7):799-804, 1969.
[7] O. M. Bucci, C. Gennarelli, C. Savarese, "Representation of
    electromagnetic fields over arbitrary surfaces by a finite and
    nonredundant number of samples," IEEE Trans. Antennas Propag.
    46(3):351-359, 1998.

Local copies in `~/Desktop/papers/`: [2] (`miller_2000_...`), [3], [4], [5]
(`pierri_moretta_...`), and a first-author review covering the material of [1]
and [7] (Bucci & Migliore, "Degrees of Freedom and Sampling Representation of
Electromagnetic Fields: Concepts and applications," IEEE Antennas Propag. Mag.;
`bucci_school_review_dof_and_sampling.pdf`). [1], [6], [7] themselves are
paywalled (IEEE/Optica). The multi-arc result in
[4] (Eq. 46: P sufficiently separated observation arcs multiply the data-space
dimension) is the template for extending `N(eps)` to the multi-frequency sweep:
stack the per-frequency operators and count the stacked spectrum's knee.
