# rice-bend — presentation outline (20 slides)

Recommendations for a talk on the novelty and importance of this codebase. Every number
below was verified against the code, docs, and `results/*/analysis.json` at commit
`be37c30` (Stage 13). Suggested visuals point at figures already in the repo.

---

## 1. Title

**rice-bend: blind localization of a 150 GHz transmitter by joint multi-frequency phase retrieval**

- Tagline: *bandwidth turns phase retrieval into localization*
- Visual: `results/scenario_caustic_hit_pm5/residual_surface_hc.png` (the basin peak standing over the true TX)

> **Notes:** One sentence framing: a receiver sees a 4 cm sliver of a millimeter-wave beam, and from that sliver we recover where the transmitter is — to 3 mm — by exploiting frequency diversity through a physically-correct source model.

## 2. The problem

- A hidden mm-wave transmitter (150 GHz, λ ≈ 2 mm); the only data is the complex field in a 40 mm receiver window
- Phase retrieval is classically ambiguous: profiles differing by 2π wraps or a global offset fit equally well
- Goal: recover the TX **location**, not just its phase profile

> **Notes:** Stress the inversion: most phase retrieval assumes you know where the aperture is. Here the aperture position is the unknown; the phase profile is just the vehicle.

## 3. The scene

- 132 mm phase plate at z = 0.3 m launches a curved "caustic" (Airy-type) beam along x(d) = 0.8d² − 0.065
- Beam bends through free space into a 40 mm RX window at z = 0 (FWHM 30 mm at the plane)
- Scene sampled at 0.25 mm over 0.6 m × 0.85 m (~2400 × 3400 cells); a steered plane-wave family exists too
- Visual: `results/scenario_caustic_hit_pm5/true_mgs_scene.png` or `mgs.png`

> **Notes:** The caustic beam is the interesting adversary — it curves, so naive "point back along the beam" fails. Five named scenario configs (hit/miss/larger/smaller) are each certified by a forward-only checker (`scripts/verify_scenario.py`).

## 4. Forward model: exact Rayleigh–Sommerfeld

- FFT convolution with the exact 2D kernel `(jkz/2r)·H₁⁽¹⁾(kr)` — no paraxial approximation
- Adjoint = conjugate kernel, verified bit-exact → analytic gradients come free
- Built-in sampling guard refuses aliased geometries, naming every offending frequency

> **Notes:** This slide buys credibility: the inverse problem sits on a defensible propagator. The sampling guard matters later — it's what gates which frequencies each grid candidate may use.

## 5. The solver: modified Gerchberg–Saxton

- Gradient descent on the aperture **phase** (amplitude fixed, support-restricted) with backtracking line search
- Loss: weighted L2 misfit of the propagated field at the RX plane
- Deterministic: fixed seed, complex64 pinned (documented as load-bearing — it flips rankings if changed)

> **Notes:** "Modified" = projected gradient descent rather than classic alternating projections. Keep this brief; it's standard machinery — the novelty is in what it solves for (slides 7–10).

## 6. Localization = 2,304 hypothesis tests

- Sweep a 48×48 grid of speculative TX planes (z, x_center); solve one phase retrieval per candidate
- The candidate whose reconstruction best explains the measurement wins (residual argmin)
- One shared seed across all candidates → residuals comparable; results byte-identical at any worker count
- Visual: `results/scenario_caustic_hit_pm5/residual_heatmap.png`

> **Notes:** The TX truth is used only to synthesize the measurement; everything right of that arrow treats it as unknown. Candidates too close to the RX plane keep only the frequencies they can validly model (per-frequency sampling gate) instead of being discarded.

## 7. Core idea 1: solve for a plate, not a phase mask

- One real unknown ψ(x) — the plate's **delay profile** — with each frequency's phase = (f/f_ref)·ψ(x)
- That is exactly how a physical plate disperses; conventional multi-λ work fits an independent mask per frequency
- One joint solve: single profile fitted against every frequency's measurement at once (loss = mean over channels)

> **Notes:** This is the central modeling decision. An independent-masks model can fit each frequency separately even at wrong locations; the shared plate cannot — the coupling is where the localization information comes from.

## 8. Why bandwidth breaks the 2π ambiguity

- Only the **true absolute delay** is consistent with every frequency at once — multi-wavelength interferometry, repurposed
- At the true TX a consistent profile exists and the loss collapses (~1e-5)
- At wrong locations no plate reconciles several frequencies of wrong geometry — losses stay pinned high

> **Notes:** The differential effect IS the signal: bandwidth doesn't just constrain the profile, it separates true from false locations. This is the physics slide worth lingering on.

## 9. The trap: the joint landscape is shattered

- Single-frequency retrieval is easy *because* it's degenerate — every 2π-wrapped variant is an acceptable minimum
- The joint constraint turns those equivalent minima into distinct basins, almost all wrong
- Measured: random-start joint descent stalls at loss 0.023 vs a verified floor of ~2e-4 — **100× too high**

> **Notes:** Honest tension: the same coupling that adds information destroys the benign optimization landscape. Escaping a wrong basin means moving whole stretches of ψ by 2π through high-loss territory, which per-point descent essentially never does.

## 10. Core idea 2: the multi-wavelength warm start

- **Solve** the channel nearest f_ref alone (easy degenerate landscape → recovers the *shape*)
- **Unwrap** its phase over the support → the unknown collapses to one scalar offset δ
- **Scan** δ over one synthetic-wavelength period (2π·f_ref/Δf_min, ~126 rad at ±5%) — essentially free, since propagation is linear
- **Descend** jointly from ψ₀ + δ*. Total cost: ~1/F extra runtime

> **Notes:** The elegant move: global search over the *only* global unknown, local descent for everything else. Each δ sample is elementwise work because adding δ just multiplies channel f's field by exp(i·(f/f_ref)·δ). Default in every solve; a strict no-op at one frequency.

## 11. Headline result

| Case | Comb | Error vs truth | Truth-cell rank |
| --- | --- | --- | --- |
| 1 frequency | 150 GHz | 8.7 mm | 10 / 2304 |
| ±5% | 142.5 / 150 / 157.5 GHz | **3.0 mm** | **1 / 2304** |
| ±10% | 135 / 150 / 165 GHz | 4.8 mm | 3 / 2304 |

> **Notes:** Three frequencies spanning ±5% cut error ~3× and moved the truth from 10th place to **first** out of 2,304 candidates. Be precise: these are 3-tone combs (the 10-tone 100–190 GHz comb belongs to other scenario variants). Claim rank + error improvement; do *not* claim the basin's global contrast sharpened — that metric doesn't support it.

## 12. Reading the residual landscape

- Flat heatmap saturates (over half the cells within 1% of max) — so residual is inverted into **relief**: best fit = tallest peak, standing over the true TX
- Every plot ships as a pair: fixed linear scale (comparable across runs) + adaptive log "high-contrast" twin
- 3D frequency-stack scatter shows per-frequency residual components of the joint solve
- Visuals: `residual_surface_hc.png`, `residual_scatter_3d.png` (pm5 run), plus the orbit mp4 if live

> **Notes:** These are instruments, not eye candy — the surface shares the heatmap's exact color norm, markers sit on occlusion-proof poles, and the residual-vs-distance scatter directly tests the premise that lower residual ⇒ closer to the truth.

## 13. Core idea 3: N_E(ε) — how much information does the receiver capture?

- SVD the **exact discrete TX→RX operator the solver uses** (columns = unit-source propagations; rows = the receiver's actual elements)
- Couple the real beam into the modes; count modes within ε = 0.1 of the channel's own capacity (σ₁·‖u₀‖)
- Noiseless variant of the EM "dimension of the data space" (Bucci, Miller, Pierri–Moretta) — no SNR parameter anywhere
- Computed into every run's `analysis.json`; multi-frequency runs sum per-frequency counts

> **Notes:** The novelty is the threshold: the literature thresholds against receiver noise; here the reference is what this geometry could deliver from this beam power, so the count falls exactly as captured energy falls. And it measures the discretized channel actually solved, not an idealized continuous one.

## 14. N_E discriminates where classical metrics fail

| Condition | classical NDF | energy in window | N(0.1) |
| --- | --- | --- | --- |
| Beam hits RX | 12 | 83.3% | **8** |
| Beam misses RX | 13 | 0.5% | **1** |

- Solver residual is actively deceptive: a miss fits near-zero data with a *low* residual
- Low N_E predicts the trap-point regime — flat, multi-modal residual maps

> **Notes:** The one-table argument for the metric. Bare operator rank can't tell hit from miss (12 vs 13) because geometry barely changes — the failure is in what the beam excites. Grounded in seven cited papers, not ad-hoc statistics.

## 15. Study 1 — bandwidth buys ranking, not accuracy

- 9-point sweep, 3-tone combs ±1%…±20% around 150 GHz (full 48×48 grid per point)
- Argmin error roughly flat at ~3–15 mm across the sweep; captured energy ~83% throughout
- Suspected bias floor: fixed 100 mm assumed candidate aperture vs the 132 mm true plate
- Visual: `results/study_frequency/study_error_vs_bandwidth.png`

> **Notes:** The honest finding: going multi-frequency is the step change (slide 11); *more* bandwidth beyond that doesn't monotonically buy accuracy — a systematic model-mismatch floor dominates. Presenting a negative result well builds trust.

## 16. Study 2 — walk the TX off the receiver: N_E < 8 is a lottery

- Rigid TX shifts 0 → 130 mm (window + caustic trajectory move together — a plate is one physical object)
- Captured energy 83% → 0.4%; N_E falls 14 → 8 → 2 → 0
- Below N_E ≈ 8, localization error decorrelates from the shift entirely — wrong answers with deceptively low residuals
- Visual: `results/study_tx_shift/study_metrics_vs_ndof.png`

> **Notes:** This is the metric doing real work: it marks the regime where the pipeline silently stops being trustworthy. At 115 mm shift (the miss scenario), the solver reports loss 1.6e-5 — as good as a hit — while capturing 0.46% of the beam.

## 17. Study 3 — a λ/2 receiver loses fidelity, not information

- Same sweep with receiver element pitch λ/2 instead of λ/20
- N_E stays ~15 and captured energy is identical — yet localization is 2–6× worse (16–60 mm vs 3–50 mm)
- Two failure axes cleanly separated: information capture vs measurement fidelity

> **Notes:** Design-relevant conclusion for real arrays: critical (λ/2) sampling keeps the information-theoretic channel intact; the accuracy loss lives in the measurement/pipeline, not in physics. Contrast with study 2, where energy loss destroys information regardless of fidelity.

## 18. Engineering rigor: bit-identical or it didn't happen

- No test suite — a **characterization harness**: 5 end-to-end runs digested to exact floats + content-addressed SHA-256 array hashes, diffed against machine-blessed baselines (~25 s)
- Serial vs 4-worker sweeps proven byte-identical; the 62-commit staged refactor was gated on the harness bit-for-bit
- Hankel-kernel hoist: 3.49× on the solver core (1.225 s → 0.351 s per candidate) with a bit-identical 800-point loss curve; the "obvious" batched-FFT alternative benchmarked *slower* and the negative result is recorded in-code

> **Notes:** Two real bugs found by measurement and memorialized in comments: the RX plane synthesized λ/8 (0.25 mm) from where it was fitted, and phase interpolation through the ±π branch cut corrupting the measurement 20.8% rel L2 — fixing it improved single-shot loss 216×. Nearly every commit co-authored with Claude (Opus 5 / Fable 5): a case study in disciplined AI-assisted research engineering.

## 19. Reproducibility as infrastructure

- Every run directory: arrays (`run.npz`), provenance (exact CLI + version), effective config *and* verbatim source YAML — round-trippable years later
- Ownership markers stop one tool clearing another's results; directories created only after the solve, so interrupts never destroy prior work
- `analysis.json` recomputed (never read back) on every replot — each replot is a free reproducibility check
- Studies are resumable, code-defined series; the experimental path (oscilloscope `.mat` captures) feeds the same solver

> **Notes:** Cheap slide, big signal: any figure in this talk can be regenerated from its run directory with one command. The bench bridge already exists — 13 former magic numbers now live in one validated `experimental:` config block.

## 20. Limits and what's next

- ~3–15 mm bias floor: fixed 100 mm candidate aperture vs 132 mm truth — aperture-width mismatch is the next thing to attack
- Coarse grid quantizes error at 6.4/13.8 mm cells (a refined-grid variant was tried and retired; Stage 13 tracks argmin + top-10 mean distance only)
- Noise-referenced N(ε) and validation on the real experimental captures are the open experimental arcs

> **Notes:** End honest and forward-looking: the pipeline, the information metric, and the bench bridge are in place; what remains is closing the model-mismatch floor and running the protocol against real data. Invite the "what about noise?" question — the decision record (`docs/information_metric.md`) already contains the plan.
