# Hierarchical Conjugate Generative Model — Status & Plan

*Working document for the Gaussian(Z) → Boltzmann(N) → Gaussian(X) MNIST effort.
All code lives in `examples/variational_mnist/` (playground; nothing has moved to
`src/goal/` yet). Updated 2026-07-12.*

## Goal

A **conjugate-by-design** hierarchical generative model — continuous top latent Z,
binary spiking population N, Gaussian observable X — that generates recognizable
MNIST digits, built entirely inside the variational-conjugation framework, and
stackable to arbitrary depth (each new layer = another Gaussian→Boltzmann unit).

## What works (validated, machine-precision unless noted)

| Component | Where | Evidence |
|---|---|---|
| Exact conjugate lower edge, non-overlapping conv (diagonal N) | `hierarchical.py::ConvBoltzmannHarmonium` | `max\|r_Y\| ~ 1e-16`; marginal == 2^n brute force |
| Exact conjugate lower edge, **overlapping** conv (chordal N) | `hierarchical.py::ConvChordalBoltzmannHarmonium` | same; node ρ = `Wᵀμ + ½diag(WᵀΣW)`, edge ρ = `(WᵀΣW)_ij` |
| Structured recognition (posterior = generative params + MLP-estimated conjugation correction ρ^X_Z) | `hierarchical.py::approximate_posterior_top` | ELBO decomposition pointwise-exact ~1e-14 |
| Reparameterized z-gradient (pathwise for Gaussian z, REINFORCE for discrete N only) | `elbo_at(reparam_z=True)` | value-identical estimator; needed *with* the conjugation penalty (alone it destabilizes) |
| Norm-preserving conjugation gradient on Θ_ZN (shape-only; forbids the Θ→0 exit) | `mnist_hierarchical.py::train` (`--norm-preserve`) | every run without it NaN'd; with it, Var[rZ]→0.1–0.3 while ‖Θ_ZN‖ grows |
| θ*_Z PD floor (finite ≠ valid; prior precision can leave the cone with all coords finite) | `bound_top_prior` | fixed the recurring NaN class |
| **Warm-start (the saddle-breaker)**: PCA-seed Θ_ZN node rows from posterior spike codes, optionally pretrain recognition MLP by regression | `--resume ckpt.npz --seed-top-pca [--seed-recog] [--seed-std]` | Z-sensitivity 0.0035 → 0.021; ELBO +33 nats at 49n |
| **Mixture top K→Z→N→X on the conv/chordal lower edge** (was plan item 1) | `hierarchical_mixture.py::build_conv_boltzmann_gaussian_mixture_hierarchy` (shares `conv_hierarchy_components`) | CV-ELBO == brute force to 0.0; `max\|r_Y\| = 8.9e-16` under the mixture top (`validate_hierarchical_mixture`) |
| **Checkpoint transfer base → mixture** (top/lower/recog blocks verbatim; stored θ*_Z inert) | `mnist_mixture.py::warm_start_from_base` | step-0 test ELBO == CW2's 738.2 exactly |
| **k-means/moment-matched mixture seeding** from aggregate-posterior z's | `mnist_mixture.py::seed_mixture_kmeans` | NMI 0.53 / purity 0.55 *at step 0*, before any mixture training |
| Mean-preserving `_project_valid` (width floor at prior's min precision eigenvalue; was plan item 2) | `hierarchical.py::_project_valid`, `_prior_precision_floor` | identity when inactive; ELBO decomposition unchanged; outcome-neutral (see below) |

**Root cause of the historical Z-collapse** (all other hypotheses eliminated by
experiment): a symmetric saddle — ∂ELBO/∂Θ_ZN ∝ E[(posterior − prior-conditional
spikes) ⊗ s_Z], which is zero when q(z|x) is uninformative, and vice versa. Not
gradient noise, not the penalty shrink-exit, not capacity. Layerwise seeding breaks it.

## Champion models

**CW2** (single-Gaussian top): `7×7` lattice, stride 4, **kernel 6×4** (vertical
overlap; treewidth-1 chordal middle, zero fill-in), td16, `--reparam-z
--lambda-z 1.0 --norm-preserve`, 25k cold + 10k node-only-PCA warm. **ELBO test
738.2, MSE 0.036, Var[rY]=0, Var[rZ]=0.27, diversity 0.065, Z-sens 0.011.**
`results/variational_mnist/znx/mnist_hierarchical_conv_7x7s4k6_..._rp_np_wsPR_st10000.{npz,png}`.

**MX1** (mixture top, current champion): CW2 warm start + K=10 k-means seed, 10k
steps, same recipe + entropy reg 0.3. **ELBO test 738.7 (beats CW2), Var[rY]=0,
Var[rZ]=0.38, NMI 0.53, purity 0.56, all 10 components alive (p(k)∈[0.09,0.12]),
‖Θ_ZN‖ 58→101 then plateau.**
`results/variational_mnist/znx/mnist_mixture_conv_7x7s4k6x4_..._wsKM_st10000.{npz,png}`.

## Results of the 2026-07-09 round (former plan items 1 and 2)

### Item 1 (mixture top): architecture works; clustering real but coarse; gate partially met

- The per-cluster regression figure is decisively fixed: rows are distinct sample
  distributions (strokes vs loops), not the historical 10 identical blobs.
- But NMI stayed at its *seeded* value (0.531 → 0.539 over 10k steps): joint
  training preserved the clustering the k-means seed provided rather than
  sharpening it. Cluster purity 0.55 ≈ the ceiling of k-means on the encoder's
  q(z|x) means — **the clustering ceiling is the encoder's, not the mixture's**.
- Individual ancestral samples remain fragmented (see decomposition below).

### Item 2 (mean-preserving projection): correct, kept, but outcome-neutral

CW3 = exact CW2 protocol retrained under the fixed projection
(`znx_fixproj/`): ELBO test 738.1, MSE 0.0362, diversity 0.063, Z-sens 0.011 —
statistically identical to CW2. Direct measurement (200 test digits, each model
under its own projection):

| | CW2 (old proj) | CW3 (fixed) |
|---|---|---|
| posterior dirs wider than prior (prior-whitened) | 26% | 31% |
| max width ratio (std) | 1.29× | 1.25× |
| decode MSE, posterior-mean z | 0.0491 | 0.0497 |
| decode MSE, sampled z | 0.0527 | 0.0530 |

**Metric correction:** the old "std to 5.5" was raw std along the prior's loosest
axes, not width relative to the prior. Whitened correctly, the recognition-width
defect is mild (≤1.3× prior) and sampling-z damage is ~7% extra MSE. Item 2 was
not where the generation gap lives. (The implemented floor is a scalar at the
prior's loosest axis; a per-direction floor — clamp Λ_q ⪰ Λ_p in prior-whitened
space — would enforce "never wider than the prior" exactly, if ever needed.)

## Revised diagnosis of the generation gap

Noise-source decomposition on MX1
(`results/variational_mnist/analysis/mx1_cluster_mean_decode.png`): decode each
cluster k three ways — (component mean z, E[N|z]), (z ~ p(z|k), E[N|z]),
(z ~ p(z|k), N sampled).

1. Component means decode to soft but **recognizable digit prototypes** (clear 0,
   9s, 8/0 loops). Prior structure is now right.
2. Within-cluster z-noise is **harmless** — row 2 ≈ row 1.
3. **Sampling N destroys both crispness and identity** — row 3 is where the mush
   comes from.

So with the prior fixed by the mixture, the dominant residual issue is the
**conditional entropy H(N|z)**: z shifts only the 49 node biases (node-only
coupling, by design — see failure modes), and p(N|z) remains too stochastic for
a single z to pin down a coherent digit. Secondary: cluster prototypes blend
digit classes because the encoder's z-space mixes them (purity 0.55).

## 2026-07-12 round: exact-N marginalization — plain gradient descent works

**The estimator was the hack generator.** For conv lower edges, ``r_Y == 0``
*pointwise*, so the learning signal is y-independent and the spike layer
integrates out of the ELBO exactly: ``L(x) = c(x) + E_{q(z|x)}[r*_Z − r_inner]``
with pathwise z-gradients only — no REINFORCE anywhere
(`hierarchical.py::marginal_elbo_at`, validated: value == sampled-y assembly at
shared z to 0.0, pathwise grad == fixed-key finite differences to ~1e-9, both
tops). Enabled via `--marginal-y` in both drivers.

**M1** = CW2 protocol, marginal estimator, **no norm-preserve, no conjugation
penalty (λ=0)**: ELBO test **738.09**, MSE 0.0364, Var[rZ] self-declined 2.2→0.7
under pure ELBO pressure, diversity 0.061, Z-sens 0.0081 (z alive), **2× faster**
(250s vs 505s / 10k steps). `znx_marginal/`. Conclusion: the norm-preserving
surgery and the conjugation-penalty ramp were compensating for score-function
gradient noise, not holding up the model. Retired for conv models. The
`_project_valid` width floor is also reverted to a bare mean-preserving PD guard
(floor 1e-3). Remaining "stability" code: Ψ variance bounds + top-prior PD clamp
+ mixture bounds — the caller-invoked-regularization pattern, kept.

**Blockiness diagnosis**: kernel 6×4 has *zero horizontal overlap* → hard seams
every 4 columns AND a vertical-edges-only coupling graph (horizontally adjacent
nodes conditionally independent given z). Kernel 6×6 fixes both (decoder
blending + king-graph couplings), exactly conjugate by the same closed form, at
treewidth 11 (max clique 12 probed; 42 induced vertical edges survive, 248
fill-in edges added). The marginal estimator removes the JT *sampling* machinery
that OOM'd before; eval trimmed via `--eval-n`.

**Full de-hacked pipeline reproduced** (all runs in `znx_marginal/`, marginal
estimator, no surgery):

| run | recipe | ELBO test | notes |
|---|---|---|---|
| M1 | single Gaussian, λ=0 | 738.09 | Var[rZ] self-declines to 0.7; Z-sens 0.008 |
| M2 | single Gaussian, λ_z=1 | 737.97 | Var[rZ] 0.2; **Z-sens 0.0108 == CW2's 0.011** |
| MX2 | mixture, λ=0, warm M1 | **738.76** | NMI 0.46 (seed 0.482 — inherits M1's z-space) |
| MX3 | mixture, λ_z=1, warm M1 | 738.49 | NMI 0.47; penalty needs no surgery |
| MX4 | mixture, λ_z=1, warm M2 | 738.47 | **seed NMI 0.525 == MX1's** (λ_z at *encoder* time shapes class-separating z) |

Mixture-phase eval NMI fluctuates ±0.04 between checkpoints, so all mixture
variants are at effective parity with MX1 (738.7 / 0.53). λ_z verdict: it is
in-framework and does real work *during encoder training* (stronger Θ_ZN
coupling → better seed); during the mixture phase it mainly holds Var[rZ] down.

**Cold starts do NOT work noise-free (K66/K66-probe finding).** The y-REINFORCE
term was zero-mean *pure noise* on the decoder gradient for conv models (the
signal is y-independent) — de facto annealing that kicked cold decoders off the
symmetric saddle. With the exact estimator a cold ``Theta_XY`` sits at exactly
zero gradient and weight decay pulls it to 0: K66 (6×6 cold, 25k) and a 6×4 cold
probe both decayed ``|Theta_XY|`` monotonically and never recruited. This is a
*feature* of the honest gradient story: initialization must do the work.
Answer: **geometry lifting** (`mnist_hierarchical.py::lift_conv_checkpoint`,
`--lift-from CKPT --lift-kernel H W`) — a smaller conv kernel embeds exactly in
a larger one (taps are center-aligned; W_lift == W_base to 0.0), conjugation is
closed-form in W so the lifted lower edge is exact from step 0; node-level top
structure transfers, fill-in edges start 0, MLP re-seeded by regression.

**K66c (6×6 lifted from M2 + seed-recog, marginal, λ_z=1, 10k)** — split verdict:

- **Texture gate PASSED**: reconstructions and samples are smooth contiguous
  strokes; the 4-pixel vertical banding is gone. Best reconstruction of the
  whole effort (MSE 0.0345), sharpest noise floor (Ψ max 0.129), survived a
  violent recognition transient (fresh MLP → Var[rZ] 3670 at step 500, fully
  recovered by 2500 — the lift's generative init is that solid).
- **Z-liveness gate FAILED**: Z-sens 0.0010, diversity 0.0043 — z is dead, and
  `Var[rZ] = 0.007` with `‖Θ_ZN‖ = 23.7` intact shows *how*: the conjugation
  penalty found a new trivial exit. With 290 lateral couplings available, θ_Y's
  internal correlations absorb the inter-patch structure that z previously
  carried (at 6×4, z was the ONLY horizontal channel), leaving z's node-bias
  modulation impotent. Capacity competition between the N-prior graph and the
  top latent — "‖Θ_ZN‖ alone is misleading", coupling-saturation edition.
- ELBO test 721.6 < M2's 738 (z contributes nothing + young recognition).

**Stacking machinery landed (`mnist_stack.py`), first result mixed.** Unit 2 =
the same `VariationalHierarchical` class on unit-1's 16-dim code space: dense
16×12 decoder → complete coupling graph on N2 → single-clique JT (exact
enumeration over 2^12) → **exactly conjugate lower edge by the existing closed
form** (`ConvChordalBoltzmannHarmonium` is generic in the interaction map;
r_Y == 0 verified). PCA decoder seeding breaks the cold-start saddle;
`MH.train` reused verbatim on codes (~0.07s/step — a 10k run is ~4 min).
**S1** (n2=12, z2=8, marginal, λ_z=1, 10k on M2 codes): trains cleanly (code
ELBO −39.3, Var[rZ2] 0.23) **but its z1-marginal underfits** — moment/coverage
metrics vs the aggregate posterior are *worse* than unit-1's own Gaussian prior
(|dcov| 24 vs 18; code→sample distance 4.9 vs 3.1). Suspected causes, in order:
Ψ2 pinned at its 2.0 cap (unstructured noise dominates the learned Boltzmann
structure — codes have per-dim std ~0.5–2, cap should be ~0.25–0.5), fresh-unit
undertraining, and no layerwise seeding of the N2 *prior* (the k-means cluster
structure should seed θ_Y2 the way it seeds the mixture — each mode ≈ a spike
pattern). `results/variational_mnist/stack/`.

**S2 (Ψ cap 0.3, 20k steps): the stack GATE PASSED.** Code ELBO −14.9 (S1:
−39), ‖Θ_XY2‖ 47 (real Boltzmann structure, noise off the cap). The stack's
z1-marginal now beats unit-1's Gaussian prior on every measure vs the aggregate
posterior: |dcov| **5.4 vs 17.9**, code→sample coverage **2.52 vs 3.12**
(self-baseline 1.68), sample→code precision **2.70 vs 4.01**. Full-stack
samples (z2→n2→z1→n1→x) are visibly more digit-like than unit-1 prior samples —
the 2^12-state N2 prior supplies the multimodal global structure the unimodal
Gaussian couldn't. Caveat, same shape as K66c: Var[rZ2] = 0.000 with
‖Θ_ZN2‖ = 1.5 — z2 is likely near-inert; at the TOP of the stack that is
acceptable (N2's Boltzmann is the working top prior), but stacking a *third*
unit would face the same absorption question.

## Plan (priority order, revised)

### 0. Stack won its gate — integrate and iterate
Candidates, all cheap on the unit-2 side: k-means-seed the N2 prior (spike
patterns per cluster), n2=14–16, whiten codes; then drop unit 2 onto the 6×6
line (K66c) — the stack is exactly the "richer top" that item 1 calls for, and
unit 2 trains in minutes against any frozen encoder. Cluster-conditional
sampling (which N2 states = which digits?) is the analysis to run next.

### 1. Resolve the 6×6 capacity competition (z vs lateral couplings)
The clean texture and the live z currently come from different runs. Candidate
levers, roughly in order of principle: (a) **stack** — a richer top (deeper
Gaussian→Boltzmann unit or the mixture) that predicts *global* structure the
lateral edges cannot (lateral edges are local; class identity is not); (b)
extend K66c training — z-recruitment through a near-zero-gradient region is the
known slow phase transition, but Var[rZ]≈0 means the penalty no longer helps;
(c) constrain θ_Y coupling magnitude (capacity control on the N-prior) — works
but is a new knob of exactly the kind we just removed. The honest bet is (a).

### 2. Drive H(N|z) down / extend MX1 training
‖Θ_ZN‖ was still growing (58→101) with the ELBO still climbing at 10k, and a
stronger Θ_ZN is exactly a lower-entropy p(N|z). Cheapest lever, zero new code:
resume MX1 for +15k (mirror the diagonal champion's 25k trajectory). Watch:
sampled-N per-cluster rows sharpen (the gate), NMI move off its seed value,
Var[rZ] stay ≤ ~0.5. Consider annealing the entropy reg (0.3 → 0.1 after
warm-up) so components can specialize; keep the anti-collapse floor MIN_PROB.

### 2. Stack a second unit (the grant story) — now the main structural lever
Z of the working unit becomes the observable of a second Gaussian→Boltzmann
layer. The **layerwise protocol is validated twice** (PCA warm start for CW2;
lossless transfer + k-means seed for MX1): train layer 1 → freeze/seed → train
layer 2 on layer-1 codes. A deeper top gives p(N|z-stack) compositional
structure that neither a single Gaussian nor a 10-component mixture provides —
this attacks both residual issues (H(N|z) and class mixing in z) at once.

### 3. Decoder capacity: channels C=2–3 on the chordal conv
Richer p(x|N) and finer codes; treewidth scales ~×C so C=2–3 is the exact-JT
limit. Subsumes the old full-2D-overlap item: same memory work
(batch 16–32, trimmed eval, `XLA_PYTHON_CLIENT_PREALLOCATE=false`, probe clique
sizes first via `ChordalBoltzmann.from_edges(...).junction_tree`).

### 4. Full-2D overlap (kernel 6×6, treewidth ~11) — demoted
Removes horizontal seams; cosmetic relative to H(N|z). Blocked on the same
memory work as (3). Keep as backlog.

## Runbook

- Train (single-Gaussian): `uv run python -m examples.variational_mnist.mnist_hierarchical
  --middle conv --conv-prior {diagonal|chordal} --conv-in H W --conv-stride H W
  --conv-kernel H W --top-dim D --reparam-z --lambda-z 1.0 --norm-preserve
  [--resume CKPT --seed-top-pca --seed-recog] --steps N --outdir znx`
- Train (mixture top): `uv run python -m examples.variational_mnist.mnist_mixture
  --middle conv --conv-prior chordal --conv-kernel 6 4 --top-dim 16 --n-clusters 10
  --reparam-z --lambda-z 1.0 --norm-preserve --resume BASE_CKPT.npz --steps N
  --outdir znx` (warm start transfers base blocks verbatim + k-means-seeds the
  mixture; NMI/purity vs held-out labels printed each log point)
- Validate: `uv run python -m examples.variational_mnist.validate_hierarchical_mixture`
  (includes the conv/chordal case: CV-ELBO identity + r_Y == 0)
- Analyze (Z-liveness + traversals):
  `uv run python -m examples.variational_mnist.analyze_znx IH IW SH SW KH KW C TD NPZ [prior]`
- Metrics & gates: Z-sensitivity (dead ≈ 0.002–0.004; alive > 0.02), generation
  diversity (> 0.05), MSE (≤ 0.05 beats mean predictor), Var[rY] (must be 0.00 —
  exactness check), Var[rZ] (soft; ~0.1–0.5 healthy), NMI (seed gives 0.53; the
  encoder is the current ceiling). **‖Θ_ZN‖ alone is misleading.**
- Training regime: ≥ 10k steps (decoder recruitment is a phase transition: ~2.5k at
  49n-diagonal, ~20k at 49n-chordal — *chordal looks dead at 10k and isn't*),
  batch 128, lr 1e-3, GPU. Long runs: launch as tracked background commands, not
  detached drivers. Warn: run tags collide across configs that differ only in
  kernel width etc. — use distinct `--outdir` per experiment family so npz
  checkpoints can't be silently overwritten.

## Known failure modes (don't rediscover)

- Warm-start does **not** transfer to fine codes (98n degraded, 196n broke at every
  seed-std/lr). Coarse-lattice (49n) codes only.
- PCA seeding must be **node-only** — seeding Θ_ZN edge rows lets z drive couplings
  and destabilizes (‖Θ_ZN‖→61, ELBO −80). Top latents modulate the layer below
  through first-order channels only (same principle as `BoltzmannNodeEmbedding`).
- Parameter transplant across geometries preserves the decoder but not recognition
  alignment; cold-train-the-geometry + seed beats transplant.
- Reparam-z without the conjugation penalty → Var[rZ] runaway → NaN.
- min_eig-style clamps that hold the natural location fixed silently move means
  (fixed in `_project_valid`; fix is outcome-neutral but principled — see above).
- Raw posterior std is a misleading width metric — always compare in
  prior-whitened space (the "std to 5.5" scare was ≤1.3× prior).
- **Cold starts are dead under the exact-N estimator** — the y-REINFORCE term
  was zero-mean annealing noise on the decoder gradient; without it, cold
  ``Theta_XY`` has exactly zero gradient and weight decay kills it (K66, 6×4
  probe). Initialize (lift/seed); don't re-inject noise.
- **Geometry lifting must carry the trained Theta_ZN edge rows** — dropping
  them (misreading the node-only *seeding* rule as a transfer rule) changes
  p(y|z) and invalidates rho0_Z (Var[rZ] 0.2 → 227). With edge rows mapped the
  lifted joint equals the base exactly (r*_Z to 7e-15).
- **Rich lateral N-couplings can absorb z** (K66c): at kernel 6×6 the
  conjugation penalty is satisfiable by letting theta_Y couplings carry the
  inter-patch structure, leaving z inert — Var[rZ] ~ 0 AND ‖Theta_ZN‖ intact
  AND z dead can co-occur. Check Z-sensitivity, not the penalty or the norm.
- Occasional huge-negative *train-eval* ELBO outliers (e.g. −13.7k for one
  64-image eval batch) under the score-form estimator with a mixture top —
  monitoring noise, not divergence; test-eval and training were unaffected.

## Library candidates awaiting review (do NOT move to `src/` yet)

`LatticeConvolution`, `EmbeddedLinearMap` (`lattice_convolution.py`);
`ConvBoltzmannHarmonium`, `ConvChordalBoltzmannHarmonium`, `BoltzmannNodeEmbedding`,
`conv_hierarchy_components` (`hierarchical.py`); the closed-form conv conjugation
formulas; `VariationalHierarchicalMixture` + conv factory + k-means seeding
(`hierarchical_mixture.py`, `mnist_mixture.py`); possibly the
norm-preserving-gradient and layerwise-seeding training utilities.
