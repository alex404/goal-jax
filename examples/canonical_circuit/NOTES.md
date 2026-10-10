# Canonical circuit: status (2026-10-10)

The three-layer circuit Z → N → X from the DFG grant (WP2), on the variational library
(`src/goal/geometry/exponential_family/variational.py`). Its purpose is to establish the framework on a case
where everything can be checked exactly: the circuit learns hierarchical structure, and the penalized ELBO
of `scratch/variational-conjugation/article.tex` reaches the performance of brute-force maximum likelihood.
If the theory fails here, the theory changes, not the example.

## Model (`model.py`)

| block | contents |
|---|---|
| θ_X | Gaussian location and precision of x (16 pixels) |
| θ_N | biases and couplings of 10 binary neurons, dense in the harmonium |
| θ_Z | held at the standard normal (one-dimensional z: an affine gauge that Θ_NZ absorbs) |
| Θ_XN | location of x × activities of n |
| Θ_NZ | activities of n × (z, z²), one shared z² coefficient (bell-shaped tuning curves, shared width) |

Two levels:

- X–N (`CanonicalCircuit`): ρ_N is the exact closed form of `BoltzmannLGM`, quadratic in n with couplings
  G = Θ_XNᵀ Σ_X Θ_XN, so the level is exactly conjugate (r_N ≡ 0).
- N–Z (`PopulationCodeLevel`): ρ_Z is an MLP of the likelihood (biases, couplings, Θ_NZ), last layer
  initialized at zero. Its precision output passes through a softplus, offset so that ρ_Z = 0 at a zero
  output, so θ_Z + ρ_Z always has positive precision: a restriction of the family of conjugation
  functions to those giving valid distributions, not a clamp. (In float32 the precision rounds to zero
  only for a raw output below about −16, a variance of order 1e7.)

The free parameters of the neurons are the generative biases and couplings θ*_N = θ_N + ρ_N, with the
couplings on a graph E: `independent` (none), `chain`, or `full`. The harmonium's θ_N = θ*_N − ρ_N is
dense. Generative and recognition models are the same family evaluated at θ*_N and at θ_N + Θ_XNᵀ x; the
generative couplings lie on E, and the recognition couplings are dense (x shifts only the biases, so the
couplings −G off E remain: explaining away between overlapping readouts). Without z, the harmonium's
marginal over the neurons is the Boltzmann machine θ*_N on E, so with a chain, long-range correlations can
only come from z.

The dense neurons are sampled exactly by enumeration (`EnumeratedBoltzmann`), which limits the example to
about 12 neurons. At scale, p̃ could be sampled by a junction tree on E, but the recognition model needs the
graph of G.

The readout is initialized with isotropic noise (s.d. 0.1). Initializing it from the data covariance
explains the correlations of the data as noise, and every objective then finds a tangled code with little
use of z.

## Training (`run.py`)

A grid of conditions: E ∈ {independent, chain, full} × {with z, without z} × {brute-force MLE, penalized
ELBO} × λ × seeds.

- Brute-force MLE: the exact log-likelihood of the harmonium, by enumeration of the neurons.
- Penalized ELBO: `mean_elbo − λ(Σ conjugation_residual_variances + Σ mean_recognition_residual_variances)`,
  all library estimators with score-function gradients, λ = 0.1 fixed (Algorithm 1 as written).
- Without z: Θ_NZ and ρ_Z are held at zero; both levels are then exact and the ELBO equals log p(x).
- Both fits: 40k steps, batch 128, Adam at a constant 2e-2, 16 samples. A λ ramp, a warmup and a cosine
  decay are options (`--ramp`, `--warmup`, `--decay`). Updates with a non-finite gradient are skipped and
  counted.

Each run is cached in `results/canonical_circuit/runs/<key>/`; the key holds a hash of the settings its
objective depends on, so changing ELBO settings does not rerun the MLE baselines. `run.py` collects the
runs of the current settings into `analysis.json`; `plot.py` makes `plot.png`. Flags select a subset of the
grid (`--couplings`, `--z`, `--fits`, `--lams`, `--seeds`) and the settings.

## Validation (`validate.py`, all pass, about 1 min on CPU)

4 neurons, one-dimensional x, each graph E, against brute-force grids: the likelihood equals
`BoltzmannLGM`'s; r_N = 0; the generative couplings off E are zero; the sampler matches the enumerated
means; exact log p(x), log p̃(x), the ELBO and the divergence of q agree to about 1e-14; the ELBO
estimator and its gradient are unbiased (largest z-score 2.3).

## Grid results (defaults: Algorithm 1, λ = 0.1, constant learning rate; 3 seeds; `plot.png`)

Test log p(x) per seed (mean), the mass of the harmonium off the curve, and the fraction of the variance of
z explained by x (exact posterior). No run skipped a step or stopped.

| graph | fit | log p(x) | log p(x) − ELBO | off the curve | use of z |
|---|---|---|---|---|---|
| independent | MLE, with z | 14.85 / 14.81 / 14.96 (14.87) | – | 0.25–0.39 | 0.97–0.99 |
| independent | ELBO, with z | 14.23 / 13.83 / 14.89 (14.31) | 0.11–0.17 | 0.49–0.85 | 0.93–0.97 |
| independent | without z | 12.46 / 12.53 / 12.29 (12.42) | – | 0.96–0.97 | 0 |
| chain | MLE, with z | 15.18 / 15.36 / 15.19 (15.24) | – | 0.03–0.24 | 0.98–0.99 |
| chain | ELBO, with z | 14.92 / 15.13 / 14.95 (15.00) | 0.15–0.27 | 0.42–0.46 | 0.93–0.96 |
| chain | without z | 13.56 / 13.59 / 14.19 (13.78) | – | 0.77–0.83 | 0 |
| full | MLE, with z | 15.65 / 15.50 / 15.19 (15.45) | – | 0.00–0.05 | 0.93–0.98 |
| full | ELBO, with z | 15.43 / 15.42 / 15.36 (15.40) | 0.00–0.01 | 0.00–0.15 | 0 |
| full | without z | 15.26 / 15.36 / 15.19 (15.27) | – | 0.01–0.42 | 0 |

- The hierarchy helps in proportion to what the couplings cannot do: with z over without, 2.4 (MLE) and
  1.9 (ELBO) nats with no couplings, 1.5 and 1.2 with a chain, 0.2 and 0.1 with all pairs.
- The ELBO trails brute-force MLE by 0.25 nats with a chain, 0.05 with all pairs, and 0.55 without
  couplings (one weak seed at 13.83), with the bounds within 0.27 nats of log p(x).
- With all pairs, brute-force MLE still uses z for a small gain, while the penalized ELBO drops it
  entirely: an unneeded z costs conjugation, and the free couplings substitute for it.
- Samples are worse under the ELBO than under brute-force MLE (chain: 0.42–0.46 of the mass off the curve
  against 0.03–0.24), but far better than without z (0.77–0.83).
- Both fits encode position in z without a monotone code (z folds or jumps); E[z | x] determines t
  (leave-one-out nearest-neighbour R² 0.99 for the ELBO and 0.98 for MLE, chain, seed 0).
- Without z, ELBO training equals brute-force training (log p(x) 12.3639 for both in the one run tried),
  so that cell is not run.
- Wall clock, 10 neurons: brute-force MLE about 3 ms per step, the ELBO about 14 ms (shared GPU,
  evaluation included). At this size enumeration is one batched product; the ELBO also samples n by
  enumeration, and only becomes cheaper once n is sampled without it.

## Search over the training settings (chain, with z, 3 seeds, test log p(x))

| setting | seeds 0 / 1 / 2 | mean |
|---|---|---|
| λ = 0.3 ramped over the first quarter, warmup, cosine decay (old default) | 14.81 / 14.54 / 14.88 | 14.74 |
| fixed λ = 0.3, warmup, cosine decay | 14.29 / 14.28 / 14.92 | 14.50 |
| fixed λ = 0.1, warmup, cosine decay | 14.69 / 14.54 / 14.91 | 14.71 |
| learning rate 1e-2 | 14.33 / 14.23 / 14.66 | 14.41 |
| 64 samples | 14.68 / 14.40 / 15.03 | 14.70 |
| MLP hidden 64 | 14.64 / 15.10 / 14.57 | 14.77 |
| constant learning rate | 14.85 / 14.90 / 15.02 | 14.92 |
| **fixed λ = 0.1, constant learning rate, no warmup (Algorithm 1 as written)** | 14.92 / 15.13 / 14.95 | **15.00** |
| fixed λ = 0.3, constant learning rate, no warmup | 14.99 / 14.34 / 14.63 | 14.65 |
| fixed λ = 0.1, constant learning rate, warmup | 14.86 / 11.38 / … | – |
| brute-force MLE (warmup, cosine decay) | 14.45 / 15.26 / 15.05 | 14.92 |

The plain algorithm is the best setting found, and is now the default. A cosine decay of the learning rate
hurts; the ramp of λ only helps at λ = 0.3. One run collapsed late (λ = 0.1 with warmup, seed 1: 14.65 →
11.43 at step 30k): the MLP of ρ_Z saturated (raw outputs +11.5 and −25.7 against about ±1 in healthy runs,
prior precision of z 52 against 2–3) and did not recover. Its inputs, the likelihood parameters, reach
magnitudes of about 75; scaling them, or a smaller learning rate for ρ_Z, are candidate remedies. NaN: two
runs of the search stopped (wider MLP, seed 1; constant learning rate, seed 1); neither reproduced on a
full rerun or a step-by-step replay, and no precision approached zero. `run.py` now skips updates with a
non-finite gradient and counts them.

## Findings so far (exploration of 2026-10-10, before the restructure)

1. A constant ρ_Z cannot represent z: q(z | x) then has natural parameters θ_Z + φ, independent of x, and
   the penalty is only satisfied by decoupling z. Removed.
2. With the chain truncation of ρ_N (the earlier parameterization), the harmonium put mass on states that
   p̃ never samples, and the variance penalties cannot see those states; log p(x) and log p̃(x) disagreed by
   0.5–5 nats. The exact ρ_N above removes this.
3. With a chain, 10 neurons, λ = 0.3: penalized ELBO with z reached test log p(x) 14.6–15.3 (5 seeds),
   every run above the circuit without z (13.2–13.8), with |log p(x) − ELBO| ≤ 0.21 and z encoding the
   bump position (|Spearman| 0.64–1.00). Brute-force MLE with z reached 15.26 (one run). Mass off the curve:
   0.23–0.30 with z, 0.73–0.82 without, 0.02 for brute-force MLE.
4. λ trades conjugation for samples: λ = 0.03 gives mass off the curve 0.09–0.18 and residual variance
   0.55–0.99; λ = 1 gives 0.20–0.57 and 0.035–0.045. With λ = 0, p̃ fits (15.1) and the harmonium does not
   (−5.7).
5. The learned ρ_Z is within 10% of its least-squares floor (measured by regression, as a diagnostic) in
   every run: the remaining residual is the harmonium's.
6. Negative: staged pretraining (brute-force MLE, then ρ_Z alone, then the ELBO) ends below the ELBO from
   scratch; the importance-sampled MLE of the article (its gradient agrees with the exact one, cosine
   ≥ 0.998) collapses z at the full budget.
7. NaN: the precision of the prior over z, θ_Z + ρ_Z at the generative biases, left the domain. At the
   first Adam step this is the zero-initialized last layer moving by about the learning rate (the warmup
   removed it); later in training it is one overshooting step (chain, seed 0: 2.07, 1.80, 1.60, then
   −0.11 at step 1589, located by a step-by-step replay on the GPU; the replays of the night before did
   not reproduce their NaNs). The softplus head above removes it: the same replay runs past 4000 steps.
8. 12 neurons (chain, λ = 0.3): z used in 2 of 3 seeds (15.23 and 14.94 against 13.20 and 13.86 without z).

## Next

- The ELBO's samples trail brute-force MLE's; λ trades sample quality for conjugation (earlier exploration:
  λ = 0.03 gave 0.09–0.18 off the curve).
- The MLP of ρ_Z can saturate (one late collapse); try scaling its inputs.
- Scaling past enumeration: a junction-tree sampler for p̃ on E, and a sampler on the graph of G for the
  recognition model.
