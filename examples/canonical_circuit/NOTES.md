# Canonical circuit: status and handover (2026-10-09)

The three-layer circuit Z → N → X from the DFG grant (WP2), rebuilt on the reviewed variational library
(`src/goal/geometry/exponential_family/variational.py`). Goal: test the standard theory of
`scratch/variational-conjugation/article.tex` and nothing else. If the theory fails here, the theory
changes, not the example. Approved plan: `~/.claude-work/plans/expressive-sprouting-teacup.md`.

## Model (`model.py`)

Notation: Z → N → X, with parameters θ_X, θ_N, θ_Z, Θ_XN, Θ_NZ.

| block | contents |
|---|---|
| θ_X | Gaussian location and precision of x (d_X = 16 or 8 pixels) |
| θ_N | Boltzmann biases and couplings on a graph E (chain or full), d_N = 10 |
| θ_Z | held at the standard normal (one-dimensional z: an affine gauge that Θ_NZ absorbs) |
| Θ_XN | location of x × activities of n |
| Θ_NZ | activities of n × (z, z²), with one shared z² coefficient (bell-shaped tuning curves, shared width) |

Two library levels, nothing else:

- `CanonicalCircuit` (X–N, a `VariationalConjugated`). ρ_N has no parameters: it is the closed form of
  `BoltzmannLGM` with the couplings off E dropped. For full E the level is exact. For a chain, its
  residual is r_N(n) = −Σ_{i<j, (i,j)∉E} G_ij nᵢnⱼ, with G = Θ_XNᵀ Σ_X Θ_XN.
- `PopulationCodeLevel` (N–Z, a `DifferentiableVariationalConjugated` over `full_normal(1)`). ρ_Z is a
  stored constant, or an MLP (hidden (32,), tanh) of the full likelihood (N-bias, Θ_NZ).

Tied parameters: `tie(u)` is linear and every block is in natural coordinates. There are no constraints,
floors or clamps. `initialize_tied` tiles the tuning curves over [−2, 2] with precision 1/spacing², sets the
prior peak logit to −1 (θ_N = target − ρ_N at the initial likelihood), and uses a quarter of the data
covariance and standard normal Θ_XN for the readout. ρ_Z starts at 0; the MLP's last layer starts at 0.

The diagnostics are used only to measure. They enumerate the 2^d_N states and use Gauss–Hermite
quadrature over z: exact log p(x) of the graphical harmonium, log p̃(x) and the ELBO of the variational
model, and KL(q(n,z|x) ‖ p(n,z|x)). The KL reduces to a divergence over z, because both share p(n | z, β).

## Training (`run.py`)

- Objective: `mean_elbo − λ(Σ conjugation_residual_variances + Σ mean_recognition_residual_variances)`. All
  three are library estimators with score-function gradients.
- λ ramps linearly over the first half of training; sweep (0, 0.3, 1, 3), 3 seeds.
- Steps: 50 chunks × 100 steps, batch 128, Adam 2e-2.
- Samples: 16 per estimate, 64 for evaluation, 40 quadrature nodes.
- A run whose loss becomes NaN is stopped and recorded (`stopped`). Its last finite parameters are
  measured.
- `min_prc` records the smallest precision of z over p(z|n), the prior and the recognition model on the
  test set. A negative value means the run left the domain. The policy is to run unconstrained and add
  checks only if runs leave the domain; goal-apps is the reference for how.
- The `*_exact` experiments are a baseline. They maximize the exact log-likelihood of the harmonium by
  enumeration. ρ_Z gets no gradient there, so ignore their log p̃ and ELBO.
- Results go to `results/canonical_circuit/<experiment>/`. The other directories there come from the old
  least-squares version, and so does `scratch/variational-conjugation/canonical-circuit-experiments.tex`.
  Both are superseded; the user decides how to replace the report.

## Validation (`validate.py`, all pass, about 4 min on CPU)

The setup is 4 neurons, d_X = 1, chain and full E, checked against brute-force grids. The likelihood matches
`BoltzmannLGM` exactly. r_N is zero for full E and matches the formula for chain E. log p(x), log p̃(x), the
ELBO and the KL all agree to about 1e-14. The ELBO estimator's mean is within 1.1 standard errors, and the
largest z-score of its gradient over all coordinates is 2.3.

## Results so far (laptop, seed 0, bump, chain E, constant ρ_Z, 4000 steps, one run each)

| | λ = 0 | λ = 1 | λ = 3 |
|---|---|---|---|
| exact log p(x) | −21.6 (falling) | 5.36 | 5.12 |
| log p̃(x) / ELBO | 5.75 / 5.73 | 5.36 / 5.36 | 5.11 / 5.11 |
| KL to exact posterior | about 24 | 0.000 | 0.000 |
| Spearman(E[z\|x], t) | – | 0.27 | −0.06 |
| E[x\|n] off the curve | – | 75% | 79% |

- With λ = 0, p̃ fits the data and the harmonium it is meant to approximate does not.
- With λ > 0, the gap closes, but z decouples: its residual variances drop to about 1e-6. At λ = 1 a gap of
  1.3 nats opened at step 3000 and closed by step 4000.
- z collapsing is the correct outcome if the lower layers explain the data. Whether they do is open. The
  fit is poor (most samples are off the curve) and log p(x) was still rising.

## Next (desktop, one experiment at a time, never in parallel)

1. `uv run python -m examples.canonical_circuit.validate` (sanity check on the new machine).
2. `uv run python -m examples.canonical_circuit.run --gpu --experiment bump_chain_exact` (3 runs). The
   question: does the exact fit use z (`corr_t`) and reach a higher log p(x) than the ELBO runs?
3. `... --experiment bump_chain_constant` (12 runs), then `... --experiment bump_chain_mlp` (12 runs).
4. `uv run python -m examples.canonical_circuit.plot --experiment <name>` for each. `plot.py` has not run
   on the new schema yet.
5. Interpret the results:
   - The exact fit uses z and gains log p(x), while the ELBO runs give it up: the penalty costs
     likelihood, which is a problem for the theory.
   - Neither uses z: the bump data does not call for hierarchy at this size.

## Notes

- Full E samples by Gibbs (`FullBoltzmann.sample`), which is slow inside training. Only chain experiments
  are defined; full E is covered by `validate.py`.
- On the laptop, CUDA failed to initialize (`cuInit` error 303); use `JAX_PLATFORMS=cpu` there.
- The port is uncommitted: the example files are modified against `fe5fa40`, and `NOTES.md` is new.
