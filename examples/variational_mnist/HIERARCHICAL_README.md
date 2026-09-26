# Hierarchical variational conjugation — chordal Boltzmann + continuous (WP2)

A local, out-of-`src` prototype of the paper's *"Variational Conjugation in
Hierarchical Models"* section, built to test the grant's WP2 claim: that we can
**iterate chordal Boltzmann spike layers with continuous variables** and train
the stack with variational conjugation while keeping the conjugation residual
small.

## Files

| file | what |
|---|---|
| `hierarchical.py` | `VariationalHierarchical[Observable, MidLatent]` — the 3-level model `p(x,y,z)=p(x|y)p(y|z)p(z)` + `build_boltzmann_gaussian_hierarchy(...)` factory |
| `validate_hierarchical.py` | correctness checks (decomposition, gradient unbiasedness, ELBO≤IWAE) |
| `hierarchical_experiment.py` | the WP2 comparison: diagonal vs chain vs chordal middle layer |
| `plot_hierarchical.py` | reads `hierarchical_experiment_results.json` → `hierarchical_experiment.png` |

## The model

`X` continuous observation ← `Y` (chordal) Boltzmann spike population ← `Z`
Gaussian top latent. The lower edge `p(x|y)` is a Gaussian-Boltzmann likelihood;
the upper edge `p(y|z)` is a Boltzmann population code. The upper edge is stored
as a whole `BoltzmannPopulationCode`, so its `conjugation_residual` supplies both
the generative residual `r*_Z` (at the generative Y-bias) and the recognition
inner residual `r_inner_Z` (at the posterior Y-bias).

Recognition is chain-factorized `q(y,z|x)=q(y|z,x)q(z|x)`; the learning signal
decomposes as `f = r_Y + r*_Z - r_inner_Z + c(x)` (verified exact to 1e-15).

### Two design points that differ from a naive reading of the paper

1. **Amortize the full top posterior, not a raw slope.** The paper writes the
   inner conjugation as an additive slope `rho^X_Z(x)`. A raw additive slope
   pushes the recognition Gaussian's precision out of the PD cone → NaN. Instead
   we amortize `q(z|x)` as a *valid* Gaussian (mean + Cholesky precision) and
   *derive* the slope as `q(z|x)_nat - (theta*_Z - rho0_Z)`. The decomposition
   identity holds for any valid `q(z|x)`, so this keeps correctness and validity.
2. **Fixed observable covariance (`obs_location_only=True`).** A fully
   y-dependent Normal likelihood collapses its variance (ELBO exceeds the data
   ceiling — the classic Gaussian degeneracy). Restricting `Theta_XY` to drive
   only the observable *mean* (via `GeneralizedGaussianLocationEmbedding`) is the
   paper's own fixed-covariance special case and removes the pathology.
   Gradient clipping + a non-finite-update guard handle the rest.

## Result

Data: an independent 6-mode mixture of Gaussians in R^10 (not model samples).
Middle layer: 16 Boltzmann units (4x4 grid for chordal). Top: 4-D Gaussian.
4000 steps. Ceiling (true-generator held-out mean log p) = -10.75.

| middle | ELBO test | gap-to-ceiling | Var[r_gen] | recon MSE |
|---|---|---|---|---|
| diagonal (independent) | -19.37 | 8.61 | 0.023 | 1.41 |
| chain | -18.57 | 7.82 | 0.052 | 1.10 |
| **chordal** (4x4 grid) | **-16.81** | **6.06** | **0.013** | **0.82** |

The chordal middle layer wins on **every** axis: best ELBO, best reconstruction,
and the *lowest* conjugation residual — i.e. correlated spike connectivity fits
the data better while remaining more conjugate. Residual variances decay over
training (chordal `Var[r_gen]`: 0.35 → 0.011), so conjugation is preserved as the
model fits. This is the WP2 thesis in miniature: chordal Boltzmann connectivity
buys representational power that variational conjugation can keep near-conjugate.

## Run it

```
uv run python -m examples.variational_mnist.validate_hierarchical   # correctness
uv run python -m examples.variational_mnist.hierarchical_experiment # ~30 min CPU
uv run python -m examples.variational_mnist.plot_hierarchical
```

## Not yet done / next steps

- Deeper stacks (L>1 spike/continuous iterations) — the class is 3-level; the
  paper's recursion `f = sum_l r_l + c(x)` extends it.
- A count/spike observable (`Binomials`/`Poissons`) variant — trivial to add
  (drop `obs_location_only`), closer to the Neuropixels setting.
- Scale to MNIST as the continuous observable (heavier; the existing
  mixture-latent tuning notes in `hierarchical_results.md` are a starting point).
- If this design settles, promote `VariationalHierarchical` into `src/goal/`.
