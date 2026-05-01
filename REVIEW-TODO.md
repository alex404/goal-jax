# Review TODO

Hand-review checklist for the new code introduced in `944da5d` (Dirichlet) and `305337b` (dynamics readdition), after streamlining.

## Commit `944da5d` — Dirichlet + multivariate example

### Library

- [x] **`src/goal/models/base/dirichlet.py`** — `Dirichlet(n_categories)` as a `Differentiable` exponential family.
  - Sufficient stat `log x`, log-partition `sum(lgamma) - lgamma(sum)`, base measure `-sum(log x)`.
  - `initialize` overrides parent's `location=0.0` default to `location=1.0` — verify this is the intended contract (parent default would clip to `1e-3`).
  - `check_natural_parameters` cast `& jnp.all(params > 0).astype(jnp.int32)` — confirm consistent with parent return type.
- [x] **`src/goal/models/__init__.py`** — `Dirichlet` re-export.

### Docs

- [x] `docs/source/models/base/dirichlet.rst` (new leaf)
- [x] `docs/source/models/base/index.rst` toctree entry
- [x] `docs/source/examples.rst` (replaces bivariate_normal listing)

### Examples

- [x] **`examples/multivariate/{run,plot,types}.py`** (new; replaces deleted `bivariate_normal/`)
  - `compute_dirichlet_panel`: 3-category Dirichlet, true α=[3,7,5], 20-sample Adam fit (500 epochs, lr=0.05).
  - `compute_normal_panel`: 2D `Normal[PositiveDefinite]`, 20-sample fit, three densities (true, source-mean, natural-param). Verify all three render in `plot.py`.
  - `dirichlet_density_grid` masks out-of-simplex points to 0.

### Tests

- [x] **`tests/dirichlet.py`** — dimensions, sufficient stat, log-partition, digamma `to_mean`, log-density, uniform-Dirichlet density = (k-1)!, sampling shape/simplex/empirical-mean, density normalization via importance sampling, `initialize` positivity. (Tautological `test_log_partition_matches_lgamma_identity` was replaced with `test_log_partition_matches_scipy_gammaln` — independent reference via `scipy.special.gammaln`.)

---

## Commit `305337b` — Dynamics (post-streamlining)

### Module rename: `linear.py` → `map.py`

- [x] **`src/goal/geometry/manifold/map.py`** — generalized from `linear.py`. Adds `Map[D, C]` ABC and `MultilayerPerceptron[D, C]` (Glorot init, configurable `hidden_dims` + `activation`). All previous `LinearMap` / `EmbeddedMap` / `BlockMap` / `AmbientMap` / `SquareMap` / `AffineMap` semantics preserved. (Renamed `MLPMap` → `MultilayerPerceptron`; removed defaults on `hidden_dims` and `activation`; CLAUDE.md updated with no-defaults policy. `MLPTransition` not yet renamed for consistency.)
- [x] **`src/goal/geometry/__init__.py`** — exports `Map` and `MultilayerPerceptron`.
- [x] **`docs/source/geometry/manifold/map.rst`** (renamed from `linear.rst`).
- [x] **`examples/variational_mnist/model.py`** — single-line import path update (`linear` → `map`).

### New geometry layer: `dynamical.py`

- [ ] **`src/goal/geometry/exponential_family/dynamical.py`**:
  - `Transition[L]` ABC: `lat_man` property + `predict(params, belief) -> Array`.
  - `transpose_harmonium` helper — used by both `AnalyticTransition.predict` and `AnalyticLatentProcess.smooth`.
  - `AnalyticTransition[L]` — wraps `AnalyticConjugated[L, L]` kernel; `predict` derives obs-marginal of swapped harmonium.
  - `LatentProcess[O, L]` — `Triple[L, SymmetricConjugated[O, L], Transition[L]]` with `filter` (BPTT-friendly scan) and `log_observable_density`.
  - `AnalyticLatentProcess[O, L]` — adds `sample` (joint), `log_density` (joint), `smooth` (forward-backward), `posterior_statistics`, `mean_posterior_statistics`, `to_natural` (per-component delegation), `expectation_maximization`.
  - **Verify**: `posterior_statistics` zips `observations[t]` with `smoothed[t]` (T-length), with `prior_means` from `smoothed_z0`. Joints come from `smooth`'s mean-coordinate output of `kernel.to_mean`.
- [ ] **`docs/source/geometry/exponential_family/dynamical.rst`**.

### New models layer: `models/dynamical/__init__.py`

- [ ] **`NormalEmission`** — type alias `NormalAnalyticLGM[PositiveDefinite]`.
- [ ] **`LinearGaussianTransition(AnalyticTransition[FullNormal])`** — kernel: `NormalAnalyticLGM[PositiveDefinite]`. Factory: `create_linear_gaussian_transition(lat_dim)`.
- [ ] **`_CategoricalConjugated`** (private base) — shared `conjugation_parameters` and `to_natural_likelihood` over `(_obs_man, _lat_man)`. **Note**: `to_natural_likelihood` contains a Python `for j in range(1, n_states)` building `int_nat` columns. `n_states` is static so this unrolls at trace time; fine for HMM EM but flag if profiling shows trace-time bloat.
- [ ] **`CategoricalKernel(n_states)`** — square `_CategoricalConjugated` (obs = lat = `Categorical(n_states)`).
- [ ] **`CategoricalEmission(n_obs, n_states)`** — rectangular `_CategoricalConjugated`.
- [ ] **`CategoricalTransition(AnalyticTransition[Categorical])`** — wraps `CategoricalKernel`. Factory: `create_categorical_transition(n_states)`.
- [ ] **`MLPTransition[L]`** — wraps `MultilayerPerceptron[L, L]`; `predict` is one-line passthrough.
- [ ] **`KalmanFilter`** — `obs_dim` + `_lat_dim` fields with public `lat_dim` property. `from_standard(A, Q, C, R, μ₀, Σ₀)` builds via precision-weighted-interaction encoding. Factory: `create_kalman_filter(obs_dim, lat_dim)`. Verify `_build_harmonium` correctly composes emission/transition.
- [ ] **`HiddenMarkovModel`** — `n_obs` + `_n_states` fields with public `n_states` property. Factory: `create_hidden_markov_model(n_obs, n_states)`.
- [ ] **`src/goal/models/__init__.py`** — re-exports the above.
- [ ] **Docs**: `docs/source/models/dynamical/{index,transitions,kalman_filter,hmm}.rst`.

### Tests

- [ ] **`tests/dynamical.py`**:
  - `TestLinearGaussianTransition` / `TestCategoricalTransition` — dim, lat_man passthrough, predict shape, BPTT gradient.
  - `TestMLPTransition` — dim, predict shape, BPTT smoke through 5-step scan.
  - `TestKalmanFilter` — dim breakdown, sample, filter, smooth, log-observable consistency, EM finite, `from_standard` round-trip.
  - `TestHiddenMarkovModel` — dim, sample, filter, smooth, EM.
  - `TestKalmanFilterCorrectness::test_em_monotone` — EM monotonicity over 5 iterations.
  - `TestHMMCorrectness::test_filter_matches_manual_forward` — strongest single test: decodes `(π, A, B)` from natural params and compares filter LL against a manual forward algorithm.
  - `TestMLPBPTTGradientDescent` — gradient descent reduces loss.
- [x] **`tests/map.py`** — LinearMap rename regression + MultilayerPerceptron dim/call/Glorot tests.

### Examples

- [ ] **`examples/kalman_filter/{run,plot,types}.py`** — damped oscillator (1D obs, 2D latent), gradient ascent vs EM (200 steps each), filter + smooth visualization.
- [ ] **`examples/hmm/{run,plot,types}.py`** — discrete state-space (4 obs, 3 latent states), gradient ascent (200 steps) vs EM (100 steps), state-probability extraction.

### CI

- [ ] **`.github/workflows/examples.yml`** — workflow updated to drop `bivariate_normal`, add `multivariate`, `kalman_filter`, `hmm`.

### Misc

- [ ] **`CLAUDE.md`** — adds dynamical-models entry; test-mapping note for `dynamical.py`.

---

## Known follow-ups (not blockers)

- [ ] Property accessors in `KalmanFilter` / `HiddenMarkovModel` construct fresh frozen dataclasses on every access (`lat_man`, `emsn_hrm`, `transition`). Doesn't cause re-tracing but inflates Python-side trace work. Caching via `cached_property` or post-init field assignment would reduce JIT compile time on small problems (see `examples/kalman_filter/` profiling: ~6s compile + 25ms run per warm EM step at LAT_DIM=2).
- [ ] No correctness test for KF smoothing against an external reference (only EM monotonicity). HMM has the manual-forward-algorithm cross-check; consider an analogous RTS-smoother check for KF.
- [ ] The `eps = 1e-10` floor inside `_CategoricalConjugated.to_natural_likelihood` silently regularizes near-zero probabilities. Document or guard.
- [ ] Pre-existing (unrelated to these commits): `graphical.py` type errors from Harmonium bounds change; `examples/variational_mnist/model.py` import drift; deleted classes still referenced in some legacy test files (per `MEMORY.md`).
