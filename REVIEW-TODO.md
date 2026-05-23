# REVIEW-TODO: manuscript-aligned variational rewrite

Tracking checklist for hand-review of the `variational.py` migration to manuscript §4.2 standard form. Plan: `/home/alex404/.claude-work/plans/alright-so-exponential-family-variationa-zany-balloon.md`.

## Automated verification (already run)

- `uvx basedpyright src/` → 0 errors, 0 warnings.
- `uvx basedpyright examples/variational_mnist/train.py examples/pendulum/run.py examples/torus_poisson/run.py` → 0 errors.
- `uv run python -m pytest tests/` → **270/270 passed** in 393s. Notable suites exercised: `tests/dynamical.py` (21 tests including `VariationalLatentProcess` filter/ELBO/SGD-monotonicity and `conjugation_metrics`/`regress_conjugation_parameters` round-trip), `tests/population_codes.py` (14 tests, all variational machinery).
- `uv run sphinx-build -W docs/source` → build succeeded, no new warnings.
- Smoke-run examples (all exit 0):
  - `examples/variational_mnist` (gradient mode, 30 steps, KL warmup 10): ELBO=-2763.67 → -2751.14, β-warmup engaged at step 0 (β=0.000), training stable.
  - `examples/pendulum` (LEARNABLE rho + REGULARIZED, 10 chunks each): both modes complete; regularized Var[RLS] drops 2.17 → 0.79 over training.
  - `examples/torus_poisson` (free + regularized + analytical, 4000 steps each): all three modes complete and produce sensible R² progressions. Final R² = 0.22 (free), 0.60 (regularized), 0.99 (analytical) — the analytical mode achieves near-perfect linear conjugation as expected.

## Mathematical correctness — sanity-check against manuscript §4.2

- [ ] **`conjugation_residual` formula matches manuscript Eq.~43.**
  - Code: `src/goal/geometry/exponential_family/variational.py:154-171`
  - Manuscript: $r(z) = \rho_Z \cdot s_Z(z) - \psi_X(\theta_X + \Theta_{XZ}\cdot s_Z(z)) + \psi_X(\theta_X)$
  - The new $+\psi_X(\theta_X)$ shift is the only behavioral change versus the old `reduced_learning_signal`. Variance / covariance / regression-with-intercept consumers are shift-invariant — the rename should be observably a no-op for them.
- [ ] **`elbo_at` decomposition $c(x) + \mathbb{E}_q[r(Z)]$ matches manuscript Eq.~43–44.**
  - Code: `src/goal/geometry/exponential_family/variational.py:173-227`
  - $c(x)$ in code = `s_X·θ_X + ψ_Z(θ̂_{Z|X}) − ψ_Z(θ_Z) − ψ_X(θ_X) + log h_X(x)`. Derivation: expand $\log p(x|z) + \log p(z) - \log q(z|x)$, cancel shared $s_Z·θ_Z$ and $s_X·Θ·s_Z$ terms, separate $z$-dependent ($r$) from $x$-dependent ($c$) pieces. The $±ψ_X(θ_X)$ is a convention to make $r$ vanish exactly at conjugation.
- [ ] **Score-function gradient matches manuscript Eq.~51.**
  - Surrogate: `c_x + direct + score - sg(score)`.
  - Direct gradient: `direct = mean(r_vals)` autodiff at sg'd $z$ gives $\mathbb{E}_q[\nabla r]$.
  - Score correction: `mean(sg(r-b) * log_q)` autodiff gives $\mathbb{E}_q[(r-b)\nabla\log q]$, which collapses to $\mathbb{E}_q[r\nabla\log q]$ under the zero-score identity.
  - Combined with $\nabla c(x)$: matches manuscript's $\nabla\mathcal{L} = \mathbb{E}_q[\nabla\log p] + \mathbb{E}_q[r\nabla\log q]$ via the identity $\log p = r + c + \log q$.
  - Manuscript says no baseline needed in the population gradient; the sample-mean $b$ in code only reduces finite-sample variance.
- [ ] **β-warmup formula $\mathcal{L}_\beta = \mathcal{L}_1 + (1-\beta)\cdot\mathrm{KL}$.**
  - Derivation: $\mathcal{L}_\beta = \mathbb{E}_q[\log p(x|z)] - \beta\mathrm{KL}$ and $\mathcal{L}_1 = \mathbb{E}_q[\log p(x|z)] - \mathrm{KL}$, so $\mathcal{L}_\beta = \mathcal{L}_1 + (1-\beta)\mathrm{KL}$.
  - Sanity: at $\beta=0$ (no KL), $\mathcal{L}_0 = \mathcal{L}_1 + \mathrm{KL} = \mathbb{E}_q[\log p(x|z)]$ ✓.
  - Applied in `examples/variational_mnist/train.py:351-357` (gradient) and `:411-419` (analytical).

## `src/goal/geometry/exponential_family/variational.py` (main rewrite)

- [ ] **Module docstring** (L1-9): manuscript §4.2 reference, $r$ as central object, β handled externally, hierarchical extension note.
- [ ] **`VariationalConjugated` class docstring** (L25-58): unchanged from previous version except surrounding context. Worth a glance to confirm it still reads coherently next to the new method order.
- [ ] **`conjugation_residual` method** (L154-171, replaces `reduced_learning_signal`):
  - Renamed in code, added `+ self.obs_man.log_partition_function(obs_params)` shift.
  - Compare to old body (in git: `git show HEAD:src/goal/geometry/exponential_family/variational.py | sed -n '246,259p'`). Only difference: the added shift term.
- [ ] **`elbo_at` body** (L173-227): full rewrite. The surrogate `c_x + direct + score - sg(score)` is value-correct as an MC estimate and gradient-correct under autodiff. Three stop-gradients (samples, $r$ inside score correction, `sg(score)`) documented in the method docstring.
- [ ] **`mean_elbo`** (L229-239): no `kl_weight` parameter; otherwise unchanged.
- [ ] **`elbo_divergence`** (L241-249): unchanged body; docstring updated to note it's no longer on the critical path of `elbo_at` but is the lever for external β-warmup.
- [ ] **Removed: `elbo_reconstruction_term`.** Was unused outside `elbo_at`; grep before/after to confirm no orphan reference.
- [ ] **`prior_conjugation_loss`** (L253-263): $\mathcal{R}_p = \mathrm{Var}_{p_Z}[r(Z)]$. Sg on prior samples; gradient flows through `params` in `conjugation_residual` at fixed $z$. Mirrors the inline pattern callers were already using.
- [ ] **`recognition_conjugation_loss_at`** (L265-275): $\mathcal{R}_q(x) = \mathrm{Var}_{q(Z|X=x)}[r(Z)]$. Same sg policy as the prior version, but samples from the recognition model. New to the codebase — not yet called anywhere.
- [ ] **`mean_recognition_conjugation_loss`** (L277-285): batch mean of the per-$x$ recognition loss. Mirrors `mean_elbo` convention.
- [ ] **`conjugation_metrics`** (free function, L367-396): refactored to delegate to `prior_conjugation_loss` for `var_f`; keeps inline $\mathrm{Var}_p[\psi_X]$ for the $R^2$ ratio. Public signature `(var_f, std_f, r_squared)` preserved; all six callers untouched.
- [ ] **`regress_conjugation_parameters`** (L309-365): body unchanged. Docstring updated to mention the shift-invariance of the intercept-absorbed regression.
- [ ] **`reconstruct` / `reconstruction_error`** (L399-425): unchanged.
- [ ] **`stop_gradient` audit (manuscript-correctness).** Five sg sites total, all annotated in code:
  1. `regress_conjugation_parameters` — sampler may be non-differentiable.
  2. `elbo_at` z-samples — same reason.
  3. `elbo_at` `r_sg = sg(r_vals)` — prevents double-counting direct gradient in score correction.
  4. `elbo_at` `score - sg(score)` — keeps value MC-clean while preserving the gradient.
  5. `prior_conjugation_loss` / `recognition_conjugation_loss_at` samples — same as #1, plus the variance-as-regularizer convention that gradient flows only through $r(\cdot;\theta)$ at fixed $z$ (matches what pendulum/torus_poisson were doing inline).

## `src/goal/geometry/exponential_family/dynamical.py`

- [ ] **`VariationalLatentProcess.filter`** (L438-475): `kl_weight` parameter removed; docstring updated to point at external β-warmup via `ems_hrm.elbo_divergence`. Pendulum is the sole caller and uses defaults — `test_filter_shapes` and `test_grad_descent_reduces_loss` in `tests/dynamical.py` exercise this and pass.
- [ ] **`VariationalLatentProcess.mean_elbo`** (L477-491): `kl_weight` removed; same notes.

## `src/goal/models/graphical/variational.py` (no code edits — review for cascade-correctness)

- [ ] **`VariationalHierarchicalMixture`** inherits all ELBO machinery from `SymmetricVariationalConjugated`. The new `elbo_at`/`conjugation_residual`/`*_conjugation_loss` methods are inherited unchanged.
  - Its override of `conjugation_parameters` returns a Prior-shaped $\rho$ via `ObservableEmbedding(mix_man).embed(rho)` — this composes correctly with `conjugation_residual` (which consumes `pst_prr_emb.embed(s_z) ⋅ conjugation_parameters(params)` for the $\rho$ term).
  - `tests/hmog.py` (22 tests) exercises the analytic hierarchical path; the variational path is exercised end-to-end via `examples/variational_mnist`. Both pass.
  - Suggested manual check: pick one concrete instance (e.g., `BinomialHierarchicalMixture` in the mnist example), call `model.conjugation_residual(params, z)` and `model.prior_conjugation_loss(key, params, n_samples)` interactively, and confirm the values look sensible (Var[r] is non-trivial when conjugation is imperfect, near zero when perfect).

## Callers (renames only, no semantic change)

- [ ] **`examples/pendulum/run.py:397`**: `reduced_learning_signal` → `conjugation_residual` inside the regularized-mode loss function. Verified by smoke run.
- [ ] **`examples/torus_poisson/run.py:308, 337`**: same rename in both `loss_fn_regularized` and `loss_fn_analytical`. Verified by smoke run across all three modes.

## `examples/variational_mnist/train.py` (β-warmup externalized — most invasive caller change)

- [ ] **`loss_fn_gradient`** (L348-389): `mean_elbo` called without `kl_weight`; `elbo_divergence` vmapped over the batch for `mean_kl`; `elbo = elbo_1 + (1.0 - beta) * mean_kl`. Logged `elbo` is the β-scaled value, matching prior semantics.
- [ ] **`loss_fn_analytical`** (L392-443): same pattern, with `params_with_rho` (not `params`) in both `mean_elbo` and the vmapped `elbo_divergence`. Verify the `params` variable used in both calls is the same.
- [ ] Cross-check: at $\beta = 1$, the new code should yield exactly the same `elbo` as the old code (within MC noise). Quick local test: snapshot the elbo trajectory before and after, with a fixed seed and `--kl-warmup-steps 0` (so β=1 from step 0).
- [ ] Sign convention: $\beta < 1$ during warmup should *increase* `elbo` (looser KL penalty). The smoke run shows step 0 with β=0.000, which makes `elbo = elbo_1 + 1.0*KL = E_q[log p(x|z)]` — confirm this matches intent.

## `docs/source/geometry/exponential_family/variational.rst`

- [ ] **No edits required.** The `autoclass :members:` directive auto-includes the new methods (`conjugation_residual`, `prior_conjugation_loss`, `recognition_conjugation_loss_at`, `mean_recognition_conjugation_loss`). Sphinx build succeeded with no warnings. Worth a render check: open `docs/build/geometry/exponential_family/variational.html` and confirm the new methods appear in the class docs with their docstrings.

## Variance regression (the main empirical risk)

- [ ] The plan explicitly accepts a small gradient-variance increase: the old recon+KL form Rao-Blackwellized the $\rho \cdot s_Z$ piece via the closed-form KL, while the new standard form estimates it via MC (bounded by $|\rho|^2 \cdot \mathrm{Var}_q[s_Z]/K$).
- [ ] **Compare training trajectories** on a representative example (torus_poisson with fixed seed is the cleanest test — three modes, well-instrumented). Run with the previous commit on `main` once for baseline, then with `HEAD`. Expectation: ELBO and conjugation-variance trajectories track within MC noise. If they diverge meaningfully, the followup is to Rao-Blackwellize $\rho \cdot s_Z$ in `elbo_at` (recovers the old form's variance without giving up the residual API).
- [ ] If variance is observably worse and matters for training stability, consider raising `n_mc_samples` (cheap workaround) or implementing the Rao-Blackwellized variant (proper fix).

## Documentation / docstring consistency

- [ ] The plan-cited paragraph in the module docstring flags that a hierarchical extension (manuscript §4.2.7) will likely require a $r_\beta + c_\beta$ residual decomposition. Make sure this matches what you actually intend to do next — if hierarchical implementation is imminent, this footnote sets expectations for collaborators reading the code now.
- [ ] `elbo_divergence` docstring positions itself as "not on the critical path" — confirm this matches your mental model. It's the public API for callers wanting β-VAE and for KL diagnostics, and that's its sole reason for existing post-rewrite.

## Followups deferred from this PR (per plan)

- Implement manuscript §4.2.7 (`HierarchicalVariationalConjugated`, MLP-parameterized $\rho_Z(x)$, joint recognition harmonium, per-block residuals). The current rewrite prepares the bivariate path for this.
- Rao-Blackwellize $\rho \cdot s_Z$ in `elbo_at` if the MC variance shows up empirically.
- Optional: lift the inline `prior_conjugation_loss` pattern out of `examples/pendulum/run.py` and `examples/torus_poisson/run.py` to call the new method directly. Pure cleanup; no behavior change.
