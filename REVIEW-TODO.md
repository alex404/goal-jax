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

- [x] **`conjugation_residual` formula matches manuscript Eq.~43.**
  - Code: `src/goal/geometry/exponential_family/variational.py:154-171`
  - Manuscript: $r(z) = \rho_Z \cdot s_Z(z) - \psi_X(\theta_X + \Theta_{XZ}\cdot s_Z(z)) + \psi_X(\theta_X)$
  - The new $+\psi_X(\theta_X)$ shift is the only behavioral change versus the old `reduced_learning_signal`. Variance / covariance / regression-with-intercept consumers are shift-invariant — the rename should be observably a no-op for them.
- [x] **`elbo_at` decomposition $c(x) + \mathbb{E}_q[r(Z)]$ matches manuscript Eq.~43–44.**
  - Code: `src/goal/geometry/exponential_family/variational.py:173-227`
  - $c(x)$ in code = `s_X·θ_X + ψ_Z(θ̂_{Z|X}) − ψ_Z(θ_Z) − ψ_X(θ_X) + log h_X(x)`. Derivation: expand $\log p(x|z) + \log p(z) - \log q(z|x)$, cancel shared $s_Z·θ_Z$ and $s_X·Θ·s_Z$ terms, separate $z$-dependent ($r$) from $x$-dependent ($c$) pieces. The $±ψ_X(θ_X)$ is a convention to make $r$ vanish exactly at conjugation.
- [x] **Score-function gradient matches manuscript Eq.~51.**
  - Surrogate: `c_x + direct + score - sg(score)`.
  - Direct gradient: `direct = mean(r_vals)` autodiff at sg'd $z$ gives $\mathbb{E}_q[\nabla r]$.
  - Score correction: `mean(sg(r-b) * log_q)` autodiff gives $\mathbb{E}_q[(r-b)\nabla\log q]$, which collapses to $\mathbb{E}_q[r\nabla\log q]$ under the zero-score identity.
  - Combined with $\nabla c(x)$: matches manuscript's $\nabla\mathcal{L} = \mathbb{E}_q[\nabla\log p] + \mathbb{E}_q[r\nabla\log q]$ via the identity $\log p = r + c + \log q$.
  - Manuscript says no baseline needed in the population gradient; the sample-mean $b$ in code only reduces finite-sample variance.
- [x] **β-warmup formula $\mathcal{L}_\beta = \mathcal{L}_1 + (1-\beta)\cdot\mathrm{KL}$.**
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

---

# REVIEW-TODO: hierarchical extension (manuscript §4.2.7)

Tracking checklist for hand-review of the `VariationalHierarchical` combinator (manuscript §4.2.7 "Variational Conjugation in Hierarchical Models"). Plan: `/home/alex404/.claude-work/plans/alright-we-layed-a-tingly-flute.md`.

Ordered for a single linear walk from the base of the dependency tree outward: start at `geometry/exponential_family/variational.py` (the only file that defines new abstractions); each subsequent module consumes the one above it. Each module section opens with a one-line orientation so you can skim past anything you've already internalized.

## Automated verification (already run)

- `uvx basedpyright` over the full project → 0 errors, 0 warnings.
- `uv run python -m pytest tests/` → **283/283 passed** in 340s; the 13 new tests in `tests/variational.py` plus all 270 prior tests.
- Smoke runs (all exit 0):
  - `create_deep_model(n_latent_mid=64, n_latent_top=32, n_clusters=10)` constructs cleanly; `model.dim = 55577` (Lower 51088 + Upper 2473 + ρ_Y + φ); initial ELBO ≈ -4320 on a sparse-x input.
  - 20-step Adam loop on a `(8, 784)` synthetic minibatch with `n_latent_mid=32, n_latent_top=16, n_clusters=4`: loss drops 3674.8 → 3430.1 (Δ ≈ -245), confirming gradient flow to all four slots in practice.
- The full `examples/variational_mnist/train.py` pipeline was **not** rewired for the deep model in this PR — see "Followups deferred" at the end.

## Big-picture architecture (read this first)

The hierarchical case differs structurally from the bivariate case in three ways. Understanding these motivates the implementation choices:

1. **Recognition is a joint harmonium, not a flat exponential family.** $q(Y, Z \mid x)$ has the same algebraic shape as `Upper`, just with shifted biases. So the recognition's "natural parameters" live in `Upper.dim` shape, and sampling/density delegate to `Upper.sample` / `Upper.log_density`.
2. **The joint log-partition is not closed-form.** This breaks the bivariate `c(x) + E_q[r]` ELBO formulation, which relied on `prr_man.log_partition_function`. The hierarchical `elbo_at` uses the explicit recon+KL form via MC + score-function correction instead.
3. **Two ρ's live at different boundaries.** $\rho_Y$ is input-independent (X–Y boundary; same as bivariate). $\rho_Z(x) = f(\hat\theta_Y(x); \phi)$ is **input-dependent** via a learnable map (Y–Z boundary inside the recognition, per Eq.~57 in the manuscript).

These three facts drive the rest: a generalization of `VariationalConjugated`'s `Conjugation` slot, a separate `VariationalHierarchical` class (not a subclass — see **§1.B** below), and a custom `elbo_at` override.

---

## §1. `src/goal/geometry/exponential_family/variational.py`

> The base file. Everything else cascades from here. All new abstractions (the relaxed `Conjugation` bound, the optional-`x` API, `HierarchicalConjugation`, `VariationalHierarchical`) are defined here.

### §1.A — `VariationalConjugated` generalization (the `Conjugation: Manifold` + optional-`x` API)

This is the change you suggested mid-implementation. It generalizes the existing class so that arbitrary `Conjugation` manifolds (including MLP-parameter manifolds) fit. Backward-compatible for all current concrete subclasses.

- [ ] **`Conjugation: ExponentialFamily` → `Conjugation: Manifold`** (`variational.py:32`). Audit: searched for ExpFam-specific method calls on `cnj_man` inside the class — only `.dim` (Manifold protocol) is used. Backward-compatible: existing concrete subclasses pass an ExpFam, which is still a Manifold.
  - Propagated to `SymmetricVariationalConjugated` (L386) and the four free-function signatures (`regress_conjugation_parameters`, `conjugation_metrics`, `reconstruct`, `reconstruction_error`).
  - Unused `ExponentialFamily` import dropped (`variational.py:26`).
- [ ] **`conjugation_parameters(params, x=None)` and `conjugation_residual(params, z, x=None)`** — optional `x` added (`variational.py:128-142, 156-175`). Default behavior unchanged: subclasses that ignore `x` (the input-independent case) get a `del x` and the original logic. Subclasses with input-dependent ρ override and use `x`.
  - `approximate_posterior_at` (L142-152) threads `x` through.
  - `elbo_at` (L218) and `recognition_conjugation_loss_at` (L275) pass `x` to `conjugation_residual`.
  - `prior_conjugation_loss` (L254-264) does **not** take `x` — it estimates $\mathrm{Var}_{p_Z}[r]$, well-defined for input-indep ρ but underspecified for input-dep ρ. For the hierarchical case use the recognition-side variants instead.
- [ ] **Docstring updates** to reflect the generalization: class header (L37-58) and `conjugation_parameters` (L128-142) both now mention the input-dependent case as a first-class option, not just an afterthought.

### §1.B — Why `VariationalHierarchical` is **not** a subclass of `VariationalConjugated`

> **RESOLVED (subsequent refactor).** The bound was relaxed from `Differentiable` to `Generative`, methods that needed closed-form $\psi$ were moved to a new `VariationalDifferentiable` subclass, and `VariationalHierarchical` now extends `VariationalConjugated` directly. The notes below describe the old design; kept for context.

Read this before reading the class itself — it explains the choice that drives the rest of §1.C–§1.D.

- [ ] **The bound problem.** `VariationalConjugated`'s `Posterior`/`Prior` type variables are bounded by `Differentiable`. The natural choice for hierarchical would be `Posterior = Prior = Upper` so `pst_man.sample(q_params, n)` returns joint $(y, z)$ samples via `Upper.sample`. But `Upper` is itself a `VariationalConjugated` — a `Manifold` but **not** a `Differentiable` (it has `sample` and `log_density` defined directly, but no closed-form joint `log_partition_function`).
- [ ] **The runtime problem.** Even if we passed `Any` to bypass the type bound, the inherited `elbo_at` calls `prr_man.log_partition_function`, which would crash at runtime for the hierarchical case.
- [ ] **The resolution.** Separate class, duck-typed to expose the same method names. Sidesteps the bound issue without compromising the existing class. Tradeoff: existing example utilities that lean on `pst_man.to_mean` (`reconstruct`, `iwae_bound`, etc.) don't automatically work; see the followups at the end.

### §1.C — `HierarchicalConjugation` Pair manifold

- [ ] **Class** (`variational.py:349-370`). A concrete `Pair[LwrCnj, RhoMap]` whose two slots hold ρ_Y (input-independent, Bernoullis-shaped) and the rho-map's parameters $\phi$. Lives in the third slot of `VariationalHierarchical`'s Triple.
  - The reason for combining into a Pair (rather than a Quadruple at the outer class level): preserves the 3-tuple `split_coords` convention so `_, lkl, _ = model.split_coords(params)` patterns from bivariate-era callers would still parse if a deep model were dropped in. Currently the example code uses `model.likelihood_function(params)` instead, so this is forward-looking. **Decision call**: keep as Triple+Pair, or switch to outer Quadruple? Both work; Triple+Pair has the slight edge in compatibility.

### §1.D — `VariationalHierarchical` class (`variational.py:478-757`, ~280 lines)

The class itself. Read in this order:

- [ ] **Class signature and type parameters** (`variational.py:478-498`):
  - Five params: `Observable: Differentiable`, `Lower`, `Upper`, `LwrCnj: Manifold`, `RhoMap: Map[Any, Any]`.
  - Inherits from `Triple[Any, Any, HierarchicalConjugation[LwrCnj, RhoMap]]` (NOT `VariationalConjugated` — see §1.B). The `Any` in slots 1 and 2 is because the actual manifold (Upper as a generic, Lower's `lkl_fun_man` for some inferred Y) can't be expressed in the type parameter list cleanly.
- [ ] **Triple slot layout** (L532-552):
  - `fst_man = self.upr_hrm` (Upper's full params live here).
  - `snd_man = self.lwr_hrm.gen_hrm.lkl_fun_man` (lower's $(\theta_X, \Theta_{XY})$ — same shape as in the bivariate Lower).
  - `trd_man = HierarchicalConjugation(_lwr_cnj_man=..., _rho_map_man=...)` — the combined $(\rho_Y, \phi)$ storage.
- [ ] **Manifold accessors** (L555-572): `obs_man = Lower.obs_man`, `pst_man = prr_man = Upper`, `cnj_man = trd_man`. These mirror `VariationalConjugated`'s names so the example code can use the same dot-access patterns.
- [ ] **Parameter accessors** (L575-586): `prior_params(params)`, `likelihood_function(params)`, `split_conjugation_params(params) → (ρ_Y, φ)`. Trivial slot extractors; named to match `VariationalConjugated`'s API.
- [ ] **`_shifted_upper_params`** (L592-628) — **the core method; everything else delegates here.** Builds the recognition's parameter vector in `Upper.dim` shape:
  - Y-bias shift: $\hat\theta_Y(x) = \theta_Y + s_X(x)\cdot\Theta_{XY} - \rho_Y$. Computed via `Lower.gen_hrm.int_man.transpose_apply(lwr_int_p, s_x)` for the $s_X(x)\cdot\Theta_{XY}$ piece (transpose-applied because `int_man` maps Y→X; we want the contribution back to Y).
  - $\rho_Z(x)$: evaluated by `rho_map(phi, hat_eta_y)`. The rho-map takes the X-shifted Y-bias as input. **Note**: $\Theta_{YZ}$ enters implicitly through the map's domain shape but is not an explicit input — confirm this matches your intent versus the manuscript's $f(\hat\theta_Y(x), \Theta_{YZ}; \phi)$.
  - $\rho_Z(x)$ is **folded into the upper's prior slot** via `Upper.conjugation_parameters(synth)` and subtracted from `upr_prior`. The subtle move: by baking it into the prior rather than leaving it in the rho slot, `Upper.sample(recog_params, …)` and `Upper.log_density(recog_params, …)` automatically respect the correction (those methods read from the prior, not the rho slot). The test `test_elbo_gradient_flows_to_all_slots` exists specifically to guard against regression here — the initial implementation put ρ_Z(x) in the rho slot and φ got zero gradient.
- [ ] **`approximate_posterior_at`** (L630-638): one-liner wrapper around `_shifted_upper_params`. Returns Upper-shaped natural params.
- [ ] **`conjugation_parameters(params, x)`** (L642-680): the public ρ accessor. Composes ρ_Y embedded into Upper's observable slot via `ObservableEmbedding(Upper.gen_hrm).embed(rho_y)` + ρ_Z(x) computed via the rho-map and passed through `Upper.conjugation_parameters` to apply Upper's own structural completion (e.g., mixture completion for `VariationalHierarchicalMixture`). Sum of the two pieces (both in Upper-shape). Raises `ValueError` if `x is None`.
- [ ] **Per-level residuals**: `lower_conjugation_residual(params, y)` (L685-692), `upper_conjugation_residual(params, z, x)` (L694-697). Both delegate to the inner harmonium's `conjugation_residual` with a synthetic params vector. The Lower one uses a zero-prior synthetic (residual doesn't use prior anyway); the Upper one uses the full shifted params so the bias is correct. Match manuscript Eq.~60–61.
- [ ] **`elbo_at`** (L702-731): full MC + score-function. Computes `f_θ = log p(x|y) + log p(y,z) - log q(y,z|x)` per sample, averages, adds score-function correction with sample-mean baseline.
  - Uses the explicit recon+KL form rather than the bivariate $c(x) + E_q[r]$ form because no closed-form `prr_man.log_partition_function` exists for the joint.
  - The score-function surrogate `direct + score - sg(score)` is the same pattern as bivariate, applied to the full $f_\theta$. The zero-score identity makes the choice of $f_\theta$ vs $r$ invariant up to baseline; the sample-mean baseline `b = mean(sg(f_vals))` reduces finite-sample variance.
  - Three stop-gradients (samples, `f_sg`, `sg(score)`) for the same reasons as the bivariate version. The `f_theta` inner function returns both the integrand and `log_q` so the score correction computes in one vmap.
  - **Variance note**: relative to the bivariate's $c + r$ decomposition, this form does NOT analytically factor out the $x$-dependent piece. Expect higher MC variance on the generative-parameter gradient. If empirically problematic, a future Rao-Blackwellization could split out the analytically-tractable parts ($\log p(x|y)$ is closed-form given $y$).
- [ ] **`mean_elbo`** (L733-743): trivial batch wrapper. No `kl_weight` parameter (consistent with the bivariate post-rewrite).
- [ ] **Per-level recognition losses** (L747-769): `lower_recognition_conjugation_loss_at`, `upper_recognition_conjugation_loss_at`. Both sample $(y, z) \sim q$ jointly via `Upper.sample(shifted, n)`, slice out the relevant component, and compute the variance of the corresponding residual. **Marginalizes** the irrelevant latent — confirm this matches your desired regularizer semantics (the alternative would be $r_Y$ variance under $q(Y \mid x)$ exactly, marginalizing $z$ analytically if possible).
- [ ] **`likelihood_at`** (L773-777): natural params of $p(x \mid y)$ at a $Y$ sample. Just `Lower.gen_hrm.lkl_fun_man(lkl, s_Y(y))`. Direct path rather than going through `Lower.likelihood_at` with a synthetic Lower params vector.
- [ ] **`sample`** (L779-797): ancestral sampling through the chain. First `Upper.sample(upr_params, n)` gives joint $(y, z)$; then for each $y$ draws $x \sim p(x \mid y)$. Returns `[x | y | z]` concatenated.
- [ ] **`log_density`** (L799-808): joint $\log p(x, y, z) = \log p(y, z) + \log p(x \mid y)$. Slices `xyz` into `x`, `yz`; the inner `yz` further splits inside `Upper.log_density` ancestrally — matches manuscript p.~22.
- [ ] **`initialize` / `initialize_from_sample`** (L812-842): build via component initializations (Upper, Lower) + zero ρ_Y + Glorot-initialized rho-map. Lower's prior slot (which the bivariate framework expects) is discarded — Lower's "prior" role is played by Upper here, so we keep only Lower's likelihood. Worth confirming this matches your structural intent.
  - `_initialize_map_params` helper (L845-849): uses `glorot_initialize` if the Map provides it (MLP does), otherwise small-scale Gaussian. Affine-shaped MLPs (`hidden_dims=()`) get Glorot.

---

## §2. `src/goal/models/graphical/variational.py`

> Downstream subclass: only one-line signature changed. No new logic.

- [ ] **`VariationalHierarchicalMixture.conjugation_parameters`** (L58-65) updated to accept `x: Array | None = None` and `del x` it. Required because the relaxed base now passes `x` through (see §1.A). The body is unchanged.
  - Without this, the inherited `conjugation_residual` (which passes `x=None`) would `TypeError`. Caught by the test suite.
  - Confirm: nothing else in this module needed touching. All other methods (`mix_man`, `cnj_man`, `get_cluster_probs`, `prior_entropy`) are unaffected.

---

## §3. `src/goal/geometry/__init__.py`

> Pure plumbing: re-exports the two new public names.

- [ ] **`VariationalHierarchical` and `HierarchicalConjugation`** added to imports (L42-46) and `__all__` (L112, 144). Verify nothing else was disturbed (`git diff` shows the two import-block additions and the two `__all__` insertions in alphabetical order).
- [ ] **`src/goal/models/__init__.py`** intentionally not touched. The new combinator lives in `geometry/` per the existing convention (variational machinery is geometry-layer; concrete models that compose it live in `models/`). If you eventually want a concrete `VariationalDeepBoltzmannMixture` for cross-example reuse, that'd go in `models/graphical/`.

---

## §4. `examples/variational_mnist/model.py`

> The example glue. One existing override needs the new signature; everything else is additive.

### §4.A — Existing override that had to follow §1.A

- [ ] **`VariationalFullMixture.conjugation_parameters`** (L133-139): same fix as `VariationalHierarchicalMixture` in §2 — accept `x: Array | None = None` and `del x` it. Body unchanged.

### §4.B — New concrete classes

- [ ] **`ConcreteFlatVariational`** (`model.py:188-208`): a `SymmetricVariationalConjugated` subclass with `_gen_hrm` field. Mirrors the existing `ConcreteVariationalHierarchicalMixture` pattern. Used as the lower in `create_deep_model`. **Notable**: this is the first non-test `SymmetricVariationalConjugated` concrete in the example *without* a mixture wrap — the simplest possible variational harmonium concretion.
- [ ] **`ConcreteVariationalHierarchical`** (`model.py:213-237`): the concrete five-type-parameter `VariationalHierarchical`. Just stores `(_lwr_hrm, _upr_hrm, _rho_map)` and exposes them via abstract properties.

### §4.C — New factory

- [ ] **`create_deep_model`** (`model.py:296-336`): the factory. Defaults to `rho_hidden_dims=()` (affine map). The MLP's domain is `y_mid` (Bernoullis of size `n_latent_mid`) and codomain is `y_top` (Bernoullis of size `n_latent_top`). Activation `jax.nn.tanh` — arbitrary, won't trigger because `hidden_dims=()`; only matters if you set `rho_hidden_dims` non-empty.
- [ ] **`create_model` unchanged** for "hierarchical" and "full" modes (`model.py:246-280`). The deep mode is opt-in via the separate `create_deep_model` factory. The existing `MixtureModel` type union does **not** include the deep model — intentional, see followups.

---

## §5. `tests/variational.py` (new file)

> 13 tests over the deep model, plus the local concrete classes needed to instantiate it without leaning on the example module.

- [ ] **Three local concrete classes** at the top of the file:
  - `_FlatVC[Observable, Latent]`: a `SymmetricVariationalConjugated` for the lower (no mixture, just flat Bernoullis-Bernoullis).
  - `_ConcreteVHM[Observable, BaseLatent]`: bare-bones `VariationalHierarchicalMixture` subclass, mirroring `examples/variational_mnist/model.py`'s `ConcreteVariationalHierarchicalMixture`.
  - `_DeepHrm[...]`: a `VariationalHierarchical` subclass with three private fields (`_lwr_hrm`, `_upr_hrm`, `_rho_map`).
- [ ] **`_make_model()` factory** (small dims: `n_x=4, n_y=3, n_z=2, n_k=3`) used by every test.
- [ ] **Test coverage** (each is a separate method on `TestVariationalHierarchical`):
  - Dimensions and split/join round-trip.
  - Conjugation slot splits into $(\rho_Y, \phi)$ with correct sub-shapes.
  - Sample shape: $(n, x_{\dim} + y_{\dim} + z_{\dim} + 1)$ (the $+1$ is the Categorical's integer index).
  - Joint log-density finite.
  - Recognition params shape equals `Upper.dim`.
  - ELBO finite (`test_elbo_finite`) and gradient flows to **every slot** including φ (`test_elbo_gradient_flows_to_all_slots`). This last one caught a real bug during development — the initial implementation had ρ_Z(x) sitting in the upper-rho slot, which `Upper.sample`/`Upper.log_density` ignore. Fixed by folding into the prior.
  - Per-level residuals (`r_Y`, `r_Z`) finite at fixed inputs.
  - Per-level recognition losses non-negative and finite.
  - `initialize_from_sample` produces finite params at the expected dim.
- [ ] **Suggested manual exploration**: load `_make_model()` in a REPL and inspect the recognition's marginal over $Y$ vs the prior — at random init, expect them close (small bias shift); after a few training steps, expect divergence as recognition starts conditioning on $x$.

---

## §6. `docs/source/geometry/exponential_family/variational.rst`

> Pure docs: adds the two new `autoclass` blocks.

- [ ] **"Hierarchical Composition" section** (L30-41) inserted between the existing class blocks and the helpers. `autoclass` blocks for `VariationalHierarchical` and `HierarchicalConjugation`, both with `:members:`. Render check: `uv run sphinx-build docs/source docs/build` and open the rendered page to confirm the inheritance diagram and method list look right.

---

## Followups deferred from this PR (per plan)

- [ ] **`train.py` integration for the deep mode.** The training script's hot paths assume `model.bas_lat_man`, `model.gen_hrm`, `model.mix_man`, `model.n_categories`, and `model.pst_man.to_mean` — all of which differ for the deep model. A clean rewire would either (a) generalize the model interface to a small common protocol, or (b) introduce a parallel `train_deep.py`. **Suggested first step**: factor out the four model-specific accessors into duck-typed module-level helpers, then a `VariationalHierarchical` arm can supply each one (e.g., reconstruction via MC samples rather than `pst_man.to_mean`).
- [ ] **Reconstruction via MC.** Specifically, `examples/variational_mnist/model.py:reconstruct` uses `pst_man.to_mean(q_params)` to get $E[s_Y]$ analytically. For the deep model, this isn't tractable (Y's marginal is a mixture of Bernoullis over $Z$, exponentially many components). The fix is straightforward — sample $y \sim q(Y \mid x)$, average $s_Y$, apply lower's likelihood — but the function signature needs to grow a `key` argument.
- [ ] **Rho-map domain considerations.** Currently `rho_map: MultilayerPerceptron[y_mid, y_top]` takes the shifted Y-bias $\hat\theta_Y(x)$ as input. The manuscript writes $\rho_Z(x) = f(\hat\theta_Y(x), \Theta_{YZ}; \phi)$ — $\Theta_{YZ}$ enters via the **input shape** (the map's domain dim matches Y) but not as an explicit input. If you want a parameterization where $\Theta_{YZ}$ explicitly conditions the map, the rho-map would need a different domain.
- [ ] **Per-level conjugation regularizers in the ELBO loss.** The class exposes `lower_recognition_conjugation_loss_at` and `upper_recognition_conjugation_loss_at` but doesn't add them to `elbo_at` automatically. To replicate the bivariate `--conj-weight` pattern from `train.py`, the caller adds them externally. **Decision**: should the class provide a `regularized_elbo_at(λ_Y, λ_Z)` convenience method, or keep regularization purely caller-side?
- [ ] **Diagnostics.** `iwae_bound`, `estimate_amortization_gap`, `compute_conjugation_metrics`, `regress_conjugation_parameters` in `examples/variational_mnist/model.py` and `variational.py` all assume the bivariate interface. None work for the deep model out of the box. Of these, IWAE is the most useful and would port cleanly (same recipe with `Upper.sample` and the joint `log_density` instead of `pst_man.sample` / `pst_man.log_density`).
