# REVIEW-TODO

Pending hand-review checklists for in-flight PRs. Each section is a self-contained review pass — move through items top-to-bottom within a section.

## Topic 1 — manuscript-aligned variational rewrite

**Status: review complete 2026-06-11 — all items ticked; skip unless revisiting.**

**Status:** already pushed to `origin/main`; this checklist tracks hand-review of that earlier PR, not anything in the currently-unpushed commits. The `a11f2ab` parking commit trimmed away the hierarchical-extension bullets that used to live in this section.

Tracking checklist for hand-review of the `variational.py` migration to manuscript §4.2 standard form. Plan: `/home/alex404/.claude-work/plans/alright-so-exponential-family-variationa-zany-balloon.md`.

> **Note on drift (added 2026-06-10).** Commits `a6d2ee5`/`dfac625` reworked `variational.py` *after* this checklist was written: the single class is now a hierarchy `VariationalConjugated` (Generative-only base with a new full-$f$ score-function `elbo_at` — new code not tracked by any item below) → `VariationalDifferentiable` (the $c(x) + \mathbb{E}_q[r]$ form, `elbo_divergence`) → `VariationalSymmetric` (the class the graphical section calls `SymmetricVariationalConjugated`). `conjugation_residual`/`conjugation_parameters` gained an optional `x` argument (input-dependent $\rho_Z(x)$ hook for manuscript §4.2.7). All L## references below are pre-rework; navigate by name. Items whose *descriptions* are stale carry inline notes.
>
> **Update 2026-06-11.** The base full-$f$ `elbo_at` is gone again: it was sample-for-sample identical to the standard form (the sample-mean baseline absorbs $c(x)$ exactly — verified by quadrature), unreachable by any concrete model, and duck-typed `log_density` on `Generative`-bounded latents. `elbo_at`/`mean_elbo`/`recognition_conjugation_loss_at`/`mean_recognition_conjugation_loss` now live on `VariationalDifferentiable`; the base keeps structure, sampling, `conjugation_residual`, and `prior_conjugation_loss`. New `tests/variational.py` (8 tests) verifies every estimator's value *and* gradient against exact quadrature ground truth on a 1-latent VonMises model, pinning the stop_gradient policy (mutation-tested: un-sg'ing $r$, dropping the score correction, leaking the score into the value, and sg'ing $\log q$ each break the suite). The stop_gradient audit item below is superseded by these tests.

### Automated verification (already run)

- `uvx basedpyright src/` → 0 errors, 0 warnings.
- `uvx basedpyright examples/variational_mnist/train.py examples/pendulum/run.py examples/torus_poisson/run.py` → 0 errors.
- `uv run python -m pytest tests/` → **270/270 passed** in 393s. Notable suites exercised: `tests/dynamical.py` (21 tests including `VariationalLatentProcess` filter/ELBO/SGD-monotonicity and `conjugation_metrics`/`regress_conjugation_parameters` round-trip), `tests/population_codes.py` (14 tests, all variational machinery).
- `uv run sphinx-build -W docs/source` → build succeeded, no new warnings.
- Smoke-run examples (all exit 0):
  - `examples/variational_mnist` (gradient mode, 30 steps, KL warmup 10): ELBO=-2763.67 → -2751.14, β-warmup engaged at step 0 (β=0.000), training stable.
  - `examples/pendulum` (LEARNABLE rho + REGULARIZED, 10 chunks each): both modes complete; regularized Var[RLS] drops 2.17 → 0.79 over training.
  - `examples/torus_poisson` (free + regularized + analytical, 4000 steps each): all three modes complete and produce sensible R² progressions. Final R² = 0.22 (free), 0.60 (regularized), 0.99 (analytical) — the analytical mode achieves near-perfect linear conjugation as expected.

### Mathematical correctness — sanity-check against manuscript §4.2

- [x] **`conjugation_residual` formula matches manuscript Eq.~43.**
  - Code: `src/goal/geometry/exponential_family/variational.py:154-171`
  - Manuscript: $r(z) = \rho_Z \cdot s_Z(z) - \psi_X(\theta_X + \Theta_{XZ}\cdot s_Z(z)) + \psi_X(\theta_X)$
  - The new $+\psi_X(\theta_X)$ shift is the only behavioral change versus the old `reduced_learning_signal`. Variance / covariance / regression-with-intercept consumers are shift-invariant — the rename should be observably a no-op for them.
- [x] **`elbo_at` decomposition $c(x) + \mathbb{E}_q[r(Z)]$ matches manuscript Eq.~43–44.**
  - Code: `src/goal/geometry/exponential_family/variational.py:173-227`
  - $c(x)$ in code = `s_X·θ_X + ψ_Z(θ̂_{Z|X}) − ψ_Z(θ_Z) − ψ_X(θ_X) + log h_X(x)`. Derivation: expand $\log p(x|z) + \log p(z) - \log q(z|x)$, cancel shared $s_Z·θ_Z$ and $s_X·Θ·s_Z$ terms, separate $z$-dependent ($r$) from $x$-dependent ($c$) pieces. The $±ψ_X(θ_X)$ is a convention to make $r$ vanish exactly at conjugation.
  - *Claude re-verify 2026-06-10 (corrected same day): both are right — the manuscript defines densities wrt the base measure ($q_X = dQ_X/d\mu_X$, Eq. 1), so its $c(x)$ correctly has no $\log h_X$ term; the code reports densities wrt counting/Lebesgue measure, so its $+\log h_X(x)$ is the convention translation, not an erratum. Gradients identical; ELBO values differ by the constant $\log \mu_X(x)$.*
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

### `src/goal/geometry/exponential_family/variational.py` (main rewrite)

> Ticked 2026-06-11: covered in the review session — every estimator re-derived against the manuscript and verified against quadrature ground truth (`tests/variational.py`), all docstrings rewritten per the new module-as-flow-document design (new text reviewable via the working-tree diff), orphan greps clean. Manuscript references removed; docs are standalone.

- [x] **Module docstring** (L1-9): manuscript §4.2 reference, $r$ as central object, β handled externally. *(Rewritten: now displays the conjugation equation, the recognition model, the residual, and the $f = c + r$ split, and names the "standard form"; manuscript reference removed.)*
- [x] **`VariationalConjugated` class docstring** (L25-58): unchanged from previous version except surrounding context. Worth a glance to confirm it still reads coherently next to the new method order. *(Rewritten: model definition + type-parameter roles; ELBO machinery moved out per the 2026-06-11 consolidation.)*
- [x] **`conjugation_residual` method** (L154-171, replaces `reduced_learning_signal`):
  - Renamed in code, added `+ self.obs_man.log_partition_function(obs_params)` shift.
  - Compare to old body (in git: `git show HEAD:src/goal/geometry/exponential_family/variational.py | sed -n '246,259p'`). Only difference: the added shift term.
- [x] **`elbo_at` body** (L173-227): full rewrite. The surrogate `c_x + direct + score - sg(score)` is value-correct as an MC estimate and gradient-correct under autodiff. Three stop-gradients (samples, $r$ inside score correction, `sg(score)`) documented in the method docstring. *(Quadrature-verified value + gradient; mutation-tested.)*
- [x] **`mean_elbo`** (L229-239): no `kl_weight` parameter; otherwise unchanged. *(Moved to `VariationalDifferentiable`.)*
- [x] **`elbo_divergence`** (L241-249): unchanged body; docstring updated to note it's no longer on the critical path of `elbo_at` but is the lever for external β-warmup. *(Argument order verified = KL(q‖p); quadrature-tested.)*
- [x] **Removed: `elbo_reconstruction_term`.** Was unused outside `elbo_at`; grep before/after to confirm no orphan reference. *(Grep clean across src/examples/tests/docs.)*
- [x] **`prior_conjugation_loss`** (L253-263): $\mathcal{R}_p = \mathrm{Var}_{p_Z}[r(Z)]$. Sg on prior samples; gradient flows through `params` in `conjugation_residual` at fixed $z$. Mirrors the inline pattern callers were already using.
- [x] **`recognition_conjugation_loss_at`** (L265-275): $\mathcal{R}_q(x) = \mathrm{Var}_{q(Z|X=x)}[r(Z)]$. Same sg policy as the prior version, but samples from the recognition model. New to the codebase — not yet called anywhere.
  - *Stale description (post-`a6d2ee5`): no longer the same sg policy as the prior version. Now delegates to the new `_variance_with_score_correction` helper, which adds the score-function term $\mathbb{E}_q[(r-\bar r)^2 \nabla \log q]$ — more manuscript-faithful (§4.2.6 says $\mathcal{R}_q$ "relies on the same score-function gradient machinery as the ELBO"), but review the helper too. Still uncalled (verified by grep 2026-06-10).*
- [x] **`mean_recognition_conjugation_loss`** (L277-285): batch mean of the per-$x$ recognition loss. Mirrors `mean_elbo` convention. *(Moved to `VariationalDifferentiable`.)*
- [x] **`conjugation_metrics`** (free function, L367-396): refactored to delegate to `prior_conjugation_loss` for `var_f`; keeps inline $\mathrm{Var}_p[\psi_X]$ for the $R^2$ ratio. Public signature `(var_f, std_f, r_squared)` preserved; all six callers untouched.
- [x] **`regress_conjugation_parameters`** (L309-365): body unchanged. Docstring updated to mention the shift-invariance of the intercept-absorbed regression.
  - *Stale description (post-`a6d2ee5`): body NOT unchanged anymore. The design matrix is now built by linearizing $\iota(\rho)\cdot\phi(s_Z)$ through `conjugation_parameters` via `jax.grad` at $\rho = 0$, with a $\rho$-independent offset subtracted from the target — so the same regression works for richer completions (mixture completion etc.). Needs a fresh read, not a rubber stamp.*
- [x] **`reconstruct` / `reconstruction_error`** (L399-425): unchanged.
- [x] **`stop_gradient` audit (manuscript-correctness).** Five sg sites total, all annotated in code:
  - *Stale count (post-`a6d2ee5`): more than five sites now — the base-class `elbo_at` (samples, `f_sg`, `sg(score)`) and `_variance_with_score_correction` (`r_sg`, `g_sg`, `sg(score)`) add their own. Each is annotated in code; audit by reading each `stop_gradient` call site rather than the list below.*
  1. `regress_conjugation_parameters` — sampler may be non-differentiable.
  2. `elbo_at` z-samples — same reason.
  3. `elbo_at` `r_sg = sg(r_vals)` — prevents double-counting direct gradient in score correction.
  4. `elbo_at` `score - sg(score)` — keeps value MC-clean while preserving the gradient.
  5. `prior_conjugation_loss` / `recognition_conjugation_loss_at` samples — same as #1, plus the variance-as-regularizer convention that gradient flows only through $r(\cdot;\theta)$ at fixed $z$ (matches what pendulum/torus_poisson were doing inline).

### `src/goal/geometry/exponential_family/dynamical.py`

- [x] **`VariationalLatentProcess.filter`** (L438-475): `kl_weight` parameter removed; docstring updated to point at external β-warmup via `ems_hrm.elbo_divergence`. Pendulum is the sole caller and uses defaults — `test_filter_shapes` and `test_grad_descent_reduces_loss` in `tests/dynamical.py` exercise this and pass. *(Reviewed 2026-06-11. Soft point: the docstring's β-warmup recipe needs per-step predicted priors, which `filter` doesn't return — a caller would re-walk the chain. Fine until a real caller wants it.)*
- [x] **`VariationalLatentProcess.mean_elbo`** (L477-491): `kl_weight` removed; same notes. *(Reviewed 2026-06-11; signature has no defaults — pendulum passes all four args positionally.)*

### `src/goal/models/graphical/variational.py` (no code edits — review for cascade-correctness)

- [x] **`VariationalHierarchicalMixture`** inherits all ELBO machinery from `SymmetricVariationalConjugated`. The new `elbo_at`/`conjugation_residual`/`*_conjugation_loss` methods are inherited unchanged. *(Reviewed 2026-06-11: shape cascade traced (cnj_man dim 8 vs mixture prior dim 26, zero-pad bridges); manual check run on `ConcreteVariationalHierarchicalMixture` — Var[r]=0.0033 at random init, Var[r]→0/R²→1 after `regress_conjugation_parameters`, and r ≡ 0 exactly with interaction+rho zeroed.)*
  - Its override of `conjugation_parameters` returns a Prior-shaped $\rho$ via `ObservableEmbedding(mix_man).embed(rho)` — this composes correctly with `conjugation_residual` (which consumes `pst_prr_emb.embed(s_z) ⋅ conjugation_parameters(params)` for the $\rho$ term).
  - `tests/hmog.py` (22 tests) exercises the analytic hierarchical path; the variational path is exercised end-to-end via `examples/variational_mnist`. Both pass.
  - Suggested manual check: pick one concrete instance (e.g., `BinomialHierarchicalMixture` in the mnist example), call `model.conjugation_residual(params, z)` and `model.prior_conjugation_loss(key, params, n_samples)` interactively, and confirm the values look sensible (Var[r] is non-trivial when conjugation is imperfect, near zero when perfect). *(Codified 2026-06-11 as `TestVariationalHierarchicalMixture` in `tests/variational.py` — 3 tests: zero-pad structure of `conjugation_parameters`, r ≡ 0 at exact conjugation, regression beats rho=0 on the prior loss.)*

### Callers (renames only, no semantic change)

- [x] **`examples/pendulum/run.py:397`**: `reduced_learning_signal` → `conjugation_residual` inside the regularized-mode loss function. Verified by smoke run. *(Re-verified 2026-06-11: now at run.py:422 inside `loss_fn_reg`; zero orphan references to the old name across src/examples/tests/docs.)*
- [x] **`examples/torus_poisson/run.py:308, 337`**: same rename in both `loss_fn_regularized` and `loss_fn_analytical`. Verified by smoke run across all three modes. *(Re-verified 2026-06-11: line numbers still accurate.)*

### `examples/variational_mnist/train.py` (β-warmup externalized — most invasive caller change)

- [x] **`loss_fn_gradient`** (L348-389): `mean_elbo` called without `kl_weight`; `elbo_divergence` vmapped over the batch for `mean_kl`; `elbo = elbo_1 + (1.0 - beta) * mean_kl`. Logged `elbo` is the β-scaled value, matching prior semantics. *(Formula verified against manuscript §4.2.1, 2026-06-10. Amended 2026-06-11: metrics now report the unscaled `elbo_1` — see the sign-convention item.)*
- [x] **`loss_fn_analytical`** (L392-443): same pattern, with `params_with_rho` (not `params`) in both `mean_elbo` and the vmapped `elbo_divergence`. Verify the `params` variable used in both calls is the same. *(Verified: `params_with_rho` in both calls, 2026-06-10.)*
- [x] Cross-check: at $\beta = 1$, the new code should yield exactly the same `elbo` as the old code (within MC noise). Quick local test: snapshot the elbo trajectory before and after, with a fixed seed and `--kl-warmup-steps 0` (so β=1 from step 0). *(Done 2026-06-11, stronger form: rebuilt the old recon−KL estimator from current APIs and compared against `elbo_at` on the full-mixture mnist model (n_latent=16, 64 MC samples, 20 seeds) — means differ by 0.0055 ± 0.0060 on an ELBO of −3847, i.e. indistinguishable. At β=1 train.py reduces to plain `mean_elbo`, so this is the whole check. Side-finding: new-form value std is ~4× the old (0.026 vs 0.007) — see variance-regression section.)*
- [x] Sign convention: $\beta < 1$ during warmup should *increase* `elbo` (looser KL penalty). The smoke run shows step 0 with β=0.000, which makes `elbo = elbo_1 + 1.0*KL = E_q[log p(x|z)]` — confirm this matches intent. *(Resolved 2026-06-11: math confirmed (KL ≥ 0 so β<1 strictly inflates the scaled value), and per review decision the logging was changed — both loss fns now report the true `elbo_1` in metrics while the β-scaled `elbo` lives only inside the loss. Smoke-verified, 30 steps.)*

### `docs/source/geometry/exponential_family/variational.rst`

- [x] **No edits required.** The `autoclass :members:` directive auto-includes the new methods (`conjugation_residual`, `prior_conjugation_loss`, `recognition_conjugation_loss_at`, `mean_recognition_conjugation_loss`). Sphinx build succeeded with no warnings. Worth a render check: open `docs/build/geometry/exponential_family/variational.html` and confirm the new methods appear in the class docs with their docstrings. *(Render check done 2026-06-11: all nine method anchors present in the built HTML, standard-form prose renders.)*

### Variance regression (the main empirical risk)

- [x] The plan explicitly accepts a small gradient-variance increase: the old recon+KL form Rao-Blackwellized the $\rho \cdot s_Z$ piece via the closed-form KL, while the new standard form estimates it via MC (bounded by $|\rho|^2 \cdot \mathrm{Var}_q[s_Z]/K$). *(Quantified 2026-06-11: direct estimator comparison on the mnist full-mixture model shows ~4× value-noise (std 0.026 vs 0.007 over 20 seeds, 64 MC samples) — relative noise ~7e-6 of the ELBO magnitude.)*
- [x] **Compare training trajectories** on a representative example (torus_poisson with fixed seed is the cleanest test — three modes, well-instrumented). Run with the previous commit on `main` once for baseline, then with `HEAD`. Expectation: ELBO and conjugation-variance trajectories track within MC noise. If they diverge meaningfully, the followup is to Rao-Blackwellize $\rho \cdot s_Z$ in `elbo_at` (recovers the old form's variance without giving up the residual API). *(Done 2026-06-11: baseline = `a6d2ee5~1` (last commit with the old `elbo_reconstruction_term` form) in a worktree vs current tree, full 4000-step runs, all three modes, seed 42. free/regularized: pointwise near-identical (ELBO RMS diff ~0.01 vs per-step noise 0.72). analytical: transient mid-run divergence (RMS 2.15) but identical endpoints — ELBO −115.88 both, R² 0.972 vs 0.974. Baseline data: `/tmp/torus_baseline_analysis.json`, logs `/tmp/torus_{baseline,head}.log`.)*
- [x] If variance is observably worse and matters for training stability, consider raising `n_mc_samples` (cheap workaround) or implementing the Rao-Blackwellized variant (proper fix). *(Moot — no observable trajectory degradation; no action.)*

### Documentation / docstring consistency

- [x] `elbo_divergence` docstring positions itself as "not on the critical path" — confirm this matches your mental model. It's the public API for callers wanting β-VAE and for KL diagnostics, and that's its sole reason for existing post-rewrite. *(Docstring rewritten in the 2026-06-10/11 pass with exactly this positioning.)*

### Followups deferred from this PR (per plan)

- Rao-Blackwellize $\rho \cdot s_Z$ in `elbo_at` if the MC variance shows up empirically.
- Optional: lift the inline `prior_conjugation_loss` pattern out of `examples/pendulum/run.py` and `examples/torus_poisson/run.py` to call the new method directly. Pure cleanup; no behavior change.

## Topic 2 — chordal & chain Boltzmann machines

**Status: review complete 2026-06-11 — all items closed (hand-review + test certification); skip unless revisiting. Working tree not yet committed at close-out.**

**Status:** substantially redesigned live during the 2026-06-11 review session; this checklist describes the **current working tree**. The pre-redesign checklists (and the original AI-gen code they tracked) are in git history: `git show HEAD:REVIEW-TODO.md`. Original plan: `/home/alex404/.claude-work/plans/cozy-crafting-brooks.md` (historical).

**Session changelog** (each round discussed and approved in-session):

1. `junction_tree.py` rewritten: 4 identity fields, one inlined `from_edges` (vectorized incremental min-fill, co-occurrence MST), plain uncached properties, disconnected-graph bug fixed, `n_neurons` → `n_nodes`. Build 25x faster at n=4096.
2. `gaussian/chordal/` subpackage (`junction_tree.py` + `sum_product.py`, both public); `boltzmann.py` deduplicated via abstract `Boltzmann[Shape]` with a `split_couplings`/`join_couplings` layout contract; concrete dense machine renamed `Boltzmann` → **`FullBoltzmann`**.
3. `ChainTree` / `ChainCouplingMatrix` / `ChainBoltzmann` reify the path-decomposition case (validation at construction, no silent dispatch); `chain_log_partition` via associative-scan transfer matrices — grad(logZ) 36.9 → 0.28 ms at n=1024.
4. `chain_sample`: parallel forward-filtering backward-sampling via random-map composition — posterior sampling 49 → 13 ms; **full mnist training step 516 → 104 ms (~5x vs Chordal)**, now observable-bound.
5. Package restructure (post-review round): `chordal/` → `coupling/`, now kernels-only — `dense.py` (enumeration + Gibbs free functions, extracted from `CouplingMatrix`), `junction_tree.py`, `sum_product.py`. ALL classes (models + the three coupling-matrix shape manifolds, each a thin wrapper over the kernels) live in `boltzmann.py`. Public API unchanged; ticked items above reference the old `chordal/` paths. Gate: 91/91 boltzmann+lgm, ruff/pyright/sphinx -W clean.

### Automated verification (already run)

- `uv run python -m pytest tests/` → **321/321** (full-suite rerun 2026-06-11 after the `from_edges` split, `jt_sample` rewrite, and categorical simplification; the chain-sampling test added afterwards brings the expected total to 322 — `tests/boltzmann.py` reran green, now at 69/69). `tests/boltzmann.py` (69 tests): logZ and sampling vs brute-force enumeration over 10+ topologies (chains, cycles, bands, K4, irregular, isolated nodes, two components, mixed separator sizes, hub), Chain-vs-Chordal equality in value *and gradient*, sampling pair statistics vs enumeration (caught a real reverse-scan orientation bug during development), split/join round-trips, JIT composition.
- `uvx basedpyright src/` → 0 errors 0 warnings; `uvx ruff check src/` clean; sphinx builds clean with the new `chordal/` pages.
- mnist chain smoke: 200 steps end-to-end, exit 0, NMI 0.4744 (pre-rewrite reference: 0.4814).
- Benchmarks: `/tmp/bench_current.txt` (baseline), `/tmp/bench_integrated.txt`, `/tmp/bench_mnist_chain.py` output in conversation.

### Core arguments worth hand-checking first (math, not code)

- [x] **Incremental min-fill update.** Fill counts are recomputed only for `touched` vertices after eliminating $v$: claim — $\mathrm{fill}(w)$ changes only if $w$ lost $v$ as a neighbour or an edge was added within $N(w)$ (i.e. $w$ adjacent to both endpoints of a fill-in). `junction_tree.py::from_edges`, triangulation section.
- [x] **Maximal-clique criterion.** Candidate $K_v = \{v\} \cup \{\text{later nbrs}\}$ is non-maximal iff contained in an *earlier-eliminated, kept* candidate containing $v$. Proof sketch: a maximal $C \supseteq K_v$ equals $K_u$ for $u$ = its earliest-eliminated member; $u \neq v$ (else $C = K_v$); $K_u$ maximal hence never dropped.
- [x] **MST + forest links.** Max-weight spanning tree of the clique graph satisfies running intersection (standard); zero-weight empty-separator links between components make sum-product multiply per-component partition functions and sampling treat them independently.
- [x] **Transfer-matrix logZ** (`chain_log_partition`): *(closed 2026-06-11: clamp justified in-session — value-exact, no policy knob, restores finite true gradients; convention-compliance now documented at the `_LOG_ZERO` definition; pinned vs enumeration + Chain≡Chordal value/grad tests.)* collect along a path = product of $2^{|S|} \times 2^{|S|}$ matrices in the log-semiring; `_LOG_ZERO = -1e30` clamp keeps gradients NaN-free through genuinely empty separator-state segments (heterogeneous separators) without changing values.
- [x] **`chain_sample` exactness — the subtlest new math.** *(2026-06-11: hand-walking this is now optional — `test_sampling_matches_joint_distribution` compares empirical frequencies against exact probabilities over ALL 2^n states (3 topologies: mixed seps, 4-cycle, plain chain), so the entire joint is certified, not just moments; nothing distributional is left for a bug to hide behind. The docstring argument remains worth reading for understanding, not for verification.)* (a) FFBS telescoping: conditionals $\propto \phi_k \cdot \beta_k(\text{out})$ masked to the incoming separator state multiply to the joint. (b) Shared-Gumbel random maps: one Gumbel vector per clique resolves the conditional draw for *every* possible incoming separator state, defining $F_k : s_{k-1} \mapsto s_k$; Gumbels independent across cliques and the realized chain consumes $F_k$ at a point independent of clique $k$'s Gumbels, so composing the maps (associative) realizes the joint exactly.
- [x] **Scan orientations.** Forward `associative_scan` feeds the lower-index product as the combine's first argument; `reverse=True` feeds the *higher*-index product. Both combines therefore multiply the new element on the left. The reverse case was initially wrong — caught by the band-2/chain-16 pair-statistics tests, now pinned (and further by the full-joint histogram tests added 2026-06-11).

### `src/goal/models/base/gaussian/chordal/junction_tree.py` (~360 lines)

- [x] Module + `JunctionTree` docstrings: 4 identity fields = `__eq__`/`__hash__`/JIT cache key; raw constructor does no validation; everything else derived on demand (uncached — recompute is cheap, hoist out of loops). *(Amended mid-review 2026-06-11: opening expanded with junction-tree context; new "seed, not preserved topology" paragraph — the model is the chordal completion, fill-ins become real couplings; binary qualifier on the cost statement; stale `_jt_inference` reference fixed.)*
- [x] `_triangulate` (was the inline triangulation section; split out with `_maximal_cliques`/`_spanning_tree` to retire the `noqa: C901`): `fill_in` closure over the zeroed-row elimination graph; `touched` superset maintenance; `argmin` deterministic lowest-index tie-break; **new**: out-of-range endpoint `ValueError` (numpy negative-index wraparound would otherwise build a wrong graph silently).
- [x] `_maximal_cliques`: `node_cliques` index makes the containment check $O(\text{cliques} \ni v)$.
- [x] `_spanning_tree`: separator weights via per-node clique co-occurrence counts; Kruskal; forest linking (the disconnected fix). `from_edges` is now a short orchestrator: validate $n$ → triangulate → cliques → treewidth gate → MST; its docstring states the seed semantics, determinism, and that `max_treewidth` bounds the *heuristic* triangulation, not the true treewidth.
- [x] Derived properties: `chordal_edges` canonical sorted order **is the off-diagonal parameter layout**; `pre_order`/`parent_of`/`collect_order` mutual consistency; `bias_owner`/`edge_owner` first-clique attribution.
- [x] `ChainTree`: `__post_init__` degree-2 validation (runs on *any* construction), `clique_path`, docstring honesty (path decomposition / pathwidth / interval graphs; PQ-tree recognition gap — chains and bands always recognized). *(Amended 2026-06-11: comment now states the check assumes the spanning-tree field invariant — for a tree, degree ≤ 2 ⟺ path; raw-construction validation is out of scope, as in the base class.)*

### `src/goal/models/base/gaussian/chordal/sum_product.py` (~470 lines)

- [x] Module docstring: four public functions; numpy tables built at trace time ("the compilation cache is the only cache"); LSB-first encoding stated once. *(Amended 2026-06-11 after external-review triage: "compile time constant in n_cliques" corrected to "not unrolled over cliques" — table sizes do grow compile time; speedup claim anchored to the measured ~100x figure.)*
- [x] `_sep_state_map`: padded states map to separator state 0 (never an all-padded segment); `_segment_logsumexp` $-\infty$ absorption.
- [x] `_ownership_tables` + `_clique_potentials`: sentinel `-1` + leading-zero-padded params gather trick.
- [x] `_collect`: sequential leaves-to-root scan (the general-tree kernel; also feeds `jt_sample`).
- [x] `chain_log_partition`: prefix scan; K=1 and K=2 short-circuits. *(Joint-histogram + equality tests certify; K=1/K=2 short-circuits separately parametrized.)*
- [x] `chain_sample`: suffix scan for $\beta$; conditional-logits tensor (clique 0's zero in-map row makes $t=0$ the unconditional draw; last clique's zero $\beta$ row); Gumbel argmax over `(K, S, 2^m)`; map-composition scan; sentinel-row scatter (overlapping writes agree by separator consistency).
- [x] `jt_sample`: collect + pre-order walk-down (general-tree sampler, still used by `ChordalCouplingMatrix`). *(Rewritten 2026-06-11: the dynamic `set_mask` + `fori_loop` writes replaced by static per-clique new-node tables — by running intersection, the already-assigned nodes at each clique are exactly the parent separator, so each node is written exactly once by its first pre-order clique (sentinel-row scatter, as in `chain_sample`). Hand-check the RIP argument in the docstring; 63/63 tests incl. hub/branching + disconnected sampling-vs-enumeration.)*

### `src/goal/models/base/gaussian/boltzmann.py` (~640 lines)

- [x] **`Boltzmann[Shape]` ABC**: *(closed by test certification — energy-identity test pins the convention; hand re-derivation waived per 2026-06-11 close-out.)* the $x^2 = x$ doctrine stated once (absorbed bias, zero location, first moment = diagonal of second, $-2/-1$ precision convention); the four GeneralizedGaussian conversions expressed through `split_couplings`/`join_couplings`. Equivalence to the old per-class code is round-trip-tested; worth one hand re-derivation of `join_location_precision`. *(2026-06-11: the sign/factor-of-two convention is now also pinned semantically — `test_location_precision_energy_identity` (Diagonal/Full/Chordal) asserts theta·s(x) = loc·x − ½xᵀΛx on binary states with the quadratic form derived independently of the conversion under test; a consistent convention error can no longer survive. Hand re-derivation now optional.)*
- [x] Layout contracts: *(round-trip + energy-identity tested across all three layouts.)* `DiagonalBoltzmann` (identity / empty off-diag), `FullBoltzmann` (packed-triu via static numpy indices), `ChordalBoltzmann` (slice/concat).
- [x] `FullBoltzmann` rename fallout: *(count-asserted replacements; full suite + pyright green.)* `lgm.py` (`BoltzmannEmbedding`, `BoltzmannLGM` internals — 12 sites), `models/__init__.py` exports both `Boltzmann` (ABC) and `FullBoltzmann`, `examples/boltzmann/run.py`.
- [x] `CouplingMatrix` / `ChordalCouplingMatrix` bodies — provenance corrected 2026-06-11: `CouplingMatrix` is Sacha's own pre-chordal code (commit `4849946`), long vetted; only `ChordalCouplingMatrix` is from the AI PR (`f40c382`). Both lightly touched (2026-06-11, external-review round): `_gibbs_step` splits the permutation key from the per-unit update keys (was reusing one key for both — hygiene, statistically harmless); `log_partition_function` enumeration rewritten as a dense quadratic-form einsum (2–8x at n=16–20, values agree to 1e-10, pinned by `test_log_partition_matches_sum` + K4 equivalence); `_unit_conditional_energy_diff` comment sharpened with the x_k-cancellation algebra; `ChordalCouplingMatrix.sample` docstring notes the vmap-hoisted collect (verified by XLA cost analysis: 256 draws cost 48x one draw). The last unread AI-gen lines (`ChordalCouplingMatrix.to_matrix`/`from_matrix`) closed by `test_to_from_matrix_round_trip` 2026-06-11: round-trip, symmetrization of asymmetric input, off-pattern dropping — `from_matrix` previously had zero callers anywhere.
- [x] `ChainCouplingMatrix` / `ChainBoltzmann`: narrowed `ChainTree` field, kernel overrides only; Liskov-clean by construction (identical layout and distribution).

### Tests, example, docs

- [x] `tests/boltzmann.py`: `TestJunctionTree` (incl. disconnected linking), `TestChordalBoltzmann` (10-topology enumeration parametrization), `TestChainBoltzmann` (validation, Chain≡Chordal value+grad, sampling vs enumeration incl. pair statistics, K=1/K=2 edge cases). Tolerances per project convention (exact: 1e-5/1e-7; 20k-sample MC: 0.02). *(Strengthened 2026-06-11: new `test_sampling_matches_joint_distribution` for both samplers — empirical frequencies vs exact probabilities over all 2^n states (jt_sample: branching hub + mixed seps; chain_sample: mixed seps, 4-cycle, chain-6), certifying the full joint incl. non-adjacent correlations that moment tests can't see; 50k samples, atol 0.01 ≈ 4.5 sigma. Also `test_sampling_matches_chain_to_mean` now checks full sufficient statistics (pairs included) vs exact to_mean, parametrized over a plain chain (n=16) and a band-2-head/chain-tail topology (n=20, separator sizes 2 and 1) — extends sampler coverage past the enumeration 2^n ceiling and through the `_LOG_ZERO` regime; measured max deviation 0.007 vs atol 0.02.)*
- [x] `examples/variational_mnist` (experimental standards): *(closed 2026-06-11 without hand-review per decision — low-priority experimental code, covered by smoke runs; the git stash "precision_floor in core (superseded)" is dead code and safe to drop at leisure.)* `_build_base_latent` → `ChainBoltzmann` for the chain graph; two assignment computations chunked via `lax.map(batch_size=256)` (OOM fix); `FlooredDiagonalNormal` shim in `model.py` (precision floor relocated out of core `Normal`; drop the git stash "precision_floor in core (superseded)" once reviewed); pre-existing `--optimizer adamw`/`--weight-decay` CLI rode along unreviewed.
- [x] Docs render check: *(waived in favor of the `sphinx -W` gate (warnings-as-errors, green after the coupling restructure) plus the earlier render checks; pages now: `boltzmann.html` (all 8 classes), `coupling/{dense,junction_tree,sum_product}.html`.)* `boltzmann.html` (ABC + Diagonal/Full/Chordal/Chain + both coupling matrices), `chordal/junction_tree.html` (incl. `ChainTree`), `chordal/sum_product.html` (four functions).

### Open follow-ups (deferred)

- **`BoltzmannLGM` widening**: still typed against `FullBoltzmann`; accepting `ChordalBoltzmann`/`ChainBoltzmann` latents needs the bound widened (no example exercises it yet).
- **Stateful-`MatrixRep` / `ChordalSymmetric` design question**: unchanged — revisit if a Gaussian chordal MRF materializes (at which point `junction_tree.py` may move to a neutral home).
- **Graph-representation future (long-range)**: when harmoniums move from recursive nesting to a proper graph representation, `from_edges` is the seam — its body can delegate to a structure library (networkx/rustworkx) while `JunctionTree` stays the frozen/hashable boundary object defining the parameter layout. Inference stays JAX-native.
- **Richer mnist topologies**: band width / user edge lists — kernels already support them, CLI plumbing only.
- **PQ-tree interval-graph recognition**: only if hub-like topologies ever need the chain kernel.
- **Distribute pass / per-clique marginals**: if diagnostics ever want calibrated beliefs.
- **Levelized collect/sample for branching trees**: process tree levels with `vmap` + scatter-add to cut depth from $O(n_\text{cliques})$ to $O(\text{height})$ — only worthwhile if a bushy-tree workload materializes; chains already have their own kernels.
