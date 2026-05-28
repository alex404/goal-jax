# REVIEW-TODO

Pending hand-review checklists for in-flight PRs. Each section is a self-contained review pass — move through items top-to-bottom within a section.

## Topic 1 — manuscript-aligned variational rewrite

**Status:** already pushed to `origin/main`; this checklist tracks hand-review of that earlier PR, not anything in the currently-unpushed commits. The `a11f2ab` parking commit trimmed away the hierarchical-extension bullets that used to live in this section.

Tracking checklist for hand-review of the `variational.py` migration to manuscript §4.2 standard form. Plan: `/home/alex404/.claude-work/plans/alright-so-exponential-family-variationa-zany-balloon.md`.

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

- [ ] **Module docstring** (L1-9): manuscript §4.2 reference, $r$ as central object, β handled externally.
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

### `src/goal/geometry/exponential_family/dynamical.py`

- [ ] **`VariationalLatentProcess.filter`** (L438-475): `kl_weight` parameter removed; docstring updated to point at external β-warmup via `ems_hrm.elbo_divergence`. Pendulum is the sole caller and uses defaults — `test_filter_shapes` and `test_grad_descent_reduces_loss` in `tests/dynamical.py` exercise this and pass.
- [ ] **`VariationalLatentProcess.mean_elbo`** (L477-491): `kl_weight` removed; same notes.

### `src/goal/models/graphical/variational.py` (no code edits — review for cascade-correctness)

- [ ] **`VariationalHierarchicalMixture`** inherits all ELBO machinery from `SymmetricVariationalConjugated`. The new `elbo_at`/`conjugation_residual`/`*_conjugation_loss` methods are inherited unchanged.
  - Its override of `conjugation_parameters` returns a Prior-shaped $\rho$ via `ObservableEmbedding(mix_man).embed(rho)` — this composes correctly with `conjugation_residual` (which consumes `pst_prr_emb.embed(s_z) ⋅ conjugation_parameters(params)` for the $\rho$ term).
  - `tests/hmog.py` (22 tests) exercises the analytic hierarchical path; the variational path is exercised end-to-end via `examples/variational_mnist`. Both pass.
  - Suggested manual check: pick one concrete instance (e.g., `BinomialHierarchicalMixture` in the mnist example), call `model.conjugation_residual(params, z)` and `model.prior_conjugation_loss(key, params, n_samples)` interactively, and confirm the values look sensible (Var[r] is non-trivial when conjugation is imperfect, near zero when perfect).

### Callers (renames only, no semantic change)

- [ ] **`examples/pendulum/run.py:397`**: `reduced_learning_signal` → `conjugation_residual` inside the regularized-mode loss function. Verified by smoke run.
- [ ] **`examples/torus_poisson/run.py:308, 337`**: same rename in both `loss_fn_regularized` and `loss_fn_analytical`. Verified by smoke run across all three modes.

### `examples/variational_mnist/train.py` (β-warmup externalized — most invasive caller change)

- [ ] **`loss_fn_gradient`** (L348-389): `mean_elbo` called without `kl_weight`; `elbo_divergence` vmapped over the batch for `mean_kl`; `elbo = elbo_1 + (1.0 - beta) * mean_kl`. Logged `elbo` is the β-scaled value, matching prior semantics.
- [ ] **`loss_fn_analytical`** (L392-443): same pattern, with `params_with_rho` (not `params`) in both `mean_elbo` and the vmapped `elbo_divergence`. Verify the `params` variable used in both calls is the same.
- [ ] Cross-check: at $\beta = 1$, the new code should yield exactly the same `elbo` as the old code (within MC noise). Quick local test: snapshot the elbo trajectory before and after, with a fixed seed and `--kl-warmup-steps 0` (so β=1 from step 0).
- [ ] Sign convention: $\beta < 1$ during warmup should *increase* `elbo` (looser KL penalty). The smoke run shows step 0 with β=0.000, which makes `elbo = elbo_1 + 1.0*KL = E_q[log p(x|z)]` — confirm this matches intent.

### `docs/source/geometry/exponential_family/variational.rst`

- [ ] **No edits required.** The `autoclass :members:` directive auto-includes the new methods (`conjugation_residual`, `prior_conjugation_loss`, `recognition_conjugation_loss_at`, `mean_recognition_conjugation_loss`). Sphinx build succeeded with no warnings. Worth a render check: open `docs/build/geometry/exponential_family/variational.html` and confirm the new methods appear in the class docs with their docstrings.

### Variance regression (the main empirical risk)

- [ ] The plan explicitly accepts a small gradient-variance increase: the old recon+KL form Rao-Blackwellized the $\rho \cdot s_Z$ piece via the closed-form KL, while the new standard form estimates it via MC (bounded by $|\rho|^2 \cdot \mathrm{Var}_q[s_Z]/K$).
- [ ] **Compare training trajectories** on a representative example (torus_poisson with fixed seed is the cleanest test — three modes, well-instrumented). Run with the previous commit on `main` once for baseline, then with `HEAD`. Expectation: ELBO and conjugation-variance trajectories track within MC noise. If they diverge meaningfully, the followup is to Rao-Blackwellize $\rho \cdot s_Z$ in `elbo_at` (recovers the old form's variance without giving up the residual API).
- [ ] If variance is observably worse and matters for training stability, consider raising `n_mc_samples` (cheap workaround) or implementing the Rao-Blackwellized variant (proper fix).

### Documentation / docstring consistency

- [ ] `elbo_divergence` docstring positions itself as "not on the critical path" — confirm this matches your mental model. It's the public API for callers wanting β-VAE and for KL diagnostics, and that's its sole reason for existing post-rewrite.

### Followups deferred from this PR (per plan)

- Rao-Blackwellize $\rho \cdot s_Z$ in `elbo_at` if the MC variance shows up empirically.
- Optional: lift the inline `prior_conjugation_loss` pattern out of `examples/pendulum/run.py` and `examples/torus_poisson/run.py` to call the new method directly. Pure cleanup; no behavior change.

## Topic 2 — tree-width-bounded Boltzmann via junction-tree inference

Tracking checklist for hand-review of the chordal Boltzmann implementation (junction-tree topology + exact sum-product inference + ancestral sampling). Plan: `/home/alex404/.claude-work/plans/cozy-crafting-brooks.md`.

> **Note on line numbers.** The `lax.scan` refactor in Topic 3 moved several helpers and renumbered most functions in `_jt_inference.py` and `junction_tree.py`. The L## references in this section are pre-refactor; treat them as historical pointers and use the file/function names plus Topic 3's notes for current-state navigation. In particular, `_clique_bits` and `_sep_state_map` (originally in `_jt_inference.py`) are now the cached property `JunctionTree.bits_table` and the module-level helper `_build_sep_state_map` respectively, both in `junction_tree.py`.

### Automated verification (already run)

- `uvx basedpyright src/` → 0 errors, 0 warnings.
- `uvx ruff check src/goal/models/base/gaussian/ tests/boltzmann.py` → all checks passed.
- `uv run python -m pytest tests/` → **291/291 passed** in 303s. New tests: `TestJunctionTree` (5 tests) and `TestChordalBoltzmann` (13 tests) in `tests/boltzmann.py`. Smoke-verified end-to-end: `log_partition_function` matches brute-force enumeration to machine precision on chain / 4-cycle / triangulated 5-cycle / irregular sparse 6-node graphs; `ChordalBoltzmann` with a fully-connected (K4) chordal pattern matches the original `Boltzmann.log_partition_function` to numerical precision.

### Mathematical / algorithmic correctness — sanity-check

- [ ] **Min-fill triangulation is correct.** `_min_fill_triangulation` greedily picks the unmarked vertex with fewest fill-in edges and returns the perfect elimination order + completed chordal edge set. Greedy heuristic — not optimal treewidth but standard practice. Sanity unit tests: chain (no fill-in), 4-cycle (adds one diagonal), K4 (no fill-in, already chordal). `tests/boltzmann.py:215-258` covers these.
- [ ] **Maximal-clique extraction from PEO.** For each $v$ in PEO, $C_v = \{v\} \cup \{\text{later neighbours of } v\}$ in the chordal graph; non-maximal cliques discarded. The "later neighbours" definition is the standard one — verify the PEO position lookup is correct (`junction_tree.py:87-112`).
- [ ] **Junction tree = maximum-weight spanning tree of the clique graph weighted by separator size.** Standard result for chordal graphs: max-weight spanning tree of the clique graph satisfies the running-intersection property automatically. Kruskal's implementation at `junction_tree.py:115-147`. Not unit-tested explicitly beyond "tree edges count = n_cliques − 1" — worth a glance.
- [ ] **Bias / edge ownership invariants.** Each bias parameter and each chordal-edge coupling is attributed to *exactly one* clique that contains it (first clique in canonical order). Required so that the energy is summed exactly once across the JT. `tests/boltzmann.py:248-258` (`test_each_param_owned_once`) checks both invariants.
- [ ] **Collect pass formula** at `_jt_inference.py:86-110`: $m_{C \to \text{parent}}(x_S) = \log \sum_{x_{C \setminus S}} \exp(\phi_C(x_C) + \text{incoming})$. Implemented via `_segment_logsumexp` over the precomputed clique-state → separator-state mapping. The mapping is built from bit-extraction in `_sep_state_map` — sanity check the LSB-first convention is consistent between clique-state encoding and separator-state encoding.
- [ ] **`log_partition_function` matches brute-force enumeration.** `tests/boltzmann.py:288-297` parametrises over chain / 4-cycle / 5-cycle / irregular sparse with `rtol=1e-5, atol=1e-7`. The K4 case is also checked against the original `Boltzmann.log_partition_function` to verify the layout conversion is correct.
- [ ] **Ancestral sampling is exact.** `_jt_inference.py:137-180` walks the rooted JT in pre-order, sampling each clique conditional on the separator with its parent (which has already been sampled). For the root, the full $2^{|C_\text{root}|}$ categorical is used; for child cliques, logits are masked to states consistent with the already-drawn separator values via `jnp.where(mask, logits, -jnp.inf)`. Empirical-vs-exact first-moment test at `tests/boltzmann.py:340-352` (`test_exact_sampling_matches_enumeration`) with 20k samples on a 5-node triangulated cycle, `atol=0.02`.
- [ ] **`to_mean` via autodiff matches empirical sampling.** Standard exponential-family identity $\eta = \nabla\psi(\theta)$. `tests/boltzmann.py:329-338` verifies on a 4-cycle, 20k samples, `atol=0.03`.

### Code-review findings (already addressed in-place)

- [ ] **Dead-code cleanup in `junction_tree.py::owned_edge_arrays`.** Removed an unused `edge_to_idx = self.edge_param_index` assignment and the dangling `del edge_to_idx  # only used to assert presence` at the end of the cached property. No behaviour change; the cached property still returns the same arrays. Worth a glance to confirm I read the intent correctly.
- [ ] **`tests/boltzmann.py::_bf_log_partition` return annotation.** The helper returns `(log_z, all_states, energies)`; the original `-> jnp.ndarray` annotation was wrong. Corrected to `tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]` and a one-line docstring addition explaining the tuple shape.

### Architectural deviation from plan — please flag

- [ ] **`ChordalSymmetric` `MatrixRep` was skipped.** Rationale: the existing `MatrixRep` system (`src/goal/geometry/manifold/matrix.py`) is built on stateless `@classmethod`s — instances are stateless and compared by type. A `MatrixRep` subclass holding a `JunctionTree` would need instance state, which would have to override classmethods with instance methods (works in Python but violates LSP and confuses basedpyright). Instead, `ChordalCouplingMatrix` holds the `JunctionTree` directly as a dataclass field and implements `to_matrix` / `from_matrix` / `sufficient_statistic` itself. The Gaussian-MRF reuse case the plan called out as future work remains open — see the open follow-ups section. Worth flagging if the user wants the MatrixRep treatment in this PR rather than later.

### `src/goal/models/base/gaussian/junction_tree.py` (new file, ~280 lines pure Python)

- [ ] **Module docstring** (L1-19): min-fill triangulation, maximal cliques, max-weight spanning tree, frozen hashable result. No JAX dependency, no networkx.
- [ ] **`_min_fill_triangulation`** (L31-85): standard greedy min-fill. Returns `(peo, chordal_edges)` where edges are sorted `(i, j)` tuples with `i < j`. `# noqa: C901` for ruff complexity (15 > 10) — algorithm is inherently nested-loopy.
- [ ] **`_maximal_cliques`** (L87-112): builds candidate cliques from PEO, then filters by strict subset check. Dedup via frozenset. O(n²) in worst case — acceptable since topology is built once.
- [ ] **`_max_spanning_tree`** (L115-147): Kruskal's MST on the clique graph with `weight = |C_i ∩ C_j|`. Sorts candidate edges descending by weight. Stops once `len(edges) == n_cliques - 1`.
- [ ] **`_root_and_collect_order`** (L150-178): roots the JT, computes `parent_of` array, and builds a post-order traversal for the collect pass (leaves before parents). Uses an iterative DFS — no recursion-depth concerns.
- [ ] **`JunctionTree` dataclass fields** (L186-225): everything stored as hashable tuples (no JAX arrays, no dicts). `chordal_edge_index` provides the canonical `(i, j) → param_idx` mapping needed by `_clique_potential`. The cached property `chordal_edges_arr` converts to a numpy `int32` array on demand for vectorised gather operations in `ChordalCouplingMatrix`.
- [ ] **`from_edges`** (L229-289): the user-facing constructor. Edge format is undirected `(i, j)` pairs; self-loops are ignored. The `max_treewidth` parameter raises `ValueError` when the triangulation exceeds the bound — this is the intended failure mode the plan called out (user can then either accept fill-in or modify the graph).
- [ ] **Edge case — disconnected graph with isolated nodes.** Handled at L249-251: isolated nodes become their own size-1 cliques. Verify this doesn't break the JT structure for a fully-isolated graph (e.g., `JunctionTree.from_edges(3, [])`).
- [ ] **Hashability invariant.** All fields are tuples / ints, so `JunctionTree` is hashable by default. This matters because it's a frozen dataclass field of `ChordalCouplingMatrix` / `ChordalBoltzmann`, which JAX traces as static topology.

### `src/goal/models/base/gaussian/_jt_inference.py` (new file, ~180 lines)

- [ ] **Module docstring** (L1-10): two public functions, complexity $O(n \cdot 2^{w+1})$, fixed input layout (diag first, off-diag second).
- [ ] **`_clique_bits`** (L25-29, `@functools.cache`): static numpy table mapping clique-state index → binary values for each clique-local position. Cached on `k` only.
- [ ] **`_sep_state_map`** (L32-45): static numpy array `(2^|C|,)` mapping each clique state to its separator state. Encoding is LSB-first in both clique and separator. **Verify this encoding is consistent throughout** — used in both `_collect` (broadcast messages) and `jt_sample` (build target separator state from sampled values).
- [ ] **`_segment_logsumexp`** (L48-53): standard 3-step logsumexp implementation using `jax.ops.segment_max`/`segment_sum`. Numerically stable by max-shifting per segment.
- [ ] **`_clique_potential`** (L59-83): builds $\phi_C$ as a flat array of length $2^{|C|}$ by iterating over owned biases and owned chordal edges within this clique. Python-loop unrolling — fine because clique sizes are small ($\le 2^w$).
- [ ] **`_collect`** (L86-110): post-order Python loop over `jt.collect_order`. Each step marginalises the child's accumulated total via segment-logsumexp and broadcasts the resulting message into the parent's total. Handles the degenerate empty-separator case (child marginalises to a scalar).
- [ ] **`jt_log_partition`** (L113-118): trivial public wrapper. `logsumexp(totals[root])` after collect.
- [ ] **`jt_sample`** (L137-180): collect + walk-down. The walk-down at each clique:
  1. Get logits from `totals[c]`.
  2. If the clique has a parent and a non-empty separator, mask logits to states consistent with the already-sampled separator values (via `_sep_state_map` comparison + `jnp.where(..., -inf)`).
  3. `jax.random.categorical` over the masked logits.
  4. Decode the sampled state into binary values for clique nodes, but only set nodes not already set (the `set_mask` machinery preserves the parent's separator values exactly).
  - **Subtle**: the `set_mask` is dynamic JAX state but read inside the Python loop. Confirm this trace pattern is what's intended — the bitwise OR/shift on `target` and the `jnp.where(set_mask[v], ...)` are all per-node-in-clique inside a Python for-loop, which JAX unrolls statically.

### `src/goal/models/base/gaussian/boltzmann.py` (modified — appended classes)

- [ ] **New imports** (L1-19): adds `_jt_inference` (free functions) and `junction_tree` (`JunctionTree`). The existing `Boltzmann`, `DiagonalBoltzmann`, `CouplingMatrix`, `BoltzmannEmbedding` are unchanged.
- [ ] **`ChordalCouplingMatrix`** (L390-483, subclass of `Differentiable`): the chordal analog of `CouplingMatrix`, but **not** a subclass of `SquareMap`/`EmbeddedMap` (see the architectural deviation note above). Fields: just `junction_tree`. Param vector layout: `[θ_ii (n,), θ_ij (n_chordal_edges,)]`.
  - `sufficient_statistic(x)` (L420-424): diagonal `x` followed by `x[i] * x[j]` for each chordal edge.
  - `log_partition_function(params)` (L432-434): delegates to `jt_log_partition`.
  - `sample(key, params, n)` (L437-444): vmap over `n` keys of `jt_sample` — confirms i.i.d. exact sampling, no Gibbs.
  - `to_matrix` (L456-468) / `from_matrix` (L470-479): scatter / gather against the chordal sparsity pattern. `from_matrix` symmetrises off-pattern entries via `0.5 * (M[i,j] + M[j,i])`. Off-pattern entries are *dropped* during gather (the projection semantics).
- [ ] **`ChordalBoltzmann`** (L485-585, `GeneralizedGaussian[Bernoullis, ChordalCouplingMatrix] + Differentiable`): mirror of `Boltzmann`. Fields: just `junction_tree`. `loc_man = Bernoullis(n)`, `shp_man = ChordalCouplingMatrix(jt)`.
  - `from_edges` static constructor (L505-512): one-line convenience over `JunctionTree.from_edges`.
  - `split_location_precision` (L549-555): loc = zeros, precision = `[-2 * diag, -off_diag]`. The off-diagonal scaling matches the convention in the original `Boltzmann.split_location_precision` — diagonals scaled by 2, off-diagonals by 1, all negated. Round-trip tested at `tests/boltzmann.py:354-362`.
  - `split_mean_second_moment` (L568-571): first moment is the diagonal slice (since $x_i^2 = x_i$ for binary, the diagonal of the moment matrix *is* $E[x_i]$). Verified at `tests/boltzmann.py:364-373`.
  - **No `to_natural` override** — same status as the original `Boltzmann`. Per memory `feedback-boltzmann-to-natural`, not adding it even for the tractable case.
  - **Not `Analytic`, only `Differentiable`** — `to_mean` comes via autodiff of `log_partition_function`. Matches the original `Boltzmann`.

### Re-exports (`src/goal/models/__init__.py`)

- [ ] Adds `ChordalBoltzmann`, `ChordalCouplingMatrix`, `JunctionTree` to both the `from` import and the `__all__` list. Otherwise unchanged.

### `tests/boltzmann.py` (modified — appended two test classes)

- [ ] **`_bf_log_partition` helper** (L209-225, module-level): brute-force enumeration over all $2^n$ binary states using the chordal-edge layout. Returns `(log_z, all_states, energies)` for reuse across multiple tests.
- [ ] **`TestJunctionTree`** (L215-258, 5 tests): chain, 4-cycle, K4, `max_treewidth` failure, parameter-ownership invariant. Hand-checked topologies — no MC.
- [ ] **`TestChordalBoltzmann`** (L261+, 13 tests): dimensions (parametrised), `log_partition_function` matches enumeration (parametrised over chain / 4-cycle / 5-cycle / irregular sparse), density normalises, K4-pattern matches original `Boltzmann`, `to_mean` via autodiff matches MC, exact sampling matches enumeration, split/join round-trips, JIT composes.
- [ ] **MC tolerances are loose** (`atol=0.02-0.03`) per project convention for sampling-based tests with 20k samples. Match the existing `TestBoltzmann.test_to_mean_via_autodiff` tolerance (0.05) and `TestBoltzmann.test_gibbs_sampling` (0.05). The chordal tests use tighter 0.02-0.03 because JT sampling is *exact* (no autocorrelation), only standard MC variance.

### Documentation

- [ ] **`docs/source/models/base/gaussian/boltzmann.rst`**: added `autoclass` blocks for `ChordalCouplingMatrix` and `ChordalBoltzmann`, plus a one-line cross-reference to the new `junction_tree.rst`. The class hierarchy diagram in this file points at `Differentiable`/`ExponentialFamily` as top classes, which already covers the new chordal classes — no diagram edit needed.
- [ ] **`docs/source/models/base/gaussian/junction_tree.rst`** (new): per project convention (1:1 module-to-RST mapping), `JunctionTree` gets its own page. The `_jt_inference.py` module is intentionally not documented (underscore-prefixed, internal — matches the `manifold/util.py` exception in the docs style guide).
- [ ] **`docs/source/models/base/gaussian/index.rst`**: added `junction_tree` to the toctree.
- [ ] **Render check**: build docs and confirm both new classes appear in `boltzmann.html`, that `junction_tree.html` renders with the dataclass field documentation, and that the cross-reference link works.

### Open follow-ups (deferred from this PR per plan)

- **`ChordalSymmetric` MatrixRep** for Gaussian-MRF reuse. Open question: does the `MatrixRep` interface need a redesign to allow instance state, or should we introduce a parallel `IndexedSymmetric` system? Either way, not blocking this PR.
- **Distribute pass + per-clique calibrated beliefs.** Currently `jt_sample` walks down without an explicit distribute pass — it uses `phi_C + collected_messages` directly, which is correct for sampling but doesn't expose per-clique marginals. If anyone wants per-clique marginal queries (e.g., for diagnostics), add a distribute pass returning `beta_C` per clique.
- **`BoltzmannLGM` widening to accept `ChordalBoltzmann`.** Currently the LGM is typed against `Boltzmann` specifically (`src/goal/models/harmonium/lgm.py:416-461`). A small refactor widens the bound; not done here because no example currently exercises it.
- **Variable-size cliques + `lax.scan`.** Done as part of Topic 3 — see Topic 3's "JT inference refactor" section.

## Topic 3 — Variational MNIST with a ChordalBoltzmann latent prior

Tracking checklist for hand-review of the chordal-Boltzmann integration into `variational_mnist`, including the prerequisite `lax.scan` refactor of `_jt_inference.py`. Plan: `/home/alex404/.claude-work/plans/cozy-crafting-brooks.md` (overwritten — same filename as Topic 2's original plan).

### Automated verification (already run)

- `uvx basedpyright src/ examples/variational_mnist/` → 0 errors, 0 warnings.
- `uvx ruff check src/ examples/variational_mnist/ tests/boltzmann.py` → all checks passed (one pre-existing SIM108 in `diagnose.py` cleaned up as collateral; one pre-existing C901 acknowledged with `# noqa`).
- `uv run python -m pytest tests/boltzmann.py` → all green, including 2 new chain-scale tests at n_neurons = 64.
- `uv run python -m pytest tests/` → see Topic-3-end automated sweep.
- **Smoke run**: `uv run python -m examples.variational_mnist.train --latent chordal_boltzmann --latent-graph chain --n-steps 200` completed end-to-end on GPU (RTX 3090). 200 steps in ~3 minutes wallclock. ELBO went from −2763.71 (step 0) → −1620.52 (final). Post-training metrics: **NMI 0.4814, purity 47.5%, accuracy 47.5%**. Conjugation barely moved (R² 0.0486, ||rho|| 10.4) because 200 steps is well short of the KL/conj warmups (1000/2000) — long-run behaviour is out of scope for the smoke test.
- **Compile-time spot-checks at n_latent = 1024 chain**: JT construction 0.4s; `jax.jit(log_partition)` first call 0.4s; `jax.jit(grad(log_partition))` first call 0.5s. Steady-state log_partition 13 ms, grad 40 ms. All well within XLA's normal operating regime.

### JT inference refactor (`src/goal/models/base/gaussian/_jt_inference.py` + `junction_tree.py`)

- [ ] **No raw Python `for` loops over cliques or tree edges.** `_collect` and `jt_sample` are now `lax.scan` over precomputed static topology arrays on the `JunctionTree`. Inner per-clique-position update inside `jt_sample` uses `lax.fori_loop` (sequential semantics — avoids the duplicate-index `.at[].set` race that a vectorised scatter would have when padded slots collide on index 0).
- [ ] **Padded shape convention.** Every per-clique array is shape `(n_cliques, 2^max_clique_size, ...)`; clique states $s \geq 2^{|C|}$ are masked to `-inf` via `clique_state_mask` so they fall out of `logsumexp`. For a 1D chain (the only `latent_graph` wired up today) every clique is size 2, no padding kicks in, and the general code reduces to what a chain-specialised implementation would emit.
- [ ] **Static topology arrays** added as `@cached_property`s on `JunctionTree` (`junction_tree.py:325+`): `bits_table`, `clique_state_mask`, `owned_bias_arrays`, `owned_edge_arrays`, `collect_step_arrays`, `pre_order`, `walk_step_arrays`. All numpy-typed (so the JT remains hashable and JIT-static).
- [ ] **Separator-state map handles padded entries safely** (`junction_tree.py::_build_sep_state_map`): real clique states $s < 2^{|C|}$ are packed by bit extraction; padded states are mapped to separator-state $0$ so `segment_logsumexp` never sees an all-padded segment (which would produce `NaN`).
- [ ] **`_clique_potentials` vmaps cleanly.** Sentinel `-1` in owned-bias/edge index arrays is handled by `params[idx + 1]` against a leading-zero-padded params vector; equivalent to mask-and-zero but cheaper. Inspect the gather/multiply structure in `_jt_inference.py:60-90`.
- [ ] **Correctness vs the previous Python-loop implementation**: existing `tests/boltzmann.py::TestChordalBoltzmann` (chain / 4-cycle / 5-cycle / 6-node irregular / K4) all pass without tolerance changes. Two new tests (`test_long_chain_log_partition_matches_dp` and `test_long_chain_sampling_matches_marginals`) exercise the scan at n_neurons = 64 chain.
- [ ] **Numerical-guard sanity-check.** `_segment_logsumexp` (now also used at scale) is bog-standard max-shift + `segment_sum`. Padded `-inf` values are absorbed into segments that already contain a finite contribution, so the formula stays well-defined.

### `src/goal/models/base/gaussian/junction_tree.py` (modified — append-only)

- [ ] **New cached properties.** `n_clique_states`, `max_separator_size`, `n_separator_states`, `bits_table`, `clique_state_mask`, `owned_bias_arrays`, `owned_edge_arrays`, `collect_step_arrays`, `pre_order`, `walk_step_arrays`. Plus a module-level helper `_build_sep_state_map`. All pure-numpy, computed lazily on first access.
- [ ] **`walk_step_arrays` returns a dict** (not a namedtuple) because the consumer `_jt_inference.jt_sample` unpacks specific keys; a flat dict was simpler than designing a one-off NamedTuple type. Reasonable?
- [ ] **`pre_order` cached** as a tuple. Previously this was a local helper in `_jt_inference.py:_pre_order`; lifting it to the JT lets the inference module stay shape-stable across calls.

### `src/goal/models/base/gaussian/_jt_inference.py` (modified — full rewrite of the two main passes)

- [ ] **`_clique_potentials` (~lines 50–90).** Vmap over cliques, gather + multiply pattern. Confirm the sentinel handling (`+ 1` offset, leading-zero padded params) does what you'd expect; an alternative would be explicit `jnp.where(mask, ...)` if you find this clearer.
- [ ] **`_collect` (~lines 100–125).** Single `lax.scan` over the precomputed `(child_idx, parent_idx, sep_maps)` tuple. The scan body does one `_segment_logsumexp` per step plus a `totals.at[parent].add(...)`.
- [ ] **`jt_sample` (~lines 140–215).** Outer `lax.scan` over the pre-order, inner `lax.fori_loop` to set the clique's variables in `x`. The inner loop is sequential (max_clique_size iterations, small static int).
- [ ] **No public API changes.** `jt_log_partition(jt, diag, off_diag)` and `jt_sample(jt, diag, off_diag, key)` keep the same signatures; downstream callers (`ChordalCouplingMatrix.log_partition_function`, `.sample`) are unaffected.

### `examples/variational_mnist/model.py` (modified)

- [ ] **`_chain_edges(n)` helper** (one-liner). Returns the 1D nearest-neighbour edge list.
- [ ] **`_build_base_latent(n_latent, latent, latent_graph) -> Any`.** Dispatches on the `latent` Literal. `latent_graph` is currently a fixed `Literal["chain"]`; the function asserts no other path is reachable via the type system (no runtime fallback `raise`).
- [ ] **`create_model` signature widened** with `latent` and `latent_graph` kwargs, both with sensible Bernoullis defaults. Docstring updated.
- [ ] **Type aliases extended.** `BinomialChordalHierarchical`, `PoissonChordalHierarchical`, `BinomialChordalFull`, `PoissonChordalFull` added to the `MixtureModel` union. No runtime impact — just keeps the return-type union accurate.
- [ ] **`normalized_reconstruction_error` chunked via `jax.lax.map`** with `batch_size=256`. Reason: the previous `jax.vmap(reconstruct)` over all 10k test samples materialised a single batched backward through the chordal `log_partition_function`'s `lax.scan` — that OOM'd the GPU at ~1.4 GiB per chunk on top of the existing model footprint (verified empirically; first smoke run failed here, second with `lax.map` succeeded). For `Bernoullis` this is a small constant-factor overhead; for `ChordalBoltzmann` it's necessary.

### `examples/variational_mnist/train.py` (modified)

- [ ] **CLI**: new `--latent {bernoullis,chordal_boltzmann}` (default `bernoullis`) and `--latent-graph {chain}` (default `chain`). Both validated by argparse `choices`, so `_build_base_latent`'s Literal-narrowing is safe.
- [ ] **Banner + config dict** carry the two new fields. `latent_graph` is only logged in the banner when `--latent chordal_boltzmann` is selected (avoids visual clutter for the default path).
- [ ] **`create_model` call** at the bottom of `main()` passes both flags through.
- [ ] **Bernoullis baseline unchanged.** Default invocation (`uv run python -m examples.variational_mnist.train`) constructs the same model as pre-PR — verify by a smoke run with `--n-steps 50` and a fixed seed.

### `examples/variational_mnist/diagnose.py` (modified)

- [ ] **`lat_dim` vs `lat_data_dim` split.** `lat_dim = model.bas_lat_man.dim` (= n + n_chordal_edges for `ChordalBoltzmann`) is used to reshape the interaction matrix; `lat_data_dim = model.bas_lat_man.data_dim` (= n) is used to slice prior/posterior samples (which are concatenated `[y, k]`). For `Bernoullis` they coincide so this is a no-op.
- [ ] **Optional unit/pair block breakdown.** When `lat_dim != lat_data_dim`, prints separate norms for the first `lat_data_dim` interaction columns (unit interactions, $x_i$) vs. the rest (pair interactions, $x_i x_j$). Quick visual sanity check that both signal types are picked up by the interaction matrix.
- [ ] **Collateral cleanup**: SIM108 ternary fix at the participation-ratio calculation; `# noqa: C901` on the function header (the +1 branch I added pushed complexity 15 → 16). No behaviour change.

### `tests/boltzmann.py` (modified)

- [ ] **`test_long_chain_log_partition_matches_dp`** (n_neurons = 64 chain). Compares the `lax.scan` collect against a Python-loop dynamic-programming reference (the standard forward sweep on a chain).
- [ ] **`test_long_chain_sampling_matches_marginals`** (n_neurons = 64 chain, 20k samples). Empirical first moments match `to_mean` (autodiff of `log_partition_function`) to `atol = 0.02`.
- [ ] Existing tests untouched and still passing.

### `src/goal/models/__init__.py`

- [ ] Imports re-sorted by ruff `--fix`. No new symbols beyond what Topic 2 added.

### Open follow-ups (deferred from this PR)

- **Richer graph topologies**: 1D band (configurable width), 2D grid, user-supplied edge list. The general `lax.scan` code path supports them already; just need the CLI plumbing and treewidth budgets.
- **Reconstruction error scale on `ChordalBoltzmann`.** The MSE-recon error is computed against `obs_man.to_mean` of the *expected* likelihood under the variational posterior, which in turn calls `pst_man.to_mean` (autodiff of `log_partition_function`). At n_latent = 1024 chain this is currently the slowest post-training step (~10s for 10k test images at `batch_size = 256`). Could be sped up by caching JIT-compiled `to_mean` and/or reducing the test-set size for during-training diagnostics.
- **Initial-parameter scale for `ChordalBoltzmann`.** Currently inherits `Differentiable.initialize` (random normal × shape 0.1). For long runs we may want to start the chordal couplings at zero (so the prior begins close to Bernoullis) and let them grow as training learns them. Not a blocker for the smoke run.
- **Longer training comparison.** The 200-step smoke run hit NMI 0.48 with R² ≈ 0.05; a proper comparison vs. the Bernoullis baseline needs ≥ several thousand steps with the conjugation regulariser engaged (`--conj-weight > 0`). Out of scope for this PR.
