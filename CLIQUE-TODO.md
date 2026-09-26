# Clique container: review order

Branch `clique-container`, last commit `aed0cfb` (2026-09-24). Replaces `REVIEW-TODO.md` and
`TODO-GPT.md`, which described earlier designs (`LinearClique`, `CliqueMap`, `BlockMap`,
`EFClique`, `CliqueSet`); recover them from git history if needed.

Review in this order; each module depends only on the ones above it.

## 1. `geometry/algebra/clique.py` (217) --- the graph

- `Cliques` is a rooted graph given as a clique cover over integer nodes; no JAX, no `Manifold`.
  Contract is `cliques` and `n_roots`; levels, level split and canonical order derive.
- `canonical_cliques` has no production callers; kept because it states the nesting the
  `LevelCliques` recursion rests on.
- `same_graph` compares cover order as well as content. Rename to `same_cover`?

## 2. `geometry/algebra/util.py` (25)

- `split_by_dims` lives here; `batched_mean` in `manifold/util.py`. One `geometry/util.py`?

## 3. `geometry/manifold/map.py` (589), `combinators.py` (301) --- `SubspaceMap`, `Product`

- `SubspaceMap`: `rep` plus `cod_embs` / `dom_embs`; direction is group membership, `trn_man`
  swaps the groups; address-free. `SubspaceMap.whole(man)` builds a root form.
- `rep` stays on `SubspaceMap` for a future convolutional `MatrixRep`.
- Field order disagrees: `SubspaceMap` is codomain first, `MatrixMap` domain first.
- `Product` is the tensor product (`Product(())` has dim 1), the counterpart of `Null` for `Tuple`.

## 4. `geometry/manifold/clique.py` (632) --- the layout (**next: walk method by method**)

- `LinearCliques.placements`: one `(members, form)` per clique, member i the node of factor i
  in `cod_embs + dom_embs` order. Placements are validated on access to `cliques`, not at
  construction.
- `LevelCliques.cross_paths` relies on downward closure of the cover (a crossing clique's
  near part is a clique), which nothing states or checks.
- `LevelCliques.cross_placement` derives storage order, arity and output group from a node set.
  `cross_paths` derives the paths `Interaction` uses and rejects a placement with other than
  one root node first.
- `CliqueEmbedding.sub_man` returns the form itself. Should it be the node space instead?
- `split_cliques` / `join_cliques` have no production callers. Prune?
- Known limitation: a hand-paired form borrowed across a level (MFA's $\theta_{XY}$) that
  arrives transposed is caught only when the two nodes' manifolds differ.

## 5. `geometry/manifold/interaction.py` (172)

- Sums path-conjugated forms: $v \mapsto \sum_t \phi_t(\Theta_t \pi_t(v))$.
- `placements` and `paths` are parallel tuples aligned only by `strict=True` zips. Replace
  with one `(form, cod_path, dom_path)` per term, which also drops the unused `members` and
  its mismatch under `trn_man`.

## 6. `geometry/exponential_family/harmonium.py` (613)

- Base is `LevelCliques[Observable, Interaction, Posterior]`; `split_level` is unchanged.
- The interaction is still consumed as a `LinearMap` through `lkl_fun_man` / `pst_fun_man`
  (`AffineMap`s). Contracting cliques directly is the agreed next structural step; large
  blast radius.
- `InteractionEmbedding` / `PosteriorEmbedding` have no callers in `models/` or `examples/`.
  Delete?
- `initialize_from_sample` passes the unsliced sample to `obs_man` (predates the branch).

## 7. Models: `graphical/mixture.py` (607), `harmonium/cca.py` (203), `graphical/hmog.py` (386)

- MFA's mixture view (`to_mixture_coords` / `from_mixture_coords`) is a block permutation
  written on the model; `CliqueCut` was removed 2026-09-25.
- MFA: three crossing cliques $(x,y)$, $(x,y,k)$, $(x,k)$; a fork, depth 2. The arity-3
  clique still executes as a matrix over the joint $(y,k)$ statistic.
- CCA: multi-root fork, conjugation as a sum of two LGMs. It is probabilistic CCA and
  exposes no canonical directions; rename or document.
- HMoG declares nothing; its chain comes from the defaults.

## 8. Tests: `clique.py`, `graphical.py` (1003), `interaction.py`, `cca.py`, `graphical_mixture.py`

- `graphical.py` is the one to trust least: several pins test layout implementation rather
  than contract.
- CCA test is gradient-step smoke coverage; compare against an independent joint covariance.

## Pre-existing sharp edges (not this branch)

- Diagonal posteriors do not conjugate exactly (residual ~4.5e-2 for `NormalLGM`).
- `NormalCovarianceEmbedding` with `Scale` is an adjoint pair, not `project ∘ embed = id`.

## Later (out of scope)

- Convolutional `MatrixRep`.
- Generic conjugation cascade over `LevelCliques`, and the variational $\rho$ slot. Trap:
  ~10 sites unpack `split_coords` positionally; rename the method in the same change.
- Short user-facing architecture page (chain, fork, arity-3 example).

## Verification

| gate | command |
|---|---|
| lint | `uvx ruff check src/ tests/` |
| types | `uvx basedpyright src/ tests/` |
| docs | `uv run sphinx-build -q docs/source docs/build` |
| suite | `uv run python -m pytest tests/ -q` (553 passed on 2026-09-06, ~17 min) |
| numeric | `uv run python -m examples.cca.run`: alignment RMSE `0.24031811353161686` |
