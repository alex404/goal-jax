# Clique container: review order

Branch `clique-container`, last commit `aed0cfb` (2026-09-24). Replaces `REVIEW-TODO.md` and
`TODO-GPT.md`, which described earlier designs (`LinearClique`, `CliqueMap`, `BlockMap`,
`EFClique`, `CliqueSet`); recover them from git history if needed.

Review in this order; each module depends only on the ones above it.

## 1. `geometry/algebra/clique.py` --- the graph (reviewed 2026-09-26)

- Reviewed method by method. `cliques` is free-form; `level_sets` and `canonical_cliques`
  are the only readers of it, and every other member derives from those two.
- `algebra/util.py` removed: `split_by_dims` moved back to `manifold/util.py`, its only callers
  being in `manifold/`.

## 2. `geometry/manifold/map.py` (589), `combinators.py`, `embedding.py` --- decided 2026-09-26, not yet done

Decisions:
- No reset to main and no re-typing. `SubspaceMap` is main's `EmbeddedMap` with a tuple of
  embeddings per side (tensor-product sides, empty side = constant/bias) plus factor accessors.
  `D, C` generics would be erased in the heterogeneous `placements` tuple and nothing reads them;
  models get their types from `Interaction[Domain, Codomain]`.
- `MatrixMap` stays (reference for `SubspaceMap` at arity 2 in tests; `SquareMap`'s parent).
- Subspaces stay bundled with the map (the `d37779f` phase stripped them; reverted in `aed0cfb`).

Next steps:
1. Done: `SubspaceMap` and `Product` moved into `manifold/clique.py` ("Subspace Maps" section).
   `combinators.py` is main's file up to ruff formatting. Rename later; leading candidate
   `CliqueForm` (the code already says "form").
2. Diff `map.py` against main. Expected only: `EmbeddedMap` -> `MatrixMap` without embeddings;
   `AmbientMap`, `BlockMap`, embedding-composition methods removed; `AffineMap` domain behind a
   private `_dom_man`. Anything else gets reviewed; then `map.py` counts as battle-tested.
3. `FirstEmbedding` / `SecondEmbedding` (new in `embedding.py`, exported, no users): delete?
4. `Product` now sits in `clique.py` beside `CliqueProduct` (a disjoint union): two
   "products" in one module. Revisit names in §3.

Checked: `../goal-apps` uses none of the removed API. Untracked `variational_mnist` scripts
still import `EmbeddedMap` / `BlockMap` (experimental, left broken).

## 3. `geometry/manifold/clique.py` (632) --- the layout (**next: walk method by method**)

- Opens with the moved `SubspaceMap`: name it, and trim (`to_tensor`/`from_tensor`/
  `_require_dense` are test-only; `cod_dims`/`dom_dims` beside `sub_dims`; public
  `project_cod`/`embed_cod`/`project_dom` wrappers). Field order vs `MatrixMap`.
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

## 4. `geometry/manifold/interaction.py` (172)

- Sums path-conjugated forms: $v \mapsto \sum_t \phi_t(\Theta_t \pi_t(v))$.
- `placements` and `paths` are parallel tuples aligned only by `strict=True` zips. Replace
  with one `(form, cod_path, dom_path)` per term, which also drops the unused `members` and
  its mismatch under `trn_man`.

## 5. `geometry/exponential_family/harmonium.py` (613)

- Base is `LevelCliques[Observable, Interaction, Posterior]`; `split_level` is unchanged.
- The interaction is still consumed as a `LinearMap` through `lkl_fun_man` / `pst_fun_man`
  (`AffineMap`s). Contracting cliques directly is the agreed next structural step; large
  blast radius.
- `InteractionEmbedding` / `PosteriorEmbedding` have no callers in `models/` or `examples/`.
  Delete?
- `initialize_from_sample` passes the unsliced sample to `obs_man` (predates the branch).

## 6. Models: `graphical/mixture.py` (607), `harmonium/cca.py` (203), `graphical/hmog.py` (386)

- MFA's mixture view (`to_mixture_coords` / `from_mixture_coords`) is a block permutation
  written on the model; `CliqueCut` was removed 2026-09-25.
- MFA: three crossing cliques $(x,y)$, $(x,y,k)$, $(x,k)$; a fork, depth 2. The arity-3
  clique still executes as a matrix over the joint $(y,k)$ statistic.
- CCA: multi-root fork, conjugation as a sum of two LGMs. It is probabilistic CCA and
  exposes no canonical directions; rename or document.
- HMoG declares nothing; its chain comes from the defaults.

## 7. Tests: `clique.py`, `graphical.py` (1003), `interaction.py`, `cca.py`, `graphical_mixture.py`

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
