# Clique container: review order

Branch `clique-container`, last commit `9a8a5db` (2026-09-29). Replaces `REVIEW-TODO.md` and
`TODO-GPT.md`, which described earlier designs (`LinearClique`, `BlockMap`, `EFClique`,
`CliqueSet`); recover them from git history if needed.

Review in this order; each module depends only on the ones above it.

## 1. `geometry/algebra/clique.py` --- the graph (reviewed 2026-09-26)

- Reviewed method by method. `cliques` is free-form; `level_sets` and `canonical_cliques`
  are the only readers of it, and every other member derives from those two.
- `algebra/util.py` removed: `split_by_dims` moved back to `manifold/util.py`, its only callers
  being in `manifold/`.

## 2. `geometry/manifold/map.py` (354), `combinators.py`, `embedding.py`

Decisions (2026-09-26):
- No reset to main and no re-typing. `D, C` generics on `CliqueMap` would be erased in a
  heterogeneous tuple and nothing reads them; models get their types from `Interaction`.
- `MatrixMap` stays (reference for `CliqueMap` at arity 2 in tests; `SquareMap`'s parent).
- Subspaces stay bundled with the map.
- `combinators.py` is main's file up to ruff formatting.

Open:
- Diff `map.py` against main. Expected only: `EmbeddedMap` -> `MatrixMap` without embeddings;
  `AmbientMap`, `BlockMap`, embedding-composition methods removed; `AffineMap` domain behind a
  private `_dom_man`. Anything else gets reviewed; then `map.py` counts as battle-tested.

Checked: `../goal-apps` uses none of the removed API. Untracked `variational_mnist` scripts
still import `EmbeddedMap` / `BlockMap` (experimental, left broken).

## 3. `geometry/manifold/clique.py` (627) --- maps and layout (restructured 2026-09-30)

State (see `CLAUDE.md`, Architecture): graph stored on the concrete model, global labels,
`RecursiveLinearCliques[Root, Deep]` is `Triple[Root, Interaction[Deep, Root], Deep]` with
`crs_man` derived, `Interaction` moved in from `interaction.py`, crossing axes built on the
subspaces of the cliques they reach, paths from `crs_embs` (identity when a partition is the
clique alone).

Open:
- Walk the module method by method, starting at `Interaction` and `crs_man`.
- `CliqueEmbedding.sub_man` returns the clique's map itself. Should it be the node space instead?
- `Interaction.cliques` and `Interaction.paths` are parallel tuples aligned only by
  `strict=True` zips. Replace with one `(map, cod_path, dom_path)` per term?
- `RootCliqueEmbedding.start` reuses the block's start in the whole layout, which is correct
  only because the root partition is stored first (stated in its docstring).

## 4. `geometry/exponential_family/harmonium.py` (580)

- Base is `RecursiveLinearCliques[Observable, Posterior]`; `int_man` returns `crs_man`.
- The interaction is still consumed as a `LinearMap` through `lkl_fun_man` / `pst_fun_man`
  (`AffineMap`s). Contracting cliques directly is the agreed next structural step; large
  blast radius.
- `InteractionEmbedding` / `PosteriorEmbedding` have no callers in `models/` or `examples/`
  (only exports and `tests/graphical.py`). Delete?
- `initialize_from_sample` passes the unsliced sample to `obs_man` (predates the branch).

## 5. Models: `graphical/mixture.py` (634), `harmonium/cca.py` (213), `graphical/hmog.py` (408)

- MFA's mixture view (`to_mixture_coords` / `from_mixture_coords`) is a block permutation
  written on the model.
- MFA: three crossing cliques $(x,y)$, $(x,y,k)$, $(x,k)$; a fork, depth 2. The arity-3
  clique still executes as a matrix over the joint $(y,k)$ statistic.
- CCA: multi-root fork, conjugation as a sum of two LGMs. It is probabilistic CCA and
  exposes no canonical directions; rename or document.

## 6. Tests: `clique.py`, `graphical.py` (677), `interaction.py`, `cca.py`, `graphical_mixture.py`

- `graphical.py` is the one to trust least: several pins test layout implementation rather
  than contract.
- `interaction.py` now tests a class in `manifold/clique.py`; merge into `graphical.py` or keep?
- CCA test is gradient-step smoke coverage; compare against an independent joint covariance.

## Pre-existing sharp edges (not this branch)

- Diagonal posteriors do not conjugate exactly (residual ~4.5e-2 for `NormalLGM`).
- `NormalCovarianceEmbedding` with `Scale` is an adjoint pair, not `project ∘ embed = id`.

## Deliberately not implemented

Decided 2026-09-30 to avoid generalizing before a model needs it.

- **Root cliques of several nodes.** The root partition holds only singletons, so neither a
  direct root coupling (e.g. $(x, y)$ in CCA) nor a crossing clique touching several roots is
  supported: a crossing clique's root part must be a clique of the root partition, as its
  deep part must be one of the deep partition. Such cliques also lack a declaration and an
  orientation. `clq_map` raises `ValueError` when unpacking the clique. Supported on
  2026-09-28, when the root partition could itself be a flat layout.
- **Partial (embedded) biases.** A bias covers its whole node (`bias_map`). Crossing axes are
  already built on the subspaces of the cliques they reach, so enabling this changes only the
  bias line of `clq_map`; harmoniums would still treat `rot_man` as the observable family.
- **Validation of the stored graph.** Nothing checks that every node has a singleton, or that
  a deep partition's graph equals its parent's `level_split()[2]`. With a single-node deep
  partition, an unknown label gets `dep_man`'s bias; a crossing clique whose deep part is not
  a deep clique fails only in `crs_embs`.

## Later (out of scope)

- Convolutional `MatrixRep`.
- Generic conjugation cascade over `RecursiveLinearCliques`, and the variational $\rho$ slot. Trap:
  ~10 sites unpack `split_coords` positionally; rename the method in the same change.
- Short user-facing architecture page (chain, fork, arity-3 example).

## Verification

| gate | command |
|---|---|
| lint | `uvx ruff check src/ tests/` |
| types | `uvx basedpyright src/ tests/` |
| docs | `uv run sphinx-build -q docs/source docs/build` |
| suite | `uv run python -m pytest tests/ -q` (553 passed on 2026-09-06, ~17 min; 316 in 11 clique-related files on 2026-09-30, before the `Interaction` merge) |
| numeric | `uv run python -m examples.cca.run` (CPU): alignment RMSE `0.24029927646203073` (HEAD c45178c and 2026-09-28 tree agree) |
