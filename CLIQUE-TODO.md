# Clique container: review order

Branch `clique-container`, last commit `6c1eb6e` (2026-09-30). Replaces `REVIEW-TODO.md` and
`TODO-GPT.md`, which described earlier designs (`LinearClique`, `BlockMap`, `EFClique`,
`CliqueSet`); recover them from git history if needed.

Review in this order; each module depends only on the ones above it.

Done: `algebra/clique.py`, and `map.py` / `combinators.py` / `embedding.py` (diffed
against main). Remaining in `src/`: `manifold/clique.py` and `cca.py` (new), the rewritten
consumers below (~800 changed lines), and `lgm.py` / `population_codes.py`, ports
following one pattern.

## Before merging

- Some `variational_mnist` scripts still import `EmbeddedMap` / `BlockMap` (experimental,
  left broken). They are committed on this branch: decide whether they belong in it.

## 1. `geometry/manifold/clique.py` --- maps and layout (restructured 2026-09-30, 2026-10-01)

State (see `CLAUDE.md`, Architecture): graph stored on the concrete model, global labels,
`RecursiveLinearCliques[Root, Deep]` is a `LinearCliques` and the
`Triple[Root, CrossMap[Root, Deep], Deep]` with `crs_man` derived, `CrossMap` moved in
from `interaction.py`, crossing axes built on the subspaces used by their root and deep parts. Each
of the root and deep partitions is a single node or a `LinearCliques` (CCA's `NormalPair`
is a flat one); `clq_map` and `part_emb` treat both sides alike.
`RootCliqueEmbedding`, `rot_nod_mans`, `nod_man` and `dep_man_rlc` were removed 2026-10-01.

Open:
- Walk the module method by method, starting at `LinearCliques`, `part_emb` and `clq_map`.
- `CliqueEmbedding.sub_man` returns the clique's map itself. Should it be the node space instead?
- `CrossMap` (2026-10-01): `terms` pairs each clique's node labels with its `CliqueMap`;
  `clq_embs(clique)` derives the clique embeddings into its sides via `part_emb`, which
  splits the clique by each side's clique group (`cod_group`/`dom_group`). `EmbeddedCliqueMap` and
  `RecursiveLinearCliques.crs_embs` are gone (embeddings are stored at the level of their
  ambient, see `CLAUDE.md`). Still keeps its own `clq_dims`/`coord_blocks`. Its single-term
  `clq_map` property is gone (2026-10-02), so becoming a `LinearCliques` with
  `clq_map(clique)` no longer collides with anything.
- Partitions are read from their clique groups, not their types (2026-10-01): one clique
  means the partition is that clique (bias, identity), several mean it must be a
  `LinearCliques` (a cast, no guard: a wrong partition fails on `clq_map`/`clq_emb`).
- `CrossMap` maps between the whole root and deep partitions, with their clique groups as
  fields (2026-10-02): the groups describe its own codomain and domain, so locating a term
  there stays within its scope. The span design (`CliqueSpan`, `SpanEmbedding`,
  `ConjugatedMap`, `crs_spans`, `span_embs`) was tried and dropped: about 150 lines, and a
  single-node partition could not carry its span's labels. `int_man` is `crs_man` again.
  A `LinearCliques` partition's `cliques` and the group come from the same graph.

## 2. `geometry/exponential_family/harmonium.py` (580)

Mostly mechanical: ~10 `split_coords`/`join_coords` -> `split_level`/`join_level`, and
`HarmoniumEmbedding` with its three subclasses moved in unchanged from the deleted
`exponential_family/graphical.py` (~100 of 193 changed lines). ~10 minutes.

- Base is `RecursiveLinearCliques[Observable, Posterior]`; `obs_man`/`pst_man` are the
  contract, `rot_man`/`dep_man` forward to them, and `int_man` is `crs_man`.
- `Conjugated.extract_likelihood_input` was removed (6e2b96d): `sample` passes the whole
  prior sample to `likelihood_at`. Inferred, not verified: correct because `pst_man` is now
  the whole deep partition and the interaction reads $y$'s block of it through `clq_emb`, so
  `pst_man.sufficient_statistic` of a joint $yk$ sample is the right input.
  `tests/hmog.py::test_sampling` checks shape and finiteness only; add a moment check.
- The interaction is still consumed as a `LinearMap` through `lkl_fun_man` / `pst_fun_man`
  (`AffineMap`s). Contracting cliques directly is the agreed next structural step; large
  blast radius.
- `InteractionEmbedding` / `PosteriorEmbedding` have no callers in `models/` or `examples/`
  (only exports and `tests/graphical.py`). Delete?
- `initialize_from_sample` passes the unsliced sample to `obs_man` (predates the branch).

## 3. Models: `graphical/mixture.py` (634), `harmonium/cca.py` (213), `graphical/hmog.py` (408)

Also `harmonium/mixture.py` (mechanical, ~10 minutes: `EmbeddedMap` `int_man` replaced by
`crs_rep`/`crs_emb_constructors`, `impose` added, `int_man.to_matrix` ->
`crs_man.clq_map.to_matrix`, most changed lines are ruff wrapping). The other two need
about an hour together:

- MFA (`graphical/mixture.py`): `RowEmbedding`, `xy_man`/`xyk_man`/`xk_man` and the
  `BlockMap` (~60 lines) became `crs_rep` and `crs_emb_constructors`. The latter (~15 lines)
  is the densest code downstream of `clique.py`: it keys the base harmonium's constructors
  by root/non-root and relies on `bas_clique`, `mix_nodes` and roots-first labelling.
- MFA's mixture view (`to_mixture_coords` / `from_mixture_coords`) is a block permutation
  written on the model; rewritten on this branch. Check the `jnp.split` offsets in
  `from_mixture_coords` (covered by `tests/graphical_mixture.py`).
- HMoG (`graphical/hmog.py`): the `*Hierarchical` bases are gone and `_HMoGBase` now
  defines `pst_prr_emb` (a `RootEmbedding`), `conjugation_parameters` (lower $\rho$ embedded
  by `ObservableEmbedding` of the upper prior), and delegates `crs_rep` /
  `crs_emb_constructors` to `lwr_hrm`. `upr_graph` + `impose` hand the upper mixture its
  graph. `AnalyticHMoG.to_natural_likelihood` is new (~6 lines).
- MFA: three crossing cliques $(x,y)$, $(x,y,k)$, $(x,k)$; a fork, depth 2. The arity-3
  clique still executes as a matrix over the joint $(y,k)$ statistic.
- CCA: multi-root fork, conjugation as a sum of two LGMs. It is probabilistic CCA and
  exposes no canonical directions; rename or document.

## 4. Tests: `clique.py`, `graphical.py`, `interaction.py`, `cca.py`, `graphical_mixture.py`

- `graphical.py` is the one to trust least: several pins test layout implementation rather
  than contract.
- `interaction.py` now tests a class in `manifold/clique.py`; merge into `graphical.py` or keep?
- CCA test is gradient-step smoke coverage; compare against an independent joint covariance.
- Root cliques of several nodes are possible since 2026-10-01 but untested.

## Pre-existing sharp edges (not this branch)

- Diagonal posteriors do not conjugate exactly (residual ~4.5e-2 for `NormalLGM`).
- `NormalCovarianceEmbedding` with `Scale` is an adjoint pair, not `project ∘ embed = id`.

## Deliberately not implemented

Decided 2026-09-30 to avoid generalizing before a model needs it.

- **Partial (embedded) biases.** A bias covers its whole node (`bias_map`). Crossing axes are
  already built on the subspaces used by their root and deep parts, so enabling this changes only the
  bias line of `clq_map`; harmoniums would still treat `rot_man` as the observable family.
- **Validation of the stored graph.** Nothing checks that every node has a singleton, that a
  partition's `cliques` equal its parent's group (`level_split()[0]` or `[2]`), or that a flat
  container's `cliques` are normalized (a repeated clique silently gets two blocks). A node
  that appears only in crossing cliques is missing from its side's group, so it is dropped
  from that side's part, silently if the rest is a clique there (raised in `clq_emb` before
  2026-10-01). A crossing clique whose part is not a clique of its partition fails only in
  `clq_emb`.
- **One role for a map read through embeddings** (decided 2026-10-03). $v \mapsto
  \iota(A(\pi(v)))$, a linear map between subspaces read in the manifolds containing them,
  is implemented twice: `CliqueMap` (its representation between the axes' subspaces, through
  `cod_emb`/`dom_emb`) and each term of `CrossMap` (the clique map, through the
  `part_emb` blocks). An abstract role with contract `cod_emb`/`map_man`/`dom_emb` would
  state it once, but saves about 10 lines for an extra class. Reconsider if a third
  occurrence appears; the role would belong in `map.py`.

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
| suite | `uv run python -m pytest tests/ -q` (531 passed on 2026-10-01, ~17 min) |
| numeric | `uv run python -m examples.cca.run` (CPU): alignment RMSE `0.24031811353161686` (2026-10-01: c45178c, 6c1eb6e and the working tree agree; the earlier `0.24029927646203073` predates an environment change) |
