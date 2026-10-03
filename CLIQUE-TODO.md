# Clique container: plan

Branch `clique-container`. Baseline: commit `1dbf826` (2026-10-03), the "integrated" design
below. `CLAUDE.md` (Architecture) describes that baseline. Older designs (`LinearClique`,
`BlockMap`, `EFClique`, `CliqueSet`, `Interaction`) are in git history.

## Baseline state (`1dbf826`)

- `manifold/clique.py`, 560 lines: `TensorProduct`, `TensorProductEmbedding`, `CliqueMap`,
  `bias_map`; `LinearCliques`, `CliqueEmbedding`, `part_emb`; `CrossMap`;
  `RecursiveLinearCliques[Root, Deep]`, `RootEmbedding`.
- `CrossMap(cod_man, dom_man, cod_group, dom_group, terms)` maps the whole deep partition
  to the whole root partition. `cod_group`/`dom_group` are the clique labels of its
  codomain and domain; `clq_embs(clique)` locates a term's blocks with
  `part_emb(partition, group, clique)`: the identity if the term's nodes there are the
  partition's only clique, `clq_emb` otherwise.
- `RecursiveLinearCliques.clq_map` ends with a bias branch: a non-crossing clique that is
  its partition's only clique gets `bias_map(partition)`.
- `Harmonium.int_man` is `crs_man`, typed `CrossMap[Observable, Posterior]`.
- Type parameters are target-first for maps and embeddings (`Map[Codomain, Domain]`,
  `Embedding[Ambient, Sub]`) and storage order for layouts (`RecursiveLinearCliques[Root,
  Deep]`, `Harmonium[Observable, Posterior]`); the two agree because the root is the
  output of the cross map.
- Single-term models get their one map from their own `clq_map(clique)`, with
  `(xz,) = self.level_split()[1]` (`CrossMap.clq_map` was removed).
- Verified at baseline: full suite 531 passed (2026-10-02), ruff and basedpyright clean,
  `examples.hmog.run` and `examples.dimensionality_reduction.run` complete.

## What we learned: where do a single node's labels live?

The user's principle: a class may hold structure --- embeddings or labels --- only about
itself or its own domain and codomain, never about a manifold containing it (`CLAUDE.md`,
"Embeddings are stored at the level of their ambient"). A multi-clique partition is a
`LinearCliques` and carries its own labels; a single node (a `Normal`, a `Categorical`)
carries none. Every design so far has had to put those labels somewhere.

1. **`EmbeddedCliqueMap`** (dropped 2026-10-01): each term stored embeddings into the whole
   partitions, which enclose it. Breaks the principle.
2. **Spans** (dropped 2026-10-02): `CrossMap` between `CliqueSpan`s of just the blocks it
   couples, `SpanEmbedding` placing them, `ConjugatedMap` lifting the map back to the
   partitions. About 150 lines and three classes, the pattern
   $v \mapsto \iota(A(\pi(v)))$ implemented three times, and a single-node partition still
   could not carry its span's labels (`SpanEmbedding` had to store them).
3. **Integrated, with groups** (baseline): works and is the smallest so far. The user's
   objection: the map stores labels for its codomain and domain that those manifolds do not
   carry themselves, and singleton handling is spread over `part_emb` and `clq_map`.
4. **`SingleClique` wrapper** (tried 2026-10-03, not committed): a one-clique
   `LinearCliques` wrapping a family, with `RecursiveLinearCliques.part_clqs` presenting
   both partitions as layouts. In `clique.py` it removed the groups, `part_emb` and the bias
   branch, and the clique tests passed (118). It failed one level up: the interaction's
   sides became wrapper objects, not the families, so `AffineMap(int_man, pst_man)` held two
   different domain objects for one space, `lkl_fun_man` stopped being
   `AffineMap[Observable, Posterior]`, and basedpyright reported 6 errors (`harmonium.py` 3,
   `dynamical.py` 2, `variational.py` 1). Making `Observable` the wrapper fails because
   `Observable: Gibbs` and the harmonium uses it as an exponential family. It also grew
   `clique.py` to 583 lines.

Conclusion: a single node's label has to live on the family itself. Then every partition
is a `LinearCliques` by type, and nothing in `clique.py` needs to know whether a partition
is a single node.

## Plan: `ExponentialFamily` as a `LinearCliques`

Make `ExponentialFamily` subclass `LinearCliques`. A single-node family has one clique, its
node, with `clq_map` = `bias_map(self)`; a harmonium already is a `LinearCliques`.

What goes away in `clique.py`: `CrossMap.cod_group`/`dom_group` (`clq_embs` reads
`cod.nodes`/`cod.clq_emb` directly), `part_emb` and its `cast`, the bias branch of
`RecursiveLinearCliques.clq_map`, and the rule "a partition's structure comes from its
group, not its type". `CrossMap[Observable, Posterior]` stays exact and `AffineMap` keeps
one domain object. Partial biases (below) become a family's own choice of `clq_map`.

Costs and open questions:

1. **Where the label is stored.** Labels are global (an LGM's latent is node 1), so each
   single-node family needs a node label, as a private keyword-only field with a default
   (the `_raw_cliques` precedent). It cannot go on `ExponentialFamily`: harmoniums derive
   their cliques from their graph, and abstract classes carry only fields every subclass
   shares. Options: one field on each concrete family (about 15 classes), or a small shared
   base class for single-node families. Unresolved.
2. **Models label their sides.** Each model builds `obs_man`/`pst_man` with the labels from
   its `level_split()`, generalizing `Mixture.impose`. Touches roughly every harmonium and
   dynamical model.
3. **Equality.** `Normal(..., node 0) != Normal(..., node 1)`. Comparisons of manifolds
   (`pst_man == prr_man`, embeddings' sub and ambient, `same_graph`, test assertions) will
   see the label. Expected to hold where both sides come from one graph; not verified.
4. **`exponential_family/base.py` changes**, and it is as well tested as `map.py`. The user
   proposed this, so it is in scope, but it is the largest change to core on this branch.

### Step 1: prototype on factor analysis (about an hour)

Start from `1dbf826`.

1. Give `Normal` a node label and make it a `LinearCliques` (one clique, `bias_map(self)`).
2. Make `FactorAnalysis` (via `LGM`) build its observable and latent with labels from its
   graph.
3. Remove the singleton handling from `clique.py`: the groups, `part_emb`, the bias branch.
4. Run `tests/lgm.py`, `tests/interaction.py` and `uvx basedpyright src tests`.

Decide from the result: how many families and models need edits, what the label field
looks like, and whether any equality check breaks. Bring the numbers back before step 2.

### Step 2: roll out (about half a day, if step 1 holds)

All single-node families, all models' sides, tests, `CLAUDE.md` Architecture, the `.rst`,
then the full suite and the CCA numeric gate.

## Other open items

### Before merging

- Some `variational_mnist` scripts still import `EmbeddedMap` / `BlockMap` (experimental,
  left broken; 124 basedpyright errors predate this work). They are committed on this
  branch: decide whether they belong in it.

### `manifold/clique.py`

- Walk the module method by method once the singleton question is settled.
- `CliqueEmbedding.sub_man` returns the clique's map itself. Should it be the node space?
- `CrossMap` keeps its own `clq_dims`/`coord_blocks`; it could become a `LinearCliques`
  with `clq_map(clique) = dict(terms)[clique]` now that nothing collides with that name.

### `exponential_family/harmonium.py`

- `Conjugated.extract_likelihood_input` was removed (6e2b96d): `sample` passes the whole
  prior sample to `likelihood_at`. Inferred, not verified: correct because `pst_man` is the
  whole deep partition and the interaction reads $y$'s block of it through `clq_emb`.
  `tests/hmog.py::test_sampling` checks shape and finiteness only; add a moment check.
- The interaction is consumed as a `LinearMap` through `lkl_fun_man` / `pst_fun_man`
  (`AffineMap`s). Contracting cliques directly is the agreed next structural step; large
  blast radius.
- `InteractionEmbedding` / `PosteriorEmbedding` have no callers in `models/` or `examples/`
  (only exports and `tests/graphical.py`). Delete?
- `initialize_from_sample` passes the unsliced sample to `obs_man` (predates the branch).

### Models

- MFA (`graphical/mixture.py`): `crs_emb_constructors` (~15 lines) is the densest code
  downstream of `clique.py`; it relies on `bas_clique`, `mix_nodes` and roots-first
  labelling. Check the `jnp.split` offsets in `from_mixture_coords`.
- HMoG (`graphical/hmog.py`): `_HMoGBase` defines `pst_prr_emb` (a `RootEmbedding`) and
  `conjugation_parameters`, and delegates `crs_rep`/`crs_emb_constructors` to `lwr_hrm`.
  `AnalyticHMoG.to_natural_likelihood` is new.
- CCA is probabilistic CCA and exposes no canonical directions; rename or document.

### Tests

- `graphical.py` is the one to trust least: several pins test layout implementation rather
  than contract.
- `interaction.py` tests a class in `manifold/clique.py`; merge into `graphical.py` or keep?
- CCA test is gradient-step smoke coverage; compare against an independent joint covariance.
- Root cliques of several nodes are possible but untested.

### Pre-existing sharp edges (not this branch)

- Diagonal posteriors do not conjugate exactly (residual ~4.5e-2 for `NormalLGM`).
- `NormalCovarianceEmbedding` with `Scale` is an adjoint pair, not `project ∘ embed = id`.

## Deliberately not implemented

Decided to avoid generalizing before a model needs it.

- **Partial (embedded) biases.** A bias covers its whole node (`bias_map`). Crossing axes
  are already built on the subspaces used by their root and deep parts. Under the plan
  above this becomes a family's choice of `clq_map`.
- **Validation of the stored graph.** Nothing checks that every node has a singleton, that a
  partition's `cliques` equal its parent's group, or that a flat container's `cliques` are
  normalized (a repeated clique silently gets two blocks). A node that appears only in
  crossing cliques is dropped from its side's part. A crossing clique whose part is not a
  clique of its partition fails only in `clq_emb` (`ValueError` from `cliques.index`).
- **One role for a map read through embeddings** (2026-10-03). $v \mapsto \iota(A(\pi(v)))$
  is implemented twice: `CliqueMap` and each term of `CrossMap`. An abstract role with
  contract `cod_emb`/`map_man`/`dom_emb` would save about 10 lines for an extra class.
  Reconsider if a third occurrence appears; it would belong in `map.py`.

## Later (out of scope)

- Convolutional `MatrixRep`.
- Generic conjugation cascade over `RecursiveLinearCliques`, and the variational $\rho$
  slot. Trap: ~10 sites unpack `split_coords` positionally; rename in the same change.
- Short user-facing architecture page (chain, fork, arity-3 example).

## Verification

| gate | command |
|---|---|
| lint | `uvx ruff check src/ tests/` |
| types | `uvx basedpyright src/ tests/` |
| docs | `uv run sphinx-build -q docs/source docs/build` |
| suite | `uv run python -m pytest tests/ -q` (531 passed on 2026-10-02, ~17 min) |
| numeric | `uv run python -m examples.cca.run` (CPU): alignment RMSE `0.24031811353161686` (2026-10-01) |
