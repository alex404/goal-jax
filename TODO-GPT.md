# `clique-container` review and handoff

Review target: the five commits on `clique-container` after `main` (`070a231..6432ee1`),
plus the current working tree. This is a review, not a claim that every item should be fixed
without a design decision first.

## Executive assessment

The intent is sound: make graphical-model structure explicit as a clique-indexed parameter
layout, then express harmoniums as one level of that layout instead of a hard-coded triple.
The branch separates integer graph combinatorics (`CliqueSet`), manifold layout
(`Cliques`/`LevelCliques`), clique statistics (`EFClique`), and concrete models. CCA proves a
multi-root fork; MFA proves a three-node, arity-3 factor scope and a nontrivial layout cut.

The branch is a successful **layout and metadata prototype**, but not yet a general graphical
model computation system. Production inference still uses the old
`[observable | LinearMap interaction | posterior]` operations (`outer_product`, transpose,
and `split_level`). `EFClique`, `partial_contract`, and `split_cliques` have no production
algorithm callers. The arity-3 MFA block is still stored and executed as an arity-2
`EmbeddedMap` whose domain happens to be a joint `(y,k)` statistic. That is a useful bridge,
but it means the graph currently describes computations that remain encoded elsewhere.

In short:

- The core direction is promising and the CCA/MFA proof cases are mathematically meaningful.
- Existing models appear preserved, and the new focused suite is strong on equivalence.
- The abstraction boundary is not settled: graph, layout, map, selectors, and factor scopes
  can disagree because no single object owns and validates all of them.
- Several public contracts accept invalid states or silently mislabel/drop coordinates.
- Documentation contains excellent mathematical explanations, but too much repeated design
  defense and some stale or false prose. It needs a concise editing pass, not more prose.

## P0 — make the reviewable branch self-contained

- [ ] Commit or intentionally discard the six currently untracked files. Four are core
      implementation/docs and two are their tests:
      `geometry/manifold/graphical.py`, `geometry/exponential_family/clique.py`, their RST
      files, `tests/ef_clique.py`, and `tests/multilinear.py`. As committed, `HEAD` does not
      contain these files even though tracked files import them. A clean checkout of the
      branch is therefore not the implementation reviewed here.
- [ ] Recompute the scope summary in `REVIEW-TODO.md` after the working tree is finalized.
      Its current file/addition counts and some model-shape claims are stale.
- [ ] Split or revert the unrelated example-formatting churn before rebasing if possible.
      It adds conflict and review surface while contributing nothing to clique semantics.

## P0 — correctness and invariant failures

### Block order can silently disagree with canonical clique order

`LevelCliques.clq_set` hands `int_members` to `CliqueSet`, which sorts cliques, while
`clique_dims` keeps `BlockMap.blocks` in declaration order. Only block count is checked.
Reversing CCA's declared members from `((0,2), (1,2))` to `((1,2), (0,2))` leaves the first
map block attached to the first canonical clique, silently relabelling the branches. With
different branch dimensions the mismatch is directly visible.

- [ ] Define one authoritative ordering. Prefer constructing explicit `(members, block)`
      records and sorting/permuting them together, or reject noncanonical declaration order.
- [ ] Validate member scope, block dimension, factor axes, and selector count together—not
      merely `len(cliques) == len(blocks)`.
- [ ] Add a regression with deliberately reversed, differently-sized CCA branches.

### `CliqueCut` is not lossless over its admitted input domain

`Cliques.cut` permits two crossing cliques to use the same near-side row. `project` converts
`cross_rows` to a dictionary, so the later clique overwrites the earlier one; `join(project(x))`
then duplicates one block and loses the other. A three-node graph with scopes `(0,1)` and
`(0,2)`, cut across `{1,2}`, reproduces this with `cross_rows == (0,0)`.

The dimension guard also checks only an integer product. It can accept a crossing scope that
does not structurally cover the claimed far-side statistics. The current implementation is
reliable for the MFA category cut (one far-node block and one crossing block per near row),
not for the general arbitrary cut described by the API.

- [ ] Either implement column slices for multiple crossing scopes per near row, or explicitly
      reject duplicate `cross_rows` and narrow the contract/name to the supported case.
- [ ] Validate structure using scopes and axes, not coincidental equal dimensions.
- [ ] Reject empty, all-node, and out-of-range `far_nodes` at construction. They currently
      build a `CliqueCut` and fail later in `jnp.concatenate` with an incidental error.
- [ ] Add round-trip/property tests for every accepted cut shape.

### `clique_axes` is false for the production arity-3 block

The contract says one axis per clique member. `MapClique.axes` returns an underlying matrix's
two-dimensional `matrix_shape`; MFA's `(x,y,k)` clique consequently reports two axes
`(x, y*k)`, not three `(x,y,k)`. Thus the graph says arity 3 while the layout introspection
says arity 2. Generic axis-wise clique code cannot trust this API.

- [ ] Decide whether nested/joint members are first-class axes. If yes, store explicit axes
      with the factor scope or derive them from a multilinear/EFClique object. If no, remove
      the one-axis-per-member claim and do not present current maps as general higher arity.
- [ ] Enforce `len(axes) == len(scope)` and `prod(axes) == block_dim` for every clique.
- [ ] Add this assertion for every real model, especially MFA.

### Invalid empty scopes break clique partitioning

`CliqueSet` accepts `()`. Because `all(...)` is vacuously true, an empty scope is both a root
and a deep clique; on a multi-level graph it appears twice in `canonical_cliques`.

- [ ] Reject empty cliques. Also decide whether repeated members should be rejected rather
      than silently deduplicated by `set(c)`.
- [ ] Add empty-scope and repeated-member tests.

## P1 — settle the central abstraction before extending it

### `CliqueBag` violates the `Cliques` contract

A concrete `CliqueBag` with more than one block cannot answer its required `clq_set` property,
even when it has `members`; those indices live in its owner's frame. This is evidence that
`members` is relational data on the owning level, not intrinsic state of a standalone clique
manifold.

- [ ] Redesign around an explicit factor/block record owned by `LevelCliques`, or give the bag
      a real local node frame plus an embedding into the owner's frame.
- [ ] Remove the state where a concrete `Cliques` object raises when its core graph property
      is read.
- [ ] Eliminate `map_span`'s one-off `isinstance(BlockMap)` interpretation once models can
      supply explicit clique/factor objects.

### Choose whether `EFClique` is the execution primitive

`EFClique` is mathematically the most general new object, but only tests use it. Models still
execute `LinearMap`; the same scope/selector information is repeated in `int_members`, map
embeddings, `CliqueBlockEmbedding`, and test-created `EFClique`s.

- [ ] Make an explicit decision:
  - integrate `EFClique` into sufficient statistics and message construction, with an ordered
    collection of executable factors; or
  - keep clique metadata as layout-only and demote/remove the unused execution API until a
    real algorithm needs it.
- [ ] If integrating it, make member identifiers and selector positions unambiguous.
      `partial_contract`/`select_joint` document member IDs but index selectors as tuple
      positions; this only works accidentally for scopes `(0,1,2)`.
- [ ] Validate nonempty arity, sorted/unique members, valid `keep` positions, argument counts,
      selector ambient manifolds, and joint-block axes at construction/use.
- [ ] Do the same bounds check in `MultilinearMap.contract`; `keep=-1` or `keep=arity`
      currently returns a scalar instead of rejecting the call.

### Replace permissive span fallbacks with an explicit escape hatch

`span_cover`, `span_blocks`, and `span_axes` silently turn any non-`Cliques` manifold into one
opaque node/block. This is convenient for variational corrections, but it can also hide a
missing graph implementation and manufacture misleading structure.

- [ ] Introduce an explicit opaque-span wrapper/protocol instead of using `isinstance` as
      implicit semantics.
- [ ] Ensure a learned variational correction is not falsely advertised as an ordinary
      factor clique if it has different meaning.

## P1 — additional correctness and API robustness

- [ ] Strengthen `RootEmbedding.__post_init__`. Equal `clq_set`s are insufficient: the root
      embedding must match both root manifolds, and cross/deep dimensions and layouts must be
      identical. Otherwise `embed` can silently return an array whose size is not
      `amb_man.dim`.
- [ ] Make clique split/join operations reject wrong total/block sizes. `_split_by_dims`
      silently ignores trailing coordinates and short slices; `join_cliques` checks block
      count but not each block's dimension. These are exactly the silent errors the project's
      validation policy says to guard.
- [ ] Change `BlockMap.blocks` from a mutable list inside a frozen dataclass to an immutable
      tuple and validate nonempty, common domain/codomain, and stable block order. The clique
      graph now relies on these properties.
- [ ] Fix or narrow `Harmonium.initialize_from_sample`. Its docstring says the sample is for
      observable biases, but it passes the unsliced input directly to `obs_man`. Observable-
      only and joint samples are currently ambiguous, and wrong shapes can yield nonfinite or
      meaningless parameters. This predates the branch but is exposed by CCA.
- [ ] Resolve `NormalCovarianceEmbedding`'s `Scale` projection/embedding mismatch and then add
      CCA/LGM conjugation tests for every advertised posterior representation. The generic
      `LinearEmbedding` claim `project(embed(x)) == x` is currently false for that case.
- [ ] Decide whether `CanonicalCorrelationAnalysis` is intentionally probabilistic CCA. The
      class implements a two-view latent Gaussian model but exposes no canonical directions,
      correlations, or transforms. Rename to `ProbabilisticCCA` or document the distinction.
- [ ] Decide compatibility policy for moved/removed public modules
      (`geometry.manifold.matrix`, `geometry.exponential_family.graphical`). Add temporary
      re-export shims and migration notes if downstream imports should survive the rebase.

## P2 — redundancy and maintainability

- [ ] Remove or justify unused public APIs: `InteractionEmbedding` and
      `PosteriorEmbedding` have no model callers; `split_cliques` has no production callers;
      `EFClique` is test-only. Public surface should reflect supported concepts, not every
      prototype explored during design.
- [ ] Reduce repeated reconstruction of the same semantic object. A harmonium exposes a raw
      `int_man`, rebuilds a `cross_man` wrapper on access, separately declares `int_members`,
      and may recreate selectors again as `EFClique`s. Consolidation will remove several
      opportunities for drift.
- [ ] Consider caching `CliqueSet` derivations only after correctness is settled. `node_levels`
      reruns BFS through most properties and recursively during canonical ordering. Graphs are
      small, so this is low priority.
- [ ] Keep `CliqueSet` separate from the internal Boltzmann `JunctionTree`; they describe
      different graph layers. The current separation is correct.

## Documentation review

### What is good

- The mathematical intent is unusually clear. The distinction between node-internal moments
  and model-level factor scopes is valuable, as is the warning that joint latent moments do
  not factor into products of marginals.
- RST files are mostly thin scaffolding and follow the documented source-of-truth policy.
- CCA's fork derivation and conjugation-as-a-sum are explained well.
- `REVIEW-TODO.md` gives a reviewer a useful dependency order and names the important design
  decisions.

### What to trim or correct

- [ ] Rewrite `models/graphical/mixture.py`'s module docstring. It advertises a nonexistent
      `harmonium_mixture_conjugation_parameters` function and reads like generic generated
      filler (“efficient”, “key function”, repeated decomposition) rather than the module.
- [ ] Remove stale historical references to deleted `RowEmbedding` from `tests/ef_clique.py`
      and `tests/graphical_mixture.py`. Tests should explain the current contract, not the
      implementation journey.
- [ ] Correct the two competing MFA graph fixtures. `tests/clique.py::MFA` and its expected
      canonical order omit the explicit `(0,2)` scope, while the real MFA and
      `tests/graphical_mixture.py` correctly contain seven blocks/scopes. The three-way scope
      induces an edge but does not replace the separate `(x,k)` factor.
- [ ] Correct `REVIEW-TODO.md` wherever it repeats the six-scope MFA description or stale
      counts. It should not claim “verification already done—so you need not redo it”; record
      commands/results with a date and let the next reviewer rerun them.
- [ ] Trim `geometry/manifold/graphical.py`. At 846 lines, much of the size is repeated
      rationale across the module, class, method, tests, and review plan. Preserve each
      invariant once at the highest useful level; method docstrings should state inputs,
      outputs, constraints, and failure modes.
- [ ] Remove autobiographical/defensive prose (“I got this wrong twice”, explanations of
      abandoned inheritance arrangements) from permanent docs. Put durable design decisions
      in a short architecture note if they cannot be inferred from contracts.
- [ ] Update `Harmonium` prose: models mostly declare factor memberships and the graph is
      derived; saying a model “declares its graph” blurs the central design claim.
- [ ] Add one concise user-facing architecture page or README section with a tiny chain, fork,
      and arity-3 factor example. Current API reference is extensive, but a user cannot easily
      learn how to define a new graph/model or what flexibility is actually supported.
- [ ] Clarify which helpers are public. `graphical.rst` documents `span_cover`/`span_blocks`,
      the top-level package exports `map_span`, and sibling helpers remain internal. Prefer a
      small explicit public API.

Overall documentation quality: mathematically strong and far above typical prototype code,
but too repetitive. The highest-value edit is subtraction: remove stale history, generic
claims, and duplicated rationale while adding one concrete extension guide.

## Test review and missing coverage

The new tests are substantial. The strongest ones compare new constructions to live model
behavior, verify the conjugation equation, distinguish joint moments from factored marginals,
and check natural/mean layout round trips. The weakest ones pin implementation details such as
exact class composition and hand-chosen canonical layouts without testing invalid permutations.

- [ ] Add property tests for graph/layout alignment under shuffled scope declaration order.
- [ ] Add exhaustive small-graph tests with nonempty scopes. A local enumeration through four
      nodes found the current BFS/ascent/canonical-permutation logic sound; retain a bounded
      version as regression coverage if runtime stays small.
- [ ] Add all negative cases listed above: empty scopes, invalid `keep`, invalid cuts,
      duplicate cut rows, axes/arity mismatch, wrong split/join sizes, and mismatched
      `RootEmbedding` span dimensions.
- [ ] Add at least one end-to-end test in which a model algorithm actually iterates executable
      cliques. Until that exists, tests establish layout compatibility, not the stated general
      graphical-computation goal.
- [ ] Strengthen CCA validation against an independently constructed joint Gaussian covariance
      or closed-form marginal likelihood. “Twenty gradient steps improve the loss” is useful
      smoke coverage but weak evidence of correct parameterization.
- [ ] Use integer category samples in `tests/ef_clique.py`; the focused suite currently emits
      four `jax.nn.one_hot` deprecation warnings because floats are passed as categories.

## Suggested execution order for Claude

1. Make the branch self-contained and freeze a clean baseline.
2. Write failing tests for block/scope ordering, duplicate-row cuts, arity/axes, empty scopes,
   and `RootEmbedding` compatibility.
3. Decide whether factors are executable (`EFClique`) or layout-only. Do not deepen both APIs
   independently.
4. Introduce one authoritative ordered factor record and derive graph, dims, axes, and
   execution from it.
5. Narrow or repair `CliqueCut`; keep the MFA repacking equivalence test.
6. Integrate one real clique-driven computation path before adding more model shapes.
7. Resolve covariance embedding and sample-initialization sharp edges.
8. Reduce public/unused API and add compatibility shims according to the chosen policy.
9. Perform the documentation subtraction pass, then update `REVIEW-TODO.md` to the final
   architecture rather than the prototype's history.
10. Rerun formatting, lint, type checking, docs with cached/offline intersphinx inventories,
    the full test suite, and representative examples.

## Validation performed during this review

- Focused tests: **191 passed** in 2m44s (`clique`, `graphical`, `ef_clique`, `multilinear`,
  `cca`, `graphical_mixture`); four float-category deprecation warnings.
- `git diff --check`: clean.
- Sphinx read and rendered all 47 sources. `-W` exited nonzero only because the sandbox could
  not fetch the Python/NumPy/JAX intersphinx inventories; no local-document warning appeared.
- Exhaustive enumeration of reachable nonempty clique covers through four nodes found no
  failure in canonical permutation or level-ascent shifting.
- Targeted counterexamples reproduced the ordering mismatch, duplicate-row `CliqueCut` data
  loss, invalid-cut late failures, empty-scope duplication, arity/axes mismatch, and invalid
  `MultilinearMap.keep` acceptance described above.
- Ruff and basedpyright were not independently rerun because their executables are not present
  in the locked environment and network installation was unavailable. `REVIEW-TODO.md` records
  a prior clean run, but rerun both after finalizing the working tree.
