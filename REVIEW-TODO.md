# Code review plan — `clique-container`

A branch-wide review, ordered so each step only depends on ones above it. Everything
describes the code **as it stands in the working tree**. Where the design changed during
development, only the current state is described — except in §11, which records what was
tried and rejected, because those are the questions a reviewer will ask.

**Scope**: 5 commits (`070a231..6432ee1`) plus the working tree, against `main` at `4b957a6`.
91 files, **+5740 / −973**. Nothing since `6432ee1` is committed. The six formerly untracked
files — `geometry/manifold/graphical.py`, `geometry/exponential_family/clique.py`, their two
RST files, `tests/ef_clique.py`, `tests/multilinear.py` — are now **staged**, so a commit
will pick them up.

**What the branch does.** Model graphs became first-class. A model declares which nodes its
interactions couple; the graph, the level structure, and the parameter layout are *derived*
from that. `Harmonium` is a layout over a graph rather than a hard-coded triple, which is
what let a fork (CCA) and a three-way interaction (MFA) be expressed at all.

**The one idea to hold onto while reading.** A clique is one inner product in the
log-density, $\langle \theta_c, \bigotimes_{i \in c} \pi_i(\mathbf s_i(x_i)) \rangle$. Arity
is the only thing that varies: 1 is a bias, 2 a matrix contraction, 3+ a tensor contraction.
So a clique **is** a multilinear map — `Clique.form` is literally `MultilinearMap(axes)`. A
covariance is *not* a clique: it is a node's internal second moment, and structured matrix
representations live inside nodes, never across them.

---

## How to use this

Ten steps, bottom of the dependency graph upward. Steps 1–5 are the design and deserve real
attention; 6–7 are how models consume it; 8–10 are checks, skims, and open questions.
**§10 is decisions only you can make.**

Each step names the files, the specific question to answer, and the test that pins the
behaviour so you can check a claim rather than take it.

---

## 1. The combinatorics — `geometry/algebra/clique.py` (244 lines)

Pure integers, no JAX, no `Manifold`. Read this first; everything else is a layout over it.

**Settled in review — no action, listed so you can see what was decided.**

- [x] **Naming.** `CliqueSet` is $(V, R, C)$: `n_nodes`, `n_roots`, `cliques`. `scopes`
      reverted to `cliques`; `levels` → `level_sets` (fibres of `node_levels`, which was
      the one ambiguous pair); `deep_cliques` kept over `higher_cliques`, since "higher"
      already means arity elsewhere.
- [x] **Levels name the property, not the algorithm.** `node_levels` is distance from the
      root set, $\ell(v) = \min_{r \in R} d(v,r)$. BFS is named in one docstring, the
      property that runs it.
- [x] **Validation is minimal by rule**: a check earns its place only if the bad input
      would describe a *different graph without failing* — empty clique, repeated member,
      negative index (**wraps silently**), duplicate clique, unreachable node. Nothing
      guards an error that raises on its own; `n_nodes >= 1` was dropped as implied by
      `1 <= n_roots <= n_nodes`. The rule is in `_validate`'s docstring.
- [x] **Each fact is computed once.** `boundary` is now level 1 and is used by
      `ascend_level` instead of being re-derived; `ascend_level` filters by `deep_cliques`;
      `canonical_cliques` tests depth one as `n_roots == n_nodes`.
- [x] **Nodes are numbered by level** — `node_levels` must be non-decreasing, checked at
      construction. This is the invariant that gives *one* organization of a clique set
      rather than two reconciled ones. Storage is `[root | cross | deep]` recursively;
      canonical order was the same decomposition reached by a different route, and the two
      could part in three places. Level-ordering closes the third: dropping the root level
      becomes a shift by `n_roots` instead of a permutation, so `canonical_cliques`
      relabels with `+ n_roots` — the same operation `Clique.shifted` performs — and a
      manifold splices its deep span in by renumbering rather than reordering.
      `deep_nodes` is deleted; it existed only to carry the permutation. All six shipped
      models already satisfy it, and `_misrooted()` — the fixture that used to corrupt
      silently — now raises from `clq_set`. **The other two divergence sites remain open**:
      block order *within* the root span and *within* the cross span are the model's
      declaration, and can only be forced canonical once `int_man` is derived from the
      cliques rather than the cliques read back off the map. That is Phase 2.

**Still needs your judgement.**

- [x] **Edges are derived** — two nodes are adjacent when some clique contains both — so the
      cover is primitive and $E$ falls out. Is that the right primitive?
- [x] **The partition convention**, easy to misread: **nodes** split two ways (root /
      non-root); **cliques** split three ways (`root_cliques` / `cross_cliques` /
      `deep_cliques`). There is no third group of *nodes* — a cross clique holds nodes from
      both sides. `boundary` is the interface, and becomes the root set one level up.
- [x] `canonical_cliques` **nests**: the ascended graph's cliques are a contiguous suffix in
      exactly the order that graph would choose alone. **No longer the layout authority** —
      see §3.
- [x] Singleton cliques are **not** required. Whether a node carries a bias is a fact about
      the family occupying it, not about the graph.

*Pinned by* `tests/clique.py` (264 lines, 45 tests, pure Python, sub-second).

---

## 2. What is left of the old combinators — `geometry/manifold/combinators.py` (267 lines)

- [ ] **Against `main`, this file is unchanged apart from its docstring** — 3 insertions,
      1 deletion. It holds the same six classes it always did: `Null`, `Tuple`, `Pair`,
      `Triple`, `Quadruple`, `Replicated`. The "717 lines down to 267" figure was measured
      against the branch tip `6432ee1`, where the clique machinery lived in this file; the
      working tree moved it to `graphical.py` and left combinators as `main` had it.
      Nothing graph-related lingers: the only graph word in the file is one docstring line
      pointing at `graphical.py`.
- [ ] `geometry/exponential_family/graphical.py` (300 lines at `main`) was **deleted**.
      Full accounting of its seven classes:
      - `ObservableEmbedding`, `InteractionEmbedding`, `PosteriorEmbedding` → moved intact
        to `harmonium.py` (lines 591 / 611 / 631). `ObservableEmbedding` has 15 references;
        the other two have 4 each, none from `models/` (see §10).
      - `LatentHarmoniumEmbedding` → **deleted with no replacement**. It had zero consumers
        at `main` — its only references were its own export lines in `__init__.py`.
      - `DifferentiableHierarchical`, `SymmetricHierarchical`, `AnalyticHierarchical` →
        **deleted; not absorbed by `Harmonium`**. Their only consumer was `hmog.py`, which
        now carries a private `_HMoGBase`. The recursion they expressed is now
        `LevelCliques`; what stayed HMoG-specific became model-local. **The judgement to
        make**: three exported generic ABCs became one private one. They were only ever
        instantiated by HMoG, so the genericity was speculative — but if you intended them
        as the public extension point for future hierarchies, that point is gone.
      - No stale references anywhere: no RST for the module, and no mention of any deleted
        name in `src/`, `tests/`, or `docs/`.

---

## 3. The clique system — `geometry/manifold/graphical.py` (757 lines)

The core of the branch. Budget the most time here. Four clique classes plus one concrete
form; two embeddings and a cut view alongside them.

- [ ] **Module position.** It sits *above* `map.py` and `embedding.py`:
      `base → combinators → embedding → map → graphical`. That is what lets a clique *be* a
      multilinear map and hold an embedding.
- [ ] **`Clique`** — `members` (a field, ascending and distinct) plus abstract `axes`, one
      factor per member. `dim = prod(axes)`; `form` is `MultilinearMap(axes)`. Arity is a
      single fact: `len(members) == len(axes)`, checked at construction, so a graph and its
      layout cannot claim different arities.
      **The question I would put to it**: `members` is a field on the clique rather than on
      the manifold that owns it. The argument is that a node index is not a property of the
      family occupying it — a `Normal` is the same `Normal` wherever it sits — so `shifted`
      and `on` renumber a cheap record rather than rebuilding a manifold. Do you agree that
      is the right side of the line?
- [ ] **`DenseClique`** — the one concrete manifold-side clique, `axes` given directly.
      It exists because Python forbids overriding an inherited `axes` *field* with a
      property, which is what `EFClique` needs to do.
- [ ] **`LinearCliques`** — the bag. Three abstract members: `clq_set`, `clique_blocks`,
      and `dim`. `clique_members`, `clique_dims`, `clique_axes`, and `clique_offsets` are
      derived from `clique_blocks`, and `clique_index` looks a clique up **by its members**.
      **This is the fix for the branch's worst bug**: an offset and the members it belongs
      to now come from the same object, so a wrong slice is unconstructible.
      **Why `dim` stays independent** rather than being `sum(clique_dims)`: `dim` is about
      coordinates, blocks are a *view* of them, and keeping them separate is what makes
      "the blocks tile the coordinate vector" a real claim `layout_problems` can check
      instead of a tautology. It also keeps `.dim` — accessed constantly — from walking the
      whole block tree on every read.
- [ ] **Storage order versus canonical order.** Nothing that slices coordinates consults
      `canonical_cliques` any more. The two can part in exactly three places, and one is
      now closed structurally by the §1 numbering invariant: **between levels**, the
      relabelling is a shift by `n_roots` in both, so they cannot disagree. The other two —
      block order *within* the root span, and *within* the cross span — are the model's
      declaration, and are checked in one place over every shipped model (§8) rather than
      at construction. Closing them needs Phase 2.
- [ ] **`LevelCliques`** — `[root | cross | deep]`, the three spans of one level ascent.
      `clique_blocks` is the root span's blocks, then `cross_blocks`, then the deep span's
      **shifted past the root nodes** — keeping the labels the deep manifold itself gives
      them, so rerooting cannot relabel anything. Levels are distances in the glued graph,
      so how the deep span roots *itself* is discarded; that discard is what makes MFA
      depth 2 rather than 3.
      **The numbering invariant now draws the line for you.** The discard is harmless when
      the glued graph still numbers its nodes by level — MFA's levels come out $(0,1,1)$ —
      and is rejected outright when it does not. `tests/graphical.py::_misrooted` is the
      failing shape: a mixture rooted at $y$ whose crossing clique reaches $k$, giving
      levels $(0,2,1)$, and `clq_set` now raises on it. So the question is no longer
      "right or merely convenient" — the cases where it would have been merely convenient
      are unconstructible.
- [ ] **`cross_blocks`** is abstract, because the cross span is a bare parameter manifold —
      a model supplies its interaction as a map — so which nodes each block couples is the
      part only the model knows.
- [ ] **`CliqueProduct`** — disjoint union, all nodes roots. This is what a multi-root root
      span is: CCA's observable pair.
- [ ] **`cut(far_node)`** / **`CliqueCut`** — the **re-rooting isomorphism**: the same
      coordinates regrouped as `(near | crossing | far)` with one node split off instead of
      the root nodes. Integers only, so it is valid identically in natural and mean
      coordinates. Cutting **one** node is what makes the row assignment total: two crossing
      cliques sharing a near part $P$ would both be $P \cup \{far\}$ and hence the same
      clique. It is narrower than `split_level`, not a generalization: it needs every
      crossing clique to couple the far node *in full*, so they share one column axis.
      **It reads the layout, not the graph** — positions come from `clique_members`, so the
      index it computes and the dimension it selects come from the same block. It used to
      read `canonical_cliques` and index `clique_dims` with the result; two layouts over
      one graph, differing only in the storage position of two equal-sized blocks, produced
      an *identical* `CliqueCut` with the near and far blocks swapped, silently. The count
      guard that stood there was unreachable once positions came from the blocks, and is
      replaced by a real one: singleton cliques are optional, so a far node may carry no
      block at all, which would give a zero-column cut with no arithmetic to contradict it.
- [ ] **`CliqueBlockEmbedding`** — one selector per member, addressing the block that holds
      their **joint** statistic and restricting it axis by axis. Reading a joint block rather
      than per-member marginals is what makes it correct in mean coordinates.
- [ ] **`RootEmbedding`** lives here rather than in `embedding.py` — it is bounded by
      `LevelCliques`, so leaving it there would make the imports circular.
- [ ] `span_nodes` / `span_blocks` each carry one documented `isinstance` fallback: a
      manifold with no clique structure is a single node with a single opaque block. That
      fallback is what lets a leaf family occupy a node without knowing it does, and what
      lets a model carry a span whose parameters are not a clique decomposition (a learned
      variational correction). Judge whether it is the right escape hatch.

*Pinned by* `tests/graphical.py` (589 lines, 40 tests).

---

## 4. Maps — `geometry/manifold/map.py` (643 lines)

- [ ] **`LinearMap` is not a `Clique`** — verified: `map.py` imports only
      `algebra.matrix`, `.base`, `.combinators`, `.embedding`. Nothing from the clique
      system. A clique *is* a multilinear form; a linear map is what a model hands over.
      **Two dangling doc references found and fixed here**: the docstrings pointed at
      `graphical.MapClique` and `graphical.map_span`, both deleted in the consolidation.
      Sphinx does not fail on these — nitpick mode is off, so they silently rendered as
      plain text instead of links. They now point at `Clique.form` and
      `LevelCliques.cross_blocks`.
- [ ] **`SquareMap` has no clique notion** — the question is not asked. A square map is a
      self-interaction *inside* one node, and this is where structured matrix
      representations live. `Covariance` and `CouplingMatrix` are both of this shape.
- [ ] **`MultilinearMap`** — `sub_dims`, `tensor`, `contract`. The arity-$n$ generalization
      of `Rectangular`, knowing nothing about graphs. `contract` now bounds-checks `keep`
      (`map.py:508`); `keep = arity` previously contracted every axis and returned a
      rank-0 scalar.
- [ ] **`BlockMap.blocks`** is a `tuple`, validated non-empty with a common domain and
      codomain (`map.py:288-298`). It was a mutable `list` inside a frozen dataclass, which
      also made `BlockMap` unhashable — relevant since models are passed as static jit
      arguments.
- [ ] `BlockMap.clique_dims` → `block_dims` — no `clique_dims` remains anywhere in
      `map.py`, so it overrides nothing.

*Pinned by* `tests/multilinear.py` (153 lines, 22 tests) — arity 2 checked against
`EmbeddedMap` on `dim`, `outer_product`, and application in **both** directions; arity 3 on
round-trip and partial-contraction associativity.

**Standing claim, spot-checked**: every cross-clique interaction in the library is dense.
All six model-level `EmbeddedMap` constructions pass `Rectangular()` — `harmonium/mixture`,
`graphical/mixture`, `lgm`, `population_codes` (twice), `cca`; structured reps enter only
through `SquareMap.__init__`. `Clique.dim = prod(axes)` now *depends* on this — a
structured cross-node factor would fail construction. That is deliberate: a non-dense block
has no per-member tensor factorization, so failing loudly is the right answer.

---

## 5. Cliques over exponential families — `geometry/exponential_family/clique.py` (170 lines)

- [ ] **`EFClique(Clique)`** — adds `selectors`, one per member, and derives `axes` from
      them. It has no `dim`, `clq_set`, or `form` of its own; the selectors *are* the axes,
      which is what makes arity one fact rather than three that can disagree.
- [ ] **The distinction that matters**, and the one place I would double-check my reasoning:
      `tensor` multiplies its members' statistics together, which is exact only when every
      member is observed. When two or more members are latent, the block is
      $\mathbb E[\bigotimes_i \mathbf s_i]$ **jointly**, and expectation does not pass
      through a tensor product. `select_joint` is the operation for that case. Using
      `tensor` there would be silently wrong, not obviously so — measured gap on MFA is
      order one.
- [ ] **The structural condition** this implies: *a clique's latent members must themselves
      be a clique of the level above*, so their joint expectation exists as a block to
      select from. `LinearCliques.clique_index` raises when it does not hold. **Is that the right
      place for the check?** It fires at use rather than at construction.
- [ ] **Known wart**: `partial_contract` and `select_joint` document their `keep` argument
      as naming *members*, but index `selectors` by tuple **position**. These coincide only
      when members are `(0..n-1)`, which is true of every clique in the library. Decide
      whether to fix the code or the docstrings; either way it needs validation.

*Pinned by* `tests/ef_clique.py` (332 lines, 44 tests) — rebuilds **every** interaction shape
in the library as an `EFClique` and checks all three operations against the live
`EmbeddedMap`: factor analysis, mixture, HMoG, both CCA branches, all three MFA blocks.

---

## 6. Harmonium — `geometry/exponential_family/harmonium.py` (647 lines)

- [ ] Base is `LevelCliques[Observable, LinearMap[Posterior, Observable], Posterior]`.
      `split_level` returns exactly `(obs, int, lat)` as before, so everything written
      against the level split is unchanged however deep the graph goes.
- [ ] **`cross_man` *is* `int_man`.** No wrapper. Which nodes the interaction's blocks couple
      is reported separately by `cross_blocks`, built from `int_man` and `int_members`.
      Confirm that is the right single seam — it is the one Phase 2 replaces with explicit
      `EFClique`s.
- [ ] **`int_members`** — which nodes each interaction block couples, defaulting to
      `((0, 1),)`, the only shape a single interaction admits. Not `None` any more: the
      default is a real declaration rather than a sentinel.
- [ ] `_block_axes` reads one factor per member off a block's codomain and domain
      sub-statistics, so a domain that is itself a joint over several nodes contributes one
      factor each. That is what makes MFA's three-way block report arity 3.
- [ ] `HarmoniumEmbedding` and its three slots address the **level split**, not individual
      cliques, so they stay correct as a model's block count grows.

---

## 7. The models

None of them changed behaviourally in the consolidation; MFA and CCA changed by one line
each (`BlockMap` now takes a tuple), and MFA's `mix_cut` passes an int.

### 7a. MFA — `models/graphical/mixture.py`

- [ ] **The graph is derived from three tuples**: `int_members = ((0,1), (0,1,2), (0,2))`.
      Node count, root count, the three biases, the $(y,k)$ coupling from the mixture above,
      the levels, and the block layout all follow.
- [ ] **The graph is a fork, not a chain.** Both $y$ and $k$ are adjacent to $x$ through the
      three-way clique, so levels are $(1,2)$ and depth is 2. Consequence:
      `_CATEGORY_NODE = 2` cannot be replaced by `levels[-1]`, because the deepest level
      holds $y$ as well as $k$. (I tried; seven tests failed.)
- [ ] **The three interaction blocks are one construction** differing only in which nodes
      they name — `CliqueBlockEmbedding(bas_pst_man, members, selectors)` for $(0,)$,
      $(0,1)$, $(1,)$.
- [ ] `to_mixture_coords` / `from_mixture_coords` are three lines each via `CliqueCut`,
      replacing ~41 lines of hard-coded offsets. `whiten_prior` and `to_natural_likelihood`
      both flow through them.
- [ ] **Still an arity-2 bridge.** The $(x,y,k)$ block is *stored and executed* as an arity-2
      `EmbeddedMap` whose domain is the joint $(y,k)$ statistic. The graph reports arity 3
      and `tests/ef_clique.py` proves an `EFClique` reproduces it exactly, but production
      does not run that path. This is the main Phase 2 item.

*Pinned by* `tests/graphical_mixture.py` (410 lines, 37 tests).

### 7b. CCA — `models/harmonium/cca.py` (222 lines)

- [ ] The first model over a **multi-root** graph: the fork $x \leftarrow z \rightarrow y$,
      declared as `int_members = ((0,2), (1,2))`, with `n_roots = 2` derived from the
      observable being a `CliqueProduct`.
- [ ] **Conjugation is a sum**, $\rho = \rho_X + \rho_Y$, delegating to two standalone
      `NormalLGM`s. Exact because `DifferentiablePair.log_partition_function` is already a
      sum — no new machinery. Residual measured at 1.4e-16.
- [ ] It is `DifferentiableConjugated`, not analytic: inverting the conjugation sum
      branch-wise needs structure a fork does not supply. **Open question for you** — if you
      know the closed form, this gains EM.

*Pinned by* `tests/cca.py` (225 lines, 23 tests).

### 7c. HMoG — `models/graphical/hmog.py`

- [ ] **Declares nothing.** Its three-node chain — `(0,)`, `(0,1)`, `(1,)`, `(1,2)`, `(2,)` —
      comes from the default cross clique plus the deep span's own blocks spliced in
      recursively. This is the extension property working: a hierarchical model needs no
      graph declaration at all.
- [ ] The asymmetry is readable off the layout: the $x$–$y$ coupling touches only the
      *location* sub-statistics; the $y$–$k$ coupling touches $y$'s **full** statistic. That
      used to be buried in embeddings.

### 7d. Leaves — `models/base/*`

- [ ] **`Node` was deleted**, along with its 12 mixin sites. It reported exactly what the
      `span_blocks` fallback already computes for an unrecognised span, and nothing anywhere
      did `isinstance(x, Node)`. Skim these diffs; they are import-line removals.

---

## 8. Tests

- [ ] **New**: `clique.py` (264, 45 tests), `graphical.py` (589, 39), `cca.py` (225, 23),
      `ef_clique.py` (332, 44), `multilinear.py` (153, 22).
- [ ] **Grown**: `graphical_mixture.py` (410, 37), `boltzmann.py`, `population_codes.py`,
      `lgm.py`, `hmog.py`.
- [ ] `tests/combinators.py` → `tests/graphical.py`, matching the module it tests.
- [ ] **The two that matter most** are in `graphical.py::TestDeclarationOrderRegressions`.
      Both reproduce silent corruption in the pre-consolidation code and both pass now:
      reversed CCA branches (`clique_index((0,2))` returned the wrong block, at the wrong
      offset, with the wrong size) and a mis-rooted deep span (nodes 1 and 2's biases
      swapped and mis-sized). They were `strict=True` xfails for one commit, then flipped.
- [ ] `TestLayoutInvariants` sweeps every shipped model through `layout_problems`, which
      states four invariants in dependency order. **Storage order == canonical order is the
      first and returns alone** — this is the single enforcement point for canonical
      ordering in the codebase. It previously *assumed* that invariant: it pulled cliques
      from `canonical_cliques` and dims from `clique_dims` and zipped them, so a real
      divergence would have blamed the wrong clique or passed by coincidence. The remaining
      three — blocks tile the vector, one axis per member, axes multiply to the block size —
      now zip `clique_members`, the order they actually measured.
- [ ] The judgement to make: do these test the *contract* or my *implementation*? The layout
      pins in `graphical.py` are the ones I would trust least to be testing the right thing.

---

## 9. Docs and examples — mostly skim

- [ ] New RST: `manifold/graphical.rst`, `exponential_family/clique.rst`,
      `models/harmonium/cca.rst`. Trimmed: `combinators.rst`, `embedding.rst`, `map.rst`.
- [ ] `CLAUDE.md`: module map and test-file table, both updated — `CliqueBag` removed, and
      `ef_clique.py` / `multilinear.py` added, which the table had been missing entirely.
- [ ] **The example churn is formatting.** ~30 example files changed; the only one touching
      the clique API is the new `examples/cca/`. Verify with
      `git diff 4b957a6 -- examples/ | grep -E "^[+-]" | grep -iE "clq_set|Clique|split_level"`.

---

## 10. Decisions only you can make

- [ ] **`InteractionEmbedding` and `PosteriorEmbedding` are unused in `models/`**, fully
      subsumed by `CliqueBlockEmbedding`. Still coherent span-level public API, still tested.
      Delete or keep? (Their sibling `ObservableEmbedding` *is* used, so this is not a
      whole-family question.)
- [ ] **Is `LinearCliques.clique_index` the right home** for the "latent members must form a clique
      above" check? It raises at use, not construction.
- [ ] **How far should "clique-based" go?** The graph and layout are clique-based everywhere,
      but the *algorithms* are not: ~24 compute sites use `split_level` and `split_cliques`
      has **zero** callers. `CliqueBlockEmbedding` is the only production compute path on
      clique addressing, and only MFA uses it. Converting the rest is churn unless something
      needs per-clique dispatch — which is exactly what interleaving exact and variational
      cliques would need.
- [ ] **Should `cut` and `CliqueCut` move to their own module?** They are ~190 lines of
      `graphical.py` and are a *view* of a layout rather than a layout. Moving them is what
      gets the module from 751 to roughly 500.
- [ ] **Whether a vector-valued product is one node or many** is a modelling choice, not a
      property of the manifold. The `span_blocks` fallback hard-codes "one". Binds if you
      ever want per-neuron couplings.
- [ ] **`EFClique`'s positional `keep`** — see §5. Fix the code or the docstrings.
- [ ] **`CanonicalCorrelationAnalysis`** exposes no canonical directions or correlations. It
      is probabilistic CCA. Rename or document the distinction.
- [ ] **Fork EM** — see §7b.

---

## 11. Tried and rejected — so you need not re-derive them

- **`Cliques.cliques` vs `CliqueSet.scopes`.** For one round the `CliqueSet` field was
  renamed `scopes` so the container could call its own `cliques`. Reverted: renaming the
  container to `LinearCliques` and its accessor to `clique_blocks` removed the collision
  outright, so the graph layer keeps the mathematical word.
- **`deep_cliques` → `higher_cliques`.** Tried for unambiguity, reverted. "Higher" is
  already carrying arity and tensor order across `exponential_family/clique.py`,
  `graphical.py`, `multilinear.py` and `ef_clique.py`; "deep" says position and nothing
  else, and stays parallel with `LevelCliques`'s `deep_man`.
- **A public `is_canonical` / `check_layout` on `LinearCliques`.** Rejected twice, on the
  same ground both times: nothing at runtime consults canonical order, so a construction-
  time or API-level guard would defend a path no code takes. The invariant is stated once,
  in the test that sweeps every shipped model.
- **Permuting `clique_dims` into canonical order.** Unsound. `clique_offsets` slices real
  coordinates, and storage order is fixed by `split_coords`, then `BlockMap.coord_blocks`,
  then the deep manifold's own layout. Permuting labels while coordinates stay put makes an
  offset point into the middle of the wrong block, with the wrong size.
- **Deriving `clq_set` from the blocks, plus a `check_layout()` guard.** Built, then removed
  as unnecessary. The corruption was never about canonical order; it was that an offset and
  its members came from two computations. One record fixes it, and canonical order stops
  being consulted by anything that slices.
- **`Clique` carrying `members` as a field on the *manifold*.** Blocked: `Node` was mixed
  into 10+ leaf families, and a base-class dataclass field becomes the first positional field
  of every subclass — `Normal(3, Diagonal())` would bind `members=3`. Deleting `Node`
  resolved it.
- **Collapsing `MapClique.n_members` into an inherited `arity` property.** Python forbids a
  dataclass field shadowing an inherited property (`AttributeError: property 'arity' has no
  setter`). `arity` turned out to be `len(axes)` and was deleted outright.
- **Unifying `CliqueSet` with `JunctionTree`** — ruled out on your instruction. A
  `ChordalBoltzmann` is a monolithic vector-valued node; its junction tree is internal to its
  own log-partition and sampling and is not exposed to model-graph algorithms.

---

## Not done, deliberately

- **Interleaving conjugated and variational-conjugated levels** — the stated objective, not
  started. Shape: a fourth span slot, `[root | cross | deep | rho]`, since $\rho$ is a
  per-level correction on the deep span's root. Orthogonal to everything above. **Trap**:
  ~10 sites destructure `split_coords` positionally into three `Array`s, so reordering slots
  breaks them *silently* — rename the method in the same change to force every site to be
  touched.
- **`EFClique` as the execution primitive.** `cross_blocks` already takes
  `tuple[Clique, ...]` and `EFClique` is a `Clique`, so the swap is a model-side change:
  declare cliques, derive `int_man`. Deferred so MFA's arity-3 bridge could be left alone.
- **`sufficient_statistic` as a flat per-node clique loop** — would need a per-node data
  split, and `x` is currently split two ways by `obs_man.data_dim` with nested splits
  handling the rest. Real risk, unclear gain.

---

## Verification — commands and results, 2026-08-25

Rerun these rather than trusting them; all are cheap except the last two.

| gate | command | result |
|---|---|---|
| lint | `uvx ruff check src/ tests/` | clean |
| format | `uvx ruff format --check src/ tests/` | 68 files formatted |
| types | `uvx basedpyright src/ tests/` | 0 errors, 0 warnings |
| docs | `uv run sphinx-build -q docs/source <out>` | exit 0 |
| suite | `uv run python -m pytest tests/` | **524 passed**, 17m48s |

Since that full run, only `algebra/clique.py`, `manifold/graphical.py` and their tests
changed. Those four files were rerun: `clique.py` + `graphical.py` **84 passed**,
`cca.py` + `graphical_mixture.py` **60 passed**. A full pass is still worth one more run
before merge.

Baseline for comparison: `main` at `4b957a6` had 430 tests; the first working state of this
branch had 505. The evidence that nothing regressed is not the count — it is that the
equivalence checks below each ran *before* the code they justify was replaced.

| check | result |
|---|---|
| `MultilinearMap` vs `EmbeddedMap`, arity 2 | exact, both contraction directions |
| `EFClique` vs live interactions | exact, 7 shapes × 3 operations |
| `CliqueBlockEmbedding` vs hand-composed | `0.0` on `project` and `embed`, all 3 MFA blocks |
| Arity-3 clique vs `xyk_man` at the E-step | exact, over 16 posterior draws |
| `CliqueCut` vs the old offset arithmetic | bit-for-bit, both directions |
| CCA fork conjugation residual | 1.4e-16 (LGM control: 2.2e-16) |
| MFA/CCA/HMoG layouts across the consolidation | `clique_dims` unchanged |

**Examples: 18/18 pass** as of the last full run before the consolidation;
`variational_mnist` skipped (needs an MNIST download). Slowest: `boltzmann_lgm` 800s,
`pendulum` 585s, `chordal_boltzmann_ppc` 273s, `torus_poisson` 227s. `cca` gives
−9.972 → −5.014, latent RMSE 0.2403, cross-covariance error 0.0070. **Not re-run since the
consolidation** — worth redoing before merge.

**Two known sharp edges, both pre-existing and neither introduced here:**

- Diagonal posteriors do not conjugate exactly — residual ~4.5e-02 for a plain `NormalLGM`,
  ~1.7e-02 for CCA; both ~1e-16 with `PositiveDefinite`. Either an intended approximation or
  a real gap. Worth deciding which.
- `initialize_from_sample` accepts wrongly-shaped data silently: its docstring says the
  sample is for observable biases, but it passes the unsliced input to `obs_man`. Observable-
  only data into a joint harmonium returns non-finite parameters for CCA, finite-but-
  meaningless for LGM.

**A correction to an earlier note in this file.** It previously claimed
`NormalCovarianceEmbedding` fails `project ∘ embed = id` by "returning twice the input".
That is wrong. `embed` acts on natural parameters and `project` on mean parameters; they are
an **adjoint** pair, satisfying $\langle \text{embed}(\theta), \eta \rangle = \langle \theta,
\text{project}(\eta) \rangle$, and the `Scale` round-trip scales by $1/d$, not 2.
`project ∘ embed = id` is not this embedding's contract — the `LinearEmbedding` docstring
that claims it for all embeddings is what needs fixing.

**Note on measuring:** do not compare example numbers while the test suite is running. JAX's
threaded CPU reductions vary summation order under contention, which moved CCA's last digits
over 8000 optimizer steps and briefly looked like a regression.
