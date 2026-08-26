# Code review plan — `clique-container`

A branch-wide review, ordered so each step only depends on ones above it. Everything
describes the code **as it stands in the working tree**. Where the design changed during
development, only the current state is described — except §9, which records what was tried
and rejected, because those are the questions a reviewer will ask.

**Scope**: 5 commits (`070a231..624badc`) plus the working tree, against `main` at `4b957a6`.
Nothing since `624badc` is committed.

**What the branch does.** Model graphs became first-class. A model declares which nodes its
interactions couple; the graph, the level structure, and the parameter layout are *derived*
from that. `Harmonium` is a layout over a graph rather than a hard-coded triple, which is
what let a fork (CCA) and a three-way interaction (MFA) be expressed at all.

**The one idea to hold onto while reading.** A clique is one inner product in the
log-density, $\langle \theta_c, \bigotimes_{i \in c} \pi_i(\mathbf s_i(x_i)) \rangle$. Arity
is the only thing that varies: 1 is a bias, 2 a matrix contraction, 3+ a tensor contraction.
So a clique **is** a multilinear map — `EFClique.form` is literally `MultilinearMap(axes)`. A
covariance is *not* a clique: it is a node's internal second moment, and structured matrix
representations live inside nodes, never across them.

---

## Status — 2026-08-26

**Two layers, not three.** `geometry/manifold/clique.py` is deleted. The clique system now
lives in exactly two places:

| module | lines | holds |
|---|---|---|
| `geometry/algebra/clique.py` | 473 | `Cliques`, `CliqueCut` — pure integers, no JAX, no `Manifold` |
| `geometry/exponential_family/clique.py` | 798 | `EFClique`, `LinearCliques`, `LevelCliques`, `CliqueProduct`, `CliqueBlockEmbedding`, `RootEmbedding`, `project_cut` / `join_cut` |

The middle layer was removed because it had no client of its own: every construct in it was
consumed by `exponential_family/` or `models/`, nothing in `manifold/`. The precedent for a
manifold-level twin (`manifold/combinators.py`) is justified by real non-EF clients —
`LinearMap`, `MultilayerPerceptron`, the embeddings. The clique layer had none.

**Why it was worth doing, beyond class count.** A clique's factor dimensions are decided by
which sub-statistics it couples, and a sub-statistic is an exponential-family notion. Because
the old layout layer could not see selectors, `Harmonium` stated its interaction as a
`LinearMap` and then reverse-engineered the factors back out of it. That reconstruction
(`_block_axes`) had two fallback branches returning `(block.dim,)`, reporting **arity 1 for a
two-node coupling**. Removing the boundary left the reconstruction with nothing to do.

**Where the layers divide**, stated once so the rest of this file can lean on it. Node
indices meet linear structure in exactly three places:

1. position → members (integers) — `clique_members`
2. position → size (integers) — `clique_dims`
3. member → axis (needs sub-statistics, hence EF) — `clique_axes` / `form`

`cut` and `split_level` use only (1) and (2). That is why the whole cut *index* computation
could move to `algebra`, and why only the array operations stayed above.

**Verified.** Full suite green at every stage boundary; ruff, basedpyright (`0 errors, 0
warnings`) and sphinx clean throughout. See the Verification table at the bottom.

**What is left for you.** Every `- [ ]` below. §1–§3 are the design and deserve real
attention; §5–§7 are how models and tests consume it; **§8 is decisions only you can make.**
Items marked `- [x]` were settled in earlier review rounds and are listed only so you can see
what was decided.

---

## 1. The combinatorics — `geometry/algebra/clique.py` (509 lines)

Pure integers, no JAX, no `Manifold`. Read this first; everything else is a layout over it.
It grew by ~265 lines this round: the partition primitive and the cut's index computation.

**New — needs your judgement.**

- [ ] **`partition(cliques, nodes) -> (inside, crossing, outside)`** is now the one primitive
      under both structural splits. At the root set it is the `[root | cross | deep]` grouping a layout
      stores its coordinates in; at a single node it is the re-rooting cut. Verified against
      the existing code on all four shipped models and seven hand-built covers before
      anything was changed. **The question**: is one primitive with two call sites the right
      unification, or does naming them separately carry information the merge loses?
- [ ] **The split is total; the row assignment is partial.** `partition` returns
      `inside`/`crossing`/`outside` as *positions*; `crossing_rows` is a separate pass. The
      cover `((0, 1),)` rooted at `{0}` — an interaction with no bias on either node, which
      `Cliques` explicitly allows — has a well-defined crossing group but no rows, because
      the near part `(1,)` is not in the cover. Folding rows into `partition` would make
      `root_cliques` raise on a cover it handles today. **Check the split is where you would
      have put it.**
- [ ] **Rows are injective only at $|S| = 1$.** Two crossing cliques sharing near part $P$
      would both be $P \cup S$, hence the same clique — an argument that needs $|S| = 1$. At
      $|S| > 1$ they collide, as CCA's two roots do (`crossing_rows == (0, 0)`). A cut is
      unaffected
      because it cuts one node; confirm nothing else assumes injectivity.
- [ ] **`positions`, not cliques.** `partition` returns positions in *whatever order it is
      handed* — `Cliques.cliques` for a graph, storage order for a layout. That is why it is
      a free function with a thin `Cliques.partition` method over it: a layout must count
      positions in the order it slices, or offsets and sizes come from different lists.
- [ ] **`CliqueCut` + `cut_indices(cliques, dims, far_node, n_nodes)`** — the cut's entire
      index computation, including the `dims[i] == height * n_cols` check, now lives here.
      This **reverses** an earlier claim in this file that the dimensional condition could
      not move to `algebra`; a dimension is an integer, and `cut` never reads anything else.
      **The seam to judge**: `algebra` now knows about block *sizes*, though still nothing
      about manifolds or JAX. Is that the right line, or should `algebra` stay purely about
      the graph?
- [ ] **The one thing combinatorics cannot decide.** A cut can be combinatorially admissible
      and dimensionally meaningless. LGM at $S = \{1\}$ has both $c \setminus S = (0,)$ and
      $c \cap S = (1,)$ in the cover, yet `cut` correctly refuses it: the interaction reaches
      only a subspace of node 1, so the crossing blocks do not share one column axis. This is
      what keeps `cut` from being a mere generalization of `split_level`, and it is why
      `cut_indices` takes `dims`.
- [ ] `root_cliques` / `cross_cliques` / `deep_cliques` are now three readbacks of
      `partition(root_nodes)` instead of three independent `node_levels[i] == 0` scans.
      `root_nodes` is new and states the equivalence the level-ordering invariant guarantees:
      `node_levels[i] == 0` iff `i < n_roots`. It is the module's only set-typed value; every
      other derived quantity is a sorted integer tuple.
- [ ] **No record classes for the partition.** `CliquePartition` was tried and removed: it
      bundled a function's three outputs with its two inputs, which is a helper, not an
      abstraction. `partition` and `crossing_rows` are now two functions of the same two
      arguments, the second recomputing the split rather than taking it as an argument.
      `CliqueCut` was kept as a class — it is a validated cut, not a bag of returns — but its
      array operations moved out to `project_cut` / `join_cut` in the EF layer, so there is no
      fields-only parent with a methods-only child.
      **Judge whether `CliqueCut` earns being a class on the same standard.**

**Settled in earlier rounds — no action.**

- [x] **Naming.** `Cliques` is $(V, R, C)$: `n_nodes`, `n_roots`, `cliques`. `scopes`
      reverted to `cliques`; `levels` → `level_sets`; `deep_cliques` kept over
      `higher_cliques`, since "higher" already means arity elsewhere.
- [x] **Levels name the property, not the algorithm.** `node_levels` is distance from the
      root set, $\ell(v) = \min_{r \in R} d(v,r)$.
- [x] **Validation is minimal by rule**: a check earns its place only if the bad input would
      describe a *different graph without failing*. The rule is in `_validate`'s docstring.
- [x] **Edges are derived** — two nodes are adjacent when some clique contains both — so the
      cover is primitive and $E$ falls out.
- [x] **Nodes are numbered by level**, checked at construction. This is the invariant that
      gives *one* organization of a clique set rather than two reconciled ones: dropping the
      root level becomes a shift by `n_roots` instead of a permutation.
- [x] **The partition convention**: **nodes** split two ways (root / non-root); **cliques**
      split three ways. A crossing clique holds nodes from both sides — there is no third
      group of *nodes*.

*Pinned by* `tests/clique.py` (445 lines, 52 test functions, **sub-second, no JAX**).
`TestPartition` and `TestCliqueCut` are new and cover every failure mode of both.

---

## 2. The clique system — `geometry/exponential_family/clique.py` (808 lines)

The core of the branch, and now the only clique module above `algebra`. Budget the most time
here. Read it in the order the file is written: clique, then layouts, then spans, then the
two views (block embedding, cut), then `RootEmbedding`.

**New — needs your judgement.**

- [ ] **`EFClique` is the only clique.** `LinearClique` is deleted and `EFClique` extends
      `Manifold` directly, carrying `members`, `selectors`, `axes` (read off the selectors),
      `dim = prod(axes)`, `form`, `shifted`, `on`, and the validation. **The judgement**: the
      old split existed so a clique could state its axes without naming families. Nothing
      needed that except scaffolding. Do you want the escape hatch back?
- [ ] **`_OpaqueClique` is deleted too.** A span with no clique structure is now
      `EFClique((0,), (IdentityEmbedding(span),))` — which is strictly better, since it knows
      its node's family and can therefore produce a sufficient statistic, where the opaque
      version could only state a number.
- [ ] **The span type bounds narrowed** as a consequence:
      `LevelCliques[Root: ExponentialFamily, Cross: Manifold, Deep: ExponentialFamily]`. The
      cross span stays a `Manifold` because it is a `LinearMap`. **The claim to check**: a
      span that occupies a graph node *is* a family. Every call site already satisfied it.
- [ ] **`CliqueBlockEmbedding` is an `EFClique`.** Same fields; one owns a block and the
      other addresses another manifold's. Its `members`/`selectors` fields and its
      `__post_init__` are inherited, and `sub_man` is just `self.form`.
- [ ] **`project` and `select_joint` were *not* merged**, though they walk the same axes.
      They take their shape from different places — `select_joint` from the selectors (the
      level above), `project` from `amb_man.clique_axes` (the ambient layout's own
      statement). Those agree for every shipped model but need not in general, and merging
      them would silently break if an ambient block were itself restricted. They share one
      `_map_axis`. **Judge whether the distinction is real or over-careful.**
- [ ] **`block_clique(block, members)`** is the bridge from the map API to the clique API,
      and what replaced `int_members` + `_block_axes`. One rule: the codomain embedding is
      the first selector, and the domain embedding supplies the rest — one selector if it
      addresses a single node, several if it addresses a joint block over a group of them.
      That second case is what makes MFA's three-way block arity 3, and it works only because
      `CliqueBlockEmbedding` is now an `EFClique`, so its `.selectors` *are* the per-member
      selectors. `members` still has to be passed: arity is readable from the block, which
      nodes it couples is not.
- [ ] **`LevelCliques` was moved, not absorbed into `Harmonium`.** The plan said absorb; the
      reason not to is `tests/graphical.py`'s `_Spans`, which declares a `Cliques` *and*
      three spans independently and checks the derived layout agrees with the declared graph.
      A `Harmonium` derives its `clq_set` from its spans, so absorption makes that check
      unexpressible and forces the fixture to become a full `Gibbs` family. `LinearCliques`
      has to survive as a class regardless — `CliqueProduct` extends it and
      `CliqueBlockEmbedding` is typed over it. **Confirm you agree; this is the one place the
      approved plan was not followed.**

**Carried forward from earlier rounds — still open.**

- [ ] **`LinearCliques`** — the layout. `clique_members`, `clique_dims`, `clique_axes`, and
      `clique_offsets` are all derived from `clique_blocks`, and `clique_index` looks a clique
      up **by its members**. **This is the fix for the branch's worst bug**: an offset and the
      members it belongs to now come from the same object, so a wrong slice is
      unconstructible.
      **Why `dim` stays independent** rather than being `sum(clique_dims)`: `dim` is about
      coordinates, blocks are a *view* of them, and keeping them separate is what makes "the
      blocks tile the coordinate vector" a real claim `layout_problems` can check instead of
      a tautology.
- [ ] **Storage order versus canonical order.** Nothing that slices coordinates consults
      `canonical_cliques`. The two can part in exactly three places; one is closed
      structurally by §1's numbering invariant (between levels, both relabel by `+n_roots`).
      The other two — block order *within* the root span and *within* the cross span — are
      the model's declaration, checked in one place over every shipped model (§6) rather than
      at construction. **Decided this round**: `Cliques` keeps *one* canonical order,
      root-set-relative; other cuts are views, never competing layouts.
- [ ] **`LevelCliques`** — `[root | cross | deep]`. `clique_blocks` is the root span's
      blocks, then `cross_blocks`, then the deep span's **shifted past the root nodes**,
      keeping the labels the deep manifold gives them. Levels are distances in the glued
      graph, so how the deep span roots *itself* is discarded; that discard is what makes MFA
      depth 2 rather than 3, and the numbering invariant rejects the cases where it would
      have been merely convenient (`tests/graphical.py::_misrooted`).
- [ ] **`cut(far_node)` / `CliqueCut`** — the **re-rooting isomorphism**: the same
      coordinates regrouped as `(near | crossing | far)`. `CliqueCut` is the validated index
      record, built in `algebra`; `project_cut` / `join_cut` are the array operations that read
      one. Everything about *which* block goes where is decided in `algebra`.
      **It reads the layout, not the graph.** It used to read `canonical_cliques` and index
      `clique_dims` with the result; two layouts over one graph differing only in the storage
      position of two equal-sized blocks produced an *identical* `CliqueCut` with the near and
      far blocks swapped, silently.
      Note `project` zero-fills: the crossing matrix has a row band for **every** near clique,
      so a near clique with no crossing partner contributes zeros. That is what makes `cross`
      a dense linear map from the far side into the whole near parameter vector rather than a
      ragged collection.
- [ ] **`RootEmbedding`** is bounded by `LevelCliques`, which is why it lives beside it
      rather than in `embedding.py`.
- [ ] `span_nodes` / `span_blocks` each carry one documented `isinstance` fallback. **Note**:
      `span_blocks`'s docstring used to justify the fallback with "a learned variational
      correction, for instance" — a case realized nowhere. That justification is removed; the
      fallback's actual job is wrapping leaf families as single nodes.
- [ ] **`tensor` versus `select_joint`**, the one place to double-check the reasoning:
      `tensor` multiplies its members' statistics together, exact only when every member is
      observed. When two or more members are latent the block is
      $\mathbb E[\bigotimes_i \mathbf s_i]$ **jointly**, and expectation does not pass through
      a tensor product. Using `tensor` there would be silently wrong — measured gap on MFA is
      order one.
- [ ] **The structural condition** this implies: *a clique's latent members must themselves be
      a clique of the level above*, so their joint expectation exists as a block to select
      from. `LinearCliques.clique_index` raises when it does not hold. **Is that the right
      place?** It fires at use rather than at construction.
- [ ] **Known wart**: `partial_contract` and `select_joint` document `keep` as naming
      *members* but index `selectors` by tuple **position**. These coincide only when members
      are `(0..n-1)`, true of every clique in the library. Fix the code or the docstrings;
      either way it needs validation.

*Pinned by* `tests/graphical.py` (620 lines, 35 test functions — layouts, spans, cuts,
ordering regressions) and `tests/ef_clique.py` (332 lines, 16 — `EFClique` against every live
interaction shape). **`tests/ef_clique.py` passed unmodified through the whole collapse**,
which is the equivalence proof the merge rests on.

---

## 3. Harmonium — `geometry/exponential_family/harmonium.py` (617 lines)

- [ ] **`cross_blocks` replaced `int_members`.** A model now declares `tuple[EFClique, ...]`
      outright instead of member tuples that something else turns into cliques. The base
      class derives the single-block case via `block_clique`; a multi-block interaction with
      no override raises a message naming the block count. **Deleted with it**: `_MapClique`,
      `_block_axes`, `_factor_dims`.
- [ ] **What this fixed.** `_block_axes` had three branches, two of which returned
      `(block.dim,)` — arity 1 for a two-node coupling. I checked every shipped `int_man`:
      all are `EmbeddedMap` or `BlockMap` of `EmbeddedMap`, all `Rectangular`, so those
      branches were dead and `prod(axes) == block.dim` holds throughout. They were a live
      trap for the next model, not a current bug.
- [ ] **The limitation this makes explicit.** A clique's `dim` is `prod(axes)`, so an
      interaction with a *constrained* matrix rep (symmetric, say) cannot be a clique at all —
      `prod(axes)` would exceed its parameter count. No shipped model has one, and failing
      loudly is arguably right: a non-dense block has no per-member tensor factorization.
      **Confirm that is the intended boundary.**
- [ ] Base is still `LevelCliques[Observable, LinearMap[Posterior, Observable], Posterior]`
      and `split_level` still returns exactly `(obs, int, lat)`, so everything written against
      the level split is unchanged however deep the graph goes.
- [ ] **`cross_man` *is* `int_man`.** No wrapper. Which nodes its blocks couple is reported
      separately by `cross_blocks`. That is the one remaining seam between the map route and
      the clique route — see §5a.
- [ ] `HarmoniumEmbedding` and its three slots address the **level split**, not individual
      cliques, so they stay correct as a model's block count grows.
- [ ] **Unchecked invariant, still unchecked.** `split_level` slices by *span dims* while
      `clique_offsets` accumulates *block dims* — two independent computations of the same
      three boundaries, across ~40 `split_level` call sites. I verified they agree on all
      four shipped models (root, cross, and deep, recursing into deep spans) but nothing in
      the code enforces it. **Worth a permanent test.**

---

## 4. What did not change — `manifold/` (skim)

- [ ] **`combinators.py` (267 lines) is unchanged apart from its docstring.** Same six
      classes: `Null`, `Tuple`, `Pair`, `Triple`, `Quadruple`, `Replicated`.
- [ ] **`map.py` (644 lines): `LinearMap` is not a clique** — verified: it imports only
      `algebra.matrix`, `.base`, `.combinators`, `.embedding`. A clique *is* a multilinear
      form; a linear map is what a model hands over.
- [ ] **`SquareMap` has no clique notion** — the question is not asked. A square map is a
      self-interaction *inside* one node, and this is where structured matrix representations
      live. `Covariance` and `CouplingMatrix` are both of this shape.
- [ ] `MultilinearMap` — `sub_dims`, `tensor`, `contract`. The arity-$n$ generalization of
      `Rectangular`, knowing nothing about graphs.
- [ ] `BlockMap.blocks` is a validated non-empty `tuple` with a common domain and codomain.
      It was a mutable `list` inside a frozen dataclass, which also made `BlockMap`
      unhashable — relevant since models are static jit arguments.
- [ ] **Cross-references to the deleted module were fixed by hand.** Sphinx does not fail on
      dangling roles — nitpick is off — so `map.py:459-460`, `combinators.py:5`, and
      `algebra/clique.py:4` were repointed manually. Re-grep for `manifold.clique` before
      merging.

---

## 5. The models

### 5a. MFA — `models/graphical/mixture.py` (610 lines)

- [ ] **Declares three cliques**, not three member tuples:
      `block_clique(xy, (0,1))`, `block_clique(xyk, (0,1,2))`, `block_clique(xk, (0,2))`.
      Node count, root count, the three biases, the $(y,k)$ coupling from the mixture above,
      the levels, and the block layout all follow.
- [ ] **The graph is a fork, not a chain.** Both $y$ and $k$ are adjacent to $x$ through the
      three-way clique, so levels are $(1,2)$ and depth is 2. Consequence:
      `_CATEGORY_NODE = 2` cannot be replaced by `levels[-1]`, because the deepest level
      holds $y$ as well as $k$. (Tried; seven tests failed.)
- [ ] `to_mixture_coords` / `from_mixture_coords` are three lines each via `CliqueCut`,
      replacing ~41 lines of hard-coded offsets. `whiten_prior` and `to_natural_likelihood`
      both flow through them.
- [ ] **Still an arity-2 bridge — the main remaining Phase 2 item.** The $(x,y,k)$ block is
      *stored and executed* as an arity-2 `EmbeddedMap` whose domain is the joint $(y,k)$
      statistic. The graph now reports arity 3 *through the selectors themselves*, and
      `tests/ef_clique.py` proves an `EFClique` reproduces it exactly including in mean
      coordinates — but production still runs the map. The gap is narrower than it was:
      `block_clique` derives the arity-3 clique from the same objects the map uses, so the two
      descriptions can no longer drift.

*Pinned by* `tests/graphical_mixture.py` (410 lines, 25 test functions).

### 5b. CCA — `models/harmonium/cca.py` (227 lines)

- [ ] The first model over a **multi-root** graph: the fork $x \leftarrow z \rightarrow y$,
      now declared as two `block_clique`s on $(0,2)$ and $(1,2)$, with `n_roots = 2` derived
      from the observable being a `CliqueProduct`.
- [ ] **Conjugation is a sum**, $\rho = \rho_X + \rho_Y$, delegating to two standalone
      `NormalLGM`s. Exact because `DifferentiablePair.log_partition_function` is already a
      sum. Residual measured at 1.4e-16.
- [ ] It is `DifferentiableConjugated`, not analytic: inverting the conjugation sum
      branch-wise needs structure a fork does not supply. **Open question** — if you know the
      closed form, this gains EM.

*Pinned by* `tests/cca.py` (225 lines, 17 test functions).

### 5c. HMoG — `models/graphical/hmog.py`

- [ ] **Declares nothing.** Its three-node chain comes from the default `cross_blocks` plus
      the deep span's own blocks spliced in recursively. This is the extension property
      working: a hierarchical model needs no graph declaration at all.
- [ ] The asymmetry is readable off the layout: the $x$–$y$ coupling touches only the
      *location* sub-statistics; the $y$–$k$ coupling touches $y$'s **full** statistic.

### 5d. Leaves — `models/base/*`

- [x] **`Node` was deleted**, along with its 12 mixin sites. It reported exactly what the
      `span_blocks` fallback computes, and nothing did `isinstance(x, Node)`.

---

## 6. Tests

- [ ] **`tests/clique.py` is the one to read first** — 445 lines, 52 test functions,
      sub-second, no JAX. It now owns the partition and cut-index logic outright, including
      the LGM partial-coupling refusal.
- [ ] **`tests/graphical.py` still tests `exponential_family/clique.py`**, which breaks
      CLAUDE.md's one-file-per-module convention. Now that `manifold/clique.py` is gone the
      name has nothing behind it. **Rename to `tests/layouts.py`? Merge into
      `tests/ef_clique.py`?** This is a naming decision, listed in §8.
- [ ] **`_Block` became `_block(members, axes)`**, a factory over `Poissons(n)` nodes
      (`Poissons(n).dim == n`, so the seven call sites keep their numbers). The fixture now
      has real node families instead of stated integers — check you find that an improvement
      rather than an obfuscation.
- [ ] **The two that matter most** are in `graphical.py::TestDeclarationOrderRegressions`.
      Both reproduce silent corruption in the pre-consolidation code and both pass: reversed
      CCA branches and a mis-rooted deep span. `_ReversedCCA` now overrides `cross_blocks`
      rather than `int_members`, and still reverses the *members* while keeping block order.
- [ ] `TestLayoutInvariants` sweeps every shipped model through `layout_problems`, four
      invariants in dependency order. **Storage order == canonical order is the first and
      returns alone** — the single enforcement point for canonical ordering in the codebase.
- [ ] The judgement to make: do these test the *contract* or the *implementation*? The layout
      pins in `graphical.py` are the ones I would trust least to be testing the right thing.

---

## 7. Docs and examples — mostly skim

- [ ] **RST now mirrors the two-module structure.** `manifold/clique.rst` deleted and dropped
      from `manifold/index.rst`; `exponential_family/clique.rst` grew an inheritance diagram
      and sections for the layouts, spans, cut views, and span embeddings;
      `algebra/clique.rst` grew Partitions and Cuts sections.
- [ ] `CLAUDE.md`'s module map and test-file table both still describe
      `manifold/clique.py`. **Not yet updated** — it is the one file this round did not touch,
      because the test-file naming decision (§8) changes what the table should say.
- [ ] **The example churn is formatting.** ~30 example files changed; the only one touching
      the clique API is the new `examples/cca/`. Verify with
      `git diff 4b957a6 -- examples/ | grep -E "^[+-]" | grep -iE "graph|Clique|split_level"`.

---

## 8. Decisions only you can make

- [ ] **Test-file naming.** `tests/graphical.py` tests `exponential_family/clique.py`. Rename,
      merge, or amend the convention in `CLAUDE.md`. This blocks the `CLAUDE.md` update.
- [ ] **Did `LevelCliques` belong in `Harmonium` after all?** §2 explains why it was moved
      instead, and the cost of absorbing it. Yours to overrule.
- [ ] **Should `algebra` know about block sizes?** `cut_indices` takes `dims`. Pure integers,
      no JAX — but it is no longer *only* about the graph.
- [ ] **`InteractionEmbedding` and `PosteriorEmbedding` are unused in `models/`**, fully
      subsumed by `CliqueBlockEmbedding`. Still coherent span-level public API, still tested.
      Delete or keep? (`ObservableEmbedding` *is* used, so this is not a whole-family
      question.)
- [ ] **Is `LinearCliques.clique_index` the right home** for the "latent members must form a
      clique above" check? It raises at use, not construction.
- [ ] **How far should "clique-based" go?** The graph and layout are clique-based everywhere,
      but the *algorithms* are not: ~40 compute sites use `split_level` and `split_cliques`
      has **zero** production callers. Converting the rest is churn unless something needs
      per-clique dispatch — which is exactly what interleaving exact and variational cliques
      would need.
- [ ] **Whether a vector-valued product is one node or many** is a modelling choice, not a
      property of the manifold. The `span_blocks` fallback hard-codes "one". Binds if you ever
      want per-neuron couplings.
- [ ] **`EFClique`'s positional `keep`** — see §2. Fix the code or the docstrings.
- [ ] **`CanonicalCorrelationAnalysis`** exposes no canonical directions or correlations. It is
      probabilistic CCA. Rename or document the distinction.
- [ ] **Fork EM** — see §5b.

---

## 9. Tried and rejected — so you need not re-derive them

**This round.**

- **`partition(S)` returning rows eagerly.** Rejected on evidence: the cover `((0, 1),)`
  rooted at `{0}` has a crossing clique with no row block, so `root_cliques` would start
  raising on a cover it handles. Rows became a separate partial property.
- **Collapsing `CliqueBlockEmbedding.project` into `EFClique.select_joint`.** They walk the
  same axes but read their shape from different places (ambient layout vs selectors). Kept
  separate over one shared `_map_axis`.
- **Absorbing `LevelCliques` into `Harmonium`.** Rejected because it makes
  `tests/graphical.py::_Spans` — declared graph vs derived layout — unexpressible, and forces
  a layout fixture to become a full `Gibbs` family. Moved to `exponential_family/clique.py`
  instead; `manifold/clique.py` is deleted either way.
- **A strict-ABC `LinearClique` with both `members` and `axes` abstract.** Built and measured:
  42 failures, all from `EFClique(members=...)` no longer being accepted once the field became
  `_members`. The hybrid (field `members`, abstract `axes`) was strictly better — and is now
  moot, since `LinearClique` is deleted entirely.
- **`CliqueProduct` in `exponential_family/combinators.py`.** The plan's destination, on the
  grounds that `ExponentialFamilyPair` is its only client. Put it in
  `exponential_family/clique.py` instead: it is a *layout*, and combinators.py is about flat
  concatenation.

**Earlier rounds.**

- **`Cliques.cliques` vs `Cliques.scopes`.** Renaming the container to `LinearCliques` and its
  accessor to `clique_blocks` removed the collision outright, so the graph layer keeps the
  mathematical word.
- **`deep_cliques` → `higher_cliques`.** "Higher" already carries arity and tensor order;
  "deep" says position and stays parallel with `LevelCliques`'s `deep_man`.
- **A public `is_canonical` / `check_layout` on `LinearCliques`.** Rejected twice: nothing at
  runtime consults canonical order, so the guard would defend a path no code takes. The
  invariant is stated once, in the test that sweeps every shipped model.
- **Permuting `clique_dims` into canonical order.** Unsound. `clique_offsets` slices real
  coordinates, and storage order is fixed by `split_coords`, then `BlockMap.coord_blocks`,
  then the deep manifold's own layout.
- **`Clique` carrying `members` as a field on the *manifold*.** Blocked: a base-class
  dataclass field becomes the first positional field of every subclass —
  `Normal(3, Diagonal())` would bind `members=3`.
- **Collapsing `MapClique.n_members` into an inherited `arity` property.** Python forbids a
  dataclass field shadowing an inherited property. `arity` turned out to be `len(axes)`.
- **Unifying `Cliques` with `JunctionTree`** — ruled out on instruction. A `ChordalBoltzmann`
  is a monolithic vector-valued node; its junction tree is internal to its own log-partition.
- **Three `*Hierarchical` ABCs absorbed by `Harmonium`.** Deleted instead; their only consumer
  was `hmog.py`, which carries a private `_HMoGBase`.

---

## Not done, deliberately

- **Interleaving conjugated and variational-conjugated levels** — the stated objective, not
  started. Shape: a fourth span slot, `[root | cross | deep | rho]`, since $\rho$ is a
  per-level correction on the deep span's root. **Trap**: ~10 sites destructure
  `split_coords` positionally into three `Array`s, so reordering slots breaks them *silently*
  — rename the method in the same change to force every site to be touched.
- **`EFClique` as the *execution* primitive.** `cross_blocks` now returns real `EFClique`s, so
  the remaining step is deriving `int_man` from them rather than the reverse. MFA's arity-3
  block is the only place the two descriptions still differ in how they execute.
- **A permanent test that span dims equal summed clique dims.** Verified by hand this round on
  all four models; nothing enforces it. See §3.
- **`sufficient_statistic` as a flat per-node clique loop** — would need a per-node data
  split. Real risk, unclear gain.

---

## Verification — commands and results, 2026-08-26

Rerun these rather than trusting them; all are cheap except the last.

| gate | command | result |
|---|---|---|
| lint | `uvx ruff check src/ tests/` | clean |
| format | `uvx ruff format --check src/ tests/` | clean |
| types | `uvx basedpyright src/ tests/` | 0 errors, 0 warnings |
| docs | `uv run sphinx-build -q docs/source <out>` | exit 0 |
| fast | `uv run python -m pytest tests/clique.py -q` | 69 passed, 0.44s |
| focused | `uv run python -m pytest tests/clique.py tests/graphical.py tests/ef_clique.py tests/multilinear.py -q` | 175 passed, 30s |
| suite | `uv run python -m pytest tests/ -q` | **549 passed**, 17m11s |

**Every stage boundary held.** Baseline 527 → 540 (partition: +13 tests) → 540 (CliqueProduct
moved) → 540 (CliqueBlockEmbedding merged) → 549 (cut indices: +9) → 549 (`int_members`
deleted) → 549 (`manifold/clique.py` deleted). No test was deleted or weakened at any point,
and `tests/ef_clique.py` passed **unmodified** throughout.

| equivalence check | result |
|---|---|
| `partition(root set)` vs `root_cliques`/`cross_cliques`/`deep_cliques` | exact, 4 shipped models + 7 hand-built covers |
| `partition({2})` vs MFA's `mix_cut` | exact on `near_idx`, `cross_idx`, `far_idx`, `cross_rows` |
| span dims vs summed clique dims | agree at every level boundary, 4 models, recursing |
| `MultilinearMap` vs `EmbeddedMap`, arity 2 | exact, both contraction directions |
| `EFClique` vs live interactions | exact, 7 shapes × 3 operations |
| Arity-3 clique vs `xyk_man` at the E-step | exact, over 16 posterior draws |
| CCA fork conjugation residual | 1.4e-16 (LGM control: 2.2e-16) |

**Examples re-run after the collapse**: `mfa`, `hmog`, `cca`, `mixture_of_gaussians` — all
four exit 0. CCA reproduces its pre-collapse numbers **exactly**: latent alignment RMSE
0.2403, cross-covariance relative error 0.0070, over 8000 optimizer steps. That is the
strongest single check here, since CCA is the multi-root fork. The other 14 examples have not
been re-run since before the consolidation; `variational_mnist` needs an MNIST download.
Slowest of the full set: `boltzmann_lgm` 800s, `pendulum` 585s, `chordal_boltzmann_ppc` 273s.

The `cuInit(0) failed: Unknown CUDA error 303` line every example prints is JAX's
plugin-discovery probe falling back to CPU. Harmless; not a failure.

**Two known sharp edges, both pre-existing and neither introduced here:**

- Diagonal posteriors do not conjugate exactly — residual ~4.5e-02 for a plain `NormalLGM`,
  ~1.7e-02 for CCA; both ~1e-16 with `PositiveDefinite`. Either an intended approximation or a
  real gap. Worth deciding which.
- `initialize_from_sample` accepts wrongly-shaped data silently: its docstring says the sample
  is for observable biases, but it passes the unsliced input to `obs_man`. Observable-only
  data into a joint harmonium returns non-finite parameters for CCA, finite-but-meaningless
  for LGM.

**A correction to an earlier note in this file.** It once claimed
`NormalCovarianceEmbedding` fails `project ∘ embed = id` by "returning twice the input". That
is wrong. `embed` acts on natural parameters and `project` on mean parameters; they are an
**adjoint** pair, and the `Scale` round-trip scales by $1/d$, not 2. `project ∘ embed = id` is
not this embedding's contract — the `LinearEmbedding` docstring that claims it for all
embeddings is what needs fixing.

**Notes on measuring.**

- Do not compare example numbers while the test suite is running. JAX's threaded CPU
  reductions vary summation order under contention, which moved CCA's last digits over 8000
  optimizer steps and briefly looked like a regression.
- One JAX job at a time — this machine has ~15 GB of RAM and stacked jobs crash it.
- `pkill -n` kills the *newest* match. Do not use it to prune a stale suite.
