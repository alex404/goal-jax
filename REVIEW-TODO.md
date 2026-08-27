# Reviewing `clique-container`

**Scope**: 7 commits (`070a231..ca0763d`) plus an uncommitted working tree, against `main` at
`4b957a6`. 90 files, +5894/−973.

**What the branch does.** Model graphs are first class. A model declares which nodes its
interactions couple; the graph, the level structure, and the parameter layout are all *derived*
from that. `Harmonium` is a layout over a graph instead of a hard-coded triple, which is what
lets a fork (CCA) and a three-way interaction (MFA) be expressed at all.

**The one idea to hold onto.** A clique is one inner product in the log-density,
$\langle \theta_c, \bigotimes_{i \in c} \pi_i(\mathbf s_i(x_i)) \rangle$. Arity is all that
varies: 1 is a bias, 2 a matrix contraction, 3+ a tensor. So a clique **is** a multilinear map
--- `EFClique.form` is literally `MultilinearMap(axes)`. A covariance is *not* a clique: it is
a node's internal second moment, and structured matrix representations live inside nodes,
never across them.

**The hierarchy under review.** `Harmonium` → `LevelCliques[Observable, LinearMap, Posterior]`
→ `LinearCliques` → `Cliques`. On `main`, `Harmonium` inherited `Tuple`.

Checkboxes are yours to tick. Each section depends only on the ones above it.

| § | file | lines | budget |
|---|---|---|---|
| 1 | `geometry/algebra/clique.py` | 281 | 20 min --- read first; everything else is a layout over it |
| 2 | `geometry/manifold/clique.py` | 423 | 45 min --- the core of the branch |
| 3 | `geometry/exponential_family/clique.py` | 410 | 30 min |
| 4 | `geometry/manifold/cut.py` | 192 | skim --- bracketed, one user |
| 5 | `geometry/exponential_family/harmonium.py` | 629 | 20 min --- small diff, one sharp decision |
| 6 | `models/` | --- | 20 min |
| 7 | tests | --- | 20 min |

---

## 1. The graph --- `algebra/clique.py`

`Cliques(ABC)` alone. A rooted graph given as a clique cover, over integer indices. No JAX, no
`Manifold`, and nothing about parameters.

- [ ] **The contract is `cliques` and `n_roots`.** Adjacency, levels, boundary, the level
      split, and canonical order are all derived from those two. Is that the right contract,
      or should `n_nodes` be declared as well?
- [ ] **`n_nodes` is derived as `max(max(c)) + 1`**, which it must be, since a layout has no
      separate declaration to read it from. The cost: a too-large index now describes a
      *bigger* graph rather than being out of range, so two declared-vs-derived validation
      cross-checks no longer exist.
- [ ] **The library ships no instance of `Cliques`.** The concrete cover lives in
      `tests/clique.py` as `Cover`. Confirm you want no bare-graph value type in `src/`.
- [ ] **`canonical_cliques` is a single sort** keyed on (lowest level touched, whether the
      clique crosses, storage position). A clique lies within one level or crosses two
      consecutive ones and there is no third case, which is what makes the key total.
- [ ] **`level_split()` is the one split**, returning root / cross / deep as *positions* in
      `cliques` so a caller can index anything held alongside the cover. It takes no argument:
      an arbitrary node set is `CliqueCut`'s business (§4). Indices ascend with level, so
      membership is a comparison and needs no traversal.
- [ ] **Members must ascend within a clique** --- `(1, 0)` raises. Nothing normalizes covers,
      so without this `(0, 1)` and `(1, 0)` are unequal tuples naming one clique and the
      duplicate check misses it. A rule the library did not have before; judge whether you
      want it.
- [ ] **`validate()` is public and nothing calls it automatically.** A layout's cover arrives
      one block at a time, so there is no moment at which it is first complete. **This is a
      real gap**: an incoherent graph is caught only when someone calls it, which is exactly
      how the `mix_man` defect in §5 survived. Decision in §9.
- [ ] **Validation is two private methods for a bad reason.** `_validate_members` and
      `_validate_graph` split only because merging them trips ruff's complexity limit at
      14 > 10. Say if you would rather see one method and a raised limit.
- [ ] **`same_graph(other)` is misnamed.** It compares cover *order* as well as content, so
      it asks whether two layouts store their blocks alike --- stricter than "same graph".
      Both callers (`RootEmbedding`) want the strict version. Consider `same_cover`.
- [ ] **Read the module docstring as mathematics**, not as design justification --- that was
      the point of the last pass over it. Check the level-gap argument and the
      numbering-by-level convention read correctly.

*Pinned by* `tests/clique.py` --- 57 tests, 0.5 s, no JAX. It defines `Cover` and an `ascend`
helper; the ascent is a test-only notion, since a layout's deep span is already the graph one
level up.

---

## 2. The layout --- `manifold/clique.py`

`Clique`, `LeafClique`, `LinearCliques`, `LevelCliques`, `CliqueProduct`. The clique-indexed
sibling of `combinators.py`: where `Pair` and `Triple` split a coordinate vector by arity,
these split one by *which nodes each block couples*.

- [ ] **The manifold *is* its graph.** `LinearCliques.cliques` reads the cover off
      `clique_blocks`, so there is no second description to disagree with the first. This is
      the fix for the branch's worst bug class: an offset and the members it belongs to now
      come from the same object, so a wrong slice is unconstructible.
- [ ] **Does the module earn its place?** It has no client inside `manifold/` --- the argument
      for it is layering by kind, that `LinearCliques` belongs beside `Tuple` regardless of
      who imports it. The alternative not taken was putting `Clique` and `LinearCliques`
      *into* `combinators.py`. Yours to overrule.
- [ ] **`Clique`'s contract is `members` and `on`.** `members` is a declared annotation rather
      than an abstract property, because a dataclass field does not satisfy `@abstractmethod`
      — verified, instantiation raises — and a property would have forced
      `EFClique(_members=...)`. `axes` defaults to `(dim,)`, the only thing a one-node block
      can mean, and `EFClique` overrides it from selectors.
- [ ] **`axes` sits on `Clique`, not only on `EFClique`.** Forced: `CliqueBlockEmbedding`
      indexes singleton blocks and reads their axes. `MultilinearMap` was already
      manifold-level, so `axes` and `form` sit beside `form`'s own type. What stays EF-only is
      *selectors*. Check the line is in the right place.
- [ ] **`LeafClique` plus `LinearCliques.blocks_of` is how a plain manifold occupies a node.**
      `blocks_of(span)` returns the span's own blocks if it is a layout, else
      `(LeafClique(span, (0,)),)`. One fallback, and the node count derives from the blocks.
- [ ] **`LevelCliques.root_blocks` is the subtlest thing in the branch.** By default a
      structured root span is *expanded*, one node per node of its own --- that is what gives
      CCA two root nodes. Overriding with a single block holds the span as one node, which is
      right whenever the level's interaction couples it as a unit and so cannot factor per
      node. `Mixture` overrides it. Read §5 before judging.
- [ ] **`split_coords` takes its two offsets from the blocks**, via `level_split()`, not from
      `root_man.dim` / `cross_man.dim`. Both readings exist; taking the split from the blocks
      is what makes their agreement structural rather than a coincidence. `validate()`
      compares them explicitly (`sum(clique_dims) == dim`).
- [ ] **`dim` stays independent of `sum(clique_dims)`.** `dim` is about coordinates and blocks
      are a *view* of them, which is what makes "the blocks tile the coordinate vector" a real
      claim rather than a tautology.
- [ ] **Storage order versus canonical order.** Nothing that slices coordinates consults
      `canonical_cliques`. They can part in exactly two places --- block order *within* the
      root span and *within* the cross span, both the model's declaration --- and the only
      enforcement is a test that sweeps every shipped model (§7). Between levels they cannot
      part, since both relabel by `+n_roots`.
- [ ] **`CliqueProduct` is the disjoint union**, every node a root, and the only route to a
      multi-root graph. Two overrides long; `exponential_family/combinators.py` is its one
      client and gives CCA its two-node root span.
- [ ] **The surface is wider than the use.** `split_cliques` and `join_cliques` have **no**
      production callers; `clique_index`, `clique_offsets`, and `clique_axes` have exactly one
      each, all inside `CliqueBlockEmbedding` (MFA-only); `cut` has one. Load-bearing
      everywhere are `clique_blocks`, `cliques`, `clique_dims`, `validate`. Prune or keep?
- [ ] **A naming tension, unresolved.** `LinearCliques` is n-ary (`split_cliques` /
      `join_cliques`) while `LevelCliques` is ternary (`split_coords` / `join_coords`): two
      combinator shapes on one hierarchy, distinguished only by method name.

*Pinned by* `tests/graphical.py` --- 53 tests, layouts and spans and cut indices.

---

## 3. Statistics on cliques --- `exponential_family/clique.py`

`EFClique`, `CliqueBlockEmbedding`, `block_clique`, `RootEmbedding`.

- [ ] **`EFClique` is the only clique that knows families.** One **selector** per member --- a
      `LinearEmbedding` from the sub-statistic the clique couples into that node's full
      statistic. `axes` is read off them, so `len(members) == len(selectors) == len(axes)`
      holds by construction, and `dim = prod(axes)`.
- [ ] **The boundary this draws.** `dim = prod(axes)` means an interaction with a *constrained*
      matrix rep (symmetric, say) cannot be a clique at all. No shipped model has one --- every
      interaction is `Rectangular` --- and failing loudly is arguably right, since a non-dense
      block has no per-member tensor factorization. Confirm that is the intended boundary.
- [ ] **`block_clique(block, members)` is the bridge from the map API.** One rule: the codomain
      embedding is the first selector, the domain embedding supplies the rest --- one selector
      if it addresses a single node, several if it addresses a joint block over a group. That
      second case is what makes MFA's block arity 3. `members` must still be passed: arity is
      readable from the block, which nodes it couples is not.
- [ ] **`tensor` versus `select_joint` --- check this reasoning.** `tensor` multiplies its
      members' statistics together, exact only when every member is *observed*. With two or
      more latent members the block is $\mathbb E[\bigotimes_i \mathbf s_i]$ **jointly**, and
      expectation does not pass through a tensor product; the joint block comes from the level
      above and the selectors are contracted into it. Using `tensor` there is silently wrong
      by order one on MFA.
- [ ] **The structural condition that implies**: a clique's latent members must themselves be
      a clique of the level above, so their joint expectation exists as a block to select
      from. `LinearCliques.clique_index` raises when it does not hold --- at use, not at
      construction. Right place?
- [ ] **`CliqueBlockEmbedding` is an `EFClique`.** Same fields; one owns a block, the other
      addresses another manifold's. `sub_man` is just `self.form`.
- [ ] **`project` and `select_joint` were not merged**, though they walk the same axes. They
      take their shape from different places --- `select_joint` from the selectors, `project`
      from `amb_man.clique_axes` --- which agree for every shipped model but need not in
      general. They share one `_map_axis`. Real distinction or over-careful?
- [ ] **Known wart**: `partial_contract` and `select_joint` document `keep` as naming *members*
      but index `selectors` by tuple **position**. These coincide only when members are
      `(0..n-1)`, true of every clique in the library. Fix the code or the docstrings.
- [ ] **`RootEmbedding` is bounded by `LevelCliques`**, which is why it lives here rather than
      in `embedding.py`.

*Pinned by* `tests/ef_clique.py` --- 44 tests, `EFClique` against every live interaction
shape, all three operations. It passed **unmodified** through every restructuring this branch
did, which is the equivalence evidence the design rests on. `tests/multilinear.py` (22) pins
`MultilinearMap` against `EmbeddedMap` at arity 2 and associativity at arity 3.

---

## 4. The re-rooting view --- `manifold/cut.py` (bracketed)

- [ ] **`CliqueCut(cliques, clique_dims, far_node)`** is an isomorphism, not a computation: the
      same coordinates regrouped as `(near | crossing | far)`. Five index tuples derived in
      `__post_init__`; `project` and `join` are methods.
- [ ] **It reads the layout, not the graph.** It used to read `canonical_cliques` and index
      `clique_dims` with the result, so two layouts over one graph differing only in the
      storage position of two equal-sized blocks produced an identical cut with the near and
      far blocks *swapped*, silently.
- [ ] **`project` zero-fills.** The crossing matrix has a row band for every near clique, so a
      near clique with no crossing partner contributes zeros --- which is what makes `cross` a
      dense linear map from the far side into the whole near parameter vector rather than a
      ragged collection.
- [ ] **One user**: `models/graphical/mixture.py`, two sites. `CliqueBlockEmbedding` and the
      arity-3 path are bracketed with it --- carried across and kept green, not redesigned.
      Deferred question: a cut is `level_split` at a single node plus dimensions, so it could
      be a refinement of the level split rather than an independent record.

---

## 5. `exponential_family/harmonium.py`

- [ ] **The base is `LevelCliques[Observable, LinearMap[Posterior, Observable], Posterior]`**
      and `split_level` still returns exactly `(obs_params, int_params, lat_params)`. However
      deep the graph, those three spans stay contiguous, so everything written against the
      level split is unchanged.
- [ ] **`cross_blocks` replaced `int_members`.** A model declares `tuple[EFClique, ...]`
      outright rather than member tuples something else turns into cliques; the base class
      derives the single-block case via `block_clique`. Deleted with it: `_MapClique`,
      `_block_axes`, `_factor_dims` --- the last of which had two branches returning
      `(block.dim,)`, i.e. arity 1 for a two-node coupling, dead for every shipped model but a
      live trap for the next one.
- [ ] **`cross_man` *is* `int_man`** --- no wrapper. Which nodes its blocks couple is reported
      separately, by `cross_blocks`. That is the one remaining seam between the map route and
      the clique route.
- [ ] **The guard, and the defect it replaces --- read this carefully.** `cross_blocks`
      hardcodes members `(0, 1)`, which names the latent only when the observable span is a
      *single* node. `mix_man` --- the `CompleteMixture` whose observable is a whole harmonium,
      and the view MFA reads itself through --- expands into two root nodes, so `(0, 1)` named
      two roots and the category sat unreachable at node 2:

      ```
      cliques  : ((0,), (0, 1), (1,), (0, 1), (2,))    <- (0,1) twice
      validate : ValueError: duplicate cliques: [(0, 1)]
      ```

      The interaction genuinely cannot be a three-node clique --- it couples the base
      harmonium's whole 21-number parameter vector, which is not a tensor product of node
      statistics --- so two nodes is the correct reading. The fix is `LevelCliques.root_blocks`
      (the hook), `Mixture.root_blocks` (holds the observable as one node), and a guard here
      that **raises** when the interaction has several blocks, or the root span occupies
      several nodes, and the subclass has not overridden. **Judge whether the guard belongs
      here or whether models should validate at construction (§9).**
- [ ] **`HarmoniumEmbedding` addresses the level split**, not individual cliques, so it stays
      correct as a model's block count grows.

---

## 6. The models

**MFA --- `models/graphical/mixture.py` (610)**

- [ ] Declares three cliques rather than three member tuples: `block_clique(xy, (0,1))`,
      `block_clique(xyk, (0,1,2))`, `block_clique(xk, (0,2))`. Node count, root count, biases,
      the $(y,k)$ coupling from the mixture above, levels, and layout all follow.
- [ ] **The graph is a fork, not a chain.** Both $y$ and $k$ are adjacent to $x$ through the
      three-way clique, so levels are $(1,2)$ and depth is 2. Consequence: `_CATEGORY_NODE = 2`
      cannot become `levels[-1]`, because the deepest level holds $y$ too.
- [ ] `to_mixture_coords` / `from_mixture_coords` are three lines each via `CliqueCut`,
      replacing ~41 lines of hard-coded offsets; `whiten_prior` and `to_natural_likelihood`
      both flow through them.
- [ ] **Still an arity-2 bridge --- the main remaining gap.** The $(x,y,k)$ block is *stored
      and executed* as an arity-2 `EmbeddedMap` over the joint $(y,k)$ statistic. The graph
      reports arity 3 through the selectors, and `tests/ef_clique.py` shows an `EFClique`
      reproduces it exactly including in mean coordinates, but production still runs the map.
      `block_clique` derives the clique from the same objects the map uses, so the two
      descriptions cannot drift.

**CCA --- `models/harmonium/cca.py` (227)**

- [ ] The first model over a **multi-root** graph: the fork $x \leftarrow z \rightarrow y$, two
      `block_clique`s on $(0,2)$ and $(1,2)$, `n_roots = 2` derived from the observable being a
      `CliqueProduct`.
- [ ] **Conjugation is a sum**, $\rho = \rho_X + \rho_Y$, delegating to two standalone
      `NormalLGM`s --- exact because `DifferentiablePair.log_partition_function` is already a
      sum. Residual 1.4e-16 (LGM control 2.2e-16).
- [ ] It is `DifferentiableConjugated`, not analytic: inverting the conjugation sum
      branch-wise needs structure a fork does not supply. If you know the closed form, this
      gains EM.
- [ ] It exposes no canonical directions or correlations --- it is probabilistic CCA. Rename or
      document the distinction.

**HMoG --- `models/graphical/hmog.py` (369)**

- [ ] **Declares nothing.** Its three-node chain comes from the default `cross_blocks` plus the
      deep span's own blocks spliced in recursively. This is the extension property working: a
      hierarchical model needs no graph declaration at all.
- [ ] The asymmetry is readable off the layout: the $x$–$y$ coupling touches only *location*
      sub-statistics, the $y$–$k$ coupling touches $y$'s full statistic.

---

## 7. Tests

- [ ] **`tests/clique.py`** (57, sub-second, no JAX) --- pure combinatorics, via its own
      `Cover` and `ascend`.
- [ ] **`tests/graphical.py`** (53) --- layouts, spans, `CliqueCut` indices with every error
      message pinned verbatim, and the ordering regressions.
- [ ] **The two that matter most** are in `TestDeclarationOrderRegressions`: reversed CCA
      branches and a mis-rooted deep span. Both reproduce silent corruption in the
      pre-consolidation code and both pass.
- [ ] **`TestLayoutInvariants`** sweeps every shipped model through `layout_problems`, four
      invariants in dependency order. Storage order == canonical order is the first and
      returns alone --- **the single enforcement point for canonical ordering in the
      codebase.**
- [ ] **`tests/graphical_mixture.py`** (39) --- includes the two `mix_man` regressions: it is a
      two-node graph, and expanding the observable is refused rather than mislabelled.
- [ ] **`_block(members, axes)`** builds fixtures over `Poissons(n)` nodes rather than stated
      integers. Improvement or obfuscation?
- [ ] **The judgement to make**: do the layout pins test the *contract* or the
      *implementation*? Those in `graphical.py` are the ones I would trust least.
- [ ] **`tests/graphical.py` covers three modules** --- `manifold/clique.py`,
      `manifold/cut.py`, and part of the EF module --- which the naming convention does not
      allow for. Decision in §9.

---

## 8. Docs and examples --- skim

- [ ] RST mirrors the module structure: `manifold/clique.rst` and `manifold/cut.rst` are new
      and listed in `manifold/index.rst`; `algebra/clique.rst` documents `Cliques` alone.
- [ ] **`examples/cca/run.py`** is the only example touching the clique API. The other ~30
      changed example files are formatting; verify with
      `git diff 4b957a6 -- examples/ | grep -E "^[+-]" | grep -iE "graph|Clique|split_level"`.
- [ ] **`CLAUDE.md` is stale** --- its module map still describes the old layout. Blocked on
      the test-file naming decision in §9. The only row updated so far is `clique.py`'s.
- [ ] Sphinx does not fail on dangling roles (nitpick is off), so cross-references to moved
      names were repointed by hand. Re-grep for `manifold.clique` and `exponential_family.clique`
      before merging.

---

## 9. Decisions only you can make

- [ ] **Should models validate their graph at construction?** `Cliques.validate()` is public
      and manual. The `mix_man` defect (§5) survived precisely because nothing called it. A
      `__post_init__` on concrete models would catch the next one, at the cost of a BFS per
      construction.
- [ ] **Test-file naming.** `tests/graphical.py` covers three modules. Rename, split, or amend
      the convention. **This blocks the `CLAUDE.md` update.**
- [ ] **Does `manifold/clique.py` earn its place** as a module with no client inside
      `manifold/`? (§2)
- [ ] **`same_graph` or `same_cover`?** (§1)
- [ ] **Prune `LinearCliques`' unused surface?** `split_cliques` / `join_cliques` have no
      production callers. (§2)
- [ ] **`InteractionEmbedding` and `PosteriorEmbedding` are unused in `models/`**, subsumed by
      `CliqueBlockEmbedding`. Still coherent span-level API, still tested. Delete or keep?
      (`ObservableEmbedding` is used, so this is not a whole-family question.)
- [ ] **How far should "clique-based" go?** The graph and layout are clique-based everywhere;
      the *algorithms* are not --- ~40 sites use `split_level`. Converting them is churn unless
      something needs per-clique dispatch, which is exactly what interleaving exact and
      variational cliques would need.
- [ ] **Whether a vector-valued product is one node or many** is a modelling choice, not a
      property of the manifold. `blocks_of` hard-codes "one". Binds if you ever want per-neuron
      couplings.
- [ ] **`EFClique`'s positional `keep`** --- fix the code or the docstrings. (§3)

---

## 10. Questions already answered

Reasons, so you need not re-derive them.

- **Why not free functions for the graph operations?** Tried and reverted. "Avoid helper
  classes" is not "prefer free functions": a class naming a mathematical concept carries the
  operations natural to it.
- **Why no bare `CliqueGraph` value class?** It was never constructed on any production path,
  and it had to sort its cover in order to hash --- a second order that disagreed with the
  layout's. Deleting it left one order in the codebase.
- **Why no `ascend_level()`?** Its only caller was `canonical_cliques`, which is now a sort.
  `LevelCliques.deep_man` is the graph one level up, as a manifold.
- **Why does `LeafClique` survive when `LeafCliques` did not?** `EFClique.selectors` is typed
  `LinearEmbedding[Any, ExponentialFamily]`, so nothing at the manifold layer can build a leaf
  block out of it. Removing `LeafClique` means making the root and deep blocks abstract and
  having every model supply its own.
- **Why is `LevelCliques` not absorbed into `Harmonium`?** Absorption makes
  `tests/graphical.py`'s declared-graph-versus-derived-layout fixture unexpressible and forces
  it to become a full `Gibbs` family. `LinearCliques` has to survive as a class regardless.
- **Why not permute `clique_dims` into canonical order?** Unsound. `clique_offsets` slices real
  coordinates, and storage order is fixed by `split_coords`, then `BlockMap.coord_blocks`, then
  the deep manifold's own layout.
- **Why no public `is_canonical` / `check_layout`?** Rejected twice: nothing at runtime consults
  canonical order, so the guard would defend a path no code takes.
- **Why is `Cliques` not unified with `JunctionTree`?** A `ChordalBoltzmann` is a monolithic
  vector-valued node; its junction tree is internal to its own log-partition.
- **Why is `CliqueProduct` not in `exponential_family/combinators.py`?** It is a *layout*;
  combinators is about flat concatenation.
- **Why `deep_cliques` and not `higher_cliques`?** "Higher" already carries arity and tensor
  order; "deep" says position and stays parallel with `deep_man`.

---

## 11. Known sharp edges --- pre-existing, none introduced here

- [ ] **Diagonal posteriors do not conjugate exactly** --- residual ~4.5e-02 for a plain
      `NormalLGM`, ~1.7e-02 for CCA; both ~1e-16 with `PositiveDefinite`. Intended
      approximation or real gap? Worth deciding which.
- [ ] **`initialize_from_sample` accepts wrongly-shaped data silently.** Its docstring says the
      sample is for observable biases, but it passes the unsliced input to `obs_man`.
      Observable-only data into a joint harmonium gives non-finite parameters for CCA,
      finite-but-meaningless for LGM.
- [ ] **`LinearEmbedding`'s docstring claims `project ∘ embed = id` for all embeddings**, which
      is false: `embed` acts on natural parameters and `project` on mean parameters, so
      `NormalCovarianceEmbedding` is an *adjoint* pair and the `Scale` round-trip scales by
      $1/d$. The docstring is what needs fixing.

---

## 12. Not done, deliberately

- **Interleaving conjugated and variational-conjugated levels** --- the stated objective, not
  started. Shape: a fourth span slot, `[root | cross | deep | rho]`, since $\rho$ is a
  per-level correction on the deep span's root. **Trap**: ~10 sites destructure `split_coords`
  positionally into three `Array`s, so reordering slots breaks them *silently* --- rename the
  method in the same change to force every site to be touched.
- **`EFClique` as the execution primitive.** `cross_blocks` returns real `EFClique`s, so the
  remaining step is deriving `int_man` from them rather than the reverse. MFA's arity-3 block
  is the only place the two descriptions still differ in how they execute.
- **Redesigning the bracketed MFA machinery** --- `CliqueCut`, `CliqueBlockEmbedding`, arity 3.
- **Splitting `tests/graphical.py`** --- blocked on §9.
- **`sufficient_statistic` as a flat per-node clique loop** --- needs a per-node data split.
  Real risk, unclear gain.

---

## Verification

Rerun rather than trusting; all cheap except the last.

| gate | command | expected |
|---|---|---|
| lint | `uvx ruff check src/ tests/` | clean |
| format | `uvx ruff format --check src/ tests/` | clean |
| types | `uvx basedpyright src/ tests/` | 0 errors, 0 warnings, 0 notes |
| docs | `uv run sphinx-build -q docs/source <out>` | exit 0 |
| fast | `uv run python -m pytest tests/clique.py -q` | 57 passed, ~0.5 s, no JAX |
| focused | `uv run python -m pytest tests/clique.py tests/graphical.py tests/ef_clique.py tests/multilinear.py tests/cca.py tests/graphical_mixture.py tests/hmog.py -q` | 260 passed, ~5 m |
| suite | `uv run python -m pytest tests/ -q` | **552 passed**, ~18 m |
| examples | `for e in hmog cca mfa mixture_of_gaussians; do uv run python -m examples.$e.run; echo "$e=$?"; done` | exit 0 each |

**The numeric check that matters most**: `examples/cca` reproduces latent alignment RMSE
`0.24031811353161686` and cross-covariance relative error `0.007013174699236388` over 8000
optimizer steps, unchanged across every restructuring. CCA is the multi-root fork and its root
span is a `CliqueProduct`, so it exercises the most of this branch at once.

Equivalences established while building, worth re-establishing if you change the corresponding
code:

| check | result |
|---|---|
| block-derived `split_level` vs span-derived | byte-identical, 10 models, splits and round-trips |
| span dims vs summed block dims, groups contiguous | exact on 18 layouts, incl. deep spans and the harmoniums nested in `hmm` / `kalman_filter` / `vm_population_code` |
| storage order vs canonical order | equal on every shipped model |
| `canonical_cliques` as a sort vs as a recursion | byte-identical on 15 layouts incl. deep spans |
| `MultilinearMap` vs `EmbeddedMap`, arity 2 | exact, both contraction directions |
| `EFClique` vs live interactions | exact, 7 shapes × 3 operations |
| arity-3 clique vs `xyk_man` at the E-step | exact, over 16 posterior draws |

**Measuring notes.** Do not compare example numbers while the suite is running --- JAX's
threaded CPU reductions vary summation order under contention, which moved CCA's last digits
and briefly looked like a regression. One JAX job at a time; this machine has ~15 GB of RAM.
`pkill -n` kills the *newest* match, so do not use it to prune a stale suite. The
`cuInit(0) failed: Unknown CUDA error 303` line every example prints is JAX's plugin probe
falling back to CPU --- harmless.

**Not re-run**: the 14 examples other than `hmog`, `cca`, `mfa`, `mixture_of_gaussians`.
Slowest of the full set are `boltzmann_lgm` 800 s, `pendulum` 585 s, `chordal_boltzmann_ppc`
273 s; `variational_mnist` needs an MNIST download.
