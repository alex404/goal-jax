# Reviewing `clique-container`

**Scope**: 7 commits (`070a231..ca0763d`) plus an uncommitted working tree, against `main` at
`4b957a6`. 90 files, +5894/−973.

**What the branch does.** Model graphs are first class. A model declares which nodes its
interactions couple; the graph, the level structure, and the parameter layout are all *derived*
from that. `Harmonium` is a layout over a graph instead of a hard-coded triple, which is what
lets a fork (CCA) and a three-way interaction (MFA) be expressed at all.

**The one idea to hold onto.** A clique is one inner product in the log-density,
$\langle \theta_c, \bigotimes_{i \in c} \pi_i(\mathbf s_i(x_i)) \rangle$. Arity is all that
varies: 1 is a bias, 2 a matrix contraction, 3+ a tensor. So a clique **is** a linear map ---
A clique is exactly the $\pi_i$ --- one embedding per node, into that node's own manifold --- and
nothing more. Which axes you keep is a *reading*, and materializing one as a `LinearMap` needs to
know what the caller holds, which is `CliqueMap`. Keeping those apart is what lets a layout check
its cliques against what actually sits on it. A covariance is *not* a clique: it is a node's internal second
moment, and structured matrix representations live inside nodes, never across them.

**The hierarchy under review.** `Harmonium` → `LevelCliques[Observable, LinearMap, Posterior]`
→ `LinearCliques` → `Cliques`. On `main`, `Harmonium` inherited `Tuple`.

Checkboxes are yours to tick. Each section depends only on the ones above it.

| § | file | lines | budget |
|---|---|---|---|
| 1 | `geometry/algebra/clique.py` | 218 | 20 min --- read first; everything else is a layout over it |
| 2 | `geometry/manifold/clique.py` | 1123 | 60 min --- the core of the branch |
| 3 | `geometry/manifold/map.py` | 439 | 20 min --- what the cliques took over from |
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
helper; the ascent is a test-only notion, since a layout's deep partition is already the graph one
level up.

---

## 2. The layout --- `manifold/clique.py`

`LinearClique`, `CliqueEmbedding`, `LinearCliques`, `LevelCliques`, `CliqueProduct`,
`RootEmbedding`, plus `node_clique` and `map_axis`. **No map class lives here any more.** The clique-indexed sibling of `combinators.py`: where `Pair` and
`Triple` **concatenate** their components' coordinates, a clique takes their **tensor
product**, and a layout splits a coordinate vector by which nodes each clique couples.

- [ ] **A clique is one embedding per node plus the direction it is read in.**
      `LinearClique` is a `Manifold` with three fields: a `rep`, `node_embs`, and
      `out_axes`. Each embedding goes from the sub-space this coupling uses *into that
      node's own manifold*. `node_mans`, `sub_dims`, `arity`, `in_axes`, `out_dims`,
      `in_dims`, `matrix_shape` and `dim` all derive. **This is the shape the round was for
      --- judge whether anything is missing from it.**
- [ ] **One reading per clique, and the others are other cliques.** `out_axes` names the
      output group, so `matrix_shape` is *the* shape of the parameters and `dim` a property
      of the manifold rather than of a fold anyone imposes. `transposed()` is the two-group
      swap --- the only re-view a structured `rep` can express without densifying --- and
      `to_tensor` is axis-ordered, hence view-independent, which is how any other reading's
      parameters are reached. **This replaced the `keep`-argument suite; judge the
      terminology.**
- [ ] **The map reading moved out.** `CliqueMap` and `TransposedCliqueMap` are **deleted**.
      A reading has to pick a partition of the axes *and* know what manifold the caller
      holds --- neither is a fact about a form, so neither belongs in this module. What
      stays here is `LevelCliques.cross_paths(members)`, which **derives** how each
      partition's coordinates reach the nodes a crossing clique couples. Verified against
      every hand-written path the library used to have --- CCA's `FirstEmbedding`, HMoG's
      `ObservableEmbedding` --- reproduced to the coordinate.
- [ ] **`Interaction` (in `manifold/interaction.py`) is the reading, and it is derived.** One object
      for *all* the crossing cliques: parameters are theirs concatenated, application sums
      them. That retires `BlockMap` from harmoniums entirely --- a fork and a three-way
      interaction are the same object as a chain, with more placements. A model states
      `obs_man`, `pst_man` and `cross_placements`; the graph, the layout and `int_man` all
      derive. `int_coupling` and `int_blocks` are gone: `int_man.clique` and
      `int_man.blocks` say the same thing on the object itself.
- [ ] **This relies on singleton cliques.** `node_emb(node)` is `clique_emb((node,))`, so a
      cover without a singleton on a node it couples cannot derive that path. Every shipped
      layout has them; `Cliques` explicitly permits covers that do not. **Known and
      accepted for now.**
- [ ] **`TransposedInteraction` is the other direction.** The canonical fold makes axis 0
      the output, so transposing puts a *group* of axes there, which a forward reading
      cannot express. Its own transpose is the map it came from.
- [ ] **What `CliqueEmbedding` is for.** One thing only: reaching *several* nodes at once.
      A form over a group is a single slice of the layout's coordinates and does not
      factorize across the nodes, so a joint reach cannot be composed out of per-node ones.
      For a single node it is an offset in an embedding's clothing --- MFA's arity-3 clique
      is the only place it earns its keep, plus CCA reaching one side of a pair. **Worth
      asking whether it survives as a class or becomes a method on the layout.**
- [ ] **Vocabulary.** A clique has **axes** (positions, `0..arity-1`); a layout has **nodes**
      (labels it assigns through `placements`). "Selector", "factor", "span" and the
      `Placement` alias are all
      gone; an embedding is a `LinearEmbedding` and the field is `node_embs`, which says the
      ambient is a node.
- [ ] **`node_mans` is the payoff, and it is derived.** Every clique states what it expects
      at each node it touches; the invariant is that cliques meeting at a node agree.
      Before this round four cliques --- `hmog`, `dhmog` and both CCA branches --- had an
      axis whose ambient was a *partition* rather than a node, because the node embedding
      and the path were pre-composed. **Judge whether deriving beats declaring.**
- [ ] **Enforcement is a test sweep, not a runtime check.** `node_mans` raises when asked;
      `layout_problems` asks it for every shipped model. Same stance as storage-vs-canonical
      order.
- [ ] **`CliqueEmbedding` is pure addressing** --- `(nodes, layout)`, `project` a slice and
      `embed` a scatter, built by `LinearCliques.clique_emb(nodes)`. It restricts nothing,
      which is what freed the axis embeddings to land on nodes.
- [ ] **`clique_emb` is the geometric way in**, the same shape `Pair` has with
      `FirstEmbedding`. `clique_index` and `clique_offsets` are still public and still its
      mechanics; making `clique_emb` the *only* way in is the follow-up.
- [ ] **A bias needs no special case.** `out_axes = (0,)` at arity 1 leaves the input group
      empty, so the column product is 1 and `project_in` returns the constant $1$ --- the
      map out of `Null`.
- [ ] **The coordinate seam is four methods on `CliqueMap`**: `node_coords` / `amb_coords`
      are the path alone, `project_axis` / `embed_axis` add the clique's own embedding, and
      `project_domain` / `embed_domain` do the contracted side. **Too much surface?**
- [ ] **`LevelCliques.root_placements`** still decides whether a structured root partition is
      expanded (CCA's two roots) or held as one node (`Mixture`). Read §5 before judging.
- [ ] **Storage order versus canonical order** is still enforced only by the test sweep.
- [ ] **A tolerance the round nearly broke, worth deciding on.** `BoltzmannLGM` applies its
      likelihood to a bare 3-vector while `int_man.dom_man` is a `FullBoltzmann` of
      dimension 6 --- it works only because `GeneralizedGaussianLocationEmbedding.project`
      slices rather than checking. Routing arity 2 through `select_joint`, which reshapes,
      exposed it as 4 failures in `tests/lgm.py` (8 before the tests were reorganised). The
      single-node fast path now lives in `LinearClique._project_group`, which is where it
      belongs --- it is a fact about the clique's own embeddings, not about paths --- so the
      tolerance survives. **It is a real looseness in `BoltzmannLGM`, not in the clique ---
      decide whether to tighten it there.**
- [ ] **Still open: `Harmonium` treats the interaction as a `LinearMap`.** That is now the
      *only* reason `CliqueMap` exists --- it holds no declared data. Removing it means
      `lkl_fun_man` and `pst_fun_man` stop being `AffineMap`s and the likelihood is computed
      by contracting each clique directly. Blast radius measured: `lkl_fun_man` 90
      references across src/tests/examples, `int_man` 122, plus every conjugation path,
      which does linear algebra through `to_matrix` (10 sites) and `coord_blocks` (9).
      **Agreed destination; a separate pass.**

*Pinned by* `tests/graphical.py` for layouts, partitions, cut indices, the five layout
invariants and the bare clique algebra, and `tests/clique_map.py` for `CliqueMap` against
every live interaction shape.

---
- [ ] **A shortening pass took the module 844 -> 775 lines (8%).** It is 38% executable ---
      292 code lines against 328 of docstring --- so the requested 25--50% was not reachable
      without deleting documentation. Cut: `LinearClique.reorder` and `LinearCliques.node_emb`
      (no callers anywhere), `LinearClique.sub_mans` (inlined into `sub_dims`, its only
      reader), `validate_placement` and `shift_placements` made private and dropped from
      `clique.rst` (no external callers, never exported), and rationale prose that repeated
      the module or parent docstring. **Judge whether the remaining density is right.**
- [ ] **The view now lives on the clique.** `out_axes` replaced the canonical fold, so
      `dim` stopped being contingent on a convention (`Symmetric` had reported `dim == 3`
      for a 16-entry tensor at `sub_dims == (2,2,4)`), the `keep` arguments are gone, and
      forward-versus-backward is a property of the object rather than of a call sequence.
      API: `contract` + `partial_contract` -> one `contract(params, *in_node_coords)`;
      `tensor` -> `outer_product(out_joint, in_joint)`; `select_joint`/`embed_joint` ->
      `project_in`/`embed_in` and `project_out`/`embed_out`; new `transposed()` and
      `transpose(params)`. Nine members became twelve, but each has one meaning.
      `TransposedInteraction` is **deleted** (`interaction.py` 373 -> 299 lines):
      `Interaction.trn_man` is now an `Interaction` over transposed forms with the two paths
      swapped, which reproduces the old backward reading term for term. Its
      `outer_product` orientation quietly changed --- the old one returned *forward*-layout
      parameters --- and no caller in `src/` was using it. **Judge the API surface.**
- [ ] **The graph is the single authority on direction.** `out_axes` was briefly declared by
      hand at twelve sites, which made it a second authority for a fact `cross_paths`
      already derives: it computes `near = members` intersect `root_nodes` and requires
      exactly one, so the output axis is `members.index(near[0])` --- always `0`, since the
      canonical numbering puts root nodes lowest. `cross_paths` now takes the whole
      placement and **rejects** a form that says otherwise, next to the sibling check it
      already made. The twelve declarations became the `cross_clique(rep, node_embs)`
      `LevelCliques.cross_placement(rep, {node: emb})` method, which takes the node *set* a
      coupling touches and derives everything positional from it: storage order is those
      nodes ascending, arity is how many there are, and the output group is the one root
      node among them. **No model writes an axis number, an axis order, or a `members`
      tuple** --- the twelve declarations and their twelve node tuples both went. A form
      cannot be paired with a node list that disagrees with it, because the node set *is*
      the arity, so `_validate_placement`'s two rules are unstateable at every site that
      goes through the method. Two survive by hand and are why the `cross_paths` check
      stays: `graphical/mixture.py`'s borrowed $\theta_{XY}$, and the root placements.
      Pinned by `test_the_reading_is_derived_from_the_graph` and
      `test_a_coupling_needs_exactly_one_root_node`.
- [ ] **`LinearCliques` is now `@dataclass(frozen=True)`** like `LevelCliques` and
      `CliqueProduct` below it. It had been the only clique class without the decorator, and
      one of three outliers library-wide (with `Boltzmann` and `GeneralizedGaussian`) among
      abstract manifolds that are otherwise all decorated.

## 3. The maps --- `manifold/map.py`

- [ ] **`BlockMap` is deleted**, on your instruction. `Interaction` sums its own cliques, so
      it had no constructors left in `src/`, `tests/` or `examples/` --- and none in
      `goal-apps` either, checked before removing a public symbol. Its two users at `HEAD`
      were MFA's three blocks and CCA's two branches, both harmoniums, both now single
      interactions with several placements. Doc references in `algebra/matrix.py` and
      `manifold/embedding.py` were reworded rather than repointed.

`Map`, `LinearMap`, `MatrixMap`, `BlockMap`, `SquareMap`, `AffineMap`, `MultilayerPerceptron`.

- [ ] **Embeddings left the map layer entirely.** A `MatrixMap`'s matrix is
      $(\dim(codomain), \dim(domain))$ --- its dimensions now mean what its domain and
      codomain say they mean. A map that addresses only part of a side says so by being a
      `LinearClique`. **This was the point of the round: judge whether the seam is in the
      right place.**
- [ ] **`MatrixMap` is the old `AmbientMap`**, unchanged in signature and behaviour, promoted
      from convenience wrapper to primitive. `SquareMap` rebases onto it; `Covariance` and
      `CouplingMatrix` are unaffected.
- [ ] **`LinearMap`'s contract is now three members** --- `trn_man`, `transpose`,
      `outer_product` --- plus `transpose_apply`. Everything about embeddings is gone.
- [ ] **`BlockMap` lost its two embedding overrides**, which had no callers once the ABC
      dropped them. It still cannot say which nodes its blocks couple: the blocks share a
      domain and codomain, so a multi-block model declares `cross_placements` outright.
- [ ] **Cost of the move**: `to_matrix` / `from_matrix` / `matrix_shape` now exist on two
      classes with the same meaning-relative-to-their-own-dimensions. Only the five
      embedding-carrying interactions changed what those dimensions *are*, and each became a
      `LinearClique` that keeps the selected shape. **Duplication worth paying?**
- [ ] **`AffineMap` is itself a two-clique layout** --- a bias on the codomain plus an
      interaction --- and now that a bias *is* a `LinearClique`, both halves are the same
      kind of object. Folding it into a layout is the obvious next step. Recorded, out of
      scope.
- [ ] **`AffineMap` now declares `Map[Domain, Codomain]`.** It had `dom_man` and `__call__`
      and every caller used it as a map, but its MRO was `AffineMap -> Pair -> Tuple ->
      Manifold`, so it was the one class in the library playing a role it did not declare.
      It gained a `cod_man` property (`self.map_man.cod_man`, which `fst_man` now returns),
      and its `dom_man` field became `_dom_man` behind the inherited abstract property ---
      it had been the only public manifold-valued field on a map-like class. Both
      construction sites are positional, so neither moved.

## 4. The re-rooting view --- `manifold/cut.py` (bracketed)

- [ ] **`CliqueCut(cliques, clique_dims, far_node)`** is an isomorphism, not a computation: the
      same coordinates regrouped as `(near | crossing | far)`. Five index tuples derived from
      the three fields; `project` and `join` are methods.
- [ ] **The five derived tuples are properties, not cached fields.** They had been
      `field(init=False)` written through `object.__setattr__` at the end of `__post_init__`
      --- the only place in the library that cached derived values in fields instead of
      recomputing them on access. `__post_init__` is now validation only, and `project` and
      `join` bind `_group()` once each rather than reading five properties. **Check the
      recomputation is genuinely free**: it is pure Python over tuples at trace time, but
      `_group()` is now called twice per `project` where it was once per construction.
- [ ] **It reads the layout, not the graph.** It used to read `canonical_cliques` and index
      `clique_dims` with the result, so two layouts over one graph differing only in the
      storage position of two equal-sized blocks produced an identical cut with the near and
      far blocks *swapped*, silently.
- [ ] **`project` zero-fills.** The crossing matrix has a row band for every near clique, so a
      near clique with no crossing partner contributes zeros --- which is what makes `cross` a
      dense linear map from the far side into the whole near parameter vector rather than a
      ragged collection.
- [ ] **One user**: `models/graphical/mixture.py`, two sites. `CliqueEmbedding` and the
      arity-3 path are bracketed with it --- carried across and kept green, not redesigned.
      Deferred question: a cut is `level_split` at a single node plus dimensions, so it could
      be a refinement of the level split rather than an independent record.

---

## 5. `exponential_family/harmonium.py`

- [ ] **`Interaction` lives in `manifold/interaction.py`**, its own module --- not here and
      not in `manifold/clique.py`. The deciding reason is the import edge: `clique.py` does
      not import `map.py`, so the module that defines forms does not know what a map is, and
      putting the reading in either neighbour would break that. It is generic over bare
      `Manifold`s and uses nothing from the exponential-family layer. A harmonium supplies
      the placements and the derived paths; `int_man.blocks` hands back each term as an
      interaction in its own right, which is what retired `BlockMap`.
- [ ] **The base is `LevelCliques[Observable, Interaction[Posterior, Observable], Posterior]`**
      and `split_level` still returns exactly `(obs_params, int_params, lat_params)`. However
      deep the graph, those three partitions stay contiguous, so everything written against the
      level split is unchanged.
- [ ] **`cross_placements` replaced `int_members`.** A model declares `tuple[LinearClique, ...]`
      outright rather than member tuples something else turns into cliques. In the
      single-block case the base class no longer *builds* a clique: `int_man` already is one,
      declared on its own local labels $(0, 1)$, and the default just shifts it into the
      level's frame as `((observable, observable + 1), int_man)`. Where the model has
      several blocks or several root nodes it declares the pairs, which is a line each in
      CCA and MFA over the maps they already build.
- [ ] **`cross_man` *is* `int_man`** --- no wrapper. The one thing still declared is which
      nodes each crossing clique couples, which nothing but the model knows: the forms share
      a domain and a codomain.
- [ ] **The guard, and the defect it replaces --- read this carefully.** `cross_placements`
      hardcodes members `(0, 1)`, which names the latent only when the observable partition is a
      *single* node. `mix_man` --- the `CompleteMixture` whose observable is a whole harmonium,
      and the view MFA reads itself through --- expands into two root nodes, so `(0, 1)` named
      two roots and the category sat unreachable at node 2:

      ```
      cliques  : ((0,), (0, 1), (1,), (0, 1), (2,))    <- (0,1) twice
      validate : ValueError: duplicate cliques: [(0, 1)]
      ```

      The interaction genuinely cannot be a three-node clique --- it couples the base
      harmonium's whole 21-number parameter vector, which is not a tensor product of node
      statistics --- so two nodes is the correct reading. The fix is
      `LevelCliques.root_placements` (the hook), `Mixture.root_placements` (holds the
      observable as one node), and a guard here
      that **raises** when the interaction has several blocks, or the root partition occupies
      several nodes, and the subclass has not overridden. **Judge whether the guard belongs
      here or whether models should validate at construction (§9).**
- [ ] **The guard now also raises on a non-clique interaction.** `cross_placements` needs
      embeddings; an interaction that is a bare `MatrixMap` has none, and says so with a
      `TypeError` rather than guessing.
- [ ] **`pst_fun_man` now holds a `TransposedClique`.** `AffineMap(int_man.trn_man, …)` is
      unchanged in shape, but the transposed half is a view rather than a clique. Nothing in
      `Harmonium` names it; check that you are happy for it to be invisible there.
- [ ] **`HarmoniumEmbedding` addresses the level split**, not individual cliques, so it stays
      correct as a model's block count grows.

---

## 6. The models

**MFA --- `models/graphical/mixture.py` (610)**

- [ ] Declares three cliques by *building* them: `xy_man` on $(0,1)$, `xyk_man` on
      $(0,1,2)$, `xk_man` on $(0,2)$, and `cross_placements` pairs those same three objects
      with their nodes.
      Node count, root count, biases, the $(y,k)$ coupling from the mixture above, levels, and
      layout all follow, and `cross_placements` pairs each with its nodes. `xy_man` and
      `xyk_man` are the base interaction re-aimed through `reaimed`; `xk_man` is built
      outright.
- [ ] **The graph is a fork, not a chain.** Both $y$ and $k$ are adjacent to $x$ through the
      three-way clique, so levels are $(1,2)$ and depth is 2. Consequence: `_CATEGORY_NODE = 2`
      cannot become `levels[-1]`, because the deepest level holds $y$ too.
- [ ] `to_mixture_coords` / `from_mixture_coords` are three lines each via `CliqueCut`,
      replacing ~41 lines of hard-coded offsets; `whiten_prior` and `to_natural_likelihood`
      both flow through them.
- [ ] **The arity-2 bridge is now internal to one object.** The $(x,y,k)$ clique is *stored
      and executed* as a matrix over the joint $(y,k)$ statistic while *reporting* arity 3
      through its embeddings --- but it is one `LinearClique`, so there are no longer two
      descriptions that could drift. `tests/clique_map.py` checks the two readings agree,
      including in mean coordinates at the E-step, which is the case that decides it.

**CCA --- `models/harmonium/cca.py` (227)**

- [ ] The first model over a **multi-root** graph: the fork $x \leftarrow z \rightarrow y$, two
      branch maps from `_branch_map`, placed at $(0,2)$ and $(1,2)$, `n_roots = 2` derived
      from the observable being a `CliqueProduct`. Its axis 0 now lands on the *branch* and
      the slot embedding is the way in --- one of the four mismatches the round removed.
- [ ] **Conjugation is a sum**, $\rho = \rho_X + \rho_Y$, delegating to two standalone
      `NormalLGM`s --- exact because `DifferentiablePair.log_partition_function` is already a
      sum. Residual 1.4e-16 (LGM control 2.2e-16).
- [ ] It is `DifferentiableConjugated`, not analytic: inverting the conjugation sum
      branch-wise needs structure a fork does not supply. If you know the closed form, this
      gains EM.
- [ ] It exposes no canonical directions or correlations --- it is probabilistic CCA. Rename or
      document the distinction.

**HMoG --- `models/graphical/hmog.py` (369)**

- [ ] **Declares nothing.** Its three-node chain comes from the default `cross_placements` plus the
      deep partition's own forms spliced in recursively. This is the extension property working: a
      hierarchical model needs no graph declaration at all. Its one rewiring site is
      `extended`, which composes the way in to node $y$ inside the upper mixture without
      touching the axes --- the other mismatch the round removed.
- [ ] The asymmetry is readable off the layout: the $x$–$y$ coupling touches only *location*
      sub-statistics, the $y$–$k$ coupling touches $y$'s full statistic.

---

## 7. Tests

- [ ] **`tests/clique.py`** (57, sub-second, no JAX) --- pure combinatorics, via its own
      `Cover` and `ascend`.
- [ ] **`tests/graphical.py`** (81) --- layouts, partitions, `CliqueCut` indices with every error
      message pinned verbatim, the ordering regressions, and the bare form algebra pinned
      against `MatrixMap` at arity 2.
- [ ] **`tests/clique_map.py`** --- `LinearClique` against every live interaction shape.
      `NODE_NAMES` now splits on whether `dom_path` is a `CliqueEmbedding` rather than on the
      old entangled embedding, which is the same split stated better.
      Replaces `tests/ef_clique.py` and `tests/multilinear.py`, both of which tested one half
      of an object that is now whole.
- [ ] **A distinction the tests had to make explicit.** The clique reading and the map reading
      coincide only when the domain embedding addresses a *single node*. Where it is a
      `CliqueEmbedding`, the clique contracts to that node's own coordinates while the map
      lands in the joint manifold's, and they agree after the embedding's own
      `project`/`embed`. `NODE_NAMES` in `clique_map.py` is that split. **Judge whether the
      asymmetry is a design smell or the honest statement of what a joint domain means.**
- [ ] **The two that matter most** are in `TestDeclarationOrderRegressions`: reversed CCA
      branches and a mis-rooted deep partition. Both reproduce silent corruption in the
      pre-consolidation code and both pass. The reversed-CCA fixture now uses `relabel`.
- [ ] **`TestLayoutInvariants`** sweeps every shipped model through `layout_problems`, now
      **five** invariants in dependency order --- the fifth being that cliques meeting at a
      node agree about what occupies it. Storage order == canonical order is the first and
      returns alone --- **the single enforcement point for canonical ordering in the
      codebase.**
- [ ] **`tests/graphical_mixture.py`** (39) --- includes the two `mix_man` regressions: it is a
      two-node graph, and expanding the observable is refused rather than mislabelled.
- [ ] **The judgement to make**: do the layout pins test the *contract* or the
      *implementation*? Those in `graphical.py` are the ones I would trust least.
- [ ] **`tests/graphical.py` and `tests/clique_map.py` now cover one module between them** ---
      `manifold/clique.py` --- plus `manifold/cut.py`. Split by concept rather than by file,
      which the naming convention does not allow for. Decision in §9.

---

## 8. Docs and examples --- skim

- [ ] RST mirrors the module structure: `manifold/clique.rst` documents `LinearClique`,
      `CliqueEmbedding` and `RootEmbedding` as well as the layouts, and no longer documents
      any map; `exponential_family/harmonium.rst` gained an `Interaction` section;
      `exponential_family/clique.rst` is deleted and dropped from that index;
      `manifold/map.rst` loses `EmbeddedMap` and renames `AmbientMap`.
- [ ] **`examples/cca/run.py`** is the only example touching the clique API.
      `examples/pendulum/run.py` and `examples/variational_mnist/model.py` each build one
      interaction and were updated to `LinearClique`.
- [ ] **`CLAUDE.md`** --- module map, the `Maps` design-pattern entry, and the test table are
      updated for this round. The test-file naming question in §9 is still open.
- [ ] Sphinx does not fail on dangling roles (nitpick is off), so cross-references to moved
      names were repointed by hand. Re-grep for `exponential_family.clique` before merging ---
      that module no longer exists.

---

## 9. Decisions only you can make

- [ ] **The ABC-versus-concrete rule is now written down**, in `CLAUDE.md` under Typing
      Strategy: abstract-and-stateless for a role a model *becomes*, concrete-with-fields for
      a value a model *builds*; abstract classes may carry only the fields every subclass
      shares, hoisted to the lowest common parent; a field backing an inherited abstract
      property is private and named for it; derived quantities are properties, never
      `field(init=False)`. An audit of all ~120 classes found the rule already held
      everywhere except the two cases fixed in §3 and §4. **Confirm the rule is the one you
      intended, since it is now the thing new code will be measured against.**

- [ ] **`Interaction` keeps `placements` and `paths` as two parallel tuples** that must stay
      index-aligned, and nothing enforces it beyond a `strict=True` zip in `blocks`. It could
      be one tuple of triples, or the interaction could hold the layout and derive the paths
      on demand. Storing them keeps `Interaction` independent of the harmonium that built it,
      which is why it is written this way --- but the alignment is a real invariant with no
      guard.
- [ ] **The `Placement` alias is deleted** on your instruction. The cost is
      `tuple[tuple[tuple[int, ...], LinearClique], ...]` at 29 signature sites. If that reads
      badly in the models, the alternative is a frozen `Placement` dataclass carrying
      `validate_placement` in `__post_init__` --- not the alias again.

- [ ] **Should models validate their graph at construction?** `Cliques.validate()` is public
      and manual. The `mix_man` defect (§5) survived precisely because nothing called it. A
      `__post_init__` on concrete models would catch the next one, at the cost of a BFS per
      construction.
- [ ] **Test-file naming.** `graphical.py` and `clique_map.py` split one module by concept.
      Rename, split differently, or amend the convention.
- [ ] **Should a bias have a transpose?** (§2) `trn_man` raises for one, because the
      canonical fold always has a row axis and never an empty head. Nothing calls it.
- [ ] **Is `TransposedCliqueMap` the right shape?** (§2) It exists because a transpose puts
      a *group* on the output side. The alternative is a general `view(keep)` on `CliqueMap`
      of which forward and transposed are two instances.
- [ ] **Derive `node_mans` or declare it?** (§2) Derived means no drift but no
      construction-time check; declared means the reverse. Currently derived, checked by the
      test sweep.
- [ ] **Is the public coordinate interface too wide?** (§2) Six methods on `CliqueMap`
      separate the path from the clique's own embedding. They are also the clearest
      statement of what applying a clique does, and the tests read better for it.
- [ ] **Is `cliques` the right place to validate placements?** (§2) It runs on every access
      rather than at construction, which is cheap but is not the moment the mistake is made.
      The alternative is a `validate()` a model calls, which is the shape that let the
      `mix_man` defect through once already.
- [ ] **Do you want the general `view`?** (§2) `reorder` gives the coordinate permutation;
      naming the manifold of a multi-node group needs a dimension-parameterized manifold that
      `geometry/` does not have (`Euclidean` lives in `models/`). **Adding one is the
      prerequisite --- say if you want it.**
- [ ] **Does `manifold/clique.py` earn its place** as a module with no client inside
      `manifold/`? It holds four concerns: the clique, its two map readings, the two
      embeddings, and the layouts. **Split candidate** --- cliques and maps in one file,
      layouts in another. (§2)
- [ ] **When to clean up `Harmonium`?** (§2, §5) `int_man` is typed as a map over
      partitions, which is the only reason `CliqueMap` needs paths at all. Inverting it ---
      declare `obs_man` / `pst_man`, derive `int_man` --- is the next step.
- [ ] **`same_graph` or `same_cover`?** (§1)
- [ ] **Prune `LinearCliques`' unused surface?** `split_cliques` / `join_cliques` have no
      production callers. (§2)
- [ ] **`LinearClique`'s form algebra has no production callers either.** `tensor`,
      `contract`, `partial_contract`, `to_tensor`, `from_tensor`, `reorder`, `select_joint`
      are exercised only by tests --- production runs `__call__`, `transpose_apply` and
      `outer_product`. Keep as the arity-$n$ statement of what a clique *is*, or prune?
- [ ] **`InteractionEmbedding` and `PosteriorEmbedding` are unused in `models/`**, subsumed by
      `CliqueEmbedding`. Still coherent partition-level API, still tested. Delete or keep?
- [ ] **How far should "clique-based" go?** The graph and layout are clique-based everywhere;
      the *algorithms* are not --- ~40 sites use `split_level`.
- [ ] **Whether a vector-valued product is one node or many** is a modelling choice, not a
      property of the manifold. `placements_of` hard-codes "one".
- [ ] **Should `rep` be on a clique at all?** Measured: 28 cliques across every shipped
      model, all `Rectangular`. A structured representation is a statement about a *pair* of
      equal-dimension axes and has no meaning under an n-way fold, which is why `to_tensor`
      needs a `_require_dense` guard. Dropping it gives `dim = prod(sub_dims)`
      unconditionally, removes `matrix_shape` and the guard, and leaves `to_matrix` as what
      it is --- the arity-2 view. Structured reps stay in `SquareMap`, inside a node.

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
- **Why is a bias a `LinearClique` and not its own class?** Because the canonical fold
  already handles it: the empty column product is 1, so an arity-1 form is a column vector,
  and applying it means feeding the constant $1$ --- the map out of `Null`. `NodeClique` was
  the class that existed only because that was not stated.
- **Why did `exponential_family/clique.py` disappear?** Nothing in it referenced an exponential
  family except a type bound on the embeddings. Once `EFClique` merged into the clique, the
  file held `CliqueEmbedding` and `RootEmbedding`, both bounded by `LinearCliques` and
  `LevelCliques` --- manifold-layer notions. Its `sufficient_statistic` and `node_mans` were
  the only EF-specific members and had no callers in `src/`.
- **Why is `CliqueEmbedding.sub_man` itself?** It has to name a manifold whose dimension is the
  product of its factors, and `geometry/` has no dimension-parameterized manifold to build one
  from. Being a `LinearClique` supplies `dim = prod(sub_dims)` directly.
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
  started. Shape: a fourth partition slot, `[root | cross | deep | rho]`, since $\rho$ is a
  per-level correction on the deep partition's root. **Trap**: ~10 sites destructure `split_coords`
  positionally into three `Array`s, so reordering slots breaks them *silently* --- rename the
  method in the same change to force every site to be touched.
- **The clique as execution primitive** --- done: `cross_placements` pairs the very objects
  `int_man` is built from with their nodes, so there is nothing left to derive in either
  direction. What
  remains is that MFA's arity-3 clique still *executes* as a matrix over a joint statistic; it
  reports arity 3 and the two readings are pinned equal, but the tensor path is not the one
  production runs.
- **Redesigning the bracketed MFA machinery** --- `CliqueCut`, `CliqueEmbedding`, arity 3.
- **A general `view`** --- blocked on a dimension-parameterized manifold in `geometry/` (§9).
- **Splitting `tests/graphical.py` further** --- blocked on §9.
- **`sufficient_statistic` as a flat per-node clique loop** --- needs a per-node data split.
  Real risk, unclear gain. The per-clique `sufficient_statistic` helper was deleted with
  `EFClique`; `LinearClique.tensor` is what it was built on.

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
| focused | `uv run python -m pytest tests/clique.py tests/graphical.py tests/clique_map.py tests/map.py tests/cca.py tests/graphical_mixture.py -q` | 242 passed, ~3 m |
| suite | `uv run python -m pytest tests/ -q` | **550 passed**, ~17 m |
| examples | `for e in hmog cca mfa mixture_of_gaussians; do uv run python -m examples.$e.run; echo "$e=$?"; done` | exit 0 each |

**The numeric check that matters most**: `examples/cca` reproduces latent alignment RMSE
`0.24031811353161686` and cross-covariance relative error `0.007013174699236388` over 8000
optimizer steps, unchanged across every restructuring. CCA is the multi-root fork and its root
partition is a `CliqueProduct`, so it exercises the most of this branch at once.

Equivalences established while building, worth re-establishing if you change the corresponding
code:

| check | result |
|---|---|
| block-derived `split_level` vs partition-derived | byte-identical, 10 models, splits and round-trips |
| partition dims vs summed block dims, groups contiguous | exact on 18 layouts, incl. deep partitions and the harmoniums nested in `hmm` / `kalman_filter` / `vm_population_code` |
| storage order vs canonical order | equal on every shipped model |
| `canonical_cliques` as a sort vs as a recursion | byte-identical on 15 layouts incl. deep partitions |
| clique form algebra vs `MatrixMap`, arity 2 | exact, both contraction directions |
| `LinearClique` vs live interactions | exact, 5 single-node-domain shapes × 3 operations |
| clique reading vs map reading, joint domain | exact after the `CliqueEmbedding`, MFA's $xy$ and $xk$ |
| eight shipped layouts: cliques, axes, dims | byte-identical before and after this round |
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
