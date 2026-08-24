# Review TODO — `clique-container` branch

Working-tree changes on top of `e5e28d1` ("broad coverage of initial clique design").
23 files, +568 / −154. Nothing committed.

Work through the sections in order — later ones depend on earlier ones. Each item names the
file and what to look at. Checkboxes are for your pass, not mine.

---

## The idea, in one paragraph

A **clique** is the parameters of one clique of the model graph, pairing against the tensor
product of its members' sufficient statistics — one inner product in the log-density.
Arity is the only thing that varies: arity 1 is a bias (a leaf family), arity 2+ is a
coupling (a linear map). A **composite** is a graph laid out as the three spans of one level
ascent. Every exponential family in `models/base/` is now a one-node clique, every
`LinearMap` is an arity-2 clique, and `Harmonium` is a composite whose graph is *derived*
from its spans rather than declared.

---

## 1. Algebra layer — `geometry/algebra/clique.py` (+12/−12)

- [ ] **`_validate_nodes` no longer requires a singleton clique per node.** Only reachability
      from the root set is enforced. Rationale: "every node carries a bias" is a fact about
      the *family*, not the graph, and it forbade an interaction's own cover `((0,1),)`.
- [ ] Class docstring and the `cliques` field docstring updated to say singletons are optional.
- [ ] **Check:** is dropping this guard acceptable to you? It was the only thing preventing a
      cover that addresses no node individually.

## 2. New abstractions — `geometry/manifold/combinators.py` (+351/−8)

The largest single change. Read this file first.

- [ ] **`CliqueManifold`** (was the three-span class, now the bare ABC). Declares `clq_set`
      and `clique_dims`; provides `split_cliques`, `join_cliques`, `cut`.
- [ ] **`Clique`** — one clique, one indivisible block. Docstring carries the dot-product
      definition. Deliberately **not** a `Tuple` (an atom has no component split).
      `single_cover(n_members)` builds its cover.
- [ ] **`Node`** — arity 1, concrete `clq_set`. Docstring makes the claim that replication and
      compound statistics do **not** multiply nodes (`Poissons(100)` is one node; a
      `ChordalBoltzmann` over a 10-variable junction tree is one node).
      **Check:** do you accept that convention? It is a convention, not a derivation.
- [ ] **`CompositeClique`** — renamed from the old `CliqueManifold`. Same `[root | cross | deep]`
      layout, `split_level` / `join_level` unchanged.
- [ ] **`CompositeClique.clq_set` is now concrete** — the glue. Root supplies root nodes, deep
      supplies the rest shifted past them, cross supplies the joining clique; levels are
      recomputed by BFS so the deep span's own `n_roots` is discarded. **Three guards**, all
      with the remedy "declare clq_set on the model instead": root span is one node, cross
      span is one arity-2 clique, and glued clique count matches `clique_dims` length.
      **Check the third guard especially** — without it MFA silently derived a wrong-but-valid
      graph.
- [ ] **`CliqueCut`** and `cut(far_nodes)` — re-view the layout across an arbitrary node cut.
      Holds only integers, so it is a block permutation and applies identically in natural and
      mean coordinates.
      **Known limitation, documented and tested:** the view exists only when every crossing
      clique couples the far side *in full*. True for a categorical node in a mixture, false
      for sub-statistic couplings (the usual Gaussian case). **`split_level` is NOT a special
      case of `cut`** — I claimed twice that it was, and it is not.
- [ ] **`span_cover` / `span_blocks`** — module-level helpers with one `isinstance` each. A span
      that declares no clique structure is one opaque block / one node.
      **Check:** this is the one isinstance in the design. It exists because
      `VariationalConjugated` is an exponential family that is *not* a clique manifold (its
      `rho` block is a learned correction, not a graph factor), which rules out putting the
      bound anywhere on the EF hierarchy.

## 3. Maps as cliques — `geometry/manifold/map.py` (+53/−4)

- [ ] **`LinearMap` is now a `Clique`** with a concrete arity-2 cover. Node 0 is the codomain,
      node 1 the domain — matching the root/deep orientation a harmonium gives its interaction.
      Attached at `LinearMap`, not at `EmbeddedMap`, so `Harmonium.int_man` has a cover whatever
      concrete map fills it.
- [ ] **`SquareMap` overrides back to arity 1** — domain and codomain are the same manifold, so
      it is a self-interaction inside one node. This is what `Covariance` and `CouplingMatrix`
      are. Done as an explicit override, *not* by putting `Node` first in the bases — that wins
      the MRO but breaks `super().__init__` (basedpyright: `reportUnsafeMultipleInheritance`).
- [ ] **`BlockMap.clique_dims`** is per-block. Its own `clq_set` stays the coarse two-node cut,
      because its blocks share a domain and codomain and cannot say which nodes each couples.
      The composite that owns it supplies the identities; the block map supplies the sizes.

## 4. Leaves as nodes — `models/base/*` and `exponential_family/combinators.py` (+34/−9)

- [ ] `Node` attached at **five points**, not per class: `GeneralizedGaussian` (covers `Normal`
      and all four Boltzmanns), `ExponentialFamilyProduct`, `LocationShape`, plus the seven
      standalone families (`Poisson`, `CoMShape`, `Binomial`, `Dirichlet`, `VonMises`,
      `Euclidean`, `Bernoulli`, `Categorical`).
- [ ] **`ExponentialFamilyPair` was deliberately left out.** It is genuinely two variables over
      disjoint data, so it is two nodes, not one. See section 8.

## 5. Harmonium — `geometry/exponential_family/harmonium.py` (+3/−12)

- [ ] Base changed to `CompositeClique`; the abstract `clq_set` redeclaration removed so the
      derived one is inherited. Nothing else in the harmonium core changed.

## 6. Derived graphs — 8 declarations deleted

- [ ] `lgm.py`, `mixture.py`, `population_codes.py` (×2), `hmog.py`, `examples/pendulum/run.py`,
      `examples/variational_mnist/model.py`, `tests/variational.py`.
- [ ] **Verified derived == declared before deleting**, for `NormalLGM`, `AnalyticMixture`,
      `PoissonVonMisesHarmonium`, and `AnalyticHMoG` (the 5-clique chain).
- [ ] **`CompleteMixtureOfHarmoniums` still declares** (`models/graphical/mixture.py:283`), and
      its graph **changed**: from a two-node coarsening to the true three-node graph, seven
      cliques, `((0,), (0,1), (0,1,2), (0,2), (1,), (1,2), (2,))`, levels `(1, 2)`.
      **Check this one carefully** — it is the only semantic change to a declared graph.
      Safe today only because nothing computes with `clq_set`.

## 7. MFA layout rewrite — `models/graphical/mixture.py` (+22/−49)

- [ ] `to_mixture_coords` / `from_mixture_coords` went from 41 lines with hard-coded offsets
      (`hrm_int_matrix[:obs_dim, :]` etc.) to three lines each, via a new `mix_cut` property.
- [ ] **Verified bit-for-bit** against the old implementation on random parameters, both
      directions, max|Δ| = 0.0.
- [ ] The docstring claim that both work identically in natural and mean coordinates is now
      evident rather than asserted.

## 8. Tests — `tests/clique.py` (+45/−12), `tests/combinators.py` (+70/−10)

- [ ] `tests/clique.py`: `test_missing_singleton` **removed** (that validation is gone); new
      `TestAtomicShapes` covering the node cover, the edge cover, a multi-clique edge cover, and
      that reachability still fires.
- [ ] `tests/combinators.py`: `CliqueManifold` → `CompositeClique` throughout; new
      `TestCliqueAddressing` (7 tests) covering `clique_dims`, `split_cliques`/`join_cliques`,
      the cut isolating the deepest node, round-tripping, the layout-disagreement guard, and
      the sub-statistic rejection.
- [ ] **`test_clique_dims_sum_to_dim` asserts equality.** That holds only while every model is
      exact. If `VariationalConjugated` ever gains a correction block the invariant becomes
      `dim == sum(clique_dims) + cnj.dim`.

## 9. Multi-root support — new (+~110 lines)

The gap that blocked every multi-root graph. Read this before section 10.

- [ ] **`CliqueProduct`** (`geometry/manifold/combinators.py`) — a `Pair` that is also a
      `CliqueManifold`: graph is the disjoint union of the components', layout is theirs
      concatenated, and every node is a root (nothing links the two sides, so a non-root node
      would be unreachable). Guards that both components are all-roots.
- [ ] **`ExponentialFamilyPair` now extends `CliqueProduct`** instead of `Pair`, so it is two
      nodes rather than one. This is the combinator I deliberately left without `Node` in
      section 4 — for the right reason, as it turns out.
- [ ] **`FirstEmbedding` / `SecondEmbedding`** (`geometry/manifold/embedding.py`) — concrete
      `TupleEmbedding`s for a `Pair`. Needed because a `BlockMap` requires its blocks to share
      a codomain, so both branch interactions must target the *pair*, each selecting its side.
      `TupleEmbedding` previously had exactly one concrete subclass.

## 10. Canonical correlation analysis — new (+~200 src, +~200 test, +~250 example)

The first model in the library over a graph with more than one root, written to test the
design rather than because it was asked for.

- [ ] **`models/harmonium/cca.py`** — `NormalPair` (two normals, disjoint data, two nodes) and
      `CanonicalCorrelationAnalysis`. The whole model is: declare `clq_set` (two roots), declare
      `int_man` as a `BlockMap` of one clique per branch, and declare `conjugation_parameters`
      as the **sum** of the two branches', delegating to two standalone `NormalLGM`s.
- [ ] **`DifferentiableConjugated`, not `AnalyticConjugated`.** Inverting the conjugation sum
      branch-wise needs structure a fork does not supply, so there is no `to_natural_likelihood`
      and no EM. Fitting is gradient-based. See the open question below.
- [ ] `clq_set` is **declared**, because the glue assumes a single root node. The remedy the
      guard names, used in earnest for the first time.
- [ ] **`tests/cca.py`** — 23 tests. `TestGraph` pins the two-root graph, the depth-2 levels, and
      that the layout matches the graph block for block; `TestConjugation` is the substance.
- [ ] **`examples/cca/`** — `run.py` / `plot.py` / `types.py`. Data comes from an explicit
      two-view process (draw a shared latent, project into each view, add noise) rather than
      from the model's own parameters, so the shared structure is known and strong.

### What the CCA work established

- [ ] **The fork conjugation is exact.** Residual of the conjugation equation over 200 random
      latents: **1.4e-16** with a full-covariance posterior. `rho = rho_X + rho_Y` is right, and
      right for the stated reason: `DifferentiablePair.log_partition_function`
      (`exponential_family/combinators.py:126-131`) is already a sum, so each branch conjugates
      on its own. No new machinery was needed to make this work.
- [ ] **The layout lines up with no intervention.** `clique_dims == (14, 6, 8, 6, 5)` against
      canonical cliques `((0,), (1,), (0,2), (1,2), (2,))` — five blocks, five cliques, summing
      to `dim`. The observable pair supplies the two root blocks in node order, the `BlockMap`
      the two cross blocks, the latent the deep one.
- [ ] **Example result:** log-likelihood -9.97 -> -5.01, latent RMSE 0.240 after linear
      alignment, cross-view covariance reproduced to **0.7%**.

### Two pre-existing sharp edges found along the way (neither introduced here)

- [ ] **Diagonal posteriors do not conjugate exactly.** With `pst_rep=Diagonal()` the
      conjugation residual is ~4.5e-02 for a **plain `NormalLGM`** and ~1.7e-02 for CCA. With
      `PositiveDefinite()` both are ~1e-16. So this is LGM behaviour, not the fork's — either
      an intended approximation for restricted posteriors or the known
      `NormalCovarianceEmbedding` statistic-vs-parameter mismatch already in the Deferred list.
      Worth deciding which.
- [ ] **`initialize_from_sample` accepts wrongly-shaped data silently.** A harmonium's
      `data_dim` counts observable *and* latent columns, but passing observable-only data raises
      nothing --- it returned non-finite parameters for CCA and finite-but-meaningless ones for
      `NormalLGM`. A shape check would have saved an hour.

## 11. Documentation

- [ ] `docs/source/geometry/manifold/combinators.rst` — new "Clique Manifolds" and "Cut Views"
      sections; `CliqueManifold` moved out of "Product Manifolds", where it no longer belongs.
- [ ] `docs/source/models/harmonium/cca.rst` — new, and added to the subpackage toctree with a
      paragraph on why the *graph* matters as much as the latent.
- [ ] `CLAUDE.md` — `cca` added to the examples list and `cca.py` to the test-file table.
- [ ] `combinators.py` reorganised: `### Clique Manifolds ###` (CliqueManifold, Clique, Node,
      CompositeClique, CliqueProduct) then `### Cut Views ###` (span_cover, span_blocks,
      CliqueCut). The cut machinery previously sat *between* `CliqueManifold` and `Clique`,
      splitting the hierarchy.
- [ ] `CompleteMixtureOfHarmoniums.mix_cut` now derives the cut from
      `clq_set.levels[-1]` rather than hard-coding node 2.

---

## Decisions I made that you may want to reverse

- [ ] **`clq_set` derived rather than declared** — you chose this, but the MFA exception means
      the codebase now has both mechanisms.
- [ ] **The one `isinstance`** in `span_cover` / `span_blocks` (section 2). You said you feel
      mixed about isinstance checks; this is the one, and the alternative was a nominal
      `GibbsClique` base that every leaf would have to inherit.
- [ ] **`Node` naming.** Was `NodeAtom`/`AtomicClique` until you pointed out an atom is just a
      clique. Renamed to `Node`/`Clique`. Note `Node` now collides conceptually with
      junction-tree nodes in `boltzmann.py` — two senses of "node" one file apart. Docstring
      addresses it; a different name would too.
- [ ] **MFA's graph refinement** (section 6). I told you earlier this would fall out of
      derivation "for free". **That was wrong** — the blocks' node identities live in their
      domain embeddings and a `BlockMap` cannot read them. MFA declares instead.

## Corrections to things I told you during the session

Recorded so you are not reviewing against claims I have since retracted.

- [ ] "Derivation refines MFA for free" — **false**. See above.
- [ ] "`split_level` is a special case of `cut`" — **false**, said twice. `cut` needs full-statistic
      coupling; a level split's cross span may couple a sub-statistic.
- [ ] "Every exponential family is a clique manifold" — **false**. `VariationalConjugated` is the
      counterexample.
- [ ] "`VariationalConjugated` isn't a clique manifold, so stage 4 is dead" — the observation was
      right, the conclusion backwards. ρ is a per-level correction living on the deep span's root.

## Not done, by decision

- [ ] **Stage 5** (`sufficient_statistic` as clique tensoring) — **dropped**. Its own stopping rule
      triggered: the method is 8 lines, one implementation, zero overrides, already correct at
      any depth. Rewriting it would not shorten it.
- [ ] **Stage 6** (transposition as graph reversal) — **scoped down**. `EmbeddedMap.trn_man` already
      *is* the factor swap. Worth doing: make the unchecked layout invariant in
      `transpose_harmonium` structural. Not worth claiming: `observable_distribution` will still
      build a fresh model, because reversing an asymmetric harmonium genuinely changes types.
- [ ] **Interleaving conjugated / variational-conjugated levels** — the stated primary objective,
      designed but not implemented. Shape: `VariationalConjugated` becomes
      `Pair[Harmonium[Obs, Pst], Conjugation]`; both slots already exist as abstract properties
      (`gen_hrm`, `cnj_man`). Also *simplifies* `approximate_posterior_at`, which currently
      reconstructs a harmonium from `prior − ρ`. **Cost:** changes the stored parameterization
      from θ_Z to `lat`, so initialization semantics and any checkpoint shift meaning.
- [ ] ~~**Multi-root graphs / CCA.**~~ **Done** — see sections 9 and 10.
- [ ] **Solver-per-cut design** — drafted at your prompting, then dropped at your call. Draft in
      the session scratchpad. It made analytic-ness a property of a field, which forces the
      whole `Conjugated` hierarchy generic — too large for what it buys.

## Open questions for you

- [ ] **Deep-variational cascade.** If a *deep* level is variational, the level below conjugates
      against an approximate log-partition. Does the Theorem 18 cascade survive, or degrade into
      something the ELBO absorbs? Decides whether "reasonably conjugate" is a guarantee or a
      training outcome. Mathematics, not typing — I cannot settle it from the code.
- [ ] **Fork EM.** The forward direction is now **verified exact** (section 10). The inverse is
      still open: `to_natural_likelihood` needs to invert the conjugation sum, and a sum is not
      invertible branch-wise without more structure. `CanonicalCorrelationAnalysis` is therefore
      `DifferentiableConjugated` only. If you know the closed form, it becomes analytic and gains
      EM; if not, gradient fitting is the honest ceiling.
- [ ] **Whether a product is one node or many** is a modeling choice, not a property of the
      manifold. `Node` hard-codes "one" at the leaf. Binds if you ever want per-neuron couplings.

## Verification status

**Gates clean:** `ruff check`, `ruff format --check` (130 files), `basedpyright`
(0 errors, 0 warnings), `sphinx-build -W`.

**Full suite: 430 passed** (`uv run python -m pytest tests/`, 16m42s), including the new
`tests/cca.py`. Two later cleanup edits were re-verified separately against
`clique`, `combinators`, `cca`, `graphical_mixture`, `whitening`: **121 passed**.

**Example runs:** `uv run python -m examples.cca.run` then `.plot` --- log-likelihood
-9.97 -> -5.01, latent RMSE 0.240, cross-view covariance error 0.7%.

### One bug I introduced and caught during cleanup

Worth knowing because the same trap is easy to hit again. I replaced MFA's
`self.cut(frozenset({2}))` with `self.cut(frozenset(self.clq_set.levels[-1]))`, thinking it
was the same thing more robustly expressed. It is not: **MFA's graph has depth two**, so
`levels[-1]` is `(1, 2)` --- both $y$ and $k$ --- and cutting there takes $y$ along with the
category node. Seven tests failed. Reverted to a named module constant `_CATEGORY_NODE = 2`
with a docstring note saying explicitly why `levels[-1]` is wrong here.

The general point: "the deepest level" and "the last node" coincide only in a chain, and
the whole reason MFA is interesting is that it is not a chain.
