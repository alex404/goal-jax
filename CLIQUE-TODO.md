# Clique container: plan

Branch `clique-container`. Current state: the **composed-graph** design (2026-10-03, uncommitted
on top of `5dd6647`), described in `CLAUDE.md` (Architecture). Earlier designs (`LinearClique`,
`BlockMap`, `EFClique`, `CliqueSet`, `Interaction`, global labels with groups) are in git
history.

## Current state

- Every `ExponentialFamily` is a `LinearCliques`. By default it is one node, `cliques = ((0,),)`,
  whose map is `bias_map(self)`. `ExponentialFamilyPair` is two nodes. A harmonium composes
  its graph.
- Node numbers are local: each `LinearCliques` numbers its own nodes $0, \ldots, n-1$.
- `CrossMap(cod_man, dom_man, terms)`: each term is `(cod_clique, dom_clique, CliqueMap)`, with
  each clique in its own side's numbering, read and written through `clq_emb`.
- `RecursiveLinearCliques[Root: LinearCliques, Deep: LinearCliques]` is no longer a
  `RecursiveCliques`.
  - A model declares `rot_man`, `dep_man`, `crs_cliques` (`Crossing` pairs: a root clique and a
    deep clique) and, per crossing, `crs_rep` and `crs_emb_constructors` (codomain constructors,
    domain constructors).
  - Derived: the numbering (root first, deep offset by the root's node count), `cliques` (root,
    crossings, deep: the triple's storage order), `clq_map`, `crs_map`, `crs_man`.
- Deleted: `part_emb`, `cod_group`/`dom_group`, the bias branch, `split_clique`, `Mixture.impose`,
  `upr_graph`, `mix_nodes`/`mix_graph`, and every `_raw_cliques`/`_root_nodes` field.
- `Harmonium` lists `RecursiveLinearCliques` before `Gibbs` among its bases, so the composed
  graph comes before the one-node default in the method order.
- `CompleteMixture` declares one crossing per clique of its observable. For a one-node
  observable this is the old single crossing. For a harmonium observable (MFA's `mix_man`) it is
  the harmonium's graph with $k$ joined to every clique, and `Mixture.cmp_int_map` reads the
  interaction as one matrix. The per-clique blocks are consecutive row bands of that matrix, as
  long as every multi-node clique of the observable is `Rectangular`.
- `tests/graphical_mixture.py` checks that `to_mixture_coords` carries every MFA clique's block to
  the same clique's block in `mix_man`, and that `cmp_int_map` and `int_man` act alike.

## Why global labels failed (2026-10-03)

The earlier plan gave each single-node family a global node label. Families are built before
the enclosing graph exists: `Mixture(n, obs_emb)` already holds its observable inside an
embedding, and HMoG takes prebuilt submodels. Relabelling would therefore have to reach into
arbitrary embeddings. There are only two ways out. One is to construct every model top-down from
a graph. The other is local numbers. We chose local numbers, with the graph composed from the
parts and the crossings declared in the parts' own numbering.

## Conjugation calculus (design note, not implemented)

This is the next problem for the move from alpha to beta: composing conjugation the way the
layout now composes. Setting: a harmonium with root partition $X$, deep partition $D$, and
crossing terms $t$, each coupling a root clique $a_t$ with a deep clique $b_t$ through
$\Theta_t$.

### 1. The crossing constraint is closure under conditioning

The posterior has natural parameters $\theta_D + \sum_t \Theta_t^\top \mathbf s_{a_t}(x)$, and
term $t$ adds into the block of $b_t$. So $b_t$ must be a clique of $D$, or the posterior leaves
its family. Likewise, the likelihood $\theta_X + \sum_t \Theta_t \mathbf s_{b_t}(d)$ needs $a_t$
to be a clique of $X$. The layout enforces both (`clq_emb` raises otherwise). This is not an
artifact of the layout.

### 2. The prior needs the fill-in

Conjugation asks that, for all $d$,
$$\psi_X\Big(\theta_X + \sum_t \Theta_t \mathbf s_{b_t}(d)\Big) = \rho \cdot \mathbf s_D(d) + \chi.$$
The left side depends on $d$ only through the deep neighbourhood $N = \bigcup_t b_t$, and in
general jointly. This is the fill-in of variable elimination: a Gaussian $\psi_X$ coupled to $y$
and to $w$ produces a $y \otimes w$ term even if no crossing contains both. So the *prior*
family must hold blocks for the fill-in of $N$, while the *posterior* needs only the $b_t$. This
is the existing posterior/prior split stated in terms of graphs. It is why an LGM with a
diagonal posterior has a full-covariance prior (the fill-in inside one node). The condition is
decomposability. `ChordalBoltzmann`'s junction-tree machinery is the relevant prior art in the
repo.

### 3. Composition rules

Split the root into components $X_1, \ldots, X_m$, with no root clique joining two of them.
Let $N_i$ be the deep neighbourhood of $X_i$.

- **Rule 1, sum over independent root components.** If $\psi_X = \sum_i \psi_{X_i}$ (an
  `ExponentialFamilyPair` root), then $\rho = \sum_i \rho_i$ and $\chi = \sum_i \chi_i$. Here
  $\rho_i$ is the conjugation of the sub-harmonium $(X_i, D)$ with the terms touching $X_i$.
  CCA: $\rho = \rho_X + \rho_Y$, as implemented in `CanonicalCorrelationAnalysis`. Root nodes
  joined by a root clique form one component and are conjugated jointly.
- **Rule 2, locality.** If $N_i$ lies inside a sub-structure $S$ of $D$ whose statistics are a
  block of $D$'s, compute $\rho_i$ on the harmonium $(X_i, S)$ and embed it into $D$ at $S$'s
  blocks. HMoG: $S$ is the mixture's observable $y$, embedded by `ObservableEmbedding`. This is
  the template on `main` (`DifferentiableHierarchical.conjugation_parameters`).
- **Rule 3, primitives.** When $N_i$ spans several deep nodes jointly, $\rho_i$ comes from a
  primitive for the joint family. MFA: $N = \{y, k\}$. For each category $k$, the LGM
  conjugation of $\theta_X + \Theta_{XK}[k] + (\Theta_{XY} + \Theta_{XYK}[k])\mathbf s_y$ gives
  $\rho_y^{(k)}, \chi^{(k)}$. Then the $y$ block is $\rho_y^{(0)}$, the $(y, k)$ block is
  $\rho_y^{(k)} - \rho_y^{(0)}$, and the $k$ block is $\chi^{(k)} - \chi^{(0)}$. This matches
  `CompleteMixtureOfConjugated.conjugation_parameters` (read 2026-10-03). Primitives are binary
  harmoniums (Normal–Normal, family–Categorical, Normal–Boltzmann) and are written by hand.

Recursion handles the rest: the deep partition is itself conjugated, so the prior's marginal
cascades upward.

### 4. Proposed implementation strategy

1. Derive the root components and each component's terms from the layout. Crossings are pairs,
   so grouping by root part is direct. $N_i$ is the union of the deep parts.
2. Have each model declare one binary primitive per component, together with the embedding of
   $N_i$'s block structure into $D$.
3. Write one generic `conjugation_parameters`: $\sum_i \iota_i(\mathrm{conj}_i(\mathrm{lkl}_i))$,
   with $\mathrm{lkl}_i$ the slice of likelihood parameters on component $i$ and its terms.
   CCA, HMoG and MFA then become three declarations of it.
4. Later: the analytic inverse (`to_natural_likelihood`) does not compose by summation. This is
   why CCA is only `DifferentiableConjugated`.
5. Open question: the same calculus says where a variational $\rho$ lives (the fill-in cliques),
   which bears on the variational $\rho$ slot below.

## Other open items

### Raised by the composed graph

- `RecursiveCliques` (`algebra/clique.py`, with `tests/clique.py`) is no longer used in `src/`:
  levels, `level_split` and canonical order are replaced by composition. Delete, or keep as a
  graph tool?
- A crossing part that is a multi-node clique with a structured (non-`Rectangular`)
  representation cannot be coupled per node: the axes are the clique's nodes, but its block
  holds fewer parameters than their product. `CompleteMixture` over such an observable fails
  loudly (shape mismatch). Coupling to a clique's block *as a flat vector* would need a second
  kind of crossing axis.
- `Mixture` with a general `obs_emb` assumes a one-node observable (one crossing). Only
  `CompleteMixture` generalizes to several cliques.
- The pendulum example's latent is a `DifferentiablePair`, so it is now two nodes with two
  crossings. Its interaction parameters are ordered differently from before (column bands rather
  than one row-major matrix), so its numbers are not comparable with earlier runs.
- Nothing checks that a `LinearCliques`' nodes are exactly $0, \ldots, n-1$, which the offset
  assumes. All current classes satisfy it by construction.

### Before merging

- Some `variational_mnist` scripts still import `EmbeddedMap` / `BlockMap` (experimental, left
  broken; 124 basedpyright errors predate this work). `variational_mnist/model.py` itself was
  updated and imports.

### `manifold/clique.py`

- Walk the module method by method.
- `CliqueEmbedding.sub_man` returns the clique's map itself. Should it be the node space?
- `CrossMap` keeps its own `clq_dims`/`coord_blocks`. It could become a `LinearCliques`, but its
  terms are pairs of cliques on two sides, not cliques of one graph.

### `exponential_family/harmonium.py`

- `RootEmbedding` moved here from `manifold/clique.py` (2026-10-03): its only role is
  `pst_prr_emb` for HMoG. Generalize it into a cliquewise posterior-to-prior embedding (one
  embedding per clique block the fill-in reaches; `TensorProductEmbedding` on crossing blocks),
  which would also replace MFA's `CompleteMixtureEmbedding`, and later derive it from the fill-in.

- `Conjugated.extract_likelihood_input` was removed (6e2b96d): `sample` passes the whole prior
  sample to `likelihood_at`. Inferred, not verified: this is correct because `pst_man` is the
  whole deep partition and the interaction reads $y$'s block of it through `clq_emb`.
  `tests/hmog.py::test_sampling` checks shape and finiteness only; add a moment check.
- The interaction is consumed as a `LinearMap` through `lkl_fun_man` / `pst_fun_man`
  (`AffineMap`s). Contracting cliques directly is the agreed next structural step; large blast
  radius.
- `InteractionEmbedding` / `PosteriorEmbedding` have no callers in `models/` or `examples/`
  (only exports and `tests/graphical.py`). Delete?
- `initialize_from_sample` passes the unsliced sample to `obs_man` (predates the branch).

### Models

- MFA (`graphical/mixture.py`): check the `jnp.split` offsets in `from_mixture_coords`. They are
  now covered by the clique-block test in `tests/graphical_mixture.py`.
- HMoG (`graphical/hmog.py`): `_HMoGBase` delegates its crossings to `lwr_hrm`. This holds
  because the lower harmonium's latent and the upper mixture's observable are each node $0$ of
  their partitions. A lower harmonium with a two-node posterior would break this.
- CCA is probabilistic CCA and exposes no canonical directions; rename or document.

### Tests

- `interaction.py` tests a class in `manifold/clique.py`; merge into `graphical.py` or keep?
- CCA test is gradient-step smoke coverage; compare against an independent joint covariance.
- Root cliques of several nodes are possible but untested.

### Pre-existing sharp edges (not this branch)

- Diagonal posteriors do not conjugate exactly (residual ~4.5e-2 for `NormalLGM`).
- `NormalCovarianceEmbedding` with `Scale` is an adjoint pair, not `project ∘ embed = id`.

## Deliberately not implemented

Decided to avoid generalizing before a model needs it.

- **Partial (embedded) biases.** A bias covers its whole node (`bias_map`). Under the composed
  graph this becomes a family's own choice of `clq_map`.
- **Validation of the composed graph.** Nothing checks that crossings touch both partitions,
  that each part is a clique of its partition (it fails only in `clq_emb`, with `ValueError`
  from `cliques.index`), or that the fill-in condition of the conjugation calculus holds.
- **One role for a map read through embeddings** (2026-10-03). $v \mapsto \iota(A(\pi(v)))$ is
  implemented twice: `CliqueMap` and each term of `CrossMap`. Reconsider if a third occurrence
  appears; it would belong in `map.py`.

## Later (out of scope)

- Convolutional `MatrixRep`.
- Generic conjugation cascade over `RecursiveLinearCliques` (see the calculus above), and the
  variational $\rho$ slot. Trap: ~10 sites unpack `split_coords` positionally; rename in the same
  change.
- Short user-facing architecture page (chain, fork, arity-3 example).

## Verification

| gate | command |
|---|---|
| lint | `uvx ruff check src/ tests/` |
| types | `uvx basedpyright src/ tests/` |
| docs | `uv run sphinx-build -q docs/source docs/build` |
| suite | `uv run python -m pytest tests/ -q` (531 passed on 2026-10-02, ~17 min) |
| numeric | `uv run python -m examples.cca.run` (CPU): alignment RMSE `0.24031811353161686` (2026-10-01) |
