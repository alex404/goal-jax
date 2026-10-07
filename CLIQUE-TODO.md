# Clique container: plan

Branch `clique-container`. Current state: the **composed-graph** design (2026-10-03, uncommitted
on top of `5dd6647`), described in `CLAUDE.md` (Architecture). Earlier designs (`LinearClique`,
`BlockMap`, `EFClique`, `CliqueSet`, `Interaction`, global labels with groups) are in git
history.

## Current state

- **Blocks (2026-10-06).** Each node has a space (`nod_mans`) and each clique one block
  (`clq_mans`), a subspace of the tensor product of its nodes' spaces. The block of a single node
  is its space. Every multi-node clique is a crossing, whose block is a two-sided `CliqueMap`
  from a subspace of the touched deep block to a subspace of the touched root block; its
  embeddings have those blocks as ambients, which is the posterior-in-family condition. A
  touched block that is itself a map is restricted by `SubMapEmbedding`. Replaced: arity-$n$
  `CliqueMap`s with per-node axes, `TensorProduct`, `TensorProductEmbedding`, `bias_map`,
  `clq_maps`, `crs_rep` and `crs_emb_constructors`.
- Every `ExponentialFamily` is a `LinearCliques`. By default it is one node, `cliques = ((0,),)`,
  whose block is the family itself. `ExponentialFamilyPair` places its components' graphs side
  by side. A harmonium composes its graph.
- A crossing's block is oriented (root block against deep block), and that orientation is part
  of the parameterization.
- Node numbers are local: each `LinearCliques` numbers its own nodes $0, \ldots, n-1$.
- `CrossMap(cod_man, dom_man, trms)`: each term is a `CrossTerm(cod_clq, dom_clq, clq_map)`, with
  each clique in its own side's numbering, read and written through `CliqueEmbedding`.
- `RecursiveLinearCliques[Root: LinearCliques, Deep: LinearCliques]` composes its graph from its
  parts.
  - A model declares `rot_man`, `dep_man` and `crs_trms` (one `CrossTerm` per crossing: a root
    clique and a deep clique with its `CliqueMap`); `crs_man` is the `CrossMap` with those terms.
  - Derived: `crs_clqs`, `crs_maps`, the numbering (root first, deep offset by the root's node
    count), `cliques` (root, crossings, deep: the triple's storage order), `clq_mans` and
    `nod_mans` (concatenated in the same order), `crs_man`.
- Deleted: `part_emb`, `cod_group`/`dom_group`, the bias branch, `split_clique`, `Mixture.impose`,
  `upr_graph`, `mix_nodes`/`mix_graph`, and every `_raw_cliques`/`_root_nodes` field.
- A clique's block manifold is `clq_man(clique)`, and its coordinates are read by
  `CliqueEmbedding(clique, man)`.
- `RecursiveCliques` is deleted. Its levels are distances from the root set, which agree with
  the partition nesting for chains but not in general: MFA's distance levels are x | {y, k},
  its nesting x | (y | k). The layout and conjugation follow the nesting.
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

- A crossing has one orientation (root block against deep block), fixed by the level that
  introduced it, and supports only that reading and its transpose. A conditional that
  groups a clique's nodes differently (MFA's $k \mid x, y$, for Gibbs over arbitrary nodes)
  needs a re-matricization `CliqueMap` does not have. For `Rectangular` cliques it would be a
  reshape and transpose. No current use needs it.
- A crossing may touch a block with a structured (non-`Rectangular`) representation: its
  embedding is into the block as a whole. `SubMapEmbedding` restricts rows and columns through
  `to_matrix`/`from_matrix`, so on a structured block `embed` keeps only the entries the
  representation stores. No current model touches a structured block; untested.
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
- `CrossMap` keeps its own `clq_dims`/`clq_coords`. It could become a `LinearCliques`, but its
  terms are pairs of cliques on two sides, not cliques of one graph.

### `exponential_family/graphical.py`

Built 2026-10-03: `GraphicalHarmonium` (a deep model plus observable harmoniums on cliques `obs_hrms_clqs`, crossings
derived), `DifferentiableGraphical` (conjugation = each observable harmonium's, placed on its clique of
the prior and summed; offsets summed), `AnalyticGraphical` (single observable harmonium). Placement and
reading go through `SubCliquesEmbedding` (2026-10-05). Openings:

- Several observable harmoniums give a right-nested `ObservablePair` (differentiable only; CCA
  uses it). The crossing offsets in `_crossing_sources` and the block splits in
  `obs_likelihoods` restate the pair's side-by-side layout instead of reading it off the pair.
- An observable harmonium's prior is placed clique by clique through its clique map, so a prior with
  more nodes than its posterior (a fill-in across observable harmoniums) is not expressible.
- `pst_prr_emb` stays declared by the model; deriving it from the observable harmoniums is open.
- Depth: a deep model that is itself graphical composes, but nothing tests depth three yet.
- MFA stays a hand-written primitive.
- `VariationalGraphical` and its `Conjugation` classes were removed (2026-10-06): a graphical
  harmonium is fit variationally as the `gen_hrm` of a `VariationalDifferentiable`, whose prior
  may itself be variational (`VariationalPrior` in `variational.py`). Open:
  - the posterior variables are read as the leading slice of a prior datapoint
    (`VariationalConjugated._latent_stats`); a `gen_hrm` whose posterior is not the prior's
    leading node needs per-node data slices;
  - residual variances are per level, not per observable harmonium;
  - pathwise gradients were dropped with `pathwise_top`;
  - `VariationalConjugated` is still a `Generative` whose `sufficient_statistic` and
    `log_base_measure` delegate to `gen_hrm`, which is not the joint over a nested prior.
- Concrete composites (`_Chain` in the tests)
  are small field-holding subclasses; a generic concrete composite in core is open.

### `exponential_family/harmonium.py`

- `RootEmbedding` (previously `LatentHarmoniumEmbedding` in `graphical.py`) moved back to
  `manifold/clique.py` (2026-10-05), typed over `RecursiveLinearCliques`, next to
  `SubCliquesEmbedding`: its only role is `pst_prr_emb` for HMoG. Generalize it into a cliquewise
  posterior-to-prior embedding (one embedding per clique block the fill-in reaches;
  `SubMapEmbedding` on crossing blocks), which would also replace MFA's
  `CompleteMixtureEmbedding`, and later derive it from the fill-in.

- `Conjugated.extract_likelihood_input` was removed (6e2b96d): `sample` passes the whole prior
  sample to `likelihood_at`. Inferred, not verified: this is correct because `pst_man` is the
  whole deep partition and the interaction reads $y$'s block of it through `clq_emb`.
  `tests/hmog.py::test_sampling` checks shape and finiteness only; add a moment check.
- The interaction is consumed as a `LinearMap` through `lkl_fun_man` / `pst_fun_man`
  (`AffineMap`s). Contracting cliques directly is the agreed next structural step; large blast
  radius.
- `InteractionEmbedding` / `PosteriorEmbedding` have no callers in `models/` or `examples/`
  (only exports and `tests/linear_clique.py`). Delete?
- `initialize_from_sample` passes the unsliced sample to `obs_man` (predates the branch).

### Models

- MFA (`graphical/mixture.py`): check the `jnp.split` offsets in `from_mixture_coords`. They are
  now covered by the clique-block test in `tests/graphical_mixture.py`.
- CCA (`harmonium/cca.py`) is a `DifferentiableGraphical` with two observable harmoniums on latent node
  $0$ (2026-10-03); its hand-written conjugation sum is gone.
- HMoG (`graphical/hmog.py`) is a `DifferentiableGraphical` with one observable harmonium, the lower
  harmonium on the mixture's node $0$ (2026-10-03); crossings, conjugation and
  `to_natural_likelihood` come from `exponential_family/graphical.py`.
- CCA is probabilistic CCA and exposes no canonical directions; rename or document.

### Tests

- `interaction.py` tests a class in `manifold/clique.py`; merge into `linear_clique.py` or keep?
- CCA test is gradient-step smoke coverage; compare against an independent joint covariance.
- Root cliques of several nodes are possible but untested.

### Pre-existing sharp edges (not this branch)

- Diagonal posteriors do not conjugate exactly (residual ~4.5e-2 for `NormalLGM`).
- `NormalCovarianceEmbedding` with `Scale` is an adjoint pair, not `project ∘ embed = id`.

## Deliberately not implemented

Decided to avoid generalizing before a model needs it.

- **Partial (embedded) biases.** A single node's block is its whole space. A partial bias would
  be a block given as an embedding into `nod_mans[i]`; crossings through that node would then
  nest inside it. The block design admits it; no model needs it yet.
- **Validation of the composed graph.** Nothing checks that crossings touch both partitions,
  that each part is a clique of its partition (it fails only in the clique lookup, with `ValueError`
  from `cliques.index`), or that the fill-in condition of the conjugation calculus holds.
- **One role for a map read through embeddings** (2026-10-03). $v \mapsto \iota(A(\pi(v)))$ is
  implemented twice: `CliqueMap` and each term of `CrossMap`. Reconsider if a third occurrence
  appears; it would belong in `map.py`.

## Later (out of scope)

- Convolutional `MatrixRep`.
- Generic conjugation cascade over `RecursiveLinearCliques` (see the calculus above), and the
  variational $\rho$ slot. Trap: ~10 sites unpack `split_coords` positionally; rename in the same
  change.
- Short user-facing architecture page (chain, fork, MFA's three-node crossing).

## Verification

| gate | command |
|---|---|
| lint | `uvx ruff check src/ tests/` |
| types | `uvx basedpyright src/ tests/` |
| docs | `uv run sphinx-build -q docs/source docs/build` |
| suite | `uv run python -m pytest tests/ -q` (531 passed on 2026-10-02, ~17 min) |
| numeric | `uv run python -m examples.cca.run` (CPU): alignment RMSE `0.24031811353161686` (2026-10-01) |
