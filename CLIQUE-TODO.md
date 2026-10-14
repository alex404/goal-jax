# Clique container: remaining items

Branch `clique-container`. The geometry work is done: coordinate blocks, the composed graph,
`CrossMap`/`CliqueMap`, `GraphicalHarmonium`/`DifferentiableGraphical`/`AnalyticGraphical`, and the
variational review (`fe5fa40`). The design is described in `CLAUDE.md` (Architecture); earlier
designs and the full design notes are in git history (this file before 2026-10-09).

## Before merging

- Every example was run and its plot inspected on 2026-10-11 (pendulum, torus_poisson,
  chordal_boltzmann_ppc and population_codes included). After the population code unification of
  2026-10-12, population_codes, torus_poisson, pendulum and chordal_boltzmann_ppc were rerun and
  their plots inspected. canonical_circuit was rewritten on 2026-10-14 (natural parameters, plain
  ELBO, one fit) and its plot inspected.
- pendulum (2026-10-14): the transition's precision floor (`precision_epsilon`) and the gradient
  clipping are removed; the transition is a perceptron whose velocity precision is a softplus, as
  in `NonnegativePrecisionMap`. $\rho$ stays constant and unconstrained: over 3000 steps the smallest
  recognition velocity precision, at a silent observation, was 0.09 at initialization and 1.7 at
  the end, and every velocity-squared interaction stayed negative. Rerun and plot inspected.
- `VariationalConjugated` is concrete, with one conjugation function per observable harmonium (see
  **Variational design** below). `VariationalHierarchicalMixture` (`models/graphical/variational.py`)
  is deleted.
- Then: full test suite, every example run and plotted (one at a time, `--gpu` allowed), and the plots
  inspected. Results from before 2026-10-12 are in the session scratchpad only, not in the repo.

## Review of `models/` and `dynamical.py`

Geometry, including `graphical.py`, is reviewed. What remains against `main`, by how much attention
it needs. Line counts are added/removed against `main`, taken on 2026-10-12 with the population code
unification uncommitted on top of `b34c832`.

### Tick off

The changes are mechanical: the removed `from __future__ import annotations`, ruff formatting,
target-first type parameters and arguments, and `int_man` → `crs_trms`/`crs_maps`.

- `base/dirichlet.py` (+0/−2), `base/gaussian/generalized.py` (+0/−2): import only.
- `base/binomial.py` (+2/−4), `base/poisson.py` (+2/−4): the same; `PopulationLocationEmbedding`
  has its type parameters flipped.
- `base/von_mises.py` (+2/−2): docstring note on the conjugation offset, no code.
- `models/__init__.py` (+33/−19): exports.
- `graphical/variational.py` (−80): deleted.
- `harmonium/mixture.py` (+61/−23): the port is a reduction. `cmp_int_map` reads the per-clique
  crossings of `CompleteMixture.crs_trms` as one matrix, which is the storage order every
  `LinearCliques` has (coordinate blocks in clique order) together with row-major `Rectangular`
  storage; exercised by the MFA and mixture-of-conjugated cases in `tests/harmonium.py` and
  `tests/graphical.py`.

The `float32` → `float` change alters dtypes: under `jax_enable_x64=True` samples and sufficient
statistics are now float64. The tests would not notice either way.

### Look at specific lines

- `graphical/mixture.py` (+106/−182): mostly deletion of `RowEmbedding`, `BlockMap` and the
  composed embeddings. `to_mixture_coords`/`from_mixture_coords` were checked: `concatenate`/`split`
  equals the old reshape/`vstack`/`ravel` under row-major storage. Look at the `pst_man` pair: the
  base returns `bas_pst_man`, `CompleteMixtureOfSymmetric` overrides with `lat_man` so the analytic
  subclass gets `to_natural`.
- `graphical/hmog.py` (+88/−16): a port to `DifferentiableGraphical`, no new mathematics. Look at
  the `RootEmbedding` in `pst_prr_emb` and the attachment `((lwr_hrm, (0,)),)`. `_HMoGBase` gained
  `@dataclass(frozen=True)`.
- `base/categorical.py` (+5/−5): `Categorical.sufficient_statistic` now casts its input to int (a
  fix for float-valued indices).
- `geometry/exponential_family/dynamical.py` (+28/−19): **reviewed 2026-10-12.** `emission_params`
  sets the latent bias to belief − $\rho$, so the conjugated prior $\theta_Z + \rho$ is the
  predicted belief; a new test checks exactly that (`test_emission_prior_is_predicted_belief`; the
  quadrature test could not, since it builds its reference through `emission_params`). The filter's
  slot is named `cnj_fun_params`.
- `variational/population_codes.py`: **reviewed 2026-10-12.** `VariationalPopulationCode` (constant
  $\rho$ on the whole latent) and `PopulationCodeProcess`, merged into one module (the process was
  `variational/population_code_process.py`). The process takes any population code: its observable
  type parameter is `Any` (it was `AnalyticTuple[Poissons]`, which did not match `lkl_fun_man`), and
  its fields are `pop_man` and `_trn_map` (were `population_code` and `transition`).

### Changes behaviour: review these

- `base/gaussian/normal.py`, `initialize_from_sample`: **reviewed 2026-10-12.** The noise factor
  $(\mathbf 1 \mathbf 1^T + AA^T/d)$ multiplies the covariance entrywise (previously $(I + AA^T/d)$
  multiplied the second moment, which reduced the off-diagonal second moments to noise, followed by
  a clamp), so the sample covariance is kept on average and the variances grow by $1 +
  \text{shape}^2$. For shape $> 0$ the result is positive definite also for a singular sample
  covariance ($AA^T$ is almost surely positive definite; Schur product theorem); at shape $= 0$ it
  is the sample-moment fit, which needs a positive definite sample covariance. The docstring
  argument was corrected (it assumed a positive definite sample covariance), and
  `test_initialize_from_sample_is_valid_at_any_noise` now includes a linearly dependent dimension.
  Every model initialized from a sample starts elsewhere than on `main`.
- `base/gaussian/boltzmann.py`: **audited and restructured** (2026-10-12, uncommitted). Each
  Boltzmann family runs its own algorithms; the shape manifolds `CouplingMatrix` and
  `ChordalCouplingMatrix` are layouts only (an `ExponentialFamily` with statistics and matrix
  packing, like `Covariance` for `Normal`), and the `Shape` bound of `Boltzmann`,
  `NormalBoltzmannMachine` and `BoltzmannPopulationCode` is `ExponentialFamily`. The `coupling/`
  package is gone: the dense kernels are methods of `FullBoltzmann` (`energies`, `states`,
  `unit_conditional_prob`, `gibbs_step`), and the junction trees moved to
  `geometry/algebra/junction_tree.py` (below). `FullBoltzmann.sample` is exact, by inverse-CDF
  sampling over the $2^n$ states; Gibbs remains as `gibbs_step`/`gibbs_chain` (the `boltzmann`
  example runs its chains through `gibbs_chain`). Benchmark (1000 samples per call, after
  compilation): enumeration takes 0.07/1.3/27 ms on CPU and 0.13/0.34/4.4 ms on GPU at
  $n = 4/16/20$, against 40/215/265 ms (CPU) and 6/23/29 ms (GPU) for 1000 vmapped Gibbs chains
  of 100 sweeps. Deleted: `EnumeratedBoltzmann` (the circuit uses `FullBoltzmann`),
  `ChainBoltzmann`, `ChainCouplingMatrix`. New on the branch: `coupled_pairs` (the order of the
  stored off-diagonal couplings).
- `geometry/algebra/junction_tree.py` (moved from `models/base/gaussian/coupling/`,
  2026-10-12, uncommitted). **Audited**: triangulation, maximal cliques, spanning tree, collect,
  ancestral sampling and both chain kernels read and checked by argument, and fuzzed against
  enumeration on 300 random graphs ($n \le 10$; $\log Z$ and its gradient to $2 \cdot
  10^{-15}$, sample frequencies to $0.005$). `JunctionTree` is a `Cliques` (its `edges` replace
  `chordal_edges`; `n_nodes` is derived), and inference is `log_partition(phi)` and
  `sample(key, phi)` on a table of clique log-potentials over binary states, which `ChainTree`
  overrides with the `associative_scan` kernels. `ChordalCouplingMatrix.clique_potentials` builds
  that table from biases and couplings, so `ChordalBoltzmann` no longer dispatches on the tree
  type. The topology and the static tables are computed by numpy on the clique × node
  `incidence` (separator sizes `B @ B.T`, owners by `argmax`, maximal cliques by one containment
  product); one traversal gives `pre_order` and `parent_of`; a `ChainTree` is rooted at an end of
  its path, so its path is its pre-order; and both samplers draw clique states and read each node
  from its first clique. 32 loop statements became 12 (triangulation and Kruskal stay sequential).
  Compiled kernels run as before (CPU, $n = 1000$ chain, 300-node band and tree); construction
  takes 107–151 ms instead of about 40 ms at $n = 1000$, from the $O(n^3)$ containment and overlap
  products. New tests in `tests/algebra.py`: sum-product at random, non-pairwise potentials against
  enumeration (log-partition, gradient as clique marginals, joint of the samples) on hand-picked
  and random graphs.
- `harmonium/lgm.py` (+91/−128): **reviewed 2026-10-12**, no changes. The exact model enumerates the
  dense prior; variational uses avoid it (the approximate canonical circuit runs at 64 chain neurons
  per layer). `NormalBoltzmannMachine[ObsRep, Shape](obs_dim, obs_rep,
  boltzmann)` replaces `BoltzmannLGM`, `DifferentiableBoltzmannLGM` and `NormalBoltzmannHarmonium`:
  the Boltzmann analogue of `NormalLGM`, any Boltzmann posterior, `FullBoltzmann` prior, exactly
  conjugated. `BoltzmannCouplingEmbedding` (any Boltzmann into dense, a scatter by `coupled_pairs`)
  generalizes `BoltzmannEmbedding` (diagonal into dense). New tests: a chordal-posterior case in
  `tests/harmonium.py`, a chordal embedding test in `tests/gaussian.py`.
- `harmonium/population_codes.py` (+99/−180): **the model set changed.**
  `PopulationCode[Observable, Latent](_pst_man)` couples one population to every clique of its latent
  through `nrn_emb`, the neuron statistics the latent modulates: all of them for
  `PoissonPopulationCode(_pst_man, n_neurons)`, the biases only for `BoltzmannPopulationCode(_pst_man,
  pop_man)`, whose couplings, and so noise correlations, do not depend on the latent. A latent with
  an edge gives non-separable tuning. Subpopulations (codes attached to different nodes of one
  latent) are not shipped: `GraphicalPopulationCode`, a `GraphicalHarmonium` with a
  `DifferentiableTuple` observable, was written and then removed on 2026-10-12, since nothing outside
  the tests used it (it is in git history). Gone: `PoissonVonMisesHarmonium`
  (on `main`); `PoissonPopulationHarmonium` and its `tuning` pairs, `BoltzmannNormalHarmonium` (the
  latent also modulated the couplings) and `PopulationCodeHarmonium` (on the branch).
  `initialize_from_tuning_curves` moved to `examples/population_codes/run.py` and
  `examples/pendulum/run.py`. On the pendulum, subpopulation and circuit layouts the old and new
  classes give identical sufficient statistics, likelihoods and posteriors (checked in a scratch
  script, not in the suite). `chordal_boltzmann_ppc` now fits the bias-only coupling, so its results
  changed.
- `harmonium/cca.py` (+166, new): **reviewed 2026-10-12**, no changes. Generalizing (several views,
  Boltzmann latents) is deferred.
- `variational/canonical_circuit.py` (+532, new). Its population code level is
  `BoltzmannPopulationCode[Any, DiagonalNormal]`, and `_generative_neuron_params` uses
  `BoltzmannCouplingEmbedding`. **Reviewed 2026-10-12**: `GaussianBoltzmannLevel` and
  `PopulationCodeLevel` replaced by the generic `VariationalLevel` (geometry, with
  `ExactConjugationFunction`); they remain as type aliases. `PopulationConjugationFunction` is a `Map`.
  Exact-enumeration docstrings state the normalizability precondition; "on $E$" corrected to the
  chordal completion. `VariationalPopulationCode` stays a `DifferentiableVariationalConjugated`
  (it needs `elbo_divergence`). A level holds one attached harmonium; several (as in
  `DifferentiableGraphical`) is not done.

## Open

- **Conjugation calculus.** `DifferentiableGraphical` sums the conjugation parameters of whole
  attached harmoniums, each placed on its clique (independent root components, locality). Not done:
  deriving the root components from an arbitrary layout; MFA as a primitive in the same scheme (it is
  hand-written); a prior with more nodes than its posterior (fill-in across attached harmoniums);
  deriving `pst_prr_emb`; where a variational $\rho$ lives (the fill-in cliques).
- **Variational.** The posterior variables are read as the leading slice of a prior datapoint
  (`likelihood_at`), so a `gph_hrm` whose posterior is not the prior's leading node needs per-node
  slices. Residual variances are per level, not per attached harmonium.
- **Crossings.** One orientation per crossing (root against deep); a conditional that regroups a
  clique's nodes needs a re-matricization `CliqueMap` does not have. A crossing touching a structured
  (non-`Rectangular`) coordinate block is untested. `Mixture` with a general `obs_emb` assumes a
  one-node observable.
- **Harmonium.** Contracting cliques directly instead of consuming the interaction as a `LinearMap`
  through `lkl_fun_man`/`pst_fun_man` (large blast radius). Generalize `RootEmbedding` into a
  cliquewise posterior-to-prior embedding (would replace `CompleteMixtureEmbedding`).
  `InteractionEmbedding`/`PosteriorEmbedding` have no callers outside exports and tests: delete?
- **Tests.** HMoG sampling checks shape only (add a moment check). Root cliques of several nodes are
  untested. "Block" survives in test names and prose (about 60 places).
- **Naming.** CCA is probabilistic CCA and exposes no canonical directions; rename or document.
- **Pre-existing.** Diagonal posteriors do not conjugate exactly (residual about 4.5e-2 for
  `NormalLGM`). `NormalCovarianceEmbedding` with `Scale` is an adjoint pair, not `project ∘ embed =
  id`.

## Variational design (decided 2026-10-13)

- **`VariationalConjugated(gph_hrm, cnj_funs, dep_man)` is concrete**, one level of a deep model:
  a graphical harmonium, one conjugation function (`Map` from likelihood to $\rho$) per observable
  harmonium, and the deep model of its latent. `VariationalLevel`, its contract
  (`cnj_fun_man`/`pst_prr_emb`/`conjugation_parameters` as abstract members) and the level aliases are
  gone; $\rho$ is the per-harmonium sum of `DifferentiableGraphical`, so several harmoniums per level
  (CCA) are expressible, untested. `DifferentiableVariationalConjugated` narrows `dep_man` and adds
  `elbo_divergence`. The latent model of each level is its prior family (no prior embedding).
- **`DeepModel`** is the contract a level needs from the model of its latent, shared by the next level
  and by the latent family that ends the recursion (`DifferentiableDeepModel`).
- **`GraphicalLevel(_obs_hrms_lat_nods, _pst_man)`** replaces `AttachedHarmonium`: the concrete
  graphical harmonium, for models assembled from parts. "Attached" is replaced by "placed on" and
  `att_clq(s)` by `lat_nods`/`lat_nodss`.
- **Constant $\rho$.** `ConstantConjugationFunction` in `variational.py`. `variational_population_code(pop_hrm)` is a factory
  (`VariationalPopulationCode` is a type alias). The bottom Gaussian-Boltzmann level of the approximate
  canonical circuit uses a `ConstantConjugationFunction`: its likelihood does not depend on the observations, so at
  given parameters $\rho_{N_1}$ is one point. This changes the approximate circuit's parameter count
  and results; the partially exact circuit and the population code are unchanged (parameters and ELBO
  equal at the same keys).
- **`ManifoldTuple`** (`manifold/combinators.py`) replaces `ConjugationTuple` and
  `GenerativeCoordinates`; `Replicated` is renamed `ManifoldProduct` (Tuple = heterogeneous, Product =
  homogeneous, at both tiers).
- **Naming scheme** (proposed, not written into `CLAUDE.md`): suffix is the role; prefix is a
  capability tier (`Generative`/`Differentiable`/`Analytic`) or a construction (`Manifold`/
  `ExponentialFamily` + `Tuple`, `Composed`, `Identity`, `Constant`, `Clique`, `Exact`, ...).
  `SymmetricConjugated` is the documented exception (structural prefix).
- **Open.** The structure is still stated twice (`gph_hrm.pst_man` and `dep_man.fst_man`), unchecked
  at run time. `_harmonium` and the `Level` alias are gone; `_place` remains. Whether the fitting functions become methods: decided to stay functions.
- **Bias conjugation functions (2026-10-13).** A learned conjugation function depends on the observable
  bias alone (`BiasConjugationFunction`): the prior and the recognition model evaluate it at the same
  interactions. Stated in the notes (`scratch/variational-conjugation/article.tex`, sections 3.1 and 4).
- **Canonical circuit (2026-10-14).** The generative coordinates (`tie`, `gen_man`) and the tied
  $\beta$ are gone: the circuit is the manifold, and the example trains its natural parameters by the
  ELBO. The population code's conjugation function adds precision (`NonnegativePrecisionMap`), so the
  prior and the recognition model over $z_k$ are Gaussian wherever $\theta_{Z_k}$ is; $\theta_{Z_k}$
  itself is unconstrained. The layers are given by `neural_connectivity` (junction trees).
  `exact_log_partition_function` and `exact_log_observable_density` are removed: they enumerated the
  undirected graphical harmonium, a different model from the directed $\tilde p$ the circuit is
  (the two agree only when every residual is constant), and their only uses were tests at
  decoupled parameters, which `tests/harmonium.py` already covers.

## Deliberately not implemented

Decided to avoid generalizing before a model needs it.

- **Partial (embedded) biases.** A single node's coordinate block is its whole space.
- **Single-node coordinate blocks as `CliqueMap`s** (rejected 2026-10-07): the case split between
  nodes and crossings moves rather than disappears.
- **Validation of the composed graph.** Nothing checks that crossings touch both partitions, that each
  part is a clique of its partition (it fails in the clique lookup), that nodes are numbered $0,
  \ldots, n - 1$, or that the fill-in condition holds.
- **One role for a map read through embeddings.** $v \mapsto \iota(A(\pi(v)))$ is implemented twice
  (`CliqueMap`, each term of `CrossMap`); move it to `map.py` if a third occurrence appears.

## Verification

| gate | command |
|---|---|
| lint | `uvx ruff check src/ tests/` |
| types | `uvx basedpyright src/ tests/ examples/` (0 errors, 2026-10-12) |
| docs | `uv run sphinx-build -q docs/source docs/build` |
| suite | `uv run python -m pytest tests/ -q` (609 passed, about 16 min on the laptop, 2026-10-14, before 12 circuit tests were removed the same day; not rerun since) |
| variational | `uv run python -m pytest tests/variational.py` (28 passed, about 3 min on the laptop, 2026-10-14) |
