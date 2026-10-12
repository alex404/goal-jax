# Clique container: remaining items

Branch `clique-container`. The geometry work is done: coordinate blocks, the composed graph,
`CrossMap`/`CliqueMap`, `GraphicalHarmonium`/`DifferentiableGraphical`/`AnalyticGraphical`, and the
variational review (`fe5fa40`). The design is described in `CLAUDE.md` (Architecture); earlier
designs and the full design notes are in git history (this file before 2026-10-09).

## Before merging

- Every example was run and its plot inspected on 2026-10-11 (pendulum, torus_poisson,
  chordal_boltzmann_ppc and population_codes included). After the population code unification of
  2026-10-12, population_codes, torus_poisson, pendulum and chordal_boltzmann_ppc were rerun and
  their plots inspected. canonical_circuit is ported onto the core module; its full grid has not
  been rerun since the port (one condition was, and matches).
- Planned follow-up: the conjugation function as a `Map[Prior, Likelihood]` and generic level
  classes. `VariationalHierarchicalMixture` (`models/graphical/variational.py`) is deleted.

## Review of `models/` and `dynamical.py`

Geometry, including `graphical.py`, is reviewed. What remains against `main`, by how much attention
it needs. Line counts are added/removed against `main`, taken on 2026-10-12 with the population code
unification uncommitted on top of `b34c832`.

### Tick off

The changes are mechanical: the removed `from __future__ import annotations`, ruff formatting,
target-first type parameters and arguments, and `int_man` → `crs_trms`/`crs_maps`.

- `base/dirichlet.py` (+0/−2), `base/gaussian/generalized.py` (+0/−2): import only.
- `base/gaussian/coupling/junction_tree.py` (+5/−6), `sum_product.py` (+14/−22), `dense.py`
  (+2/−4): formatting, `float32` → `float`.
- `base/binomial.py` (+2/−4), `base/poisson.py` (+2/−4): the same; `PopulationLocationEmbedding`
  has its type parameters flipped.
- `base/von_mises.py` (+2/−2): docstring note on the conjugation offset, no code.
- `models/__init__.py` (+33/−19): exports.
- `graphical/variational.py` (−80): deleted.

The `float32` → `float` change alters dtypes: under `jax_enable_x64=True` samples and sufficient
statistics are now float64. The tests would not notice either way.

### Look at specific lines

- `harmonium/mixture.py` (+61/−23): the port is a reduction. Look at `cmp_int_map` against
  `CompleteMixture.crs_trms`: with a multi-clique observable, the interaction is one crossing per
  clique, and `cmp_int_map` reads them as one matrix. This relies on the clique coordinate blocks
  being consecutive row bands under row-major storage; documented, but an implicit invariant.
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
- `base/gaussian/boltzmann.py` (+54/−4): new `EnumeratedBoltzmann` (exact sampling by enumeration;
  ignores `n_burnin`/`n_thin`, as documented), and the contract `coupled_pairs` (the order of the
  stored off-diagonal couplings) on every Boltzmann.
- `geometry/exponential_family/dynamical.py` (+28/−19): look at `emission_params` (latent bias =
  belief − $\rho$); the rest is the rename to the new variational API.
- `variational/population_code_process.py` (+53, new): three one-line overrides, over a
  `VariationalPopulationCode`.
- `variational/population_codes.py` (+53, new): `VariationalPopulationCode`, four one-line
  overrides (constant $\rho$ on the whole latent).

### Changes behaviour: review these

- `base/gaussian/normal.py`, `initialize_from_sample` (+20/−25): **initialization behaviour
  changed.** Previously the noise factor $(I + AA^T/d)$ multiplied the second moment entrywise,
  which reduced the off-diagonal second moments to noise, followed by a clamp. Now
  $(\mathbf 1 \mathbf 1^T + AA^T/d)$ multiplies the covariance, so the sample covariance is kept up
  to noise, and positive definiteness follows from the Schur product theorem. Every model
  initialized from a sample now starts elsewhere; no test checks initialization.
- `harmonium/lgm.py` (+91/−128): `NormalBoltzmannMachine[ObsRep, Shape](obs_dim, obs_rep,
  boltzmann)` replaces `BoltzmannLGM`, `DifferentiableBoltzmannLGM` and `NormalBoltzmannHarmonium`:
  the Boltzmann analogue of `NormalLGM`, any Boltzmann posterior, `FullBoltzmann` prior, exactly
  conjugated. `BoltzmannCouplingEmbedding` (any Boltzmann into dense, a scatter by `coupled_pairs`)
  generalizes `BoltzmannEmbedding` (diagonal into dense). New tests: a chordal-posterior case in
  `tests/harmonium.py`, a chordal embedding test in `tests/gaussian.py`.
- `harmonium/population_codes.py` (+99/−180): **the model set changed.**
  `PopulationCode[Observable, Latent](latent)` couples one population to every clique of its latent
  through `nrn_emb`, the neuron statistics the latent modulates: all of them for
  `PoissonPopulationCode(latent, n_neurons)`, the biases only for `BoltzmannPopulationCode(latent,
  neurons)`, whose couplings, and so noise correlations, do not depend on the latent. A latent with
  an edge gives non-separable tuning. `GraphicalPopulationCode(latent, codes_att_clqs)` attaches
  several codes to one latent as subpopulations; it overrides `obs_man` with a `DifferentiableTuple`,
  since the variational ELBO needs its log-partition function. Gone: `PoissonVonMisesHarmonium`
  (on `main`); `PoissonPopulationHarmonium` and its `tuning` pairs, `BoltzmannNormalHarmonium` (the
  latent also modulated the couplings) and `PopulationCodeHarmonium` (on the branch).
  `initialize_from_tuning_curves` moved to `examples/population_codes/run.py` and
  `examples/pendulum/run.py`. On the pendulum, subpopulation and circuit layouts the old and new
  classes give identical sufficient statistics, likelihoods and posteriors (checked in a scratch
  script, not in the suite). `chordal_boltzmann_ppc` now fits the bias-only coupling, so its results
  changed.
- `harmonium/cca.py` (+166, new).
- `variational/canonical_circuit.py` (+532, new). Its population code level is
  `BoltzmannPopulationCode[Any, DiagonalNormal]`; `GaussianBoltzmannLevel` computes exact $\rho_N$
  from its own `gb_hrm`, and `_generative_neuron_params` uses `BoltzmannCouplingEmbedding`.

## Open

- **Conjugation calculus.** `DifferentiableGraphical` sums the conjugation parameters of whole
  attached harmoniums, each placed on its clique (independent root components, locality). Not done:
  deriving the root components from an arbitrary layout; MFA as a primitive in the same scheme (it is
  hand-written); a prior with more nodes than its posterior (fill-in across attached harmoniums);
  deriving `pst_prr_emb`; where a variational $\rho$ lives (the fill-in cliques).
- **Variational.** The posterior variables are read as the leading slice of a prior datapoint
  (`likelihood_at`), so a `gen_hrm` whose posterior is not the prior's leading node needs per-node
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
| suite | `uv run python -m pytest tests/ -q` |
| variational | `uv run python -m pytest tests/variational.py` (54 passed, about 2.5 min on the laptop, 2026-10-12) |
