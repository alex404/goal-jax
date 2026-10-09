# Clique container: remaining items

Branch `clique-container`. The geometry work is done: coordinate blocks, the composed graph,
`CrossMap`/`CliqueMap`, `GraphicalHarmonium`/`DifferentiableGraphical`/`AnalyticGraphical`, and the
variational review (`fe5fa40`). The design is described in `CLAUDE.md` (Architecture); earlier
designs and the full design notes are in git history (this file before 2026-10-09).

## Before merging

- `examples/variational_mnist` is broken (imports `EmbeddedMap`/`BlockMap`/`VariationalSymmetric`,
  calls `coord_blocks`). Experimental; fix or leave out of the merge.
- Audit every example built on the variational classes (pendulum, torus_poisson,
  chordal_boltzmann_ppc, population_codes, variational_mnist, canonical_circuit); they were only
  type- and import-checked after the review. The pendulum example also needs a run.
- Planned follow-up: the conjugation function as a `Map[Prior, Likelihood]`, generic level classes,
  and deleting `VariationalHierarchicalMixture` (`models/graphical/variational.py`).

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
| types | `uvx basedpyright src/ tests/` (known: `mixture.py:468`) |
| docs | `uv run sphinx-build -q docs/source docs/build` |
| suite | `uv run python -m pytest tests/ -q` |
| variational | `uv run python -m pytest tests/graphical_variational.py tests/variational.py` (85 passed, about 13 min on the laptop, 2026-10-09) |
