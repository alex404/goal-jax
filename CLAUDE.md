# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

### General Points

1. Ask, don't assume. If something is unclear, ask before writing a single line. Never make silent assumptions about intent, architecture, or requirements. When running unattended, pick the most reasonable interpretation, proceed, and record the assumption rather than blocking.

2. Implement the simplest solution for simple problems, better solutions for harder problems. Do not over-engineer or add flexibility that isn't needed yet. 

3. Don't touch unrelated code but please do surface bad code or design smells you discover with me so we can address them as a separate issue.

4. Flag uncertainty explicitly. If you're unsure about something, see point 1 above. If it makes sense to do so, conduct a small, localised and low-risk experiment and bring the hypothesis and results to me to discuss. Confidence without certainty causes more damage than admitting a gap.

5. I'm always open to ideas on better ways to do things. Please don't hesitate to suggest a better way, or one that has long lasting impact over a tactical change. (as a few examples)

## Project Overview

goal-jax is a JAX implementation of information geometric algorithms and exponential families. The library provides geometric optimization tools for statistical models, focusing on manifold-based optimization and exponential family distributions.

### Library boundaries (goal-jax vs goal-apps)

goal-jax is the **core library**; `../goal-apps` is the application layer built on top of it (CLI training framework with plugin-based models/datasets, Hydra configs, W&B integration). The division of responsibility:

- **Core model classes stay mathematically exact.** A `log_partition_function` computes the actual log-partition; closed-form duals and autodiff gradients agree everywhere. Do NOT bake training-stability policy into core classes — no clamps, floors, or clipping controlled by construction-time state (a `precision_floor` field on `Normal` was rejected for exactly this reason).
- **Core may expose explicit, caller-invoked regularization tools** (e.g. `Normal.regularize_covariance(means, jitter, min_var)`, `Covariance.add_jitter`). The tool lives in core; the *decision to use it* does not.
- **Stability policy lives in the application layer**: goal-apps trainer configs carry `min_var`/`jitter_var`/`grad_clip` and trainers apply bounds at the right points in the loop. Within this repo, examples may define local shims (e.g. a subclass overriding `log_partition_function` with a floor) — see `examples/variational_mnist/model.py::FlooredDiagonalNormal`.

### Experimental code: examples/variational_mnist

`examples/variational_mnist` is an experimental block slated to be factored out of the library (eventually into goal-apps). Its standards are deliberately looser than core: local model subclasses with stability shims, dataclass-field defaults, extra diagnostic scripts, and CLI surface area are all acceptable there but should not migrate into `src/goal/`.

## Development Environment

This project uses `uv` for Python environment and dependency management. Use `uv run` to execute commands within the project environment — do not manually activate the venv.

### Package management strategy

This is a **library**, not an application. The dependency policy is:

- `pyproject.toml` — the source of truth for what the project needs (bare package names, no version pins)
- `uv.lock` — committed snapshot for reproducible dev environments; regenerate with `uv sync --all-extras` when deps change
- Project `.venv` — should always match the lockfile; restore with `uv sync`

**Do not** manually `uv pip install` into the project env — it creates drift from the lockfile. Instead:
- Add permanent dependencies with `uv add` (updates both `pyproject.toml` and `uv.lock`)
- Use `uvx <tool>` or `uv run --with <pkg> <cmd>` for one-off tools/packages

## Development Commands

### Testing
- Run all tests: `uv run python -m pytest tests/`
- Run specific test files:
  - `uv run python -m pytest tests/lgm.py`
  - `uv run python -m pytest tests/normal.py`
  - `uv run python -m pytest tests/graphical_mixture.py`
  - `uv run python -m pytest tests/matrix.py`
  - `uv run python -m pytest tests/clique.py` (pure Python, sub-second)

### Code Quality
- Type checking: `uvx basedpyright`

### Syncing the environment
- Sync all extras (recommended after pulling): `uv sync --all-extras`
- Sync only core deps: `uv sync`

### Examples
Examples are located in the `examples/` directory and organized by topic:
- Run example: `uv run python -m examples.multivariate.run`
- Generate plots: `uv run python -m examples.multivariate.plot`
- Available examples: boltzmann, boltzmann_lgm, boltzmann_lgm_cd, cca, chordal_boltzmann_ppc, dimensionality_reduction, hmm, hmog, kalman_filter, mfa, mixture_of_gaussians, multivariate, pendulum, poisson_mixture, population_codes, torus_poisson, univariate_analytic, univariate_differentiable, variational_mnist

### Documentation
- Build documentation: `uv run sphinx-build docs/source docs/build` or `cd docs/ && make html`
- Live documentation: https://goal-jax.readthedocs.io/

## Architecture

### Core Structure
The library is organized into three main modules under `src/goal/`:

1. **geometry/**: Core geometric abstractions
   - `algebra/`: Stateless parameter-layout descriptors that never reference `Manifold` (`matrix.py`, `clique.py`: `Cliques`, a graph given by its cliques in storage order, with the nodes, adjacency and edges they present)
   - `manifold/`: Riemannian manifolds, linear maps, embeddings, and combinators. `map.py` holds the map hierarchy. `clique.py` lays coordinates out over the cliques of a graph. **Each node has a space and each clique one coordinate block**: a `LinearCliques` is a `Cliques` and a `Manifold` with `nod_mans` (the space of each node) and `clq_mans` (the coordinate block of each clique, in storage order) as its contract, and `clq_dims`/`clq_coords`/`clq_man(clique)` derived. A coordinate block is a subspace of the tensor product of its nodes' spaces: the coordinate block of a single node is its space, and every multi-node clique is a crossing, whose coordinate block is a `CliqueMap`. Arrays are named by coordinate system (`coords`, `params`, `means`), never as blocks. It is a container of its own kind, not a `Tuple`, because its subclasses may already be tuples over coarser components (its dimension comes from the subclass and must equal the summed coordinate blocks). **Every `ExponentialFamily` is a `LinearCliques`**: by default one node whose coordinate block is the family itself (`nod_mans = clq_mans = (self,)`); `ExponentialFamilyTuple` places its elements' (`elm_mans`) graphs side by side, each offset by the node counts before it (`elm_nod_offsets`), and a harmonium composes its graph. **Node numbers are local**: each `LinearCliques` numbers its own nodes $0, \ldots, n - 1$, the numbers mean nothing outside it, and no class stores another manifold's numbers or relabels a part. A `CliqueMap(rep, cod_emb, dom_emb)` is a structured matrix from a subspace of one coordinate block to a subspace of another: `dom_emb`/`cod_emb` are embeddings *into the coordinate blocks it touches*, so `dom_man`/`cod_man` are those coordinate blocks, and `trn_man` swaps them. When a touched coordinate block is itself a map, `SubMapEmbedding(amb_man, cod_emb, dom_emb)` selects a sub-block by restricting its rows and columns; the sub-block is again a `CliqueMap` on nested subspaces (`LinearComposedEmbedding`). `CrossMap` sums several `CliqueMap`s into one map between two `LinearCliques`: each of its terms `trms` is a `CrossTerm(cod_clq, dom_clq, clq_map)`, a clique map pinned to a clique on each side (each in its own side's numbering), whose coordinate blocks it reads and writes through `CliqueEmbedding`. A `RecursiveLinearCliques[Root, Deep]` is a `LinearCliques` and the `Triple[Root, CrossMap[Root, Deep], Deep]` of its partitions `[root | cross | deep]`, and its **graph is composed from its parts**. A concrete model declares `rot_man`, `dep_man` and `crs_trms`: one `CrossTerm` per crossing, a clique of the root partition and a clique of the deep partition (each in its partition's numbering) with its `CliqueMap`, whose embeddings have the touched coordinate blocks as ambients (`rot_man.clq_man(near)`, `dep_man.clq_man(far)`). That containment is the condition that keeps the likelihood and the posterior in their families; it holds by declaration and is checked over every shipped model in `tests/linear_clique.py`, not at run time. Everything else is derived: `crs_clqs` and `crs_maps` read the terms; the composed numbering puts the root's nodes first and the deep partition's after them, offset by the root's node count; `cliques`, `clq_mans` and `nod_mans` are the root's, then the crossings (for cliques and coordinate blocks), then the deep partition's, which is exactly the triple's storage order; and `crs_man` is the `CrossMap` with the terms `crs_trms`. Submodels are used exactly as given. MFA's $(x,y,k)$ maps from a `SubMapEmbedding` of the mixture's $(y,k)$ coordinate block (the location of $y$, all of $k$) to the location of $x$. `CompleteMixture` declares one crossing per clique of its observable, each over the whole touched coordinate block, and `Mixture.cmp_int_map` reads the result as one matrix. `Harmonium.int_man` is `crs_man`; code that needs one crossing's map reads it from `crs_maps`, e.g. `(xz_map,) = self.crs_maps`. Features deliberately not implemented (partial biases, graph validation) are listed in `CLIQUE-TODO.md`
   - `exponential_family/`: Exponential family abstractions and harmoniums. `graphical.py` holds the composites: a `GraphicalHarmonium` is a deep model (`pst_man`) with observable harmoniums, each paired with the clique of the deep model its posterior sits on (`obs_hrms_att_clqs`); the observable, crossings and their maps are read off the observable harmoniums. It assumes no conjugation: its observable is a `GenerativeTuple`, and a plain harmonium becomes one as an `AttachedHarmonium`. `DifferentiableGraphical` derives conjugation from them: each observable harmonium's conjugation parameters placed on its clique of the prior deep model by a `SubCliquesEmbedding` and summed, offsets summed. Its observable is always a `DifferentiableTuple` (an `AnalyticTuple` for `AnalyticGraphical`) of the observable harmoniums' observables, also when there is only one; code that needs Normal methods on HMoG's observable reads `lwr_hrm.obs_man`. HMoG is one linear Gaussian model attached to node $0$ of a mixture; CCA is two attached to the same latent node. `AnalyticGraphical` converts mean parameters to the likelihood harmonium by harmonium, so any number of analytic attached harmoniums is supported (`AnalyticHMoG`, `AnalyticCanonicalCorrelationAnalysis`). A graphical harmonium whose observable harmoniums are not conjugated is fit variationally by a `VariationalConjugated` (`variational.py`), a fork of the graphical hierarchy that wraps it (`gen_hrm`) and approximates `DifferentiableGraphical`'s operations; it is not an exponential family. Its parameters are `[hrm | cnj | dep]`: the graphical harmonium's joint natural parameters, this level's conjugation coordinates, and the deep model's. `conjugation_parameters(lkl_params, cnj_params)` gives $\rho$ from the likelihood alone (tied $\rho$), the prior is $\theta_Z + \rho$ (`prior`), and the recognition model is the deep model at `gen_hrm`'s posterior (`posterior_at`); this is a directed VAE in other coordinates ($\theta^{dir}_Y = \theta_Y + \rho_Y$). The deep model is exact in `ExactPriorVariational` (a `Differentiable` prior family) or another fork in `NestedPriorVariational` (`dep_vrt`, whose graphical harmonium is the prior family and whose conjugation coordinates are stored in `dep`), so the recursion goes through `pst_man`. Residual slopes are read off the parameters actually used, so overriding `prior` or `posterior_at` (where an example's stability shim goes) keeps the ELBO decomposition exact; residual-variance penalties are returned per level. Stability policy (positive-definite clamps, clipping) stays out of it. The graph layout itself lives in `manifold/clique.py`, with three embeddings between layouts: `CliqueEmbedding` (one clique's coordinate block), `SubCliquesEmbedding` (a smaller layout on some cliques of a larger one, coordinate block for coordinate block) and `RootEmbedding` (same graph, restricted family on the root partition)

2. **models/**: Concrete statistical models
   - `base/`: Fundamental distributions (Normal, Categorical, Poisson, Von Mises)
   - `harmonium/`: Bipartite models (Mixtures, Linear Gaussian Models)
   - `graphical/`: Complex graphical models (Hierarchical MoG)
   - `dynamical/`: State-space and Markov-process models (Kalman filter, HMM, MLP-based hybrid filter, homogeneous Gaussian Markov chain)

### Key Abstractions

**Manifold Hierarchy:**
- `Manifold`: Base class for differentiable manifolds

**Exponential Family Hierarchy:**
- `ExponentialFamily`: Base class extending Manifold
- `Analytic`: Closed-form computations available
- `Differentiable`: Gradient-based optimization support

**Matrix Representations:**
- `MatrixRep`: Base for matrix-valued manifolds
- Specialized forms: `Symmetric`, `PositiveDefinite`, `Diagonal`, etc.
- Automatic constraint handling and optimization

### Key Design Patterns
1. **Analytic vs Differentiable**: Models can provide exact computations or gradient-based approximations
2. **Harmoniums**: Conjugate relationship modeling between latent and observed variables
   - `SymmetricConjugated`: Posterior and prior use the same manifold (`pst_man == prr_man`)
   - `DifferentiableConjugated[Obs, Pst, Prr]`: Supports asymmetric cases where posterior embeds into prior (`pst_man ⊂ prr_man` via `pst_prr_emb`)
3. **Maps**: `Map[C, D]` is the generic ABC for parameterized functions between manifolds; concrete `LinearMap[C, D]` leaves are `MatrixMap` and `SquareMap` (matrix-rep backed, acting on the full domain and codomain), `CliqueMap` (a matrix-rep backed map between subspaces of two coordinate blocks, given by an embedding into each), and `CrossMap` (several `CliqueMap`s, each on a labelled clique, summed into one map between two `LinearCliques`); `AffineMap` adds a bias, and `MultilayerPerceptron[C, D]` is the nonlinear leaf.
4. **Transitions and LatentProcess**: `Transition[L]` is a predict map on belief natural parameters used by the BPTT-friendly filter scan. `AnalyticTransition[L]` wraps a `SymmetricConjugated[L, L]` kernel to enable smoothing and exact EM. `LatentProcess[O, L]` composes (prior, conjugated emission, transition) into a Triple; `AnalyticLatentProcess[O, L]` adds joint sampling, smoothing, and EM.
5. **Combinators**: Composable building blocks for complex models (Pair, Triple, Replicated; for exponential families `LocationShape`, the heterogeneous `ExponentialFamilyTuple` chain, and the homogeneous `ExponentialFamilyProduct` chain)
6. **Embeddings**: Flexible transformations between manifolds (e.g., `NormalCovarianceEmbedding` embeds `DiagonalNormal` into `FullNormal`)

### Key Model Classes
- **Normal distributions**: `Normal[Rep]` parameterized by covariance representation
- **Linear Gaussian Models**: `NormalLGM[ObsRep, PstRep]`, `FactorAnalysis`, `PrincipalComponentAnalysis`
- **Mixtures**: `Mixture[Observable]`, `CompleteMixture[Observable]`, `AnalyticMixture[Observable]`
- **Canonical correlation analysis**: `CanonicalCorrelationAnalysis` (gradient-based), `AnalyticCanonicalCorrelationAnalysis` (exact EM)
- **Graphical models**: `CompleteMixtureOfConjugated[Obs, PstLatent, PrrLatent]` for mixture of factor analyzers
- **Dynamical models**: `KalmanFilter`, `HiddenMarkovModel`

## Typing Strategy

This codebase uses Python 3.12+ modern generic syntax with a pragmatic approach to type safety:

### Philosophy
- **Pragmatic over purist**: Accept type system limitations rather than fight them when the code is functionally correct

### Abstract versus concrete

Whether a class is an ABC with abstract properties or a concrete class with fields is decided by **is-a versus has-a**:

- **Abstract, stateless** when the class is a *role a model becomes*. A model subclasses it and derives the properties from its own fields: `Manifold`, `Tuple`/`Pair`/`Triple`/`Quadruple`, `Replicated`, `ExponentialFamilyProduct` and its chain, `Map`/`LinearMap`, `Embedding`/`LinearEmbedding`/`TupleEmbedding`, `LinearCliques`/`RecursiveLinearCliques`, `ExponentialFamily` and its chain, `Harmonium` and its conjugacy variants.
- **Concrete, with fields** when the class is a *value a model builds*. It occurs with multiplicity inside a model and its fields are already primitive --- there is nothing more basic to derive them from: `MatrixMap`, `SquareMap`, `AffineMap`, `MultilayerPerceptron`, `CliqueMap`, `CrossMap`, the `ExponentialFamilyTuple` chain (its field is `elm_mans`), `CliqueEmbedding`, `SubCliquesEmbedding`, `RootEmbedding`, `IdentityEmbedding`, `ComposedEmbedding`.

**Embeddings are stored at the level of their ambient.** A class may hold an embedding into itself or into one of its parts (for a map, its domain or codomain), never into a manifold that contains it: if $M \supset N \supset L$, then $N$ may store $L \to N$ but not $L \to M$. Locating a component in an enclosing manifold is the enclosing class's job --- `CrossMap` locates its terms in its own codomain and domain, by those manifolds' own cliques, and knows nothing of the model around them. The same holds for node numbers: a manifold numbers only its own nodes, and an enclosing class translates (`RecursiveLinearCliques` offsets its deep partition's).

A model is exactly one `RecursiveLinearCliques`, so it can *be* one; it holds many `CliqueMap`s, so it builds them. `MatrixMap` is concrete for the same reason `LinearMap` is abstract.

**Abstract classes may carry fields**, but only the fields *every* subclass shares, and only when the subclasses differ solely in behavior: `HarmoniumEmbedding(hrm_man)`, `LGM(obs_dim, obs_rep)`, `CompleteMixtureOfHarmoniums(n_categories, bas_hrm)`. Hoist a field to the lowest parent all its holders share --- `ChainBoltzmann` inherits `junction_tree` from `ChordalBoltzmann`, while `n_neurons` stays on `DiagonalBoltzmann` and `FullBoltzmann` separately because those two share no parent below `Boltzmann`.

**A field backing an inherited abstract property is private**, named for the property it serves: `_dom_man`, `_cod_man`, `_amb_man`, `_sub_man`. A field that is not shadowing a property stays public (`rep`, `map_man`, `hidden_dims`, `junction_tree`).

**Derived quantities are properties, not cached fields.** `field(init=False)` written through `object.__setattr__` is not used; `__post_init__` is for validation only. Recomputation is cheap because manifolds are stateless and the work happens at trace time.

### Dataclass field defaults
Dataclass fields generally do **not** have default values. Modeling and architectural choices (e.g. `Binomial.n_trials`, `MultilayerPerceptron.hidden_dims`, `MultilayerPerceptron.activation`) must be made explicitly at every call site — defaults silently encode design decisions and tend to mask the degenerate case that callers shouldn't actually want (e.g. `Binomial(n_trials=1) == Bernoulli`). The narrow exception is **numerical/implementation details that callers shouldn't need to reason about** (e.g. `CoMPoisson.window_size = 200`, a truncation bound for an infinite series). When in doubt, omit the default.

### Target-first order
Maps and embeddings list their manifolds target first, the way a matrix lists rows before columns: `Map[Codomain, Domain]`, `LinearMap[C, D]`, `CrossMap[Root, Deep]`, `Embedding[Ambient, Sub]`, `ComposedEmbedding[Ambient, Mid, Sub]`. A harmonium's interaction $\Theta_{XZ}$ is `CrossMap[Observable, Posterior]`. Dataclass fields and constructor arguments follow the same order (`MatrixMap(rep, cod_man, dom_man)`, `NormalCovarianceEmbedding(amb_man, sub_man)`, `ComposedEmbedding(mid_emb, sub_emb)`).

### Type Aliases
The codebase defines convenient type aliases for common parameterized types:
- `FullNormal = Normal[PositiveDefinite]` - Full covariance normal
- `DiagonalNormal = Normal[Diagonal]` - Diagonal covariance normal
- `IsotropicNormal = Normal[Scale]` - Isotropic (scalar variance) normal
- `StandardNormal = Normal[Identity]` - Standard normal (identity covariance)

### Current State
- **Core codebase**: Zero type errors, fully functional and well-typed
- **External library integration**: JAX, optax, and scipy types appropriately suppressed where incomplete
- **Complex generics**: Some inference limitations remain in deep hierarchical models (expected with current Python typing)

## Documentation Strategy

### Source of truth
Python docstrings are the single source of truth. Sphinx `.rst` files should be thin scaffolding (title, `automodule`, inheritance diagrams, section headings) --- no duplicated prose. `index.rst` files get a brief orientation paragraph and module listing, nothing more.

### Docstring structure
Lead with what the class or function *does* in concrete terms (what arrays it operates on, what it returns, when you'd use it). Then, when the underlying mathematics adds clarity, introduce it with a **"Mathematically,"** marker. This signals "here comes the formal version" --- readers who don't need the math can stop. Keep the math precise but brief.

Place mathematical definitions at the highest appropriate level in the class hierarchy. Subclasses should not repeat them --- they inherit the concept and just state their specialization.

### Coordinate system naming
Variable names encode the coordinate system that a flat array lives in. This convention replaces what a richer type system would enforce:

- `params` --- natural parameters (the full vector for a model)
- `*_params` (e.g. `obs_params`, `lat_params`, `int_params`) --- slices of a natural parameter vector
- `means` --- mean parameters
- `coords` --- generic, coordinate-system-agnostic (used in the manifold layer)

Docstrings should naturally reiterate which coordinate system their inputs live in --- e.g. "at the given natural parameters", "convert mean parameters to natural parameters" --- so that the coordinate system is always clear without needing explicit Args blocks.

### Args/Returns blocks
Only document a parameter when its name and type aren't enough to use it correctly. Shape conventions, non-obvious defaults, and semantic constraints the type system can't express earn Args entries. Self-evident parameters (e.g., `coords: Array` on a method called `split_coords`) do not.

### Guards and validation
Only add runtime checks for errors that might slip through silently. If the operation would crash with a clear error anyway (e.g., wrong-shaped array in `reshape`), don't add a redundant guard.

### Class body organization
Within each class, order members as follows:

1. **Fields** --- dataclass fields that define the class
2. **Contract** --- abstract properties and methods that subclasses must implement
3. **Overrides** --- implementations of parent abstract properties and methods
4. **Methods** --- new concrete functionality specific to this class

Use comment headers (`# Fields`, `# Contract`, `# Overrides`, `# Methods`) to separate sections. Omit headers when the class is small enough that the structure is obvious. Within each section, properties naturally precede methods.

### RST file convention

RST files live under `docs/source/` and mirror the Python package hierarchy under `src/goal/`:

- **1:1 mapping**: each source module gets one RST file; RST paths mirror Python package paths (e.g., `models/base/poisson.py` → `docs/source/models/base/poisson.rst`)
- **RST filenames must match Python module filenames** (e.g., `population_codes.rst` for `population_codes.py`)
- **Leaf RST template**: title, `automodule` (docstring only, `:noindex: :no-members:`), optional `inheritance-diagram`, `autoclass` sections, optional factory functions
- **Index RST template**: title, one-line description, `toctree` listing child modules
- **Each class is documented in the RST of its defining module** --- no cross-module duplication
- **Internal utilities** (e.g., `manifold/util.py`) may be skipped

### Style
- No Unicode math in docstrings or comments --- use LaTeX notation throughout (`\\theta`, `\\mathcal M`, etc.)
- Use `\\\\` (doubled backslash) in docstrings for LaTeX commands (Python string escaping); single `\\` in comments
- Matplotlib labels use raw strings with `$...$` for LaTeX rendering

## Test Design

### File naming
Test files drop the `test_` prefix (pytest is configured with `python_files = ["*.py"]` in `pyproject.toml`). Each file is named after the source module it tests:

| Test file | Source module(s) |
|---|---|
| `matrix.py` | `geometry/algebra/matrix.py` |
| `clique.py` | `geometry/algebra/clique.py` (`Cliques`: the nodes, `graph` and `edges` its cliques present; defines the concrete `Graph` it tests with) |
| `linear_clique.py` | `geometry/manifold/clique.py` (`LinearCliques`/`RecursiveLinearCliques` partitions, `ExponentialFamilyTuple` composition, the composed graph and its storage order, `CliqueEmbedding`; every shipped model's crossings lie in the coordinate blocks they touch, and a single node's coordinate block is its space; `CliqueMap` vs `MatrixMap`; `SubMapEmbedding`) |
| `graphical.py` | `geometry/exponential_family/graphical.py` (crossings and conjugation of a composite read off its observable harmoniums, checked against HMoG; `SubCliquesEmbedding` and `RootEmbedding` on HMoG) |
| `graphical_variational.py` | Nested `VariationalConjugated` in `geometry/exponential_family/variational.py` (all-exact Gaussian chains at depths 2 and 3, an exact harmonium as prior, and a CCA-like fork as `gen_hrm`: ELBO = closed-form log-marginal, and prior, log-partition and $c(x)$ equal `DifferentiableGraphical`'s; zero-shift conditioning gives the prior; ELBO and log-density decompositions against brute force for Boltzmann/Gaussian levels at depths 2--4, also with `prior`/`posterior_at` overridden; exact levels have vanishing residuals; estimator gradients vs enumeration + quadrature at depths 2 and 3, also from one or two samples; recognition and prior residual variances with their full gradients, incl. the sampling-distribution terms) |
| `interaction.py` | `CrossMap` in `geometry/manifold/clique.py` (its `CliqueMap`s vs every live interaction shape, all three operations, incl. MFA's three-node crossing in mean coordinates) |
| `map.py` | `geometry/manifold/map.py` (LinearMap rename regression + MultilayerPerceptron) |
| `normal.py` | `models/base/gaussian/normal.py` |
| `boltzmann.py` | `models/base/gaussian/boltzmann.py` |
| `categorical.py` | `models/base/categorical.py` (Categorical, Bernoulli, Bernoullis) |
| `binomial.py` | `models/base/binomial.py` (Binomial, Binomials) |
| `poisson.py` | `models/base/poisson.py` |
| `von_mises.py` | `models/base/von_mises.py` |
| `dirichlet.py` | `models/base/dirichlet.py` |
| `cca.py` | `models/harmonium/cca.py` (`CanonicalCorrelationAnalysis`: two-node observable, fork conjugation as a sum; `AnalyticCanonicalCorrelationAnalysis`: branch-by-branch mean-to-natural conversion, EM) |
| `lgm.py` | `models/harmonium/lgm.py` |
| `population_codes.py` | `models/harmonium/population_codes.py` |
| `graphical_mixture.py` | `models/graphical/mixture.py` |
| `variational.py` | `geometry/exponential_family/variational.py` and `models/graphical/variational.py` (estimator values + gradients vs quadrature ground truth; VariationalHierarchicalMixture conjugation cascade) |
| `hmog.py` | `models/graphical/hmog.py` (AnalyticHMoG, DifferentiableHMoG) |
| `dynamical.py` | `geometry/exponential_family/dynamical.py` and `models/dynamical/` (`AnalyticTransition` over both kernel types, `KalmanFilter`, `HiddenMarkovModel`, and `MultilayerPerceptron[L, L]` as a transition map) |
| `whitening.py` | Cross-cutting: whitening on FA, MFA, HMoG |

### Structure within files
- Each file starts with both JAX config lines (`jax_platform_name=cpu`, `jax_enable_x64=True`) and a module docstring stating what it tests
- One test class per model/concept; use `@pytest.mark.parametrize` for variation rather than class-level fixtures
- Construct models and keys inline (`jax.random.PRNGKey(...)`) rather than through shared fixtures

### Standard test categories for exponential families
- **Dimensions**: `dim`, `data_dim` match expectations
- **Sufficient statistic**: correct shape and known values
- **Density normalization**: pmf sums to 1 (discrete) or density integrates to 1 (continuous)
- **Parameter conversions**: `to_natural(to_mean(params)) == params` round-trip
- **Log partition**: matches known closed form (e.g., softplus for Bernoulli)
- **Sampling**: correct shape, domain constraints, empirical mean close to theoretical

### Tolerances
- Analytic comparisons: `rtol=1e-5, atol=1e-7` (or `1e-4/1e-6` for less precise models)
- Sampling comparisons use 50k samples and compare `average_sufficient_statistic(samples)` against `to_mean(params)`:
  - Simple EFs (Bernoulli, Categorical, Poisson, VonMises): `atol=0.01–0.02`
  - Normal (includes second moments with higher variance): data-space mean `atol=0.03`, full sufficient stats `atol=0.1`

## Examples Design

Each example lives in `examples/<name>/` with a standard structure:

### File layout
- `run.py` — computation: generates data, fits models, saves `analysis.json` via `paths.save_analysis()`
- `plot.py` — visualization: loads `analysis.json`, produces `plot.png` via `paths.save_plot()`
- `types.py` — typed dataclass(es) defining the JSON schema exchanged between run and plot

### run.py conventions
- **Entry point**: `main()` with shape `jax_cli()` → `paths = example_paths(__file__)` → `key` → compute → `paths.save_analysis(results)`
- **No module-level state**: all models, hyperparameters, and configuration live inside `main()` as local variables
- **JAX-idiomatic loops**: use `jax.lax.scan` for training loops, not Python for-loops. For loops needing periodic expensive metrics, use a chunked pattern (outer Python loop, inner `jax.lax.scan`)
- **No direct file I/O**: always use `paths.save_analysis()` / `paths.load_analysis()`, never raw `json.dump`

### plot.py conventions
- **Style**: call `apply_style(paths)` once; do not override `fontsize`, `grid`, or `tight_layout` — `default.mplstyle` handles these
- **Colors**: use `colors["ground_truth"]`, `model_color(i)`, `metric_color(name)` from `shared.py`
- **Figure sizes**: use `figure_size("small"|"medium"|"large"|"wide"|"tall")` from `shared.py`
- **Scatter helpers**: use `scatter_samples()` / `scatter_points()` for consistent styling
- **GridSpec**: pass `figure=fig` when using `gridspec.GridSpec` (required for `constrained_layout`)

### Shared infrastructure (`examples/shared.py`)
- `ExamplePaths` — manages result/plot paths and serialization
- `example_paths(module_path)` — factory from `__file__`
- `jax_cli()` — parses `--gpu` and `--no-jit` flags
- Grid, bounds, density contour, and training history plot helpers

## Dependencies
- **JAX**: Core computation backend for automatic differentiation
- **Optax**: Optimization algorithms compatible with JAX
- **pytest**: Testing framework
- **ruff**: Fast Python linter/formatter
- **basedpyright**: Static type checker
