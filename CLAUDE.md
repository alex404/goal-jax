# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

goal-jax is a JAX implementation of information geometric algorithms and exponential families. The library provides geometric optimization tools for statistical models, focusing on manifold-based optimization and exponential family distributions.

## Development Environment

This project uses `uv` for Python environment and dependency management. Always activate the virtual environment:

```bash
source .venv/bin/activate
```

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
- Run all tests: `python -m pytest tests/`
- Run specific test files:
  - `python -m pytest tests/lgm.py`
  - `python -m pytest tests/normal.py`
  - `python -m pytest tests/graphical_mixture.py`
  - `python -m pytest tests/matrix.py`

### Code Quality
- Type checking: `uvx basedpyright` (or `basedpyright` with venv active)

### Syncing the environment
- Sync all extras (recommended after pulling): `uv sync --all-extras`
- Sync only core deps: `uv sync`

### Examples
Examples are located in the `examples/` directory and organized by topic:
- Run example: `python -m examples.bivariate_normal.run`
- Generate plots: `python -m examples.bivariate_normal.plot`
- Available examples: bivariate_normal, boltzmann, boltzmann_lgm, boltzmann_lgm_cd, dimensionality_reduction, hmog, mfa, mixture_of_gaussians, poisson_mixture, population_codes, torus_poisson, univariate_analytic, univariate_differentiable, variational_mnist

### Documentation
- Build documentation: `sphinx-build docs/source docs/build` or `cd docs/ && make html`
- Documentation is also configured for MkDocs: `mkdocs serve` (uses mkdocs.yml)
- Live documentation: https://goal-jax.readthedocs.io/

## Architecture

### Core Structure
The library is organized into three main modules under `src/goal/`:

1. **geometry/**: Core geometric abstractions
   - `manifold/`: Riemannian manifolds, matrix representations, linear maps, embeddings
   - `exponential_family/`: Exponential family abstractions, harmoniums, hierarchical models

2. **models/**: Concrete statistical models
   - `base/`: Fundamental distributions (Normal, Categorical, Poisson, Von Mises)
   - `harmonium/`: Bipartite models (Mixtures, Linear Gaussian Models)
   - `graphical/`: Complex graphical models (Hierarchical MoG)

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
3. **Combinators**: Composable building blocks for complex models (Product, Pair, Replicated)
4. **Embeddings**: Flexible transformations between manifolds (e.g., `NormalCovarianceEmbedding` embeds `DiagonalNormal` into `FullNormal`)

### Key Model Classes
- **Normal distributions**: `Normal[Rep]` parameterized by covariance representation
- **Linear Gaussian Models**: `NormalLGM[ObsRep, PstRep]`, `FactorAnalysis`, `PrincipalComponentAnalysis`
- **Mixtures**: `Mixture[Observable]`, `CompleteMixture[Observable]`, `AnalyticMixture[Observable]`
- **Graphical models**: `CompleteMixtureOfConjugated[Obs, PstLatent, PrrLatent]` for mixture of factor analyzers

## Typing Strategy

This codebase uses Python 3.12+ modern generic syntax with a pragmatic approach to type safety:

### Philosophy
- **Pragmatic over purist**: Accept type system limitations rather than fight them when the code is functionally correct

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

### Style
- No Unicode math in docstrings or comments --- use LaTeX notation throughout (`\\theta`, `\\mathcal M`, etc.)
- Use `\\\\` (doubled backslash) in docstrings for LaTeX commands (Python string escaping); single `\\` in comments
- Matplotlib labels use raw strings with `$...$` for LaTeX rendering

## Test Design

### File naming
Test files drop the `test_` prefix (pytest is configured with `python_files = ["*.py"]` in `pyproject.toml`). Each file is named after the source module it tests:

| Test file | Source module(s) |
|---|---|
| `matrix.py` | `geometry/manifold/matrix.py` |
| `normal.py` | `models/base/gaussian/normal.py` |
| `boltzmann.py` | `models/base/gaussian/boltzmann.py` |
| `categorical.py` | `models/base/categorical.py` (Categorical, Bernoulli, Bernoullis) |
| `binomial.py` | `models/base/binomial.py` (Binomial, Binomials) |
| `poisson.py` | `models/base/poisson.py` |
| `von_mises.py` | `models/base/von_mises.py` |
| `lgm.py` | `models/harmonium/lgm.py` |
| `population_codes.py` | `models/harmonium/population_codes.py` |
| `graphical_mixture.py` | `models/graphical/mixture.py` |
| `hmog.py` | `models/graphical/hmog.py` (AnalyticHMoG, DifferentiableHMoG) |
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

## Dependencies
- **JAX**: Core computation backend for automatic differentiation
- **Optax**: Optimization algorithms compatible with JAX
- **pytest**: Testing framework
- **ruff**: Fast Python linter/formatter
- **basedpyright**: Static type checker
