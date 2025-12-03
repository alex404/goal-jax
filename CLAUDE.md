# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

goal-jax is a JAX implementation of information geometric algorithms and exponential families. The library provides geometric optimization tools for statistical models, focusing on manifold-based optimization and exponential family distributions.

## Development Environment

This project uses `uv` for Python environment and dependency management. Always activate the virtual environment:

```bash
source .venv/bin/activate
```

## Development Commands

### Testing
- Run all tests: `python -m pytest tests/`
- Run specific test files:
  - `python -m pytest tests/linear_gaussian_model.py`
  - `python -m pytest tests/normal.py`
  - `python -m pytest tests/univariate.py`
  - `python -m pytest tests/matrix.py`

### Code Quality
- Run linter: `ruff check src/`
- Run formatter: `ruff format src/`
- Type checking: `basedpyright` (configured via pyproject.toml, available via uv)

### Installation and Dependencies
- Install for development: `pip install -e ".[test]"` or `uv pip install -e ".[test]"`
- Install for examples: `pip install -e ".[examples]"` or `uv pip install -e ".[examples]"`
- Install for documentation: `pip install -e ".[docs]"` or `uv pip install -e ".[docs]"`
- GPU support: `pip install -e ".[gpu]"`

### Examples
Examples are located in the `examples/` directory and organized by topic:
- Run example: `python -m examples.bivariate_normal.run`
- Generate plots: `python -m examples.bivariate_normal.plot`
- Available examples: bivariate_normal, dimensionality_reduction, hmog, mixture_of_gaussians, univariate_analytic, univariate_differentiable, poisson_mixture

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
3. **Combinators**: Composable building blocks for complex models (Product, Pair, Replicated)
4. **Embeddings**: Flexible transformations between manifolds

## Typing Strategy

This codebase uses Python 3.13+ modern generic syntax with a pragmatic approach to type safety:

### Philosophy
- **Pragmatic over purist**: Accept type system limitations rather than fight them when the code is functionally correct

### Suppression Strategy
```toml
**Global configuration**: Backend/external library type issues handled via pyproject.toml:
[tool.pyright]
reportUnknownParameterType = "none"         # JAX integration challenges
reportUnknownMemberType = "none"           # External library type inference
reportUnknownArgumentType = "none"         # Complex generic type inference
reportUnknownVariableType = "none"  # JAX, scipy, etc. have incomplete types
```

### Current State
- **Core codebase**: Zero type errors, fully functional and well-typed
- **External library integration**: JAX, optax, and scipy types appropriately suppressed where incomplete
- **Complex generics**: Some inference limitations remain in deep hierarchical models (expected with current Python typing)

## Testing Notes
- Test files correspond to major components: matrix representations, normal distributions, univariate models, linear Gaussian models

## Dependencies
- **JAX**: Core computation backend for automatic differentiation
- **Optax**: Optimization algorithms compatible with JAX
- **pytest**: Testing framework
- **ruff**: Fast Python linter/formatter
- **basedpyright**: Static type checker
