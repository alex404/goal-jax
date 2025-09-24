# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

goal-jax is a JAX implementation of information geometric algorithms and exponential families. The library provides geometric optimization tools for statistical models, focusing on manifold-based optimization and exponential family distributions.

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
- Type checking: `pyright` (configured via pyproject.toml)

### Installation and Dependencies
- Install for development: `pip install -e ".[test]"`
- Install for examples: `pip install -e ".[examples]"`
- Install for documentation: `pip install -e ".[docs]"`
- GPU support: `pip install -e ".[gpu]"`

### Examples
Examples are located in the `examples/` directory and organized by topic:
- Run example: `python -m examples.bivariate_normal.run`
- Generate plots: `python -m examples.bivariate_normal.plot`
- Available examples: bivariate_normal, dimensionality_reduction, hmog, mixture_of_gaussians, univariate_analytic, univariate_differentiable, poisson_mixture

### Documentation
- Build documentation: `sphinx-build docs/source docs/build`
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
   - `graphical/`: Complex models (Mixtures, Linear Gaussian Models, Hierarchical MoG)

### Key Abstractions

**Manifold Hierarchy:**
- `Manifold`: Base class for differentiable manifolds
- `Point`: Type wrapper for points on manifolds
- `Coordinates`: Type system for different coordinate systems
- `Dual`: Automatic dual coordinate system derivation

**Exponential Family Hierarchy:**
- `ExponentialFamily`: Base class extending Manifold
- `Analytic`: Closed-form computations available
- `Differentiable`: Gradient-based optimization support
- Natural vs Mean parameterizations with automatic conversion

**Matrix Representations:**
- `MatrixRep`: Base for matrix-valued manifolds
- Specialized forms: `Symmetric`, `PositiveDefinite`, `Diagonal`, etc.
- Automatic constraint handling and optimization

### Coordinate Systems
The library uses a sophisticated type system to track coordinate transformations:
- `Natural`: Natural parameters of exponential families
- `Mean = Dual[Natural]`: Mean parameters (dual to natural)
- Automatic conversions between coordinate systems

### Key Design Patterns
1. **Analytic vs Differentiable**: Models can provide exact computations or gradient-based approximations
2. **Harmoniums**: Conjugate relationship modeling between latent and observed variables
3. **Combinators**: Composable building blocks for complex models (Product, Pair, Replicated)
4. **Embeddings**: Flexible transformations between manifolds

## Typing Strategy

This codebase uses Python 3.13+ modern generic syntax with a pragmatic approach to type safety:

### Philosophy
- **Pragmatic over purist**: Accept type system limitations rather than fight them when the code is functionally correct
- **Modern syntax preferred**: Use `class Point[C, M]` syntax over `typing.Generic` despite some rough edges
- **Document compromises**: Type ignores serve as breadcrumbs documenting known type system limitations

### Key Patterns
- **Coordinate type parameters**: `Point[C: Coordinates, M: Manifold]` tracks transformations between coordinate systems
- **Invariant type parameters**: `Point` uses invariant generics, so `Point[C, Subclass] ≠ Point[C, Superclass]`
- **Method override variance**: Suppressed via `reportIncompatibleMethodOverride = "none"` due to type system limitations with inheritance

### Type Checker Configuration
```toml
[tool.pyright]
reportIncompatibleMethodOverride = "none"  # Python typing limitations with method variance
reportUnknownParameterType = "none"         # JAX integration challenges
reportUnknownMemberType = "none"           # External library type inference
reportUnknownArgumentType = "none"         # Complex generic type inference
```

### Suppression Strategy
**Global configuration**: Backend/external library type issues handled via pyproject.toml:
```toml
reportUnknownVariableType = "none"  # JAX, scipy, etc. have incomplete types
```

**Import-level suppressions**: Module-wide external library issues:
```python
from optax import GradientTransformation  # pyright: ignore[reportMissingTypeStubs]
```

**Inline suppressions**: Local design/Python typing system limitations:
```python
def point[C: Coordinates](...) -> Point[C, Self]:  # pyright: ignore[reportInvalidTypeVarUse]
return components[self.tup_idx]  # type: ignore[return-value]
```

**Principle**: Suppress at the most appropriate scope - global config for systematic backend issues, import-level for module-wide issues, inline for specific local limitations.

### Current State
- **Core codebase**: Zero type errors, fully functional and well-typed
- **External library integration**: JAX, optax, and scipy types appropriately suppressed where incomplete
- **Complex generics**: Some inference limitations remain in deep hierarchical models (expected with current Python typing)

### Type System Maintenance
- New JAX/external library usage should follow the suppression patterns established
- Complex generic inference issues can often be resolved with explicit type annotations
- The type checker configuration balances strictness with practicality for mathematical computing

## Testing Notes
- Tests are minimal and focused on core functionality
- Test files correspond to major components: matrix representations, normal distributions, univariate models, linear Gaussian models
- No comprehensive test suite - tests serve as integration checks