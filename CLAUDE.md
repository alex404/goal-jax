# CLAUDE.md

Guidance for Claude Code in this repository: design, conventions and expectations. Detailed documentation lives in the docstrings.

## General Points

1. Ask, don't assume: never make silent assumptions about intent, architecture or requirements. When running unattended, pick the most reasonable interpretation, proceed, and record the assumption. Where it helps, run a small, local, low-risk experiment and bring the hypothesis and results to discuss.

2. Implement the simplest solution for simple problems, better solutions for harder problems. Do not over-engineer or add flexibility that isn't needed yet.

3. Don't touch unrelated code, but do surface bad code, design smells, and better long-term alternatives to a tactical change, as separate issues.

## Project Overview

goal-jax is a JAX library of exponential families, harmoniums and information-geometric algorithms. `../goal-apps` is the application layer built on it (CLI training, Hydra configs, W&B).

### Library boundaries
- **Core model classes stay mathematically exact.** A `log_partition_function` computes the actual log-partition. No training-stability policy in core classes: no clamps, floors or clipping set at construction (a `precision_floor` field on `Normal` was rejected for this reason).
- **Core may expose caller-invoked regularization tools** (e.g. `Normal.regularize_covariance`, `Covariance.add_jitter`); the decision to use them is not core's.
- **Stability policy lives in the application layer** (goal-apps trainer configs). Examples in this repo may define local overrides that keep parameters valid.

## Development

`uv` manages the environment; run everything through `uv run`, never activate the venv or `uv pip install` into it. `pyproject.toml` lists bare dependencies (no pins), `uv.lock` is committed; add dependencies with `uv add`, use `uvx <tool>` or `uv run --with <pkg>` for one-off tools.

| Task | Command |
|---|---|
| Tests | `uv run python -m pytest tests/` (or one file, e.g. `tests/variational.py`) |
| Lint, format | `uvx ruff check`, `uvx ruff format` |
| Types | `uvx basedpyright` (zero errors expected) |
| Docs | `uv run sphinx-build docs/source docs/build` |
| Example | `uv run python -m examples.<name>.run`, then `.plot` |
| Sync | `uv sync --all-extras` |

CI (`.github/workflows/`) runs the tests, basedpyright, and every example with its plot on a CPU runner.

## Architecture

Module and class docstrings are the detailed reference; this section is a map and the invariants that hold everywhere.

### Layout of `src/goal/`
- `geometry/algebra/`: stateless parameter layouts that never reference `Manifold`: matrix representations (`matrix.py`), graphs given by their cliques (`clique.py`), junction trees with exact sum-product over binary variables (`junction_tree.py`).
- `geometry/manifold/`: manifolds, maps (`Map`, `LinearMap`, `MatrixMap`, `CliqueMap`, `CrossMap`, `AffineMap`, `MultilayerPerceptron`), embeddings, combinators (`Pair`, `Triple`, `ManifoldTuple`, `ManifoldProduct`), and the clique layout (`clique.py`: `LinearCliques`, `RecursiveLinearCliques`).
- `geometry/exponential_family/`: the capability chain `ExponentialFamily` → `Generative` → `Differentiable` → `Analytic`; harmoniums and their conjugacy variants (`harmonium.py`); graphical harmoniums (`graphical.py`); variational levels (`variational.py`); latent processes (`dynamical.py`).
- `models/`: concrete models: `base/` (distributions), `harmonium/` (mixtures, linear Gaussian models, CCA, population codes), `graphical/` (HMoG, mixtures of harmoniums), `dynamical/` (Kalman filter, HMM), `variational/` (variational population codes, `PopulationCodeProcess`, the canonical circuit). `harmonium/`, `graphical/` and `dynamical/` hold exact families only.

### Invariants
- **Each node has a space and each clique one coordinate block.** Every `ExponentialFamily` is a `LinearCliques` (by default one node); a harmonium is a `RecursiveLinearCliques` `[root | cross | deep]` whose graph is composed from its parts. A concrete model declares `rot_man`, `dep_man` and one `CrossTerm` per crossing; everything else is derived.
- **Node numbers are local.** A manifold numbers only its own nodes; an enclosing class translates.
- **Crossings lie in the coordinate blocks they touch.** This keeps the likelihood and the posterior in their families; it holds by declaration and is checked over every shipped model in `tests/manifold.py`, not at run time.
- **Conjugation is placed, not derived globally.** A graphical harmonium places each observable harmonium's posterior on nodes of its latent model, and sums their conjugation parameters there. A variational level (`VariationalConjugated(gph_hrm, cnj_funs, dep_man)`) does the same with one learned or exact conjugation function per observable harmonium; $\rho$ lives in the harmonium's posterior family, so its prior family plays no role. Levels nest through `dep_man`, the next level or, at the top, a `DifferentiableDeepModel`.
- **Capability is in the class.** `Analytic` models give closed forms (EM), `Differentiable` ones the log-partition function and its gradient; a `VariationalConjugated` is not an exponential family.

## Code Conventions

Python 3.14+ (PEP 695 generics, no `from __future__ import annotations`). Pragmatic typing: accept type-system limitations rather than fight them when the code is correct; JAX, optax and scipy types are suppressed where incomplete.

### Abstract versus concrete
Decided by **is-a versus has-a**:
- **Abstract, stateless** when the class is a role a model becomes; the model derives its properties from its own fields (`Manifold`, `Pair`/`Triple`, `Map`, `Embedding`, `LinearCliques`, `ExponentialFamily`, `Harmonium` and their chains).
- **Concrete, with fields** when the class is a value a model builds, occurring with multiplicity and with primitive fields (`MatrixMap`, `AffineMap`, `CliqueMap`, `CrossMap`, `ManifoldTuple`, `MultilayerPerceptron`, the conjugation functions, `GraphicalLevel`, `VariationalConjugated`, the clique embeddings).
- **Abstract classes may carry fields** only when every subclass shares them and the subclasses differ only in behaviour. Hoist a field to the lowest parent all its holders share.
- **Embeddings are stored at the level of their ambient.** A class may hold an embedding into itself or into one of its parts, never into a manifold that contains it; locating a component in an enclosing manifold is the enclosing class's job.
- **A field backing an inherited abstract property is private**, named for the property (`_dom_man`, `_cod_man`); other fields are public.
- **Derived quantities are properties, not cached fields.** `__post_init__` is for validation only.
- **No defaults on dataclass fields** for modeling choices (`Binomial.n_trials`, `MultilayerPerceptron.hidden_dims`): defaults hide design decisions. The exception is numerical details callers need not reason about (`CoMPoisson.window_size`).

### Target-first order
Maps and embeddings list their manifolds target first, as a matrix lists rows before columns: `Map[Codomain, Domain]`, `Embedding[Ambient, Sub]`; a harmonium's interaction $\Theta_{XZ}$ is `CrossMap[Observable, Posterior]`. Fields and constructor arguments follow the same order.

### Coordinate system naming
Flat arrays are named by coordinate system: `params` (natural parameters of a model), `*_params` (slices of them, e.g. `obs_params`), `means` (mean parameters), `coords` (agnostic, in the manifold layer). Docstrings say which system their inputs are in ("at the given natural parameters").

### Class body organization
Order members as `# Fields`, `# Contract` (abstract members), `# Overrides`, `# Methods`, with these comment headers unless the class is small; properties before methods within a section.

### Type aliases
`FullNormal`, `DiagonalNormal`, `IsotropicNormal` and `StandardNormal` are `Normal` with `PositiveDefinite`, `Diagonal`, `Scale` and `Identity` covariance.

### Guards
Only add runtime checks for errors that would otherwise pass silently.

## Documentation

- **Docstrings are the single source of truth.** RST files under `docs/source/` are thin scaffolding (title, `automodule` with `:noindex: :no-members:`, inheritance diagram, `autoclass` sections); `index.rst` files get a short orientation and a toctree.
- **RST mirrors the package:** one RST file per module, same path and filename (`models/base/poisson.py` → `docs/source/models/base/poisson.rst`); each class documented only in its defining module's file. Internal utilities may be skipped.
- **Docstring structure:** lead with what it does in concrete terms (arrays in, arrays out, when to use it). Mathematics follows after a **"Mathematically,"** marker, precise and brief, at the highest class where it applies; subclasses state only their specialization.
- **Module docstrings:** a summary line and at most one paragraph (about 100 words) of pointers. Equations, costs and layouts go on the class or method.
- **Args blocks** only when name and type are not enough (shapes, constraints, non-obvious defaults).
- **Math notation:** LaTeX, never Unicode, in docstrings and comments; doubled backslashes in docstrings, single in comments; matplotlib labels as raw strings with `$...$`.

## Tests

Files drop the `test_` prefix (`python_files = ["*.py"]`). One flat file per geometry module plus one for the Gaussian family; a concrete model is tested where its abstraction lives, by one test parametrized over the models it applies to.

| File | Covers |
|---|---|
| `algebra.py` | matrix representations, `Cliques`, junction trees (sum-product against enumeration) |
| `manifold.py` | layout contracts over every shipped model, embedding laws, `CliqueMap`/`CrossMap` against dense matrices, `MultilayerPerceptron` |
| `exponential_family.py` | every base family: normalization, sampling, round trips, entropy, products, closed forms |
| `gaussian.py` | `Normal` against scipy, Boltzmann machines against enumeration |
| `harmonium.py` | every conjugated harmonium, graphical ones included: conjugation, marginals by brute force, round trips, EM |
| `graphical.py` | mixtures of harmoniums, whitening |
| `variational.py` | single levels against quadrature, exact levels against their harmonium, the canonical circuit's ELBO |
| `dynamical.py` | Kalman filter, HMM, EM, the variational filter against quadrature |

- **Tests use shipped classes only**; helpers are plain functions (quadrature, enumeration, references).
- **Every test checks a contract or an independent ground truth** (closed form, scipy, enumeration, quadrature, dense reference, sampling against exact means). Shape, dimension and `isinstance` checks, and tests that restate the implementation, are not tests.
- Each file starts with `jax_platform_name=cpu` and `jax_enable_x64=True` and a module docstring; one test class per concept, `@pytest.mark.parametrize` for variation, models and keys built inline (no shared fixtures).
- **Tolerances:** analytic `rtol=1e-5, atol=1e-7` (`1e-4/1e-6` for less precise models). Sampling uses 50k samples against `to_mean`: `atol=0.01–0.02` for simple families; for `Normal`, `0.03` on the mean and `0.1` on all sufficient statistics.

## Examples

Each example is `examples/<name>/` with `run.py` (computes, saves `analysis.json` via `paths.save_analysis()`), `plot.py` (loads it, saves `plot.png` via `paths.save_plot()`), and `types.py` (the typed JSON schema between them).

- **One-shot:** `main()` is `jax_cli()` → `example_paths(__file__)` → key → compute → save, with every setting a local of `main()`; no module-level state, no CLI flags beyond `jax_cli`'s. Grids and sweeps belong in `scratch/` (gitignored).
- **Loops** use `jax.lax.scan`; for periodic metrics, an outer Python loop over chunks of an inner scan.
- **No direct file I/O**; use `paths`.
- **Plots** use `apply_style(paths)` once, colours and figure sizes from `shared.py` (`colors`, `model_color`, `metric_color`, `figure_size`), short Title-Case titles, no `fontsize`/`grid`/`tight_layout` overrides, and `GridSpec(..., figure=fig)`.
