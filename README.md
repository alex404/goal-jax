![Tests](https://github.com/alex404/goal-jax/actions/workflows/tests.yml/badge.svg)
![Type Check](https://github.com/alex404/goal-jax/actions/workflows/typecheck.yml/badge.svg)
![Examples](https://github.com/alex404/goal-jax/actions/workflows/examples.yml/badge.svg)

# goal-jax: Geometric OptimizAtion Libraries (in JAX!)

A JAX framework for statistical modeling grounded in information geometry and exponential families. Goal provides machine learning algorithms based on the theory of statistical manifolds --- mathematical spaces where every point is a probability distribution. Models are lightweight, stateless objects that define operations on flat JAX arrays, enabling high-level algorithms for inference, learning, and model evaluation in complex latent variable models.

## Installation

```bash
pip install goal-jax
```

For development:

```bash
git clone https://github.com/alex404/goal-jax.git
cd goal-jax
uv sync --all-extras
```

## Quick Start

Fit a 3-component Gaussian mixture model via expectation-maximization:

```python
import jax
import jax.numpy as jnp
from goal.geometry import PositiveDefinite
from goal.models import AnalyticMixture, Normal

# 3-component 2D Gaussian mixture with full covariance
model = AnalyticMixture(Normal(2, PositiveDefinite()), n_components=3)

# Generate synthetic data from a known mixture
key = jax.random.PRNGKey(0)
ground_truth = model.initialize(key)
sample = model.observable_sample(key, ground_truth, n=500)

# Fit via expectation-maximization
key, subkey = jax.random.split(key)
params = model.initialize(subkey)

def em_step(params, _):
    return model.expectation_maximization(params, sample), None

params, _ = jax.lax.scan(em_step, params, None, length=50)

# Evaluate fit
ll = model.average_log_observable_density(params, sample)
```

The full example compares full, diagonal, and isotropic covariance across EM iterations --- see [`examples/mixture_of_gaussians/`](examples/mixture_of_gaussians/):

![Mixture of Gaussians](docs/source/_static/mixture_of_gaussians.png)

## Documentation

https://goal-jax.readthedocs.io/

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
