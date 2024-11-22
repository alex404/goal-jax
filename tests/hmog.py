import json
from typing import Tuple, cast

import jax
import jax.numpy as jnp
from jax import random

from goal.exponential_family import MultivariateGaussian
from goal.harmonium import Categorical, Harmonium, create_gaussian_mixture

# Constants from Haskell implementation
WIDTH = 0.1
HEIGHT = 8
SEP = 5
N_OBSERVATIONS = 400
N_EPOCHS = 200
PLOT_RESOLUTION = 100


def create_true_model() -> Harmonium:
    """Create the true model matching Haskell implementation"""
    mixture = create_gaussian_mixture(dim=2, n_components=2)

    # Set up visible Gaussian (trusx in Haskell)
    return mixture.replace(
        visible=MultivariateGaussian(
            dim=2,
            mean=jnp.array([0.0, 0.0]),
            covariance=jnp.diag(jnp.array([WIDTH, HEIGHT])),
        ),
        # Interaction matrix (trusfa in Haskell)
        interaction=jnp.array([[1.0], [0.0]]),
    )


def initialize_hmog() -> Harmonium:
    """Initialize model for training"""
    mixture = create_gaussian_mixture(dim=2, n_components=2)

    key = random.PRNGKey(0)
    keys = random.split(key, 3)

    # Initialize with small random perturbations
    visible = MultivariateGaussian(
        dim=2,
        mean=jnp.zeros(2) + 0.1 * random.normal(keys[0], (2,)),
        covariance=2.0 * jnp.eye(2),
    )

    interaction = 0.1 * random.normal(keys[1], (2, 1))

    return mixture.replace(visible=visible, interaction=interaction)


def expectation_step(
    model: Harmonium, observations: jnp.ndarray
) -> Tuple[jnp.ndarray, float]:
    """E-step of EM algorithm"""
    gaussian = cast(MultivariateGaussian, model.visible)
    categorical = cast(Categorical, model.hidden)

    n_samples = len(observations)
    log_responsibilities = jnp.zeros((n_samples, categorical.n_categories))

    for k in range(categorical.n_categories):
        mean_k = gaussian.mean + model.interaction[:, k]
        log_prob = sum(
            MultivariateGaussian(
                dim=2, mean=mean_k, covariance=gaussian.covariance
            ).log_density(x)
            for x in observations
        )
        log_responsibilities = log_responsibilities.at[:, k].set(log_prob)

    # Normalize responsibilities
    log_normalizer = jax.scipy.special.logsumexp(
        log_responsibilities, axis=1, keepdims=True
    )
    log_responsibilities = log_responsibilities - log_normalizer

    return jnp.exp(log_responsibilities), jnp.mean(log_normalizer)


def maximization_step(
    model: Harmonium, observations: jnp.ndarray, responsibilities: jnp.ndarray
) -> Harmonium:
    """M-step of EM algorithm"""
    gaussian = cast(MultivariateGaussian, model.visible)
    categorical = cast(Categorical, model.hidden)

    # Update Gaussian parameters
    weighted_mean = (
        jnp.sum(responsibilities[:, :, None] * observations[:, None, :], axis=0)
        / jnp.sum(responsibilities, axis=0)[:, None]
    )

    # Update interaction matrix
    new_interaction = jnp.zeros_like(model.interaction)
    for k in range(categorical.n_categories):
        resp_k = responsibilities[:, k]
        weighted_sum = jnp.sum(resp_k)
        mean_k = jnp.sum(resp_k[:, None] * observations, axis=0) / weighted_sum
        new_interaction = new_interaction.at[:, k].set(mean_k - gaussian.mean)

    return model.replace(interaction=new_interaction)


def compute_densities(model: Harmonium, grid_points: jnp.ndarray) -> jnp.ndarray:
    """Compute model densities on grid points"""
    gaussian = cast(MultivariateGaussian, model.visible)
    categorical = cast(Categorical, model.hidden)

    densities = jnp.zeros(len(grid_points))
    for k in range(categorical.n_categories):
        mean_k = gaussian.mean + model.interaction[:, k]
        component_k = MultivariateGaussian(
            dim=2, mean=mean_k, covariance=gaussian.covariance
        )
        densities += (
            jnp.exp(jnp.array([component_k.log_density(x) for x in grid_points]))
            / categorical.n_categories
        )

    return densities


def main():
    # Create true model and generate data
    true_model = create_true_model()
    key = random.PRNGKey(0)

    # Generate samples
    categorical = cast(Categorical, true_model.hidden)
    gaussian = cast(MultivariateGaussian, true_model.visible)

    z = random.categorical(
        key, jnp.zeros(categorical.n_categories), shape=(N_OBSERVATIONS,)
    )

    observations = []
    for zi in z:
        key, subkey = random.split(key)
        mean_i = gaussian.mean + true_model.interaction[:, zi]
        x_i = random.multivariate_normal(subkey, mean_i, gaussian.covariance)
        observations.append(x_i)
    observations = jnp.array(observations)

    # Initialize and train model
    model = initialize_hmog()
    log_likelihoods = []
    models = [model]

    for epoch in range(N_EPOCHS):
        responsibilities, ll = expectation_step(model, observations)
        log_likelihoods.append(float(ll))

        if epoch % 10 == 0:
            print(f"Iteration {epoch} Log-Likelihood: {ll:.4f}")

        model = maximization_step(model, observations, responsibilities)
        models.append(model)

    # Generate grid points for density evaluation
    x1_range = jnp.linspace(-10, 10, PLOT_RESOLUTION)
    x2_range = jnp.linspace(-10, 10, PLOT_RESOLUTION)
    y_range = jnp.linspace(-6, 6, PLOT_RESOLUTION)

    grid_points = jnp.array([[x1, x2] for x2 in x2_range for x1 in x1_range])

    # Compute densities
    true_densities = compute_densities(true_model, grid_points)
    initial_densities = compute_densities(models[0], grid_points)
    final_densities = compute_densities(models[-1], grid_points)

    # Save results in format expected by plotting script
    results = {
        "plot-range-x1": x1_range.tolist(),
        "plot-range-x2": x2_range.tolist(),
        "observations": observations.tolist(),
        "log-likelihoods": log_likelihoods,
        "observable-densities": [
            true_densities.tolist(),
            initial_densities.tolist(),
            final_densities.tolist(),
        ],
        "plot-range-y": y_range.tolist(),
        "mixture-densities": [
            compute_densities(true_model, y_range[:, None]).tolist(),
            compute_densities(models[0], y_range[:, None]).tolist(),
            compute_densities(models[-1], y_range[:, None]).tolist(),
        ],
    }

    with open("hmog.json", "w") as f:
        json.dump(results, f)


if __name__ == "__main__":
    main()
