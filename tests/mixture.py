from typing import cast

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from jax import random

from goal.exponential_family import MultivariateGaussian
from goal.harmonium import Categorical, create_gaussian_mixture


def test_gaussian_mixture():
    """Test Gaussian mixture model implemented as Harmonium"""
    # Create figure
    _, ax1 = plt.subplots(1, 1, figsize=(8, 6))

    # Create a 2-component mixture in 1D
    mixture = create_gaussian_mixture(dim=1, n_components=2)

    # Update parameters for better visualization
    mixture = mixture.replace(
        visible=MultivariateGaussian(
            dim=1, mean=jnp.array([0.0]), covariance=jnp.array([[1.0]])
        ),
        interaction=jnp.array([[2.0, -2.0]]),  # Centers at -2, 2
    )

    # Get distribution objects with proper types
    categorical = cast(Categorical, mixture.hidden)
    gaussian = cast(MultivariateGaussian, mixture.visible)

    # Generate samples
    key = random.PRNGKey(0)
    n_samples = 1000

    # Sample component assignments
    z = random.categorical(key, jnp.zeros(categorical.n_categories), shape=(n_samples,))

    # Sample observations given components
    x = []
    for zi in z:
        mean_i = gaussian.mean + mixture.interaction[:, zi]
        x_i = random.multivariate_normal(key, mean_i, gaussian.covariance)
        key, _ = random.split(key)
        x.append(x_i)
    x = jnp.array(x)

    # Plot samples
    sns.histplot(data=np.array(x), stat="density", ax=ax1)
    ax1.set_title("Sample Distribution")

    # Plot mixture density
    x_range = np.linspace(-6, 6, 200)
    density = np.zeros_like(x_range)
    for k in range(categorical.n_categories):
        mean_k = gaussian.mean + mixture.interaction[:, k]
        gauss_k = MultivariateGaussian(
            dim=1, mean=mean_k, covariance=gaussian.covariance
        )
        density_k = np.array(
            [np.exp(gauss_k.log_density(jnp.array([xi]))) for xi in x_range]
        )
        density += density_k / categorical.n_categories

    ax1.plot(x_range, density, "r-", label="True Density")
    ax1.legend()

    plt.tight_layout()
    plt.savefig("mixture_tests.png")
    plt.close()


if __name__ == "__main__":
    test_gaussian_mixture()
