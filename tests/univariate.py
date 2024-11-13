import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

from goal.exponential_family import MultivariateGaussian


def plot_distributions():
    """Test and plot both distributions"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Test Gaussian
    mu = jnp.array([2.0])
    sigma = jnp.array([[1.5]])
    gaussian = MultivariateGaussian.create(mean=mu, covariance=sigma)

    # Plot density
    x = np.linspace(-2, 6, 200)
    x_points = jnp.array(x)[:, None]
    densities = jnp.exp(jnp.array([gaussian.log_density(xi) for xi in x_points]))
    scipy_densities = norm.pdf(x, loc=float(mu[0]), scale=np.sqrt(float(sigma[0, 0])))

    ax1.plot(x, densities, "b-", label="Our Implementation")
    ax1.plot(x, scipy_densities, "r--", label="Scipy")
    ax1.set_title("Gaussian Density")
    ax1.legend()

    # Test Categorical
    # probs = jnp.array([0.1, 0.2, 0.4, 0.2, 0.1])
    # categorical = Categorical.create(n_categories=5, probs=probs)
    #
    # # Plot PMF
    # categories = np.arange(5)
    # cat_probs = jnp.exp(jnp.array([categorical.log_density(i) for i in categories]))
    #
    # ax2.bar(categories, np.array(cat_probs), alpha=0.5, label="Implementation")
    # ax2.bar(categories, np.array(probs), alpha=0.5, label="True Probabilities")
    # ax2.set_title("Categorical PMF")
    # ax2.legend()

    plt.tight_layout()
    plt.show()

    # Test coordinate transforms
    print("\nTesting coordinate transforms:")

    # Gaussian
    natural_gaussian = gaussian.to_natural()
    mean_gaussian = natural_gaussian.to_mean()
    test_point = jnp.array([1.5])

    print("\nGaussian at test point:", float(test_point[0]))
    print(f"Original log density: {float(gaussian.log_density(test_point)):.4f}")
    print(
        f"Natural form log density: {float(natural_gaussian.log_density(test_point)):.4f}"
    )
    print(f"Mean form log density: {float(mean_gaussian.log_density(test_point)):.4f}")

    # Categorical
    # natural_cat = categorical.to_natural()
    # mean_cat = natural_cat.to_mean()
    # test_cat = 2
    #
    # print("\nCategorical at category:", test_cat)
    # print(f"Original log probability: {float(categorical.log_density(test_cat)):.4f}")
    # print(
    #     f"Natural form log probability: {float(natural_cat.log_density(test_cat)):.4f}"
    # )
    # print(f"Mean form log probability: {float(mean_cat.log_density(test_cat)):.4f}")


if __name__ == "__main__":
    plot_distributions()
