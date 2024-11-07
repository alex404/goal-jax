import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from jax import random
from scipy.stats import norm

from goal.exponential_family import MultivariateGaussian


def test_1d_gaussian():
    """Test 1D Gaussian distribution properties"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))

    # Initialize a 1D Gaussian
    mu = jnp.array([2.0])
    sigma = jnp.array([[1.5]])
    gaussian = MultivariateGaussian(dim=1, mean=mu, covariance=sigma)

    # Test 1: Plot density curve against scipy implementation
    x = np.linspace(-2, 6, 200)  # Using numpy for scipy compatibility
    x_points = jnp.array(x)[:, None]  # Shape for our implementation
    log_densities = jnp.array([gaussian.log_density(xi) for xi in x_points])
    densities = np.exp(np.array(log_densities))  # Convert to numpy for plotting

    # Compare with scipy - using norm.pdf which is definitely available
    scipy_densities = norm.pdf(x, loc=mu[0], scale=np.sqrt(sigma[0, 0]))

    ax1.plot(x, densities, "b-", label="Our Implementation")
    ax1.plot(x, scipy_densities, "r--", label="Scipy")
    ax1.set_title("Density Comparison")
    ax1.legend()

    # Test 2: Generate and plot samples
    key = random.PRNGKey(0)
    n_samples = 1000
    samples = jnp.array(
        [
            random.multivariate_normal(key, gaussian.mean, gaussian.covariance)
            for key in random.split(key, n_samples)
        ]
    )

    # Convert to numpy for seaborn
    samples_np = np.array(samples)
    sns.histplot(data=samples_np, stat="density", ax=ax2)
    ax2.plot(x, densities, "r-", label="True Density")
    ax2.set_title("Sample Distribution")
    ax2.legend()

    # Test 3: MLE estimation
    key = random.PRNGKey(1)
    true_mu = jnp.array([1.0])
    true_sigma = jnp.array([[2.0]])
    true_gaussian = MultivariateGaussian(dim=1, mean=true_mu, covariance=true_sigma)

    # Generate samples from true distribution
    samples = jnp.array(
        [
            random.multivariate_normal(key, true_mu, true_sigma)
            for key in random.split(key, n_samples)
        ]
    )

    # Compute MLE
    mle_mu = jnp.mean(samples, axis=0)
    centered = samples - mle_mu
    mle_sigma = jnp.mean(jnp.array([jnp.outer(x, x) for x in centered]), axis=0)
    mle_gaussian = MultivariateGaussian(dim=1, mean=mle_mu, covariance=mle_sigma)

    # Plot true vs MLE densities
    true_densities = jnp.exp(
        jnp.array([true_gaussian.log_density(xi) for xi in x_points])
    )
    mle_densities = jnp.exp(
        jnp.array([mle_gaussian.log_density(xi) for xi in x_points])
    )

    # Convert to numpy for plotting
    ax3.plot(x, np.array(true_densities), "b-", label="True Distribution")
    ax3.plot(x, np.array(mle_densities), "r--", label="MLE Estimate")
    ax3.hist(np.array(samples), bins=50, density=True, alpha=0.3, label="Samples")
    ax3.set_title("MLE Estimation")
    ax3.legend()

    # Test 4: Natural parameters
    test_point = jnp.array([1.5])
    suff_stat = gaussian.sufficient_statistic(test_point)
    natural_params = gaussian.to_natural()

    # Compare different ways of computing density
    raw_density = gaussian.log_density(test_point)
    natural_density = (
        jnp.dot(natural_params, suff_stat)
        + gaussian.log_base_measure(test_point)
        - gaussian.potential(natural_params)
    )

    text_info = [
        f"Test Point: {float(test_point[0]):.2f}",
        f"Sufficient Statistics: {np.array(suff_stat)}",
        f"Natural Parameters: {np.array(natural_params)}",
        f"Log Density (direct): {float(raw_density):.2f}",
        f"Log Density (natural): {float(natural_density):.2f}",
        f"Difference: {float(abs(raw_density - natural_density)):.2e}",
    ]

    ax4.text(0.1, 0.5, "\n".join(text_info), transform=ax4.transAxes)
    ax4.set_title("Parameter Analysis")
    ax4.axis("off")

    plt.tight_layout()
    plt.savefig("gaussian_tests.png")
    plt.close()


if __name__ == "__main__":
    test_1d_gaussian()
