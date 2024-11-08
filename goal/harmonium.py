import equinox as eqx
import jax
import jax.numpy as jnp

from .exponential_family import ExponentialFamily


class Categorical(ExponentialFamily):
    """Categorical distribution as exponential family"""

    n_categories: int
    logits: jnp.ndarray

    def __init__(self, n_categories: int):
        self.n_categories = n_categories
        self.logits = jnp.zeros(n_categories)

    def sufficient_statistic(self, x: jnp.ndarray) -> jnp.ndarray:
        """One-hot encoding"""
        return jax.nn.one_hot(x, self.n_categories)

    def log_base_measure(self, x: jnp.ndarray) -> float:
        """Base measure is constant 1, so log is 0"""
        return 0.0

    def potential(self, natural_params: jnp.ndarray) -> float:
        """Log sum exp of natural parameters"""
        return float(jax.nn.logsumexp(natural_params))

    def to_natural(self) -> jnp.ndarray:
        return self.logits


class Harmonium(ExponentialFamily):
    """Base Harmonium model - product of exponential families with interaction"""

    visible: ExponentialFamily
    hidden: ExponentialFamily
    interaction: jnp.ndarray

    def sufficient_statistic(self, x: jnp.ndarray) -> jnp.ndarray:
        """Compute sufficient statistics from joint observation (x, z)"""
        x_vis, x_hid = x  # Unpack the tuple
        tx = self.visible.sufficient_statistic(x_vis)
        tz = self.hidden.sufficient_statistic(x_hid)
        return jnp.concatenate([tx, tz, (tx[:, None] * tz[None, :]).ravel()])

    def log_base_measure(self, x: jnp.ndarray) -> float:
        """Sum of component base measures"""
        x_vis, x_hid = x  # Unpack the tuple
        return float(
            self.visible.log_base_measure(x_vis) + self.hidden.log_base_measure(x_hid)
        )

    def potential(self, natural_params: jnp.ndarray) -> float:
        """Log partition function"""
        n_visible = len(self.visible.to_natural())
        n_hidden = len(self.hidden.to_natural())
        nx = natural_params[:n_visible]
        nz = natural_params[n_visible : n_visible + n_hidden]
        nxz = natural_params[n_visible + n_hidden :].reshape(n_visible, n_hidden)

        return float(
            self.visible.potential(nx)
            + self.hidden.potential(nz)
            + jnp.sum(nxz * self.interaction)
        )

    def to_natural(self) -> jnp.ndarray:
        """Convert to natural parameters"""
        nx = self.visible.to_natural()
        nz = self.hidden.to_natural()
        return jnp.concatenate([nx, nz, self.interaction.ravel()])

    def replace(self, **updates) -> "Harmonium":
        """Create new Harmonium with updated fields"""
        return eqx.tree_at(
            lambda m: [getattr(m, k) for k in updates],
            self,
            [updates[k] for k in updates],
        )


def create_gaussian_mixture(dim: int, n_components: int) -> Harmonium:
    """Create a Gaussian mixture model as a Harmonium"""
    from .exponential_family import MultivariateGaussian

    # Component distributions
    observable = MultivariateGaussian(
        dim=dim, mean=jnp.zeros(dim), covariance=jnp.eye(dim)
    )
    latent = Categorical(n_components)

    # Interaction matrix
    interaction = jnp.zeros((dim, n_components))

    return Harmonium(visible=observable, hidden=latent, interaction=interaction)
