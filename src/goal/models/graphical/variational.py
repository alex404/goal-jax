"""Variational hierarchical mixture model.

Extends VariationalConjugated for models with mixture posterior structure,
where the latent space is a CompleteMixture over a base latent manifold.
"""

from dataclasses import dataclass
from typing import Any, override

import jax.numpy as jnp
from jax import Array

from ...geometry import Differentiable, LinearEmbedding, ObservableEmbedding
from ...geometry.exponential_family.variational import VariationalConjugated
from ..harmonium.mixture import CompleteMixture


@dataclass(frozen=True)
class VariationalHierarchicalMixture[
    Observable: Differentiable,
    BaseLatent: Differentiable,
](
    VariationalConjugated[Observable, Any, BaseLatent],
):
    """Variational harmonium with mixture latent structure.

    For a three-level hierarchy (Observable <-> BaseLatent <-> Categorical),
    the rho correction only affects the BaseLatent (observable) component
    of the mixture, leaving interaction and categorical unchanged.

    Model structure:
        Observable (X) <-> Base Latent (Y) <-> Categorical (K)
           |                    |                    |
        e.g., Binomials     Bernoullis        Mixture component

    Joint distribution: p(x, y, k) = p(k) * p(y|k) * p(x|y)

    The posterior is a CompleteMixture[BaseLatent], and the rho correction
    is applied only to the observable (BaseLatent) bias, not to the
    mixture interaction or categorical parameters.
    """

    @property
    def mix_man(self) -> CompleteMixture[BaseLatent]:
        """Complete mixture over base latent."""
        return self.hrm.pst_man  # type: ignore[return-value]

    @property
    def n_categories(self) -> int:
        """Number of mixture components."""
        return self.mix_man.n_categories

    @property
    def bas_lat_man(self) -> BaseLatent:
        """Base latent manifold (before mixture)."""
        return self.mix_man.obs_man  # type: ignore[return-value]

    @property
    @override
    def rho_emb(self) -> LinearEmbedding[BaseLatent, Any]:
        """Embedding from BaseLatent to mixture's observable component.

        The rho correction only affects the observable (BaseLatent) bias,
        not the mixture interaction or categorical parameters.
        """
        return ObservableEmbedding(self.mix_man)

    def get_cluster_probs(self, mixture_params: Array) -> Array:
        """Extract cluster probabilities from mixture parameters.

        For a mixture q(y,k), the marginal over k is:
            q(k) proportional to exp(theta_K[k] + psi_Y(theta_Y[k]))

        Uses the harmonium prior() which computes effective categorical natural
        parameters (cat_params + conjugation_parameters), then converts to
        probabilities via to_mean() (softmax) and to_probs() (prepend p_0).

        Args:
            mixture_params: Natural parameters of the mixture

        Returns:
            Array of shape (n_categories,) with cluster probabilities
        """
        cat_natural = self.mix_man.prior(mixture_params)
        cat_means = self.mix_man.lat_man.to_mean(cat_natural)
        return self.mix_man.lat_man.to_probs(cat_means)

    def prior_entropy(self, params: Array) -> Array:
        """Compute entropy of the prior's marginal cluster distribution.

        This is useful for regularization to prevent cluster collapse.

        Args:
            params: Full model parameters

        Returns:
            Entropy of prior cluster distribution (scalar)
        """
        _, hrm_params = self.split_coords(params)
        _, _, prior_mixture_params = self.hrm.split_coords(hrm_params)
        probs = self.get_cluster_probs(prior_mixture_params)
        probs_safe = jnp.clip(probs, 1e-10, 1.0)
        return -jnp.sum(probs_safe * jnp.log(probs_safe))
