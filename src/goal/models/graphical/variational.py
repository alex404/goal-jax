"""Variational hierarchical mixture model.

Extends VariationalConjugated for hierarchical models with mixture posterior structure: the underlying harmonium has interaction structurally restricted to the observable (BaseLatent) component of the mixture posterior --- no per-component observable offsets, so the analytic mixture completion of the conjugation collapses to a simple zero-pad embedding of the learned $\\rho$ into the mixture-shaped Prior space.
"""

from abc import ABC
from typing import Any, override

from jax import Array

from ...geometry import Differentiable, ObservableEmbedding
from ...geometry.exponential_family.variational import SymmetricVariationalConjugated
from ..harmonium.mixture import CompleteMixture


class VariationalHierarchicalMixture[
    Observable: Differentiable,
    BaseLatent: Differentiable,
](
    SymmetricVariationalConjugated[Observable, Any, BaseLatent],
    ABC,
):
    """Variational harmonium with hierarchical mixture latent structure.

    Three-level hierarchy ``Observable (X) <-> BaseLatent (Y) <-> Categorical (K)``, where the underlying harmonium's interaction only acts on the BaseLatent component of the mixture (constructed via :class:`~goal.geometry.exponential_family.graphical.ObservableEmbedding`). Posterior = Prior = ``CompleteMixture[BaseLatent]``; Conjugation = ``BaseLatent``.

    Because the interaction is structurally restricted to the BaseLatent slot, the marginal $p(z, k) = \\int p(x|y) p(x, y, k) dx$ only picks up conjugation along the BaseLatent direction. The analytic mixture completion therefore zero-pads the learned $\\rho_y$ into the (mixture-interaction, categorical) slots of the Prior, mirroring how the analytic-side :class:`~goal.geometry.exponential_family.graphical.DifferentiableHierarchical` zero-pads via :class:`~goal.geometry.exponential_family.graphical.ObservableEmbedding`.

    The contrast with a hypothetical full-mixture variant (built atop :class:`~goal.models.graphical.mixture.CompleteMixtureOfHarmoniums`): the analytic mixture completion would supply a *non-zero* categorical contribution from the three-block interaction (``xy``, ``xyk``, ``xk``), reflecting log-partition differences across component-specific observable offsets.
    """

    @property
    def mix_man(self) -> CompleteMixture[BaseLatent]:
        """Complete mixture over base latent."""
        return self.gen_hrm.pst_man

    @property
    def n_categories(self) -> int:
        """Number of mixture components."""
        return self.mix_man.n_categories

    @property
    def bas_lat_man(self) -> BaseLatent:
        """Base latent manifold (before mixture)."""
        return self.mix_man.obs_man

    @property
    @override
    def lat_man(self) -> Any:
        return self.gen_hrm.pst_man

    @property
    @override
    def cnj_man(self) -> BaseLatent:
        return self.bas_lat_man

    @override
    def conjugation_parameters(self, params: Array) -> Array:
        """Zero-pad the learned $\\rho_y$ into the full mixture-shape Prior conjugation.

        In the hierarchical case the underlying harmonium's interaction only touches the BaseLatent slot, so the analytic mixture completion has $\\rho_{yz} = 0$ and $\\rho_z = 0$. The full conjugation is just $\\rho_y$ embedded via :class:`~goal.geometry.exponential_family.graphical.ObservableEmbedding` into the mixture.
        """
        _, _, rho = self.split_coords(params)
        return ObservableEmbedding(self.mix_man).embed(rho)

    def get_cluster_probs(self, mixture_params: Array) -> Array:
        """Extract cluster probabilities from mixture natural parameters by marginalizing out the observable."""
        cat_natural = self.mix_man.prior(mixture_params)
        cat_means = self.mix_man.lat_man.to_mean(cat_natural)
        return self.mix_man.lat_man.to_probs(cat_means)

    def prior_entropy(self, params: Array) -> Array:
        """Compute entropy of the prior's marginal cluster distribution.

        This is useful for regularization to prevent cluster collapse.
        """
        cat_params = self.mix_man.prior(self.prior_params(params))
        cat_means = self.mix_man.lat_man.to_mean(cat_params)
        return -self.mix_man.lat_man.negative_entropy(cat_means)
