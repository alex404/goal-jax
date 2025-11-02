"""Mixture model utilities for conjugated harmoniums.

This module provides helper functions for computing conjugation parameters in hierarchical
mixture models, enabling efficient inference in models where conjugated harmoniums are composed
into mixture structures.

**Key function**: `harmonium_mixture_conjugation_parameters` decomposes the joint conjugation
parameters of a mixture of conjugated harmoniums into three components:

1. **Y conjugation biases**: Base conjugation parameters from the first component
2. **K conjugation biases**: Differences in log partition between components
3. **K-Y interaction**: Changes in conjugation parameters across components

This decomposition preserves the mixture structure and enables efficient conjugation parameter
computation for hierarchical models where the upper level is a mixture over harmoniums.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Self, override

import jax
import jax.numpy as jnp
from jax import Array

from ...geometry import (
    AffineMap,
    AnalyticConjugated,
    Conjugated,
    Differentiable,
    DifferentiableConjugated,
    LinearEmbedding,
    Mean,
    Natural,
    Point,
    Product,
    Rectangular,
)
from ..base.categorical import Categorical
from ..harmonium.mixture import AnalyticMixture, CompleteMixture, Mixture

### Embeddings ###


@dataclass(frozen=True)
class MixtureComponentEmbedding[
    SubComponent: Differentiable,
    AmbientComponent: Differentiable,
](LinearEmbedding[CompleteMixture[SubComponent], CompleteMixture[AmbientComponent]]):
    """Embedding between complete mixtures via per-component observable embedding.

    This embedding enables transformations on each component of a complete mixture (where
    Observable = IntObservable). Both mixtures must have the same number of components.

    **Example**: Embedding mixture parameters (Mixture[Euclidean, Euclidean])
    into posterior distributions (Mixture[FullNormal, FullNormal]) by applying a
    component embedding (Euclidean → FullNormal) to each mixture component.

    **Structure**:
    - Sub-manifold: Complete mixture with observable type SubComponent
    - Ambient manifold: Complete mixture with observable type AmbientComponent
    - Same number of components (n_categories)
    - Derived from single component embedding: sub_man and amb_man are constructed from cmp_emb

    **Coordinate Systems**:
    - `embed`: Works in Natural coordinates
    - `project`: Works in Mean coordinates

    **Operations**: Applies component embedding/projection to observable parameters while
    preserving interaction and latent parameters, since the interaction structure depends only
    on the observable dimension (which is preserved in complete mixtures).
    """

    # Fields

    n_categories: int
    """Number of mixture components."""

    cmp_emb: LinearEmbedding[SubComponent, AmbientComponent]
    """The embedding to apply to each observable component."""

    # Overrides

    @property
    @override
    def sub_man(self) -> CompleteMixture[SubComponent]:
        """The sub-manifold symmetric mixture."""
        return CompleteMixture(
            self.cmp_emb.sub_man,
            n_categories=self.n_categories,
        )

    @property
    @override
    def amb_man(self) -> CompleteMixture[AmbientComponent]:
        """The ambient manifold symmetric mixture."""
        return CompleteMixture(
            self.cmp_emb.amb_man,
            n_categories=self.n_categories,
        )

    @override
    def embed(
        self, p: Point[Natural, CompleteMixture[SubComponent]]
    ) -> Point[Natural, CompleteMixture[AmbientComponent]]:
        """Embed by applying the component embedding to each component in Natural coordinates.

        Decomposes the mixture into component natural parameters and prior using
        `split_natural_mixture`. Maps the component embedding over the product of components.
        Reconstructs the mixture using `join_natural_mixture`.
        """
        components, prior = self.sub_man.split_natural_mixture(p)
        emb_components = self.sub_man.cmp_man.man_map(self.cmp_emb.embed, components)
        return self.amb_man.join_natural_mixture(Point(emb_components.array), prior)

    @override
    def project(
        self, p: Point[Mean, CompleteMixture[AmbientComponent]]
    ) -> Point[Mean, CompleteMixture[SubComponent]]:
        """Project by applying the component projection to each component in Mean coordinates.

        Decomposes the mixture into component means and weights using `split_mean_mixture`.
        Maps the component projection over the product of components. Reconstructs the
        mixture using `join_mean_mixture`.
        """
        components, weights = self.amb_man.split_mean_mixture(p)
        prj_components = self.amb_man.cmp_man.man_map(self.cmp_emb.project, components)
        return self.sub_man.join_mean_mixture(Point(prj_components.array), weights)


### Mixture of Conjugated Harmoniums ###


class MixtureOfConjugated[
    Observable: Differentiable,
    Component: DifferentiableConjugated[Any, Any, Any, Any, Any, Any],
    IntComponent: DifferentiableConjugated[Any, Any, Any, Any, Any, Any],
    ComponentPosterior: Differentiable,
    IntComponentPosterior: Differentiable,
    ComponentPrior: Differentiable,
](
    Mixture[Component, IntComponent],
):
    """Mixture of conjugated harmoniums where each component is an independent harmonium.

    **Structure**: A mixture where each of K components is a conjugated harmonium. The joint
    distribution is:

    $$p(x,y,z) = p(z) \\prod_{k=1}^K \\mathbb{1}_{z=k} p(x,y|\\theta_k)$$

    where each $p(x,y|\\theta_k)$ is a conjugated harmonium with natural parameters $\\theta_k$.

    **Parameters**: Naturally split as $(\\theta_1, \\ldots, \\theta_K, \\pi)$ where:
    - Each $\\theta_k$ are natural parameters of a ConHarm harmonium
    - $\\pi$ are parameters of the Categorical prior over z

    **Note**: While this class extends Conjugated, the "observable" in the API refers to
    the joint (x,y) from the component harmoniums, and "latent" refers to the component
    index z. This allows reuse of the conjugated structure while properly modeling
    independent component harmoniums.
    """

    def component_posteriors_at(
        self, params: Point[Natural, Self], x: Array
    ) -> tuple[
        Point[Natural, Product[ComponentPosterior]], Point[Natural, Categorical]
    ]:
        hrm_cmps, cat_params = self.split_natural_mixture(params)

        def component_posterior_at_x(
            hrm_params: Point[Natural, Component],
        ) -> Point[Natural, ComponentPosterior]:
            return self.int_obs_emb.amb_man.posterior_at(hrm_params, x)

        pst_cmps = self.cmp_man.man_map(component_posterior_at_x, hrm_cmps)
        return pst_cmps, cat_params  # pyright: ignore[reportReturnType]

    def component_likelihoods_at(
        self, params: Point[Natural, Self], y: Array
    ) -> tuple[Point[Natural, Product[Observable]], Point[Natural, Categorical]]:
        hrm_cmps, cat_params = self.split_natural_mixture(params)

        def component_likelihood_at_x(
            hrm_params: Point[Natural, Component],
        ) -> Point[Natural, Observable]:
            return self.int_obs_emb.amb_man.likelihood_at(hrm_params, y)

        lkl_cmps = self.cmp_man.man_map(component_likelihood_at_x, hrm_cmps)
        return lkl_cmps, cat_params  # pyright: ignore[reportReturnType]

    @override
    def log_observable_density(
        self,
        params: Point[Natural, Self],
        x: Array,
    ) -> Array:
        hrm_cmps, cat_params = self.split_natural_mixture(params)

        def harmonium_log_density_at_x(
            hrm_params: Point[Natural, Component],
        ) -> Array:
            return self.obs_man.log_observable_density(hrm_params, x)

        lds = self.cmp_man.map(harmonium_log_density_at_x, hrm_cmps)
        wghts = self.lat_man.to_probs(self.lat_man.to_mean(cat_params))
        return jax.scipy.special.logsumexp(lds + jnp.log(wghts))


class CompleteMixtureOfConjugated[
    Observable: Differentiable,
    Component: DifferentiableConjugated[Any, Any, Any, Any, Any, Any],
    ComponentPosterior: Differentiable,
    ComponentPrior: Differentiable,
](
    MixtureOfConjugated[
        Observable,
        Component,
        Component,
        ComponentPosterior,
        ComponentPosterior,
        ComponentPrior,
    ],
    CompleteMixture[Component],
):
    @property
    def mix_pst_man(
        self,
    ) -> CompleteMixture[ComponentPosterior]:
        """Latent manifold for the mixture of conjugated harmoniums."""
        return CompleteMixture(self.obs_man.pst_man, self.n_categories)

    @property
    def mix_prr_man(
        self,
    ) -> CompleteMixture[ComponentPrior]:
        """Latent manifold for the mixture of conjugated harmoniums."""
        return CompleteMixture(self.obs_man.prr_man, self.n_categories)

    @override
    def posterior_at(
        self, params: Point[Natural, Self], x: Array
    ) -> Point[Natural, CompleteMixture[ComponentPosterior]]:
        pst_cmps, cat_params = self.component_posteriors_at(params, x)
        return self.mix_pst_man.join_natural_mixture(pst_cmps, cat_params)


class AnalyticMixtureOfConjugated[
    Observable: Differentiable,
    Component: AnalyticConjugated[Any, Any, Any, Any, Any],
    ComponentPosterior: Differentiable,
    ComponentPrior: Differentiable,
](
    MixtureOfConjugated[
        Observable,
        Component,
        Component,
        ComponentPosterior,
        ComponentPosterior,
        ComponentPrior,
    ],
    AnalyticMixture[Component],
):
    @property
    def mix_pst_man(
        self,
    ) -> CompleteMixture[ComponentPosterior]:
        """Latent manifold for the mixture of conjugated harmoniums."""
        return CompleteMixture(self.obs_man.pst_man, self.n_categories)

    @property
    def mix_prr_man(
        self,
    ) -> CompleteMixture[ComponentPrior]:
        """Latent manifold for the mixture of conjugated harmoniums."""
        return CompleteMixture(self.obs_man.prr_man, self.n_categories)

    @override
    def posterior_at(
        self, params: Point[Natural, Self], x: Array
    ) -> Point[Natural, CompleteMixture[ComponentPosterior]]:
        pst_cmps, cat_params = self.component_posteriors_at(params, x)
        return self.mix_pst_man.join_natural_mixture(pst_cmps, cat_params)


### Helper Functions ###


def harmonium_mixture_conjugation_parameters[
    Harm: Conjugated[Any, Any, Any, Any, Any, Any],
](
    mixture_likelihood: AffineMap[Rectangular, Categorical, Harm, Harm],
    lkl_params: Point[Natural, AffineMap[Rectangular, Categorical, Harm, Harm]],
) -> tuple[Point[Natural, Any], Point[Natural, Any], Point[Natural, Any]]:
    """Compute conjugation parameters for a mixture of conjugated harmoniums.

    This decomposes the joint conjugation parameters into three components:
    1. Y conjugation biases (latent variable biases from component 0)
    2. K conjugation biases (categorical variable biases from observable component differences)
    3. K-Y interaction columns (differences in conjugation parameters across components)

    The function extracts observable biases from each component harmonium and computes
    a mixture-like conjugation parameter structure encoding the joint (Y, K) latent space.

    Parameters
    ----------
    mixture_model : Mixture[Observable, IntObservable]
        The mixture of conjugated harmoniums manifold
    params : Point[Natural, Mixture[Observable, IntObservable]]
        Natural parameters on the mixture

    Returns
    -------
    Point[Natural, Any]
        Conjugation parameters for the mixture of harmoniums
    """
    # Split mixture parameters into component harmonium parameters and categorical prior
    hrm_params_0, int_mat = mixture_likelihood.split_params(lkl_params)
    int_man = mixture_likelihood.snd_man
    int_comps = int_man.to_columns(int_mat)

    hrm_man = mixture_likelihood.cod_emb.amb_man
    lkl_params_0 = hrm_man.likelihood_function(hrm_params_0)
    rho_y = hrm_man.conjugation_parameters(lkl_params_0)
    obs_params_0, _, _ = hrm_man.split_params(hrm_params_0)
    lp_0 = hrm_man.obs_man.log_partition_function(obs_params_0)

    def compute_rho_ks(comp_params: Point[Natural, Harm]) -> Array:
        obs_params_k, _, _ = hrm_man.split_params(comp_params)
        adjusted_obs = hrm_man.int_obs_emb.translate(obs_params_0, obs_params_k)
        return hrm_man.obs_man.log_partition_function(adjusted_obs) - lp_0

    def compute_rho_yks(
        comp_params: Point[Natural, Harm],
    ) -> Point[Natural, Harm]:
        adjusted_hrm = hrm_man.int_obs_emb.translate(hrm_params_0, comp_params)
        adjusted_lkl = hrm_man.likelihood_function(adjusted_hrm)
        rho_yk0 = hrm_man.conjugation_parameters(adjusted_lkl)
        return rho_yk0 - rho_y

    # rho_z shape: (n_categories - 1,)
    rho_k0 = int_man.col_man.map(compute_rho_ks, int_comps)
    rho_k = mixture_likelihood.dom_man.natural_point(rho_k0)

    rho_yk_comps = int_man.col_man.man_map(compute_rho_yks, int_comps)
    rho_yk = int_man.from_columns(rho_yk_comps)

    return (rho_y, rho_yk, rho_k)
