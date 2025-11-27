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
from typing import override

from jax import Array

from ...geometry import (
    Differentiable,
    DifferentiableConjugated,
    ExponentialFamily,
    LinearEmbedding,
    MatrixRep,
    Rectangular,
    SymmetricConjugated,
)
from ..harmonium.mixture import CompleteMixture

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
        self,
        params: Array,
    ) -> Array:
        """Embed by applying the component embedding to each component in Natural coordinates.

        Decomposes the mixture into component natural parameters and prior using
        `split_natural_mixture`. Maps the component embedding over the product of components.
        Reconstructs the mixture using `join_natural_mixture`.

        Parameters
        ----------
        params : Array
            Natural parameters in sub-manifold.

        Returns
        -------
        Array
            Natural parameters in ambient manifold.
        """
        components, prior = self.sub_man.split_natural_mixture(params)
        emb_components = self.sub_man.cmp_man.map(self.cmp_emb.embed, components)
        return self.amb_man.join_natural_mixture(emb_components, prior)

    @override
    def project(  # pyright: ignore[reportIncompatibleMethodOverride]
        self,
        mean_params: Array,
    ) -> Array:
        """Project by applying the component projection to each component in Mean coordinates.

        Decomposes the mixture into component means and weights using `split_mean_mixture`.
        Maps the component projection over the product of components. Reconstructs the
        mixture using `join_mean_mixture`.

        Parameters
        ----------
        mean_params : Array
            Mean parameters in ambient manifold.

        Returns
        -------
        Array
            Mean parameters in sub-manifold.
        """
        components, weights = self.amb_man.split_mean_mixture(mean_params)
        prj_components = self.amb_man.cmp_man.map(self.cmp_emb.project, components)
        return self.sub_man.join_mean_mixture(prj_components, weights)


### Mixture of Conjugated Harmoniums ###


class LatentMixtureOfConjugated[
    IntRep: MatrixRep,
    Observable: Differentiable,
    IntObservable: ExponentialFamily,
    IntLatent: ExponentialFamily,
    Posterior: Differentiable,
    Prior: Differentiable,
](
    CompleteMixture[
        DifferentiableConjugated[Observable, IntObservable, IntLatent, Posterior, Prior]
    ],
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

    @property
    def mix_pst_man(
        self,
    ) -> CompleteMixture[Posterior]:
        """Latent manifold representing the mixture component assignments."""
        return CompleteMixture(
            self.obs_man.pst_man,
            self.n_categories,
        )

    @property
    def mix_prr_man(
        self,
    ) -> CompleteMixture[Prior]:
        """Latent manifold representing the mixture component assignments."""
        return CompleteMixture(
            self.obs_man.prr_man,
            self.n_categories,
        )

    def mix_posterior_at(
        self,
        params: Array,
        x: Array,
    ) -> Array:
        """Compute natural parameters of posterior distribution $p(z \\mid x)$.

        Given an observation $x$ with sufficient statistics $s(x)$, computes natural
        parameters $\\theta_Z$ of the conditional distribution:

        $$p(z \\mid x) \\propto \\exp(\\theta_Z \\cdot s_Z(z) + s_X(x) \\cdot \\Theta_{XZ} \\cdot s_Z(z))$$

        Parameters
        ----------
        params : Array
            Natural parameters for the mixture of conjugated harmoniums.
        x : Array
            Observation data.

        Returns
        -------
        Array
            Natural parameters of posterior distribution.
        """
        ssx = self.obs_man.obs_man.sufficient_statistic(x)
        prj_ssx = self.obs_man.int_obs_emb.project(ssx)
        obs_params, int_params, lat_params = self.split_coords(params)
        _, obs_int_params, obs_lat_params = self.obs_man.split_coords(obs_params)

        def component_int_fun0(
            comp_params: Array,
        ) -> Array:
            _, _, lat_params = self.obs_man.split_coords(comp_params)
            return lat_params

        def component_int_fun1(
            comp_params: Array,
        ) -> Array:
            _, int_params, _ = self.obs_man.split_coords(comp_params)
            prj_int_params = self.obs_man.int_man.transpose_apply(int_params, prj_ssx)
            return self.obs_man.int_pst_emb.embed(prj_int_params)

        def component_lat_fun(
            comp_params: Array,
        ) -> Array:
            obs_params, _, _ = self.obs_man.split_coords(comp_params)
            return jnp.dot(obs_params, ssx)

        pst_obs_params0 = self.obs_man.int_man.transpose_apply(obs_int_params, prj_ssx)
        pst_obs_params = obs_lat_params + self.obs_man.int_pst_emb.embed(
            pst_obs_params0
        )

        int_cols = self.int_man.to_columns(int_params)

        pst_int_params0 = self.int_man.col_man.map(component_int_fun0, int_cols)
        pst_int_params = self.mix_pst_man.int_man.from_columns(pst_int_params0)
        pst_int_params1 = self.int_man.col_man.map(component_int_fun1, int_cols)
        pst_int_params += self.mix_pst_man.int_man.from_columns(pst_int_params1)

        pst_lat_params0 = self.int_man.col_man.map(component_lat_fun, int_cols)
        pst_lat_params = lat_params + pst_lat_params0

        return self.mix_pst_man.join_coords(
            pst_obs_params, pst_int_params, pst_lat_params
        )

    def mix_conjugation_parameters(
        self,
        lkl_params: Array,
    ) -> Array:
        """Compute conjugation parameters for a mixture of conjugated harmoniums.

        This decomposes the joint conjugation parameters into three components:
        1. Y conjugation biases (latent variable biases from component 0)
        2. K conjugation biases (categorical variable biases from observable component differences)
        3. K-Y interaction columns (differences in conjugation parameters across components)

        The function extracts observable biases from each component harmonium and computes
        a mixture-like conjugation parameter structure encoding the joint (Y, K) latent space.

        Parameters
        ----------
        lkl_params : Array
            Natural parameters for likelihood function in AffineMap structure.

        Returns
        -------
        Array
            Natural parameters for conjugation in CompleteMixture[Prior] space.
        """
        # Split mixture parameters into component harmonium parameters and categorical prior
        hrm_params_0, int_mat = self.lkl_fun_man.split_coords(lkl_params)
        int_man = self.lkl_fun_man.snd_man
        int_cols = int_man.to_columns(int_mat)

        hrm_man = self.lkl_fun_man.cod_emb.amb_man
        lkl_params_0 = hrm_man.likelihood_function(hrm_params_0)
        rho_y = hrm_man.conjugation_parameters(lkl_params_0)
        obs_params_0, _, _ = hrm_man.split_coords(hrm_params_0)
        lp_0 = hrm_man.obs_man.log_partition_function(obs_params_0)

        def compute_rho_ks(
            comp_params: Array,
        ) -> Array:
            obs_params_k, _, _ = hrm_man.split_coords(comp_params)
            adjusted_obs = obs_params_0 + obs_params_k
            return hrm_man.obs_man.log_partition_function(adjusted_obs) - lp_0

        def compute_rho_yks(
            comp_params: Array,
        ) -> Array:
            adjusted_hrm = hrm_params_0 + comp_params
            adjusted_lkl = hrm_man.likelihood_function(adjusted_hrm)
            rho_yk0 = hrm_man.conjugation_parameters(adjusted_lkl)
            return rho_yk0 - rho_y

        # rho_z shape: (n_categories - 1,)
        rho_k0 = int_man.col_man.map(compute_rho_ks, int_cols)
        rho_k = rho_k0

        rho_yk_comps = int_man.col_man.man_map(compute_rho_yks, int_cols)
        rho_yk = self.mix_prr_man.int_man.from_columns(rho_yk_comps)

        return self.mix_prr_man.join_coords(rho_y, rho_yk, rho_k)


class MixtureOfConjugated[
    IntRep: MatrixRep,
    Observable: Differentiable,
    IntObservable: Differentiable,
    IntLatent: Differentiable,
    Latent: Differentiable,
](
    DifferentiableConjugated[Observable, IntObservable, IntLatent, Latent, Latent],
    SymmetricConjugated[
        Observable,
        IntObservable,
        IntLatent,
        Latent,
    ],
):
    @property
    @override
    def int_rep(self) -> Rectangular:
        return 0

    @property
    @override
    def int_obs_emb(
        self,
    ) -> LinearEmbedding[
        CompleteMixture[IntObservable],
        CompleteMixture[Observable],
    ]:
        return 0

    @property
    @override
    def int_pst_emb(
        self,
    ) -> LinearEmbedding[
        CompleteMixture[IntLatent],
        CompleteMixture[Latent],
    ]:
        return 0

    # def mix_log_observable_density(
    #     self,
    #     params: Array,
    #     x: Array,
    # ) -> Array:
    #     hrm_cmps, cat_params = self.split_natural_mixture(params)
    #
    #     def harmonium_log_density_at_x(
    #         hrm_params: Array,
    #     ) -> Array:
    #         return self.obs_man.log_observable_density(hrm_params, x)
    #
    #     lds = self.cmp_man.map(harmonium_log_density_at_x, hrm_cmps)
    #     wghts = self.lat_man.to_probs(self.lat_man.to_mean(cat_params))
    #     return jax.scipy.special.logsumexp(lds + jnp.log(wghts))
    #


### Helper Functions ###
