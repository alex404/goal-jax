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

import jax
from jax import Array

from ...geometry import (
    BlockMap,
    Differentiable,
    DifferentiableConjugated,
    EmbeddedMap,
    IdentityEmbedding,
    InteractionEmbedding,
    LinearEmbedding,
    LinearMap,
    Manifold,
    ObservableEmbedding,
    PosteriorEmbedding,
    Rectangular,
    SymmetricConjugated,
)
from ..base.categorical import Categorical
from ..harmonium.mixture import CompleteMixture, Mixture

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
    def embed(  # pyright: ignore[reportIncompatibleMethodOverride]
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


@dataclass(frozen=True)
class MixtureOfConjugated[
    Observable: Differentiable,
    Latent: Differentiable,
](
    DifferentiableConjugated[
        Observable,
        CompleteMixture[Latent],
        CompleteMixture[Latent],
    ],
    SymmetricConjugated[
        Observable,
        CompleteMixture[Latent],
    ],
):
    """Mixture of conjugated harmoniums.

    This class implements the theory of mixing arbitrary conjugated harmoniums,
    where a base harmonium $\\mathcal{H}^G_{XY}$ (observable X, latent Y) is
    composed with a categorical mixture indicator Z to produce a conjugated harmonium with:

    - Observable: X (unchanged from base)
    - Latent: (Y, Z) represented as CompleteMixture[Latent]
    - Interaction: Three-block structure $(\\theta_{XY}, \\theta^0_{XYZ}, \\theta_{XZ})$

    **Mathematical structure**: The sufficient statistics decompose as:

    $$s_{XYZ} = s_X \\oplus S_{X(YZ)} \\oplus s_{YZ}$$

    where $S_{X(YZ)}$ has three blocks encoding base interaction, component-specific
    interactions, and observable bias adjustments.

    **Conjugation parameters**: Computed using the formula:

    - $\\rho_Y$: Base conjugation from component 0
    - $\\rho^i_Z = \\psi_X(\\theta_X + \\theta^i_{XZ}) - \\psi_X(\\theta_X)$
    - $\\rho^i_{YZ} = \\rho^i_Y - \\rho_Y$

    **Fields**:
        - hrm: Base conjugated harmonium (lower level)
        - n_categories: Number of mixture components

    **Structure**:
        - Lower harmonium: Base conjugated harmonium with embeddings from hrm
        - Upper harmonium: CompleteMixture[Latent] with identity embeddings
    """

    n_categories: int
    """Number of mixture components."""

    hrm: DifferentiableConjugated[Observable, Latent, Latent]
    """Base conjugated harmonium (lower level)."""

    @property
    @override
    def lat_man(self) -> CompleteMixture[Latent]:
        """Latent manifold: complete mixture over base latent."""
        return CompleteMixture(self.hrm.pst_man, self.n_categories)

    @property
    def xy_man(self) -> LinearMap[CompleteMixture[Latent], Observable]:
        """Base harmonium interaction embedded into mixture structure."""
        emb = ObservableEmbedding(self.lat_man)
        return self.hrm.int_man.prepend_embedding(emb)  # pyright: ignore[reportReturnType]

    @property
    def xyk_man(self) -> LinearMap[CompleteMixture[Latent], Observable]:
        """Component-specific interactions embedded into mixture structure."""

        def tensorize_emb[SubLatent: Manifold](
            emb: LinearEmbedding[SubLatent, Latent],
        ) -> LinearEmbedding[
            LinearMap[Categorical, Latent], LinearMap[Categorical, Latent]
        ]:
            return IdentityEmbedding(
                Mixture(
                    self.n_categories,
                    emb,
                ).int_man
            )

        xyk_man0 = self.hrm.int_man.map_domain_embedding(tensorize_emb)
        return xyk_man0.prepend_embedding(InteractionEmbedding(self.lat_man))  # pyright: ignore[reportReturnType]

    @property
    def xk_man(self) -> LinearMap[CompleteMixture[Latent], Observable]:
        """Observable bias adjustments per component."""
        return EmbeddedMap(
            Rectangular(),
            PosteriorEmbedding(self.lat_man),
            IdentityEmbedding(self.obs_man),
        )  # pyright: ignore[reportReturnType]

    @property
    @override
    def int_man(self) -> BlockMap[CompleteMixture[Latent], Observable]:
        """Interaction matrix with three blocks: mxy, mxyz, mxz.

        Following the theory:
        - mxy: Base harmonium interaction $(\\theta_{XY})$
        - mxyk: Component-specific interactions $(\\theta^0_{XYZ})$
        - mxk: Observable bias adjustments per component $(\\theta_{XZ})$
        """
        return BlockMap([self.xy_man, self.xyk_man, self.xk_man])

    @override
    def conjugation_parameters(self, lkl_params: Array) -> Array:
        """Compute conjugation parameters for mixture of conjugated harmoniums.

        Implements the theory:
        - $\\rho_Y$: Base conjugation from component 0
        - $\\rho^i_K$: Log partition differences for categorical (K) variable
        - $\\rho^i_{YK}$: Differences in Y conjugation across components

        Returns parameters in CompleteMixture[Latent] space.
        """
        # Extract base harmonium parameters and interaction matrix
        x_params, int_mat = self.lkl_fun_man.split_coords(lkl_params)

        xy_params = int_mat[: self.xy_man.dim]
        xk_params = int_mat[self.xy_man.dim : self.xy_man.dim + self.xk_man.dim]
        xyk_params = int_mat[self.xy_man.dim + self.xk_man.dim :]

        # reshape xk_params into (obs_dim, n_categories-1)
        xk_params = xk_params.reshape(-1, self.n_categories - 1)
        # reshape xyk_params into (obs_dim * latent_dim, n_categories-1)
        xyk_params = xyk_params.reshape(-1, self.n_categories - 1)

        # Compute base conjugation parameters from component 0
        aff_0 = self.hrm.lkl_fun_man.join_coords(
            x_params,
            xy_params,
        )
        rho_y = self.hrm.conjugation_parameters(aff_0)

        def conjugate_k(x_offset: Array, xy_offset: Array) -> tuple[Array, Array]:
            """Compute conjugation parameters and log partition difference for component k."""
            # Compute affine parameters for component k
            aff_k = self.hrm.lkl_fun_man.join_coords(
                x_params + x_offset,
                xy_params + xy_offset,
            )

            # Compute conjugation parameters for component k
            rho_yi = self.hrm.conjugation_parameters(aff_k)

            # Compute log partition difference
            rho_zi = self.hrm.obs_man.log_partition_function(
                x_params + x_offset
            ) - self.hrm.obs_man.log_partition_function(x_params)

            return rho_yi, rho_zi

        # Vmap over columns - returns tuple of (rho_yk, rho_z)
        rho_yk, rho_z = jax.vmap(conjugate_k, in_axes=(1, 1))(xk_params, xyk_params)

        # Compute differences from base component
        rho_yz = rho_yk - rho_y

        # Join into complete mixture coordinates
        return self.lat_man.join_coords(rho_y, rho_yz, rho_z)
