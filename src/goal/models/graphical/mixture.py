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
import jax.numpy as jnp
from jax import Array

from ...geometry import (
    AmbientMap,
    BlockMap,
    Differentiable,
    DifferentiableConjugated,
    EmbeddedMap,
    IdentityEmbedding,
    InteractionEmbedding,
    LinearComposedEmbedding,
    LinearEmbedding,
    LinearMap,
    ObservableEmbedding,
    PosteriorEmbedding,
    Rectangular,
    SymmetricConjugated,
)
from ..base.categorical import Categorical
from ..harmonium.mixture import CompleteMixture

### Embeddings ###


@dataclass(frozen=True)
class TensorizedEmbedding[
    Domain: Differentiable,
    Sub: Differentiable,
    Amb: Differentiable,
](LinearEmbedding[EmbeddedMap[Domain, Sub], EmbeddedMap[Domain, Amb]]):
    """Tensorized embedding that applies a base embedding to each 'row' of a linear map.

    Takes a LinearEmbedding[Sub, Amb] and lifts it to LinearEmbedding[LinearMap[Dom, Sub], LinearMap[Dom, Amb]]
    by applying the base embedding independently to each component (row) of the linear map.

    This is useful for mixture models where we want to apply an embedding to each mixture component
    without dealing with the complexity of the full mixture structure.

    **Example**: If we have an embedding Latent -> SubLatent and n_categories components,
    we can create a tensorized embedding LinearMap[Categorical, Latent] -> LinearMap[Categorical, SubLatent]
    that applies the base embedding to each of the n_categories components independently.
    """

    dom_man: Domain
    """The domain manifold (determines number of rows)."""

    cod_emb: LinearEmbedding[Sub, Amb]
    """The base embedding to apply to each row."""

    @property
    @override
    def sub_man(self) -> EmbeddedMap[Domain, Sub]:
        """LinearMap from Dom to Sub manifold."""
        return AmbientMap(Rectangular(), self.dom_man, self.cod_emb.sub_man)

    @property
    @override
    def amb_man(self) -> EmbeddedMap[Domain, Amb]:
        """LinearMap from Dom to Amb manifold."""
        return AmbientMap(Rectangular(), self.dom_man, self.cod_emb.amb_man)

    @override
    def embed(self, params: Array) -> Array:  # pyright: ignore[reportIncompatibleMethodOverride]
        """Embed by applying base embedding to each column.

        Parameters
        ----------
        params : Array
            Parameters in LinearMap[Dom, Sub].

        Returns
        -------
        Array
            Parameters in LinearMap[Dom, Amb].
        """
        # Reshape to matrix
        matrix = self.sub_man.to_matrix(params)

        # Apply base embedding to each column using vmap
        embedded_matrix = jax.vmap(self.cod_emb.embed, in_axes=1, out_axes=1)(matrix)

        # Flatten back
        return embedded_matrix.ravel()

    @override
    def project(self, mean_params: Array) -> Array:  # pyright: ignore[reportIncompatibleMethodOverride]
        """Project by applying base projection to each column.

        Parameters
        ----------
        mean_params : Array
            Mean parameters in LinearMap[Dom, Amb].

        Returns
        -------
        Array
            Mean parameters in LinearMap[Dom, Sub].
        """
        # Reshape to matrix
        matrix = self.amb_man.to_matrix(mean_params)

        # Apply base projection to each column using vmap
        projected_matrix = jax.vmap(self.cod_emb.project, in_axes=1, out_axes=1)(matrix)

        # Flatten back
        return projected_matrix.ravel()


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

        def tensorize_emb[SubLatent: Differentiable](
            emb: LinearEmbedding[SubLatent, Latent],
        ) -> LinearEmbedding[
            LinearMap[Categorical, SubLatent], CompleteMixture[Latent]
        ]:
            tns_emb = TensorizedEmbedding(
                Categorical(self.n_categories),
                emb,
            )
            int_emb = InteractionEmbedding(self.lat_man)
            return LinearComposedEmbedding(  # pyright: ignore[reportReturnType]
                tns_emb,
                int_emb,
            )

        return self.hrm.int_man.map_domain_embedding(tensorize_emb)  # pyright: ignore[reportArgumentType]

    @property
    def xk_man(self) -> LinearMap[CompleteMixture[Latent], Observable]:
        """Observable bias adjustments per component."""
        return EmbeddedMap(
            Rectangular(),
            PosteriorEmbedding(self.lat_man),
            IdentityEmbedding(self.hrm.obs_man),
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

        # Section interaction matrix into blocks: xy, xyk, xk
        xy_params, xyk_params, xk_params = self.int_man.coord_blocks(int_mat)

        # Compute base conjugation parameters from component 0
        aff_0 = self.hrm.lkl_fun_man.join_coords(
            x_params,
            xy_params,
        )
        rho_y = self.hrm.conjugation_parameters(aff_0)

        # Handle trivial case: n_categories=1 means no mixture
        if self.n_categories == 1:
            # No additional components, return base conjugation with empty interaction/categorical
            return self.lat_man.join_coords(rho_y, jnp.array([]), jnp.array([]))

        # reshape xk_params into (obs_dim, n_categories-1)
        xk_params = xk_params.reshape(-1, self.n_categories - 1)
        # reshape xyk_params into (obs_dim * latent_dim, n_categories-1)
        xyk_params = xyk_params.reshape(-1, self.n_categories - 1)

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
        # rho_yz has shape (n_categories-1, lat_dim), need to flatten to ((n_categories-1) * lat_dim,)
        return self.lat_man.join_coords(rho_y, rho_yz.ravel(), rho_z)

    def posterior_categorical(self, params: Array, x: Array) -> Array:
        """Compute posterior categorical distribution p(Z|x) in natural coordinates.

        Returns the natural parameters of the categorical distribution over
        mixture components given an observation.

        Args:
            params: Model parameters (natural coordinates)
            x: Observable data point

        Returns:
            Array of shape (n_components-1,) with categorical natural parameters
        """
        # Compute posterior harmonium parameters
        posterior = self.posterior_at(params, x)

        return self.lat_man.prior(posterior)

    def posterior_soft_assignments(self, params: Array, x: Array) -> Array:
        """Compute posterior assignment probabilities p(Z|x).

        Returns the posterior probability distribution over mixture components,
        often called "responsibilities" in EM literature.

        Args:
            params: Model parameters (natural coordinates)
            x: Observable data point

        Returns:
            Array of shape (n_components,) giving p(z_k|x) for each component k
        """
        cat_natural = self.posterior_categorical(params, x)
        return self.lat_man.lat_man.to_probs(cat_natural)

    def posterior_hard_assignment(self, params: Array, x: Array) -> Array:
        """Compute hard assignment to most probable mixture component.

        Returns the index of the mixture component with the highest posterior
        probability given the observation.

        Args:
            params: Model parameters (natural coordinates)
            x: Observable data point

        Returns:
            Integer Array giving index of most probable component
        """
        soft_assignments = self.posterior_soft_assignments(params, x)
        return jnp.argmax(soft_assignments)
