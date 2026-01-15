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
    ExponentialFamily,
    Harmonium,
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
from ..harmonium.mixture import CompleteMixture, Mixture

### Embeddings ###


@dataclass(frozen=True)
class RowEmbedding[
    Domain: Differentiable,
    Sub: Differentiable,
    Ambient: Differentiable,
](LinearEmbedding[EmbeddedMap[Domain, Sub], EmbeddedMap[Domain, Ambient]]):
    """Embedding that applies a base embedding to each 'row' of a linear map.

    Takes a LinearEmbedding[Sub, Amb] and lifts it to LinearEmbedding[LinearMap[Dom, Sub], LinearMap[Dom, Amb]]
    by applying the base embedding independently to each component (row) of the linear map.

    **Example**: If we have an embedding Sub -> Ambient and n_categories components,
    we can create a tensorized embedding LinearMap[Categorical, Sub] -> LinearMap[Categorical, Ambient]
    that applies the base embedding to each of the n_categories components independently.
    """

    dom_man: Domain
    """The domain manifold (determines number of rows)."""

    cod_emb: LinearEmbedding[Sub, Ambient]
    """The base embedding to apply to each row."""

    @property
    @override
    def sub_man(self) -> EmbeddedMap[Domain, Sub]:
        """LinearMap from Dom to Sub manifold."""
        return AmbientMap(Rectangular(), self.dom_man, self.cod_emb.sub_man)

    @property
    @override
    def amb_man(self) -> EmbeddedMap[Domain, Ambient]:
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
    def project(self, means: Array) -> Array:  # pyright: ignore[reportIncompatibleMethodOverride]
        """Project by applying base projection to each column.

        Parameters
        ----------
        means : Array
            Mean parameters in LinearMap[Dom, Amb].

        Returns
        -------
        Array
            Mean parameters in LinearMap[Dom, Sub].
        """
        # Reshape to matrix
        matrix = self.amb_man.to_matrix(means)

        # Apply base projection to each column using vmap
        projected_matrix = jax.vmap(self.cod_emb.project, in_axes=1, out_axes=1)(matrix)

        # Flatten back
        return projected_matrix.ravel()


@dataclass(frozen=True)
class PosteriorPriorMixtureEmbedding[
    SubPosterior: Differentiable,
    Posterior: Differentiable,
    Prior: Differentiable,
](LinearEmbedding[Mixture[Posterior], CompleteMixture[Prior]]):
    """Embedding between Mixture types.

    Given a LinearEmbedding[Sub, Ambient], creates an embedding
    Mixture[Sub] -> Mixture[Ambient] by applying the base embedding to:
    - The observable bias (base component parameters)
    - Each column of the interaction matrix (component differences)

    The categorical parameters pass through unchanged.

    The sub and ambient mixtures are constructed with the appropriate observable
    embeddings derived from the base embedding.
    """

    pst_prr_emb: LinearEmbedding[Posterior, Prior]
    """The base embedding to lift (also serves as obs_emb for ambient mixture)."""

    sub_pst_emb: LinearEmbedding[SubPosterior, Posterior]
    """Observable embedding for sub mixture."""

    n_categories: int
    """Number of mixture components."""

    @property
    @override
    def sub_man(self) -> Mixture[Posterior]:
        return Mixture(self.n_categories, self.sub_pst_emb)

    @property
    @override
    def amb_man(self) -> CompleteMixture[Prior]:
        # Ambient obs_emb is the composition: sub_obs_emb then base_emb
        return CompleteMixture(self.pst_prr_emb.amb_man, self.n_categories)

    @override
    def embed(self, params: Array) -> Array:  # pyright: ignore[reportIncompatibleMethodOverride]
        """Embed by applying base embedding to observable and interaction columns."""
        obs_params, int_params, lat_params = self.sub_man.split_coords(params)

        # Embed observable
        emb_obs_params = self.pst_prr_emb.embed(obs_params)

        # Embed each column of interaction matrix
        int_matrix = self.sub_man.int_man.to_matrix(int_params)
        emb_int_matrix = jax.vmap(self.pst_prr_emb.embed, in_axes=1, out_axes=1)(
            int_matrix
        )
        emb_int_params = emb_int_matrix.ravel()

        return self.amb_man.join_coords(emb_obs_params, emb_int_params, lat_params)

    @override
    def project(self, means: Array) -> Array:  # pyright: ignore[reportIncompatibleMethodOverride]
        """Project by applying base projection to observable and interaction columns."""
        obs_means, int_means, lat_means = self.amb_man.split_coords(means)

        # Project observable
        prj_obs = self.pst_prr_emb.project(obs_means)

        # Project each column of interaction matrix
        int_matrix = self.amb_man.int_man.to_matrix(int_means)
        prj_int_matrix = jax.vmap(self.pst_prr_emb.project, in_axes=1, out_axes=1)(
            int_matrix
        )
        prj_int = prj_int_matrix.ravel()

        return self.sub_man.join_coords(prj_obs, prj_int, lat_means)


@dataclass(frozen=True)
class SimpleHarmonium[
    Observable: ExponentialFamily,
    Posterior: ExponentialFamily,
](Harmonium[Observable, Posterior]):
    """Concrete harmonium with explicit interaction manifold.

    A minimal implementation of Harmonium that takes an explicit interaction
    manifold. Used as a lightweight wrapper when a concrete Harmonium type
    is needed (e.g., as the sub_man of an embedding).
    """

    _int_man: LinearMap[Posterior, Observable]
    """The interaction manifold."""

    @property
    @override
    def int_man(self) -> LinearMap[Posterior, Observable]:
        return self._int_man


@dataclass(frozen=True)
class HarmoniumEmbedding[
    SubObservable: Differentiable,
    SubPosterior: Differentiable,
    Observable: Differentiable,
    Posterior: Differentiable,
    Prior: Differentiable,
](
    LinearEmbedding[
        Harmonium[SubObservable, SubPosterior],
        DifferentiableConjugated[Observable, Posterior, Prior],
    ]
):
    """Embedding of a triple of submanifolds into a harmonium.

    This embedding connects a submanifold (represented as a Triple of observable,
    interaction, and latent components) to a harmonium by applying three independent
    embeddings to each component.

    **Structure**:

    The embedding is composed of three sub-embeddings:
    - obs_emb: LinearEmbedding[SubObs, Observable] for observable biases
    - int_emb: LinearEmbedding[SubInt, LinearMap[Posterior, Observable]] for interaction
    - lat_emb: LinearEmbedding[SubPst, Posterior] for latent biases

    **Parameters**:

    For a point (sub_obs, sub_int, sub_lat) in the submanifold:
    - Embedding: (obs_emb.embed(sub_obs), int_emb.embed(sub_int), lat_emb.embed(sub_lat))
    - Projection: (obs_emb.project(obs), int_emb.project(int), lat_emb.project(lat))
    """

    obs_emb: LinearEmbedding[SubObservable, Observable]
    """Embedding for observable biases."""

    int_emb: LinearEmbedding[
        LinearMap[SubPosterior, SubObservable], LinearMap[Posterior, Observable]
    ]
    """Embedding for interaction parameters."""

    pst_emb: LinearEmbedding[SubPosterior, Posterior]
    """Embedding for latent biases."""

    hrm_man: DifferentiableConjugated[Observable, Posterior, Prior]
    """The target harmonium manifold."""

    @property
    @override
    def sub_man(self) -> Harmonium[SubObservable, SubPosterior]:
        return SimpleHarmonium(self.int_emb.sub_man)

    @property
    @override
    def amb_man(self) -> DifferentiableConjugated[Observable, Posterior, Prior]:
        return self.hrm_man

    @override
    def project(self, means: Array) -> Array:  # pyright: ignore[reportIncompatibleMethodOverride]
        """Project harmonium parameters to submanifold.

        Args:
            means: Parameters in harmonium space

        Returns:
            Parameters in submanifold space (Triple)
        """
        obs_params, int_params, lat_params = self.amb_man.split_coords(means)
        sub_obs = self.obs_emb.project(obs_params)
        sub_int = self.int_emb.project(int_params)
        sub_lat = self.pst_emb.project(lat_params)
        return self.sub_man.join_coords(sub_obs, sub_int, sub_lat)

    @override
    def embed(self, params: Array) -> Array:  # pyright: ignore[reportIncompatibleMethodOverride]
        """Embed submanifold parameters into harmonium space.

        Args:
            params: Parameters in submanifold space (Triple)

        Returns:
            Parameters in harmonium space
        """
        sub_obs, sub_int, sub_lat = self.sub_man.split_coords(params)
        obs_params = self.obs_emb.embed(sub_obs)
        int_params = self.int_emb.embed(sub_int)
        lat_params = self.pst_emb.embed(sub_lat)
        return self.amb_man.join_coords(obs_params, int_params, lat_params)


### Mixture of Conjugated Harmoniums ###


@dataclass(frozen=True)
class MixtureOfConjugated[
    SubObservable: Differentiable,
    Observable: Differentiable,
    SubPosterior: Differentiable,
    Posterior: Differentiable,
    Prior: Differentiable,
](
    DifferentiableConjugated[
        Observable,
        Mixture[Posterior],
        Mixture[Prior],
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

    hrm_emb: HarmoniumEmbedding[
        SubObservable, SubPosterior, Observable, Posterior, Prior
    ]

    @property
    def bas_hrm(
        self,
    ) -> DifferentiableConjugated[Observable, Posterior, Prior]:
        """Base conjugated harmonium (lower level)."""
        return self.hrm_emb.amb_man

    @property
    @override
    def pst_prr_emb(
        self,
    ) -> LinearEmbedding[Mixture[Posterior], Mixture[Prior]]:
        """Embedding from posterior mixture to prior mixture.

        Composes lat_emb (SubPst -> Posterior) with hrm.pst_prr_emb (Posterior -> Prior)
        and lifts to the mixture structure, applying to:
        - The observable bias (base component Y params)
        - Each column of the interaction matrix (Y param differences)
        """
        return PosteriorPriorMixtureEmbedding(
            self.bas_hrm.pst_prr_emb, self.hrm_emb.pst_emb, self.n_categories
        )

    @property
    def raw_mix_man(
        self,
    ) -> Mixture[DifferentiableConjugated[Observable, Posterior, Prior]]:
        """Mixture manifold over component harmoniums.

        This provides an alternative representation of the mixture model where the observable manifold is the entire base harmonium."""
        return Mixture(self.n_categories, self.hrm_emb)

    @property
    def xy_man(self) -> LinearMap[Mixture[Posterior], Observable]:
        """Base harmonium interaction embedded into mixture structure."""
        return self.bas_hrm.int_man.prepend_embedding(ObservableEmbedding(self.pst_man))  # pyright: ignore[reportReturnType]

    @property
    def xyk_man(self) -> LinearMap[Mixture[Posterior], Observable]:
        """Component-specific interactions embedded into mixture structure."""

        def tensorize_emb[MidPosterior: Differentiable](
            hrm_dom_emb: LinearEmbedding[MidPosterior, Posterior],
        ) -> LinearEmbedding[LinearMap[Categorical, MidPosterior], Mixture[Posterior]]:
            row_emb = RowEmbedding(
                Categorical(self.n_categories),
                hrm_dom_emb,
            )
            int_emb = InteractionEmbedding(self.pst_man)
            return LinearComposedEmbedding(  # pyright: ignore[reportReturnType]
                row_emb,
                int_emb,
            )

        return self.bas_hrm.int_man.map_domain_embedding(tensorize_emb)  # pyright: ignore[reportArgumentType]

    # @property
    # def xyk_man(self) -> LinearMap[Mixture[Posterior], Observable]:
    #     """Component-specific interactions embedded into mixture structure."""
    #
    #     def new_emb[SubLatent: Differentiable](
    #         emb: LinearEmbedding[SubLatent, Posterior],
    #     ) -> LinearEmbedding[LinearMap[Categorical, SubLatent], Mixture[Posterior]]:
    #         emb_emb = LinearComposedEmbedding(self.hrm_emb.int_emb, emb)
    #         tns_emb = RowEmbedding(
    #             Categorical(self.n_categories),
    #             emb_emb,
    #         )
    #         int_emb = InteractionEmbedding(self.pst_man)
    #         return LinearComposedEmbedding(  # pyright: ignore[reportReturnType]
    #             tns_emb,
    #             int_emb,
    #         )
    #
    #     return self.bas_hrm.int_man.map_domain_embedding(new_emb)  # pyright: ignore[reportArgumentType]
    #
    @property
    def xk_man(self) -> LinearMap[Mixture[Posterior], Observable]:
        """Observable bias adjustments per component."""
        return EmbeddedMap(
            Rectangular(),
            PosteriorEmbedding(self.pst_man),
            self.hrm_emb.obs_emb,
        )  # pyright: ignore[reportReturnType]

    @property
    @override
    def int_man(self) -> BlockMap[Mixture[Posterior], Observable]:
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
        x_params, int_params = self.lkl_fun_man.split_coords(lkl_params)

        # Section interaction matrix into blocks: xy, xyk, xk
        xy_params, xyk_params, xk_params = self.int_man.coord_blocks(int_params)

        # Compute base conjugation parameters from component 0
        aff_0 = self.bas_hrm.lkl_fun_man.join_coords(
            x_params,
            xy_params,
        )
        rho_y = self.bas_hrm.conjugation_parameters(aff_0)

        # Handle trivial case: n_categories=1 means no mixture
        if self.n_categories == 1:
            # No additional components, return base conjugation with empty interaction/categorical
            return self.pst_man.join_coords(rho_y, jnp.array([]), jnp.array([]))

        # reshape xk_params into (obs_dim, n_categories-1)
        xk_params = xk_params.reshape(-1, self.n_categories - 1)
        # reshape xyk_params into (obs_dim * latent_dim, n_categories-1)
        xyk_params = xyk_params.reshape(-1, self.n_categories - 1)

        def conjugate_k(x_offset: Array, xy_offset: Array) -> tuple[Array, Array]:
            """Compute conjugation parameters and log partition difference for component k."""
            # Compute affine parameters for component k
            aff_k = self.bas_hrm.lkl_fun_man.join_coords(
                x_params + x_offset,
                xy_params + xy_offset,
            )

            # Compute conjugation parameters for component k
            rho_yi = self.bas_hrm.conjugation_parameters(aff_k)

            # Compute log partition difference
            rho_zi = self.bas_hrm.obs_man.log_partition_function(
                x_params + x_offset
            ) - self.bas_hrm.obs_man.log_partition_function(x_params)

            return rho_yi, rho_zi

        # Vmap over columns - returns tuple of (rho_yk, rho_z)
        rho_yk, rho_z = jax.vmap(conjugate_k, in_axes=(1, 1))(xk_params, xyk_params)

        # Compute differences from base component
        rho_yz = rho_yk - rho_y

        # Join into complete mixture coordinates
        # rho_yz has shape (n_categories-1, lat_dim), need to transpose to (lat_dim, n_categories-1)
        # before flattening to match lat_man's expected interaction layout
        return self.lat_man.join_coords(rho_y, rho_yz.T.ravel(), rho_z)

    def to_mixture_params(self, params: Array) -> Array:
        """Convert MixtureOfConjugated parameters to Mixture[Harmonium] parameters.

        This converts from the shared-base-plus-offsets representation to the
        redundant representation where each component harmonium has independent
        parameters.

        Args:
            params: Parameters in MixtureOfConjugated representation

        Returns:
            Parameters in Mixture[Harmonium] representation (mix_man)
        """
        # Step 1: Split MixtureOfConjugated parameters
        x_params, int_params, yk_harm_params = self.split_coords(params)

        # Step 2: Extract interaction blocks
        xy_params, xyk_params, xk_params = self.int_man.coord_blocks(int_params)

        # Step 3: Extract latent components
        y_params, yk_int_params, k_params = self.lat_man.split_coords(yk_harm_params)

        # Step 4: Build base harmonium (component 0)
        base_hrm_params = self.bas_hrm.join_coords(x_params, xy_params, y_params)

        # Step 5: Build interaction matrix for mix_man
        # All offset blocks have (n_categories-1) columns when viewed as matrices
        n_cols = self.n_categories - 1

        xk_matrix = xk_params.reshape(-1, n_cols)
        xyk_matrix = xyk_params.reshape(-1, n_cols)
        yk_matrix = yk_int_params.reshape(-1, n_cols)

        # Stack to form (hrm_dim, n_cat-1) matrix
        hrm_int_matrix = jnp.vstack([xk_matrix, xyk_matrix, yk_matrix])

        # Flatten to get mix_man interaction params
        mix_int_params = hrm_int_matrix.ravel()

        # Step 6: Join into mix_man parameters
        return self.raw_mix_man.join_coords(base_hrm_params, mix_int_params, k_params)

    def from_mixture_params(self, mix_params: Array) -> Array:
        """Convert Mixture[Harmonium] parameters to MixtureOfConjugated parameters.

        This converts from the redundant representation (independent component
        harmoniums) to the shared-base-plus-offsets representation.

        Args:
            mix_params: Parameters in Mixture[Harmonium] representation (mix_man)

        Returns:
            Parameters in MixtureOfConjugated representation
        """
        # Step 1: Split mix_man parameters
        base_hrm_params, mix_int_params, k_params = self.raw_mix_man.split_coords(
            mix_params
        )

        # Step 2: Extract base harmonium components
        x_params, xy_params, y_params = self.bas_hrm.split_coords(base_hrm_params)

        # Step 3: Split mix_man interaction matrix into blocks
        n_cols = self.n_categories - 1
        hrm_int_matrix = mix_int_params.reshape(-1, n_cols)

        # Split along rows according to harmonium structure
        obs_dim = self.bas_hrm.obs_man.dim
        int_dim = self.bas_hrm.int_man.dim

        xk_matrix = hrm_int_matrix[:obs_dim, :]
        xyk_matrix = hrm_int_matrix[obs_dim : obs_dim + int_dim, :]
        yk_matrix = hrm_int_matrix[obs_dim + int_dim :, :]

        # Flatten back to parameter vectors
        xk_params = xk_matrix.ravel()
        xyk_params = xyk_matrix.ravel()
        yk_int_params = yk_matrix.ravel()

        # Step 4: Join interaction blocks
        int_params = jnp.concatenate([xy_params, xyk_params, xk_params])

        # Step 5: Join latent components
        yk_harm_params = self.lat_man.join_coords(y_params, yk_int_params, k_params)

        # Step 6: Join into MixtureOfConjugated parameters
        return self.join_coords(x_params, int_params, yk_harm_params)

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
        cat_mean = self.lat_man.lat_man.to_mean(cat_natural)
        return self.lat_man.lat_man.to_probs(cat_mean)

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


@dataclass(frozen=True)
class CompleteMixtureOfConjugated[
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

    bas_hrm: DifferentiableConjugated[Observable, Latent, Latent]
    """Base conjugated harmonium (lower level)."""

    @property
    @override
    def lat_man(self) -> CompleteMixture[Latent]:
        """Latent manifold: complete mixture over base latent."""
        return CompleteMixture(self.bas_hrm.pst_man, self.n_categories)

    @property
    def mix_man(
        self,
    ) -> CompleteMixture[DifferentiableConjugated[Observable, Latent, Latent]]:
        """Mixture manifold over component harmoniums.

        This provides an alternative representation of the mixture model where each
        component is a full harmonium, rather than the shared-base-plus-offsets
        representation used by MixtureOfConjugated.
        """
        return CompleteMixture(self.bas_hrm, self.n_categories)

    @property
    def xy_man(self) -> LinearMap[CompleteMixture[Latent], Observable]:
        """Base harmonium interaction embedded into mixture structure."""
        emb = ObservableEmbedding(self.lat_man)
        return self.bas_hrm.int_man.prepend_embedding(emb)  # pyright: ignore[reportReturnType]

    @property
    def xyk_man(self) -> LinearMap[CompleteMixture[Latent], Observable]:
        """Component-specific interactions embedded into mixture structure."""

        def tensorize_emb[SubLatent: Differentiable](
            emb: LinearEmbedding[SubLatent, Latent],
        ) -> LinearEmbedding[
            LinearMap[Categorical, SubLatent], CompleteMixture[Latent]
        ]:
            tns_emb = RowEmbedding(
                Categorical(self.n_categories),
                emb,
            )
            int_emb = InteractionEmbedding(self.lat_man)
            return LinearComposedEmbedding(  # pyright: ignore[reportReturnType]
                tns_emb,
                int_emb,
            )

        return self.bas_hrm.int_man.map_domain_embedding(tensorize_emb)  # pyright: ignore[reportArgumentType]

    @property
    def xk_man(self) -> LinearMap[CompleteMixture[Latent], Observable]:
        """Observable bias adjustments per component."""
        return EmbeddedMap(
            Rectangular(),
            PosteriorEmbedding(self.lat_man),
            IdentityEmbedding(self.bas_hrm.obs_man),
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
        x_params, int_params = self.lkl_fun_man.split_coords(lkl_params)

        # Section interaction matrix into blocks: xy, xyk, xk
        xy_params, xyk_params, xk_params = self.int_man.coord_blocks(int_params)

        # Compute base conjugation parameters from component 0
        aff_0 = self.bas_hrm.lkl_fun_man.join_coords(
            x_params,
            xy_params,
        )
        rho_y = self.bas_hrm.conjugation_parameters(aff_0)

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
            aff_k = self.bas_hrm.lkl_fun_man.join_coords(
                x_params + x_offset,
                xy_params + xy_offset,
            )

            # Compute conjugation parameters for component k
            rho_yi = self.bas_hrm.conjugation_parameters(aff_k)

            # Compute log partition difference
            rho_zi = self.bas_hrm.obs_man.log_partition_function(
                x_params + x_offset
            ) - self.bas_hrm.obs_man.log_partition_function(x_params)

            return rho_yi, rho_zi

        # Vmap over columns - returns tuple of (rho_yk, rho_z)
        rho_yk, rho_z = jax.vmap(conjugate_k, in_axes=(1, 1))(xk_params, xyk_params)

        # Compute differences from base component
        rho_yz = rho_yk - rho_y

        # Join into complete mixture coordinates
        # rho_yz has shape (n_categories-1, lat_dim), need to transpose to (lat_dim, n_categories-1)
        # before flattening to match lat_man's expected interaction layout
        return self.lat_man.join_coords(rho_y, rho_yz.T.ravel(), rho_z)

    def to_mixture_params(self, params: Array) -> Array:
        """Convert MixtureOfConjugated parameters to Mixture[Harmonium] parameters.

        This converts from the shared-base-plus-offsets representation to the
        redundant representation where each component harmonium has independent
        parameters.

        Args:
            params: Parameters in MixtureOfConjugated representation

        Returns:
            Parameters in Mixture[Harmonium] representation (mix_man)
        """
        # Step 1: Split MixtureOfConjugated parameters
        x_params, int_params, yk_harm_params = self.split_coords(params)

        # Step 2: Extract interaction blocks
        xy_params, xyk_params, xk_params = self.int_man.coord_blocks(int_params)

        # Step 3: Extract latent components
        y_params, yk_int_params, k_params = self.lat_man.split_coords(yk_harm_params)

        # Step 4: Build base harmonium (component 0)
        base_hrm_params = self.bas_hrm.join_coords(x_params, xy_params, y_params)

        # Step 5: Build interaction matrix for mix_man
        # All offset blocks have (n_categories-1) columns when viewed as matrices
        n_cols = self.n_categories - 1

        xk_matrix = xk_params.reshape(-1, n_cols)
        xyk_matrix = xyk_params.reshape(-1, n_cols)
        yk_matrix = yk_int_params.reshape(-1, n_cols)

        # Stack to form (hrm_dim, n_cat-1) matrix
        hrm_int_matrix = jnp.vstack([xk_matrix, xyk_matrix, yk_matrix])

        # Flatten to get mix_man interaction params
        mix_int_params = hrm_int_matrix.ravel()

        # Step 6: Join into mix_man parameters
        return self.mix_man.join_coords(base_hrm_params, mix_int_params, k_params)

    def from_mixture_params(self, mix_params: Array) -> Array:
        """Convert Mixture[Harmonium] parameters to MixtureOfConjugated parameters.

        This converts from the redundant representation (independent component
        harmoniums) to the shared-base-plus-offsets representation.

        Args:
            mix_params: Parameters in Mixture[Harmonium] representation (mix_man)

        Returns:
            Parameters in MixtureOfConjugated representation
        """
        # Step 1: Split mix_man parameters
        base_hrm_params, mix_int_params, k_params = self.mix_man.split_coords(
            mix_params
        )

        # Step 2: Extract base harmonium components
        x_params, xy_params, y_params = self.bas_hrm.split_coords(base_hrm_params)

        # Step 3: Split mix_man interaction matrix into blocks
        n_cols = self.n_categories - 1
        hrm_int_matrix = mix_int_params.reshape(-1, n_cols)

        # Split along rows according to harmonium structure
        obs_dim = self.bas_hrm.obs_man.dim
        int_dim = self.bas_hrm.int_man.dim

        xk_matrix = hrm_int_matrix[:obs_dim, :]
        xyk_matrix = hrm_int_matrix[obs_dim : obs_dim + int_dim, :]
        yk_matrix = hrm_int_matrix[obs_dim + int_dim :, :]

        # Flatten back to parameter vectors
        xk_params = xk_matrix.ravel()
        xyk_params = xyk_matrix.ravel()
        yk_int_params = yk_matrix.ravel()

        # Step 4: Join interaction blocks
        int_params = jnp.concatenate([xy_params, xyk_params, xk_params])

        # Step 5: Join latent components
        yk_harm_params = self.lat_man.join_coords(y_params, yk_int_params, k_params)

        # Step 6: Join into MixtureOfConjugated parameters
        return self.join_coords(x_params, int_params, yk_harm_params)

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
        cat_mean = self.lat_man.lat_man.to_mean(cat_natural)
        return self.lat_man.lat_man.to_probs(cat_mean)

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
