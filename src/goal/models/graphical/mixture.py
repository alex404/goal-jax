"""Mixtures over harmoniums, and the mixture of factor analyzers.

A :class:`CompleteMixtureOfHarmoniums` puts a categorical node above a base harmonium, so
the graph gains a third node $k$ and three crossing cliques: $(x,y)$ from the base
interaction, $(x,y,k)$ for the component-specific coupling, and $(x,k)$ for per-component
observable shifts. That fork --- $y$ and $k$ both adjacent to $x$ --- is why the graph has
depth two rather than being a chain.

The same coordinates read two ways. :meth:`to_mixture_coords` re-roots the layout at $k$
via :meth:`~goal.geometry.manifold.clique.LinearCliques.cut`, turning the model into a
:class:`~goal.models.harmonium.mixture.CompleteMixture` whose observable is the base
harmonium; :meth:`from_mixture_coords` inverts it. Conjugation and whitening are written
against whichever view makes them a one-liner.
"""

from __future__ import annotations

from abc import ABC
from dataclasses import dataclass
from typing import override

import jax
import jax.numpy as jnp
from jax import Array

from ...geometry import (
    Analytic,
    AnalyticConjugated,
    CliqueCut,
    Diagonal,
    Differentiable,
    DifferentiableConjugated,
    Harmonium,
    IdentityEmbedding,
    LinearClique,
    LinearEmbedding,
    Rectangular,
    SymmetricConjugated,
)
from ..base.categorical import Categorical
from ..base.gaussian.normal import FullNormal, Normal
from ..harmonium.lgm import NormalAnalyticLGM
from ..harmonium.mixture import AnalyticMixture, CompleteMixture

# Embeddings


@dataclass(frozen=True)
class CompleteMixtureEmbedding[Sub: Differentiable, Ambient: Differentiable](
    LinearEmbedding[CompleteMixture[Sub], CompleteMixture[Ambient]]
):
    """Embedding that lifts a component embedding to work on CompleteMixture structures.

    Given an embedding Sub -> Ambient, this creates an embedding
    CompleteMixture[Sub] -> CompleteMixture[Ambient] by applying the base embedding to:
    1. The observable component (first component parameters)
    2. Each column of the interaction matrix (component-specific offsets)

    The categorical parameters remain unchanged since they live in the same space.
    """

    n_categories: int
    """Number of mixture components."""

    component_emb: LinearEmbedding[Sub, Ambient]
    """The base embedding to apply to each component."""

    @property
    @override
    def sub_man(self) -> CompleteMixture[Sub]:
        """CompleteMixture over the sub-manifold."""
        return CompleteMixture(self.component_emb.sub_man, self.n_categories)

    @property
    @override
    def amb_man(self) -> CompleteMixture[Ambient]:
        """CompleteMixture over the ambient manifold."""
        return CompleteMixture(self.component_emb.amb_man, self.n_categories)

    @override
    def embed(self, coords: Array) -> Array:
        """Embed by applying base embedding to observable and interaction components."""
        obs_params, int_params, cat_params = self.sub_man.split_level(coords)

        # Embed observable component
        emb_obs = self.component_emb.embed(obs_params)

        # Embed each column of the interaction matrix
        if self.n_categories > 1:
            # Reshape to matrix: (sub_obs_dim, n_categories-1)
            int_matrix = self.sub_man.int_man.to_matrix(int_params)
            # Apply embedding to each column
            emb_int_matrix = jax.vmap(self.component_emb.embed, in_axes=1, out_axes=1)(
                int_matrix
            )
            # Flatten back using ambient int_man's convention
            emb_int = emb_int_matrix.ravel()
        else:
            emb_int = jnp.array([])

        # Categorical params unchanged
        return self.amb_man.join_level(emb_obs, emb_int, cat_params)

    @override
    def project(self, coords: Array) -> Array:
        """Project by applying base projection to observable and interaction components."""
        obs_means, int_means, cat_means = self.amb_man.split_level(coords)

        # Project observable component
        proj_obs = self.component_emb.project(obs_means)

        # Project each column of the interaction matrix
        if self.n_categories > 1:
            # Reshape to matrix: (amb_obs_dim, n_categories-1)
            int_matrix = self.amb_man.int_man.to_matrix(int_means)
            # Apply projection to each column
            proj_int_matrix = jax.vmap(
                self.component_emb.project, in_axes=1, out_axes=1
            )(int_matrix)
            # Flatten back using sub int_man's convention
            proj_int = proj_int_matrix.ravel()
        else:
            proj_int = jnp.array([])

        # Categorical params unchanged
        return self.sub_man.join_level(proj_obs, proj_int, cat_means)

    @override
    def translate(self, p_coords: Array, q_coords: Array) -> Array:
        """Translate by embedding and adding componentwise."""
        return p_coords + self.embed(q_coords)


# Mixture of Harmoniums


_CATEGORY_NODE = 2
"""Index of the category node $k$ in the $x - y - k$ graph declared below."""


@dataclass(frozen=True)
class CompleteMixtureOfHarmoniums[
    Observable: Differentiable,
    Posterior: Differentiable,
](
    Harmonium[Observable, CompleteMixture[Posterior]],
    ABC,
):
    """Harmonium over $x$, $y$, and $k$ with a three-clique interaction.

    Given a base harmonium over (Observable, Posterior), this constructs a harmonium whose
    latent space is ``CompleteMixture[Posterior]`` = $(Y, K)$. The interaction is three
    cliques rather than one, and which nodes each couples is what :attr:`cross_placements`
    declares:

    - $\\theta_{XY}$ on $(x, y)$: the base interaction, shared across components
    - $\\theta_{XYK}$ on $(x, y, k)$: component-specific interaction offsets
    - $\\theta_{XK}$ on $(x, k)$: per-component observable bias shifts

    The graph is a **fork, not a chain**: both $y$ and $k$ are adjacent to $x$ through the
    three-way clique, so the levels come out $(1, 2)$ and the depth is two. This matters
    wherever the deepest level is used --- it holds $y$ as well as $k$.

    This class does NOT require conjugation — it provides the pure harmonium
    structure that can be wrapped in either:
    - DifferentiableConjugated (via CompleteMixtureOfConjugated) for closed-form conjugation
    - VariationalConjugated for learned conjugation parameters

    **Fields**:
        - bas_hrm: Base harmonium (lower level)
        - n_categories: Number of mixture components
    """

    n_categories: int
    """Number of mixture components."""

    bas_hrm: Harmonium[Observable, Posterior]
    """Base harmonium (lower level)."""

    @property
    def bas_pst_man(self) -> CompleteMixture[Posterior]:
        """Complete mixture over base posterior (avoids circular dependency with pst_man)."""
        return CompleteMixture(self.bas_hrm.pst_man, self.n_categories)

    @property
    def _bas_clique(self) -> LinearClique:
        """The base harmonium's single crossing clique --- the form this level extends."""
        return self.bas_hrm.int_man.clique

    @property
    def _cat_emb(self) -> IdentityEmbedding[Categorical]:
        """The category node in full: every component gets its own coefficient."""
        return IdentityEmbedding(Categorical(self.n_categories))

    @property
    def xy_clique(self) -> LinearClique:
        """$\\theta_{XY}$: the base interaction, which couples $x$ to $y$ unchanged."""
        return self._bas_clique

    @property
    @override
    def cross_placements(self) -> tuple[tuple[tuple[int, ...], LinearClique], ...]:
        """The three interaction blocks couple $(x,y)$, $(x,y,k)$, and $(x,k)$.

        Node $0$ is $x$, node $1$ is $y$, node $2$ is $k$. Everything else about the graph
        follows: the three biases come from the partitions, the $(y,k)$ coupling comes from the
        mixture one level up, and the levels come out $(1, 2)$ rather than a chain --- both
        $y$ and $k$ are adjacent to $x$, so the graph has depth two, not three.

        This has to be declared: the three forms share a domain and a codomain, so nothing
        but the model knows which nodes each couples. Each says only what it uses at each
        node --- the arity, the storage order and the reading all follow from the node set,
        which is why the middle one needs no mention of being arity three even though it
        reaches the mixture's joint $(y,k)$ clique rather than either node alone.

        $\\theta_{XY}$ is the base harmonium's own form, borrowed whole. It is the one
        placement here that names its nodes by hand, because a borrowed form arrives
        address-free and this level is asserting where it lands --- which is also why the
        next line can read its embeddings off as $x$ and $y$.
        """
        x_emb, y_emb = self._bas_clique.node_embs
        rect = Rectangular()
        return (
            ((0, 1), self.xy_clique),
            self.cross_placement(rect, {0: x_emb, 1: y_emb, 2: self._cat_emb}),
            self.cross_placement(
                rect,
                {0: IdentityEmbedding(self.bas_hrm.obs_man), 2: self._cat_emb},
            ),
        )

    @property
    @override
    def obs_man(self) -> Observable:
        """The base harmonium's observable."""
        return self.bas_hrm.obs_man

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
        # Compute posterior harmonium parameters (in pst_man space)
        posterior = self.posterior_at(params, x)

        return self.bas_pst_man.prior(posterior)

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
        cat_mean = self.bas_pst_man.lat_man.to_mean(cat_natural)
        return self.bas_pst_man.lat_man.to_probs(cat_mean)

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

    @property
    def mix_man(
        self,
    ) -> CompleteMixture[Harmonium[Observable, Posterior]]:  # pyright: ignore[reportInvalidTypeArguments]
        """Mixture manifold over component harmoniums.

        This provides an alternative representation of the mixture model where each
        component is a full harmonium, rather than the shared-base-plus-offsets
        representation used by CompleteMixtureOfHarmoniums.
        """
        return CompleteMixture(self.bas_hrm, self.n_categories)  # pyright: ignore[reportArgumentType]

    @property
    def mix_cut(self) -> CliqueCut:
        """Re-view of the graph across the cut $\\{x, y\\} \\mid \\{k\\}$.

        The mixture layout is this model's own layout under a different bipartition: rather
        than splitting off the root node $x$, it splits off the category node $k$ and
        gathers everything coupling to it --- $\\theta_{XK}$, $\\theta_{XYK}$,
        $\\theta_{YK}$ --- into one matrix whose rows follow the base harmonium's clique
        order. That is exactly ``mix_man``'s three partitions.

        Note this is *not* ``levels[-1]``: the graph has depth two, so its deepest level
        holds both $y$ and $k$, and cutting there would take $y$ with it.
        """
        return self.cut(_CATEGORY_NODE)

    def to_mixture_coords(self, coords: Array) -> Array:
        """Repack coordinates from this model's layout to ``mix_man``'s.

        Works identically in natural and mean coordinates: a ``CliqueCut`` is a block
        permutation, which is the same linear operation in both dual spaces.
        """
        return self.mix_man.join_level(*self.mix_cut.project(coords))

    def from_mixture_coords(self, mix_coords: Array) -> Array:
        """Repack coordinates from ``mix_man``'s layout back to this model's."""
        return self.mix_cut.join(*self.mix_man.split_level(mix_coords))


# Mixture of Conjugated Harmoniums


@dataclass(frozen=True)
class CompleteMixtureOfConjugated[
    Observable: Differentiable,
    PstLatent: Differentiable,
    PrrLatent: Differentiable,
](
    CompleteMixtureOfHarmoniums[Observable, PstLatent],
    DifferentiableConjugated[
        Observable,
        CompleteMixture[PstLatent],
        CompleteMixture[PrrLatent],
    ],
):
    """Mixture of conjugated harmoniums.

    Extends CompleteMixtureOfHarmoniums with conjugation structure, requiring a
    DifferentiableConjugated base harmonium with closed-form conjugation_parameters.

    **Conjugation parameters**: Computed using the formula:

    - $\\rho_Y$: Base conjugation from component 0
    - $\\rho^i_Z = \\psi_X(\\theta_X + \\theta^i_{XZ}) - \\psi_X(\\theta_X)$
    - $\\rho^i_{YZ} = \\rho^i_Y - \\rho_Y$

    **Fields**:
        - bas_hrm: Base conjugated harmonium (lower level)
        - n_categories: Number of mixture components
    """

    n_categories: int
    """Number of mixture components."""

    bas_hrm: DifferentiableConjugated[Observable, PstLatent, PrrLatent]
    """Base conjugated harmonium (lower level)."""

    @property
    @override
    def pst_man(self) -> CompleteMixture[PstLatent]:
        """A complete mixture over the base posterior.

        The symmetric subclasses get theirs from ``lat_man`` instead, which is the analytic
        mixture --- same coordinates, more operations.
        """
        return self.bas_pst_man

    @property
    @override
    def pst_prr_emb(
        self,
    ) -> CompleteMixtureEmbedding[PstLatent, PrrLatent]:
        """Embedding of posterior mixture into prior mixture."""
        return CompleteMixtureEmbedding(self.n_categories, self.bas_hrm.pst_prr_emb)

    @override
    def conjugation_parameters(self, lkl_params: Array) -> Array:
        """Compute conjugation parameters for mixture of conjugated harmoniums.

        Implements the theory:
        - $\\rho_Y$: Base conjugation from component 0
        - $\\rho^i_K$: Log partition differences for categorical (K) variable
        - $\\rho^i_{YK}$: Differences in Y conjugation across components

        Returns parameters in CompleteMixture[PrrLatent] space (the prior).
        """
        # Extract base harmonium parameters and interaction matrix
        x_params, int_params = self.lkl_fun_man.split_coords(lkl_params)

        # Section interaction matrix into blocks: xy, xyk, xk
        xy_params, xyk_params, xk_params = self.int_man.coord_blocks(int_params)

        # Compute base conjugation parameters from component 0
        # Note: conjugation_parameters returns params in bas_hrm.prr_man space
        aff_0 = self.bas_hrm.lkl_fun_man.join_coords(
            x_params,
            xy_params,
        )
        rho_y = self.bas_hrm.conjugation_parameters(aff_0)

        # Handle trivial case: n_categories=1 means no mixture
        if self.n_categories == 1:
            # No additional components, return base conjugation with empty interaction/categorical
            return self.prr_man.join_level(rho_y, jnp.array([]), jnp.array([]))

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

            # Compute conjugation parameters for component k (in prr_man space)
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

        # Join into complete mixture coordinates using prr_man (prior manifold)
        # rho_yz has shape (n_categories-1, prr_lat_dim), need to transpose to (prr_lat_dim, n_categories-1)
        # before flattening to match prr_man's expected interaction layout
        return self.prr_man.join_level(rho_y, rho_yz.T.ravel(), rho_z)


@dataclass(frozen=True)
class CompleteMixtureOfSymmetric[
    Observable: Differentiable,
    Latent: Analytic,
](
    CompleteMixtureOfConjugated[Observable, Latent, Latent],
    SymmetricConjugated[Observable, CompleteMixture[Latent]],
):
    """Mixture of symmetric conjugated harmoniums.

    A specialized version of CompleteMixtureOfConjugated for harmoniums where
    the posterior and prior latent manifolds are identical (pst_man == prr_man).
    This is the common case for models like Factor Analysis.

    Provides convenient access to ``lat_man`` (the shared latent manifold).
    """

    n_categories: int
    """Number of mixture components."""

    bas_hrm: AnalyticConjugated[Observable, Latent]
    """Base analytic conjugated harmonium (lower level)."""

    @property
    @override
    def lat_man(self) -> CompleteMixture[Latent]:
        """The shared latent manifold (posterior == prior)."""
        return CompleteMixture(self.bas_hrm.lat_man, self.n_categories)

    @property
    @override
    def pst_man(self) -> CompleteMixture[Latent]:
        """Symmetric: posterior and prior are one manifold, so this is :attr:`lat_man`.

        Overrides the base's ``bas_pst_man``, which is the same coordinates under a class
        with fewer operations --- the analytic subclass needs ``to_natural``.
        """
        return self.lat_man


@dataclass(frozen=True)
class CompleteMixtureOfAnalytic[  # pyright: ignore[reportGeneralTypeIssues,reportIncompatibleMethodOverride]
    Observable: Differentiable,
    Latent: Analytic,
](
    CompleteMixtureOfSymmetric[Observable, Latent],
    AnalyticConjugated[Observable, AnalyticMixture[Latent]],
):
    """Mixture of analytic conjugated harmoniums.

    Extends CompleteMixtureOfSymmetric with full analytic structure via
    ``AnalyticConjugated``, inheriting ``to_natural``, ``negative_entropy``,
    and ``expectation_maximization``. The two bases parameterize
    ``SymmetricConjugated`` with ``CompleteMixture[Latent]`` vs
    ``AnalyticMixture[Latent]`` respectively — invariant generics make this
    a type error, but ``lat_man`` returns ``AnalyticMixture`` at runtime.
    """

    n_categories: int
    """Number of mixture components."""

    bas_hrm: AnalyticConjugated[Observable, Latent]
    """Base analytic conjugated harmonium (lower level)."""

    @property
    @override
    def lat_man(self) -> AnalyticMixture[Latent]:
        """The shared latent manifold as an AnalyticMixture (supports to_natural)."""
        return AnalyticMixture(self.bas_hrm.lat_man, self.n_categories)

    @override
    def to_natural_likelihood(self, means: Array) -> Array:
        """Convert mean parameters to natural likelihood parameters.

        Algorithm:
        1. Convert to mix_man means (per-component base harmonium means)
        2. Apply ``bas_hrm.to_natural_likelihood`` to each component
        3. Use component 0 as anchor; compute offsets for remaining components
        4. Pack into three-block interaction format [xy, xyk, xk]
        """
        mix_means = self.to_mixture_coords(means)
        comp_means, _ = self.mix_man.split_mean_mixture(mix_means)

        # Reshape to (n_categories, bas_hrm.dim) and apply per-component
        comp_means_2d = comp_means.reshape(self.n_categories, -1)
        comp_lkls = jax.vmap(self.bas_hrm.to_natural_likelihood)(comp_means_2d)
        # comp_lkls: (n_categories, bas_hrm.lkl_fun_man.dim)

        # Base component (k=0) sets the anchor likelihood params
        x_params_0, xy_params_0 = self.bas_hrm.lkl_fun_man.split_coords(comp_lkls[0])

        if self.n_categories == 1:
            return self.lkl_fun_man.join_coords(x_params_0, xy_params_0)

        # Compute offsets for remaining components (k=1..K-1)
        def compute_offset(lkl_k: Array) -> Array:
            x_k, xy_k = self.bas_hrm.lkl_fun_man.split_coords(lkl_k)
            return jnp.concatenate([x_k - x_params_0, xy_k - xy_params_0])

        rest_offsets = jax.vmap(compute_offset)(comp_lkls[1:])
        # rest_offsets: (n_cols, obs_dim + int_dim)

        obs_dim = self.bas_hrm.obs_man.dim
        # Transpose to (obs_dim, n_cols) before ravel to match the xk clique storage
        xk_coords = rest_offsets[:, :obs_dim].T.ravel()
        xyk_coords = rest_offsets[:, obs_dim:].T.ravel()

        int_params = jnp.concatenate([xy_params_0, xyk_coords, xk_coords])
        return self.lkl_fun_man.join_coords(x_params_0, int_params)


@dataclass(frozen=True)
class MixtureOfFactorAnalyzers(
    CompleteMixtureOfAnalytic[Normal[Diagonal], FullNormal],
):
    """Mixture of Factor Analyzers: K factor analysis components with shared structure.

    Each component is a FactorAnalysis model (diagonal observable noise,
    full-covariance Gaussian latent prior). Provides whiten_prior to reparameterize
    all components to standard-normal latent priors while preserving p(x), and
    expectation_maximization for closed-form EM with whitening.
    """

    n_categories: int
    """Number of mixture components."""

    bas_hrm: NormalAnalyticLGM[Diagonal]
    """Base factor analysis harmonium (lower level)."""

    @override
    def expectation_maximization(self, params: Array, xs: Array) -> Array:
        """Perform a single EM iteration with latent-prior whitening.

        MFA has the same latent-space non-identifiability as FA/PCA. After the
        E-step, whiten the latent prior in mean coordinates before mapping back
        to natural coordinates.
        """
        q = self.mean_posterior_statistics(params, xs)
        return self.to_natural(self.whiten_prior(q))

    def whiten_prior(self, means: Array) -> Array:
        """Whiten MFA priors in CompleteMixtureOfHarmoniums mean coordinates.

        The input/output are mean parameters on ``MixtureOfFactorAnalyzers`` itself.
        Internally, we convert to ``mix_man`` means, whiten each FA component, then
        convert back to the three-block mean layout.
        """
        mix_means = self.to_mixture_coords(means)
        comp_means, cat_means = self.mix_man.split_mean_mixture(mix_means)
        whitened_comp_means = self.mix_man.cmp_man.map(
            self.bas_hrm.whiten_prior,
            comp_means,
            flatten=True,
        )
        whitened_mix_means = self.mix_man.join_mean_mixture(
            whitened_comp_means, cat_means
        )
        return self.from_mixture_coords(whitened_mix_means)
