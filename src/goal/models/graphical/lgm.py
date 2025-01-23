"""Linear Gaussian Models.

This module implements several variations of linear Gaussian models:

1. Core Structure:

    - Linear relationship between observed and latent variables
    - Gaussian distributions for both observable and latent spaces
    - Conjugate structure enabling exact inference and EM

2. Specialized Variants:

    - LinearModel: Full covariance structure
    - FactorAnalysis: Diagonal observation noise
    - PrincipalComponentAnalysis: Isotropic observation noise

The joint distribution takes the form:

$$p(x,z) \\propto \\exp(\\theta_x \\cdot s_x(x) + \\theta_z \\cdot s_z(z) + x \\cdot \\Theta^m_{xz} \\cdot z)$$

where $x$ represents observations and $z$ represents latent variables.

Implementation Structure:

- Helper functions for matrix operations
- Base LinearGaussianModel class with flexible covariance
- Specialized model classes for constrained covariances

#### Class Hierarchy

![Class Hierarchy](lgm.svg)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Self, override

import jax.numpy as jnp
from jax import Array

from ...geometry import (
    AffineMap,
    AnalyticConjugated,
    Coordinates,
    Diagonal,
    Dual,
    LinearMap,
    MatrixRep,
    Mean,
    Natural,
    Point,
    PositiveDefinite,
    Rectangular,
    Scale,
    Subspace,
    expand_dual,
    reduce_dual,
)
from ..base.gaussian.normal import (
    Covariance,
    Euclidean,
    FullNormal,
    Normal,
    cov_to_lin,
)

### Helper Functions ###


@dataclass(frozen=True)
class NormalLocationSubspace[Rep: PositiveDefinite](Subspace[Normal[Rep], Euclidean]):
    """Subspace relationship for a product manifold $M \\times N$."""

    nor_man: Normal[Rep]

    @property
    @override
    def sup_man(self) -> Normal[Rep]:
        return self.nor_man

    @property
    @override
    def sub_man(self) -> Euclidean:
        return self.nor_man.loc_man

    @override
    def project[C: Coordinates](self, p: Point[C, Normal[Rep]]) -> Point[C, Euclidean]:
        first, _ = self.sup_man.split_params(p)
        return first

    @override
    def translate[C: Coordinates](
        self, p: Point[C, Normal[Rep]], q: Point[C, Euclidean]
    ) -> Point[C, Normal[Rep]]:
        first, second = self.sup_man.split_params(p)
        return self.sup_man.join_params(first + q, second)


### Private Functions ###


def _dual_composition[
    Coords: Coordinates,
    HRep: MatrixRep,
    GRep: MatrixRep,
    FRep: MatrixRep,
](
    h: LinearMap[HRep, Euclidean, Euclidean],
    h_params: Point[Coords, LinearMap[HRep, Euclidean, Euclidean]],
    g: LinearMap[GRep, Euclidean, Euclidean],
    g_params: Point[Dual[Coords], LinearMap[GRep, Euclidean, Euclidean]],
    f: LinearMap[FRep, Euclidean, Euclidean],
    f_params: Point[Coords, LinearMap[FRep, Euclidean, Euclidean]],
) -> tuple[
    LinearMap[MatrixRep, Euclidean, Euclidean],
    Point[Coords, LinearMap[Rectangular, Euclidean, Euclidean]],
]:
    """Three-way matrix multiplication that respects coordinate duality.

    Computes h @ g @ f where g is in dual coordinates.
    """
    # First multiply g @ f
    rep_gf, shape_gf, params_gf = g.rep.matmat(
        g.shape, g_params.array, f.rep, f.shape, f_params.array
    )

    # Then multiply h @ (g @ f)
    rep_hgf, shape_hgf, params_hgf = h.rep.matmat(
        h.shape, h_params.array, rep_gf, shape_gf, params_gf
    )
    out_man = LinearMap(rep_hgf, Euclidean(shape_hgf[1]), Euclidean(shape_hgf[0]))
    return out_man, Point(params_hgf)


def _change_of_basis[
    Coords: Coordinates,
    LinearRep: MatrixRep,
    CovRep: PositiveDefinite,
](
    f: LinearMap[LinearRep, Euclidean, Euclidean],
    f_params: Point[Coords, LinearMap[LinearRep, Euclidean, Euclidean]],
    g: LinearMap[CovRep, Euclidean, Euclidean],
    g_params: Point[Dual[Coords], LinearMap[CovRep, Euclidean, Euclidean]],
) -> tuple[
    Covariance[PositiveDefinite],
    Point[Coords, Covariance[PositiveDefinite]],
]:
    """Linear change of basis transformation.

    Computes f.T @ g @ f where g is in dual coordinates.
    """
    f_trans = f.transpose_manifold()
    f_trans_params = f.transpose(f_params)
    fgf_man, fgf_params = _dual_composition(
        f_trans,
        f_trans_params,
        g,
        g_params,
        f,
        f_params,
    )
    cov_man = Covariance(fgf_man.shape[0], PositiveDefinite)
    out_mat = cov_man.from_dense(fgf_man.to_dense(fgf_params))  # pyright: ignore[reportArgumentType]
    return cov_man, out_mat


@dataclass(frozen=True)
class LinearGaussianModel[
    ObsRep: PositiveDefinite,
](
    AnalyticConjugated[Rectangular, Normal[ObsRep], Euclidean, Euclidean, FullNormal],
):
    """A linear Gaussian model (LGM) implemented as a harmonium with Gaussian latent variables.


    Args:
        obs_dim: Dimension of observable Gaussian
        obs_rep: Representation of observable covariance
        lat_dim: Dimension of latent Gaussian

    The joint distribution takes the form:

    $p(x,z) \\propto e^{\\theta_x \\cdot s_x(x)+ \\theta_z \\cdot s_z(z) + s_x(x) \\cdot \\Theta_{xz} \\cdot s_z(z)}$

    where:

    - $s_x(x) = (x, \\text{tril}(x \\otimes x))$ is the sufficient statistic of the observable Gaussian
    - $s_z(z) = (z, \\text{tril}(z \\otimes z))$ is the sufficient statistic of the latent Gaussian
    - $\\Theta_{xz}$ has the block structure:

        $\\Theta_{xz} = \\begin{pmatrix} \\Theta^m_{xz} & 0 \\\\ 0 & 0 \\end{pmatrix}$

    This constraint ensures the model only captures first-order interactions between x and z,
    making it conjugate with respect to the Gaussian family. The conjugation parameters are:

    - $\\chi = -\\frac{1}{4} \\theta^m_x \\cdot {\\Theta_x^{\\sigma}}^{-1} \\cdot \\theta^m_x - \\frac{1}{2}\\log |-2 \\Theta^{\\sigma}_x|$
    - $\\rho^m = -\\frac{1}{2} \\Theta^m_{zx} \\cdot {\\Theta_x^{\\sigma}}^{-1} \\cdot \\theta^m_x$
    - $P^{\\sigma} = -\\frac{1}{4} \\Theta^m_{zx} \\cdot {\\Theta_x^{\\sigma}}^{-1} \\cdot \\Theta^m_{xz}$

    where:

    - $\\theta_x^{m}$ and $\\Theta_x^{\\sigma}$ are the natural location and shape parameters of the observable Gaussian
    - $\\Theta^m_{zx}$ is the transpose of $\\Theta^m_{xz}$
    - $\\rho^m$ and $P^{\\sigma}$ are conjugation parameters for natural location and shape parameters
    """

    # Fields

    obs_dim: int
    obs_rep: type[ObsRep]
    lat_dim: int

    # Overrides

    @property
    @override
    def int_rep(self) -> Rectangular:
        """Representation of interaction matrix."""
        return Rectangular()

    @property
    @override
    def obs_sub(self) -> NormalLocationSubspace[ObsRep]:
        """Representation of interaction matrix."""
        return NormalLocationSubspace(Normal(self.obs_dim, self.obs_rep))

    @property
    @override
    def lat_sub(self) -> NormalLocationSubspace[PositiveDefinite]:
        """Representation of interaction matrix."""
        return NormalLocationSubspace(Normal(self.lat_dim, PositiveDefinite))

    @override
    def conjugation_parameters(
        self,
        lkl_params: Point[
            Natural, AffineMap[Rectangular, Euclidean, Euclidean, Normal[ObsRep]]
        ],
    ) -> tuple[Array, Point[Natural, FullNormal]]:
        """Compute conjugation parameters for a linear model.

        Args:
            lkl_params: Linear model parameters in natural coordinates

        Returns:
            chi: Log normalization parameter
            rho: Natural parameters of conjugate prior
        """
        # Get parameters
        obs_cov_man = self.obs_man.cov_man
        obs_bias, int_mat = self.lkl_man.split_params(lkl_params)
        obs_loc, obs_prec = self.obs_man.split_location_precision(obs_bias)

        # Intermediate computations
        obs_sigma = obs_cov_man.inverse(obs_prec)
        log_det = obs_cov_man.logdet(obs_sigma)
        obs_mean = obs_cov_man(obs_sigma, expand_dual(obs_loc))

        # Conjugation parameters
        chi = 0.5 * self.obs_man.loc_man.dot(
            obs_mean, expand_dual(obs_cov_man(obs_prec, obs_mean))
        )
        chi += 0.5 * log_det
        rho_mean = self.int_man.transpose_apply(int_mat, obs_mean)
        _, rho_shape = _change_of_basis(
            self.int_man, int_mat, obs_cov_man, cov_to_lin(obs_sigma)
        )
        rho_shape *= -1

        # Join parameters into moment parameters
        rho = self.lat_man.join_location_precision(rho_mean, rho_shape)

        return chi, rho

    @override
    def to_natural_likelihood(
        self, params: Point[Mean, Self]
    ) -> Point[Natural, AffineMap[Rectangular, Euclidean, Euclidean, Normal[ObsRep]]]:
        # Deconstruct parameters
        obs_cov_man = self.obs_man.cov_man
        lat_cov_man = self.lat_man.cov_man
        obs_means, int_means, lat_means = self.split_params(params)
        obs_mean, obs_cov = self.obs_man.split_mean_covariance(obs_means)
        lat_mean, lat_cov = self.lat_man.split_mean_covariance(lat_means)
        int_cov = int_means - self.int_man.outer_product(obs_mean, lat_mean)

        # Construct precisions
        lat_prs = lat_cov_man.inverse(lat_cov)
        int_man_t = self.int_man.transpose_manifold()
        int_cov_t = self.int_man.transpose(int_cov)
        cob_man, cob = _change_of_basis(
            int_man_t, int_cov_t, lat_cov_man, cov_to_lin(lat_prs)
        )
        shaped_cob = obs_cov_man.from_dense(cob_man.to_dense(cob))
        obs_prs = obs_cov_man.inverse(obs_cov - shaped_cob)
        _, int_params = _dual_composition(
            obs_cov_man,
            cov_to_lin(obs_prs),
            self.int_man,
            expand_dual(int_cov),
            lat_cov_man,
            cov_to_lin(lat_prs),
        )
        # Construct observable location params
        obs_loc0 = obs_cov_man(obs_prs, expand_dual(obs_mean))
        obs_loc1 = self.int_man(int_params, expand_dual(lat_mean))
        obs_loc = obs_loc0 - obs_loc1

        # Return natural parameters
        obs_params = self.obs_man.join_location_precision(
            reduce_dual(obs_loc), reduce_dual(obs_prs)
        )
        return self.lkl_man.join_params(obs_params, reduce_dual(int_params))

    def transform_observable_rep[TargetRep: PositiveDefinite](
        self,
        trg_man: LinearGaussianModel[TargetRep],
        p: Point[Natural, Self],
    ) -> Point[Natural, LinearGaussianModel[TargetRep]]:
        """Transform observable parameters to target representation.

        This transforms the observable component of the model between different matrix
        representations while preserving the latent and interaction components.
        Acts as an embedding when going to more complex representations and as a
        projection when going to simpler ones.
        """
        obs_params, int_params, lat_params = self.split_params(p)
        trn_obs_params = self.obs_man.embed_rep(trg_man.obs_man, obs_params)

        return trg_man.join_params(trn_obs_params, int_params, lat_params)

    def to_normal(self, p: Point[Natural, Self]) -> Point[Natural, FullNormal]:
        """Convert a linear model to a normal model."""
        new_man: LinearGaussianModel[PositiveDefinite] = LinearGaussianModel(
            obs_dim=self.obs_man.data_dim,
            obs_rep=PositiveDefinite,
            lat_dim=self.lat_man.data_dim,
        )
        p_embedded = self.transform_observable_rep(new_man, p)
        obs_params, int_params, lat_params = new_man.split_params(p_embedded)
        obs_loc, obs_prs = new_man.obs_man.split_location_precision(obs_params)
        lat_loc, lat_prs = new_man.lat_man.split_location_precision(lat_params)
        nor_loc: Point[Natural, Euclidean] = Point(
            jnp.concatenate([obs_loc.array, lat_loc.array])
        )
        obs_prs_array = new_man.obs_man.cov_man.to_dense(obs_prs)
        lat_prs_array = new_man.lat_man.cov_man.to_dense(lat_prs)
        int_array = -self.int_man.to_dense(int_params)
        joint_shape_array = jnp.block(
            [[obs_prs_array, int_array], [int_array.T, lat_prs_array]]
        )
        nor_man = Normal(self.data_dim, PositiveDefinite)
        return nor_man.join_location_precision(
            nor_loc, nor_man.cov_man.from_dense(joint_shape_array)
        )


# @dataclass(frozen=True)
# class LinearModel(LinearGaussianModel[PositiveDefinite]):
#     """A linear model with Gaussian latent variables."""
#
#     def __init__(self, obs_dim: int, lat_dim: int):
#         super().__init__(obs_dim, PositiveDefinite, lat_dim)
#
#
@dataclass(frozen=True)
class FactorAnalysis(LinearGaussianModel[Diagonal]):
    """A factor analysis model with Gaussian latent variables."""

    def __init__(self, obs_dim: int, lat_dim: int):
        super().__init__(obs_dim, Diagonal, lat_dim)

    @override
    def expectation_maximization(
        self, params: Point[Natural, Self], xs: Array
    ) -> Point[Natural, Self]:
        """Perform a single iteration of the EM algorithm. Without further constraints the latent Normal of FA is not identifiable, and so we hold it fixed at standard normal."""
        # E-step: Compute expectations
        q = self.expectation_step(params, xs)
        p1 = self.to_natural(q)
        lkl_params = self.likelihood_function(p1)
        z = self.lat_man.to_natural(self.lat_man.standard_normal())
        return self.join_conjugated(lkl_params, z)

    def to_natural_from_loadings(
        self,
        loadings: Array,
        means: Array,
        diags: Array,
    ) -> Point[Natural, Self]:
        """Convert source parameters to natural parameters for factor analysis.

        For a factor analysis model with generative form:
        $$p(x \\mid z) = \\mathcal{N}(Az + \\mu, D)$$

        where:
        - $A$ is the loading matrix
        - $\\mu$ is the mean vector
        - $D$ is a diagonal covariance matrix
        - $z \\sim \\mathcal{N}(0, I)$

        This method converts these source parameters to natural parameters
        $$p(x \\mid z) \\propto \\exp(\\theta_x \\cdot s_x(x) + s_x(x) \\cdot \\Theta_{xz} \\cdot z)$$

        where:
        - $\\theta_x$ are observable natural parameters derived from $\\mu$ and $D$
        - $\\Theta_{xz} = D^{-1}A$ is the precision-scaled loading matrix

        Args:
            loadings: Loading matrix $A$ of shape (obs_dim, lat_dim)
            means: Mean vector $\\mu$ for observable variables
            diags: Diagonal entries of $D$

        Returns:
            Natural parameters for factor analysis model
        """
        # Initialize interaction matrix scaled by precision
        with self.obs_man as om:
            mu = om.loc_man.mean_point(means)
            cov = om.cov_man.mean_point(diags)
            obs_params = om.to_natural(om.join_mean_covariance(mu, cov))
            obs_prs = om.split_location_precision(obs_params)[1]
            dns_prs = om.cov_man.to_dense(obs_prs)

        int_mat = self.int_man.from_dense(dns_prs @ loadings)

        # Combine parameters
        lkl_params = self.lkl_man.join_params(obs_params, int_mat)
        z = self.lat_man.to_natural(self.lat_man.standard_normal())
        return self.join_conjugated(lkl_params, z)


@dataclass(frozen=True)
class PrincipalComponentAnalysis(LinearGaussianModel[Scale]):
    """A principal component analysis model with Gaussian latent variables."""

    def __init__(self, obs_dim: int, lat_dim: int):
        super().__init__(obs_dim, Scale, lat_dim)

    @override
    def expectation_maximization(
        self, params: Point[Natural, Self], xs: Array
    ) -> Point[Natural, Self]:
        """Perform a single iteration of the EM algorithm. Without further constraints the latent Normal of PCA is not identifiable, and so we hold it fixed at standard normal."""
        # E-step: Compute expectations
        q = self.expectation_step(params, xs)
        p1 = self.to_natural(q)
        lkl_params = self.likelihood_function(p1)
        z = self.lat_man.to_natural(self.lat_man.standard_normal())
        return self.join_conjugated(lkl_params, z)
