"""Core definitions for harmonium models - product exponential families combining observable and latent variables."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Self

import jax.numpy as jnp
from jax import Array

from ..geometry import (
    AffineMap,
    BackwardConjugated,
    Coordinates,
    Dual,
    LinearMap,
    LocationSubspace,
    MatrixRep,
    Mean,
    Natural,
    Point,
    PositiveDefinite,
    Rectangular,
    expand_dual,
)
from .normal import (
    Covariance,
    Euclidean,
    FullNormal,
    Normal,
)

### Helper Functions ###


### Private Functions ###


def _dual_composition[
    Coords: Coordinates,
    OuterRep: MatrixRep,
    InnerRep: MatrixRep,
](
    h: LinearMap[OuterRep, Euclidean, Euclidean],
    h_params: Point[Coords, LinearMap[OuterRep, Euclidean, Euclidean]],
    g: LinearMap[InnerRep, Euclidean, Euclidean],
    g_params: Point[Dual[Coords], LinearMap[InnerRep, Euclidean, Euclidean]],
    f: LinearMap[OuterRep, Euclidean, Euclidean],
    f_params: Point[Coords, LinearMap[OuterRep, Euclidean, Euclidean]],
) -> tuple[
    LinearMap[MatrixRep, Euclidean, Euclidean],
    Point[Coords, LinearMap[PositiveDefinite, Euclidean, Euclidean]],
]:
    """Three-way matrix multiplication that respects coordinate duality.

    Computes h @ g @ f where g is in dual coordinates.
    """
    # First multiply g @ f
    rep_gf, shape_gf, params_gf = g.rep.matmat(
        g.shape, g_params.params, f.rep, f.shape, f_params.params
    )

    # Then multiply h @ (g @ f)
    rep_hgf, shape_hgf, params_hgf = h.rep.matmat(
        h.shape, h_params.params, rep_gf, shape_gf, params_gf
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
    g: Covariance[CovRep],
    g_params: Point[Dual[Coords], Covariance[CovRep]],
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
    out_mat = cov_man.from_dense(fgf_man.to_dense(fgf_params))
    return cov_man, out_mat


@dataclass(frozen=True)
class LinearGaussianModel[
    Rep: PositiveDefinite,
](
    BackwardConjugated[Rectangular, Normal[Rep], Euclidean, FullNormal, Euclidean],
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

    def __init__(self, obs_dim: int, obs_rep: type[Rep], lat_dim: int):
        """Initialize a linear Gaussian model."""
        obs_man = Normal(obs_dim, obs_rep)
        lat_man = Normal(lat_dim)
        super().__init__(
            obs_man,
            lat_man,
            LinearMap(Rectangular(), lat_man.loc_man, obs_man.loc_man),
            LocationSubspace(obs_man.loc_man, obs_man.cov_man),
            LocationSubspace(lat_man.loc_man, lat_man.cov_man),
        )

    def conjugation_parameters(
        self,
        lkl_params: Point[
            Natural, AffineMap[Rectangular, Euclidean, Normal[Rep], Euclidean]
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
        chi = 0.5 * self.obs_man.loc_man.dot(obs_mean, obs_cov_man(obs_prec, obs_mean))
        chi += 0.5 * log_det
        rho_mean = self.int_man.transpose_apply(int_mat, obs_mean)
        _, rho_shape = _change_of_basis(self.int_man, int_mat, obs_cov_man, obs_sigma)
        rho_shape *= -1

        # Join parameters into moment parameters
        rho = self.lat_man.join_location_precision(rho_mean, rho_shape)

        return chi, rho

    def to_natural_likelihood(
        self, p: Point[Mean, Self]
    ) -> Point[Natural, AffineMap[Rectangular, Euclidean, Normal[Rep], Euclidean]]:
        # Deconstruct parameters
        obs_cov_man = self.obs_man.cov_man
        lat_cov_man = self.lat_man.cov_man
        (obs_means, lat_means, int_means) = self.split_params(p)
        obs_mean, obs_cov = self.obs_man.split_mean_covariance(obs_means)
        lat_mean, lat_cov = self.lat_man.split_mean_covariance(lat_means)
        int_cov = int_means - self.int_man.outer_product(obs_mean, lat_mean)

        # Construct precisions
        lat_prs = lat_cov_man.inverse(lat_cov)
        int_man_t = self.int_man.transpose_manifold()
        int_cov_t = self.int_man.transpose(int_cov)
        cob_man, cob = _change_of_basis(int_man_t, int_cov_t, lat_cov_man, lat_prs)
        shaped_cob = obs_cov_man.from_dense(cob_man.to_dense(cob))
        obs_prs = obs_cov_man.inverse(obs_cov - shaped_cob)
        _, int_params = _dual_composition(
            obs_cov_man,
            obs_prs,
            self.int_man,
            expand_dual(int_cov),
            lat_cov_man,
            lat_prs,
        )
        # Construct observable location params
        obs_loc0 = obs_cov_man(obs_prs, expand_dual(obs_mean))
        obs_loc1 = self.int_man(int_params, expand_dual(lat_mean))
        obs_loc = obs_loc0 - obs_loc1

        # Return natural parameters
        obs_params = self.obs_man.join_location_precision(obs_loc, obs_prs)
        return self.lkl_man.join_params(obs_params, int_params)

    def transform_observable_rep[TargetRep: PositiveDefinite](
        self,
        target_man: LinearGaussianModel[TargetRep],
        p: Point[Natural, Self],
    ) -> Point[Natural, LinearGaussianModel[TargetRep]]:
        """Transform observable parameters to target representation.

        This transforms the observable component of the model between different matrix
        representations while preserving the latent and interaction components.
        Acts as an embedding when going to more complex representations and as a
        projection when going to simpler ones.
        """
        obs_bias, lat_bias, int_mat = self.split_params(p)
        transformed_bias = self.obs_man.transform_rep(target_man.obs_man, obs_bias)
        return target_man.join_params(transformed_bias, lat_bias, int_mat)

    def to_normal(self, p: Point[Natural, Self]) -> Point[Natural, FullNormal]:
        """Convert a linear model to a normal model."""
        new_man: LinearGaussianModel[PositiveDefinite] = LinearGaussianModel(
            obs_dim=self.obs_man.data_dim,
            obs_rep=PositiveDefinite,
            lat_dim=self.lat_man.data_dim,
        )
        p_embedded = self.transform_observable_rep(new_man, p)
        obs_params, lat_params, int_params = new_man.split_params(p_embedded)
        obs_loc, obs_prs = new_man.obs_man.split_location_precision(obs_params)
        lat_loc, lat_prs = new_man.lat_man.split_location_precision(lat_params)
        nor_loc: Point[Natural, Euclidean] = Point(
            jnp.concatenate([obs_loc.params, lat_loc.params])
        )
        obs_prs_array = new_man.obs_man.cov_man.to_dense(obs_prs)
        lat_prs_array = new_man.lat_man.cov_man.to_dense(lat_prs)
        int_array = -self.int_man.to_dense(int_params)
        joint_shape_array = jnp.block(
            [[obs_prs_array, int_array], [int_array.T, lat_prs_array]]
        )
        nor_man = Normal(self.data_dim)
        return nor_man.join_location_precision(
            nor_loc, nor_man.cov_man.from_dense(joint_shape_array)
        )
