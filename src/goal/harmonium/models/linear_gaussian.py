"""Core definitions for harmonium models - product exponential families combining observable and latent variables."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Self

import jax.numpy as jnp
from jax import Array

from ...exponential_family import (
    LocationSubspace,
    Mean,
    Natural,
)
from ...exponential_family.distributions import (
    Covariance,
    Euclidean,
    FullNormal,
    Normal,
)
from ...manifold import Coordinates, Dual, Point, expand_dual
from ...transforms import (
    AffineMap,
    LinearMap,
    MatrixRep,
    PositiveDefinite,
    Rectangular,
)
from ..harmonium import BackwardConjugated

### Helper Functions ###


### Private Functions ###


# TODO: Refactor helper functions with proper co/contravariance
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
    LinearMap[PositiveDefinite, Euclidean, Euclidean],
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
    out_man = Covariance(f.dom_man.dim, PositiveDefinite)
    # params_hgf is is going to be square, but we know it can be positive definite
    out_mat = out_man.from_dense(rep_hgf.to_dense(shape_hgf, params_hgf))
    return out_man, out_mat


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
    fgf_man, fgf = _dual_composition(
        f_trans,
        f_trans_params,
        g,
        g_params,
        f,
        f_params,
    )
    cov_man = Covariance(fgf_man.shape[0], PositiveDefinite)
    return cov_man, Point(fgf.params)


@dataclass(frozen=True)
class LinearGaussianModel[
    ObsRep: PositiveDefinite,
](
    BackwardConjugated[Rectangular, Normal[ObsRep], Euclidean, FullNormal, Euclidean],
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

    def __init__(self, obs_dim: int, obs_rep: type[ObsRep], lat_dim: int):
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
            Natural, AffineMap[Rectangular, Euclidean, Normal[ObsRep], Euclidean]
        ],
    ) -> tuple[Array, Point[Natural, FullNormal]]:
        """Compute conjugation parameters for a linear model.

        Args:
            lkl_params: Linear model parameters in natural coordinates

        Returns:
            chi: Log normalization parameter
            rho: Natural parameters of conjugate prior
        """
        obs_cov_man = self.obs_man.cov_man
        # Split affine map into parts
        obs_bias, int_mat = self.lkl_man.split_params(lkl_params)

        # Get Gaussian parameters
        obs_loc, obs_prec = self.obs_man.split_params(obs_bias)

        # Compute inverse and log determinant
        obs_sigma = obs_cov_man.inverse(obs_prec)
        log_det = obs_cov_man.logdet(obs_sigma)

        # Intermediate computations
        obs_mean = obs_cov_man(obs_sigma, expand_dual(obs_loc))
        chi = (
            self.obs_man.loc_man.dot(obs_mean, obs_cov_man(obs_prec, obs_mean))
            + log_det
        )
        chi *= 0.5

        # Compute rho parameters
        rho_mean = self.int_man.transpose_apply(int_mat, obs_mean)
        _, rho_shape = _change_of_basis(self.int_man, int_mat, obs_cov_man, obs_sigma)
        rho_shape *= 0.5

        # Join parameters into moment parameters
        rho = self.lat_man.join_params(rho_mean, rho_shape)

        return chi, rho

    def to_natural_likelihood(
        self, p: Point[Mean, Self]
    ) -> Point[Natural, AffineMap[Rectangular, Euclidean, Normal[ObsRep], Euclidean]]:
        obs_cov_man = self.obs_man.cov_man
        lat_cov_man = self.lat_man.cov_man
        (obs_means, lat_means, int_means) = self.split_params(p)
        obs_mean, obs_cov = self.obs_man.to_mean_and_covariance(obs_means)
        lat_mean, lat_cov = self.lat_man.to_mean_and_covariance(lat_means)
        int_cov = int_means - self.int_man.outer_product(lat_mean, obs_mean)
        lat_prs = lat_cov_man.inverse(lat_cov)
        cob_man, cob = _change_of_basis(self.int_man, int_cov, lat_cov_man, lat_prs)
        obs_prs = obs_cov_man.inverse(
            obs_cov - obs_cov_man.from_dense(cob_man.to_dense(cob))
        )
        _, int_params = _dual_composition(
            obs_cov_man,
            obs_prs,
            self.int_man,
            expand_dual(int_cov),
            lat_cov_man,
            lat_prs,
        )
        obs_loc = obs_cov_man(obs_prs, expand_dual(obs_mean)) - self.int_man(
            int_params, expand_dual(lat_mean)
        )
        obs_params = self.obs_man.join_natural_params(obs_loc, obs_prs)
        return self.lkl_man.join_params(obs_params, int_params)

    def to_normal[C: Coordinates](self, p: Point[C, Self]) -> Point[C, FullNormal]:
        obs_params, lat_params, int_params = self.split_params(p)
        obs_loc, obs_shape = self.obs_man.split_params(obs_params)
        lat_loc, lat_shape = self.lat_man.split_params(lat_params)
        joint_loc: Point[C, Euclidean] = Point(
            jnp.concatenate([obs_loc.params, lat_loc.params])
        )
        obs_shape_array = self.obs_man.cov_man.to_dense(obs_shape)
        lat_shape_array = self.lat_man.cov_man.to_dense(lat_shape)
        int_array = self.int_man.to_dense(int_params)
        joint_shape_array = jnp.block(
            [[obs_shape_array, int_array], [int_array.T, lat_shape_array]]
        )
        nor_man = Normal(self.data_dim)
        return nor_man.join_params(
            joint_loc, nor_man.cov_man.from_dense(joint_shape_array)
        )
