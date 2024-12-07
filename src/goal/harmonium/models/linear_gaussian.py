"""Core definitions for harmonium models - product exponential families combining observable and latent variables."""

from __future__ import annotations

from dataclasses import dataclass

from jax import Array

from ...exponential_family import (
    Backward,
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


def dual_composition[
    Coords: Coordinates,
    LinearRep: MatrixRep,
    CovarianceRep: PositiveDefinite,
](
    h: LinearMap[LinearRep, Euclidean, Euclidean],
    h_params: Point[Coords, LinearMap[LinearRep, Euclidean, Euclidean]],
    g: Covariance[CovarianceRep],
    g_params: Point[Dual[Coords], Covariance[CovarianceRep]],
    f: LinearMap[LinearRep, Euclidean, Euclidean],
    f_params: Point[Coords, LinearMap[LinearRep, Euclidean, Euclidean]],
) -> tuple[
    Covariance[PositiveDefinite],
    Point[Coords, Covariance[PositiveDefinite]],
]:
    """Three-way matrix multiplication that respects coordinate duality.

    Computes h @ g @ f where g is in dual coordinates.
    """
    # First multiply g @ f
    rep_gf, shape_gf, params_gf = g.rep.matmat(
        g.shape, g_params.params, f.rep, f.shape, f_params.params
    )

    # Then multiply h @ (g @ f)
    _, _, params_hgf = h.rep.matmat(
        h.shape, h_params.params, rep_gf, shape_gf, params_gf
    )

    result_map = Covariance(f.dom_man.dim, PositiveDefinite)
    return result_map, Point(params_hgf)


def change_of_basis[
    Coords: Coordinates,
    LinearRep: MatrixRep,
    CovRep: PositiveDefinite,
](
    f: LinearMap[LinearRep, Euclidean, Euclidean],
    f_params: Point[Coords, LinearMap[LinearRep, Euclidean, Euclidean]],
    g: Covariance[CovRep],
    g_params: Point[Dual[Coords], Covariance[CovRep]],
) -> Point[Coords, Covariance[PositiveDefinite]]:
    """Linear change of basis transformation.

    Computes f.T @ g @ f where g is in dual coordinates.
    """
    f_trans = f.transpose_manifold()
    f_trans_params = f.transpose(f_params)
    _, fgf = dual_composition(
        f_trans,
        f_trans_params,
        g,
        g_params,
        f,
        f_params,
    )
    return fgf


@dataclass(frozen=True)
class LinearGaussianModel[Rep: PositiveDefinite, Observable: Backward](
    BackwardConjugated[Rectangular, Normal[Rep], Euclidean, FullNormal, Euclidean],
):
    """A linear Gaussian model (LGM) implemented as a harmonium with Gaussian latent variables.

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

    def conjugation_parameters(
        self,
        lkl_params: Point[
            Natural, AffineMap[Rectangular, Euclidean, Normal[Rep], Euclidean]
        ],
    ) -> tuple[Array, Point[Natural, FullNormal]]:
        """Compute conjugation parameters for a linear model.

        Args:
            aff: Linear model parameters in natural coordinates

        Returns:
            Tuple of:
            - chi: Log normalization parameter
            - rho: Natural parameters of conjugate prior
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
        rho_shape = change_of_basis(self.int_man, int_mat, obs_cov_man, obs_sigma)
        rho_shape *= 0.5

        # Join parameters into moment parameters
        rho = self.lat_man.join_params(rho_mean, rho_shape)

        return chi, rho
