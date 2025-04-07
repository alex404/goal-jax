"""This module provides implementations of linear Gaussian models (LGMs), including factor analysis and principal component analysis. LGMs model linear, Gaussian relationships between observable and latent variables. The conjugacy of LGMs enables exact inference and EM."""

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
    DifferentiableConjugated,
    Dual,
    LinearEmbedding,
    LinearMap,
    MatrixRep,
    Mean,
    Natural,
    Point,
    PositiveDefinite,
    Rectangular,
    Scale,
    TupleEmbedding,
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
class NormalLocationEmbedding[Rep: PositiveDefinite](
    TupleEmbedding[Euclidean, Normal[Rep]],
):
    """Embedding of the mean of a normal distribution into a normal with particular covariance structure."""

    nor_man: Normal[Rep]
    """The normal distribution with the specified covariance structure."""

    @property
    @override
    def tup_idx(
        self,
    ) -> int:
        return 0

    @property
    @override
    def amb_man(self) -> Normal[Rep]:
        return self.nor_man

    @property
    @override
    def sub_man(self) -> Euclidean:
        return self.nor_man.loc_man


@dataclass(frozen=True)
class NormalCovarianceEmbedding[SubRep: PositiveDefinite, AmbientRep: PositiveDefinite](
    LinearEmbedding[Normal[SubRep], Normal[AmbientRep]]
):
    """Embedding of a normal distribution with a simpler covariance structure into a more complex one."""

    # Fields

    _amb_man: Normal[AmbientRep]
    """The super-manifold with the more complex covariance structure."""

    _sub_man: Normal[SubRep]
    """The sub-manifold with the simpler covariance structure."""

    def __post_init__(self):
        if not isinstance(self.sub_man.cov_man.rep, self.amb_man.cov_man.rep.__class__):
            raise TypeError(
                f"Sub-manifold rep {self.sub_man.cov_man.rep} must be simpler than super-manifold rep {self.amb_man.cov_man.rep}"
            )

    @property
    @override
    def amb_man(self) -> Normal[AmbientRep]:
        """Super-manifold."""
        return self._amb_man

    @property
    @override
    def sub_man(self) -> Normal[SubRep]:
        """Sub-manifold."""
        return self._sub_man

    @override
    def project(
        self, p: Point[Mean, Normal[AmbientRep]]
    ) -> Point[Mean, Normal[SubRep]]:
        return self.amb_man.project_rep(self.sub_man, p)

    @override
    def embed(
        self, p: Point[Natural, Normal[SubRep]]
    ) -> Point[Natural, Normal[AmbientRep]]:
        return self.sub_man.embed_rep(self.amb_man, p)

    @override
    def translate(
        self, p: Point[Natural, Normal[AmbientRep]], q: Point[Natural, Normal[SubRep]]
    ) -> Point[Natural, Normal[AmbientRep]]:
        embedded_q = self.sub_man.embed_rep(self.amb_man, q)
        return p + embedded_q


@dataclass(frozen=True)
class DifferentiableLinearGaussianModel[
    ObsRep: PositiveDefinite,
    PostRep: PositiveDefinite,
](
    DifferentiableConjugated[
        Rectangular, Normal[ObsRep], Euclidean, Euclidean, Normal[PostRep], FullNormal
    ],
):
    """A linear Gaussian model (LGM) implemented as a harmonium with Gaussian latent variables.

    Linear Gaussian Models represent a joint distribution over observable variables $X$ and latent variables $Z$ where both are Gaussian and the relationship between them is linear. In generative terms, this can be viewed as:

    $$x = Az + \\mu + \\epsilon$$

    where:
        - $z$ is drawn from a multivariate normal (typically a standard normal),
        - $A$ is the loading matrix mapping latent to observable space,
        - $\\mu$ is the observable bias term, and
        - $\\epsilon \\sim \\mathcal{N}(0, \\Sigma)$ is Gaussian noise.

    As a harmonium, the joint distribution takes the form

    $$p(x,z) \\propto \\exp(\\theta_X \\cdot s_X(x) + \\theta_Z \\cdot s_Z(z) + x \\cdot \\Theta^m_{XZ} \\cdot z),$$

    where

    - $s_X(x) = (x, \\text{tril}(x \\otimes x))$ is the sufficient statistic of the observable normal,
    - $s_Z(z) = (z, \\text{tril}(z \\otimes z))$ is the sufficient statistic of the latent normal, and
    - and $\\Theta^m_{XZ}$ are the first-order interaction terms between $X$ and $Z$.

    The conjugation parameters are $\\rho = (\\rho^m, P^{\\sigma})$ where
        - $\\rho^m = -\\frac{1}{2} \\Theta^m_{ZX} \\cdot {\\Theta_X^{\\sigma}}^{-1} \\cdot \\theta^m_X$
        - $P^{\\sigma} = -\\frac{1}{4} \\Theta^m_{ZX} \\cdot {\\Theta_X^{\\sigma}}^{-1} \\cdot \\Theta^m_{XZ}$
    """

    # Fields

    obs_dim: int
    """Dimension of the observable variables."""

    obs_rep: type[ObsRep]
    """Covariance structure of the observable variables."""

    lat_dim: int
    """Dimension of the latent variables."""

    lat_rep: type[PostRep]
    """Covariance structure of the latent variables."""

    ### Methods ###

    def observable_distribution(
        self, params: Point[Natural, Self]
    ) -> tuple[FullNormal, Point[Natural, FullNormal]]:
        """Returns the marginal normal distribution over observable variables."""
        # Build transposed LGM with full covariance observable variables
        transposed_lgm = AnalyticLinearGaussianModel(
            obs_dim=self.lat_dim,  # Original latent becomes observable
            obs_rep=PositiveDefinite,
            lat_dim=self.obs_dim,  # Original observable becomes latent
        )

        # Construct parameters for transposed model
        obs_params, int_params, lat_params = self.split_params(params)
        nor_man = transposed_lgm.lat_man
        obs_params_emb = self.obs_man.embed_rep(nor_man, obs_params)
        lat_params_emb = self.pst_lat_man.embed_rep(transposed_lgm.obs_man, lat_params)

        # Join parameters with interaction matrix transposed
        transposed_params = transposed_lgm.join_params(
            lat_params_emb,  # Original latent becomes observable
            self.int_man.transpose(int_params),
            obs_params_emb,
        )

        # Use harmonium prior to get marginal distribution
        return nor_man, transposed_lgm.prior(transposed_params)

    # Overrides

    @property
    @override
    def int_rep(self) -> Rectangular:
        return Rectangular()

    @property
    @override
    def obs_emb(self) -> NormalLocationEmbedding[ObsRep]:
        return NormalLocationEmbedding(Normal(self.obs_dim, self.obs_rep))

    @property
    @override
    def int_lat_emb(self) -> NormalLocationEmbedding[PostRep]:
        return NormalLocationEmbedding(Normal(self.lat_dim, self.lat_rep))

    @property
    @override
    def pst_lat_emb(self) -> NormalCovarianceEmbedding[PostRep, PositiveDefinite]:
        return NormalCovarianceEmbedding(
            Normal(self.lat_dim, PositiveDefinite),
            Normal(self.lat_dim, self.lat_rep),
        )

    @override
    def conjugation_parameters(
        self,
        lkl_params: Point[
            Natural, AffineMap[Rectangular, Euclidean, Euclidean, Normal[ObsRep]]
        ],
    ) -> Point[Natural, FullNormal]:
        # Get parameters
        obs_cov_man = self.obs_man.cov_man
        obs_bias, int_mat = self.lkl_man.split_params(lkl_params)
        obs_loc, obs_prec = self.obs_man.split_location_precision(obs_bias)

        # Intermediate computations
        obs_sigma = obs_cov_man.inverse(obs_prec)
        obs_mean = obs_cov_man(obs_sigma, expand_dual(obs_loc))

        # Conjugation parameters
        rho_mean = self.int_man.transpose_apply(int_mat, obs_mean)
        _, rho_shape = _change_of_basis(
            self.int_man, int_mat, obs_cov_man, cov_to_lin(obs_sigma)
        )
        rho_shape *= -1

        # Join parameters into moment parameters
        return self.lat_man.join_location_precision(rho_mean, rho_shape)

    def to_normal(self, p: Point[Natural, Self]) -> Point[Natural, FullNormal]:
        """Convert a linear model to a normal model."""
        new_man: DifferentiableLinearGaussianModel[
            PositiveDefinite, PositiveDefinite
        ] = DifferentiableLinearGaussianModel(
            obs_dim=self.obs_man.data_dim,
            obs_rep=PositiveDefinite,
            lat_dim=self.lat_man.data_dim,
            lat_rep=PositiveDefinite,
        )
        obs_params, int_params, lat_params = self.split_params(p)
        emb_obs_params = self.obs_man.embed_rep(new_man.obs_man, obs_params)
        emb_lat_params = self.pst_lat_man.embed_rep(new_man.lat_man, lat_params)

        obs_loc, obs_prs = new_man.obs_man.split_location_precision(emb_obs_params)
        lat_loc, lat_prs = new_man.lat_man.split_location_precision(emb_lat_params)
        nor_man = Normal(self.data_dim, PositiveDefinite)
        nor_loc: Point[Natural, Euclidean] = nor_man.loc_man.point(
            jnp.concatenate([obs_loc.array, lat_loc.array])
        )
        obs_prs_array = new_man.obs_man.cov_man.to_dense(obs_prs)
        lat_prs_array = new_man.lat_man.cov_man.to_dense(lat_prs)
        int_array = -self.int_man.to_dense(int_params)
        joint_shape_array = jnp.block(
            [[obs_prs_array, int_array], [int_array.T, lat_prs_array]]
        )
        return nor_man.join_location_precision(
            nor_loc, nor_man.cov_man.from_dense(joint_shape_array)
        )


@dataclass(frozen=True)
class AnalyticLinearGaussianModel[
    ObsRep: PositiveDefinite,
](
    DifferentiableLinearGaussianModel[ObsRep, PositiveDefinite],
    AnalyticConjugated[Rectangular, Normal[ObsRep], Euclidean, Euclidean, FullNormal],
):
    """Analytic Linear Gaussian Model that extends the differentiable LGM with full analytical tractability, adding conversions between mean and natural coordinates, and a closed-form implementation of EM."""

    def __init__(self, obs_dim: int, obs_rep: type[ObsRep], lat_dim: int):
        super().__init__(obs_dim, obs_rep, lat_dim, PositiveDefinite)

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
        int_man_t = self.int_man.trn_man
        int_cov_t = self.int_man.transpose(int_cov)
        cob_man, cob = _change_of_basis(
            int_man_t, int_cov_t, lat_cov_man, cov_to_lin(lat_prs)
        )
        shaped_cob: Point[Mean, Covariance[ObsRep]] = obs_cov_man.from_dense(
            cob_man.to_dense(cob)
        )
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


@dataclass(frozen=True)
class FactorAnalysis(AnalyticLinearGaussianModel[Diagonal]):
    """A factor analysis model with Gaussian latent variables."""

    def __init__(self, obs_dim: int, lat_dim: int):
        super().__init__(obs_dim, Diagonal, lat_dim)

    @override
    def expectation_maximization(
        self, params: Point[Natural, Self], xs: Array
    ) -> Point[Natural, Self]:
        """Perform a single iteration of the EM algorithm. Without further constraints the latent Normal of FA is not identifiable, and so we hold it fixed at standard normal."""
        # E-step: Compute expectations
        q = self.posterior_statistics(params, xs)
        p1 = self.to_natural(q)
        lkl_params = self.likelihood_function(p1)
        z = self.lat_man.to_natural(self.lat_man.standard_normal())
        return self.join_conjugated(lkl_params, z)

    def from_loadings(
        self,
        loadings: Array,
        means: Array,
        diags: Array,
    ) -> Point[Natural, Self]:
        """Convert standard factor analysis parameters to natural parameters."""
        # Initialize interaction matrix scaled by precision
        with self.obs_man as om:
            mu = om.loc_man.mean_point(means)
            cov = om.cov_man.mean_point(diags)
            obs_params = om.to_natural(om.join_mean_covariance(mu, cov))
            obs_prs = om.split_location_precision(obs_params)[1]
            dns_prs = om.cov_man.to_dense(obs_prs)

        int_mat: Point[Natural, LinearMap[Rectangular, Euclidean, Euclidean]] = (
            self.int_man.from_dense(dns_prs @ loadings)
        )

        # Combine parameters
        lkl_params = self.lkl_man.join_params(obs_params, int_mat)
        z = self.lat_man.to_natural(self.lat_man.standard_normal())
        return self.join_conjugated(lkl_params, z)


@dataclass(frozen=True)
class PrincipalComponentAnalysis(AnalyticLinearGaussianModel[Scale]):
    """A principal component analysis model with Gaussian latent variables."""

    def __init__(self, obs_dim: int, lat_dim: int):
        super().__init__(obs_dim, Scale, lat_dim)

    @override
    def expectation_maximization(
        self, params: Point[Natural, Self], xs: Array
    ) -> Point[Natural, Self]:
        """Perform a single iteration of the EM algorithm. Without further constraints the latent Normal of PCA is not identifiable, and so we hold it fixed at standard normal."""
        # E-step: Compute expectations
        q = self.posterior_statistics(params, xs)
        p1 = self.to_natural(q)
        lkl_params = self.likelihood_function(p1)
        z = self.lat_man.to_natural(self.lat_man.standard_normal())
        return self.join_conjugated(lkl_params, z)


### Helper Functions ###


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
    LinearMap[Rectangular, Euclidean, Euclidean],
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
    dense_params_hgf = rep_hgf.to_dense(shape_hgf, params_hgf)
    dense_rep_hgf = Rectangular()
    params_hgf1 = dense_rep_hgf.from_dense(dense_params_hgf)
    out_man = LinearMap(dense_rep_hgf, Euclidean(shape_hgf[1]), Euclidean(shape_hgf[0]))
    out_mat: Point[Coords, LinearMap[Rectangular, Euclidean, Euclidean]] = (
        out_man.point(params_hgf1)
    )
    return out_man, out_mat


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
    f_trans = f.trn_man
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
    out_mat: Point[Coords, Covariance[PositiveDefinite]] = cov_man.from_dense(
        fgf_man.to_dense(fgf_params)
    )
    return cov_man, out_mat
