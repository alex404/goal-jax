"""This module provides implementations of linear Gaussian models (LGMs), including factor analysis and principal component analysis. LGMs model linear, Gaussian relationships between observable and latent variables. The conjugacy of LGMs enables exact inference and EM."""

from __future__ import annotations

from abc import ABC
from dataclasses import dataclass
from typing import Any, override

import jax.numpy as jnp
from jax import Array

from ...geometry import (
    AnalyticConjugated,
    Diagonal,
    DifferentiableConjugated,
    EmbeddedMap,
    IdentityEmbedding,
    LinearEmbedding,
    MatrixRep,
    PositiveDefinite,
    Rectangular,
    Scale,
    SymmetricConjugated,
)
from ..base.gaussian.boltzmann import Boltzmann
from ..base.gaussian.generalized import Euclidean, GeneralizedGaussian
from ..base.gaussian.normal import (
    Covariance,
    Normal,
)

### Helper Functions ###


@dataclass(frozen=True)
class GeneralizedGaussianLocationEmbedding[G: GeneralizedGaussian[Any]](
    LinearEmbedding[Euclidean, G],
):
    """Embedding of the Euclidean location component into a GeneralizedGaussian distribution.

    Projects a GeneralizedGaussian point in mean coordinates to its Euclidean location component, or embeds
    a location vector in natural coordinates into a GeneralizedGaussian with zero shape parameters.
    """

    gau_man: G
    """The GeneralizedGaussian distribution."""

    @property
    @override
    def amb_man(self) -> G:
        return self.gau_man

    @property
    @override
    def sub_man(self) -> Euclidean:
        return self.gau_man.loc_man

    @override
    def project(self, means: Array) -> Array:  # pyright: ignore[reportIncompatibleMethodOverride]
        """Project to Euclidean location component.

        Works on mean coordinates, extracting the location component from the full
        generalized Gaussian parameterization. If given a data point (size == data_dim),
        converts it to sufficient statistics (mean coordinates) first.

        Parameters
        ----------
        means : Array
            Mean coordinate parameters or data point in GeneralizedGaussian space.

        Returns
        -------
        Array
            Mean parameters in Euclidean space (location only).
        """
        # Convert data points to sufficient statistics (mean coordinates)
        if means.size == self.gau_man.data_dim and means.size != self.gau_man.dim:
            means = self.gau_man.sufficient_statistic(means)

        loc, _ = self.gau_man.split_mean_second_moment(means)
        return loc

    @override
    def embed(self, params: Array) -> Array:  # pyright: ignore[reportIncompatibleMethodOverride]
        """Embed Euclidean location into GeneralizedGaussian with zero shape.

        Parameters
        ----------
        params : Array
            Natural parameters in Euclidean space.

        Returns
        -------
        Array
            Natural parameters in GeneralizedGaussian space.
        """
        zero_shape = self.gau_man.shp_man.zeros()
        return self.gau_man.join_location_precision(params, zero_shape)

    @override
    def translate(  # pyright: ignore[reportIncompatibleMethodOverride]
        self,
        params: Array,
        delta: Array,
    ) -> Array:
        """Translate by adding Euclidean offset to location.

        Parameters
        ----------
        params : Array
            Natural parameters in GeneralizedGaussian space.
        delta : Array
            Euclidean offset to add.

        Returns
        -------
        Array
            Translated natural parameters in GeneralizedGaussian space.
        """
        loc, shape = self.gau_man.split_location_precision(params)
        new_loc = loc + delta
        return self.gau_man.join_location_precision(new_loc, shape)


@dataclass(frozen=True)
class NormalCovarianceEmbedding(LinearEmbedding[Normal, Normal]):
    """Embedding of a normal distribution with a simpler covariance structure into a more complex one."""

    # Fields

    _sub_man: Normal
    """The sub-manifold with the simpler covariance structure."""

    _amb_man: Normal
    """The super-manifold with the more complex covariance structure."""

    def __post_init__(self):
        if not isinstance(self.sub_man.cov_man.rep, type(self.amb_man.cov_man.rep)):
            raise TypeError(
                f"Sub-manifold rep {self.sub_man.cov_man.rep} must be simpler than super-manifold rep {self.amb_man.cov_man.rep}"
            )

    @property
    @override
    def amb_man(self) -> Normal:
        return self._amb_man

    @property
    @override
    def sub_man(self) -> Normal:
        return self._sub_man

    @override
    def project(self, means: Array) -> Array:  # pyright: ignore[reportIncompatibleMethodOverride]
        """Project from ambient to sub-manifold representation.

        Parameters
        ----------
        means : Array
            Mean parameters in ambient manifold.

        Returns
        -------
        Array
            Mean parameters in sub-manifold.
        """
        return self.amb_man.project_rep(self.sub_man, means)

    @override
    def embed(self, params: Array) -> Array:  # pyright: ignore[reportIncompatibleMethodOverride]
        """Embed from sub-manifold to ambient representation.

        Parameters
        ----------
        params : Array
            Natural parameters in sub-manifold.

        Returns
        -------
        Array
            Natural parameters in ambient manifold.
        """
        return self.sub_man.embed_rep(self.amb_man, params)

    @override
    def translate(  # pyright: ignore[reportIncompatibleMethodOverride]
        self,
        params: Array,
        delta: Array,
    ) -> Array:
        """Translate by embedding and adding.

        Parameters
        ----------
        params : Array
            Natural parameters in ambient manifold.
        delta : Array
            Natural parameters in sub-manifold to add.

        Returns
        -------
        Array
            Translated natural parameters in ambient manifold.
        """
        embedded_q = self.sub_man.embed_rep(self.amb_man, delta)
        return params + embedded_q


@dataclass(frozen=True)
class LGM[
    PostGaussian: GeneralizedGaussian[Any],
    PriorGaussian: GeneralizedGaussian[Any],
](
    DifferentiableConjugated[Normal, PostGaussian, PriorGaussian],
    ABC,
):
    """A linear Gaussian model (LGM) implemented as a harmonium with Gaussian latent variables.

    Linear Gaussian Models represent a joint distribution over observable variables $X$ and latent variables $Z$ where both are Gaussian and the relationship between them is linear. In generative terms, this can be viewed as:

    $$x = Az + \\mu + \\epsilon$$

    where:
        - $z$ is drawn from a multivariate normal (typically a standard normal),
        - $A$ is the loading matrix mapping latent to observable space,
        - $\\mu$ is the observable bias term, and
        - $\\epsilon \\sim \\mathcal{N}(0, \\Sigma)$ is Gaussian noise.

    **Posterior vs Prior Structure**: The posterior latent distribution (conditioned on observables) uses the `PostGaussian` parameterization, which may employ a restricted covariance structure (e.g., diagonal) for computational efficiency during frequent inference. The prior latent distribution uses the `PriorGaussian` parameterization, whose shape is dictated by the conjugation parameters. When `PostGaussian` is more restricted than `PriorGaussian`, the prior is constructed by embedding the restricted posterior covariance structure into the fuller prior structure, ensuring compatibility with the required conjugation parameter computation.

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

    obs_rep: PositiveDefinite
    """Covariance structure of the observable variables."""

    ### Methods ###

    # Overrides

    @property
    @override
    def obs_man(self) -> Normal:
        """Override to construct directly from fields, avoiding circular dependency."""
        return Normal(self.obs_dim, self.obs_rep)

    @property
    def int_obs_emb(self) -> GeneralizedGaussianLocationEmbedding[Normal]:
        return GeneralizedGaussianLocationEmbedding(self.obs_man)

    @property
    def int_pst_emb(self) -> LinearEmbedding[Euclidean, PostGaussian]:
        """Embedding of Euclidean location into posterior latent - general for all GeneralizedGaussians."""
        return GeneralizedGaussianLocationEmbedding(self.pst_man)

    @property
    @override
    def int_man(self) -> EmbeddedMap[PostGaussian, Normal]:
        return EmbeddedMap(
            Rectangular(),
            self.int_pst_emb,
            self.int_obs_emb,
        )

    @override
    def conjugation_parameters(
        self,
        lkl_params: Array,
    ) -> Array:
        """Compute conjugation parameters for linear Gaussian model.

        Parameters
        ----------
        lkl_params : Array
            Natural parameters for likelihood function.

        Returns
        -------
        Array
            Natural parameters for conjugation in PriorGaussian space.
        """
        # Get parameters
        obs_cov_man = self.obs_man.cov_man
        obs_bias, int_mat = self.lkl_fun_man.split_coords(lkl_params)
        obs_loc, obs_prec = self.obs_man.split_location_precision(obs_bias)

        # Intermediate computations
        obs_sigma = obs_cov_man.inverse(obs_prec)
        obs_mean = obs_cov_man(obs_sigma, obs_loc)

        # Conjugation parameters

        with self.int_man as im:
            int_mat_trn = im.transpose(int_mat)
            rho_mean = im.trn_man.rep.matvec(
                im.trn_man.matrix_shape, int_mat_trn, obs_mean
            )

            _, rho_shape = _change_of_basis(
                im.matrix_shape,
                im.rep,
                int_mat,
                obs_cov_man.rep,
                obs_sigma,
            )
        rho_shape *= -1

        # Join parameters into moment parameters
        return self.prr_man.join_location_precision(rho_mean, rho_shape)


@dataclass(frozen=True)
class NormalLGM(
    LGM[Normal, Normal],
):
    """Differentiable Linear Gaussian Model with Normal latent variables.

    Extends the abstract LGM with Normal-specific implementations for computing
    observable distributions and converting to joint Normal form.
    """

    lat_dim: int
    """Dimension of the latent variables."""

    pst_lat_rep: PositiveDefinite

    # Overrides

    @property
    @override
    def pst_man(self) -> Normal:
        """Override to construct directly from fields, avoiding circular dependency."""
        return Normal(self.lat_dim, self.pst_lat_rep)

    @property
    @override
    def pst_prr_emb(self) -> LinearEmbedding[Normal, Normal]:
        """Embedding of posterior Normal into prior Normal via covariance structure."""
        prior_gau = Normal(self.lat_dim, PositiveDefinite())
        return NormalCovarianceEmbedding(self.pst_man, prior_gau)

    # Methods

    def observable_distribution(
        self,
        params: Array,
    ) -> tuple[Normal, Array]:  # (Normal, Natural[Normal])
        """Returns the marginal normal distribution over observable variables.

        Parameters
        ----------
        params : Array
            Natural parameters for the linear Gaussian model.

        Returns
        -------
        tuple[Normal, Array]
            The Normal manifold and its natural parameters.
        """
        # Build transposed LGM with full covariance observable variables
        transposed_lgm = NormalAnalyticLGM(
            obs_dim=self.pst_man.data_dim,  # Original latent becomes observable
            obs_rep=PositiveDefinite(),
            lat_dim=self.obs_dim,  # Original observable becomes latent
        )

        # Construct parameters for transposed model
        obs_params, int_params, lat_params = self.split_coords(params)
        nor_man = transposed_lgm.prr_man
        obs_params_emb = self.obs_man.embed_rep(nor_man, obs_params)
        lat_params_emb = self.pst_man.embed_rep(transposed_lgm.obs_man, lat_params)

        # Join parameters with interaction matrix transposed
        transposed_params = transposed_lgm.join_coords(
            lat_params_emb,  # Original latent becomes observable
            self.int_man.transpose(int_params),
            obs_params_emb,
        )

        # Use harmonium prior to get marginal distribution
        return nor_man, transposed_lgm.prior(transposed_params)

    def to_normal(self, params: Array) -> Array:
        """Convert a linear model to a normal model.

        Parameters
        ----------
        params : Array
            Natural parameters for the linear Gaussian model.

        Returns
        -------
        Array
            Natural parameters for the joint Normal distribution.
        """
        lat_dim = self.prr_man.data_dim
        new_man: NormalLGM = NormalLGM(
            obs_dim=self.obs_man.data_dim,
            obs_rep=PositiveDefinite(),
            lat_dim=lat_dim,
            pst_lat_rep=PositiveDefinite(),
        )
        obs_params, int_params, lat_params = self.split_coords(params)
        emb_obs_params = self.obs_man.embed_rep(new_man.obs_man, obs_params)
        emb_lat_params = self.pst_man.embed_rep(new_man.prr_man, lat_params)

        obs_loc, obs_prs = new_man.obs_man.split_location_precision(emb_obs_params)
        lat_loc, lat_prs = new_man.prr_man.split_location_precision(emb_lat_params)
        nor_man = Normal(self.data_dim, PositiveDefinite())
        nor_loc = jnp.concatenate([obs_loc, lat_loc])
        obs_prs_array = new_man.obs_man.cov_man.to_matrix(obs_prs)
        lat_prs_array = new_man.prr_man.cov_man.to_matrix(lat_prs)
        int_array = -self.int_man.to_matrix(int_params)
        joint_shape_array = jnp.block(
            [[obs_prs_array, int_array], [int_array.T, lat_prs_array]]
        )
        return nor_man.join_location_precision(
            nor_loc, nor_man.cov_man.from_dense(joint_shape_array)
        )


@dataclass(frozen=True)
class BoltzmannLGM(
    SymmetricConjugated[Normal, Boltzmann],
    LGM[Boltzmann, Boltzmann],
):
    """Differentiable Linear Gaussian Model with Boltzmann latent variables.

    This model combines a Normal observable distribution with Boltzmann (binary)
    latent variables. The latent states are discrete binary vectors, making this
    suitable for discrete representation learning and binary latent factor models.

    The observable distribution remains Gaussian (continuous), while the latent
    distribution is a Boltzmann machine (discrete binary). This enables learning
    discrete latent representations of continuous data.
    """

    lat_dim: int
    """Number of binary latent units."""

    # Overrides

    @property
    @override
    def pst_man(self) -> Boltzmann:
        """Override to construct directly from fields, avoiding circular dependency."""
        return Boltzmann(self.lat_dim)

    @property
    @override
    def pst_prr_emb(self) -> LinearEmbedding[Boltzmann, Boltzmann]:
        """Embedding of posterior Boltzmann into prior Boltzmann.

        For Boltzmann machines, both posterior and prior use the same manifold
        structure (no covariance simplification like in Normal case), so we use
        the identity embedding.
        """
        return IdentityEmbedding(self.pst_man)

    @property
    @override
    def lat_man(self) -> Boltzmann:
        """The latent manifold is a Boltzmann machine."""
        return Boltzmann(self.lat_dim)


@dataclass(frozen=True)
class NormalAnalyticLGM(
    AnalyticConjugated[Normal, Normal],
    NormalLGM,
):
    """Analytic Linear Gaussian Model that extends the differentiable LGM with full analytical tractability, adding conversions between mean and natural coordinates, and a closed-form implementation of EM."""

    def __init__(self, obs_dim: int, obs_rep: PositiveDefinite, lat_dim: int):
        super().__init__(
            obs_dim=obs_dim,
            obs_rep=obs_rep,
            lat_dim=lat_dim,
            pst_lat_rep=PositiveDefinite(),
        )

    @property
    @override
    def lat_man(self) -> Normal:
        """The latent manifold is a full Normal distribution."""
        return Normal(self.lat_dim, PositiveDefinite())

    @override
    def to_natural_likelihood(
        self,
        means: Array,  # Mean[Self]
    ) -> Array:
        """Convert mean parameters to natural likelihood parameters.

        Parameters
        ----------
        means : Array
            Mean parameters for the analytic linear Gaussian model.

        Returns
        -------
        Array
            Natural parameters for likelihood function.
        """
        # Get relevant manifolds
        ocm = self.obs_man.cov_man
        lcm = self.lat_man.cov_man
        im = self.int_man

        # Deconstruct parameters
        obs_means, int_means, lat_means = self.split_coords(means)
        obs_mean, obs_cov = self.obs_man.split_mean_covariance(obs_means)
        lat_mean, lat_cov = self.lat_man.split_mean_covariance(lat_means)
        int_cov = int_means - im.rep.outer_product(obs_mean, lat_mean)

        # Construct precisions
        lat_prs = lcm.inverse(lat_cov)
        im_trn = im.trn_man
        int_cov_t = im.transpose(int_cov)
        # cob_man, cob = _change_of_basis(im_trn, int_cov_t, lat_cov_man, lat_prs)
        cob_man, cob = _change_of_basis(
            im_trn.matrix_shape,
            im_trn.rep,
            int_cov_t,
            lcm.rep,
            lat_prs,
        )

        shaped_cob = ocm.from_dense(cob_man.to_matrix(cob))
        obs_prs = ocm.inverse(obs_cov - shaped_cob)
        sizes = (
            ocm.matrix_shape[0],
            ocm.matrix_shape[1],
            im.matrix_shape[1],
            lcm.matrix_shape[1],
        )
        _, _, int_params = _dual_composition(
            sizes,
            ocm.rep,
            obs_prs,
            im.rep,
            int_cov,
            lcm.rep,
            lat_prs,
        )
        obs_loc0 = ocm(obs_prs, obs_mean)
        # obs_loc1 = self._int_man_internal(int_params, lat_mean)

        obs_loc1 = im.rep.matvec(im.matrix_shape, int_params, lat_mean)
        obs_loc = obs_loc0 - obs_loc1

        # Return natural parameters
        obs_params = self.obs_man.join_location_precision(obs_loc, obs_prs)
        return self.lkl_fun_man.join_coords(obs_params, int_params)


@dataclass(frozen=True)
class FactorAnalysis(NormalAnalyticLGM):
    """A factor analysis model with Gaussian latent variables."""

    def __init__(self, obs_dim: int, lat_dim: int):
        super().__init__(obs_dim, Diagonal(), lat_dim)

    @override
    def expectation_maximization(
        self,
        params: Array,
        xs: Array,
    ) -> Array:
        """Perform a single iteration of the EM algorithm.

        Without further constraints the latent Normal of FA is not identifiable,
        and so we hold it fixed at standard normal.

        Parameters
        ----------
        params : Array
            Current natural parameters.
        xs : Array
            Observation data.

        Returns
        -------
        Array
            Updated natural parameters.
        """
        # E-step: Compute expectations
        q = self.mean_posterior_statistics(params, xs)
        p1 = self.to_natural(q)
        lkl_params = self.likelihood_function(p1)
        z = self.lat_man.to_natural(self.lat_man.standard_normal())
        return self.join_conjugated(lkl_params, z)

    def from_loadings(
        self,
        loadings: Array,
        means: Array,
        diags: Array,
    ) -> Array:
        """Convert standard factor analysis parameters to natural parameters.

        Parameters
        ----------
        loadings : Array
            Loading matrix (obs_dim, lat_dim).
        means : Array
            Observation means.
        diags : Array
            Diagonal noise variances.

        Returns
        -------
        Array
            Natural parameters for the factor analysis model.
        """
        # Initialize interaction matrix scaled by precision
        with self.obs_man as om:
            mu = means
            cov = diags
            obs_params = om.to_natural(om.join_mean_covariance(mu, cov))
            obs_prs = om.split_location_precision(obs_params)[1]
            dns_prs = om.cov_man.to_matrix(obs_prs)

        int_mat = self.int_man.from_dense(dns_prs @ loadings)

        # Combine parameters
        lkl_params = self.lkl_fun_man.join_coords(obs_params, int_mat)
        z = self.lat_man.to_natural(self.lat_man.standard_normal())
        return self.join_conjugated(lkl_params, z)


@dataclass(frozen=True)
class PrincipalComponentAnalysis(NormalAnalyticLGM):
    """A principal component analysis model with Gaussian latent variables."""

    def __init__(self, obs_dim: int, lat_dim: int):
        super().__init__(obs_dim, Scale(), lat_dim)

    @override
    def expectation_maximization(
        self,
        params: Array,
        xs: Array,
    ) -> Array:
        """Perform a single iteration of the EM algorithm.

        Without further constraints the latent Normal of PCA is not identifiable,
        and so we hold it fixed at standard normal.

        Parameters
        ----------
        params : Array
            Current natural parameters.
        xs : Array
            Observation data.

        Returns
        -------
        Array
            Updated natural parameters.
        """
        # E-step: Compute expectations
        q = self.mean_posterior_statistics(params, xs)
        p1 = self.to_natural(q)
        lkl_params = self.likelihood_function(p1)
        z = self.lat_man.to_natural(self.lat_man.standard_normal())
        return self.join_conjugated(lkl_params, z)


### Helper Functions ###


def _dual_composition(
    sizes: tuple[int, int, int, int],
    h_rep: MatrixRep,
    h_params: Array,  # Parameters in some coordinate system
    g_rep: MatrixRep,
    g_params: Array,  # Parameters in dual coordinates
    f_rep: MatrixRep,
    f_params: Array,  # Parameters in original coordinate system
) -> tuple[
    MatrixRep,
    tuple[int, int],  # Output shape
    Array,  # Parameters in original coordinate system
]:
    """Three-way matrix multiplication that respects coordinate duality.

    Computes h @ g @ f where g is in dual coordinates.
    """
    # First multiply g @ f
    h_shape = (sizes[0], sizes[1])
    g_shape = (sizes[1], sizes[2])
    f_shape = (sizes[2], sizes[3])
    rep_gf, shape_gf, params_gf = g_rep.matmat(
        g_shape,
        g_params,
        f_rep,
        f_shape,
        f_params,
    )

    # Then multiply h @ (g @ f)
    return h_rep.matmat(h_shape, h_params, rep_gf, shape_gf, params_gf)


# TODO: Could probably try and reduce the number of to/from_matrix calls here and throughout the module
def _change_of_basis(
    f_size: tuple[int, int],
    f_rep: MatrixRep,
    f_params: Array,  # Parameters in some coordinate system
    g_rep: PositiveDefinite,
    g_params: Array,  # Parameters in dual coordinates
) -> tuple[
    Covariance,
    Array,  # Parameters in original coordinate system
]:
    """Linear change of basis transformation.

    Computes f.T @ g @ f where g is in dual coordinates.
    """
    sizes = (f_size[1], f_size[0], f_size[0], f_size[1])
    f_trans_params = f_rep.transpose(f_size, f_params)
    fgf_rep, fgf_sizes, fgf_params = _dual_composition(
        sizes,
        f_rep,
        f_trans_params,
        g_rep,
        g_params,
        f_rep,
        f_params,
    )
    # If fgf_rep is diagonal or stricter, leave it, otherwise positivedefinite
    if isinstance(fgf_rep, (Diagonal, Scale, IdentityEmbedding)):
        cov_man = Covariance(fgf_sizes[0], fgf_rep)
    else:
        cov_man = Covariance(fgf_sizes[0], PositiveDefinite())
        fgf_params = cov_man.from_dense(
            fgf_rep.to_matrix(cov_man.matrix_shape, fgf_params)
        )
    return cov_man, fgf_params
