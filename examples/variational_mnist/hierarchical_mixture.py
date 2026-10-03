"""Hierarchical variational conjugation with a *mixture* top (``p(x, y, z, k)``).

This extends :mod:`.hierarchical` by replacing the single-Gaussian top prior
``p(z)`` with a mixture of Gaussians ``p(z, k) = p(z | k) p(k)`` -- an
:class:`AnalyticMixture` over the top latent. The purpose is unsupervised
*clustering*: the categorical ``k`` labels the top-level Gaussian components, so
the model can group observations ``x`` by which component their inferred ``z``
belongs to.

    p(x, y, z, k) = p(x | y) . p(y | z) . p(z | k) . p(k),

    X observable -- continuous data,
    Y middle     -- a (chordal) Boltzmann spike population,
    Z top latent -- a continuous Gaussian,
    K cluster    -- a categorical over Gaussian components.

Why this stays cheap
--------------------
Two facts make the mixture top almost free on top of the base model:

1. **The Y|Z and X|Y conjugation machinery is prior-independent.** The residuals
   ``r_Y``, ``r*_Z``, ``r_inner_Z`` and ``top_var.conjugation_residual`` depend
   only on biases, interactions, and conjugation slopes -- never on the top
   prior. So every conjugation regularizer and residual is inherited verbatim.

2. **``k`` integrates out exactly.** The Z|K edge is an :class:`AnalyticMixture`,
   so the responsibility ``p(k | z)`` is a closed-form categorical. We never
   amortize ``q(k | x)``; given a sampled ``z`` the responsibilities are exact.
   The clustering readout is ``r_k(x) = E_{z ~ q(z|x)}[p(k | z)]``.

ELBO by control variate
-----------------------
The single-Gaussian ELBO decomposition ``c(x) + E_q[r_Y + r*_Z - r_inner_Z]``
telescopes only because ``log p(z)`` is one exponential-family term. A mixture
breaks that exact cancellation, so we keep the validated single-Gaussian ELBO as
a **control variate** and add the exact correction

    L_mix(x) = E_q[f_single(z; theta_ref)] + E_q[log p_mix(z) - log p_ref(z)],

which equals ``E_q[f_mix]`` for *any* reference ``theta_ref`` (only the variance
depends on it). We use ``theta_ref = moment-match of the mixture`` under a
stop-gradient, so the reference tracks the mixture and the correction stays
small. Concretely this needs only two overrides: fold the correction into
:meth:`learning_signal`, and swap ``theta_ref`` into :meth:`conjugation_baseline`
-- everything else (sampling gradient, score-function surrogate, ``mean_elbo``)
is inherited.
"""

# pyright: reportAttributeAccessIssue=false
# pyright: reportArgumentType=false
# pyright: reportMissingTypeArgument=false

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import override

import jax
import jax.numpy as jnp
from jax import Array

from goal.geometry import (
    Differentiable,
    EmbeddedMap,
    IdentityEmbedding,
    Rectangular,
)
from goal.geometry.manifold.combinators import Pair
from goal.geometry.manifold.map import MultilayerPerceptron
from goal.models import (
    AnalyticMixture,
    Boltzmann,
    ChainBoltzmann,
    ChordalBoltzmann,
    DiagonalBoltzmann,
    FullNormal,
    full_normal,
)
from goal.models.harmonium.lgm import GeneralizedGaussianLocationEmbedding
from goal.models.harmonium.population_codes import (
    BoltzmannNormalHarmonium,
    BoltzmannPopulationCode,
)

from .hierarchical import (
    HierarchicalRecognition,
    VariationalHierarchical,
    _chain_edges,
    conv_hierarchy_components,
)
from .model import ConcreteHarmonium


@dataclass(frozen=True)
class RecognitionAndPrior[MidLatent: Differentiable](
    Pair[HierarchicalRecognition[MidLatent], AnalyticMixture[FullNormal]]
):
    """Third Triple slot: ``(recognition extras, mixture top prior p(z, k))``.

    Bundling the mixture prior into the recognition slot keeps the model a
    :class:`Triple`, so all of :class:`VariationalHierarchical`'s parameter
    plumbing is reused; only the accessors that peel this pair are overridden.
    """

    _recog: HierarchicalRecognition[MidLatent]
    _prior: AnalyticMixture[FullNormal]

    @property
    @override
    def fst_man(self) -> HierarchicalRecognition[MidLatent]:
        return self._recog

    @property
    @override
    def snd_man(self) -> AnalyticMixture[FullNormal]:
        return self._prior


@dataclass(frozen=True)
class VariationalHierarchicalMixture[
    Observable: Differentiable,
    MidLatent: Differentiable,
](VariationalHierarchical[Observable, MidLatent]):
    """``VariationalHierarchical`` with a mixture-of-Gaussians top prior.

    Parameter layout ``[top_params | lower_lkl | (recog | mixture)]``: identical
    to the base model except the third block now carries the mixture prior
    ``p(z, k)`` alongside the recognition extras.
    """

    top_prior: AnalyticMixture[FullNormal]
    """Top prior p(z, k) as a mixture of Gaussians over the top latent Z."""

    # --- Triple slot / manifold access (bundle mixture into slot 3) --------

    @property
    @override
    def trd_man(self) -> RecognitionAndPrior[MidLatent]:
        return RecognitionAndPrior(self.recog_man, self.top_prior)

    @property
    def n_clusters(self) -> int:
        """Number of top-level Gaussian clusters K."""
        return self.top_prior.n_categories

    # --- Parameter access (peel the (recog, mixture) pair) ----------------

    @override
    def split_recog(self, params: Array) -> tuple[Array, Array]:
        """``(rho_Y, phi)`` recognition extras, peeled from the bundled slot."""
        _, _, third = self.split_coords(params)
        recog_params, _ = self.trd_man.split_coords(third)
        return self.recog_man.split_coords(recog_params)

    def split_mixture(self, params: Array) -> Array:
        """Mixture prior parameters ``p(z, k)`` from the bundled slot."""
        _, _, third = self.split_coords(params)
        _, mix_params = self.trd_man.split_coords(third)
        return mix_params

    @override
    def split_top(self, params: Array) -> tuple[Array, Array, Array]:
        """``(theta_ref, (theta_Y, Theta_YZ), rho0_Z)`` with the moment-matched reference.

        The stored ``top_var`` prior slot is replaced by the control-variate
        reference :meth:`reference_prior`, so every base-class consumer of the top
        prior -- the inner slope ``rho^X_Z`` and the baseline ``c(x)`` -- reads the
        *same* ``theta_ref`` that :meth:`learning_signal` subtracts. Keeping these
        consistent is what makes the control-variate assembly equal the exact
        mixture ELBO. (``residual_top_*`` pass the prior slot to
        ``conjugation_residual``, which ignores it, so this substitution is inert
        there.)
        """
        _, top_lkl, rho0_z = super().split_top(params)
        return self.reference_prior(params), top_lkl, rho0_z

    # --- Reference Gaussian (control-variate baseline for the z-prior) ----

    def reference_prior(self, params: Array) -> Array:
        """Moment-matched single Gaussian ``theta_ref`` of the mixture (stop-grad).

        Used only as the control-variate reference in the ELBO: the estimator is
        exact for any reference, so matching the mixture's first two moments
        keeps the ``log p_mix - log p_ref`` correction small.
        """
        mix = self.split_mixture(params)
        mean, cov = self.top_prior.observable_mean_covariance(mix)
        # Jitter before inversion: a collapsing component can make the mixture
        # covariance singular, and a NaN reference would poison the ELBO. The
        # reference is a stop-grad variance control, so jitter is harmless.
        cov = cov + 1e-4 * jnp.eye(self.top_man.data_dim)
        precision = jnp.linalg.inv(cov)
        location = precision @ mean
        prec_params = self.top_man.cov_man.rep.from_matrix(precision)
        ref = self.top_man.join_location_precision(location, prec_params)
        return jax.lax.stop_gradient(ref)

    # --- ELBO: single-Gaussian control variate + exact mixture correction -

    @override
    def learning_signal_z(self, params: Array, x: Array, z: Array) -> Array:
        """Base z-signal plus the mixture correction ``log p_mix(z) - log p_ref(z)``.

        The correction is z-only, so extending :meth:`learning_signal_z` covers
        both estimators at once: the sampled-y ``learning_signal`` (which adds
        ``r_Y`` on top) and the exact-N :meth:`marginal_elbo_at`. With
        ``theta_ref`` also used in :meth:`conjugation_baseline`, ``c(x) +
        E_q[signal]`` equals the exact mixture ELBO ``E_q[f_mix]``.
        """
        base = super().learning_signal_z(params, x, z)
        mix = self.split_mixture(params)
        ref = self.reference_prior(params)
        log_p_mix = self.top_prior.log_observable_density(mix, z)
        log_p_ref = self.top_man.log_density(ref, z)
        return base + log_p_mix - log_p_ref

    # --- Generative prior over z is now the mixture -----------------------

    def _sample_top_z(self, key: Array, params: Array, n: int) -> Array:
        """Ancestral z-samples from the mixture prior ``p(z) = sum_k p(k) p(z | k)``."""
        return self.top_prior.observable_sample(key, self.split_mixture(params), n)

    @override
    def sample(self, key: Array, params: Array, n: int = 1) -> Array:
        """Ancestral sample z ~ p(z) [mixture], y ~ p(y | z), x ~ p(x | y)."""
        kz, ky, kx = jax.random.split(key, 3)
        _, top_lkl, _ = self.split_top(params)
        _, lower_lkl, _ = self.split_coords(params)

        z_samples = self._sample_top_z(kz, params, n)

        def sample_y(subkey: Array, z: Array) -> Array:
            s_z = self.top_man.sufficient_statistic(z)
            y_params = self.top_var.gen_hrm.lkl_fun_man(top_lkl, s_z)
            return self.top_var.obs_man.sample(subkey, y_params, 1)[0]

        y_samples = jax.vmap(sample_y)(jax.random.split(ky, n), z_samples)

        def sample_x(subkey: Array, y: Array) -> Array:
            s_y = self.mid_man.sufficient_statistic(y)
            x_params = self.lower_hrm.lkl_fun_man(lower_lkl, s_y)
            return self.obs_man.sample(subkey, x_params, 1)[0]

        x_samples = jax.vmap(sample_x)(jax.random.split(kx, n), y_samples)
        return jnp.concatenate([x_samples, y_samples, z_samples], axis=-1)

    @override
    def log_density_joint(self, params: Array, x: Array, y: Array, z: Array) -> Array:
        """log p(x, y, z) = log p_mix(z) + log p(y | z) + log p(x | y)."""
        _, top_lkl, _ = self.split_top(params)
        _, lower_lkl, _ = self.split_coords(params)

        log_pz = self.top_prior.log_observable_density(self.split_mixture(params), z)
        s_z = self.top_man.sufficient_statistic(z)
        y_params = self.top_var.gen_hrm.lkl_fun_man(top_lkl, s_z)
        log_py = self.mid_man.log_density(y_params, y)
        s_y = self.mid_man.sufficient_statistic(y)
        x_params = self.lower_hrm.lkl_fun_man(lower_lkl, s_y)
        log_px = self.obs_man.log_density(x_params, x)
        return log_pz + log_py + log_px

    @override
    def prior_conjugation_loss_components(
        self, key: Array, params: Array, n_samples: int
    ) -> tuple[Array, Array]:
        """Per-edge penalties ``(Var_p[r_Y], Var_p[r*_Z])`` over *mixture*-prior samples.

        Same as the base class except ``z`` is drawn from the mixture prior
        ``p(z) = sum_k p(k) p(z | k)`` rather than a single Gaussian; the summed
        :meth:`prior_conjugation_loss` is inherited on top of this.
        """
        kz, ky = jax.random.split(key)
        _, top_lkl, _ = self.split_top(params)
        z_samples = jax.lax.stop_gradient(self._sample_top_z(kz, params, n_samples))
        r_z = jax.vmap(lambda z: self.residual_top_generative(params, z))(z_samples)
        if self.lower_edge_exact:  # r_Y == 0 pointwise: skip the y-sampling
            return jnp.zeros(()), jnp.var(r_z)

        y_keys = jax.random.split(ky, n_samples)

        def sample_y(subkey: Array, z: Array) -> Array:
            s_z = self.top_man.sufficient_statistic(z)
            y_params = self.top_var.gen_hrm.lkl_fun_man(top_lkl, s_z)
            return self.top_var.obs_man.sample(subkey, y_params, 1)[0]

        y_samples = jax.lax.stop_gradient(jax.vmap(sample_y)(y_keys, z_samples))

        r_y = jax.vmap(lambda y: self.residual_lower(params, y))(y_samples)
        return jnp.var(r_y), jnp.var(r_z)

    # --- Clustering readout -----------------------------------------------

    def responsibilities(self, key: Array, params: Array, x: Array, n: int) -> Array:
        """Cluster responsibilities ``r_k(x) = E_{z ~ q(z|x)}[p(k | z)]`` (shape ``[K]``).

        Samples ``z`` from the recognition posterior ``q(z | x)`` and averages the
        exact mixture responsibility ``p(k | z)`` over them.
        """
        q_top = self.approximate_posterior_top(params, x)
        z_samples = self.top_man.sample(key, q_top, n)
        mix = self.split_mixture(params)
        cat = self.top_prior.lat_man

        def resp(z: Array) -> Array:
            cat_nat = self.top_prior.posterior_at(mix, z)
            return cat.to_probs(cat.to_mean(cat_nat))

        return jnp.mean(jax.vmap(resp)(z_samples), axis=0)

    def cluster_assignments(
        self, key: Array, params: Array, xs: Array, n: int
    ) -> Array:
        """Hard cluster labels ``argmax_k r_k(x)`` for a batch (shape ``[batch]``)."""
        keys = jax.random.split(key, xs.shape[0])
        resp = jax.vmap(lambda k, x: self.responsibilities(k, params, x, n))(keys, xs)
        return jnp.argmax(resp, axis=-1)

    # --- Initialization ---------------------------------------------------

    def _join_with_mixture(
        self, top: Array, lower_lkl: Array, recog: Array, mixture: Array
    ) -> Array:
        third = self.trd_man.join_coords(recog, mixture)
        return self.join_coords(top, lower_lkl, third)

    @override
    def initialize(
        self, key: Array, location: float = 0.0, shape: float = 0.1
    ) -> Array:
        k_top, k_low, k_mlp, k_mix = jax.random.split(key, 4)
        top = self.top_var.initialize(k_top, location, shape)
        low_hrm = self.lower_hrm.initialize(k_low, location, shape)
        theta_x, theta_xy, _ = self.lower_hrm.split_coords(low_hrm)
        lower_lkl = self.lower_hrm.lkl_fun_man.join_coords(theta_x, theta_xy)
        rho_y = jnp.zeros(self.mid_man.dim)
        phi = self.mlp_man.glorot_initialize(k_mlp)
        recog = self.recog_man.join_coords(rho_y, phi)
        mixture = self.top_prior.initialize(k_mix, location, shape)
        return self._join_with_mixture(top, lower_lkl, recog, mixture)

    @override
    def initialize_from_sample(
        self, key: Array, sample: Array, location: float = 0.0, shape: float = 0.1
    ) -> Array:
        k_top, k_low, k_mlp, k_mix = jax.random.split(key, 4)
        top = self.top_var.initialize(k_top, location, shape)
        low_hrm = self.lower_hrm.initialize_from_sample(k_low, sample, location, shape)
        theta_x, theta_xy, _ = self.lower_hrm.split_coords(low_hrm)
        lower_lkl = self.lower_hrm.lkl_fun_man.join_coords(theta_x, theta_xy)
        rho_y = jnp.zeros(self.mid_man.dim)
        phi = self.mlp_man.glorot_initialize(k_mlp)
        recog = self.recog_man.join_coords(rho_y, phi)
        mixture = self.top_prior.initialize(k_mix, location, shape)
        return self._join_with_mixture(top, lower_lkl, recog, mixture)


# --- Factory ----------------------------------------------------------------


def build_boltzmann_gaussian_mixture_hierarchy(
    obs_man: Differentiable,
    n_mid: int,
    top_dim: int,
    n_clusters: int,
    mid_kind: str = "chain",
    mid_edges: Sequence[tuple[int, int]] | None = None,
    mlp_hidden: tuple[int, ...] = (128,),
    mlp_activation: Callable[[Array], Array] = jax.nn.gelu,
    max_treewidth: int | None = None,
    obs_location_only: bool = False,
) -> VariationalHierarchicalMixture:
    """Assemble ``p(x, y, z, k)`` -- the base hierarchy with a mixture top prior.

    Same arguments as :func:`.hierarchical.build_boltzmann_gaussian_hierarchy`
    plus ``n_clusters``, the number of top-level Gaussian components K.
    """
    mid_man: Boltzmann
    if mid_kind == "diagonal":
        mid_man = DiagonalBoltzmann(n_neurons=n_mid)
    elif mid_kind == "chordal":
        edges = list(mid_edges) if mid_edges is not None else _chain_edges(n_mid)
        mid_man = ChordalBoltzmann.from_edges(n_mid, edges, max_treewidth)
    else:  # "chain"
        mid_man = ChainBoltzmann.from_edges(n_mid, _chain_edges(n_mid))

    top_var = BoltzmannPopulationCode(BoltzmannNormalHarmonium(mid_man, top_dim))

    obs_emb = (
        GeneralizedGaussianLocationEmbedding(obs_man)
        if obs_location_only
        else IdentityEmbedding(obs_man)
    )
    lower_int = EmbeddedMap(Rectangular(), IdentityEmbedding(mid_man), obs_emb)
    lower_hrm = ConcreteHarmonium(lower_int)

    mlp = MultilayerPerceptron(
        full_normal(top_dim), mid_man, mlp_hidden, mlp_activation
    )
    recog = HierarchicalRecognition(mid_man, mlp)

    top_prior = AnalyticMixture(full_normal(top_dim), n_clusters)

    return VariationalHierarchicalMixture(
        top_var=top_var,
        lower_hrm=lower_hrm,
        recog_man=recog,
        top_prior=top_prior,
    )


def build_conv_boltzmann_gaussian_mixture_hierarchy(
    obs_man: Differentiable,
    in_lattice: tuple[int, ...],
    stride: tuple[int, ...],
    kernel_shape: tuple[int, ...],
    top_dim: int,
    n_clusters: int,
    in_channels: int = 1,
    out_channels: int = 1,
    prior_graph: str = "chordal",
    max_treewidth: int | None = None,
    mlp_hidden: tuple[int, ...] = (128,),
    mlp_activation: Callable[[Array], Array] = jax.nn.gelu,
) -> VariationalHierarchicalMixture:
    """Assemble ``p(x, y, z, k)`` with a **convolutional** lower decoder.

    The mixture-top counterpart of
    :func:`.hierarchical.build_conv_boltzmann_gaussian_hierarchy`: the identical
    conv/chordal lower edge and Boltzmann population-code upper edge (see
    :func:`.hierarchical.conv_hierarchy_components`), with the single-Gaussian
    top prior replaced by an :class:`AnalyticMixture` over K components. Because
    the three shared slots match the single-Gaussian model parameter-for-
    parameter, a trained ``VariationalHierarchical`` checkpoint transfers
    verbatim into the first two blocks and the recognition half of the third.
    """
    top_var, lower_hrm, recog = conv_hierarchy_components(
        obs_man,
        in_lattice,
        stride,
        kernel_shape,
        top_dim,
        in_channels=in_channels,
        out_channels=out_channels,
        prior_graph=prior_graph,
        max_treewidth=max_treewidth,
        mlp_hidden=mlp_hidden,
        mlp_activation=mlp_activation,
    )
    top_prior = AnalyticMixture(full_normal(top_dim), n_clusters)
    return VariationalHierarchicalMixture(
        top_var=top_var,
        lower_hrm=lower_hrm,
        recog_man=recog,
        top_prior=top_prior,
    )
