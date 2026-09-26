"""Hierarchical variational conjugation (3-level ``p(x, y, z)``), playground build.

This is a *local, out-of-core* implementation of the hierarchical variational
conjugation scheme from the "Variational Conjugation in Hierarchical Models"
section of the conjugated-graphical-harmoniums article. It is deliberately kept
in ``examples/`` rather than ``src/goal/`` until the design settles.

Model
-----
A three-level directed hierarchy

    p(x, y, z) = p(x | y) . p(y | z) . p(z),

with natural-parameter conditionals

    eta_{X|Y}(y) = theta_X + Theta_XY . s_Y(y),
    eta_{Y|Z}(z) = theta_Y + Theta_YZ . s_Z(z),
    p(z) = prior with natural parameters theta*_Z.

Mapping to the grant's WP2 spike/continuous iteration:

- ``X`` observable  -- continuous data,
- ``Y`` middle      -- a (chordal) Boltzmann spike population,
- ``Z`` top latent  -- a continuous Gaussian.

So the lower edge ``p(x | y)`` is a Gaussian-Boltzmann likelihood (psi_X
closed-form quadratic) and the upper edge ``p(y | z)`` is a Boltzmann population
code (psi_Y computed by the chordal junction tree).

Composition
-----------
The upper edge ``p(y, z)`` is *itself* a bivariate variational-conjugation model
-- a :class:`BoltzmannPopulationCode` (``VariationalSymmetric[Boltzmann, Normal,
Normal]``). We store it whole as ``top_var`` and reuse its
``conjugation_residual`` to evaluate both

- ``r*_Z`` -- the top-edge residual at the *generative* Y-bias theta_Y with the
  input-independent slope rho0_Z, and
- ``r_inner_Z`` -- the *same* residual form at the *posterior* Y-bias
  eta_hat_{Y|X}(x) with the amortized slope rho^X_Z(x) produced by an MLP.

The lower-edge residual ``r_Y`` uses psi_X and mirrors the bivariate residual.

Recognition (chain factorization)
---------------------------------
    q(y, z | x) = q(y | z, x) . q(z | x), where
    eta_hat_{Y|X}(x)   = theta_Y - rho_Y + s_X(x) . Theta_XY,
    eta_hat_{Z|X}(x)   = (theta*_Z - rho0_Z) + MLP_phi(eta_hat_{Y|X}(x)),
    eta_hat_{Y|Z,X}    = eta_hat_{Y|X}(x) + Theta_YZ . s_Z(z).

Because each conditional is normalized by construction, the chain ``q`` is a
valid distribution for *any* MLP, so the resulting objective is a genuine ELBO
regardless of inner-slope quality -- the MLP error only loosens the bound.

Learning signal
---------------
    f(x, y, z) = log p(x, y, z) - log q(y, z | x)
               = r_Y(y) + r*_Z(z) - r_inner_Z(z; x) + c(x),
    c(x) = s_X(x).theta_X - psi_X(theta_X) - psi_Y(theta_Y) - psi_Z(theta*_Z)
           + psi_Y(eta_hat_{Y|X}(x)) + psi_Z(eta_hat_{Z|X}(x)),

with ``c(x)`` collecting the entire x-dependence (the exact score-function
baseline) and ``L(x) = c(x) + E_q[r_Y + r*_Z - r_inner_Z]``. The two generative
residuals ``r_Y, r*_Z`` drive the generative model toward conjugation; the inner
residual ``r_inner_Z`` drives the recognition MLP toward tractability.
"""

# pyright: reportAttributeAccessIssue=false
# pyright: reportArgumentType=false
# pyright: reportMissingTypeArgument=false

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from math import prod
from typing import override

import jax
import jax.numpy as jnp
from jax import Array

from goal.geometry import (
    Diagonal,
    Differentiable,
    EmbeddedMap,
    Generative,
    IdentityEmbedding,
    LinearEmbedding,
    Rectangular,
)
from goal.geometry.exponential_family.harmonium import SymmetricConjugated
from goal.geometry.manifold.combinators import Pair, Triple
from goal.geometry.manifold.map import AffineMap, LinearMap, MultilayerPerceptron
from goal.models import (
    Bernoullis,
    Boltzmann,
    ChainBoltzmann,
    ChordalBoltzmann,
    DiagonalBoltzmann,
    FullNormal,
    Normal,
    full_normal,
)
from goal.models.harmonium.lgm import GeneralizedGaussianLocationEmbedding
from goal.models.harmonium.population_codes import (
    BoltzmannNormalHarmonium,
    BoltzmannPopulationCode,
)

from .lattice_convolution import EmbeddedLinearMap, LatticeConvolution
from .model import ConcreteHarmonium

# --- Recognition-extras manifold: (rho_Y storage, MLP phi) ------------------


@dataclass(frozen=True)
class HierarchicalRecognition[MidLatent: Differentiable](
    Pair[MidLatent, MultilayerPerceptron[MidLatent, FullNormal]]
):
    """Storage for the two recognition tiers not held inside ``top_var``.

    ``fst`` -- rho_Y, the input-independent lower-edge conjugation vector, stored
    in ``MidLatent`` (spike) coordinates.
    ``snd`` -- the weights of the amortized inner-slope map
    ``rho^X_Z: eta_hat_{Y|X}(x) |-> Normal-shaped slope``.
    """

    _mid_man: MidLatent
    _mlp: MultilayerPerceptron[MidLatent, FullNormal]

    @property
    @override
    def fst_man(self) -> MidLatent:
        return self._mid_man

    @property
    @override
    def snd_man(self) -> MultilayerPerceptron[MidLatent, FullNormal]:
        return self._mlp


# --- The hierarchical model -------------------------------------------------


@dataclass(frozen=True)
class VariationalHierarchical[
    Observable: Differentiable,
    MidLatent: Differentiable,
](
    Generative,
    Triple[
        BoltzmannPopulationCode[MidLatent],
        AffineMap[MidLatent, Observable],
        HierarchicalRecognition[MidLatent],
    ],
):
    """Three-level variational-conjugation harmonium ``p(x, y, z)``.

    Parameter layout ``[top_params | lower_lkl_params | recog_params]``:

    - ``top_params``   -- a full :class:`BoltzmannPopulationCode` vector
      ``[theta*_Z, (theta_Y, Theta_YZ), rho0_Z]`` (upper edge + top prior),
    - ``lower_lkl``    -- ``(theta_X, Theta_XY)`` of the Gaussian-Boltzmann
      likelihood ``p(x | y)``,
    - ``recog``        -- ``(rho_Y, phi)`` recognition extras.

    ``MidLatent`` is a :class:`Boltzmann` variant (chordal, chain, diagonal);
    ``Observable`` is any :class:`Differentiable` observable with a closed-form
    log-partition (Normal, Binomials, Poissons).
    """

    top_var: BoltzmannPopulationCode[MidLatent]
    """Upper edge p(y, z) as a bivariate variational-conjugation model."""

    lower_hrm: ConcreteHarmonium[Observable, MidLatent]
    """Lower edge harmonium p(x, y); supplies the likelihood p(x | y)."""

    recog_man: HierarchicalRecognition[MidLatent]
    """Recognition extras manifold (rho_Y, MLP phi)."""

    # Triple slots

    @property
    @override
    def fst_man(self) -> BoltzmannPopulationCode[MidLatent]:
        return self.top_var

    @property
    @override
    def snd_man(self) -> AffineMap[MidLatent, Observable]:
        return self.lower_hrm.lkl_fun_man

    @property
    @override
    def trd_man(self) -> HierarchicalRecognition[MidLatent]:
        return self.recog_man

    # Manifold access

    @property
    def obs_man(self) -> Observable:
        """Observable manifold M_X."""
        return self.lower_hrm.obs_man

    @property
    def mid_man(self) -> MidLatent:
        """Middle (spike) latent manifold M_Y."""
        return self.lower_hrm.pst_man

    @property
    def top_man(self) -> FullNormal:
        """Top (continuous) latent manifold M_Z."""
        return self.top_var.lat_man

    @property
    def mlp_man(self) -> MultilayerPerceptron[MidLatent, FullNormal]:
        """Amortized inner-slope map manifold."""
        return self.recog_man.snd_man

    # --- Parameter access -------------------------------------------------

    def split_top(self, params: Array) -> tuple[Array, Array, Array]:
        """``(theta*_Z, (theta_Y, Theta_YZ), rho0_Z)`` from the top-edge block."""
        top, _, _ = self.split_coords(params)
        return self.top_var.split_coords(top)

    def split_lower(self, params: Array) -> tuple[Array, Array]:
        """``(theta_X, Theta_XY)`` of the lower likelihood."""
        _, lower, _ = self.split_coords(params)
        return self.lower_hrm.lkl_fun_man.split_coords(lower)

    def split_recog(self, params: Array) -> tuple[Array, Array]:
        """``(rho_Y, phi)`` recognition extras."""
        _, _, recog = self.split_coords(params)
        return self.recog_man.split_coords(recog)

    def mid_bias(self, params: Array) -> Array:
        """Generative middle bias theta_Y (the top-edge observable bias)."""
        _, top_lkl, _ = self.split_top(params)
        theta_y, _ = self.top_var.gen_hrm.lkl_fun_man.split_coords(top_lkl)
        return theta_y

    # --- Recognition model ------------------------------------------------

    @property
    def lower_edge_exact(self) -> bool:
        """Whether the lower edge is analytically conjugate (``r_Y == 0`` pointwise).

        True for the conv harmoniums, whose closed-form rho_Y makes the residual
        vanish identically. This is what licenses :meth:`marginal_elbo_at`: the
        spike layer integrates out of the ELBO's residual term exactly.
        """
        return isinstance(
            self.lower_hrm, (ConvBoltzmannHarmonium, ConvChordalBoltzmannHarmonium)
        )

    def lower_conjugation(self, params: Array) -> Array:
        """Lower-edge conjugation vector rho_Y.

        When the lower harmonium is analytically conjugate (a
        :class:`ConvBoltzmannHarmonium`), rho_Y is the closed-form conjugation
        parameter, so ``r_Y == 0`` and ``p(N | x)`` is exact --- no learned
        vector on this edge. Otherwise it falls back to the stored, learned
        recognition vector rho_Y.
        """
        if self.lower_edge_exact:
            _, lower_lkl, _ = self.split_coords(params)
            return self.lower_hrm.conjugation_parameters(lower_lkl)
        rho_y, _ = self.split_recog(params)
        return rho_y

    def posterior_mid_bias(self, params: Array, x: Array) -> Array:
        """eta_hat_{Y|X}(x) = theta_Y - rho_Y + s_X(x) . Theta_XY (spike-space).

        Built by running the lower harmonium's posterior map with its latent
        bias set to ``theta_Y - rho_Y``.
        """
        theta_x, theta_xy = self.split_lower(params)
        theta_y = self.mid_bias(params)
        rho_y = self.lower_conjugation(params)
        lat_bias = theta_y - rho_y
        hrm_params = self.lower_hrm.join_coords(theta_x, theta_xy, lat_bias)
        return self.lower_hrm.posterior_at(hrm_params, x)

    def _project_valid(self, natural: Array, floor: float | Array = 1e-3) -> Array:
        """Mean-preservingly project Normal natural params onto {precision >= floor}.

        A pure positive-definiteness guard: an imperfect amortized slope can push
        the additive posterior ``(theta*_Z - rho0_Z) + rho^X_Z(x)`` out of the PD
        cone and produce NaN log-partitions, so eigenvalues are floored at a tiny
        ``floor``. Mean-preserving: the v1 projection clamped eigenvalues holding
        the natural *location* fixed, which silently moves the mean
        ``mu = Lambda^{-1} theta_1`` exactly in the clamped directions; here we
        recover ``mu`` first (from a PD-safe inverse so it stays bounded even off
        the cone), clamp, and rebuild the location as ``Lambda' mu``. Identity
        whenever all eigenvalues clear the floor -- a safety net, not a
        reparameterization. (A prior-scale width floor was tried and was
        outcome-neutral; see HIERARCHICAL_PLAN.md.)
        """
        loc, prec_params = self.top_man.split_location_precision(natural)
        cov = self.top_man.cov_man
        d = self.top_man.data_dim
        prec = cov.rep.to_matrix((d, d), prec_params)
        prec = 0.5 * (prec + prec.T)
        evals, evecs = jnp.linalg.eigh(prec)
        mu_rot = (evecs.T @ loc) / jnp.maximum(evals, 1e-3)  # mu in the eigenbasis
        clamped = jnp.maximum(evals, floor)
        new_loc = evecs @ (clamped * mu_rot)
        prec_new = (evecs * clamped) @ evecs.T
        return self.top_man.join_location_precision(new_loc, cov.rep.from_matrix(prec_new))

    def approximate_posterior_top(self, params: Array, x: Array) -> Array:
        """q(z | x) natural parameters eta_hat_{Z|X}(x) = (theta*_Z - rho0_Z) + rho^X_Z(x).

        The recognition is the *generative* top prior, conjugation-adjusted by the
        input-independent rho0_Z, plus the amortized conjugation slope rho^X_Z(x)
        that the MLP estimates from the posterior Y-bias eta_hat_{Y|X}(x). The MLP
        supplies only the intractable, data-dependent conjugation *correction* --
        it does not synthesize the posterior from scratch -- keeping recognition
        inside the variational-conjugation framework (article Sec. 4.2.7).
        :meth:`_project_valid` is a positive-definite safety net.
        """
        theta_z, _, rho0_z = self.split_top(params)
        _, phi = self.split_recog(params)
        rho_x_z = self.mlp_man(phi, self.posterior_mid_bias(params, x))
        return self._project_valid((theta_z - rho0_z) + rho_x_z)

    def inner_slope(self, params: Array, x: Array) -> Array:
        """Effective amortized slope rho^X_Z(x) = eta_hat_{Z|X}(x) - (theta*_Z - rho0_Z).

        Equal to the MLP's estimated correction when the PD projection is inactive;
        defined as the difference so the decomposition ``f = r_Y + r*_Z -
        r_inner_Z + c(x)`` stays exact even when :meth:`_project_valid` clamps.
        """
        theta_z, _, rho0_z = self.split_top(params)
        return self.approximate_posterior_top(params, x) - (theta_z - rho0_z)

    def _top_lkl_at_posterior(self, params: Array, x: Array) -> Array:
        """Upper likelihood affine params with the Y-bias swapped for the posterior one.

        ``(eta_hat_{Y|X}(x), Theta_YZ)`` -- used both to sample q(y | z, x) and to
        rebuild the top-edge residual at the posterior bias.
        """
        _, top_lkl, _ = self.split_top(params)
        _, theta_yz = self.top_var.gen_hrm.lkl_fun_man.split_coords(top_lkl)
        mid_bias = self.posterior_mid_bias(params, x)
        return self.top_var.gen_hrm.lkl_fun_man.join_coords(mid_bias, theta_yz)

    def posterior_mid_at(self, params: Array, x: Array, z: Array) -> Array:
        """q(y | z, x) natural params eta_hat_{Y|X}(x) + Theta_YZ . s_Z(z)."""
        top_lkl = self._top_lkl_at_posterior(params, x)
        s_z = self.top_man.sufficient_statistic(z)
        return self.top_var.gen_hrm.lkl_fun_man(top_lkl, s_z)

    def sample_posterior(
        self, key: Array, params: Array, x: Array, n: int
    ) -> tuple[Array, Array]:
        """Chain sampling: z ~ q(z | x), then y ~ q(y | z, x). Returns (y, z)."""
        kz, ky = jax.random.split(key)
        q_top = self.approximate_posterior_top(params, x)
        z_samples = self.top_man.sample(kz, q_top, n)

        y_keys = jax.random.split(ky, n)

        def sample_y(subkey: Array, z: Array) -> Array:
            y_params = self.posterior_mid_at(params, x, z)
            return self.mid_man.sample(subkey, y_params, 1)[0]

        y_samples = jax.vmap(sample_y)(y_keys, z_samples)
        return y_samples, z_samples

    # --- Residuals --------------------------------------------------------

    def residual_lower(self, params: Array, y: Array) -> Array:
        """r_Y(y) = rho_Y . s_Y(y) - psi_X(theta_X + Theta_XY . s_Y(y)) + psi_X(theta_X)."""
        _, lower_lkl, _ = self.split_coords(params)
        theta_x, _ = self.split_lower(params)
        rho_y = self.lower_conjugation(params)
        s_y = self.mid_man.sufficient_statistic(y)
        rho_term = jnp.dot(rho_y, s_y)
        lkl_at_y = self.lower_hrm.lkl_fun_man(lower_lkl, s_y)
        psi_at_y = self.obs_man.log_partition_function(lkl_at_y)
        psi_at_bias = self.obs_man.log_partition_function(theta_x)
        return rho_term - psi_at_y + psi_at_bias

    def residual_top_generative(self, params: Array, z: Array) -> Array:
        """r*_Z(z): the top-edge residual at generative bias theta_Y, slope rho0_Z."""
        top, _, _ = self.split_coords(params)
        return self.top_var.conjugation_residual(top, z)

    def residual_top_inner(self, params: Array, x: Array, z: Array) -> Array:
        """r_inner_Z(z; x): top-edge residual at posterior bias with amortized slope.

        Rebuilds a top-edge parameter vector whose observable bias is
        eta_hat_{Y|X}(x) and whose rho is rho^X_Z(x), then reuses
        ``top_var.conjugation_residual`` (which depends only on the bias, the
        interaction Theta_YZ, and rho -- not on the top prior).
        """
        theta_z, _, _ = self.split_top(params)
        top_lkl = self._top_lkl_at_posterior(params, x)
        rho_inner = self.inner_slope(params, x)
        top_params = self.top_var.join_coords(theta_z, top_lkl, rho_inner)
        return self.top_var.conjugation_residual(top_params, z)

    def learning_signal_z(self, params: Array, x: Array, z: Array) -> Array:
        """The z-only part of the learning signal: r*_Z(z) - r_inner_Z(z; x).

        Everything in ``f - c(x)`` except the lower residual r_Y(y). When the
        lower edge is exactly conjugate (``r_Y == 0`` pointwise) this IS the full
        learning signal, and the spike layer integrates out of the ELBO exactly
        (:meth:`marginal_elbo_at`). Subclasses with a different top prior extend
        this (not :meth:`learning_signal`) so both estimators stay consistent.
        """
        return self.residual_top_generative(params, z) - self.residual_top_inner(
            params, x, z
        )

    def learning_signal(
        self, params: Array, x: Array, y: Array, z: Array
    ) -> Array:
        """f(x,y,z) - c(x) = r_Y(y) + r*_Z(z) - r_inner_Z(z; x)."""
        return self.residual_lower(params, y) + self.learning_signal_z(params, x, z)

    def conjugation_baseline(self, params: Array, x: Array) -> Array:
        """c(x), the z-independent ELBO term (exact log-marginal at conjugation)."""
        theta_x, _ = self.split_lower(params)
        theta_y = self.mid_bias(params)
        theta_z, _, _ = self.split_top(params)

        s_x = self.obs_man.sufficient_statistic(x)
        mid_bias = self.posterior_mid_bias(params, x)
        top_post = self.approximate_posterior_top(params, x)

        return (
            jnp.dot(s_x, theta_x)
            - self.obs_man.log_partition_function(theta_x)
            - self.mid_man.log_partition_function(theta_y)
            - self.top_man.log_partition_function(theta_z)
            + self.mid_man.log_partition_function(mid_bias)
            + self.top_man.log_partition_function(top_post)
            + self.obs_man.log_base_measure(x)
        )

    # --- ELBO -------------------------------------------------------------

    def reparam_top_samples(self, key: Array, params: Array, x: Array, n: int) -> Array:
        """Differentiable samples z = mu(x) + L(x) . eps from q(z | x) (pathwise).

        Converts the natural-parameter Gaussian q(z | x) to (mu, Sigma) and
        applies the reparameterization trick: with precision Lambda = C C^T
        (Cholesky), Sigma = C^{-T} C^{-1}, so z = mu + C^{-T} eps. Gradients flow
        through mu and C into the recognition parameters --- the low-variance
        pathwise estimator a continuous Gaussian latent affords, in contrast to
        the score-function estimator required for the discrete spike layer.
        """
        q_top = self.approximate_posterior_top(params, x)
        loc, prec_params = self.top_man.split_location_precision(q_top)
        d = self.top_man.data_dim
        lam = self.top_man.cov_man.rep.to_matrix((d, d), prec_params)
        lam = 0.5 * (lam + lam.T)
        chol = jnp.linalg.cholesky(lam)
        mu = jax.scipy.linalg.cho_solve((chol, True), loc)
        eps = jax.random.normal(key, (n, d))
        z = mu + jax.scipy.linalg.solve_triangular(chol.T, eps.T, lower=False).T
        return z

    def elbo_at(
        self, key: Array, params: Array, x: Array, n_samples: int,
        reparam_z: bool = False,
    ) -> Array:
        """Standard-form ELBO L(x) = c(x) + E_q[r_Y + r*_Z - r_inner_Z].

        Baseline c(x) analytic; residual term by Monte Carlo over chain samples.
        The estimated *value* is the same either way; the flag selects the
        gradient estimator for the continuous top latent:

        - ``reparam_z=False``: score-function estimator for both y and z
          (``score - sg(score)`` surrogate on the full chain log q).
        - ``reparam_z=True``: pathwise (reparameterized) gradient through the
          Gaussian z samples; the score-function correction then covers only the
          discrete y layer, whose sampling distribution q(y | z, x) still moves
          with z through the correction's log q(y | z, x) term.
        """
        if not reparam_z:
            y_samples, z_samples = self.sample_posterior(key, params, x, n_samples)
            y_samples = jax.lax.stop_gradient(y_samples)
            z_samples = jax.lax.stop_gradient(z_samples)

            c_x = self.conjugation_baseline(params, x)

            signal = jax.vmap(lambda y, z: self.learning_signal(params, x, y, z))(
                y_samples, z_samples
            )
            log_q = jax.vmap(lambda y, z: self.log_q(params, x, y, z))(
                y_samples, z_samples
            )

            direct = jnp.mean(signal)
            sig_sg = jax.lax.stop_gradient(signal)
            b = jnp.mean(sig_sg)
            score = jnp.mean((sig_sg - b) * log_q)
            return c_x + direct + score - jax.lax.stop_gradient(score)

        kz, ky = jax.random.split(key)
        z_samples = self.reparam_top_samples(kz, params, x, n_samples)  # NOT stopped
        y_keys = jax.random.split(ky, n_samples)

        def sample_y(subkey: Array, z: Array) -> Array:
            return self.mid_man.sample(subkey, self.posterior_mid_at(params, x, z), 1)[0]

        y_samples = jax.lax.stop_gradient(jax.vmap(sample_y)(y_keys, z_samples))

        c_x = self.conjugation_baseline(params, x)
        signal = jax.vmap(lambda y, z: self.learning_signal(params, x, y, z))(
            y_samples, z_samples
        )
        log_qy = jax.vmap(
            lambda y, z: self.mid_man.log_density(self.posterior_mid_at(params, x, z), y)
        )(y_samples, z_samples)

        direct = jnp.mean(signal)  # pathwise z-gradient rides through signal
        sig_sg = jax.lax.stop_gradient(signal)
        b = jnp.mean(sig_sg)
        score = jnp.mean((sig_sg - b) * log_qy)  # y-layer REINFORCE only
        return c_x + direct + score - jax.lax.stop_gradient(score)

    def mean_elbo(
        self, key: Array, params: Array, xs: Array, n_samples: int,
        reparam_z: bool = False,
    ) -> Array:
        """Mean ELBO over a batch."""
        keys = jax.random.split(key, xs.shape[0])
        elbos = jax.vmap(
            lambda k, x: self.elbo_at(k, params, x, n_samples, reparam_z=reparam_z)
        )(keys, xs)
        return jnp.mean(elbos)

    def marginal_elbo_at(self, key: Array, params: Array, x: Array, n_samples: int) -> Array:
        """Exact-N ELBO: L(x) = c(x) + E_{q(z|x)}[r*_Z - r_inner_Z], z pathwise.

        Requires an analytically conjugate lower edge: with ``r_Y == 0``
        pointwise, the learning signal is y-independent, so the expectation over
        the spike layer q(y | z, x) is the identity and the discrete layer never
        needs to be sampled. The only Monte Carlo left is the Gaussian top latent
        via :meth:`reparam_top_samples`, whose gradient is pathwise -- no
        score-function estimator anywhere in the objective.
        """
        if not self.lower_edge_exact:
            raise NotImplementedError(
                "marginal_elbo_at requires an exactly conjugate lower edge "
                "(conv harmonium); use elbo_at for learned-rho_Y models."
            )
        z_samples = self.reparam_top_samples(key, params, x, n_samples)  # NOT stopped
        c_x = self.conjugation_baseline(params, x)
        sig = jax.vmap(lambda z: self.learning_signal_z(params, x, z))(z_samples)
        return c_x + jnp.mean(sig)

    def mean_marginal_elbo(
        self, key: Array, params: Array, xs: Array, n_samples: int
    ) -> Array:
        """Mean exact-N ELBO over a batch."""
        keys = jax.random.split(key, xs.shape[0])
        elbos = jax.vmap(
            lambda k, x: self.marginal_elbo_at(k, params, x, n_samples)
        )(keys, xs)
        return jnp.mean(elbos)

    def log_q(self, params: Array, x: Array, y: Array, z: Array) -> Array:
        """log q(y, z | x) = log q(z | x) + log q(y | z, x) (chain factorization)."""
        q_top = self.approximate_posterior_top(params, x)
        log_qz = self.top_man.log_density(q_top, z)
        q_mid = self.posterior_mid_at(params, x, z)
        log_qy = self.mid_man.log_density(q_mid, y)
        return log_qz + log_qy

    # --- Conjugation regularizers ----------------------------------------

    def prior_conjugation_loss_components(
        self, key: Array, params: Array, n_samples: int
    ) -> tuple[Array, Array]:
        """The two generative conjugation penalties separately: ``(Var_p[r_Y],
        Var_p[r*_Z])`` over ancestral samples.

        ``r_Y`` is the *bottom* (Gaussian-Boltzmann) edge residual -- the
        non-analytically-conjugate one, whose quadratic ``psi_X`` a sparse
        Boltzmann can't absorb -- and ``r*_Z`` the *top* (Boltzmann population
        code ``Y|Z``) edge, the genuinely variational-conjugate one. Splitting
        them lets a caller regularize the top toward conjugation while leaving the
        expressive bottom decoder free. Samples ``z ~ p(z)``, ``y ~ p(y|z)``;
        autodiff flows through the residuals at fixed samples.
        """
        kz, ky = jax.random.split(key)
        theta_z, top_lkl, _ = self.split_top(params)
        z_samples = jax.lax.stop_gradient(self.top_man.sample(kz, theta_z, n_samples))
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

    def prior_conjugation_loss(
        self, key: Array, params: Array, n_samples: int
    ) -> Array:
        """Generative-side penalty Var_p[r_Y] + Var_p[r*_Z] over ancestral samples.

        Drives the *generative* model toward conjugation. See
        :meth:`prior_conjugation_loss_components` for the per-edge split.
        """
        var_r_y, var_r_z = self.prior_conjugation_loss_components(key, params, n_samples)
        return var_r_y + var_r_z

    def recognition_inner_loss_at(
        self, key: Array, params: Array, x: Array, n_samples: int
    ) -> Array:
        """Inner-slope penalty Var_q[r_inner_Z] at x, the MLP's tractability signal.

        The residual depends only on z, so z is drawn from q(z | x) directly --
        no need to run the y-chain of :meth:`sample_posterior` (same marginal).
        """
        q_top = self.approximate_posterior_top(params, x)
        z_samples = jax.lax.stop_gradient(self.top_man.sample(key, q_top, n_samples))
        r_inner = jax.vmap(lambda z: self.residual_top_inner(params, x, z))(z_samples)
        return jnp.var(r_inner)

    def mean_recognition_inner_loss(
        self, key: Array, params: Array, xs: Array, n_samples: int
    ) -> Array:
        """Mean inner-slope penalty over a batch."""
        keys = jax.random.split(key, xs.shape[0])
        losses = jax.vmap(
            lambda k, x: self.recognition_inner_loss_at(k, params, x, n_samples)
        )(keys, xs)
        return jnp.mean(losses)

    # --- Generative EF contract on the joint (x, y, z) --------------------

    @property
    @override
    def data_dim(self) -> int:
        return self.obs_man.data_dim + self.mid_man.data_dim + self.top_man.data_dim

    @override
    def sufficient_statistic(self, x: Array) -> Array:
        raise NotImplementedError("Joint sufficient statistic unused for this model.")

    @override
    def log_base_measure(self, x: Array) -> Array:
        obs = x[..., : self.obs_man.data_dim]
        mid = x[..., self.obs_man.data_dim : self.obs_man.data_dim + self.mid_man.data_dim]
        top = x[..., self.obs_man.data_dim + self.mid_man.data_dim :]
        return (
            self.obs_man.log_base_measure(obs)
            + self.mid_man.log_base_measure(mid)
            + self.top_man.log_base_measure(top)
        )

    @override
    def sample(self, key: Array, params: Array, n: int = 1) -> Array:
        """Ancestral sample z ~ p(z), y ~ p(y | z), x ~ p(x | y). Returns [x, y, z]."""
        kz, ky, kx = jax.random.split(key, 3)
        theta_z, top_lkl, _ = self.split_top(params)
        _, lower_lkl, _ = self.split_coords(params)

        z_samples = self.top_man.sample(kz, theta_z, n)

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

    def log_density_joint(self, params: Array, x: Array, y: Array, z: Array) -> Array:
        """log p(x, y, z) = log p(z) + log p(y | z) + log p(x | y)."""
        theta_z, top_lkl, _ = self.split_top(params)
        _, lower_lkl, _ = self.split_coords(params)

        log_pz = self.top_man.log_density(theta_z, z)
        s_z = self.top_man.sufficient_statistic(z)
        y_params = self.top_var.gen_hrm.lkl_fun_man(top_lkl, s_z)
        log_py = self.mid_man.log_density(y_params, y)
        s_y = self.mid_man.sufficient_statistic(y)
        x_params = self.lower_hrm.lkl_fun_man(lower_lkl, s_y)
        log_px = self.obs_man.log_density(x_params, x)
        return log_pz + log_py + log_px

    # --- Initialization ---------------------------------------------------

    @override
    def initialize(self, key: Array, location: float = 0.0, shape: float = 0.1) -> Array:
        """Initialize with zero conjugation vectors and a small-weight MLP."""
        k_top, k_low, k_mlp = jax.random.split(key, 3)
        top = self.top_var.initialize(k_top, location, shape)
        low_hrm = self.lower_hrm.initialize(k_low, location, shape)
        theta_x, theta_xy, _ = self.lower_hrm.split_coords(low_hrm)
        lower_lkl = self.lower_hrm.lkl_fun_man.join_coords(theta_x, theta_xy)
        rho_y = jnp.zeros(self.mid_man.dim)
        phi = self.mlp_man.glorot_initialize(k_mlp)
        recog = self.recog_man.join_coords(rho_y, phi)
        return self.join_coords(top, lower_lkl, recog)

    @override
    def initialize_from_sample(
        self, key: Array, sample: Array, location: float = 0.0, shape: float = 0.1
    ) -> Array:
        """Like :meth:`initialize` but seeds the observable bias from data."""
        k_top, k_low, k_mlp = jax.random.split(key, 3)
        top = self.top_var.initialize(k_top, location, shape)
        low_hrm = self.lower_hrm.initialize_from_sample(k_low, sample, location, shape)
        theta_x, theta_xy, _ = self.lower_hrm.split_coords(low_hrm)
        lower_lkl = self.lower_hrm.lkl_fun_man.join_coords(theta_x, theta_xy)
        rho_y = jnp.zeros(self.mid_man.dim)
        phi = self.mlp_man.glorot_initialize(k_mlp)
        recog = self.recog_man.join_coords(rho_y, phi)
        return self.join_coords(top, lower_lkl, recog)


# --- Node-only interaction embedding ----------------------------------------


@dataclass(frozen=True)
class BoltzmannNodeEmbedding[B: Boltzmann](LinearEmbedding[Bernoullis, B]):
    """Expose only the first-order (node-activity) subspace of a Boltzmann.

    A Boltzmann's sufficient statistic packs node activities ``y_i`` on the
    coupling *diagonal* and pairwise edge products ``y_i y_j`` off-diagonal
    (:meth:`Boltzmann.split_couplings`). Routing the lower interaction through
    this embedding makes it couple the observable to the nodes ``y_i`` *only* --
    the conventional observable<->hidden-unit wiring -- with the pairwise
    structure left in the Boltzmann prior.

    Coupling to the full statistic (edges included, an ``IdentityEmbedding``) is
    what breaks conjugacy: the observable mean would depend on second-order
    latent stats, so the quadratic ``psi_X`` squared against them produces
    fourth-order terms in ``y`` that no latent (Boltzmann or Gaussian) can
    absorb. Coupling to nodes only keeps ``psi_X`` quadratic in ``y``, so a
    sufficiently rich prior graph can absorb it.
    """

    bol_man: B

    @property
    @override
    def amb_man(self) -> B:
        return self.bol_man

    @property
    @override
    def sub_man(self) -> Bernoullis:
        return self.bol_man.loc_man

    @override
    def project(self, coords: Array) -> Array:
        diag, _ = self.bol_man.split_couplings(coords)
        return diag

    @override
    def embed(self, coords: Array) -> Array:
        off = jnp.zeros(self.bol_man.dim - self.bol_man.data_dim)
        return self.bol_man.join_couplings(coords, off)


# --- Exact analytic conv Gaussian-Boltzmann lower edge ----------------------


@dataclass(frozen=True)
class ConvBoltzmannHarmonium(SymmetricConjugated[Normal[Diagonal], DiagonalBoltzmann]):
    """Diagonal-Gaussian observable, diagonal-Boltzmann latent, conv interaction.

    An *analytically conjugate* lower edge ``p(x | N)``: with a non-overlapping
    conv (kernel == stride) the decoder columns have disjoint support, so
    ``1/2 W^T Sigma W`` is diagonal and the diagonal spike prior is exactly
    conjugate. :meth:`conjugation_parameters` returns the closed-form ``rho_N``
    (linear + diagonal-quadratic part of ``psi_X(theta_X + Theta . s_N)``),
    validated against brute-force enumeration to machine precision. Slotting this
    in as the lower harmonium makes ``r_Y == 0`` and ``p(N | x)`` exact --- no
    learned conjugation vector on this edge.
    """

    _int_man: EmbeddedLinearMap[DiagonalBoltzmann, Normal[Diagonal]]
    _lat_man: DiagonalBoltzmann

    @property
    @override
    def int_man(self) -> LinearMap[DiagonalBoltzmann, Normal[Diagonal]]:
        return self._int_man

    @property
    @override
    def lat_man(self) -> DiagonalBoltzmann:
        return self._lat_man

    @override
    def conjugation_parameters(self, lkl_params: Array) -> Array:
        obs_bias, _ = self.lkl_fun_man.split_coords(lkl_params)
        obs_loc, obs_prec = self.obs_man.split_location_precision(obs_bias)
        sigma = 1.0 / jnp.asarray(obs_prec)  # diagonal covariance
        mu = sigma * obs_loc  # observable mean
        loc0, _ = self.obs_man.split_location_precision(obs_bias)

        n = self.lat_man.data_dim

        def col(i: Array) -> Array:
            s_n = jnp.zeros(n).at[i].set(1.0)
            loc, _ = self.obs_man.split_location_precision(self.lkl_fun_man(lkl_params, s_n))
            return loc - loc0

        w = jax.vmap(col)(jnp.arange(n)).T  # (obs_dim, n_nodes)
        return w.T @ mu + 0.5 * (w.T * w.T) @ sigma


@dataclass(frozen=True)
class ConvChordalBoltzmannHarmonium(SymmetricConjugated[Normal[Diagonal], ChordalBoltzmann]):
    """Diagonal-Gaussian observable, *chordal*-Boltzmann latent, conv interaction.

    The overlapping-kernel counterpart of :class:`ConvBoltzmannHarmonium`: with
    kernel > stride the decoder columns overlap, so ``W^T Sigma W`` has
    off-diagonal support --- exactly the conv's induced coupling graph, which the
    chordal latent's edge set contains by construction. Conjugation stays exact:

    - node part:  ``rho_i  = (W^T mu)_i + 1/2 (W^T Sigma W)_ii``
    - edge part:  ``rho_ij = (W^T Sigma W)_ij`` per chordal edge (the 1/2 cancels
      against the two off-diagonal terms; fill-in edges get exactly 0).

    This buys smooth, seam-free receptive fields while keeping ``r_Y == 0``.
    """

    _int_man: EmbeddedLinearMap[ChordalBoltzmann, Normal[Diagonal]]
    _lat_man: ChordalBoltzmann

    @property
    @override
    def int_man(self) -> LinearMap[ChordalBoltzmann, Normal[Diagonal]]:
        return self._int_man

    @property
    @override
    def lat_man(self) -> ChordalBoltzmann:
        return self._lat_man

    @override
    def conjugation_parameters(self, lkl_params: Array) -> Array:
        obs_bias, _ = self.lkl_fun_man.split_coords(lkl_params)
        obs_loc, obs_prec = self.obs_man.split_location_precision(obs_bias)
        sigma = 1.0 / jnp.asarray(obs_prec)
        mu = sigma * obs_loc
        loc0, _ = self.obs_man.split_location_precision(obs_bias)

        n = self.lat_man.data_dim

        def col(i: Array) -> Array:
            s_n = jnp.zeros(self.lat_man.dim).at[i].set(1.0)  # node slots come first
            loc, _ = self.obs_man.split_location_precision(self.lkl_fun_man(lkl_params, s_n))
            return loc - loc0

        w = jax.vmap(col)(jnp.arange(n)).T  # (obs_dim, n_nodes)
        wsw = w.T @ (sigma[:, None] * w)  # (n, n) = W^T Sigma W
        rho_diag = w.T @ mu + 0.5 * jnp.diagonal(wsw)
        edges = self.lat_man.junction_tree.chordal_edges_arr
        rho_off = (
            wsw[edges[:, 0], edges[:, 1]] if edges.shape[0] > 0 else jnp.zeros(0)
        )
        return self.lat_man.join_couplings(rho_diag, rho_off)


# --- Factory ----------------------------------------------------------------


def _chain_edges(n: int) -> list[tuple[int, int]]:
    return [(i, i + 1) for i in range(n - 1)]


def build_boltzmann_gaussian_hierarchy(
    obs_man: Differentiable,
    n_mid: int,
    top_dim: int,
    mid_kind: str = "chain",
    mid_edges: Sequence[tuple[int, int]] | None = None,
    mlp_hidden: tuple[int, ...] = (128,),
    mlp_activation: Callable[[Array], Array] = jax.nn.gelu,
    max_treewidth: int | None = None,
    obs_location_only: bool = False,
    couple_edges: bool = False,
) -> VariationalHierarchical:
    """Assemble ``p(x, y, z)`` with a (chordal) Boltzmann middle and Gaussian top.

    Args:
        obs_man: observable manifold M_X (Normal / Binomials / Poissons).
        n_mid: number of middle Boltzmann spike units.
        top_dim: dimension of the top Gaussian latent.
        mid_kind: ``"chain"``, ``"chordal"``, or ``"diagonal"``.
        mid_edges: seed edges for ``"chordal"`` (defaults to a chain).
        mlp_hidden: hidden widths of the amortized inner-slope MLP.
        mlp_activation: MLP activation.
        max_treewidth: triangulation cap for chordal graphs.
        obs_location_only: if ``True`` (Normal observable only), the lower
            interaction ``Theta_XY`` drives *only* the observable mean, leaving
            the covariance a global bias parameter. This is the fixed-covariance
            Gaussian-Boltzmann model, which avoids the variance-collapse
            degeneracy of a fully y-dependent Normal likelihood.
        couple_edges: if ``False`` (default), the observable couples to the
            Boltzmann *node activities* only (:class:`BoltzmannNodeEmbedding`) --
            the conjugation-friendly, conventional wiring. If ``True``, it
            couples to the full Boltzmann statistic (nodes + edges), whose
            second-order terms make the model non-conjugate for any latent.
    """
    mid_man: Boltzmann
    if mid_kind == "diagonal":
        mid_man = DiagonalBoltzmann(n_neurons=n_mid)
    elif mid_kind == "chordal":
        edges = list(mid_edges) if mid_edges is not None else _chain_edges(n_mid)
        mid_man = ChordalBoltzmann.from_edges(n_mid, edges, max_treewidth)
    else:  # "chain"
        mid_man = ChainBoltzmann.from_edges(n_mid, _chain_edges(n_mid))

    # Upper edge p(y, z): Boltzmann observable, Gaussian latent.
    top_var = BoltzmannPopulationCode(BoltzmannNormalHarmonium(mid_man, top_dim))

    # Lower edge harmonium p(x, y): observable X, Boltzmann latent Y. When
    # obs_location_only, the observable side embeds a Euclidean location into the
    # Normal with zero shape, so Theta_XY only shifts the mean.
    obs_emb = (
        GeneralizedGaussianLocationEmbedding(obs_man)
        if obs_location_only
        else IdentityEmbedding(obs_man)
    )
    lat_emb = IdentityEmbedding(mid_man) if couple_edges else BoltzmannNodeEmbedding(mid_man)
    lower_int = EmbeddedMap(Rectangular(), lat_emb, obs_emb)
    lower_hrm = ConcreteHarmonium(lower_int)

    mlp = MultilayerPerceptron(mid_man, full_normal(top_dim), mlp_hidden, mlp_activation)
    recog = HierarchicalRecognition(mid_man, mlp)

    return VariationalHierarchical(
        top_var=top_var, lower_hrm=lower_hrm, recog_man=recog
    )


def conv_hierarchy_components(
    obs_man: Differentiable,
    in_lattice: tuple[int, ...],
    stride: tuple[int, ...],
    kernel_shape: tuple[int, ...],
    top_dim: int,
    in_channels: int = 1,
    out_channels: int = 1,
    prior_graph: str = "chordal",
    max_treewidth: int | None = None,
    mlp_hidden: tuple[int, ...] = (128,),
    mlp_activation: Callable[[Array], Array] = jax.nn.gelu,
) -> tuple[
    BoltzmannPopulationCode,
    ConvBoltzmannHarmonium | ConvChordalBoltzmannHarmonium,
    HierarchicalRecognition,
]:
    """Build the three shared slots ``(top_var, lower_hrm, recog)`` of a conv hierarchy.

    Shared by :func:`build_conv_boltzmann_gaussian_hierarchy` and the mixture-top
    factory (:mod:`.hierarchical_mixture`), which assemble the identical
    conv/chordal lower edge and differ only in the top prior.

    The lower interaction ``Theta_XY`` is a strided, multi-channel transposed
    convolution (:class:`LatticeConvolution`) from a coarse latent lattice to the
    observable location. It couples the observable to Boltzmann *node activities*
    only (:class:`BoltzmannNodeEmbedding`) and drives *only* the observable mean
    (:class:`GeneralizedGaussianLocationEmbedding`), so the model is a
    convolutional linear-Gaussian likelihood with a binary code.

    ``prior_graph`` decides how the middle Boltzmann's couplings relate to the
    convolution's *induced coupling graph* (:meth:`LatticeConvolution.\
induced_coupling_graph`), the support the conjugation parameter
    ``P^sigma = 1/2 W^T Sigma W`` can occupy:

    - ``"chordal"``: prior graph = triangulated induced graph, so ``P^sigma`` is
      *representable on the couplings* -> analytic/exact conjugation. But with
      ``in_channels`` channels the induced graph tensors a complete C-block, so
      treewidth scales ~x``in_channels`` -- exact junction-tree inference caps
      channels at ~2-3.
    - ``"diagonal"``: independent-node prior (``DiagonalBoltzmann``); ``psi_Y`` is
      trivial and treewidth is ~0, so **channel count is free**. ``P^sigma``'s
      off-diagonal part is then *not* structurally representable, so conjugation is
      no longer analytic -- it must be driven down *softly* (``rho_Y`` node terms +
      the generative conjugation penalty). This is the many-channel regime.

    Args:
        obs_man: observable manifold M_X, a Normal over ``prod(out_lattice) *
            out_channels`` coordinates (``out_lattice = in_lattice * stride``).
        in_lattice: coarse latent lattice extent (e.g. ``(7, 7)``).
        stride: upsampling factor per axis (e.g. ``(4, 4)`` -> 28x28).
        kernel_shape: convolution kernel extent (e.g. ``(6, 6)``).
        top_dim: dimension of the top Gaussian latent.
        in_channels: latent feature channels per lattice cell (= number of kernels).
        out_channels: observable channels per output cell.
        prior_graph: ``"chordal"`` (exact conjugation, treewidth ~x channels) or
            ``"diagonal"`` (free channels, soft conjugation).
        max_treewidth: triangulation cap for the ``"chordal"`` middle graph.
    """
    n_mid = prod(in_lattice) * in_channels
    bernoullis = Bernoullis(n_neurons=n_mid)
    conv = LatticeConvolution.create(
        bernoullis, obs_man.loc_man, in_lattice, stride, kernel_shape,
        in_channels, out_channels,
    )

    mid_man: Boltzmann
    if prior_graph == "diagonal":
        mid_man = DiagonalBoltzmann(n_neurons=n_mid)
    elif prior_graph == "chordal":
        mid_man = ChordalBoltzmann.from_edges(n_mid, conv.induced_edges(), max_treewidth)
    else:
        raise ValueError(f"prior_graph must be 'chordal' or 'diagonal', got {prior_graph!r}")
    top_var = BoltzmannPopulationCode(BoltzmannNormalHarmonium(mid_man, top_dim))

    lat_emb = BoltzmannNodeEmbedding(mid_man)
    obs_emb = GeneralizedGaussianLocationEmbedding(obs_man)
    lower_int = EmbeddedLinearMap(conv, lat_emb, obs_emb)
    # Both priors get an exact analytically conjugate lower edge (r_Y == 0):
    # diagonal for non-overlapping kernels, chordal (node+edge rho) for overlapping.
    lower_hrm: ConvBoltzmannHarmonium | ConvChordalBoltzmannHarmonium
    if prior_graph == "diagonal":
        lower_hrm = ConvBoltzmannHarmonium(lower_int, mid_man)
    else:
        assert isinstance(mid_man, ChordalBoltzmann)
        lower_hrm = ConvChordalBoltzmannHarmonium(lower_int, mid_man)

    mlp = MultilayerPerceptron(mid_man, full_normal(top_dim), mlp_hidden, mlp_activation)
    recog = HierarchicalRecognition(mid_man, mlp)

    return top_var, lower_hrm, recog


def build_conv_boltzmann_gaussian_hierarchy(
    obs_man: Differentiable,
    in_lattice: tuple[int, ...],
    stride: tuple[int, ...],
    kernel_shape: tuple[int, ...],
    top_dim: int,
    in_channels: int = 1,
    out_channels: int = 1,
    prior_graph: str = "chordal",
    max_treewidth: int | None = None,
    mlp_hidden: tuple[int, ...] = (128,),
    mlp_activation: Callable[[Array], Array] = jax.nn.gelu,
) -> VariationalHierarchical:
    """Assemble ``p(x, y, z)`` with a **convolutional** lower decoder.

    See :func:`conv_hierarchy_components` for the architecture and argument
    semantics; this factory pairs those components with the single-Gaussian top
    prior held inside ``top_var``.
    """
    top_var, lower_hrm, recog = conv_hierarchy_components(
        obs_man, in_lattice, stride, kernel_shape, top_dim,
        in_channels=in_channels, out_channels=out_channels,
        prior_graph=prior_graph, max_treewidth=max_treewidth,
        mlp_hidden=mlp_hidden, mlp_activation=mlp_activation,
    )
    return VariationalHierarchical(top_var=top_var, lower_hrm=lower_hrm, recog_man=recog)
