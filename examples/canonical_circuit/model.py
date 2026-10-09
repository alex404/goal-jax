"""The three-layer canonical circuit $X - N - Z$ and its exact computations.

The circuit is a hierarchical harmonium over an observable $x \\in \\mathbb R^{d_X}$, a full Boltzmann
machine $n \\in \\{0, 1\\}^{d_N}$, and a Gaussian $z \\in \\mathbb R^{d_Z}$. The edge $X - N$ is a
:class:`~goal.models.BoltzmannLGM`, which is conjugated, with conjugation parameters $\\rho_N$ in closed
form. The edge $N - Z$ is a :class:`PopulationCodeHarmonium`, whose conjugation parameters
$\\rho_Z(\\beta)$ are a weighted least-squares fit at the Gauss-Hermite nodes of a reference Gaussian,
found by fixed-point iteration.

The neurons are summed exactly, and so the log-likelihood, the posterior over $z$ and the joint samples
are exact. The log-likelihood identity of the variational framework,

$$\\log p_X(x) = c(x) + \\log \\mathbb E_q[e^{-r^X_Z}] - \\log \\mathbb E_{\\tilde p_Z}[e^{-r_Z}],$$

is evaluated by Gauss-Hermite quadrature at the nodes of the recognition model and of the prior, and is
checked against the exact value. Experimental code: see ``scratch/variational-conjugation/canonical-circuit.tex``.
"""

from dataclasses import dataclass
from typing import Literal, override

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.scipy.special import logsumexp

from goal.geometry import (
    CliqueMap,
    CrossTerm,
    Harmonium,
    IdentityEmbedding,
    PositiveDefinite,
    Rectangular,
)
from goal.models import BoltzmannLGM, FullBoltzmann, FullNormal, full_normal
from goal.models.harmonium.lgm import GeneralizedGaussianLocationEmbedding

type Unconstrained = dict[str, Array]
type NoiseCorrelations = Literal["harmonium", "none", "chain", "full"]


@dataclass(frozen=True)
class PopulationCodeHarmonium(Harmonium[FullBoltzmann, FullNormal]):
    """Harmonium with a Boltzmann observable and a Normal latent, coupled through the node activities only.

    Neuron $i$ has the logit $\\theta_{N,i} + \\mathbf w_i \\cdot \\mathbf s_Z(z)$, so the couplings of
    the Boltzmann machine do not depend on $z$, and $p(z \\mid n)$ is Gaussian with natural parameters
    affine in $n$. Unlike :class:`~goal.models.BoltzmannNormalHarmonium`, the pairwise statistics of the
    Boltzmann machine are not coupled to $z$.
    """

    # Fields

    n_neurons: int
    lat_dim: int

    # Overrides

    @property
    @override
    def crs_trms(self) -> tuple[CrossTerm, ...]:
        """The node activities and the latent, coupled."""
        int_map = CliqueMap(
            Rectangular(),
            GeneralizedGaussianLocationEmbedding(self.obs_man),
            IdentityEmbedding(self.pst_man),
        )
        return (CrossTerm((0,), (0,), int_map),)

    @property
    @override
    def obs_man(self) -> FullBoltzmann:
        return FullBoltzmann(self.n_neurons)

    @property
    @override
    def pst_man(self) -> FullNormal:
        return full_normal(self.lat_dim)


@dataclass(frozen=True)
class CanonicalCircuit:
    """The three-layer circuit, with its parameters stored as $[\\theta_{XN} | \\Theta_{NZ} | \\theta_Z]$.

    The first block is the natural parameters of the :class:`~goal.models.BoltzmannLGM` (which holds
    $\\theta_N$), the second the interaction of the population code, and the third the bias of $Z$.
    """

    # Fields

    obs_dim: int
    n_neurons: int
    lat_dim: int

    noise_correlations: NoiseCorrelations
    """The couplings of the neurons given $Z$ under the generative model.

    For ``"none"``, ``"chain"`` and ``"full"``, $\\theta^*_N = \\theta_N + \\rho_N$ is stored directly, with
    no couplings, couplings between neurons $i$ and $i + 1$ only, or all couplings, and $\\theta_N =
    \\theta^*_N - \\rho_N$. These are the stimulus-independent noise correlations of $p(n \\mid z)$. For
    ``"harmonium"``, $\\theta_N$ is stored with free couplings, which is the same family as ``"full"`` in
    other coordinates.
    """

    latent: bool
    """Whether the neurons interact with $Z$. Without, $\\Theta_{NZ} = 0$ and the circuit reduces to
    the Gaussian-Boltzmann harmonium $X - N$."""

    exact_reference: bool
    """Whether the reference of the least-squares conjugation parameters is the moment-matched exact
    distribution over $z$ (available by enumeration), or the fixed point $\\theta_Z + \\rho_Z(\\beta)$."""

    n_nodes: int = 12
    """Gauss-Hermite nodes per latent dimension."""

    n_fixed_point: int = 4
    """Fixed-point iterations for the reference of the least-squares conjugation parameters."""

    min_precision: float = 1e-3
    """Floor on the eigenvalues of the recognition and prior precisions (an example-level guard that keeps them positive definite)."""

    # Manifolds

    @property
    def lgm(self) -> BoltzmannLGM[PositiveDefinite]:
        return BoltzmannLGM(self.obs_dim, PositiveDefinite(), self.n_neurons)

    @property
    def pch(self) -> PopulationCodeHarmonium:
        return PopulationCodeHarmonium(self.n_neurons, self.lat_dim)

    @property
    def obs_man(self):
        return self.lgm.obs_man

    @property
    def blz_man(self) -> FullBoltzmann:
        return self.lgm.pst_man

    @property
    def lat_man(self) -> FullNormal:
        return self.pch.pst_man

    @property
    def dim(self) -> int:
        return self.lgm.dim + self.pch.int_man.dim + self.lat_man.dim

    # Parameters

    def split_params(self, params: Array) -> tuple[Array, Array]:
        """Split circuit parameters into the parameters of the two harmoniums."""
        n_lgm, n_int = self.lgm.dim, self.pch.int_man.dim
        lgm_params = params[:n_lgm]
        nz_params = params[n_lgm : n_lgm + n_int]
        z_params = params[n_lgm + n_int :]
        _, _, n_params = self.lgm.split_coords(lgm_params)
        return lgm_params, self.pch.join_coords(n_params, nz_params, z_params)

    def constrain(self, u: Unconstrained) -> Array:
        """Map unconstrained parameters to circuit natural parameters.

        Unless the noise correlations are ``"harmonium"``, the biases and ``couplings`` below are those of
        $\\theta^*_N$.
        Precisions are stored through Cholesky factors with log diagonals. The population code is stored
        by preferred stimuli $\\mu_i$, a shared tuning width $A$ and peak logits $u_i$, so that the logit of
        neuron $i$ is $u_i - \\frac{1}{2}(z - \\mu_i) \\cdot A \\cdot (z - \\mu_i)$ plus the input from $X$.
        """
        obs_man, lat_man, blz = self.obs_man, self.lat_man, self.blz_man
        prc_x = _chol_matrix(u["x_chol"])
        obs_params = obs_man.join_location_precision(
            prc_x @ u["x_mean"], obs_man.cov_man.from_matrix(prc_x)
        )
        (xn_map,) = self.lgm.crs_maps
        xn_params = xn_map.from_matrix(u["xn"])

        a = _chol_matrix(u["a_chol"])
        mus = u["preferred"]
        ws = mus @ a
        biases = u["peak"] - 0.5 * jnp.einsum("ni,ij,nj->n", mus, a, mus)
        off_diag = (
            jnp.zeros(blz.dim - self.n_neurons)
            .at[self._coupling_indices]
            .set(u["couplings"])
        )
        n_params = blz.join_couplings(biases, off_diag)
        if self.noise_correlations != "harmonium":
            lkl_params = self.lgm.lkl_fun_man.join_coords(obs_params, xn_params)
            n_params = n_params - self.lgm.conjugation_parameters(lkl_params)
        a_coords = lat_man.cov_man.from_matrix(a)
        rows = jax.vmap(lat_man.join_location_precision, in_axes=(0, None))(
            ws, a_coords
        )
        (nz_map,) = self.pch.crs_maps
        nz_params = nz_map.from_matrix(rows) * float(self.latent)

        prc_z = _chol_matrix(u["z_chol"])
        z_params = lat_man.join_location_precision(
            prc_z @ u["z_mean"], lat_man.cov_man.from_matrix(prc_z)
        )
        lgm_params = self.lgm.join_coords(obs_params, xn_params, n_params)
        return jnp.concatenate([lgm_params, nz_params, z_params])

    def initialize(self, key: Array, xs: Array, z_range: float = 2.0) -> Unconstrained:
        """Initialize the readout from the data and the tuning curves tiling $[-r, r]^{d_Z}$.

        Neighbouring tuning curves overlap by one spacing, the peak firing rates are low, and the couplings
        are zero. The readout has a quarter of the data covariance and standard normal interactions with
        the neurons: with small interactions, $p(n \\mid x)$ does not depend on $x$ and the gradient of the
        log-likelihood with respect to the interactions vanishes.
        """
        if self.lat_dim != 1:
            raise NotImplementedError(
                "Tiling initialization is implemented for d_Z = 1."
            )
        mean = jnp.mean(xs, axis=0)
        cov = jnp.cov(xs.T) / 4 + 1e-6 * jnp.eye(self.obs_dim)
        preferred = jnp.linspace(-z_range, z_range, self.n_neurons)[:, None]
        spacing = 2 * z_range / (self.n_neurons - 1)
        return {
            "x_mean": mean,
            "x_chol": _chol_coords(jnp.linalg.inv(cov)),
            "xn": jax.random.normal(key, (self.obs_dim, self.n_neurons)),
            "peak": jnp.full(self.n_neurons, -1.0),
            "couplings": jnp.zeros(len(self._coupling_indices)),
            "preferred": preferred,
            "a_chol": _chol_coords(jnp.eye(1) / spacing**2),
            "z_mean": jnp.zeros(1),
            "z_chol": _chol_coords(jnp.eye(1)),
        }

    # Conjugation of the population code edge

    def shifted_log_partitions(self, nz_params: Array, beta: Array, zs: Array) -> Array:
        """$\\psi_N(\\beta + \\Theta_{NZ} \\cdot \\mathbf s_Z(z_k))$ at nodes $z_k$, for Boltzmann natural parameters $\\beta$.

        The interaction shifts only the biases, so the energies of the states at $\\beta$ are computed
        once, and each node adds $n \\cdot \\Theta_{NZ} \\cdot \\mathbf s_Z(z_k)$.
        """
        blz = self.blz_man
        energies = jax.vmap(blz.sufficient_statistic)(blz.states) @ beta
        (nz_map,) = self.pch.crs_maps
        sts = jax.vmap(self.lat_man.sufficient_statistic)(zs)
        shifts = sts @ nz_map.to_matrix(nz_params).T
        return logsumexp(energies[None, :] + shifts @ blz.states.T, axis=1)

    def quadrature(self, nrm_params: Array) -> tuple[Array, Array]:
        """Gauss-Hermite nodes and weights (summing to one) of a Gaussian in natural parameters."""
        mean, cov = _mean_covariance(self.lat_man, nrm_params)
        us, ws = _standard_nodes(self.lat_dim, self.n_nodes)
        return mean + us @ jnp.linalg.cholesky(cov).T, ws

    def least_squares(
        self, nz_params: Array, beta: Array, ref_params: Array
    ) -> tuple[Array, Array]:
        """Weighted least-squares conjugation parameters at the nodes of a reference, and the residual variance.

        The fit is solved in the standardized coordinates $u = L^{-1}(z - m)$ of the reference, where the
        design is well conditioned however narrow the reference is, and mapped back to $z$: the quadratic
        $-\\frac{1}{2} u \\cdot P_u \\cdot u + h_u \\cdot u$ is $-\\frac{1}{2} z \\cdot P_z \\cdot z + h_z \\cdot z$ up to a
        constant, with $P_z = L^{-\\top} P_u L^{-1}$ and $h_z = P_z m + L^{-\\top} h_u$.
        """
        lat = self.lat_man
        mean, cov = _mean_covariance(lat, ref_params)
        low = jnp.linalg.cholesky(cov)
        us, ws = _standard_nodes(self.lat_dim, self.n_nodes)
        zs = mean + us @ low.T
        psis = self.shifted_log_partitions(nz_params, beta, zs)
        sts = jax.vmap(lat.sufficient_statistic)(us)
        design = jnp.concatenate([sts, jnp.ones((us.shape[0], 1))], axis=1)
        sw = jnp.sqrt(ws)[:, None]
        coef, *_ = jnp.linalg.lstsq(sw * design, sw[:, 0] * psis)
        errors = psis - design @ coef
        h_u, prc_u = lat.split_location_precision(coef[:-1])
        low_inv = jnp.linalg.inv(low)
        prc_z = low_inv.T @ lat.cov_man.to_matrix(prc_u) @ low_inv
        h_z = prc_z @ mean + low_inv.T @ h_u
        rho = lat.join_location_precision(h_z, lat.cov_man.from_matrix(prc_z))
        return rho, jnp.sum(ws * errors**2)

    def conjugation_parameters(self, params: Array, beta: Array) -> tuple[Array, Array]:
        """Least-squares conjugation parameters $\\rho_Z(\\beta)$ at the reference, and the residual variance there.

        The reference is either the moment-matched exact distribution of $z$ at the biases $\\beta$ (the
        posterior at $\\hat\\theta_{N|X}(x)$, the marginal at $\\theta^*_N$), or the fixed point of
        $\\theta_Z + \\rho_Z(\\beta)$. It is held fixed when differentiating, which by the envelope theorem
        gives the gradient of the minimal residual variance. The returned $\\rho_Z$ has the precision of
        $\\theta_Z + \\rho_Z$ floored.
        """
        _, pch_params = self.split_params(params)
        _, nz_params, z_params = self.pch.split_coords(pch_params)
        if self.exact_reference:
            ref = self._floor(
                self.lat_man.to_natural(self.exact_latent_means(params, beta))
            )
        else:

            def step(ref: Array, _: None) -> tuple[Array, None]:
                rho, _ = self.least_squares(nz_params, beta, ref)
                return self._floor(z_params + rho), None

            ref, _ = jax.lax.scan(step, self._floor(z_params), None, self.n_fixed_point)
        ref = jax.lax.stop_gradient(ref)
        rho, var = self.least_squares(nz_params, beta, ref)
        return self._floor(z_params + rho) - z_params, var

    def residuals(self, nz_params: Array, beta: Array, rho: Array, zs: Array) -> Array:
        """The conjugation residual $r(z) = \\mathbf s_Z(z) \\cdot \\rho - \\psi_N(\\beta + \\Theta_{NZ} \\cdot \\mathbf s_Z(z)) + \\psi_N(\\beta)$ at nodes $z_k$."""
        sts = jax.vmap(self.lat_man.sufficient_statistic)(zs)
        return (
            sts @ rho
            - self.shifted_log_partitions(nz_params, beta, zs)
            + self.blz_man.log_partition_function(beta)
        )

    # Inference

    def recognition(self, params: Array, x: Array) -> tuple[Array, Array, Array]:
        """Recognition biases $\\hat\\theta_{N|X}(x)$, natural parameters $\\hat\\theta_{Z|X}(x)$, and residual variance."""
        lgm_params, pch_params = self.split_params(params)
        _, _, z_params = self.pch.split_coords(pch_params)
        beta = self.lgm.posterior_at(lgm_params, x)
        rho, var = self.conjugation_parameters(params, beta)
        return beta, z_params + rho, var

    def generative_prior(self, params: Array) -> tuple[Array, Array, Array]:
        """Prior biases $\\theta^*_N = \\theta_N + \\rho_N$, prior $\\theta^*_Z = \\theta_Z + \\rho_Z(\\theta^*_N)$, and residual variance."""
        lgm_params, pch_params = self.split_params(params)
        _, _, z_params = self.pch.split_coords(pch_params)
        beta = self.lgm.prior(lgm_params)
        rho, var = self.conjugation_parameters(params, beta)
        return beta, z_params + rho, var

    # Exact computations by enumeration

    def state_latents(self, params: Array) -> Array:
        """Natural parameters of $p(z \\mid n) = \\theta_Z + n \\cdot \\Theta_{NZ}$ for every state $n$."""
        _, pch_params = self.split_params(params)
        return jax.vmap(self.pch.posterior_at, in_axes=(None, 0))(
            pch_params, self.blz_man.states
        )

    def log_partition_function(self, params: Array) -> Array:
        """Exact $\\psi_{XNZ}$, by enumeration of the neurons and closed-form integrals over $x$ and $z$."""
        lgm_params, _ = self.split_params(params)
        obs_params, _, _ = self.lgm.split_coords(lgm_params)
        return self.obs_man.log_partition_function(obs_params) + self._log_sum_states(
            params, self.lgm.prior(lgm_params)
        )

    def log_observable_density(self, params: Array, x: Array) -> Array:
        """Exact $\\log p_X(x)$."""
        lgm_params, _ = self.split_params(params)
        obs_params, _, _ = self.lgm.split_coords(lgm_params)
        beta = self.lgm.posterior_at(lgm_params, x)
        return (
            self.obs_man.log_base_measure(x)
            + jnp.dot(self.obs_man.sufficient_statistic(x), obs_params)
            + self._log_sum_states(params, beta)
            - self.log_partition_function(params)
        )

    def exact_posterior_means(self, params: Array, x: Array) -> Array:
        """Mean parameters of the exact posterior $p(z \\mid x)$, a mixture over the states of $N$."""
        lgm_params, _ = self.split_params(params)
        return self.exact_latent_means(params, self.lgm.posterior_at(lgm_params, x))

    def exact_latent_means(self, params: Array, beta: Array) -> Array:
        """Mean parameters of the exact distribution of $z$ when the biases of $N$ are $\\beta$.

        At $\\beta = \\hat\\theta_{N|X}(x)$ this is the posterior $p(z \\mid x)$, and at $\\beta = \\theta^*_N$
        the marginal $p_Z$.
        """
        lat_params = self.state_latents(params)
        log_wts = self._state_log_weights(lat_params, beta)
        means = jax.vmap(self.lat_man.to_mean)(lat_params)
        return jax.nn.softmax(log_wts) @ means

    def sample(self, key: Array, params: Array, n: int) -> tuple[Array, Array, Array]:
        """Exact joint samples $(x, n, z)$."""
        lgm_params, _ = self.split_params(params)
        lat_params = self.state_latents(params)
        log_wts = self._state_log_weights(lat_params, self.lgm.prior(lgm_params))
        key_n, key_x, key_z = jax.random.split(key, 3)
        idx = jax.random.categorical(key_n, log_wts, shape=(n,))
        ns = self.blz_man.states[idx]
        x_params = jax.vmap(self.lgm.likelihood_at, in_axes=(None, 0))(lgm_params, ns)
        xs = jax.vmap(lambda k, p: self.obs_man.sample(k, p, 1)[0])(
            jax.random.split(key_x, n), x_params
        )
        zs = jax.vmap(lambda k, p: self.lat_man.sample(k, p, 1)[0])(
            jax.random.split(key_z, n), lat_params[idx]
        )
        return xs, ns, zs

    # The log-likelihood identity

    def log_likelihood_identity(
        self, params: Array, x: Array
    ) -> tuple[Array, Array, Array]:
        """$\\log p_X(x)$ by the identity, with Gauss-Hermite quadrature at the recognition and prior nodes.

        Returns the estimate and the effective sample sizes of the weights $w_k e^{-r_k}$ relative to
        those of $w_k$, at the recognition and prior nodes.
        """
        _, pch_params = self.split_params(params)
        _, nz_params, z_params = self.pch.split_coords(pch_params)
        beta_q, q_params, _ = self.recognition(params, x)
        beta_p, p_params, _ = self.generative_prior(params)
        c = self._offset(params, x, beta_q, q_params, beta_p, p_params)
        log_q, ess_q = self._log_mean_exp_neg_residual(
            nz_params, beta_q, q_params - z_params, q_params
        )
        log_p, ess_p = self._log_mean_exp_neg_residual(
            nz_params, beta_p, p_params - z_params, p_params
        )
        return c + log_q - log_p, ess_q, ess_p

    def variational_bounds(self, params: Array, x: Array) -> tuple[Array, Array]:
        """$\\log \\tilde p_X(x)$ of the variational model and its ELBO $\\mathcal L(x)$, by quadrature at the recognition nodes.

        The variational model is $\\tilde p_Z(z) \\tilde p(n \\mid z) p(x \\mid n)$, with the Gaussian prior
        $\\theta^*_Z$ and $\\tilde p(n \\mid z)$ the Boltzmann machine with biases $\\theta^*_N + \\Theta_{NZ}
        \\cdot \\mathbf s_Z(z)$. With $f = c + r_Z - r^X_Z$, where $r_Z$ is the residual at the prior biases
        and $r^X_Z$ at the recognition biases, $\\log \\tilde p_X = c + \\log \\mathbb E_q[e^{r_Z - r^X_Z}]$
        and $\\mathcal L = c + \\mathbb E_q[r_Z - r^X_Z]$.
        """
        _, pch_params = self.split_params(params)
        _, nz_params, z_params = self.pch.split_coords(pch_params)
        beta_q, q_params, _ = self.recognition(params, x)
        beta_p, p_params, _ = self.generative_prior(params)
        c = self._offset(params, x, beta_q, q_params, beta_p, p_params)
        zs, ws = self.quadrature(q_params)
        r_x = self.residuals(nz_params, beta_q, q_params - z_params, zs)
        r_p = self.residuals(nz_params, beta_p, p_params - z_params, zs)
        log_tilde = c + logsumexp(jnp.log(ws) + r_p - r_x)
        return log_tilde, c + jnp.sum(ws * (r_p - r_x))

    # Internal

    @property
    def _coupling_indices(self) -> np.ndarray:
        """Positions of the stored couplings among the off-diagonal coordinates of the Boltzmann machine."""
        rows, cols = np.triu_indices(self.n_neurons, 1)
        match self.noise_correlations:
            case "none":
                return np.zeros(0, dtype=int)
            case "chain":
                return np.flatnonzero(cols == rows + 1)
            case "harmonium" | "full":
                return np.arange(rows.size)

    def _offset(
        self,
        params: Array,
        x: Array,
        beta_q: Array,
        q_params: Array,
        beta_p: Array,
        p_params: Array,
    ) -> Array:
        """$c(x) = \\log p_X$ under exact conjugation, given the recognition and prior biases and Gaussians."""
        lgm_params, _ = self.split_params(params)
        obs_params, _, _ = self.lgm.split_coords(lgm_params)
        blz, lat, obs = self.blz_man, self.lat_man, self.obs_man
        return (
            obs.log_base_measure(x)
            + jnp.dot(obs.sufficient_statistic(x), obs_params)
            - obs.log_partition_function(obs_params)
            + blz.log_partition_function(beta_q)
            - blz.log_partition_function(beta_p)
            + lat.log_partition_function(q_params)
            - lat.log_partition_function(p_params)
        )

    def _floor(self, nrm_params: Array) -> Array:
        loc, prc = self.lat_man.split_location_precision(nrm_params)
        mat = self.lat_man.cov_man.to_matrix(prc)
        vals, vecs = jnp.linalg.eigh(mat)
        floored = (vecs * jnp.maximum(vals, self.min_precision)) @ vecs.T
        return self.lat_man.join_location_precision(
            loc, self.lat_man.cov_man.from_matrix(floored)
        )

    def _state_log_weights(self, lat_params: Array, beta: Array) -> Array:
        stats = jax.vmap(self.blz_man.sufficient_statistic)(self.blz_man.states)
        psi_z = jax.vmap(self.lat_man.log_partition_function)(lat_params)
        return stats @ beta + psi_z

    def _log_sum_states(self, params: Array, beta: Array) -> Array:
        return logsumexp(self._state_log_weights(self.state_latents(params), beta))

    def _log_mean_exp_neg_residual(
        self, nz_params: Array, beta: Array, rho: Array, nrm_params: Array
    ) -> tuple[Array, Array]:
        zs, ws = self.quadrature(nrm_params)
        rs = self.residuals(nz_params, beta, rho, zs)
        log_wts = jnp.log(ws) - rs
        log_mean = logsumexp(log_wts)
        nrm_wts = jnp.exp(log_wts - log_mean)
        ess = 1.0 / jnp.sum(nrm_wts**2)
        return log_mean, ess * jnp.sum(ws**2)


def _chol_matrix(coords: Array) -> Array:
    """Positive definite matrix $L L^\\top$ from a lower-triangular matrix with a log diagonal."""
    low = jnp.tril(coords, -1) + jnp.diag(jnp.exp(jnp.diag(coords)))
    return low @ low.T


def _chol_coords(matrix: Array) -> Array:
    """Inverse of :func:`_chol_matrix`."""
    low = jnp.linalg.cholesky(matrix)
    return jnp.tril(low, -1) + jnp.diag(jnp.log(jnp.diag(low)))


def _mean_covariance(lat_man: FullNormal, params: Array) -> tuple[Array, Array]:
    loc, prc = lat_man.split_location_precision(params)
    cov = jnp.linalg.inv(lat_man.cov_man.to_matrix(prc))
    return cov @ loc, cov


def _standard_nodes(dim: int, n_nodes: int) -> tuple[Array, Array]:
    """Tensor-product Gauss-Hermite nodes and weights for the standard normal in $\\mathbb R^d$."""
    us, ws = np.polynomial.hermite_e.hermegauss(n_nodes)
    ws = ws / ws.sum()
    grids = np.meshgrid(*([us] * dim), indexing="ij")
    wgrids = np.meshgrid(*([ws] * dim), indexing="ij")
    nodes = np.stack([g.ravel() for g in grids], axis=1)
    weights = np.prod(np.stack([g.ravel() for g in wgrids], axis=1), axis=1)
    return jnp.asarray(nodes), jnp.asarray(weights)
