"""The three-layer canonical circuit $Z \\to N \\to X$ as a two-level variational model.

The circuit is a graphical harmonium over an observable $x \\in \\mathbb R^{d_X}$, neurons $n \\in \\{0,
1\\}^{d_N}$ and a latent $z \\in \\mathbb R$, with natural parameters

- $\\theta_X$: the location and precision of $x$;
- $\\theta_N$: the biases and pairwise couplings of the neurons;
- $\\theta_Z$: the location and precision of $z$;
- $\\Theta_{XN}$: the location of $x$ times the activities of $n$;
- $\\Theta_{NZ}$: the activities of $n$ times $(z, z^2)$, with a shared coefficient of $z^2$, so that the
  tuning curves are bell-shaped with a shared width.

It is fit as a :class:`~goal.geometry.VariationalConjugated` with two levels:

- :class:`CanonicalCircuit` is the $X - N$ level. Its conjugation parameters $\\rho_N$ have no
  parameters of their own: they are the exact closed form of :class:`~goal.models.BoltzmannLGM`, which
  is quadratic in $n$ with couplings $G = \\Theta_{XN}^\\top \\Sigma_X \\Theta_{XN}$. The level is
  therefore exactly conjugate.
- :class:`PopulationCodeLevel` is the $N - Z$ level, over the Gaussian prior family. Its conjugation
  parameters $\\rho_Z$ are a multilayer perceptron of the likelihood $(\\theta_N, \\Theta_{NZ})$.

The free parameters of the neurons are their generative biases and couplings $\\theta^*_N = \\theta_N +
\\rho_N$, with the couplings restricted to a graph $E$ (none, a chain, or all pairs); the harmonium's
$\\theta_N = \\theta^*_N - \\rho_N$ is dense. Given $z$, the generative model over the neurons then has
couplings on $E$ only, and without $z$ the marginal of the harmonium over the neurons is the Boltzmann
machine $\\theta^*_N$ on $E$. The recognition model over the neurons has the dense couplings of
$\\theta_N$, since $x$ shifts only the biases.

The diagnostics enumerate the neurons and integrate $z$ by Gauss-Hermite quadrature: the exact $\\log
p(x)$ of the graphical harmonium, $\\log \\tilde p(x)$ and the ELBO of the variational model, and the
divergence of the recognition model from the exact posterior. Experimental code: see
``scratch/variational-conjugation/canonical-circuit.tex``.
"""

import itertools
from dataclasses import dataclass
from typing import Any, Literal, override

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.scipy.special import logsumexp

from goal.geometry import (
    AttachedHarmonium,
    CliqueMap,
    CrossTerm,
    DifferentiableVariationalConjugated,
    GraphicalHarmonium,
    Harmonium,
    IdentityEmbedding,
    MultilayerPerceptron,
    PositiveDefinite,
    Rectangular,
    VariationalConjugated,
)
from goal.models import (
    BoltzmannLGM,
    Euclidean,
    FullBoltzmann,
    FullNormal,
    NormalBoltzmannHarmonium,
    full_normal,
)
from goal.models.harmonium.lgm import GeneralizedGaussianLocationEmbedding

type Couplings = Literal["independent", "chain", "full"]
type Tied = dict[str, Array]


@dataclass(frozen=True)
class EnumeratedBoltzmann(FullBoltzmann):
    """A dense Boltzmann machine sampled exactly, by enumerating its states."""

    @override
    def sample(
        self,
        key: Array,
        params: Array,
        n: int = 1,
        n_burnin: int = 1000,
        n_thin: int = 10,
    ) -> Array:
        """Exact samples; ``n_burnin`` and ``n_thin`` of the Gibbs sampler are ignored."""
        states = self.states
        logits = jax.vmap(self.sufficient_statistic)(states) @ params
        return states[jax.random.categorical(key, logits, shape=(n,))]


@dataclass(frozen=True)
class PopulationCodeHarmonium(Harmonium[EnumeratedBoltzmann, FullNormal]):
    """Harmonium with a Boltzmann observable and a Normal latent, coupled through the node activities only.

    Neuron $i$ has the logit $\\theta_{N,i} + \\Theta_{NZ,i} \\cdot \\mathbf s_Z(z)$, so the couplings of
    the neurons do not depend on $z$, and $p(z \\mid n)$ is Gaussian with natural parameters affine in
    $n$. Unlike :class:`~goal.models.BoltzmannNormalHarmonium`, the pairwise statistics of the neurons
    are not coupled to $z$.
    """

    # Fields

    neurons: EnumeratedBoltzmann
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
    def obs_man(self) -> EnumeratedBoltzmann:
        return self.neurons

    @property
    @override
    def pst_man(self) -> FullNormal:
        return full_normal(self.lat_dim)


@dataclass(frozen=True)
class PopulationCodeLevel(
    DifferentiableVariationalConjugated[
        AttachedHarmonium[FullNormal], FullNormal, MultilayerPerceptron[FullNormal, Any]
    ]
):
    """The $N - Z$ level, with conjugation parameters $\\rho_Z$ a map of the likelihood $(\\theta_N, \\Theta_{NZ})$.

    The map's output is read as a location and a raw precision. The precision of $\\rho_Z$ is
    $\\mathrm{softplus}(a + c) - 1$, with $c$ such that it vanishes at $a = 0$, so that with $\\theta_Z$
    the standard normal (as :meth:`CanonicalCircuit.tie` holds it), the prior and the recognition model
    over $z$ always have positive precision. This restricts the family of conjugation functions to those
    that give valid distributions, which every exact conjugation function of a normalizable harmonium
    does; it does not clamp anything.
    """

    # Fields

    pop_hrm: PopulationCodeHarmonium
    mlp: MultilayerPerceptron[FullNormal, Any]

    # Overrides

    @property
    @override
    def gen_hrm(self) -> AttachedHarmonium[FullNormal]:
        return AttachedHarmonium(self.pop_hrm, self.pop_hrm.pst_man)

    @property
    @override
    def pst_prr_emb(self) -> IdentityEmbedding[FullNormal]:
        return IdentityEmbedding(self.pop_hrm.pst_man)

    @property
    @override
    def cnj_fun_man(self) -> MultilayerPerceptron[FullNormal, Any]:
        return self.mlp

    @override
    def conjugation_parameters(self, lkl_params: Array, cnj_fun_params: Array) -> Array:
        lat = self.pop_hrm.pst_man
        loc, raw = lat.split_location_precision(self.mlp(cnj_fun_params, lkl_params))
        prc = jax.nn.softplus(raw + np.log(np.e - 1.0)) - 1.0
        return lat.join_location_precision(loc, prc)


@dataclass(frozen=True)
class ReadoutHarmonium(GraphicalHarmonium[Any]):
    """The $X - N$ harmonium attached to the neurons of the population code harmonium, its latent model."""

    # Fields

    rdt_hrm: NormalBoltzmannHarmonium[PositiveDefinite, Any]
    pop_hrm: AttachedHarmonium[FullNormal]

    # Overrides

    @property
    @override
    def pst_man(self) -> AttachedHarmonium[FullNormal]:
        return self.pop_hrm

    @property
    @override
    def obs_hrms_att_clqs(
        self,
    ) -> tuple[tuple[Harmonium[Any, Any], tuple[int, ...]], ...]:
        return ((self.rdt_hrm, tuple(range(self.rdt_hrm.pst_man.n_nodes))),)


@dataclass(frozen=True)
class CanonicalCircuit(VariationalConjugated[ReadoutHarmonium, Euclidean]):
    """The circuit: the $X - N$ level over the population code level.

    Its parameters are $[\\theta_X, \\Theta_{XN}, (\\theta_N, \\Theta_{NZ}, \\theta_Z) | \\phi_Z]$, with
    $\\phi_Z$ the parameters of $\\rho_Z$; $\\rho_N$ has none. ``edges`` is the graph $E$ of the
    generative couplings $\\theta^*_N$, as pairs $i < j$.
    """

    # Fields

    obs_dim: int
    edges: tuple[tuple[int, int], ...]
    dep: PopulationCodeLevel

    # Overrides

    @property
    @override
    def gen_hrm(self) -> ReadoutHarmonium:
        rdt_hrm = NormalBoltzmannHarmonium(
            self.obs_dim, PositiveDefinite(), self.neurons
        )
        return ReadoutHarmonium(rdt_hrm, self.dep.gen_hrm)

    @property
    @override
    def cnj_fun_man(self) -> Euclidean:
        return Euclidean(0)

    @property
    @override
    def pst_prr_emb(self) -> IdentityEmbedding[Any]:
        return IdentityEmbedding(self.dep.gen_hrm)

    @property
    @override
    def dep_man(self) -> PopulationCodeLevel:
        return self.dep

    @override
    def conjugation_parameters(self, lkl_params: Array, cnj_fun_params: Array) -> Array:
        """$\\rho_N$: the exact closed form of :class:`~goal.models.BoltzmannLGM`, on the biases and couplings of the neurons."""
        rho = BoltzmannLGM(
            self.obs_dim, PositiveDefinite(), self.neurons.n_neurons
        ).conjugation_parameters(lkl_params)
        pop_hrm = self.dep.gen_hrm
        return pop_hrm.join_coords(
            rho, pop_hrm.int_man.zeros(), pop_hrm.pst_man.zeros()
        )

    # Manifolds

    @property
    def neurons(self) -> EnumeratedBoltzmann:
        return self.dep.pop_hrm.neurons

    @property
    def lat_man(self) -> FullNormal:
        return self.dep.pop_hrm.pst_man

    @property
    def states(self) -> Array:
        """Every state of the neurons, as rows."""
        return jnp.array(
            list(itertools.product([0.0, 1.0], repeat=self.neurons.n_neurons))
        )

    # Tied parameters

    def generative_couplings(self, stored: Array) -> Array:
        """$\\theta^*_N$ in the dense layout, from the biases followed by the couplings on ``edges``."""
        n_neurons = self.neurons.n_neurons
        rows, cols = np.triu_indices(n_neurons, 1)
        index = {(int(i), int(j)): k for k, (i, j) in enumerate(zip(rows, cols))}
        positions = np.array([index[e] for e in self.edges], dtype=int)
        off_diag = jnp.zeros(rows.size).at[positions].set(stored[n_neurons:])
        return self.neurons.join_couplings(stored[:n_neurons], off_diag)

    def readout_conjugation(self, obs_params: Array, xn_params: Array) -> Array:
        """$\\rho_N$ on the neurons at the readout $(\\theta_X, \\Theta_{XN})$."""
        lkl_params = self.gen_hrm.lkl_fun_man.join_coords(obs_params, xn_params)
        rho, _, _ = self.dep.gen_hrm.split_coords(
            self.conjugation_parameters(lkl_params, self.cnj_fun_man.zeros())
        )
        return rho

    def tie(self, u: Tied) -> Array:
        """Circuit parameters from the stored blocks, with the shared tuning width and $\\theta_Z$ standard normal.

        ``neurons`` holds the generative biases and the couplings on ``edges``, and $\\theta_N =
        \\theta^*_N - \\rho_N$. ``nz_loc`` holds the coefficient of $z$ for each neuron and ``nz_prc`` the
        shared precision coordinates of the coefficient of $z^2$. $\\theta_Z$ is held at the standard
        normal: for a one-dimensional $z$ its location and precision are an affine change of $z$, which
        $\\Theta_{NZ}$ absorbs. Every block other than ``neurons`` enters linearly.
        """
        lat = self.lat_man
        theta_n = self.generative_couplings(u["neurons"]) - self.readout_conjugation(
            u["obs"], u["xn"]
        )
        rows = jax.vmap(lat.join_location_precision, in_axes=(0, None))(
            u["nz_loc"][:, None], u["nz_prc"]
        )
        (nz_map,) = self.dep.pop_hrm.crs_maps
        z_params = lat.to_natural(lat.standard_normal())
        pop_params = self.dep.gen_hrm.join_coords(
            theta_n, nz_map.from_matrix(rows), z_params
        )
        hrm_params = self.gen_hrm.join_coords(u["obs"], u["xn"], pop_params)
        return self.join_coords(hrm_params, u["cnj"])

    def initialize_tied(
        self, key: Array, xs: Array, z_range: float, obs_sd: float
    ) -> Tied:
        """Tuning curves tiling $[-r, r]$ at low prior firing, an isotropic readout, and $\\rho_Z = 0$.

        Neighbouring tuning curves overlap by one spacing $s$ (precision $1 / s^2$), and the generative
        logit $\\theta^*_N + \\Theta_{NZ} \\cdot \\mathbf s_Z(z)$ peaks at $-1$, with no couplings. The
        readout has isotropic noise of s.d. ``obs_sd`` around the data mean and standard normal
        interactions with the neurons. A noise covariance fitted to the data would explain the
        correlations of the data as noise and leave nothing to the neurons. $\\rho_Z$ starts at zero:
        the last layer of the map is zero.
        """
        k_xn, k_mlp = jax.random.split(key)
        n_neurons = self.neurons.n_neurons
        (obs_nrm,) = self.gen_hrm.obs_man.elm_mans
        prc = jnp.eye(self.obs_dim) / obs_sd**2
        obs_params = obs_nrm.join_location_precision(
            prc @ jnp.mean(xs, axis=0), obs_nrm.cov_man.from_matrix(prc)
        )
        (xn_map,) = self.gen_hrm.crs_maps
        xn_params = xn_map.from_matrix(
            jax.random.normal(k_xn, (self.obs_dim, n_neurons))
        )
        preferred = jnp.linspace(-z_range, z_range, n_neurons)
        width_prc = (n_neurons - 1) ** 2 / (2 * z_range) ** 2
        mlp = self.dep.mlp
        cnj = mlp.glorot_initialize(k_mlp)
        last = mlp.layer_dims[-2] * mlp.layer_dims[-1] + mlp.layer_dims[-1]
        return {
            "obs": obs_params,
            "xn": xn_params,
            "neurons": jnp.concatenate(
                [-1.0 - 0.5 * width_prc * preferred**2, jnp.zeros(len(self.edges))]
            ),
            "nz_loc": width_prc * preferred,
            "nz_prc": self.lat_man.cov_man.from_matrix(jnp.full((1, 1), width_prc)),
            "cnj": cnj.at[-last:].set(0.0),
        }

    # Exact computations by enumeration and quadrature

    def state_log_weights(self, pop_params: Array) -> Array:
        """$\\beta \\cdot \\mathbf s_N(n) + \\psi_Z(\\theta_Z + \\Theta_{NZ}^\\top n)$ for every state, at parameters of the population code harmonium with biases $\\beta$: the log-weights of $n$ with $z$ integrated out."""
        pop_hrm = self.dep.gen_hrm
        bias, _, _ = pop_hrm.split_coords(pop_params)
        stats = jax.vmap(pop_hrm.obs_man.sufficient_statistic)(self.states)
        z_params = jax.vmap(pop_hrm.posterior_at, in_axes=(None, 0))(
            pop_params, self.states
        )
        return stats @ bias + jax.vmap(self.lat_man.log_partition_function)(z_params)

    def exact_state_log_weights(self, params: Array) -> Array:
        """Unnormalized $\\log p(n)$ of the graphical harmonium for every state: $x$ and $z$ integrated in closed form."""
        hrm_params, _ = self.split_coords(params)
        _, _, pop_params = self.gen_hrm.split_coords(hrm_params)
        psi_x = jax.vmap(
            lambda n: self.obs_man.log_partition_function(
                self.likelihood_at(params, jnp.append(n, 0.0))
            )
        )(self.states)
        return self.state_log_weights(pop_params) + psi_x

    def exact_log_partition_function(self, params: Array) -> Array:
        """$\\psi$ of the graphical harmonium."""
        return logsumexp(self.exact_state_log_weights(params))

    def exact_log_observable_density(self, params: Array, x: Array) -> Array:
        """$\\log p(x)$ of the graphical harmonium."""
        obs_params, _ = self.gen_hrm.lkl_fun_man.split_coords(
            self.likelihood_function(params)
        )
        return (
            self.obs_man.log_base_measure(x)
            + jnp.dot(self.obs_man.sufficient_statistic(x), obs_params)
            + logsumexp(self.state_log_weights(self.posterior_at(params, x)))
            - self.exact_log_partition_function(params)
        )

    def recognition_quadrature(
        self, params: Array, x: Array, n_nodes: int
    ) -> tuple[Array, Array]:
        """Gauss-Hermite nodes and weights (summing to one) of the recognition model's marginal over $z$, the Gaussian $\\theta_Z + \\rho_Z$ at the recognition biases."""
        q_params, _ = self.dep.dep_man.split_coords(
            self.dep.conjugated_prior_params(self.recognition_at(params, x))
        )
        loc, prc = self.lat_man.split_location_precision(q_params)
        var = 1.0 / self.lat_man.cov_man.to_matrix(prc)[0, 0]
        us, ws = np.polynomial.hermite_e.hermegauss(n_nodes)
        zs = var * loc[0] + jnp.sqrt(var) * jnp.asarray(us)
        return zs[:, None], jnp.asarray(ws / ws.sum())

    def variational_bounds(
        self, params: Array, x: Array, n_nodes: int
    ) -> tuple[Array, Array]:
        """$\\log \\tilde p(x)$ and the ELBO of the variational model, with $z$ by quadrature and $n$ by enumeration.

        With $f$ the summed residual (:meth:`elbo_residual`), $\\log \\tilde p(x) = c(x) + \\log \\mathbb
        E_q[e^f]$ and $\\mathcal L(x) = c(x) + \\mathbb E_q[f]$. The residual of this level depends on $n$
        and those of the population code level on $z$, and $q(n, z \\mid x) = q(z \\mid x) p(n \\mid z,
        \\beta)$ with $\\beta$ the recognition biases.
        """
        dep = self.dep
        dep_prior = self.conjugated_prior_params(params)
        dep_post = self.recognition_at(params, x)
        zs, ws = self.recognition_quadrature(params, x, n_nodes)
        r_n = jax.vmap(lambda n: self.conjugation_residual(params, jnp.append(n, 0.0)))(
            self.states
        )

        def at_node(z: Array) -> tuple[Array, Array]:
            lkl = dep.likelihood_at(dep_post, z)
            log_pn = jax.vmap(dep.obs_man.log_density, in_axes=(None, 0))(
                lkl, self.states
            )
            r_z = dep.conjugation_residual(dep_prior, z) - dep.conjugation_residual(
                dep_post, z
            )
            return (
                r_z + jnp.exp(log_pn) @ r_n,
                r_z + logsumexp(log_pn + r_n),
            )

        means, log_means = jax.vmap(at_node)(zs)
        c = self.conjugation_baseline(params, x)
        return c + logsumexp(jnp.log(ws) + log_means), c + ws @ means

    def recognition_divergence(self, params: Array, x: Array, n_nodes: int) -> Array:
        """$\\mathrm{KL}(q(n, z \\mid x) \\Vert p(n, z \\mid x))$ from the recognition model to the exact posterior of the graphical harmonium.

        Both have $p(n \\mid z, \\beta)$ as the conditional of $n$, so it is the divergence of the
        marginals over $z$: the Gaussian $q(z \\mid x)$ and the mixture $p(z \\mid x)$.
        """
        pop_hrm = self.dep.gen_hrm
        post = self.posterior_at(params, x)
        bias, _, _ = pop_hrm.split_coords(post)
        stats = jax.vmap(pop_hrm.obs_man.sufficient_statistic)(self.states)
        z_params = jax.vmap(pop_hrm.posterior_at, in_axes=(None, 0))(post, self.states)
        log_norm = logsumexp(self.state_log_weights(post))
        q_params, _ = self.dep.dep_man.split_coords(
            self.dep.conjugated_prior_params(self.recognition_at(params, x))
        )
        zs, ws = self.recognition_quadrature(params, x, n_nodes)

        def log_ratio(z: Array) -> Array:
            s_z = self.lat_man.sufficient_statistic(z)
            log_p = (
                logsumexp(stats @ bias + z_params @ s_z)
                + self.lat_man.log_base_measure(z)
                - log_norm
            )
            return self.lat_man.log_density(q_params, z) - log_p

        return ws @ jax.vmap(log_ratio)(zs)



def canonical_circuit(
    obs_dim: int, n_neurons: int, couplings: Couplings, hidden_dims: tuple[int, ...]
) -> CanonicalCircuit:
    """The circuit with generative couplings on no pairs, a chain, or all pairs, and a map with ``hidden_dims`` for $\\rho_Z$."""
    pairs = itertools.combinations(range(n_neurons), 2)
    edges = {
        "independent": (),
        "chain": tuple((i, i + 1) for i in range(n_neurons - 1)),
        "full": tuple(pairs),
    }[couplings]
    pop_hrm = PopulationCodeHarmonium(EnumeratedBoltzmann(n_neurons), 1)
    mlp = MultilayerPerceptron(
        pop_hrm.pst_man,
        AttachedHarmonium(pop_hrm, pop_hrm.pst_man).lkl_fun_man,
        hidden_dims,
        jax.nn.tanh,
    )
    return CanonicalCircuit(obs_dim, edges, PopulationCodeLevel(pop_hrm, mlp))
