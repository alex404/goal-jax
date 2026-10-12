"""The canonical circuit: a deep model of alternating Gaussian and binary populations, fit variationally.

A circuit with $L$ layers is the chain $X \\leftarrow N_1 \\leftarrow Z_1 \\leftarrow N_2 \\leftarrow
\\cdots \\leftarrow N_L \\leftarrow Z_L$, with $x \\in \\mathbb R^{d_X}$ observed, binary neurons $n_k \\in
\\{0, 1\\}^{n_k}$, and Gaussian latents $z_k \\in \\mathbb R^{d_k}$. Each layer is a Gaussian-Boltzmann edge, which couples
the location of the Gaussian below to the activities of the neurons, and a population code, which
couples the activities of the neurons to the statistics of the Gaussian above. Given $z_k$ the
neurons $n_k$ have the couplings of their own family, and given the neurons every $z_k$ is Gaussian.

The circuit is a chain of $2L$ levels
(:class:`~goal.geometry.exponential_family.variational.VariationalLevel`), bottom first, the bottom level being the :class:`CanonicalCircuit` itself. Each layer has a Gaussian-Boltzmann
level, a :class:`~goal.models.NormalBoltzmannMachine` whose conjugation function gives $\\rho_N$ on the neurons, and
a population code level, a :class:`~goal.models.BoltzmannPopulationCode` whose conjugation function
(:class:`PopulationConjugationFunction`) gives $\\rho_{Z_k}$ on its latent. It comes in two variants:

- **Partially exact**: the neurons are dense (:class:`~goal.models.FullBoltzmann`, sampled exactly by enumeration), and the
  conjugation function of each Gaussian-Boltzmann level is the exact closed form of
  :class:`~goal.models.NormalBoltzmannMachine` (:class:`~goal.geometry.exponential_family.variational.ExactConjugationFunction`), quadratic in $n$ with dense couplings. Only the population
  codes are approximated. Exact computations enumerate the neurons, so the populations must be small.
- **Approximate**: the neurons are a :class:`~goal.models.ChordalBoltzmann` on the chordal completion of a graph $E$, in the
  harmonium, the prior and the recognition model, and the conjugation parameters of each Gaussian-Boltzmann
  level are learned in that family. The neurons are sampled and normalized by junction tree.

In both, every conjugation function that is learned is a multilayer perceptron of the likelihood
parameters of its level. The circuit is trained in its generative coordinates (:meth:`CanonicalCircuit.tie`):
the generative biases and couplings $\\theta^*_N$ on the chordal completion of $E$ and the tuning curves of the neurons, with the
harmonium's $\\theta_N = \\theta^*_N - \\rho_N$ and $\\theta_{Z_k} = \\theta^*_{Z_k} - \\rho_{Z_k}$, where every
$\\theta^*_{Z_k}$ is the standard normal. The directed model $\\tilde p$ then does not depend on the conjugation
functions.
"""

import itertools
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal, cast, override

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.scipy.special import logsumexp

from ...geometry import (
    AttachedHarmonium,
    Diagonal,
    MultilayerPerceptron,
    PositiveDefinite,
)
from ...geometry.exponential_family.variational import (
    DeepModel,
    DifferentiableDeepModel,
    ExactConjugationFunction,
    VariationalLevel,
)
from ...geometry.manifold.base import Manifold
from ...geometry.manifold.combinators import Pair, Tuple
from ...geometry.manifold.map import Map
from ...geometry.manifold.util import split_by_dims
from ..base.gaussian.boltzmann import (
    Boltzmann,
    ChordalBoltzmann,
    FullBoltzmann,
)
from ..base.gaussian.generalized import Euclidean
from ..base.gaussian.normal import DiagonalNormal, diagonal_normal
from ..harmonium.lgm import BoltzmannCouplingEmbedding, NormalBoltzmannMachine
from ..harmonium.population_codes import BoltzmannPopulationCode

type Variant = Literal["partially_exact", "approximate"]


### Levels ###

type GaussianBoltzmannLevel = VariationalLevel[
    NormalBoltzmannMachine[Any, Any], Map[Any, Any]
]
type PopulationCodeLevel = VariationalLevel[
    BoltzmannPopulationCode[Any, DiagonalNormal], PopulationConjugationFunction
]


@dataclass(frozen=True)
class PopulationConjugationFunction(
    Pair[MultilayerPerceptron[DiagonalNormal, Any], Euclidean],
    Map[DiagonalNormal, Any],
):
    """The conjugation function of a population code level: a multilayer perceptron of the likelihood parameters, and a log-scale $\\beta$ per dimension of the latent.

    The perceptron reads the likelihood parameters $\\ell = (\\theta_N, \\Theta_{NZ})$ and outputs a location
    $l(\\ell)$ and a raw precision $a(\\ell)$. Mathematically, $\\rho_Z(\\ell)$ has location $l(\\ell)$ and
    precision $e^{\\beta} (e^{a(\\ell)} - 1)$, so a latent with precision $e^{\\beta}$ has precision
    $e^{\\beta + a(\\ell)} > 0$ after $\\rho_Z$ is added, at every likelihood. :meth:`CanonicalCircuit.tie`
    sets $e^{\\beta}$ to the precision of $\\theta_Z$, so that the prior and the recognition model over $z$
    have positive precision at every likelihood. This restricts the family of conjugation functions to
    those that give valid distributions; it does not clamp anything.
    """

    # Fields

    mlp: MultilayerPerceptron[DiagonalNormal, Any]

    # Overrides

    @property
    @override
    def fst_man(self) -> MultilayerPerceptron[DiagonalNormal, Any]:
        return self.mlp

    @property
    @override
    def snd_man(self) -> Euclidean:
        return Euclidean(self.mlp.cod_man.data_dim)

    @property
    @override
    def dom_man(self) -> Any:
        return self.mlp.dom_man

    @property
    @override
    def cod_man(self) -> DiagonalNormal:
        return self.mlp.cod_man

    @override
    def __call__(self, f_coords: Array, v_coords: Array) -> Array:
        """$\\rho_Z$ in natural coordinates of the latent."""
        mlp_params, beta = self.split_coords(f_coords)
        lat = self.mlp.cod_man
        loc, raw = lat.split_location_precision(self.mlp(mlp_params, v_coords))
        return lat.join_location_precision(loc, jnp.exp(beta) * jnp.expm1(raw))

    # Methods

    def raw_precision(self, mlp_params: Array, lkl_params: Array) -> Array:
        """$a(\\ell)$, from the parameters of the perceptron."""
        _, raw = self.mlp.cod_man.split_location_precision(
            self.mlp(mlp_params, lkl_params)
        )
        return raw


### Circuit ###


@dataclass(frozen=True)
class GenerativeCoordinates(Tuple):
    """The generative coordinates of a canonical circuit: a plain product of its free blocks, as listed by :attr:`CanonicalCircuit.generative_man`."""

    # Fields

    elm_mans: tuple[Manifold, ...]

    # Overrides

    @property
    @override
    def dim(self) -> int:
        return sum(elm.dim for elm in self.elm_mans)

    @override
    def split_coords(self, coords: Array) -> tuple[Array, ...]:
        return split_by_dims(coords, tuple(elm.dim for elm in self.elm_mans))

    @override
    def join_coords(self, *components: Array) -> Array:
        return jnp.concatenate(components)


@dataclass(frozen=True)
class CanonicalCircuit(
    VariationalLevel[NormalBoltzmannMachine[Any, Any], Map[Any, Any]]
):
    """The canonical circuit: the Gaussian-Boltzmann level of the first layer, over every level above it.

    ``edges`` is the graph $E_k$ of each layer, as pairs $i < j$; the generative couplings live on its chordal completion.
    """

    # Fields

    edges: tuple[tuple[tuple[int, int], ...], ...]

    # Methods

    @property
    def layers(self) -> tuple[tuple[GaussianBoltzmannLevel, PopulationCodeLevel], ...]:
        """The Gaussian-Boltzmann and population code level of each layer, bottom first."""
        layers: list[tuple[GaussianBoltzmannLevel, PopulationCodeLevel]] = []
        gb: GaussianBoltzmannLevel = self
        while True:
            pop = cast(PopulationCodeLevel, gb.dep_man)
            layers.append((gb, pop))
            if not isinstance(pop.dep_man, VariationalLevel):
                return tuple(layers)
            gb = cast(GaussianBoltzmannLevel, pop.dep_man)

    @property
    def generative_neuron_mans(self) -> tuple[ChordalBoltzmann, ...]:
        """The family of the generative biases and couplings of each layer: a Boltzmann machine on the chordal completion of $E_k$."""
        return tuple(
            ChordalBoltzmann.from_edges(gb.att_hrm.pst_man.data_dim, edges)
            for (gb, _), edges in zip(self.layers, self.edges)
        )

    @property
    def generative_man(self) -> GenerativeCoordinates:
        """The free blocks of the circuit, bottom first.

        Per layer: the observable of the Gaussian-Boltzmann harmonium (first layer only; above it, the
        latent below is the standard normal), its interaction, the generative biases and couplings
        $\\theta^*_N$ on the chordal completion of $E_k$, and the coefficients of the location and of the precision of $z_k$ for
        each neuron and dimension, so that every neuron has its own tuning width. Last, the parameters of
        the perceptron of every learned conjugation function, level by level (:meth:`tie` sets the
        $\\beta$ of the population codes).
        """
        mans: list[Manifold] = []
        for k, ((gb, pop), gen_nrn) in enumerate(
            zip(self.layers, self.generative_neuron_mans)
        ):
            if k == 0:
                mans.append(gb.att_hrm.obs_man)
            mans.append(gb.att_hrm.int_man)
            mans.append(gen_nrn)
            lat_dim = pop.att_hrm.pst_man.data_dim
            mans.append(Euclidean(gen_nrn.data_dim * lat_dim))
            mans.append(Euclidean(gen_nrn.data_dim * lat_dim))
        mans.append(GenerativeCoordinates(tuple(self._perceptron_mans)))
        return GenerativeCoordinates(tuple(mans))

    def tie(self, gen_params: Array) -> Array:
        """Circuit parameters from generative coordinates (:attr:`generative_man`).

        Every natural parameter of the harmonium is its generative value minus its conjugation
        parameters, each evaluated where $\\tilde p$ evaluates it: $\\theta_{N_k} = \\theta^*_{N_k} -
        \\rho_{N_k}$, with $\\rho_{N_k}$ at the likelihood of the Gaussian-Boltzmann level given its
        Gaussian at $\\theta^*$ (the observable, or the standard normal), and $\\theta_{Z_k} =
        \\theta^*_{Z_k} - \\rho_{Z_k}$, with $\\rho_{Z_k}$ at the generative neurons $\\theta^*_{N_k}$ and
        $\\beta$ such that the precision of $\\theta_{Z_k}$ is $e^{\\beta}$. So $\\tilde p(z_L)$ and every
        $\\tilde p(z_{k-1} \\mid n_k)$ have base $\\theta^*_{Z} = \\mathcal N(0, I)$, and $\\tilde p$ does not
        depend on the conjugation functions. A free $\\theta^*_Z$ would add nothing: an affine change of
        each coordinate of $z_k$ is absorbed by the interactions and tuning curves it enters.
        """
        blocks = list(self.generative_man.split_coords(gen_params))
        perceptron_params = list(
            GenerativeCoordinates(tuple(self._perceptron_mans)).split_coords(
                blocks.pop()
            )
        )
        layers = self.layers
        obs_params = blocks.pop(0)
        cnj_paramss: list[Array] = []
        per_layer: list[tuple[Array, Array, Array, Array, Array]] = []
        for k, ((gb, pop), gen_nrn) in enumerate(
            zip(layers, self.generative_neuron_mans)
        ):
            int_params, gen_nrn_params, nz_loc, nz_prc = blocks[4 * k : 4 * k + 4]
            if k > 0:
                obs_params = _standard_normal_params(gb.att_hrm.obs_man)
            lkl_params = gb.att_hrm.lkl_fun_man.join_coords(obs_params, int_params)
            gb_cnj = perceptron_params[2 * k]
            rho_n = gb.cnj_fun(gb_cnj, lkl_params)
            gen_nrn_full = _generative_neuron_params(
                gb.att_hrm.pst_man, gen_nrn, gen_nrn_params
            )
            lat = pop.att_hrm.pst_man
            shape = (gen_nrn.data_dim, lat.data_dim)
            rows = jax.vmap(lat.join_location_precision)(
                nz_loc.reshape(shape), nz_prc.reshape(shape)
            )
            (nz_map,) = pop.att_hrm.crs_maps
            nz_params = nz_map.from_matrix(rows)
            pop_lkl = pop.gph_hrm.lkl_fun_man.join_coords(gen_nrn_full, nz_params)
            mlp_params = perceptron_params[2 * k + 1]
            # the prior's precision of z is exp(beta + a) = 1
            beta = -pop.cnj_fun.raw_precision(mlp_params, pop_lkl)
            pop_cnj = pop.cnj_fun.join_coords(mlp_params, beta)
            rho_z = pop.cnj_fun(pop_cnj, pop_lkl)
            theta_z = _standard_normal_params(lat) - rho_z
            cnj_paramss += [gb_cnj, pop_cnj]
            per_layer.append(
                (obs_params, int_params, gen_nrn_full - rho_n, nz_params, theta_z)
            )

        hrm_params = per_layer[-1][4]
        for k in reversed(range(len(layers))):
            gb, pop = layers[k]
            obs, int_, nrn, nz, _ = per_layer[k]
            hrm_params = pop.gph_hrm.join_coords(nrn, nz, hrm_params)
            # above the first layer, the Gaussian of the harmonium is the latent of the layer below
            obs = obs if k == 0 else per_layer[k - 1][4]
            hrm_params = gb.gph_hrm.join_coords(obs, int_, hrm_params)
        return self.join_coords(hrm_params, self.snd_man.join_coords(*cnj_paramss))

    @property
    def _perceptron_mans(self) -> list[Manifold]:
        """The parameters of the perceptron of each level's conjugation function, bottom first."""
        return [man for gb, pop in self.layers for man in (gb.cnj_fun, pop.cnj_fun.mlp)]

    # Exact computations by enumeration

    def exact_log_partition_function(self, params: Array) -> Array:
        """$\\psi$ of the graphical harmonium, by enumeration of the joint states of all neurons.

        Given the neurons, $x$ and every $z_k$ are Gaussian and integrate in closed form. The cost
        is $2^{\\sum_k n_k}$, so this is for small circuits, e.g. as ground truth. The graphical harmonium
        is normalizable only if every $z_k$ given the neurons has positive precision; where it is not,
        the result is not finite. The directed model $\\tilde p$ has no such condition.
        """
        log_weights, x_params = self._joint_state_log_weights(params)
        psi_x = jax.vmap(self.att_hrm.obs_man.log_partition_function)(x_params)
        return logsumexp(log_weights + psi_x)

    def exact_log_observable_density(self, params: Array, x: Array) -> Array:
        """$\\log p(x)$ of the graphical harmonium, by enumeration; see :meth:`exact_log_partition_function`."""
        obs_man = self.att_hrm.obs_man
        log_weights, x_params = self._joint_state_log_weights(params)
        log_joint = jax.vmap(
            lambda p: obs_man.log_density(p, x) + obs_man.log_partition_function(p)
        )(x_params)
        return logsumexp(log_weights + log_joint) - self.exact_log_partition_function(
            params
        )

    def _joint_state_log_weights(self, params: Array) -> tuple[Array, Array]:
        """For every joint state of the neurons: the log-weight with every $z_k$ integrated out, and the natural parameters of $p(x \\mid n_1)$."""
        layers = self.layers
        hrm_params, _ = self.split_coords(params)
        per_layer: list[tuple[Array, Array, Array, Array]] = []
        for gb, pop in layers:
            obs, int_, pop_params = gb.gph_hrm.split_coords(hrm_params)
            nrn, nz, hrm_params = pop.gph_hrm.split_coords(pop_params)
            per_layer.append((obs, int_, nrn, nz))
        top_lat_params = hrm_params
        sizes = [gb.att_hrm.pst_man.data_dim for gb, _ in layers]
        states = jnp.array(
            list(itertools.product([0.0, 1.0], repeat=sum(sizes))), dtype=float
        )

        def gaussian_likelihood_at(k: int, n: Array) -> Array:
            gb, _ = layers[k]
            obs, int_, _, _ = per_layer[k]
            gb_params = gb.att_hrm.join_coords(obs, int_, gb.att_hrm.pst_man.zeros())
            return gb.att_hrm.likelihood_at(gb_params, n)

        def at_state(state: Array) -> tuple[Array, Array]:
            ns = jnp.split(state, np.cumsum(sizes)[:-1])
            log_weight = jnp.asarray(0.0)
            for k, (_, pop) in enumerate(layers):
                _, _, nrn, nz = per_layer[k]
                lat_params = (
                    top_lat_params
                    if k == len(layers) - 1
                    else gaussian_likelihood_at(k + 1, ns[k + 1])
                )
                pop_params = pop.att_hrm.join_coords(nrn, nz, lat_params)
                z_params = pop.att_hrm.posterior_at(pop_params, ns[k])
                log_weight = (
                    log_weight
                    + jnp.dot(pop.att_hrm.obs_man.sufficient_statistic(ns[k]), nrn)
                    + pop.att_hrm.pst_man.log_partition_function(z_params)
                )
            return log_weight, gaussian_likelihood_at(0, ns[0])

        return jax.vmap(at_state)(states)


def _standard_normal_params(nrm: Any) -> Array:
    """Natural parameters of the standard normal."""
    return nrm.join_location_precision(
        jnp.zeros(nrm.data_dim), nrm.cov_man.from_matrix(jnp.eye(nrm.data_dim))
    )


def _generative_neuron_params(
    neurons: Boltzmann[Any], gen_nrn: ChordalBoltzmann, gen_params: Array
) -> Array:
    """The generative biases and couplings on $E$, in the layout of the neurons of the harmonium."""
    if neurons == gen_nrn:
        return gen_params
    if not isinstance(neurons, FullBoltzmann):
        raise TypeError("Neurons must be the generative family or a FullBoltzmann")
    return BoltzmannCouplingEmbedding(neurons, gen_nrn).embed(gen_params)


def canonical_circuit(
    obs_dim: int,
    obs_rep: PositiveDefinite,
    n_neurons: Sequence[int],
    edges: Sequence[Sequence[tuple[int, int]]],
    lat_dims: Sequence[int],
    variant: Variant,
    hidden_dims: tuple[int, ...],
) -> CanonicalCircuit:
    """A canonical circuit with one entry of ``n_neurons``, ``edges`` and ``lat_dims`` per layer, bottom first.

    ``edges`` is the graph $E_k$ of each layer, whose chordal completion carries the generative couplings. With ``variant``
    ``"partially_exact"`` the neurons are dense and the Gaussian-Boltzmann levels exact; with
    ``"approximate"`` the neurons are a chordal Boltzmann machine on the same completion and the Gaussian-Boltzmann
    levels learned. Every learned
    conjugation function is a multilayer perceptron with ``hidden_dims`` and a tanh activation.
    """
    edgess = tuple(tuple((int(i), int(j)) for i, j in es) for es in edges)
    above: GaussianBoltzmannLevel | None = None
    for k in reversed(range(len(n_neurons))):
        neurons: Boltzmann[Any] = (
            FullBoltzmann(n_neurons[k])
            if variant == "partially_exact"
            else ChordalBoltzmann.from_edges(n_neurons[k], edgess[k])
        )
        pop_hrm = BoltzmannPopulationCode(diagonal_normal(lat_dims[k]), neurons)
        pop_deep: DeepModel[Any] = (
            DifferentiableDeepModel(pop_hrm.pst_man) if above is None else above
        )
        pop_dom = AttachedHarmonium(pop_hrm, pop_deep.fst_man).lkl_fun_man
        pop: PopulationCodeLevel = VariationalLevel(
            pop_hrm,
            PopulationConjugationFunction(
                MultilayerPerceptron(pop_hrm.pst_man, pop_dom, hidden_dims, jax.nn.tanh)
            ),
            pop_deep,
        )
        gb_hrm: NormalBoltzmannMachine[Any, Any] = (
            NormalBoltzmannMachine(obs_dim, obs_rep, neurons)
            if k == 0
            else NormalBoltzmannMachine(lat_dims[k - 1], Diagonal(), neurons)
        )
        cnj_fun: Map[Any, Any] = (
            ExactConjugationFunction(gb_hrm)
            if variant == "partially_exact"
            else MultilayerPerceptron(
                neurons,
                AttachedHarmonium(gb_hrm, pop.gph_hrm).lkl_fun_man,
                hidden_dims,
                jax.nn.tanh,
            )
        )
        if k == 0:
            return CanonicalCircuit(gb_hrm, cnj_fun, pop, edgess)
        above = VariationalLevel(gb_hrm, cnj_fun, pop)
    raise ValueError("A canonical circuit needs at least one layer")
