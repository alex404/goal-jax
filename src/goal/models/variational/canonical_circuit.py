"""The canonical circuit: a deep model of alternating Gaussian and binary populations, fit variationally.

A circuit with $L$ layers is the chain $X \\leftarrow N_1 \\leftarrow Z_1 \\leftarrow N_2 \\leftarrow
\\cdots \\leftarrow N_L \\leftarrow Z_L$, with $x \\in \\mathbb R^{d_X}$ observed, binary neurons $n_k \\in
\\{0, 1\\}^{n_k}$, and Gaussian latents $z_k \\in \\mathbb R^{d_k}$. Each layer is a Gaussian-Boltzmann edge, which couples
the location of the Gaussian below to the activities of the neurons, and a population code, which
couples the activities of the neurons to the statistics of the Gaussian above. Given $z_k$ the
neurons $n_k$ have the couplings of their own family, and given the neurons every $z_k$ is Gaussian.

The circuit is a :class:`~goal.geometry.exponential_family.variational.VariationalConjugated` with
$2L$ levels, bottom first: a :class:`GaussianBoltzmannLevel` and a :class:`PopulationCodeLevel` per layer, the
bottom level being the :class:`CanonicalCircuit` itself. It comes in two variants:

- **Partially exact**: the neurons are dense (:class:`~goal.models.EnumeratedBoltzmann`), and the
  conjugation parameters of each Gaussian-Boltzmann level are the exact closed form of
  :class:`~goal.models.NormalBoltzmannMachine`, quadratic in $n$ with dense couplings. Only the population
  codes are approximated. Exact computations enumerate the neurons, so the populations must be small.
- **Approximate**: the neurons are a :class:`~goal.models.ChordalBoltzmann` on a graph $E$, in the
  harmonium, the prior and the recognition model, and the conjugation parameters of each Gaussian-Boltzmann
  level are learned in that family. The neurons are sampled and normalized by junction tree.

In both, every conjugation function that is learned is a multilayer perceptron of the likelihood
parameters of its level. The circuit is trained in its generative coordinates (:meth:`CanonicalCircuit.tie`):
the generative biases and couplings $\\theta^*_N$ on $E$ and the tuning curves of the neurons, with the
harmonium's $\\theta_N = \\theta^*_N - \\rho_N$ and $\\theta_{Z_k} = \\theta^*_{Z_k} - \\rho_{Z_k}$, where every
$\\theta^*_{Z_k}$ is the standard normal. The directed model $\\tilde p$ then does not depend on the conjugation
functions.
"""

import itertools
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal, override

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.scipy.special import logsumexp

from ...geometry import (
    AttachedHarmonium,
    Diagonal,
    IdentityEmbedding,
    MultilayerPerceptron,
    PositiveDefinite,
    SubCliquesEmbedding,
)
from ...geometry.exponential_family.variational import (
    DeepModel,
    DifferentiableDeepModel,
    VariationalConjugated,
)
from ...geometry.manifold.base import Manifold
from ...geometry.manifold.combinators import Pair, Tuple
from ...geometry.manifold.util import split_by_dims
from ..base.gaussian.boltzmann import (
    Boltzmann,
    ChordalBoltzmann,
    EnumeratedBoltzmann,
    FullBoltzmann,
)
from ..base.gaussian.generalized import Euclidean
from ..base.gaussian.normal import DiagonalNormal
from ..harmonium.lgm import BoltzmannCouplingEmbedding, NormalBoltzmannMachine
from ..harmonium.population_codes import PopulationCodeHarmonium

type Variant = Literal["partially_exact", "approximate"]


### Levels ###


@dataclass(frozen=True)
class GaussianBoltzmannLevel(VariationalConjugated[AttachedHarmonium[Any], Manifold]):
    """The level of a Gaussian-Boltzmann edge: a Gaussian (the observable, or the latent of the layer below) over the neurons of a layer.

    Its deep model is the population code level of the same layer. With ``cnj_map`` ``None``,
    $\\rho_N$ is the exact closed form of the :class:`~goal.models.NormalBoltzmannMachine`, which has
    dense couplings, so the neurons must be a :class:`~goal.models.FullBoltzmann`. Otherwise $\\rho_N$
    is ``cnj_map`` of the likelihood parameters, in the family of the neurons.
    """

    # Fields

    gb_hrm: NormalBoltzmannMachine[Any, Any]
    cnj_map: MultilayerPerceptron[Any, Any] | None
    deep: PopulationCodeLevel

    def __post_init__(self) -> None:
        if self.cnj_map is None and not isinstance(self.neurons, FullBoltzmann):
            raise ValueError(
                "Exact conjugation parameters have dense couplings: the neurons must be a FullBoltzmann"
            )

    # Overrides

    @property
    @override
    def gen_hrm(self) -> AttachedHarmonium[Any]:
        return AttachedHarmonium(self.gb_hrm, self.deep.gen_hrm)

    @property
    @override
    def cnj_fun_man(self) -> Manifold:
        return Euclidean(0) if self.cnj_map is None else self.cnj_map

    @property
    @override
    def pst_prr_emb(self) -> IdentityEmbedding[Any]:
        return IdentityEmbedding(self.deep.gen_hrm)

    @property
    @override
    def dep_man(self) -> PopulationCodeLevel:
        return self.deep

    @override
    def conjugation_parameters(self, lkl_params: Array, cnj_fun_params: Array) -> Array:
        """$\\rho_N$, placed on the neurons of the deep model."""
        (att_clq,) = self.gen_hrm.att_clqs
        rho = self.neuron_conjugation_parameters(lkl_params, cnj_fun_params)
        return SubCliquesEmbedding(att_clq, self.prr_man, self.neurons).embed(rho)

    # Methods

    @property
    def neurons(self) -> Boltzmann[Any]:
        return self.gb_hrm.pst_man

    def neuron_conjugation_parameters(
        self, lkl_params: Array, cnj_fun_params: Array
    ) -> Array:
        """$\\rho_N$ in natural coordinates of the neurons."""
        if self.cnj_map is None:
            return self.gb_hrm.conjugation_parameters(lkl_params)
        return self.cnj_map(cnj_fun_params, lkl_params)


@dataclass(frozen=True)
class PopulationConjugationFunction(
    Pair[MultilayerPerceptron[DiagonalNormal, Any], Euclidean]
):
    """The conjugation function of a population code level: a multilayer perceptron of the likelihood parameters, and a log-scale $\\beta$ per dimension of the latent.

    The perceptron reads the likelihood parameters $\\ell = (\\theta_N, \\Theta_{NZ})$ and outputs a location
    $l(\\ell)$ and a raw precision $a(\\ell)$. Mathematically, $\\rho_Z(\\ell)$ has location $l(\\ell)$ and
    precision $e^{\\beta} (e^{a(\\ell)} - 1)$, so a latent with precision $e^{\\beta}$ has precision
    $e^{\\beta + a(\\ell)} > 0$ after $\\rho_Z$ is added, at every likelihood.
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

    # Methods

    def __call__(self, params: Array, lkl_params: Array) -> Array:
        """$\\rho_Z$ in natural coordinates of the latent."""
        mlp_params, beta = self.split_coords(params)
        lat = self.mlp.cod_man
        loc, raw = lat.split_location_precision(self.mlp(mlp_params, lkl_params))
        return lat.join_location_precision(loc, jnp.exp(beta) * jnp.expm1(raw))

    def raw_precision(self, mlp_params: Array, lkl_params: Array) -> Array:
        """$a(\\ell)$, from the parameters of the perceptron."""
        _, raw = self.mlp.cod_man.split_location_precision(
            self.mlp(mlp_params, lkl_params)
        )
        return raw


@dataclass(frozen=True)
class PopulationCodeLevel(
    VariationalConjugated[AttachedHarmonium[Any], PopulationConjugationFunction]
):
    """The level of a population code: the neurons of a layer over its Gaussian latent.

    Its deep model is the Gaussian-Boltzmann level of the next layer, or the latent's family at the top.
    $\\rho_Z$ is ``cnj_map`` of the likelihood parameters $(\\theta_N, \\Theta_{NZ})$. Its precision is
    relative to $e^{\\beta}$, which :meth:`CanonicalCircuit.tie` sets to the precision of $\\theta_Z$, so that
    the prior and the recognition model over $z$ have positive precision at every likelihood. This
    restricts the family of conjugation functions to those that give valid distributions; it does not
    clamp anything.
    """

    # Fields

    pop_hrm: PopulationCodeHarmonium
    cnj_map: PopulationConjugationFunction
    deep: DeepModel[Any]

    # Overrides

    @property
    @override
    def gen_hrm(self) -> AttachedHarmonium[Any]:
        return AttachedHarmonium(self.pop_hrm, self.deep.fst_man)

    @property
    @override
    def cnj_fun_man(self) -> PopulationConjugationFunction:
        return self.cnj_map

    @property
    @override
    def pst_prr_emb(self) -> IdentityEmbedding[Any]:
        return IdentityEmbedding(self.deep.fst_man)

    @property
    @override
    def dep_man(self) -> DeepModel[Any]:
        return self.deep

    @override
    def conjugation_parameters(self, lkl_params: Array, cnj_fun_params: Array) -> Array:
        """$\\rho_Z$, placed on the latent of the deep model."""
        (att_clq,) = self.gen_hrm.att_clqs
        rho = self.cnj_map(cnj_fun_params, lkl_params)
        return SubCliquesEmbedding(att_clq, self.prr_man, self.pop_hrm.pst_man).embed(
            rho
        )


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
class CanonicalCircuit(GaussianBoltzmannLevel):
    """The canonical circuit: the Gaussian-Boltzmann level of the first layer, over every level above it.

    ``edges`` is the graph $E_k$ of the generative couplings of each layer, as pairs $i < j$.
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
            pop = gb.deep
            layers.append((gb, pop))
            if not isinstance(pop.deep, GaussianBoltzmannLevel):
                return tuple(layers)
            gb = pop.deep

    @property
    def generative_neuron_mans(self) -> tuple[ChordalBoltzmann, ...]:
        """The family of the generative biases and couplings of each layer: a Boltzmann machine on $E_k$."""
        return tuple(
            ChordalBoltzmann.from_edges(gb.neurons.data_dim, edges)
            for (gb, _), edges in zip(self.layers, self.edges)
        )

    @property
    def generative_man(self) -> GenerativeCoordinates:
        """The free blocks of the circuit, bottom first.

        Per layer: the observable of the Gaussian-Boltzmann harmonium (first layer only; above it, the
        latent below is the standard normal), its interaction, the generative biases and couplings
        $\\theta^*_N$ on $E_k$, and the coefficients of the location and of the precision of $z_k$ for
        each neuron and dimension, so that every neuron has its own tuning width. Last, the parameters of
        the perceptron of every learned conjugation function, level by level (:meth:`tie` sets the
        $\\beta$ of the population codes).
        """
        mans: list[Manifold] = []
        for k, ((gb, pop), gen_nrn) in enumerate(
            zip(self.layers, self.generative_neuron_mans)
        ):
            if k == 0:
                mans.append(gb.gb_hrm.obs_man)
            mans.append(gb.gb_hrm.int_man)
            mans.append(gen_nrn)
            lat_dim = pop.pop_hrm.lat_dim
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
                obs_params = _standard_normal_params(gb.gb_hrm.obs_man)
            lkl_params = gb.gb_hrm.lkl_fun_man.join_coords(obs_params, int_params)
            gb_cnj = perceptron_params[2 * k]
            rho_n = gb.neuron_conjugation_parameters(lkl_params, gb_cnj)
            gen_nrn_full = _generative_neuron_params(
                gb.neurons, gen_nrn, gen_nrn_params
            )
            lat = pop.pop_hrm.pst_man
            shape = (gen_nrn.data_dim, lat.data_dim)
            rows = jax.vmap(lat.join_location_precision)(
                nz_loc.reshape(shape), nz_prc.reshape(shape)
            )
            (nz_map,) = pop.pop_hrm.crs_maps
            nz_params = nz_map.from_matrix(rows)
            pop_lkl = pop.gen_hrm.lkl_fun_man.join_coords(gen_nrn_full, nz_params)
            mlp_params = perceptron_params[2 * k + 1]
            # the prior's precision of z is exp(beta + a) = 1
            beta = -pop.cnj_map.raw_precision(mlp_params, pop_lkl)
            pop_cnj = pop.cnj_map.join_coords(mlp_params, beta)
            rho_z = pop.cnj_map(pop_cnj, pop_lkl)
            theta_z = _standard_normal_params(lat) - rho_z
            cnj_paramss += [gb_cnj, pop_cnj]
            per_layer.append(
                (obs_params, int_params, gen_nrn_full - rho_n, nz_params, theta_z)
            )

        hrm_params = per_layer[-1][4]
        for k in reversed(range(len(layers))):
            gb, pop = layers[k]
            obs, int_, nrn, nz, _ = per_layer[k]
            hrm_params = pop.gen_hrm.join_coords(nrn, nz, hrm_params)
            # above the first layer, the Gaussian of the harmonium is the latent of the layer below
            obs = obs if k == 0 else per_layer[k - 1][4]
            hrm_params = gb.gen_hrm.join_coords(obs, int_, hrm_params)
        return self.join_coords(hrm_params, self.snd_man.join_coords(*cnj_paramss))

    @property
    def _perceptron_mans(self) -> list[Manifold]:
        """The parameters of the perceptron of each level's conjugation function, bottom first."""
        return [
            man for gb, pop in self.layers for man in (gb.cnj_fun_man, pop.cnj_map.mlp)
        ]

    # Exact computations by enumeration

    def exact_log_partition_function(self, params: Array) -> Array:
        """$\\psi$ of the graphical harmonium, by enumeration of the joint states of all neurons.

        Given the neurons, $x$ and every $z_k$ are Gaussian and integrate in closed form. The cost
        is $2^{\\sum_k n_k}$, so this is for small circuits, e.g. as ground truth.
        """
        log_weights, x_params = self._joint_state_log_weights(params)
        psi_x = jax.vmap(self.gb_hrm.obs_man.log_partition_function)(x_params)
        return logsumexp(log_weights + psi_x)

    def exact_log_observable_density(self, params: Array, x: Array) -> Array:
        """$\\log p(x)$ of the graphical harmonium, by enumeration; see :meth:`exact_log_partition_function`."""
        obs_man = self.gb_hrm.obs_man
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
            obs, int_, pop_params = gb.gen_hrm.split_coords(hrm_params)
            nrn, nz, hrm_params = pop.gen_hrm.split_coords(pop_params)
            per_layer.append((obs, int_, nrn, nz))
        top_lat_params = hrm_params
        sizes = [gb.neurons.data_dim for gb, _ in layers]
        states = jnp.array(
            list(itertools.product([0.0, 1.0], repeat=sum(sizes))), dtype=float
        )

        def gaussian_likelihood_at(k: int, n: Array) -> Array:
            gb, _ = layers[k]
            obs, int_, _, _ = per_layer[k]
            gb_params = gb.gb_hrm.join_coords(obs, int_, gb.neurons.zeros())
            return gb.gb_hrm.likelihood_at(gb_params, n)

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
                pop_params = pop.pop_hrm.join_coords(nrn, nz, lat_params)
                z_params = pop.pop_hrm.posterior_at(pop_params, ns[k])
                log_weight = (
                    log_weight
                    + jnp.dot(pop.pop_hrm.obs_man.sufficient_statistic(ns[k]), nrn)
                    + pop.pop_hrm.pst_man.log_partition_function(z_params)
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

    ``edges`` is the graph $E_k$ of the generative couplings of each layer. With ``variant``
    ``"partially_exact"`` the neurons are dense and the Gaussian-Boltzmann levels exact; with
    ``"approximate"`` the neurons are a chordal Boltzmann machine on $E_k$ and the Gaussian-Boltzmann
    levels learned. Every learned
    conjugation function is a multilayer perceptron with ``hidden_dims`` and a tanh activation.
    """
    edgess = tuple(tuple((int(i), int(j)) for i, j in es) for es in edges)
    deep: GaussianBoltzmannLevel | None = None
    for k in reversed(range(len(n_neurons))):
        neurons: Boltzmann[Any] = (
            EnumeratedBoltzmann(n_neurons[k])
            if variant == "partially_exact"
            else ChordalBoltzmann.from_edges(n_neurons[k], edgess[k])
        )
        pop_hrm = PopulationCodeHarmonium(neurons, lat_dims[k])
        pop_deep: DeepModel[Any] = (
            DifferentiableDeepModel(pop_hrm.pst_man) if deep is None else deep
        )
        pop_dom = AttachedHarmonium(pop_hrm, pop_deep.fst_man).lkl_fun_man
        pop = PopulationCodeLevel(
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
        cnj_map = (
            None
            if variant == "partially_exact"
            else MultilayerPerceptron(
                neurons,
                AttachedHarmonium(gb_hrm, pop.gen_hrm).lkl_fun_man,
                hidden_dims,
                jax.nn.tanh,
            )
        )
        if k == 0:
            return CanonicalCircuit(gb_hrm, cnj_map, pop, edgess)
        deep = GaussianBoltzmannLevel(gb_hrm, cnj_map, pop)
    raise ValueError("A canonical circuit needs at least one layer")
