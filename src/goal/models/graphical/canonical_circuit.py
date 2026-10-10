"""The canonical circuit: a deep model of alternating Gaussian and binary populations, fit variationally.

A circuit with $L$ layers is the chain $X \\leftarrow N_1 \\leftarrow Z_1 \\leftarrow N_2 \\leftarrow
\\cdots \\leftarrow N_L \\leftarrow Z_L$, with $x \\in \\mathbb R^{d_X}$ observed, binary neurons $n_k \\in
\\{0, 1\\}^{n_k}$, and Gaussian latents $z_k \\in \\mathbb R^{d_k}$. Each layer is a readout, which couples
the location of the Gaussian below to the activities of the neurons, and a population code, which
couples the activities of the neurons to the statistics of the Gaussian above. Given $z_k$ the
neurons $n_k$ have the couplings of their own family, and given the neurons every $z_k$ is Gaussian.

The circuit is a :class:`~goal.geometry.exponential_family.variational.VariationalConjugated` with
$2L$ levels, bottom first: a :class:`ReadoutLevel` and a :class:`PopulationCodeLevel` per layer, the
bottom level being the :class:`CanonicalCircuit` itself. It comes in two variants:

- **Partially exact**: the neurons are dense (:class:`~goal.models.EnumeratedBoltzmann`), and the
  conjugation parameters of each readout are the exact closed form of
  :class:`~goal.models.BoltzmannLGM`, quadratic in $n$ with dense couplings. Only the population
  codes are approximated. Exact computations enumerate the neurons, so the populations must be small.
- **Approximate**: the neurons are a :class:`~goal.models.ChordalBoltzmann` on a graph $E$, in the
  harmonium, the prior and the recognition model, and the conjugation parameters of each readout are
  learned in that family. The neurons are sampled and normalized by junction tree.

In both, every conjugation function that is learned is a multilayer perceptron of the likelihood
parameters of its level. The circuit is trained in its generative coordinates (:meth:`CanonicalCircuit.tie`):
the generative biases and couplings $\\theta^*_N = \\theta_N + \\rho_N$ on $E$, with the harmonium's
$\\theta_N = \\theta^*_N - \\rho_N$, and every $\\theta_{Z_k}$ held at the standard normal.
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
    CliqueMap,
    CrossTerm,
    Diagonal,
    IdentityEmbedding,
    MultilayerPerceptron,
    PositiveDefinite,
    Rectangular,
    SubCliquesEmbedding,
)
from ...geometry.exponential_family.harmonium import Harmonium
from ...geometry.exponential_family.variational import (
    DeepModel,
    DifferentiableDeepModel,
    VariationalConjugated,
)
from ...geometry.manifold.base import Manifold
from ...geometry.manifold.combinators import Tuple
from ...geometry.manifold.util import split_by_dims
from ..base.gaussian.boltzmann import (
    Boltzmann,
    ChordalBoltzmann,
    EnumeratedBoltzmann,
    FullBoltzmann,
)
from ..base.gaussian.generalized import Euclidean
from ..base.gaussian.normal import DiagonalNormal, diagonal_normal
from ..harmonium.lgm import (
    BoltzmannLGM,
    GeneralizedGaussianLocationEmbedding,
    NormalBoltzmannHarmonium,
)

type Variant = Literal["partially_exact", "approximate"]


### Harmonium ###


@dataclass(frozen=True)
class PopulationCodeHarmonium(Harmonium[Boltzmann[Any], DiagonalNormal]):
    """Harmonium with Boltzmann neurons as its observable and a diagonal Normal latent, coupled through the activities of the neurons only.

    Neuron $i$ has the logit $\\theta_{N,i} + \\Theta_{NZ,i} \\cdot \\mathbf s_Z(z)$, so its tuning
    curve is the exponential of a quadratic in $z$, and the couplings of the neurons do not depend
    on $z$. Given $n$, $z$ is Gaussian with natural parameters affine in $n$. Unlike
    :class:`~goal.models.BoltzmannNormalHarmonium`, the pairwise statistics of the neurons are not
    coupled to $z$.
    """

    # Fields

    neurons: Boltzmann[Any]
    lat_dim: int

    # Overrides

    @property
    @override
    def crs_trms(self) -> tuple[CrossTerm, ...]:
        """The activities of the neurons and the statistics of the latent, coupled."""
        int_map = CliqueMap(
            Rectangular(),
            GeneralizedGaussianLocationEmbedding(self.obs_man),
            IdentityEmbedding(self.pst_man),
        )
        return (CrossTerm((0,), (0,), int_map),)

    @property
    @override
    def obs_man(self) -> Boltzmann[Any]:
        return self.neurons

    @property
    @override
    def pst_man(self) -> DiagonalNormal:
        return diagonal_normal(self.lat_dim)


### Levels ###


@dataclass(frozen=True)
class ReadoutLevel(VariationalConjugated[AttachedHarmonium[Any], Manifold]):
    """The level of a readout: a Gaussian (the observable, or the latent of the layer below) over the neurons of a layer.

    Its deep model is the population code level of the same layer. With ``cnj_map`` ``None``,
    $\\rho_N$ is the exact closed form of :class:`~goal.models.BoltzmannLGM`, which has dense
    couplings, so the neurons must be a :class:`~goal.models.FullBoltzmann`. Otherwise $\\rho_N$
    is ``cnj_map`` of the likelihood parameters, in the family of the neurons.
    """

    # Fields

    rdt_hrm: NormalBoltzmannHarmonium[Any, Any]
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
        return AttachedHarmonium(self.rdt_hrm, self.deep.gen_hrm)

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
        return self.rdt_hrm.pst_man

    def neuron_conjugation_parameters(
        self, lkl_params: Array, cnj_fun_params: Array
    ) -> Array:
        """$\\rho_N$ in natural coordinates of the neurons."""
        if self.cnj_map is None:
            lgm = BoltzmannLGM(
                self.rdt_hrm.obs_dim, self.rdt_hrm.obs_rep, self.neurons.data_dim
            )
            return lgm.conjugation_parameters(lkl_params)
        return self.cnj_map(cnj_fun_params, lkl_params)


@dataclass(frozen=True)
class PopulationCodeLevel(
    VariationalConjugated[AttachedHarmonium[Any], MultilayerPerceptron[Any, Any]]
):
    """The level of a population code: the neurons of a layer over its Gaussian latent.

    Its deep model is the readout level of the next layer, or the latent's family at the top.
    $\\rho_Z$ is ``cnj_map`` of the likelihood parameters $(\\theta_N, \\Theta_{NZ})$, read as a
    location and a raw precision $a$. The precision of $\\rho_Z$ is $\\mathrm{softplus}(a + c) - 1$,
    with $c$ such that it vanishes at $a = 0$, so that with $\\theta_Z$ the standard normal (as
    :meth:`CanonicalCircuit.tie` holds it) the prior and the recognition model over $z$ always have
    positive precision. This restricts the family of conjugation functions to those that give
    valid distributions; it does not clamp anything.
    """

    # Fields

    pop_hrm: PopulationCodeHarmonium
    cnj_map: MultilayerPerceptron[DiagonalNormal, Any]
    deep: DeepModel[Any]

    # Overrides

    @property
    @override
    def gen_hrm(self) -> AttachedHarmonium[Any]:
        return AttachedHarmonium(self.pop_hrm, self.deep.fst_man)

    @property
    @override
    def cnj_fun_man(self) -> MultilayerPerceptron[DiagonalNormal, Any]:
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
        lat = self.pop_hrm.pst_man
        loc, raw = lat.split_location_precision(self.cnj_map(cnj_fun_params, lkl_params))
        prc = jax.nn.softplus(raw + np.log(np.e - 1.0)) - 1.0
        (att_clq,) = self.gen_hrm.att_clqs
        rho = lat.join_location_precision(loc, prc)
        return SubCliquesEmbedding(att_clq, self.prr_man, lat).embed(rho)


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
class CanonicalCircuit(ReadoutLevel):
    """The canonical circuit: the readout level of the first layer, over every level above it.

    ``edges`` is the graph $E_k$ of the generative couplings of each layer, as pairs $i < j$.
    """

    # Fields

    edges: tuple[tuple[tuple[int, int], ...], ...]

    # Methods

    @property
    def layers(self) -> tuple[tuple[ReadoutLevel, PopulationCodeLevel], ...]:
        """The readout and population code level of each layer, bottom first."""
        layers: list[tuple[ReadoutLevel, PopulationCodeLevel]] = []
        rdt: ReadoutLevel = self
        while True:
            pop = rdt.deep
            layers.append((rdt, pop))
            if not isinstance(pop.deep, ReadoutLevel):
                return tuple(layers)
            rdt = pop.deep

    @property
    def generative_neuron_mans(self) -> tuple[ChordalBoltzmann, ...]:
        """The family of the generative biases and couplings of each layer: a Boltzmann machine on $E_k$."""
        return tuple(
            ChordalBoltzmann.from_edges(rdt.neurons.data_dim, edges)
            for (rdt, _), edges in zip(self.layers, self.edges)
        )

    @property
    def generative_man(self) -> GenerativeCoordinates:
        """The free blocks of the circuit, bottom first.

        Per layer: the observable of the readout (first layer only; above it, the latent below is
        held at the standard normal), the readout interaction, the generative biases and
        couplings $\\theta^*_N$ on $E_k$, the coefficients of the location of $z_k$ for each neuron,
        and the coefficients of the precision of $z_k$, shared by the neurons, so that the tuning
        curves have a common width. Last, the conjugation function parameters of every level.
        """
        mans: list[Manifold] = []
        for k, ((rdt, pop), gen_nrn) in enumerate(
            zip(self.layers, self.generative_neuron_mans)
        ):
            if k == 0:
                mans.append(rdt.rdt_hrm.obs_man)
            mans.append(rdt.rdt_hrm.int_man)
            mans.append(gen_nrn)
            lat_dim = pop.pop_hrm.lat_dim
            mans.append(Euclidean(gen_nrn.data_dim * lat_dim))
            mans.append(Euclidean(lat_dim))
        mans.append(self.snd_man)
        return GenerativeCoordinates(tuple(mans))

    def tie(self, gen_params: Array) -> Array:
        """Circuit parameters from generative coordinates (:attr:`generative_man`).

        $\\theta_{N_k} = \\theta^*_{N_k} - \\rho_{N_k}$, with $\\rho_{N_k}$ the conjugation parameters
        of the readout of layer $k$ at its likelihood. Every $\\theta_{Z_k}$ is the standard normal:
        an affine change of $z_k$ is absorbed by the interactions it enters.
        """
        blocks = list(self.generative_man.split_coords(gen_params))
        cnj_params = blocks.pop()
        cnj_paramss = self.snd_man.split_coords(cnj_params)
        layers = self.layers
        obs_params = blocks.pop(0)
        per_layer: list[tuple[Array, Array, Array, Array]] = []
        for k, ((rdt, pop), gen_nrn) in enumerate(
            zip(layers, self.generative_neuron_mans)
        ):
            int_params, gen_nrn_params, nz_loc, nz_prc = blocks[4 * k : 4 * k + 4]
            if k > 0:
                obs_params = _standard_normal_params(rdt.rdt_hrm.obs_man)
            lkl_params = rdt.rdt_hrm.lkl_fun_man.join_coords(obs_params, int_params)
            rho = rdt.neuron_conjugation_parameters(lkl_params, cnj_paramss[2 * k])
            nrn_params = (
                _generative_neuron_params(rdt.neurons, gen_nrn, gen_nrn_params) - rho
            )
            lat = pop.pop_hrm.pst_man
            rows = jax.vmap(lat.join_location_precision, in_axes=(0, None))(
                nz_loc.reshape(gen_nrn.data_dim, lat.data_dim), nz_prc
            )
            (nz_map,) = pop.pop_hrm.crs_maps
            per_layer.append((obs_params, int_params, nrn_params, nz_map.from_matrix(rows)))

        _, top_pop = layers[-1]
        hrm_params = _standard_normal_params(top_pop.pop_hrm.pst_man)
        for (rdt, pop), (obs, int_, nrn, nz) in reversed(list(zip(layers, per_layer))):
            hrm_params = pop.gen_hrm.join_coords(nrn, nz, hrm_params)
            hrm_params = rdt.gen_hrm.join_coords(obs, int_, hrm_params)
        return self.join_coords(hrm_params, cnj_params)

    # Exact computations by enumeration

    def exact_log_partition_function(self, params: Array) -> Array:
        """$\\psi$ of the graphical harmonium, by enumeration of the joint states of all neurons.

        Given the neurons, $x$ and every $z_k$ are Gaussian and integrate in closed form. The cost
        is $2^{\\sum_k n_k}$, so this is for small circuits, e.g. as ground truth.
        """
        log_weights, x_params = self._joint_state_log_weights(params)
        psi_x = jax.vmap(self.rdt_hrm.obs_man.log_partition_function)(x_params)
        return logsumexp(log_weights + psi_x)

    def exact_log_observable_density(self, params: Array, x: Array) -> Array:
        """$\\log p(x)$ of the graphical harmonium, by enumeration; see :meth:`exact_log_partition_function`."""
        obs_man = self.rdt_hrm.obs_man
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
        for rdt, pop in layers:
            obs, int_, pop_params = rdt.gen_hrm.split_coords(hrm_params)
            nrn, nz, hrm_params = pop.gen_hrm.split_coords(pop_params)
            per_layer.append((obs, int_, nrn, nz))
        top_lat_params = hrm_params
        sizes = [rdt.neurons.data_dim for rdt, _ in layers]
        states = jnp.array(
            list(itertools.product([0.0, 1.0], repeat=sum(sizes))), dtype=float
        )

        def readout_at(k: int, n: Array) -> Array:
            rdt, _ = layers[k]
            obs, int_, _, _ = per_layer[k]
            rdt_params = rdt.rdt_hrm.join_coords(obs, int_, rdt.neurons.zeros())
            return rdt.rdt_hrm.likelihood_at(rdt_params, n)

        def at_state(state: Array) -> tuple[Array, Array]:
            ns = jnp.split(state, np.cumsum(sizes)[:-1])
            log_weight = jnp.asarray(0.0)
            for k, (_, pop) in enumerate(layers):
                _, _, nrn, nz = per_layer[k]
                lat_params = (
                    top_lat_params if k == len(layers) - 1 else readout_at(k + 1, ns[k + 1])
                )
                pop_params = pop.pop_hrm.join_coords(nrn, nz, lat_params)
                z_params = pop.pop_hrm.posterior_at(pop_params, ns[k])
                log_weight = (
                    log_weight
                    + jnp.dot(pop.pop_hrm.obs_man.sufficient_statistic(ns[k]), nrn)
                    + pop.pop_hrm.pst_man.log_partition_function(z_params)
                )
            return log_weight, readout_at(0, ns[0])

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
    n_neurons = neurons.n_neurons
    rows, cols = np.triu_indices(n_neurons, 1)
    index = {(int(i), int(j)): k for k, (i, j) in enumerate(zip(rows, cols))}
    positions = np.array(
        [index[(int(i), int(j))] for i, j in gen_nrn.junction_tree.chordal_edges],
        dtype=int,
    )
    diag, off_diag = gen_nrn.split_couplings(gen_params)
    dense_off = jnp.zeros(rows.size).at[positions].set(off_diag)
    return neurons.join_couplings(diag, dense_off)


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
    ``"partially_exact"`` the neurons are dense and the readouts exact; with ``"approximate"`` the
    neurons are a chordal Boltzmann machine on $E_k$ and the readouts learned. Every learned
    conjugation function is a multilayer perceptron with ``hidden_dims`` and a tanh activation.
    """
    edgess = tuple(tuple((int(i), int(j)) for i, j in es) for es in edges)
    deep: ReadoutLevel | None = None
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
            MultilayerPerceptron(pop_hrm.pst_man, pop_dom, hidden_dims, jax.nn.tanh),
            pop_deep,
        )
        rdt_hrm: NormalBoltzmannHarmonium[Any, Any] = (
            NormalBoltzmannHarmonium(obs_dim, obs_rep, neurons)
            if k == 0
            else NormalBoltzmannHarmonium(lat_dims[k - 1], Diagonal(), neurons)
        )
        cnj_map = (
            None
            if variant == "partially_exact"
            else MultilayerPerceptron(
                neurons,
                AttachedHarmonium(rdt_hrm, pop.gen_hrm).lkl_fun_man,
                hidden_dims,
                jax.nn.tanh,
            )
        )
        if k == 0:
            return CanonicalCircuit(rdt_hrm, cnj_map, pop, edgess)
        deep = ReadoutLevel(rdt_hrm, cnj_map, pop)
    raise ValueError("A canonical circuit needs at least one layer")
