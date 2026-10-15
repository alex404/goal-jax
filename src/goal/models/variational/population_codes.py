"""Variational population codes: a population code fit with learned, constant conjugation parameters, a filter over one, and the canonical circuit.

The population codes themselves are exact harmoniums, in
:mod:`goal.models.harmonium.population_codes`. A :class:`PopulationCodeProcess` is a latent process
whose state follows a learned (deterministic) map on beliefs, with each observation the response of
the population code. Filtering alternates the prediction of the transition with the conjugate update
of the population code, which is exact up to its conjugation residual.

The canonical circuit with $L$ layers is the chain $X \\leftarrow N_1 \\leftarrow Z_1 \\leftarrow
\\cdots \\leftarrow N_L \\leftarrow Z_L$, with Gaussian $x$ and $z_k$ and binary neurons $n_k$.
Each layer is two levels: a Gaussian-Boltzmann level (:class:`~goal.models.NormalBoltzmannMachine`)
and a population code level (:class:`~goal.models.BoltzmannPopulationCode`). It is a nested
:class:`~goal.geometry.exponential_family.variational.VariationalConjugated`, which
:func:`canonical_circuit` assembles from
:func:`~goal.geometry.exponential_family.variational.variational_level`, and variants of it are
assembled the same way.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal, override

import jax

from ...geometry import (
    Diagonal,
    GraphicalLevel,
    JunctionTree,
    MultilayerPerceptron,
    PositiveDefinite,
)
from ...geometry.exponential_family.base import Differentiable
from ...geometry.exponential_family.dynamical import VariationalLatentProcess
from ...geometry.exponential_family.variational import (
    BiasConjugationFunction,
    ConstantConjugationFunction,
    DeepModel,
    DifferentiableDeepModel,
    DifferentiableVariationalConjugated,
    ExactConjugationFunction,
    VariationalConjugated,
    variational_level,
)
from ...geometry.manifold.map import Map
from ..base.gaussian.boltzmann import Boltzmann, ChordalBoltzmann, FullBoltzmann
from ..base.gaussian.normal import diagonal_normal
from ..harmonium.lgm import NormalBoltzmannMachine
from ..harmonium.population_codes import BoltzmannPopulationCode, PopulationCode

type VariationalPopulationCode[Latent: Differentiable] = (
    DifferentiableVariationalConjugated[GraphicalLevel[Latent], Latent]
)
"""A population code with constant conjugation parameters (:func:`variational_population_code`)."""

type CanonicalCircuit = VariationalConjugated[GraphicalLevel[Any]]
"""A canonical circuit (:func:`canonical_circuit`): its bottom level, over every level above it."""

### Population codes ###


def variational_population_code[Latent: Differentiable](
    pop_hrm: PopulationCode[Any, Latent],
) -> VariationalPopulationCode[Latent]:
    """A population code with constant conjugation parameters: its conjugation function is a :class:`~goal.geometry.exponential_family.variational.ConstantConjugationFunction`, whose parameters are $\\rho$."""
    lat = pop_hrm.pst_man
    return variational_level(
        ((pop_hrm, ConstantConjugationFunction(lat, pop_hrm.lkl_fun_man)),),
        DifferentiableDeepModel(lat),
    )


@dataclass(frozen=True)
class PopulationCodeProcess[Latent: Differentiable, Transition: Map[Any, Any]](
    VariationalLatentProcess[Any, Latent]
):
    """State-space model with a population code emission and a transition map on beliefs.

    The transition maps natural parameters of the latent family to natural parameters of the
    latent family, independently of the observations; it must return valid parameters, which
    is the caller's choice of map (e.g. a :class:`~goal.geometry.MultilayerPerceptron`, or a
    subclass that keeps a precision positive).
    """

    # Fields

    pop_man: VariationalPopulationCode[Latent]
    _trn_map: Transition

    # Overrides

    @property
    @override
    def lat_man(self) -> Latent:
        return self.pop_man.pst_man

    @property
    @override
    def ems_hrm(self) -> VariationalPopulationCode[Latent]:
        return self.pop_man

    @property
    @override
    def trn_map(self) -> Transition:
        return self._trn_map


### Canonical circuit ###


def canonical_circuit(
    obs_dim: int,
    obs_rep: PositiveDefinite,
    neural_connectivity: Sequence[JunctionTree],
    lat_dims: Sequence[int],
    variant: Literal["partially_exact", "approximate"],
    hidden_dims: tuple[int, ...],
) -> CanonicalCircuit:
    """A canonical circuit with one entry of ``neural_connectivity`` and ``lat_dims`` per layer, bottom first.

    ``neural_connectivity`` is the chordal graph of the neurons of each layer: e.g.
    ``JunctionTree.from_edges(n, edges)``, or a :class:`~goal.geometry.ChainTree` for chains and
    bands. With ``"partially_exact"`` the neurons are a :class:`~goal.models.FullBoltzmann`,
    normalized by enumeration, so populations must be small and only the number of nodes of the
    graph is used, and each Gaussian-Boltzmann level has its exact conjugation function. With
    ``"approximate"`` the neurons are a :class:`~goal.models.ChordalBoltzmann` on the graph, and
    every conjugation function is learned. A learned conjugation function is a multilayer
    perceptron of the observable bias with ``hidden_dims`` and a tanh activation
    (:class:`~goal.geometry.exponential_family.variational.BiasConjugationFunction`), except at the
    bottom level, whose likelihood does not depend on the observations, so a constant suffices.
    The circuit is built from the top layer down, each level over the levels above it.
    """
    n_layers = len(neural_connectivity)
    if n_layers == 0:
        raise ValueError("A canonical circuit needs at least one layer")

    deep: DeepModel[Any] | None = None
    for k in reversed(range(n_layers)):
        neurons: Boltzmann[Any] = (
            FullBoltzmann(neural_connectivity[k].n_nodes)
            if variant == "partially_exact"
            else ChordalBoltzmann(neural_connectivity[k])
        )

        # Population code level
        pop_hrm = BoltzmannPopulationCode(diagonal_normal(lat_dims[k]), neurons)
        if deep is None:
            deep = DifferentiableDeepModel(pop_hrm.pst_man)
        pop_fun = BiasConjugationFunction(
            MultilayerPerceptron(
                pop_hrm.pst_man, pop_hrm.obs_man, hidden_dims, jax.nn.tanh
            ),
            pop_hrm.lkl_fun_man,
        )
        deep = variational_level(((pop_hrm, pop_fun),), deep)

        # Gaussian-Boltzmann level
        gb_hrm: NormalBoltzmannMachine[Any, Any] = (
            NormalBoltzmannMachine(obs_dim, obs_rep, neurons)
            if k == 0
            else NormalBoltzmannMachine(lat_dims[k - 1], Diagonal(), neurons)
        )
        gb_fun: Map[Any, Any] = (
            ExactConjugationFunction(gb_hrm)
            if variant == "partially_exact"
            else ConstantConjugationFunction(neurons, gb_hrm.lkl_fun_man)
            if k == 0
            else BiasConjugationFunction(
                MultilayerPerceptron(neurons, gb_hrm.obs_man, hidden_dims, jax.nn.tanh),
                gb_hrm.lkl_fun_man,
            )
        )
        deep = variational_level(((gb_hrm, gb_fun),), deep)

    assert isinstance(deep, VariationalConjugated)
    return deep
