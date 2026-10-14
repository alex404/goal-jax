"""The canonical circuit: a deep model of alternating Gaussian and binary populations, fit variationally.

A circuit with $L$ layers is the chain $X \\leftarrow N_1 \\leftarrow Z_1 \\leftarrow \\cdots
\\leftarrow N_L \\leftarrow Z_L$, with Gaussian $x$ and $z_k$ and binary neurons $n_k$. Each layer
is two levels (:class:`~goal.geometry.exponential_family.variational.VariationalConjugated`): a
Gaussian-Boltzmann level (:class:`~goal.models.NormalBoltzmannMachine`) and a population code level
(:class:`~goal.models.BoltzmannPopulationCode`). :func:`canonical_circuit` builds it, partially
exact or approximate.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal

import jax

from ...geometry import (
    Diagonal,
    GraphicalLevel,
    JunctionTree,
    MultilayerPerceptron,
    PositiveDefinite,
)
from ...geometry.exponential_family.variational import (
    BiasConjugationFunction,
    ConstantConjugationFunction,
    DeepModel,
    DifferentiableDeepModel,
    ExactConjugationFunction,
    VariationalConjugated,
)
from ...geometry.manifold.map import Map
from ..base.gaussian.boltzmann import (
    Boltzmann,
    ChordalBoltzmann,
    FullBoltzmann,
)
from ..base.gaussian.normal import diagonal_normal
from ..harmonium.lgm import NormalBoltzmannMachine
from ..harmonium.population_codes import BoltzmannPopulationCode

### Conjugation functions ###


### Circuit ###


@dataclass(frozen=True)
class CanonicalCircuit(VariationalConjugated[GraphicalLevel[Any]]):
    """The canonical circuit: the Gaussian-Boltzmann level of the first layer, over every level above it."""

    # Methods

    @property
    def lvl_mans(self) -> tuple[VariationalConjugated[Any], ...]:
        """The levels of the circuit, bottom first: the Gaussian-Boltzmann and the population code level of each layer, in turn."""
        levels: list[VariationalConjugated[Any]] = [self]
        while isinstance(dep_man := levels[-1].dep_man, VariationalConjugated):
            levels.append(dep_man)
        return tuple(levels)

    @property
    def hrm_pars(self) -> tuple[tuple[Any, Any], ...]:
        """The Gaussian-Boltzmann and the population code harmonium of each layer, bottom first."""
        hrms = [level.gph_hrm.obs_hrms[0] for level in self.lvl_mans]
        return tuple(zip(hrms[::2], hrms[1::2], strict=True))


def _place(hrm: Any, lat_man: Any) -> GraphicalLevel[Any]:
    """A harmonium as a graphical harmonium, its posterior on the leading nodes of the latent model."""
    return GraphicalLevel(((hrm, tuple(range(hrm.pst_man.n_nodes))),), lat_man)


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
    """
    n_layers = len(neural_connectivity)
    if n_layers == 0:
        raise ValueError("A canonical circuit needs at least one layer")

    def gaussian_boltzmann(
        k: int,
    ) -> tuple[
        GraphicalLevel[Any],
        tuple[Map[Any, Any]],
        VariationalConjugated[Any],
    ]:
        """The graphical harmonium, conjugation function and deep model of the Gaussian-Boltzmann level of layer $k$."""
        neurons: Boltzmann[Any] = (
            FullBoltzmann(neural_connectivity[k].n_nodes)
            if variant == "partially_exact"
            else ChordalBoltzmann(neural_connectivity[k])
        )
        pop = population_code(k, neurons)
        gb_hrm: NormalBoltzmannMachine[Any, Any] = (
            NormalBoltzmannMachine(obs_dim, obs_rep, neurons)
            if k == 0
            else NormalBoltzmannMachine(lat_dims[k - 1], Diagonal(), neurons)
        )
        cnj_fun: Map[Any, Any] = (
            ExactConjugationFunction(gb_hrm)
            if variant == "partially_exact"
            else ConstantConjugationFunction(neurons, gb_hrm.lkl_fun_man)
            if k == 0
            else BiasConjugationFunction(
                MultilayerPerceptron(neurons, gb_hrm.obs_man, hidden_dims, jax.nn.tanh),
                gb_hrm.lkl_fun_man,
            )
        )
        return _place(gb_hrm, pop.gph_hrm), (cnj_fun,), pop

    def population_code(k: int, neurons: Boltzmann[Any]) -> VariationalConjugated[Any]:
        """The population code level of layer $k$, over every level above it."""
        pop_hrm = BoltzmannPopulationCode(diagonal_normal(lat_dims[k]), neurons)
        deep: DeepModel[Any] = (
            DifferentiableDeepModel(pop_hrm.pst_man)
            if k == n_layers - 1
            else VariationalConjugated(*gaussian_boltzmann(k + 1))
        )
        cnj_fun = BiasConjugationFunction(
            MultilayerPerceptron(
                pop_hrm.pst_man, pop_hrm.obs_man, hidden_dims, jax.nn.tanh
            ),
            pop_hrm.lkl_fun_man,
        )
        return VariationalConjugated(_place(pop_hrm, deep.fst_man), (cnj_fun,), deep)

    return CanonicalCircuit(*gaussian_boltzmann(0))
