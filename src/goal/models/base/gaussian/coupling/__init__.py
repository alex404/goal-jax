"""Coupling kernels and topology for Boltzmann machines.

Algorithmic backends for the coupling-matrix shape manifolds in
:mod:`~goal.models.base.gaussian.boltzmann`, as free functions over static
structure and a parameter vector.
:mod:`~goal.models.base.gaussian.coupling.dense` enumerates and Gibbs-samples
the dense case; :mod:`~goal.models.base.gaussian.coupling.junction_tree`
builds static junction-tree topology (triangulation, maximal cliques,
spanning tree); :mod:`~goal.models.base.gaussian.coupling.sum_product` runs
exact inference over it.
"""

from .dense import (
    dense_gibbs_step,
    dense_log_partition,
    dense_sample,
    dense_states,
    dense_unit_conditional_prob,
)
from .junction_tree import ChainTree, JunctionTree
from .sum_product import chain_log_partition, chain_sample, jt_log_partition, jt_sample

__all__ = [
    "ChainTree",
    "JunctionTree",
    "chain_log_partition",
    "chain_sample",
    "dense_gibbs_step",
    "dense_log_partition",
    "dense_sample",
    "dense_states",
    "dense_unit_conditional_prob",
    "jt_log_partition",
    "jt_sample",
]
