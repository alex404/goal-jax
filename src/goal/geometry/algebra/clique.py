"""Clique sets: the combinatorial skeleton of a graphical model.

A ``CliqueSet`` says which nodes a model has, which of them are observable, and which
groups of nodes interact. It is pure combinatorics over integer indices --- no manifolds,
no sufficient statistics, no JAX --- so it can be built, validated, and reasoned about
before any statistical structure is attached. ``CliqueManifold`` in
``manifold/graphical.py`` is what turns a clique set into a parameter layout.

Mathematically, a clique set describes an undirected graph $G = (V, E)$ together with a
cover of $V$ by complete subgraphs. Nodes $0, \\ldots, n_o - 1$ are observable and the
rest are latent. Edges are *derived*: two nodes are adjacent exactly when some clique
contains both. A harmonium's interaction structure $\\Theta = \\sum_C I^C_1 \\Theta^C
I^C_2$ has one block per clique of size at least two, and one bias per singleton clique.

Levels are derived too, by breadth-first search from the observable set: level $k$ is the
set of nodes at graph distance $k$ from an observable node. This is what recovers
hierarchy without declaring it --- a hierarchical mixture of Gaussians comes out as
levels $(1, 1, 1)$, a mixture of factor analyzers as $(1, 2)$, and canonical correlation
analysis as $(2, 1)$, from the cliques alone. Because cliques are complete and adjacency
implies a level gap of at most one, every clique automatically lies within a single level
or spans two consecutive levels; neither condition needs checking.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass


@dataclass(frozen=True)
class CliqueSet:
    """The nodes, observable/latent split, and interaction cliques of a graphical model.

    Cliques are index tuples. Every node must carry a singleton clique --- that is the
    slot its bias parameters live in --- and every node must be reachable from the
    observable set, so that its level is defined.

    Members are sorted within each clique and cliques are sorted among themselves at
    construction, so two equivalent descriptions compare and hash equal. The order
    parameters are actually laid out in is :attr:`canonical_cliques`, which is computed
    independently of storage order.
    """

    # Fields

    n_nodes: int
    """Total number of nodes. Observable nodes are ``0`` to ``n_observable - 1``."""

    n_observable: int
    """Number of observable nodes."""

    cliques: tuple[tuple[int, ...], ...]
    """The interacting groups of nodes, including one singleton per node."""

    def __post_init__(self) -> None:
        normalized = tuple(sorted(tuple(sorted(set(c))) for c in self.cliques))
        object.__setattr__(self, "cliques", normalized)
        self._validate()

    # Properties

    @property
    def latent_nodes(self) -> tuple[int, ...]:
        """The latent nodes, in index order."""
        return tuple(range(self.n_observable, self.n_nodes))

    @property
    def edges(self) -> tuple[tuple[int, int], ...]:
        """The derived adjacency: every pair of nodes sharing a clique, sorted."""
        pairs = {
            (i, j)
            for clique in self.cliques
            for k, i in enumerate(clique)
            for j in clique[k + 1 :]
        }
        return tuple(sorted(pairs))

    @property
    def node_levels(self) -> tuple[int, ...]:
        """The breadth-first distance of each node from the observable set."""
        return tuple(self._bfs_levels())

    @property
    def levels(self) -> tuple[tuple[int, ...], ...]:
        """Nodes grouped by breadth-first distance from the observable set.

        Level ``0`` is exactly the observable nodes; level ``k`` is the nodes first
        reached after $k$ steps. The length of this tuple is the depth of the model.
        """
        node_levels = self.node_levels
        depth = max(node_levels) + 1
        return tuple(
            tuple(i for i, lvl in enumerate(node_levels) if lvl == level)
            for level in range(depth)
        )

    @property
    def observable_cliques(self) -> tuple[tuple[int, ...], ...]:
        """Cliques lying wholly within level 0."""
        node_levels = self.node_levels
        return tuple(c for c in self.cliques if all(node_levels[i] == 0 for i in c))

    @property
    def crossing_cliques(self) -> tuple[tuple[int, ...], ...]:
        """Cliques spanning levels 0 and 1 --- the interactions the conjugation solve sees."""
        node_levels = self.node_levels
        return tuple(
            c
            for c in self.cliques
            if any(node_levels[i] == 0 for i in c)
            and any(node_levels[i] == 1 for i in c)
        )

    @property
    def boundary(self) -> tuple[int, ...]:
        """The level-1 nodes that interact directly with level 0.

        Mathematically, this is $B \\cap W$ in the graphical conjugation lemma: the
        latent-side nodes of the boundary, whose induced subgraph carries the conjugation
        parameters.
        """
        node_levels = self.node_levels
        return tuple(
            sorted({i for c in self.crossing_cliques for i in c if node_levels[i] == 1})
        )

    @property
    def tail_nodes(self) -> tuple[int, ...]:
        """Original indices of the nodes at level 1 and beyond, in tail index order.

        Level-1 nodes come first, so they become the tail's observable nodes.
        """
        node_levels = self.node_levels
        level_1 = [i for i in range(self.n_nodes) if node_levels[i] == 1]
        deeper = [i for i in range(self.n_nodes) if node_levels[i] > 1]
        return tuple(level_1 + deeper)

    @property
    def canonical_cliques(self) -> tuple[tuple[int, ...], ...]:
        """The cliques in parameter-layout order.

        Level-0 cliques first, then crossing cliques, then the tail --- with the same
        rule applied recursively inside the tail. For a two-node model this reproduces
        the ``[obs | int | lat]`` layout of a harmonium, and for a chain it nests so that
        each tail span is byte-identical to the layout of the model living on that tail.
        """
        head = self.observable_cliques + self.crossing_cliques
        if len(self.levels) == 1:
            return head
        tail_nodes = self.tail_nodes
        tail = self.tail()
        return head + tuple(
            tuple(sorted(tail_nodes[i] for i in c)) for c in tail.canonical_cliques
        )

    # Methods

    def tail(self) -> CliqueSet:
        """The clique set on levels 1 and beyond, reindexed from zero.

        Level-1 nodes become the tail's observable nodes, so the tail's own levels are
        this model's levels shifted down by one. Peeling tails is how a deep model is
        decomposed into the sequence of flat observable/latent splits that the
        conjugation lemma applies to.
        """
        node_levels = self.node_levels
        tail_nodes = self.tail_nodes
        index = {node: i for i, node in enumerate(tail_nodes)}
        cliques = tuple(
            tuple(index[i] for i in c)
            for c in self.cliques
            if all(node_levels[i] >= 1 for i in c)
        )
        n_observable = sum(1 for i in tail_nodes if node_levels[i] == 1)
        return CliqueSet(len(tail_nodes), n_observable, cliques)

    @classmethod
    def chain(cls, n_nodes: int) -> CliqueSet:
        """A chain ``0 --- 1 --- ... --- (n_nodes - 1)`` with node 0 observable.

        This covers the two-node harmonium (``chain(2)``) and the hierarchical mixture of
        Gaussians (``chain(3)``), whose levels come out as one node each.
        """
        singletons = tuple((i,) for i in range(n_nodes))
        links = tuple((i, i + 1) for i in range(n_nodes - 1))
        return cls(n_nodes, 1, singletons + links)

    # Private

    def _bfs_levels(self) -> list[int]:
        neighbours: list[set[int]] = [set() for _ in range(self.n_nodes)]
        for i, j in self.edges:
            neighbours[i].add(j)
            neighbours[j].add(i)

        levels = [-1] * self.n_nodes
        queue = deque(range(self.n_observable))
        for i in queue:
            levels[i] = 0
        while queue:
            i = queue.popleft()
            for j in neighbours[i]:
                if levels[j] < 0:
                    levels[j] = levels[i] + 1
                    queue.append(j)
        return levels

    def _validate(self) -> None:
        if self.n_nodes < 1:
            raise ValueError(f"n_nodes must be at least 1, got {self.n_nodes}")
        if not 1 <= self.n_observable <= self.n_nodes:
            raise ValueError(
                f"n_observable must be in 1..{self.n_nodes}, got {self.n_observable}"
            )
        self._validate_cliques()
        self._validate_nodes()

    def _validate_cliques(self) -> None:
        for clique in self.cliques:
            for i in clique:
                if not 0 <= i < self.n_nodes:
                    raise ValueError(
                        f"clique {clique} references out-of-range node {i}"
                    )
        if len(set(self.cliques)) != len(self.cliques):
            duplicates = sorted({c for c in self.cliques if self.cliques.count(c) > 1})
            raise ValueError(f"duplicate cliques: {duplicates}")

    def _validate_nodes(self) -> None:
        for i in range(self.n_nodes):
            if (i,) not in self.cliques:
                raise ValueError(f"node {i} has no singleton clique to hold its biases")
        for i, level in enumerate(self._bfs_levels()):
            if level < 0:
                raise ValueError(f"node {i} is not reachable from the observable nodes")
