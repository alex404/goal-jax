"""Subspace relationships between manifolds.

An embedding defines how one manifold sits inside another: it provides maps for injecting points into the ambient space (``embed``), and for pulling back cotangent vectors --- i.e. gradients --- from the ambient space to the subspace (``pullback``). Linear embeddings additionally support projection and translation.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import override

import jax
from jax import Array

from .base import Manifold
from .combinators import Null, Tuple

### Linear Subspaces ###


@dataclass(frozen=True)
class Embedding[Sub: Manifold, Ambient: Manifold](ABC):
    """Defines how a smaller model space sits inside a larger one.

    Use an embedding when you want to convert parameters from a constrained model to a more general one (``embed``), or transform gradients computed in the general space back to the constrained space (``pullback``). For example, embedding a diagonal-covariance normal into a full-covariance normal.

    Mathematically, an injection $\\phi: \\mathcal{M} \\to \\mathcal{N}$ from a submanifold into an ambient space. Subclasses implement ``embed``; the default ``pullback`` is computed via JAX's VJP.
    """

    # Contract

    @property
    @abstractmethod
    def sub_man(self) -> Sub:
        """The submanifold."""

    @property
    @abstractmethod
    def amb_man(self) -> Ambient:
        """The ambient manifold."""

    @abstractmethod
    def embed(self, coords: Array) -> Array:
        """Embed a point from the submanifold into the ambient manifold."""

    def pullback(self, at_point: Array, cotangent_vector: Array) -> Array:
        """Transform a gradient from the ambient space to the subspace.

        This is what makes constrained optimization work: compute a gradient in the full model space, then pull it back to get the gradient in the restricted space.

        Mathematically, given the embedding $\\phi: \\mathcal M \\to \\mathcal N$, at $w = \\phi(v)$ the pullback is the mapping
        $$
        \\phi^*: T^*_w \\mathcal N \\to T^*_v \\mathcal M
        $$
        from the cotangent space on $\\mathcal N$ at $w$ to the cotangent space on $\\mathcal M$ at $v$. Where
        $\\omega \\in T^*_w \\mathcal N$ is a cotangent vector, the pullback is given by
        $$
        \\phi^*(\\omega) = \\omega \\cdot \\frac{\\partial \\phi}{\\partial v},
        $$
        where $\\frac{\\partial \\phi}{\\partial v}$ is the Jacobian of $\\phi$ at $v$.

        Args:
            at_point: Point on the submanifold
            cotangent_vector: Cotangent vector on the ambient manifold

        Returns:
            Pulled back cotangent vector on the submanifold
        """

        # Compute VJP function at the current point
        _, vjp_fn = jax.vjp(self.embed, at_point)

        # Apply VJP to transform the cotangent vector
        return vjp_fn(cotangent_vector)[0]


@dataclass(frozen=True)
class LinearEmbedding[Sub: Manifold, Ambient: Manifold](Embedding[Sub, Ambient], ABC):
    """An embedding that additionally supports projection and translation.

    Use a linear embedding when you need to move coordinates in both directions: ``embed`` to go from the smaller space to the larger, and ``project`` to extract the relevant components back out. Translation lets you update just the subspace components of an ambient point without touching the rest.

    Mathematically, a linear embedding $\\phi: \\mathcal{M} \\to \\mathcal{N}$ comes with:

    - A projection $\\pi: \\mathcal{N} \\to \\mathcal{M}$ satisfying $\\pi \\circ \\phi = \\text{id}_{\\mathcal{M}}$, and
    - a translation $\\tau(p, q) = p + \\phi(q)$.

    For linear embeddings the pullback is location-independent and reduces to projection.
    """

    # Contract

    @abstractmethod
    def project(self, coords: Array) -> Array:
        """Project from the ambient space to the subspace."""

    def translate(self, p_coords: Array, q_coords: Array) -> Array:
        """Translate an ambient point by a subspace displacement: $p + \\phi(q)$."""
        return p_coords + self.embed(q_coords)

    # Overrides

    @override
    def pullback(self, at_point: Array, cotangent_vector: Array) -> Array:
        """For linear embeddings, pullback reduces to projection."""
        return self.project(cotangent_vector)


@dataclass(frozen=True)
class IdentityEmbedding[M: Manifold](LinearEmbedding[M, M]):
    """The trivial case where sub and ambient are the same manifold.

    Used as the default embedding when a linear map operates on the full space without restriction (see ``AmbientMap``).
    """

    man: M

    @property
    @override
    def amb_man(self) -> M:
        return self.man

    @property
    @override
    def sub_man(self) -> M:
        return self.man

    @override
    def project(self, coords: Array) -> Array:
        return coords

    @override
    def embed(self, coords: Array) -> Array:
        return coords

    @override
    def translate(self, p_coords: Array, q_coords: Array) -> Array:
        return p_coords + q_coords


@dataclass(frozen=True)
class ComposedEmbedding[Sub: Manifold, Mid: Manifold, Ambient: Manifold](
    Embedding[Sub, Ambient], ABC
):
    """Chain two embeddings when a model space is nested two levels deep.

    For example, a diagonal normal embeds into a full normal, which in turn embeds into a harmonium --- composing these lets you pull gradients all the way back to the diagonal parameters.

    Mathematically, $\\phi = \\phi_2 \\circ \\phi_1: \\mathcal{M} \\to \\mathcal{L} \\to \\mathcal{N}$, with pullback in reverse order: $\\phi^*(\\omega) = \\phi_1^*(\\phi_2^*(\\omega))$.
    """

    # Fields

    sub_emb: Embedding[Sub, Mid]
    mid_emb: Embedding[Mid, Ambient]

    # Overrides

    @property
    @override
    def amb_man(self) -> Ambient:
        return self.mid_emb.amb_man

    @property
    def mid_man(self) -> Mid:
        return self.mid_emb.sub_man

    @property
    @override
    def sub_man(self) -> Sub:
        return self.sub_emb.sub_man

    @override
    def embed(self, coords: Array) -> Array:
        return self.mid_emb.embed(self.sub_emb.embed(coords))

    @override
    def pullback(self, at_point: Array, cotangent_vector: Array) -> Array:
        mid_point = self.sub_emb.embed(at_point)
        mid_cotangent = self.mid_emb.pullback(mid_point, cotangent_vector)
        return self.sub_emb.pullback(at_point, mid_cotangent)


@dataclass(frozen=True)
class LinearComposedEmbedding[
    Sub: Manifold,
    Mid: Manifold,
    Ambient: Manifold,
](ComposedEmbedding[Sub, Mid, Ambient], LinearEmbedding[Sub, Ambient]):
    """Linear version of ``ComposedEmbedding``, preserving projection and translation through the chain.

    Mathematically, $\\pi = \\pi_1 \\circ \\pi_2$ and $\\tau(p, q) = \\tau_2(p, \\tau_1(0_{\\mathcal{L}}, q))$.
    """

    # Fields

    sub_emb: LinearEmbedding[Sub, Mid]
    mid_emb: LinearEmbedding[Mid, Ambient]

    # Overrides

    @override
    def project(self, coords: Array) -> Array:
        mid = self.mid_emb.project(coords)
        return self.sub_emb.project(mid)

    @override
    def translate(self, p_coords: Array, q_coords: Array) -> Array:
        mid_zero = self.mid_man.zeros()
        mid = self.sub_emb.translate(mid_zero, q_coords)
        return self.mid_emb.translate(p_coords, mid)


@dataclass(frozen=True)
class TrivialEmbedding[Ambient: Manifold](LinearEmbedding[Null, Ambient]):
    """Embedding from a zero-dimensional manifold, used to mark a component as fixed.

    In a mixture or product structure, a trivial embedding on a component means that component carries no free parameters. ``embed`` returns zeros; ``project`` returns an empty array.
    """

    _amb_man: Ambient
    """The ambient manifold."""

    @property
    @override
    def amb_man(self) -> Ambient:
        return self._amb_man

    @property
    @override
    def sub_man(self) -> Null:
        return Null()

    @override
    def project(self, coords: Array) -> Array:
        """Project to zero-dimensional space (returns empty array)."""
        return jax.numpy.array([])

    @override
    def embed(self, coords: Array) -> Array:
        """Embed from zero-dimensional space (returns zeros)."""
        return self._amb_man.zeros()

    @override
    def translate(self, p_coords: Array, q_coords: Array) -> Array:
        """Translation does nothing since q_coords is empty."""
        return p_coords


@dataclass(frozen=True)
class TupleEmbedding[Component: Manifold, TupleMan: Tuple](
    LinearEmbedding[Component, TupleMan], ABC
):
    """Embeds a single component of a ``Tuple`` manifold, used to isolate one factor of a product space.

    For example, in a harmonium (which is a ``Triple`` of observable, interaction, and latent), a ``TupleEmbedding`` lets you extract or update just the observable parameters. Projection extracts the ``tup_idx``-th component; embedding sets that component and zeros the rest; translation adds to that component only.
    """

    # Contract

    @property
    @abstractmethod
    def tup_idx(self) -> int:
        """The index of the component in the tuple manifold."""

    def __post_init__(self):
        """Validate that tup_idx is in range."""
        zero_tuple = self.amb_man.zeros()
        components = self.amb_man.split_coords(zero_tuple)
        if self.tup_idx >= len(components):
            raise IndexError(
                f"Component index {self.tup_idx} out of range for {type(self.amb_man)}"
            )

    # Overrides

    @override
    def project(self, coords: Array) -> Array:
        return self.amb_man.split_coords(coords)[self.tup_idx]

    @override
    def embed(self, coords: Array) -> Array:
        components = list(self.amb_man.split_coords(self.amb_man.zeros()))
        components[self.tup_idx] = coords
        return self.amb_man.join_coords(*components)

    @override
    def translate(self, p_coords: Array, q_coords: Array) -> Array:
        components = list(self.amb_man.split_coords(p_coords))
        components[self.tup_idx] = components[self.tup_idx] + q_coords
        return self.amb_man.join_coords(*components)
