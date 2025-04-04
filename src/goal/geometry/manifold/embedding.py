"""Defines subspace relationships between manifolds.

This module provides the fundamental embedding operations that define how manifolds relate to each other through subspace relationships.

In practice, embeddings allow you to transform parameters between different model spaces, project complex models to simpler ones, and define how changes in one manifold affect another.

In theory, embeddings formalize the essential concept from information geometry where one statistical model (manifold) can be viewed as a submanifold of another. This corresponds to statistical concepts like sufficiency, model constraints, and parameter restrictions.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import override

import jax
import jax.numpy as jnp
from jax import Array

from .base import Coordinates, Dual, Manifold, Point
from .combinators import Tuple

### Linear Subspaces ###


@dataclass(frozen=True)
class Embedding[Sub: Manifold, Ambient: Manifold](ABC):
    """Embeddings let you transform points between manifolds, enabling operations like embedding and pullback that are crucial for model specification and constrained optimization.

    In theory, an embedding relationship between manifolds $\\mathcal{M} \\subset \\mathcal{N}$ is characterized by an embedding map $\\phi: \\mathcal{M} \\to \\mathcal{N}$ that injects the submanifold into the ambient space.
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
    def embed[C: Coordinates](self, p: Point[C, Sub]) -> Point[C, Ambient]:
        """Embed a point from the submanifold in the ambient manifold."""

    def pullback[C: Coordinates](
        self, at_point: Point[C, Sub], cotangent_vector: Point[Dual[C], Ambient]
    ) -> Point[Dual[C], Sub]:
        """Pull back a cotangent vector from the ambient manifold to the submanifold.

        In practice, this transforms gradients computed in the larger model space to the constrained model space, enabling optimization in the lower-dimensional space. It's essential for efficiently computing natural gradients on submodels and implementing constrained optimization.

        In theory, given the embedding $\\phi: \\mathcal M \\to \\mathcal N$. At $w = \\phi(v)$, the pullback is the mapping
        $$
        \\phi^*: T^*_w \\mathcal N \\to T^*_v \\mathcal M
        $$
        from the cotangent space on $\\mathcal N$ at $w$ to the cotangent space on $\\mathcal M$ at $v$. Where $\\omega \\in T^*_w \\mathcal N$ is a cotangent vector, the pullback is given by
        $$
        \\phi^*(\\omega) = \\omega \\cdot \\frac{\\partial \\phi}{\\partial v},
        $$
        where $\\frac{\\partial \\phi}{\\partial v}$ is the Jacobian of $\\phi$ at $v$.
        """

        def embed_func(sub_array: Array) -> Array:
            sub_point: Point[C, Sub] = self.sub_man.point(sub_array)
            embedded_point = self.embed(sub_point)
            return embedded_point.array

        # Compute VJP function at the current point
        _, vjp_fn = jax.vjp(embed_func, at_point.array)

        # Apply VJP to transform the cotangent vector
        pulled_back_array = vjp_fn(cotangent_vector.array)[0]

        return self.sub_man.point(pulled_back_array)


@dataclass(frozen=True)
class LinearEmbedding[Sub: Manifold, Ambient: Manifold](Embedding[Sub, Ambient], ABC):
    """A structure-preserving embedding with projection and translation operations.

    In practice, linear embeddings enable direct conversion of parameters between model spaces through projection, while preserving vector operations like translation. These operations are fundamental to algorithms like EM and natural gradient descent on constrained models.

    A linear embedding $\\phi: \\mathcal{M} \\to \\mathcal{N}$ supports

        - A projection map $\\pi: \\mathcal{N} \\to \\mathcal{M}$ satisfying $\\pi \\circ \\phi = \\text{id}_{\\mathcal{M}}$ (the identity on $\\mathcal{M}$), and
        - A translation operation $\\tau(p,q) = p + \\phi(q)$ that preserves the vector space structure.
    """

    # Contract

    @abstractmethod
    def project[C: Coordinates](self, p: Point[C, Ambient]) -> Point[C, Sub]:
        """Project a point from the ambient space to the subspace."""

    def translate[C: Coordinates](
        self, p: Point[C, Ambient], q: Point[C, Sub]
    ) -> Point[C, Ambient]:
        """Translate point by subspace components."""
        return p + self.embed(q)

    # Overrides

    @override
    def pullback[C: Coordinates](
        self, at_point: Point[C, Sub], cotangent_vector: Point[Dual[C], Ambient]
    ) -> Point[Dual[C], Sub]:
        """For linear subspaces, pullback is location-independent."""
        return self.project(cotangent_vector)


@dataclass(frozen=True)
class IdentityEmbedding[M: Manifold](LinearEmbedding[M, M]):
    """Identity subspace relationship for a manifold M."""

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
    def project[C: Coordinates](self, p: Point[Dual[C], M]) -> Point[Dual[C], M]:
        return p

    @override
    def embed[C: Coordinates](self, p: Point[C, M]) -> Point[C, M]:
        return p

    @override
    def translate[C: Coordinates](self, p: Point[C, M], q: Point[C, M]) -> Point[C, M]:
        return p + q


@dataclass(frozen=True)
class ComposedEmbedding[Sub: Manifold, Mid: Manifold, Ambient: Manifold](
    Embedding[Sub, Ambient], ABC
):
    """Composed embeddings allow you to chain multiple model transformations, creating hierarchical model reductions that apply constraints in stages. This is particularly valuable for complex model relationships with multiple levels of abstraction.

    In theory, given embeddings $\\phi_1: \\mathcal{M} \\to \\mathcal{L}$ and $\\phi_2: \\mathcal{L} \\to \\mathcal{N}$, their composition $\\phi = \\phi_2 \\circ \\phi_1: \\mathcal{M} \\to \\mathcal{N}$ forms a new embedding with:

        - Embedding map: $\\phi(p) = \\phi_2(\\phi_1(p))$ for $p \\in \\mathcal{M}$
        - Pullback operation: $\\phi^*(\\omega) = \\phi_1^*(\\phi_2^*(\\omega))$ for cotangent vectors $\\omega \\in T^*\\mathcal{N}$
    """

    # Fields

    mid_emb: Embedding[Mid, Ambient]
    sub_emb: Embedding[Sub, Mid]

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
    def embed[C: Coordinates](self, p: Point[C, Sub]) -> Point[C, Ambient]:
        mid = self.sub_emb.embed(p)
        return self.mid_emb.embed(mid)

    @override
    def pullback[C: Coordinates](
        self, at_point: Point[C, Sub], cotangent_vector: Point[Dual[C], Ambient]
    ) -> Point[Dual[C], Sub]:
        # First, embed the point to get its location in the middle space
        mid_point = self.sub_emb.embed(at_point)

        # Pull back through the upper subspace
        mid_cotangent = self.mid_emb.pullback(mid_point, cotangent_vector)

        # Pull back through the lower subspace
        return self.sub_emb.pullback(at_point, mid_cotangent)


@dataclass(frozen=True)
class LinearComposedEmbedding[
    Sub: Manifold,
    Mid: Manifold,
    Ambient: Manifold,
](ComposedEmbedding[Sub, Mid, Ambient], LinearEmbedding[Sub, Ambient]):
    """Linear composed embeddings maintain projection and translation operations through nested embeddings.

    In theory, given linear embeddings $(\\phi_1, \\pi_1, \\tau_1)$ from $\\mathcal{M} \\to \\mathcal{L}$ and $(\\phi_2, \\pi_2, \\tau_2)$ from $\\mathcal{L} \\to \\mathcal{N}$, their composition forms a linear embedding $(\\phi, \\pi, \\tau)$ from $\\mathcal{M} \\to \\mathcal{N}$ where

        - Projection map: $\\pi = \\pi_1 \\circ \\pi_2$
        - Translation operation: $\\tau(p, q) = \\tau_2(p, \\tau_1(0_{\\mathcal{L}}, q))$, where $0_{\\mathcal{L}}$ is the zero point in $\\mathcal{L}$
    """

    # Fields

    mid_emb: LinearEmbedding[Mid, Ambient]
    sub_emb: LinearEmbedding[Sub, Mid]

    # Overrides

    @override
    def project[C: Coordinates](
        self, p: Point[Dual[C], Ambient]
    ) -> Point[Dual[C], Sub]:
        mid = self.mid_emb.project(p)
        return self.sub_emb.project(mid)

    @override
    def translate[C: Coordinates](
        self, p: Point[C, Ambient], q: Point[C, Sub]
    ) -> Point[C, Ambient]:
        mid_zero: Point[C, Mid] = self.mid_man.point(
            jnp.zeros(self.mid_emb.sub_man.dim)
        )
        mid = self.sub_emb.translate(mid_zero, q)
        return self.mid_emb.translate(p, mid)


@dataclass(frozen=True)
class TupleEmbedding[Component: Manifold, TupleMan: Tuple](
    LinearEmbedding[Component, TupleMan], ABC
):
    """Embedding that projects to or embeds from a specific component of a tuple manifold.

    This embedding provides a way to work with individual components of tuple manifolds like Pair and Triple, enabling operations such as projection, embedding, and translation on specific components.
    """

    # Fields
    cmp_idx: int
    cmp_man: Component
    tup_man: TupleMan

    def __post_init__(self):
        """Validate that the component manifold matches the expected component at the given index."""
        # Get all components
        zero_tuple = self.tup_man.zeros()
        components = self.tup_man.split_params(zero_tuple)

        # Check if index is valid
        if self.cmp_idx >= len(components):
            raise IndexError(
                f"Component index {self.cmp_idx} out of range for {type(self.tup_man)}"
            )

    @property
    @override
    def amb_man(self) -> TupleMan:
        """The tuple manifold (super-manifold)."""
        return self.tup_man

    @property
    @override
    def sub_man(self) -> Component:
        """The component manifold (sub-manifold)."""
        return self.cmp_man

    @override
    def project[C: Coordinates](self, p: Point[C, TupleMan]) -> Point[C, Component]:
        """Project a point from the tuple manifold to the specified component."""
        # Split the tuple and get the component at the specified index
        components = self.amb_man.split_params(p)
        return components[self.cmp_idx]  # type: ignore[return-value]

    @override
    def embed[C: Coordinates](self, p: Point[C, Component]) -> Point[C, TupleMan]:
        """Creates a new point in the tuple manifold with the specified component set to p and all other components set to zero."""
        # Split a zero tuple to get zero components
        zero_tuple = self.amb_man.zeros()
        components = list(self.amb_man.split_params(zero_tuple))

        # Replace the component at the specified index
        components[self.cmp_idx] = p

        # Join the components back into a tuple
        return self.amb_man.join_params(*components)

    @override
    def translate[C: Coordinates](
        self, p: Point[C, TupleMan], q: Point[C, Component]
    ) -> Point[C, TupleMan]:
        """Updates the component at the specified index by adding q, leaving other components unchanged."""
        # Split the tuple into components
        components = list(self.amb_man.split_params(p))

        # Update the component at the specified index
        components[self.cmp_idx] = components[self.cmp_idx] + q

        # Join the components back into a tuple
        return self.amb_man.join_params(*components)
