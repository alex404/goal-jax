"""Defines subspace relationships between manifolds.

This module provides the fundamental embedding operations that define how manifolds relate to each other through subspace relationships.

In practice, embeddings allow you to transform parameters between different model spaces, project complex models to simpler ones, and define how changes in one manifold affect another.

In theory, embeddings formalize the essential concept from information geometry where one statistical model (manifold) can be viewed as a submanifold of another. This corresponds to statistical concepts like sufficiency and model constraints.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import override

import jax
from jax import Array

from .base import Manifold
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
    def embed(self, coords: Array) -> Array:
        """Embed a point from the submanifold in the ambient manifold.

        Args:
            coords: Point on the submanifold

        Returns:
            Embedded point on the ambient manifold
        """

    def pullback(self, at_point: Array, cotangent_vector: Array) -> Array:
        """Pull back a cotangent vector from the ambient manifold to the submanifold.

        In practice, this transforms gradients computed in the larger model space to the constrained model space,
        enabling optimization in the lower-dimensional space. It's essential for efficiently computing natural
        gradients on submodels and implementing constrained optimization.

        In theory, given the embedding $\\phi: \\mathcal M \\to \\mathcal N$. At $w = \\phi(v)$, the pullback is the mapping
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
    """A structure-preserving embedding with projection and translation operations.

    In practice, linear embeddings enable direct conversion of coordinates between model spaces through projection, while preserving vector operations like translation.

    A linear embedding $\\phi: \\mathcal{M} \\to \\mathcal{N}$ supports

        - A projection map $\\pi: \\mathcal{N} \\to \\mathcal{M}$ satisfying $\\pi \\circ \\phi = \\text{id}_{\\mathcal{M}}$ (the identity on $\\mathcal{M}$), and
        - A translation operation $\\tau(p,q) = p + \\phi(q)$ that preserves the vector space structure.
    """

    # Contract

    @abstractmethod
    def project(self, coords: Array) -> Array:
        """Project a point from the ambient space to the subspace.

        Args:
            coords: Point on the ambient manifold

        Returns:
            Projected point on the submanifold
        """

    def translate(self, p_coords: Array, q_coords: Array) -> Array:
        """Translate point by subspace components.

        Args:
            p_coords: Point on the ambient manifold
            q_coords: Point on the submanifold

        Returns:
            Translated point on the ambient manifold
        """
        return p_coords + self.embed(q_coords)

    # Overrides

    @override
    def pullback(self, at_point: Array, cotangent_vector: Array) -> Array:
        """For linear subspaces, pullback is location-independent.

        Args:
            at_point: Point on the submanifold
            cotangent_vector: Cotangent vector on the ambient manifold

        Returns:
            Pulled back cotangent vector on the submanifold
        """
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
    def project(self, coords: Array) -> Array:
        """Project a point (identity operation).

        Args:
            coords: Point on the manifold

        Returns:
            Same point
        """
        return coords

    @override
    def embed(self, coords: Array) -> Array:
        """Embed a point (identity operation).

        Args:
            coords: Point on the manifold

        Returns:
            Same point
        """
        return coords

    @override
    def translate(self, p_coords: Array, q_coords: Array) -> Array:
        """Translate point by adding (identity operation).

        Args:
            p_coords: First point
            q_coords: Second point

        Returns:
            Sum of the two points
        """
        return p_coords + q_coords


@dataclass(frozen=True)
class ComposedEmbedding[Sub: Manifold, Mid: Manifold, Ambient: Manifold](
    Embedding[Sub, Ambient], ABC
):
    """Composed embeddings allow you to chain multiple embeddings.

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
    def embed(self, coords: Array) -> Array:
        """Embed through composed embeddings.

        Args:
            coords: Point on the submanifold

        Returns:
            Embedded point on the ambient manifold
        """
        mid = self.sub_emb.embed(coords)
        return self.mid_emb.embed(mid)

    @override
    def pullback(self, at_point: Array, cotangent_vector: Array) -> Array:
        """Pull back through composed embeddings.

        Args:
            at_point: Point on the submanifold
            cotangent_vector: Cotangent vector on the ambient manifold

        Returns:
            Pulled back cotangent vector on the submanifold
        """
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
    def project(self, coords: Array) -> Array:
        mid = self.mid_emb.project(coords)
        return self.sub_emb.project(mid)

    @override
    def translate(self, p_coords: Array, q_coords: Array) -> Array:
        mid_zero = self.mid_man.zeros()
        mid = self.sub_emb.translate(mid_zero, q_coords)
        return self.mid_emb.translate(p_coords, mid)


@dataclass(frozen=True)
class TupleEmbedding[Component: Manifold, TupleMan: Tuple](
    LinearEmbedding[Component, TupleMan], ABC
):
    """Embedding that projects to or embeds from a specific component of a tuple manifold.

    This embedding provides a way to work with individual components of tuple manifolds like Pair and Triple, enabling operations such as projection, embedding, and translation on specific components.
    """

    # Contract

    @property
    @abstractmethod
    def tup_idx(self) -> int:
        """The index of the component in the tuple manifold."""

    def __post_init__(self):
        """Validate that the component manifold matches the expected component at the given index."""
        # Get all components
        zero_tuple = self.amb_man.zeros()
        components = self.amb_man.split_coords(zero_tuple)

        # Check if index is valid
        if self.tup_idx >= len(components):
            raise IndexError(
                f"Component index {self.tup_idx} out of range for {type(self.amb_man)}"
            )

    # Overrides

    @property
    @override
    def amb_man(self) -> TupleMan:
        """The tuple manifold (super-manifold)."""
        return self.amb_man

    @property
    @override
    def sub_man(self) -> Component:
        """The component manifold (sub-manifold)."""
        return self.sub_man

    @override
    def project(self, coords: Array) -> Array:
        """Project a point from the tuple manifold to the specified component.

        Args:
            p_coords: Point on the tuple manifold

        Returns:
            Component at the specified index
        """
        # Split the tuple and get the component at the specified index
        components = self.amb_man.split_coords(coords)
        return components[self.tup_idx]

    @override
    def embed(self, coords: Array) -> Array:
        """Creates a new point in the tuple manifold with the specified component set to p and all other components set to zero.

        Args:
            p_coords: Point on the component manifold

        Returns:
            Point on the tuple manifold
        """
        # Split a zero tuple to get zero components
        zero_tuple = self.amb_man.zeros()
        components = list(self.amb_man.split_coords(zero_tuple))

        # Replace the component at the specified index
        components[self.tup_idx] = coords

        # Join the components back into a tuple
        return self.amb_man.join_coords(*components)

    @override
    def translate(self, p_coords: Array, q_coords: Array) -> Array:
        """Updates the component at the specified index by adding q, leaving other components unchanged.

        Args:
            p_coords: Point on the tuple manifold
            q_coords: Point on the component manifold

        Returns:
            Updated point on the tuple manifold
        """
        # Split the tuple into components
        components = list(self.amb_man.split_coords(p_coords))

        # Update the component at the specified index
        components[self.tup_idx] = components[self.tup_idx] + q_coords

        # Join the components back into a tuple
        return self.amb_man.join_coords(*components)
