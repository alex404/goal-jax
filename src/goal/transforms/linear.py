"""Linear and affine transformations between manifolds."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Self

from jax import Array

from ..manifold import Coordinates, Dual, Manifold, Pair, Point
from .matrix import MatrixRep, Square

### Linear Maps ###


@dataclass(frozen=True)
class LinearMap[Rep: MatrixRep, Domain: Manifold, Codomain: Manifold](Manifold):
    """Linear map between manifolds using a specific matrix representation.

    A linear map $L: V \\to W$ between vector spaces satisfies:

    $$L(\\alpha x + \\beta y) = \\alpha L(x) + \\beta L(y)$$

    The map is stored in a specific representation (full, symmetric, etc) defined by R.

    Args:
        rep: The matrix representation strategy
        dom_man: The source/domain manifold $V$
        cod_man: The target/codomain manifold $W$
    """

    rep: Rep
    dom_man: Domain
    cod_man: Codomain

    @property
    def dim(self) -> int:
        return self.rep.num_params(self.shape)

    @property
    def shape(self) -> tuple[int, int]:
        """Shape of the linear maps."""
        return (self.cod_man.dim, self.dom_man.dim)

    def __call__[C: Coordinates](
        self,
        f: Point[C, Self],
        p: Point[Dual[C], Domain],
    ) -> Point[C, Codomain]:
        """Apply the linear map to transform a point."""
        return Point(self.rep.matvec(self.shape, f.params, p.params))

    def from_dense(self, matrix: Array) -> Point[Any, Self]:
        """Create point from dense matrix."""
        return Point(self.rep.from_dense(matrix))

    def outer_product[C: Coordinates](
        self, w: Point[C, Codomain], v: Point[C, Domain]
    ) -> Point[C, Self]:
        """Outer product of points."""
        return Point(self.rep.outer_product(w.params, v.params))

    def transpose[C: Coordinates](
        self: LinearMap[Rep, Domain, Codomain],
        f: Point[C, LinearMap[Rep, Domain, Codomain]],
    ) -> Point[C, LinearMap[Rep, Codomain, Domain]]:
        """Transpose of the linear map."""
        return Point(self.rep.transpose(self.shape, f.params))

    def to_dense[C: Coordinates](self, f: Point[C, Self]) -> Array:
        """Convert to dense matrix representation."""
        return self.rep.to_dense(self.shape, f.params)

    def to_columns[C: Coordinates](self, f: Point[C, Self]) -> list[Point[C, Codomain]]:
        """Split linear map into list of column vectors as points in the codomain."""
        cols = self.rep.to_cols(self.shape, f.params)
        return [Point(col) for col in cols]

    def from_columns[C: Coordinates](
        self: LinearMap[Rep, Domain, Codomain], cols: list[Point[C, Codomain]]
    ) -> Point[C, LinearMap[Rep, Domain, Codomain]]:
        """Construct linear map from list of column vectors as points."""
        col_arrays = [col.params for col in cols]
        params = self.rep.from_cols(col_arrays)
        return Point(params)


@dataclass(frozen=True)
class SquareMap[R: Square, M: Manifold](LinearMap[R, M, M]):
    """Square linear map with domain and codomain the same manifold.

    Args:
        rep: Matrix representation strategy
        domain: Source and target manifold

    """

    def __init__(self, rep: R, domain: M):
        super().__init__(rep, domain, domain)

    def inverse[C: Coordinates](self, f: Point[C, Self]) -> Point[Dual[C], Self]:
        """Matrix inverse (requires square matrix)."""
        return Point(self.rep.inverse(self.shape, f.params))

    def logdet[C: Coordinates](self, f: Point[C, Self]) -> Array:
        """Log determinant (requires square matrix)."""
        return self.rep.logdet(self.shape, f.params)


### Linear Subspaces ###


@dataclass(frozen=True)
class LinearSubspace[Super: Manifold, Sub: Manifold](ABC):
    """Defines how a sub-manifold $\\mathcal N$ embeds into a super-manifold $\\mathcal M$.

    A linear subspace relationship between manifolds allows us to:
    1. Project a point in the super manifold N to its M-manifold components
    2. Translate a point in N by a point in M
    """

    sup_man: Super
    sub_man: Sub

    @abstractmethod
    def project[C: Coordinates](self, p: Point[C, Super]) -> Point[C, Sub]:
        """Project point to subspace components."""
        ...

    @abstractmethod
    def translate[C: Coordinates](
        self, p: Point[C, Super], q: Point[C, Sub]
    ) -> Point[C, Super]:
        """Translate point by subspace components."""
        ...


@dataclass(frozen=True)
class IdentitySubspace[M: Manifold](LinearSubspace[M, M]):
    """Identity subspace relationship for a manifold M."""

    def __init__(self, man: M):
        super().__init__(man, man)

    def project[C: Coordinates](self, p: Point[C, M]) -> Point[C, M]:
        return p

    def translate[C: Coordinates](self, p: Point[C, M], q: Point[C, M]) -> Point[C, M]:
        return p + q


@dataclass(frozen=True)
class PairSubspace[First: Manifold, Second: Manifold](
    LinearSubspace[Pair[First, Second], First]
):
    """Subspace relationship for a product manifold $M \\times N$."""

    def __init__(self, fst_man: First, snd_man: Second):
        super().__init__(Pair(fst_man, snd_man), fst_man)

    def project[C: Coordinates](
        self, p: Point[C, Pair[First, Second]]
    ) -> Point[C, First]:
        first, _ = self.sup_man.split_params(p)
        return first

    def translate[C: Coordinates](
        self, p: Point[C, Pair[First, Second]], q: Point[C, First]
    ) -> Point[C, Pair[First, Second]]:
        first, second = self.sup_man.split_params(p)
        return self.sup_man.join_params(first + q, second)


### Affine Maps ###


@dataclass(frozen=True)
class AffineSubMap[
    Rep: MatrixRep,
    Domain: Manifold,
    Codomain: Manifold,
    Subcodomain: Manifold,
](Pair[Codomain, LinearMap[Rep, Domain, Subcodomain]]):
    """Affine transformation targeting a subspace of the codomain."""

    subspace: LinearSubspace[Codomain, Subcodomain]

    def __call__[C: Coordinates](
        self,
        f: Point[C, Self],
        p: Point[Dual[C], Domain],
    ) -> Point[C, Codomain]:
        """Apply the affine transformation."""
        bias: Point[C, Codomain]
        bias, linear = self.split_params(f)
        subshift: Point[C, Subcodomain] = self.snd_man(linear, p)
        return self.subspace.translate(bias, subshift)


@dataclass(frozen=True)
class AffineMap[Rep: MatrixRep, Domain: Manifold, Codomain: Manifold](
    AffineSubMap[Rep, Domain, Codomain, Codomain]
):
    """Affine transformation $f(x) = A \\cdot x + b$.

    Args:
        rep: Matrix representation strategy
        domain: Source manifold
        codomain: Target manifold
    """

    def __init__(self, rep: Rep, domain: Domain, codomain: Codomain):
        super().__init__(
            codomain, LinearMap(rep, domain, codomain), IdentitySubspace(codomain)
        )
