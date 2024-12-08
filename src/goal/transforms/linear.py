"""Linear and affine transformations between manifolds."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Generic, Self, TypeVar

from jax import Array

from ..manifold import Coordinates, Dual, Manifold, Pair, Point, Triple
from .matrix import MatrixRep, Square

### Linear Maps ###

Rep = TypeVar("Rep", bound="MatrixRep", covariant=True)
Domain = TypeVar("Domain", bound="Manifold", contravariant=True)
Codomain = TypeVar("Codomain", bound="Manifold", covariant=True)


@dataclass(frozen=True)
class LinearMap(Generic[Rep, Domain, Codomain], Manifold):
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

    def transpose_manifold(self) -> LinearMap[Rep, Codomain, Domain]:
        """Transpose the linear map."""
        return LinearMap(self.rep, self.cod_man, self.dom_man)

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

    def transpose_apply[C: Coordinates](
        self: LinearMap[Rep, Domain, Codomain],
        f: Point[C, LinearMap[Rep, Domain, Codomain]],
        p: Point[Dual[C], Codomain],
    ) -> Point[C, Domain]:
        """Apply the transpose of the linear map."""
        trn_man = self.transpose_manifold()
        f_trn = self.transpose(f)
        return trn_man(f_trn, p)

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

Super = TypeVar("Super", bound="Manifold", contravariant=True)
Sub = TypeVar("Sub", bound="Manifold", contravariant=True)


@dataclass(frozen=True)
class Subspace(Generic[Super, Sub], ABC):
    """Defines how a super-manifold $\\mathcal M$ can be projected onto a sub-manifold $\\mathcal N$.

    A subspace relationship between manifolds allows us to:
    1. Project a point in the super manifold $\\mathcal M$ to its $\\mathcal N$-manifold components
    2. Partially translate a point in $\\mathcal M$ by a point in $\\mathcal N$
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
class IdentitySubspace[M: Manifold](Subspace[M, M]):
    """Identity subspace relationship for a manifold M."""

    def __init__(self, man: M):
        super().__init__(man, man)

    def project[C: Coordinates](self, p: Point[C, M]) -> Point[C, M]:
        return p

    def translate[C: Coordinates](self, p: Point[C, M], q: Point[C, M]) -> Point[C, M]:
        return p + q


@dataclass(frozen=True)
class PairSubspace[First: Manifold, Second: Manifold](
    Subspace[Pair[First, Second], First]
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


@dataclass(frozen=True)
class TripleSubspace[First: Manifold, Second: Manifold, Third: Manifold](
    Subspace[Triple[First, Second, Third], First]
):
    """Subspace relationship for a product manifold $M \\times N$."""

    def __init__(self, fst_man: First, snd_man: Second, trd_man: Third):
        super().__init__(Triple(fst_man, snd_man, trd_man), fst_man)

    def project[C: Coordinates](
        self, p: Point[C, Triple[First, Second, Third]]
    ) -> Point[C, First]:
        first, _, _ = self.sup_man.split_params(p)
        return first

    def translate[C: Coordinates](
        self, p: Point[C, Triple[First, Second, Third]], q: Point[C, First]
    ) -> Point[C, Triple[First, Second, Third]]:
        first, second, third = self.sup_man.split_params(p)
        return self.sup_man.join_params(first + q, second, third)


### Affine Maps ###


@dataclass(frozen=True)
class AffineMap[
    Rep: MatrixRep,
    Domain: Manifold,
    Codomain: Manifold,
    SubCodomain: Manifold,
](Pair[Codomain, LinearMap[Rep, Domain, SubCodomain]]):
    """Affine transformation targeting a subspace of the codomain.

    Args:
        rep: Matrix representation strategy
        domain: Source manifold
        cod_sub: Target manifold and subspace

    """

    cod_sub: Subspace[Codomain, SubCodomain]

    def __init__(
        self, rep: Rep, domain: Domain, cod_sub: Subspace[Codomain, SubCodomain]
    ):
        super().__init__(cod_sub.sup_man, LinearMap(rep, domain, cod_sub.sub_man))
        object.__setattr__(self, "cod_sub", cod_sub)

    def __call__[C: Coordinates](
        self,
        f: Point[C, Self],
        p: Point[Dual[C], Domain],
    ) -> Point[C, Codomain]:
        """Apply the affine transformation."""
        bias: Point[C, Codomain]
        bias, linear = self.split_params(f)
        subshift: Point[C, SubCodomain] = self.snd_man(linear, p)
        return self.cod_sub.translate(bias, subshift)
