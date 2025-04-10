"""Hierarchical harmoniums enable modeling complex, multi-level dependencies by stacking harmonium models. This stacked structure supports deep representations where information flows between multiple levels of latent variables, while maintaining analytical tractability through conjugate relationships.

In theory, a hierarchical harmonium is a conjugated harmonium where the latent manifold is itself a conjugated harmonium. This creates a structured joint distribution:

- A lower harmonium provides likelihood $p(x|y)$ between observable $x$ and first latent $y$
- An upper harmonium provides likelihood $p(y|z)$ between first latent $y$ and second latent $z$
- Together they form a joint distribution $p(x,y,z) = p(x|y)p(y|z)p(z)$

Key algorithms (conjugation, natural parameters, sampling) are implemented recursively through the model hierarchy, enabling efficient inference and learning despite the model's complexity.

Note: Implementation of these models pushes the boundaries of Python's type system, requiring careful handling of generic types to maintain proper relationships between layers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Self, override

import jax
import jax.numpy as jnp
from jax import Array

from ..manifold.base import Point
from ..manifold.embedding import (
    LinearComposedEmbedding,
    LinearEmbedding,
    TupleEmbedding,
)
from ..manifold.linear import AffineMap
from ..manifold.matrix import MatrixRep
from .base import (
    Analytic,
    Differentiable,
    ExponentialFamily,
    Mean,
    Natural,
)
from .harmonium import (
    AnalyticConjugated,
    DifferentiableConjugated,
    Harmonium,
    SymmetricConjugated,
)

### Helper Classes ###


@dataclass(frozen=True)
class ObservableEmbedding[
    Rep: MatrixRep,
    Observable: ExponentialFamily,
    IntObservable: ExponentialFamily,
    IntLatent: ExponentialFamily,
    PostLatent: ExponentialFamily,
    Latent: ExponentialFamily,
](
    TupleEmbedding[
        Observable,
        Harmonium[Rep, Observable, IntObservable, IntLatent, PostLatent, Latent],
    ]
):
    """Embedding of the observable manifold of a harmonium in to the harmonium itself."""

    # Fields

    hrm_man: Harmonium[Rep, Observable, IntObservable, IntLatent, PostLatent, Latent]
    """The harmonium that contains the observable manifold."""

    # Overrides

    @property
    @override
    def tup_idx(
        self,
    ) -> int:
        return 0

    @property
    @override
    def amb_man(
        self,
    ) -> Harmonium[Rep, Observable, IntObservable, IntLatent, PostLatent, Latent]:
        return self.hrm_man

    @property
    @override
    def sub_man(self) -> Observable:
        return self.hrm_man.obs_man


@dataclass(frozen=True)
class LatentHarmoniumEmbedding[
    Rep: MatrixRep,
    Observable: ExponentialFamily,
    PostObservable: ExponentialFamily,
    IntObservable: ExponentialFamily,
    IntLatent: ExponentialFamily,
    PostLatent: ExponentialFamily,
    Latent: ExponentialFamily,
](
    LinearEmbedding[
        Harmonium[Rep, PostObservable, IntObservable, IntLatent, PostLatent, Latent],
        Harmonium[Rep, Observable, IntObservable, IntLatent, PostLatent, Latent],
    ]
):
    """Embedding of one harmonium into another, where the observable space of the former can be embedded into the latter. This is designed for hierarchical harmoniums with constrained posterior spaces."""

    # Fields

    pst_hrm: Harmonium[
        Rep, PostObservable, IntObservable, IntLatent, PostLatent, Latent
    ]
    """The harmonium that contains the contrained observable manifold."""

    hrm: Harmonium[Rep, Observable, IntObservable, IntLatent, PostLatent, Latent]
    """The harmonium that contains the unconstrained observable manifold."""

    pst_obs_emb: LinearEmbedding[PostObservable, Observable]
    """The embedding of the constrained observable manifold into the unconstrained observable manifold."""

    # Overrides

    @property
    @override
    def amb_man(
        self,
    ) -> Harmonium[Rep, Observable, IntObservable, IntLatent, PostLatent, Latent]:
        return self.hrm

    @property
    @override
    def sub_man(
        self,
    ) -> Harmonium[Rep, PostObservable, IntObservable, IntLatent, PostLatent, Latent]:
        return self.pst_hrm

    @override
    def project(
        self,
        p: Point[
            Mean,
            Harmonium[Rep, Observable, IntObservable, IntLatent, PostLatent, Latent],
        ],
    ) -> Point[
        Mean,
        Harmonium[Rep, PostObservable, IntObservable, IntLatent, PostLatent, Latent],
    ]:
        obs_params, int_params, lat_params = self.amb_man.split_params(p)
        prj_obs_params = self.pst_obs_emb.project(obs_params)
        return self.sub_man.join_params(prj_obs_params, int_params, lat_params)

    @override
    def embed(
        self,
        p: Point[
            Natural,
            Harmonium[
                Rep, PostObservable, IntObservable, IntLatent, PostLatent, Latent
            ],
        ],
    ) -> Point[
        Natural,
        Harmonium[Rep, Observable, IntObservable, IntLatent, PostLatent, Latent],
    ]:
        obs_params, int_params, lat_params = self.sub_man.split_params(p)
        emb_obs_params = self.pst_obs_emb.embed(obs_params)
        return self.amb_man.join_params(emb_obs_params, int_params, lat_params)


### Undirected Harmoniums ###


@dataclass(frozen=True)
class StrongDifferentiableUndirected[
    IntRep: MatrixRep,
    Observable: Differentiable,
    IntObservable: ExponentialFamily,
    IntLatent: ExponentialFamily,
    PostLatent: Differentiable,
    Latent: Differentiable,
    LowerHarmonium: Differentiable,
    PostUpperHarmonium: Differentiable,
    UpperHarmonium: Differentiable,
](
    DifferentiableConjugated[
        IntRep, Observable, IntObservable, IntLatent, PostUpperHarmonium, UpperHarmonium
    ],
):
    """Class for hierarchical harmoniums with deep conjugate structure.

    This class provides the algorithms needed for hierarchical harmoniums while ensuring proper typing and conjugate relationships between layers. It subclasses `DifferentiableConjugated` to maintain the exponential family structure while adding hierarchical capabilities.

    Notes
    -----
    This class pushes past the boundary of the python type system, and requires a bit of fiddling to behave correctly. If python would support higher kinded types, its type parameters would rather be

        [ IntRep: MatrixRep,
        Observable: Generative,
        IntObservable: ExponentialFamily,
        IntLatent: ExponentialFamily,
        PostLatent: Differentiable,
        Latent: Differentiable,
        LowerHarmonium: DifferentiableConjugated,
        PostUpperHarmonium: Differentiable,
        UpperHarmonium: Differentiable ]

    and `lwr_hrm` would have type

        LowerHarmonium[IntRep, Observable, IntObservable, IntLatent, PostLatent, Latent]

    instead of `LowerHarmonium`.
    """

    # Fields

    lwr_hrm: LowerHarmonium
    """The lower harmonium in the hierarchy."""

    pst_upr_hrm: PostUpperHarmonium
    """The posterior harmonium in the hierarchy. This is a harmonium with a constrained observable space that is embedded into the upper harmonium."""

    upr_hrm: UpperHarmonium
    """The upper harmonium in the hierarchy. This is a harmonium with an unconstrained observable space that contains the posterior harmonium."""

    # Private Properties

    @property
    def _abc_lwr_hrm(
        self,
    ) -> DifferentiableConjugated[
        IntRep, Observable, IntObservable, IntLatent, PostLatent, Latent
    ]:
        """Accessor for the lower harmonium viewed as an instance of `DifferentiableConjugated`."""
        return self.lwr_hrm  # pyright: ignore[reportReturnType]

    @property
    def _hrm_lat_emb(self) -> LinearEmbedding[Latent, UpperHarmonium]:
        """The embedding of the shared latent manifold into the upper harmonium."""
        return ObservableEmbedding(self.upr_hrm)  # pyright: ignore[reportReturnType, reportUnknownVariableType, reportArgumentType]

    # Overrides

    @property
    @override
    def int_rep(self) -> IntRep:
        return self._abc_lwr_hrm.snd_man.rep

    @property
    @override
    def obs_emb(self) -> LinearEmbedding[IntObservable, Observable]:
        return self._abc_lwr_hrm.obs_emb

    @property
    @override
    def int_lat_emb(self) -> LinearEmbedding[IntLatent, UpperHarmonium]:
        return LinearComposedEmbedding(  # pyright: ignore[reportReturnType, reportUnknownVariableType]
            ObservableEmbedding(self.pst_upr_hrm),  # pyright: ignore[reportArgumentType]
            self._abc_lwr_hrm.int_lat_emb,
        )

    @property
    @override
    def pst_lat_emb(self) -> LinearEmbedding[PostUpperHarmonium, UpperHarmonium]:
        return LatentHarmoniumEmbedding(  # pyright: ignore[reportReturnType, reportUnknownVariableType]
            self.upr_hrm,  # pyright: ignore[reportArgumentType]
            self.pst_upr_hrm,  # pyright: ignore[reportArgumentType]
            self._abc_lwr_hrm.pst_lat_emb,
        )

    @override
    def sample(self, key: Array, params: Point[Natural, Self], n: int = 1) -> Array:
        # Split up the sampling key
        key1, key2 = jax.random.split(key)

        # Sample from adjusted latent distribution p(z)
        nat_prior = self.prior(params)
        yz_sample = self.lat_man.sample(key1, nat_prior, n)
        y_sample = yz_sample[:, : self._abc_lwr_hrm.lat_man.data_dim]

        # Vectorize sampling from conditional distributions
        x_params = jax.vmap(self.likelihood_at, in_axes=(None, 0))(params, y_sample)

        # Sample from conditionals p(x|z) in parallel
        x_sample = jax.vmap(self.obs_man.sample, in_axes=(0, 0, None))(
            jax.random.split(key2, n), x_params, 1
        ).reshape((n, -1))

        # Concatenate samples along data dimension
        return jnp.concatenate([x_sample, yz_sample], axis=-1)

    @override
    def conjugation_parameters(
        self,
        lkl_params: Point[
            Natural, AffineMap[IntRep, IntLatent, IntObservable, Observable]
        ],
    ) -> Point[Natural, UpperHarmonium]:
        return self._hrm_lat_emb.embed(
            self._abc_lwr_hrm.conjugation_parameters(lkl_params)
        )


class StrongSymmetricUndirected[
    IntRep: MatrixRep,
    Observable: Differentiable,
    IntObservable: ExponentialFamily,
    IntLatent: ExponentialFamily,
    Latent: Differentiable,
    LowerHarmonium: Differentiable,
    UpperHarmonium: Differentiable,
](
    SymmetricConjugated[IntRep, Observable, IntObservable, IntLatent, UpperHarmonium],
    StrongDifferentiableUndirected[
        IntRep,
        Observable,
        IntObservable,
        IntLatent,
        Latent,
        Latent,
        LowerHarmonium,
        UpperHarmonium,
        UpperHarmonium,
    ],
):
    """This class simplifies hierarchical models where the posterior latent space is not restricted to a submanifold of the full latent space. This enables more direct parameter transformations and inference procedures by eliminating the need to handle embedding constraints."""

    def __init__(
        self,
        lwr_hrm: LowerHarmonium,
        upr_hrm: UpperHarmonium,
    ):
        """Initialize hierarchical harmonium with conjugate structure."""
        super().__init__(lwr_hrm, upr_hrm, upr_hrm)

    # Overrides

    @property
    @override
    def _abc_lwr_hrm(
        self,
    ) -> SymmetricConjugated[IntRep, Observable, IntObservable, IntLatent, Latent]:
        """Accessor for the lower harmonium viewed as an instance of `SymmetricConjugated`."""
        return self.lwr_hrm  # pyright: ignore[reportReturnType]


class StrongAnalyticUndirected[
    IntRep: MatrixRep,
    Observable: Differentiable,
    IntObservable: ExponentialFamily,
    IntLatent: ExponentialFamily,
    Latent: Analytic,
    LowerHarmonium: Analytic,
    UpperHarmonium: Analytic,
](
    AnalyticConjugated[IntRep, Observable, IntObservable, IntLatent, UpperHarmonium],
    StrongSymmetricUndirected[
        IntRep,
        Observable,
        IntObservable,
        IntLatent,
        Latent,
        LowerHarmonium,
        UpperHarmonium,
    ],
):
    """This class enables closed-form parameter conversion, entropy calculation, and efficient implementation of the EM algorithm for hierarchical models. It represents the most analytically tractable form of hierarchical harmoniums."""

    # Overrides

    @property
    @override
    def _abc_lwr_hrm(
        self,
    ) -> AnalyticConjugated[IntRep, Observable, IntObservable, IntLatent, Latent]:
        """Accessor for the lower harmonium viewed as an instance of `AnalyticConjugated`."""
        return self.lwr_hrm  # pyright: ignore[reportReturnType]

    @override
    def to_natural_likelihood(
        self,
        params: Point[Mean, Self],
    ) -> Point[Natural, AffineMap[IntRep, IntLatent, IntObservable, Observable]]:
        """Convert mean parameters to natural parameters for lower likelihood."""
        obs_means, lwr_int_means, lat_means = self.split_params(params)
        lwr_lat_means = self._hrm_lat_emb.project(lat_means)
        lwr_means = self._abc_lwr_hrm.join_params(
            obs_means, lwr_int_means, lwr_lat_means
        )
        return self._abc_lwr_hrm.to_natural_likelihood(lwr_means)


### Front End Classes ###


@dataclass(frozen=True)
class DifferentiableUndirected[
    LowerHarmonium: Differentiable,
    PostUpperHarmonium: Differentiable,
    UpperHarmonium: Differentiable,
](
    StrongDifferentiableUndirected[
        Any, Any, Any, Any, Any, Any, LowerHarmonium, PostUpperHarmonium, UpperHarmonium
    ]
):
    """This front-end class provides a simplified interface for creating hierarchical models, with runtime checks to ensure the provided components form a valid hierarchical structure. It handles the technical implementation details while exposing a clean interface for model building.

    In theory, this wraps the strong implementation with appropriate runtime validation to ensure:
    - `LowerHarmonium`, `PostUpperHarmonium`, and `UpperHarmonium` must be subtypes of `DifferentiableConjugated`
    - The latent manifold of the lower harmonium must match the observable manifold of the upper harmonium and posterior harmoniums.
    """

    # Fields

    def __post_init__(self):
        # Check that the subspaces are compatible - both should be DifferentiableConjugated, and lwr_hrm.lat_man should match upr_hrm.obs_man
        assert isinstance(self.lwr_hrm, DifferentiableConjugated)
        assert isinstance(self.pst_upr_hrm, DifferentiableConjugated)
        assert isinstance(self.upr_hrm, DifferentiableConjugated)
        assert self.lwr_hrm.lat_man == self.upr_hrm.obs_man
        assert self.lwr_hrm.pst_lat_man == self.pst_upr_hrm.obs_man


@dataclass(frozen=True)
class SymmetricUndirected[
    LowerHarmonium: Differentiable,
    UpperHarmonium: Differentiable,
](StrongSymmetricUndirected[Any, Any, Any, Any, Any, LowerHarmonium, UpperHarmonium]):
    """This front-end class provides a simplified interface for creating hierarchical models where posterior and prior spaces are the same. It incorporates runtime validation to ensure the component models satisfy the required properties."""

    # Fields

    def __init__(
        self,
        lwr_hrm: LowerHarmonium,
        upr_hrm: UpperHarmonium,
    ):
        """Initialize hierarchical harmonium with conjugate structure."""
        super().__init__(lwr_hrm, upr_hrm)

    def __post_init__(self):
        # Check that the subspaces are compatible - both should be DifferentiableConjugated, and lwr_hrm.lat_man should match upr_hrm.obs_man
        assert isinstance(self.lwr_hrm, SymmetricConjugated)
        assert isinstance(self.upr_hrm, SymmetricConjugated)
        assert self.lwr_hrm.lat_man == self.upr_hrm.obs_man


@dataclass(frozen=True)
class AnalyticUndirected[
    LowerHarmonium: Analytic,
    UpperHarmonium: Analytic,
](StrongAnalyticUndirected[Any, Any, Any, Any, Any, LowerHarmonium, UpperHarmonium]):
    """This front-end class provides a simplified interface for creating hierarchical models
    where posterior and prior spaces are the same. It incorporates runtime validation to ensure the
    component models satisfy the required properties.
    """

    # Fields

    def __init__(
        self,
        lwr_hrm: LowerHarmonium,
        upr_hrm: UpperHarmonium,
    ):
        """Initialize hierarchical harmonium with conjugate structure."""
        super().__init__(lwr_hrm, upr_hrm)

    def __post_init__(self):
        # Check that the subspaces are compatible - both should be DifferentiableConjugated, and lwr_hrm.lat_man should match upr_hrm.obs_man
        assert isinstance(self.lwr_hrm, AnalyticConjugated)
        assert isinstance(self.upr_hrm, AnalyticConjugated)
        assert self.lwr_hrm.lat_man == self.upr_hrm.obs_man
