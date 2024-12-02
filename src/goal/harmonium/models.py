"""Core definitions for harmonium models - product exponential families combining observable and latent variables."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Self

import jax.numpy as jnp
from jax import Array

from ..exponential_family import (
    ClosedForm,
    Mean,
    Natural,
)
from ..exponential_family.distributions import Categorical
from ..manifold import Point
from ..transforms import LinearMap, Rectangular
from .core import ClosedFormConjugated


@dataclass(frozen=True)
class Mixture[O: ClosedForm](ClosedFormConjugated[Rectangular, O, Categorical]):
    """A mixture model implemented as a harmonium with categorical latent variables.

    The joint distribution takes the form:

    $$p(x,z) \\propto e^{\\theta_x \\cdot s_x(x)} e^{\\theta_z \\cdot s_z} e^{s_x(x) \\cdot \\Theta_{xz} \\cdot s_z}$$

    where $s_z$ is the one-hot encoding of the categorical variable z.

    This can be rewritten as:

    $$p(x,z=k) \\propto e^{\\theta_x \\cdot s_x(x)} e^{\\theta_k} e^{s_x(x) \\cdot \\theta_{k}}$$

    where $\\theta_k$ and $\\theta_{k}$ are the k-th components of $\\theta_z$ and $\\Theta_{xz}$ respectively.
    """

    def conjugation_parameters(
        self,
        obs_bias: Point[Natural, O],
        int_mat: Point[Natural, LinearMap[Rectangular, Categorical, O]],
    ) -> tuple[Array, Point[Natural, Categorical]]:
        """Compute conjugation parameters for categorical mixture.

        For a categorical mixture model:

        - $\\rho_0 = \\psi(\\theta_x)$
        - $\\rho_z = \\{\\psi(\\theta_x + \\theta_{xz,k}) - \\rho_0\\}_k$
        """
        # Compute base term from observable bias
        rho_0 = self.obs_man.log_partition_function(obs_bias)

        # Get component parameter vectors as columns of interaction matrix
        int_cols = self.inter_man.to_columns(int_mat)

        # Compute adjusted partition function for each component
        rho_z_components = []
        for comp_params in int_cols:
            # For each component k, compute psi(theta_x + theta_xz_k)
            adjusted_obs = obs_bias + comp_params
            comp_term = self.obs_man.log_partition_function(adjusted_obs) - rho_0
            rho_z_components.append(comp_term)

        return rho_0, Point(jnp.array(rho_z_components))

    def to_natural_likelihood(
        self, p: Point[Mean, Self]
    ) -> tuple[
        Point[Natural, O], Point[Natural, LinearMap[Rectangular, Categorical, O]]
    ]:
        """Map mean harmonium parameters to natural likelihood parameters.

        For categorical mixtures, we:
        1. Get categorical probabilities from mean latent parameters
        2. Use these to weight the mean parameters to recover component means
        3. Convert each component mean to natural parameters
        4. Extract base and interaction terms
        """
        # Split mean parameters
        mean_obs, mean_lat, mean_int = self.split_params(p)

        # Get weights from categorical mean parameters
        weights = self.lat_man.to_probs(mean_lat)

        # Get component parameter vectors from interaction matrix
        int_cols = self.inter_man.to_columns(mean_int)

        # Reconstruct component means:
        # First component is base mean
        # Other components are base + interaction columns
        comp_means = [mean_obs]  # First component
        for col in int_cols:
            comp_mean = mean_obs + col
            comp_means.append(comp_mean)

        # Scale by weights to get original means
        orig_comp_means = [comp / w for comp, w in zip(comp_means, weights)]

        # Convert to natural parameters
        nat_comps = [self.obs_man.to_natural(comp) for comp in orig_comp_means]

        # First component gives observable bias
        nat_obs = nat_comps[0]

        # Differences from first component give interaction terms
        nat_int_cols = [comp - nat_obs for comp in nat_comps[1:]]

        # Reconstruct interaction matrix
        nat_int = self.inter_man.from_columns(nat_int_cols)

        return nat_obs, nat_int
