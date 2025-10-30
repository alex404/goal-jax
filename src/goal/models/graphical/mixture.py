"""Mixture model utilities for conjugated harmoniums.

This module provides helper functions for computing conjugation parameters in hierarchical
mixture models, enabling efficient inference in models where conjugated harmoniums are composed
into mixture structures.

**Key function**: `harmonium_mixture_conjugation_parameters` decomposes the joint conjugation
parameters of a mixture of conjugated harmoniums into three components:

1. **Y conjugation biases**: Base conjugation parameters from the first component
2. **K conjugation biases**: Differences in log partition between components
3. **K-Y interaction**: Changes in conjugation parameters across components

This decomposition preserves the mixture structure and enables efficient conjugation parameter
computation for hierarchical models where the upper level is a mixture over harmoniums.
"""

from __future__ import annotations

from typing import Any

from jax import Array

from ...geometry import (
    AffineMap,
    Conjugated,
    Natural,
    Point,
    Rectangular,
)
from ..base.categorical import Categorical


def harmonium_mixture_conjugation_parameters[
    Harm: Conjugated[Any, Any, Any, Any, Any, Any],
](
    mixture_likelihood: AffineMap[Rectangular, Categorical, Harm, Harm],
    lkl_params: Point[Natural, AffineMap[Rectangular, Categorical, Harm, Harm]],
) -> tuple[Point[Natural, Any], Point[Natural, Any], Point[Natural, Any]]:
    """Compute conjugation parameters for a mixture of conjugated harmoniums.

    This decomposes the joint conjugation parameters into three components:
    1. Y conjugation biases (latent variable biases from component 0)
    2. K conjugation biases (categorical variable biases from observable component differences)
    3. K-Y interaction columns (differences in conjugation parameters across components)

    The function extracts observable biases from each component harmonium and computes
    a mixture-like conjugation parameter structure encoding the joint (Y, K) latent space.

    Parameters
    ----------
    mixture_model : Mixture[Observable, SubObservable]
        The mixture of conjugated harmoniums manifold
    params : Point[Natural, Mixture[Observable, SubObservable]]
        Natural parameters on the mixture

    Returns
    -------
    Point[Natural, Any]
        Conjugation parameters for the mixture of harmoniums
    """
    # Split mixture parameters into component harmonium parameters and categorical prior
    hrm_params_0, int_mat = mixture_likelihood.split_params(lkl_params)
    int_man = mixture_likelihood.snd_man
    int_comps = int_man.to_columns(int_mat)

    hrm_man = mixture_likelihood.cod_emb.amb_man
    lkl_params_0 = hrm_man.likelihood_function(hrm_params_0)
    rho_y = hrm_man.conjugation_parameters(lkl_params_0)
    obs_params_0, _, _ = hrm_man.split_params(hrm_params_0)
    lp_0 = hrm_man.obs_man.log_partition_function(obs_params_0)

    def compute_rho_ks(comp_params: Point[Natural, Harm]) -> Array:
        obs_params_k, _, _ = hrm_man.split_params(comp_params)
        adjusted_obs = hrm_man.int_obs_emb.translate(obs_params_0, obs_params_k)
        return hrm_man.obs_man.log_partition_function(adjusted_obs) - lp_0

    def compute_rho_yks(
        comp_params: Point[Natural, Harm],
    ) -> Point[Natural, Harm]:
        adjusted_hrm = hrm_man.int_obs_emb.translate(hrm_params_0, comp_params)
        adjusted_lkl = hrm_man.likelihood_function(adjusted_hrm)
        rho_yk0 = hrm_man.conjugation_parameters(adjusted_lkl)
        return rho_yk0 - rho_y

    # rho_z shape: (n_categories - 1,)
    rho_k0 = int_man.col_man.map(compute_rho_ks, int_comps)
    rho_k = mixture_likelihood.dom_man.natural_point(rho_k0)

    rho_yk_comps = int_man.col_man.man_map(compute_rho_yks, int_comps)
    rho_yk = int_man.from_columns(rho_yk_comps)

    return (rho_y, rho_yk, rho_k)

    # # Extract observable biases from each component harmonium
    # def extract_obs_bias(
    #     hrm_params: Point[Natural, Harm],
    # ) -> Point[Natural, Differentiable]:
    #     """Extract observable bias from a component harmonium."""
    #     obs_params, _, _ = hrm_man.split_params(hrm_params)
    #     return obs_params
    #
    # # Map extraction across all components to get observable biases
    # obs_biases = mixture_model.cmp_man.man_map(extract_obs_bias, comp_hrm_params)
    #
    # # obs_0 = mixture_model.cmp_man.get_replicate(obs_biases, jnp.asarray(0))
    # # Need to extract this manually by extracting from array
    # obs_0: Point[Natural, Differentiable] = mixture_model.obs_man.obs_man.natural_point(
    #     obs_biases.array[0]
    # )
    #
    # # Step 1: Y conjugation biases from component 0
    # # The base log partition value serves as the Y bias baseline
    # rho_0 = mixture_model.obs_man.obs_man.log_partition_function(obs_0)
    #
    # # Step 2 & 3: K conjugation biases and K-Y interaction columns
    # # For each component k > 0, compute log partition difference from baseline
    # def compute_rhos(
    #     obs_params: Point[Natural, Differentiable],
    # ) -> Array:
    #     return mixture_model.obs_man.obs_man.log_partition_function(obs_params - obs_0)
    #
    # # Apply to all components and extract interactions from components 1..K-1
    # all_rhos = mixture_model.cmp_man.man_map(compute_rho_k, obs_biases)
    # # Extract rho differences for components 1..K-1 (skip component 0)
    # interactions_cols = jnp.asarray(all_rhos.array[1:]).T
    #
    # # Reconstruct output parameters on the mixture structure
    # # Y biases become the observable parameters
    # # K-Y interactions become the interaction matrix
    # # K prior becomes the categorical parameters
    # output_obs_params = mixture_model.obs_man.natural_point(rho_0)
    # output_int_params = mixture_model.int_man.point(
    #     mixture_model.int_man.rep.from_dense(interactions_cols)
    # )
    #
    # return mixture_model.join_conjugated(
    #     mixture_model.lkl_man.join_params(output_obs_params, output_int_params),
    #     cat_params,
    # )
