"""Hidden Markov Model example with gradient-based and EM learning.

Discrete analogue of the Kalman filter example: sample sequences from an HMM, run filtering and smoothing under the true parameters, fit a fresh model via gradient ascent on $\\log p(x_{1:T})$ and via expectation maximization, and compare.
"""

from typing import Any

import jax
import jax.numpy as jnp
import optax
from jax import Array

from goal.models import HiddenMarkovModel

from ..shared import example_paths, jax_cli
from .types import HMMResults

jax.config.update("jax_enable_x64", True)

N_OBS = 4
N_STATES = 3
N_STEPS = 50
N_SEQUENCES = 20
N_GRAD_STEPS = 200
N_EM_STEPS = 100


def avg_log_lik(
    model: HiddenMarkovModel, params: Array, observations_batch: Array
) -> Array:
    return jnp.mean(
        jax.vmap(lambda obs: model.log_observable_density(params, obs))(
            observations_batch
        )
    )


def fit_gradient(
    key: Array,
    model: HiddenMarkovModel,
    observations_batch: Array,
    n_steps: int,
) -> tuple[Array, Array, Array]:
    init_params = model.initialize(key, shape=0.2)

    def obj(p: Array) -> Array:
        return avg_log_lik(model, p, observations_batch)

    optimizer = optax.adam(learning_rate=2e-2)

    def grad_step(
        carry: tuple[Array, Any], _: None
    ) -> tuple[tuple[Array, Any], Array]:
        params, opt_state = carry
        ll = obj(params)
        grads = jax.grad(obj)(params)
        updates, new_opt_state = optimizer.update(
            jax.tree.map(jnp.negative, grads), opt_state
        )
        new_params = jnp.asarray(optax.apply_updates(params, updates))
        return (new_params, new_opt_state), ll

    opt_state = optimizer.init(init_params)
    (final_params, _), lls = jax.lax.scan(
        grad_step, (init_params, opt_state), None, length=n_steps
    )
    return lls.ravel(), init_params, final_params


fit_gradient = jax.jit(fit_gradient, static_argnames=["model", "n_steps"])


def fit_em(
    key: Array,
    model: HiddenMarkovModel,
    observations_batch: Array,
    n_steps: int,
) -> tuple[Array, Array, Array]:
    init_params = model.initialize(key, shape=0.2)

    def em_step(params: Array, _: Any) -> tuple[Array, Array]:
        ll = avg_log_lik(model, params, observations_batch)
        return model.expectation_maximization(params, observations_batch), ll

    final_params, lls = jax.lax.scan(em_step, init_params, None, length=n_steps)
    return lls.ravel(), init_params, final_params


fit_em = jax.jit(fit_em, static_argnames=["model", "n_steps"])


def extract_state_probs(
    model: HiddenMarkovModel, params: Array, observations: Array
) -> tuple[Array, Array]:
    """Filter and smooth, then convert per-step natural params to state probabilities."""
    filtered, _ = model.filter(params, observations)
    _, smoothed_seq, _ = model.smooth(params, observations)

    def to_probs(nat: Array) -> Array:
        return model.lat_man.to_probs(model.lat_man.to_mean(nat))

    filt_probs = jax.vmap(to_probs)(filtered)
    smth_probs = jax.vmap(to_probs)(smoothed_seq)
    return filt_probs, smth_probs


def main() -> None:
    jax_cli()
    paths = example_paths(__file__)

    key = jax.random.PRNGKey(0)
    key_model, key_sample, key_train = jax.random.split(key, 3)

    # Build a ground-truth HMM by random initialization with larger noise so
    # that the sampled trajectories are non-trivial.
    model = HiddenMarkovModel(n_obs=N_OBS, n_states=N_STATES)
    true_params = model.initialize(key_model, shape=0.8)

    # Sample training trajectories
    keys_sample = jax.random.split(key_sample, N_SEQUENCES)
    obs_batch, state_batch = jax.vmap(
        lambda k: model.sample(k, true_params, N_STEPS)
    )(keys_sample)

    observations = obs_batch[0]
    true_states = state_batch[0]

    # Oracle inference under true params
    true_filt_probs, true_smth_probs = extract_state_probs(
        model, true_params, observations
    )

    true_avg_ll = float(avg_log_lik(model, true_params, obs_batch))
    init_params = model.initialize(key_train, shape=0.2)
    initial_ll = float(avg_log_lik(model, init_params, obs_batch))

    # Training
    grad_lls, _, grad_final = fit_gradient(
        key_train, model, obs_batch, N_GRAD_STEPS
    )
    em_lls, _, em_final = fit_em(key_train, model, obs_batch, N_EM_STEPS)

    learned: dict[str, tuple[Array, Array]] = {
        "Gradient": extract_state_probs(model, grad_final, observations),
        "EM": extract_state_probs(model, em_final, observations),
    }

    # Convert observations and states from (T, 1) integer arrays to flat lists.
    obs_int = observations.reshape(-1).astype(jnp.int32).tolist()
    state_int = true_states.reshape(-1).astype(jnp.int32).tolist()

    results = HMMResults(
        n_obs=N_OBS,
        n_states=N_STATES,
        n_steps=N_STEPS,
        n_sequences=N_SEQUENCES,
        observations=obs_int,
        true_states=state_int,
        filtered_probs=true_filt_probs.tolist(),
        smoothed_probs=true_smth_probs.tolist(),
        log_likelihoods={
            "Gradient": grad_lls.tolist(),
            "EM": em_lls.tolist(),
        },
        learned_filtered_probs={k: v[0].tolist() for k, v in learned.items()},
        learned_smoothed_probs={k: v[1].tolist() for k, v in learned.items()},
        true_log_lik=true_avg_ll,
        initial_log_lik=initial_ll,
    )
    paths.save_analysis(results)


if __name__ == "__main__":
    main()
