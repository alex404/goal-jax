from typing import TypedDict


class GaussianMarkovChainResults(TypedDict):
    lat_dim: int
    n_steps: int
    n_sequences: int
    true_log_likelihood: float
    learned_log_likelihood: float
    param_error: float
    true_trajectories: list[list[float]]
    learned_trajectories: list[list[float]]
    transition_pairs_x: list[float]
    transition_pairs_y: list[float]
