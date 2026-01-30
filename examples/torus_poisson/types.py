"""Type definitions for the torus_poisson example."""

from typing import TypedDict


class TrainingHistory(TypedDict):
    """Training metrics tracked during optimization."""

    elbos: list[float]
    reconstruction_errors: list[float]
    r_squared: list[float]
    var_rls: list[float]  # Variance of reduced learning signal
    rho_norms: list[float]
    conjugation_errors: list[float]


class TuningParams(TypedDict):
    """Extracted tuning parameters for visualization."""

    preferred_theta1: list[float]  # First torus dimension preferred angles
    preferred_theta2: list[float]  # Second torus dimension preferred angles
    gains: list[float]  # Gain magnitudes for each neuron
    baselines: list[float]  # Log baseline rates


class GTConjugationMetrics(TypedDict):
    """Ground truth conjugation metrics for reference."""

    optimal_rho: list[float]  # Optimal rho via least squares
    optimal_rho_norm: float
    r_squared: float  # R² with optimal rho
    var_rls: float  # Var[RLS] with optimal rho
    var_psi: float  # Var[ψ_X(z)] for reference


class GroundTruth(TypedDict):
    """Ground truth model parameters for evaluation."""

    weight_matrix: list[list[float]]  # (n_neurons, 2*n_latent)
    baselines: list[float]  # (n_neurons,)
    prior_params: list[float]  # Von Mises prior parameters
    tuning: TuningParams
    conjugation: GTConjugationMetrics  # Ground truth conjugation metrics


class ModeResults(TypedDict):
    """Results for a single training mode."""

    history: TrainingHistory
    final_rho_norm: float
    final_r_squared: float
    final_var_rls: float  # Final variance of reduced learning signal
    final_elbo: float
    final_reconstruction_error: float
    learned_weight_matrix: list[list[float]]
    learned_baselines: list[float]
    learned_tuning: TuningParams


class TrainingConfig(TypedDict):
    """Configuration for training."""

    n_neurons: int
    n_latent: int
    n_samples: int
    n_steps: int
    batch_size: int
    learning_rate: float
    n_mc_samples: int
    seed: int
    distortion: float
    coverage: float
    baseline_rate: float
    gain: float
    prior_concentration: float
    conj_weight: float
    analytical_samples_mult: int


class Results(TypedDict):
    """Complete results from training."""

    # Training results per mode
    models: dict[str, ModeResults]

    # Ground truth
    ground_truth: GroundTruth

    # Best model
    best_model: str

    # Configuration
    config: TrainingConfig
