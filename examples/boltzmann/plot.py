"""Plotting code for Boltzmann pattern learning analysis."""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from ..shared import example_paths
from .types import BoltzmannPatternResults


def plot_probability_comparison(ax: Axes, results: BoltzmannPatternResults) -> None:
    """Compare true vs learned probabilities for all states."""
    true_probs = np.array(results['true_probabilities'])
    learned_probs = np.array(results['learned_probabilities'])
    state_labels = results['state_labels']

    # Create bar plot comparison
    x = np.arange(len(state_labels))
    width = 0.35

    ax.bar(x - width/2, true_probs, width, label='True', alpha=0.8, color='blue')
    ax.bar(x + width/2, learned_probs, width, label='Fitted', alpha=0.8, color='red')

    ax.set_xlabel('Binary States')
    ax.set_ylabel('Probability')
    ax.set_title('True vs. Fitted Probabilities')
    ax.set_xticks(x)
    ax.set_xticklabels(state_labels, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3)


def plot_pattern_frequency_comparison(ax: Axes, results: BoltzmannPatternResults) -> None:
    """Compare pattern frequencies between ground truth, training data, and generated samples."""
    n_patterns = len(results['ground_truth_patterns'])
    x = np.arange(n_patterns)
    width = 0.25

    # Get the pattern frequencies
    gt_weights = [0.15, 0.15, 0.15, 0.15, 0.10, 0.10, 0.10, 0.10]  # From create_ground_truth_patterns
    pattern_freqs = results['pattern_frequencies']
    generated_freqs = results['generated_frequencies']

    ax.bar(x - width, gt_weights[:n_patterns], width, label='Ground Truth', alpha=0.8, color='blue')
    ax.bar(x, pattern_freqs, width, label='Training Data', alpha=0.8, color='gray')
    ax.bar(x + width, generated_freqs, width, label='Generated', alpha=0.8, color='red')

    ax.set_xlabel('Pattern Index')
    ax.set_ylabel('Frequency')
    ax.set_title('Pattern Frequencies')
    ax.set_xticks(x)
    ax.set_xticklabels([f'P{i+1}' for i in range(n_patterns)])
    ax.legend()
    ax.grid(True, alpha=0.3)


def plot_training_loss(ax: Axes, results: BoltzmannPatternResults) -> None:
    """Plot training loss history."""
    losses = results['training_losses']
    ax.plot(losses, color='blue', linewidth=2)
    ax.set_xlabel('Training Step')
    ax.set_ylabel('Cross-Entropy Loss')
    ax.set_title('Training History')
    ax.grid(True, alpha=0.3)

    # Add final loss value as text
    ax.text(0.95, 0.95, f'Final Loss: {losses[-1]:.4f}',
           transform=ax.transAxes, ha='right', va='top',
           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))


def plot_sampling_convergence_by_samples(ax: Axes, results: BoltzmannPatternResults) -> None:
    """Show convergence vs. number of samples collected (current form)."""
    sample_sizes = results['sampling_sizes']
    thin_levels = results['sampling_thin_levels']
    errors_matrix = results['sampling_errors_matrix']

    # Plot convergence for different thinning levels
    colors = ['blue', 'red', 'green', 'orange']

    for j, thin in enumerate(thin_levels):
        errors_for_thin = [errors_matrix[i][j] for i in range(len(sample_sizes))]
        color = colors[j % len(colors)]
        ax.loglog(sample_sizes, errors_for_thin, 'o-', color=color,
                 linewidth=2, markersize=4, label=f'thin={thin}')

    ax.set_xlabel('Number of Samples Collected')
    ax.set_ylabel('L2 Error')
    ax.set_title('Convergence vs. Samples')
    ax.grid(True, alpha=0.3)
    ax.legend()

    # Add theoretical 1/sqrt(n) convergence line for reference
    theoretical_slope = np.array(sample_sizes) ** (-0.5)
    theoretical_slope = theoretical_slope * errors_matrix[0][0] / theoretical_slope[0]  # Normalize
    ax.loglog(sample_sizes, theoretical_slope, 'k--', alpha=0.5, linewidth=1, label='1/√n')


def plot_sampling_convergence_by_computation(ax: Axes, results: BoltzmannPatternResults) -> None:
    """Show convergence vs. total computational effort (Gibbs steps)."""
    sample_sizes = results['sampling_sizes']
    thin_levels = results['sampling_thin_levels']
    errors_matrix = results['sampling_errors_matrix']

    n_burnin = 50  # From the implementation

    # Plot convergence vs total Gibbs steps
    colors = ['blue', 'red', 'green', 'orange']

    for j, thin in enumerate(thin_levels):
        errors_for_thin = [errors_matrix[i][j] for i in range(len(sample_sizes))]
        # Calculate total Gibbs steps for each sample size with this thinning
        total_steps = [n_burnin + (n_samples * thin) for n_samples in sample_sizes]
        color = colors[j % len(colors)]
        ax.loglog(total_steps, errors_for_thin, 'o-', color=color,
                 linewidth=2, markersize=4, label=f'thin={thin}')

    ax.set_xlabel('Total Gibbs Steps')
    ax.set_ylabel('L2 Error')
    ax.set_title('Convergence vs. Computational Effort')
    ax.grid(True, alpha=0.3)
    ax.legend()


def create_boltzmann_visualization(results: BoltzmannPatternResults) -> Figure:
    """Create the complete Boltzmann pattern learning visualization."""
    # Set up clean figure style
    plt.style.use('default')
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Boltzmann Machine Pattern Learning', fontsize=16, fontweight='bold')

    # Two main plots on top row: probabilities and training history
    plot_probability_comparison(axes[0, 0], results)
    plot_training_loss(axes[0, 1], results)

    # Two sampling convergence plots on bottom row
    plot_sampling_convergence_by_samples(axes[1, 0], results)
    plot_sampling_convergence_by_computation(axes[1, 1], results)

    plt.tight_layout()
    return fig


### Main ###

def main():
    """Create and save the Boltzmann pattern learning visualization."""
    paths = example_paths(__file__)

    # Load analysis results
    results = paths.load_analysis()

    # Create visualization
    fig = create_boltzmann_visualization(results)

    # Save plot
    paths.save_plot(fig)
    print(f"Plot saved to {paths.plot_path}")


if __name__ == "__main__":
    main()