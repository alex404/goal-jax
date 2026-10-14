"""Plotting for Hierarchical Mixture of Gaussians."""

from typing import cast

import matplotlib.pyplot as plt
import numpy as np

from ..shared import (
    apply_style,
    colors,
    example_paths,
    model_color,
    plot_density_contours,
)
from .types import HMoGResults


def main():
    paths = example_paths(__file__)
    apply_style(paths)

    results = cast(HMoGResults, paths.load_analysis())

    sample = np.array(results["observations"])
    x1s = np.array(results["plot_range_x1"])
    x2s = np.array(results["plot_range_x2"])
    ys = np.array(results["plot_range_y"])
    x1_mesh, x2_mesh = np.meshgrid(x1s, x2s)

    true_obs = np.array(results["observable_densities"]["Ground Truth"])
    iso_init = np.array(results["observable_densities"]["Isotropic Initial"])
    iso_final = np.array(results["observable_densities"]["Isotropic Final"])
    dia_init = np.array(results["observable_densities"]["Diagonal Initial"])
    dia_final = np.array(results["observable_densities"]["Diagonal Final"])

    true_mix = np.array(results["mixture_densities"]["Ground Truth"])
    iso_init_mix = np.array(results["mixture_densities"]["Isotropic Initial"])
    iso_final_mix = np.array(results["mixture_densities"]["Isotropic Final"])
    dia_init_mix = np.array(results["mixture_densities"]["Diagonal Initial"])
    dia_final_mix = np.array(results["mixture_densities"]["Diagonal Final"])

    iso_lls = np.array(results["log_likelihoods"]["Isotropic"])
    dia_lls = np.array(results["log_likelihoods"]["Diagonal"])

    test_lls = results["test_log_likelihoods"]

    fig = plt.figure(figsize=(22.5, 7.5))
    gs = fig.add_gridspec(2, 6)

    ax_gt = fig.add_subplot(gs[0, 0])
    ax_iso_init = fig.add_subplot(gs[0, 1])
    ax_dia_init = fig.add_subplot(gs[0, 2])
    ax_ll = fig.add_subplot(gs[0, 3])
    ax_iso_final = fig.add_subplot(gs[1, 0])
    ax_dia_final = fig.add_subplot(gs[1, 1])
    ax_lat_init = fig.add_subplot(gs[1, 2])
    ax_lat_final = fig.add_subplot(gs[1, 3])

    # Ground truth
    plot_density_contours(
        ax_gt, x1_mesh, x2_mesh, [true_obs], ["True"], sample, [colors["ground_truth"]]
    )
    ax_gt.set_title("Ground Truth")

    # Initial fits
    plot_density_contours(
        ax_iso_init, x1_mesh, x2_mesh, [true_obs, iso_init], ["True", "Initial"], sample
    )
    ax_iso_init.set_title("Isotropic Initial")

    plot_density_contours(
        ax_dia_init, x1_mesh, x2_mesh, [true_obs, dia_init], ["True", "Initial"], sample
    )
    ax_dia_init.set_title("Diagonal Initial")

    # Training history
    ax_ll.plot(
        iso_lls,
        label=f"Isotropic (test {test_lls['Isotropic']:.4f})",
        color=model_color(0),
    )
    ax_ll.plot(
        dia_lls,
        label=f"Diagonal (test {test_lls['Diagonal']:.4f})",
        color=model_color(1),
    )
    ax_ll.plot(
        [], [], color="none", label=f"Ground truth test {test_lls['Ground Truth']:.4f}"
    )
    ax_ll.set_xscale("log")
    ax_ll.set_xlabel("EM Step")
    ax_ll.set_ylabel("Log Likelihood")
    ax_ll.set_title("Training History")
    ax_ll.legend()

    # Final fits
    plot_density_contours(
        ax_iso_final, x1_mesh, x2_mesh, [true_obs, iso_final], ["True", "Final"], sample
    )
    ax_iso_final.set_title("Isotropic Final")

    plot_density_contours(
        ax_dia_final, x1_mesh, x2_mesh, [true_obs, dia_final], ["True", "Final"], sample
    )
    ax_dia_final.set_title("Diagonal Final")

    # Latent densities
    lat_colors = [colors["ground_truth"], model_color(0), model_color(1)]
    ax_lat_init.plot(ys, true_mix, color=lat_colors[0], label="True")
    ax_lat_init.plot(ys, iso_init_mix, color=lat_colors[1], label="Isotropic")
    ax_lat_init.plot(ys, dia_init_mix, color=lat_colors[2], label="Diagonal")
    ax_lat_init.set_xlabel("y")
    ax_lat_init.set_ylabel("Density")
    ax_lat_init.set_title("Initial Latent Distributions")
    ax_lat_init.legend()

    ax_lat_final.plot(ys, true_mix, color=lat_colors[0], label="True")
    ax_lat_final.plot(ys, iso_final_mix, color=lat_colors[1], label="Isotropic")
    ax_lat_final.plot(ys, dia_final_mix, color=lat_colors[2], label="Diagonal")
    ax_lat_final.set_xlabel("y")
    ax_lat_final.set_ylabel("Density")
    ax_lat_final.set_title("Final Latent Distributions")
    ax_lat_final.legend()

    # Variational fit
    var = results["variational"]
    ax_var = fig.add_subplot(gs[0, 4])
    ax_var.plot(
        var["steps"],
        var["log_likelihood"],
        color=model_color(1),
        label="Log likelihood",
    )
    ax_var.plot(
        var["steps"], var["elbo"], color=model_color(2), linestyle="--", label="ELBO"
    )
    ax_var.axhline(dia_lls[-1], color=colors["ground_truth"], linestyle=":", label="EM")
    ax_var.set_xlabel("Adam Step")
    ax_var.set_ylabel("Log Likelihood")
    ax_var.set_title("Diagonal, Variational")
    ax_var.legend()

    ax_rho = fig.add_subplot(gs[1, 4])
    rho = np.array(var["rho"])
    exact_rho = np.array(var["exact_rho"])
    for i, name in enumerate((r"$y$", r"$y^2$")):
        ax_rho.plot(
            var["steps"],
            exact_rho[:, i],
            color=model_color(i),
            label=f"{name} coefficient, exact",
        )
        ax_rho.plot(
            var["steps"],
            rho[:, i],
            color=model_color(i),
            linestyle="--",
            label=f"{name} coefficient, learned",
        )
    ax_rho.set_xlabel("Adam Step")
    ax_rho.set_ylabel("Conjugation Parameters")
    ax_rho.set_title("Learned and Exact $\\rho_Y$")
    ax_rho.legend()

    ax_rec = fig.add_subplot(gs[:, 5])
    exact_logits = np.array(var["exact_logits"])
    rec_logits = np.array(var["recognition_logits"])
    lim = float(np.max(np.abs(exact_logits)))
    ax_rec.plot([-lim, lim], [-lim, lim], color=colors["ground_truth"], linestyle=":")
    ax_rec.scatter(exact_logits, rec_logits, s=8, color=model_color(1))
    ax_rec.set_xlabel("Exact logit of $k = 1$")
    ax_rec.set_ylabel("Recognition logit of $k = 1$")
    ax_rec.set_title("Recognition Model of $k$ ($\\rho_K$)")

    paths.save_plot(fig)


if __name__ == "__main__":
    main()
