import numpy as np
import copy
import os
import matplotlib.pyplot as plt
import matplotlib
from matplotlib import gridspec

import flowrect
from flowrect.simulations.util import calculate_age, calculate_mt, eta_SRM
from flowrect.simulations import particle_population, ASA1, quasi_renewal_pde

from flowrect.simulations.pdes.QR import quasi_renewal_pde


def plot_leaky_memory(
    params,
    N,
    savepath="",
    savename="",
    save=False,
    usetex=False,
    figsize=(6, 4),
    dpi=None,
    title=None,
    simple_m=False,
    font_family="serif",
    font_size="12",
    noshow=False,
):
    if usetex:
        plt.rc("text", usetex=True)
        plt.rc("font", family=font_family, size=font_size)
        matplotlib.rcParams["text.latex.preamble"] = [r"\usepackage{amsmath}"]

    np.random.seed(124)
    ts, M, spikes, A, _ = particle_population(**params)
    mask = spikes.T == 1

    dt = params["dt"]

    for i in range(3):
        fig = plt.figure(figsize=(7, 5))
        gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1])

        # Leaky memory plot
        ax1 = plt.subplot(gs[0])
        ax1.plot(ts, M[:, 0, 1], "-k", linewidth=0.9)
        ax1.set_ylabel(r"Leaky memory $M$ (mV)")

        ax1.annotate(
            r"decreases exp. at rate $\tau^{-1}$",
            color="grey",
            xy=(0.05, M[int(0.05 / dt), 0, 1]),
            xycoords="data",
            xytext=(0.5, 0),
            textcoords="axes fraction",
            arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=-0.3"),
            horizontalalignment="right",
            verticalalignment="bottom",
        )

        text = None
        if i == 1:
            text = r"spikes with probability $dt \cdot f(M + h_t)$" " \n"
        if i == 2:
            text = (
                r"spikes with probability $dt \cdot f(M + h_t)$"
                " \n"
                r"and $M$ jumps with size of $\Gamma\tau^{-1}$"
            )
        if i > 0:
            ax1.annotate(
                text,
                color="grey",
                xy=(0.51, 1.5),
                xycoords="data",
                xytext=(0.32, 0.45),
                textcoords="axes fraction",
                arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0.4"),
                horizontalalignment="left",
                verticalalignment="top",
            )

        # Spike plot
        ax2 = plt.subplot(gs[1], sharex=ax1)
        ax2.eventplot(
            ts[mask[0]],
            lineoffsets=0.5,
            colors="black",
            linewidths=0.5,
        )
        ax2.set_xlabel(r"$t$ (s)")
        ax2.set_yticks([])
        ax2.set_ylabel("Spikes")
        ax2.set_ylim(0, 1)
        plt.tight_layout()

        if save:
            if dpi:
                fig.savefig(os.path.join(savepath, f"{i}_" + savename), dpi=dpi, transparent=True)
            else:
                fig.savefig(os.path.join(savepath, f"{i}_" + savename), transparent=True)
    if not noshow:
        plt.show()
