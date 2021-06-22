import time
import copy
import os
from multiprocessing import Pool

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.animation import FuncAnimation
import matplotlib.animation as animation

import flowrect

from flowrect.simulations.util import calculate_age, calculate_mt, eta_SRM
from flowrect.simulations import particle_population


def plot_kernel(
    params,
    N,
    time_kernel=None,
    plot_spikes=True,
    seed=None,
    savepath="",
    savename="",
    save=False,
    usetex=False,
    figsize=(8, 5),
    margin=0.05,
    dpi=None,
    title=None,
    font_family="serif",
    font_size="12",
    noshow=False,
):
    if usetex:
        plt.rc("text", usetex=True)
        plt.rc("font", family=font_family, size=font_size)
    num_plots = 1 + int(plot_spikes)

    if num_plots == 1:
        height_ratios = [1]
    elif num_plots == 2:
        height_ratios = [3, 1]

    if seed:
        np.random.seed(seed)

    Lambda = params["Lambda"]
    Gamma = params["Gamma"]
    if "use_LambdaGamma" in params and params["use_LambdaGamma"]:
        Gamma = Lambda * Gamma

    print(f"Particle simulation")
    t = time.time()
    ts, M, spikes, A, H = particle_population(**params, N=N)
    print(f"{time.time() - t:.2f}")

    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(num_plots, 1, height_ratios=height_ratios)

    # Kernel
    ax1 = plt.subplot(gs[0])

    t = params["time_end"]
    if time_kernel:
        t = time_kernel
    ts_kernel = np.linspace(0, t, 1000)
    eta = eta_SRM(ts_kernel, Gamma, Lambda)
    ax1.plot(ts_kernel, eta, "-r")
    ax1.set_xlim(-margin, t + margin)
    ax1.plot(ts_kernel, np.zeros(len(ts_kernel)), "-k", linewidth=0.5)
    if title:
        ax1.set_title(title)
    else:
        ax1.set_title(r"Kernel $\eta(t)$")
    ax1.set_ylabel(r"$\eta(t)$")
    ax1.set_xlabel(r"$t$ (s)")

    # Spike train
    ax2 = plt.subplot(gs[1])
    mask = spikes.T == 1
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
    ax2.set_xlim(-margin, params["time_end"] + margin)
    plt.tight_layout()

    if save:
        if dpi:
            fig.savefig(os.path.join(savepath, savename), dpi=dpi, transparent=True)
        else:
            fig.savefig(os.path.join(savepath, savename), transparent=True)
    if not noshow:
        plt.show()