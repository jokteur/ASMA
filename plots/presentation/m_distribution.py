import time
import copy
import os

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.text as mtext
from matplotlib import gridspec
from matplotlib.animation import FuncAnimation
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import matplotlib.animation as animation
from numba import jit, prange
from scipy.interpolate import griddata
import matplotlib

import flowrect
from flowrect.simulations.util import calculate_age, calculate_mt, eta_SRM
from flowrect.simulations import particle_population, ASA1, quasi_renewal_pde

from flowrect.simulations.pdes.QR import quasi_renewal_pde

# Plot saving parameters
save = False
usetex = False
save_path = "final\\plots"
save_name = "m_t_distr.pdf"


def moving_average(x, w):
    return np.convolve(x, np.ones(w), "valid") / w


def calculate_hist(array, **args):
    hist, bins = np.histogram(array, **args)
    bins = (bins[1:] + bins[:-1]) / 2
    return hist, bins  # moving_average(hist, w)


# @jit(nopython=True, cache=True)
def fast_hist(array, num_bins=25, density=True):
    nums = array.shape[0]
    N = array.shape[1]
    hist_points = np.zeros((nums, num_bins))
    bins_points = np.zeros((nums, num_bins))
    mask = ~np.isnan(array)
    for i in range(nums):
        hist, bins = np.histogram(array[i][mask[i]], bins=num_bins, density=False)
        bins = (bins[1:] + bins[:-1]) / 2
        hist_points[i] = hist
        bins_points[i] = bins

    return hist_points, bins_points


def right_after_spike(M, spikes):
    M_after_spike = M
    mask = spikes == 0
    M_after_spike[mask] = np.nan
    return M_after_spike


def plot_m_distribution(
    params,
    N,
    im_res=100,
    num_bins=200,
    margin=1,
    time_before_input=1,
    I_ylim=None,
    params_p=None,
    plot_I=True,
    savepath="",
    savename="",
    cmap="Greens",
    save=False,
    usetex=False,
    figsize=(12, 5),
    inset=None,
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

    dim = len(params["Lambda"])
    dt = params["dt"]

    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(1, dim)
    axs = [plt.subplot(gs[i]) for i in range(dim)]

    bins_min = [1e9] * dim
    bins_max = [-1e9] * dim

    # Particle simulation
    t = time.time()
    ts, M, spikes, A, H = particle_population(**params, N=N)
    m_tp = calculate_mt(M, spikes)
    print(f"Particle simulation done in {time.time()- t:.2f}s")

    # Figure out the indices
    begin_idx = int((params["I_ext_time"] - time_before_input) / dt)
    end_idx = int(params["I_ext_time"] / dt)

    # Figure out the distribution of m after spike

    time_points = np.repeat(ts[begin_idx:], num_bins)
    for a in range(dim):
        # Subplots
        ax = axs[a]
        hist_points, bin_points = fast_hist(M[begin_idx:, :, a], num_bins)

        hist_points = hist_points.flatten()
        bin_points = bin_points.flatten()

        label = r"Density over $m$"
        if usetex:
            label = r"Density over $\boldsymbol{m}$"
        scatter = ax.scatter(
            time_points,
            bin_points,
            s=0.2,
            marker="s",
            c=hist_points,
            cmap=cmap,
            label=label,
        )

        # m_t distribution
        ax.set_xlabel(fr"Time $t$ (s)")
        ax.set_ylabel(fr"$m_{a+1}$ (mV)")
        plt.tight_layout()
        if a == 0:
            handles, _ = scatter.legend_elements(alpha=0.6)
            ax.legend(
                [handles[-1]],
                [label],
                loc="lower left",
            )

    if save:
        if dpi:
            fig.savefig(os.path.join(savepath, savename), dpi=dpi, transparent=True)
        else:
            fig.savefig(os.path.join(savepath, savename), transparent=True)
    if not noshow:
        plt.show()