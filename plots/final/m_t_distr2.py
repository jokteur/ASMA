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
    # for i in range(1, M.shape[0]):
    #     mask = spikes[i] == 1
    #     M_after_spike[i][mask] = M[i][mask]
    #     M_after_spike[i][~mask] = M_after_spike[i - 1][~mask]
    return M_after_spike


def plot_m_t_distr(
    params,
    N,
    Deltas,
    I_exts,
    im_res=100,
    num_bins=25,
    margin=1,
    time_before_input=1,
    a_cutoff=10,
    plot_QR=True,
    I_ylim=None,
    params_p=None,
    plot_I=True,
    plot_H=False,
    savepath="",
    savename="",
    save=False,
    usetex=False,
    figsize=(12, 5),
    inset=None,
    dpi=None,
    title=None,
    font_family="serif",
    font_size="12",
    noshow=False,
):
    if usetex:
        plt.rc("text", usetex=True)
        plt.rc("font", family=font_family, size=font_size)

    dim = len(params["Lambda"])
    dt = params["dt"]

    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(dim, len(Deltas))
    axs = [plt.subplot(gs[i]) for i in range(dim * len(Deltas))]

    if title:
        fig.suptitle(title)
        fr"$m$ distribution just after spike"

    bins_min = [1e9] * dim
    bins_max = [-1e9] * dim
    for i, (delta, I_ext) in enumerate(zip(Deltas, I_exts)):
        params["Delta"] = delta
        params["I_ext"] = I_ext

        # Particle simulation
        t = time.time()
        ts, M, spikes, A, H = particle_population(**params, N=N)
        m_tp = calculate_mt(M, spikes)
        print(f"Particle simulation done in {time.time()- t:.2f}s")
        # ASMA simulation
        t = time.time()
        (ts_ASMA, a_grid_ASMA, rho_t_ASMA, m_t_ASMA, x_t_ASMA, en_cons_ASMA, A_t_ASMA,) = ASA1(
            a_cutoff=a_cutoff,
            **params,
        )
        print(f"ASMA simulation done in {time.time()- t:.2f}s")

        # Figure out the indices
        begin_idx = int((params["I_ext_time"] - time_before_input) / dt)
        end_idx = int(params["I_ext_time"] / dt)

        # Figure out the distribution of m after spike
        M_after_spike = right_after_spike(M, spikes)

        bins = num_bins - 3 * i
        time_points = np.repeat(ts[begin_idx:], bins)
        for a in range(dim):
            # Subplots
            ax = axs[a * len(Deltas) + i]
            hist_points, bin_points = fast_hist(M_after_spike[begin_idx:, :, a], bins)

            hist_points = hist_points.flatten()
            bin_points = bin_points.flatten()

            bin_min, bin_max = np.min(bin_points) - margin, np.max(bin_points) + margin
            t_min, t_max = np.min(ts), np.max(ts)

            # target grid to interpolate to
            # xi = time_points
            # yi = np.linspace(bin_min, bin_max, im_res)
            # xi, yi = np.meshgrid(xi, yi)

            # interpolate
            # zi = griddata((time_points, bin_points), hist_points, (xi, yi), method="cubic")

            # cs = ax.contourf(
            #     xi, yi, zi, levels=np.linspace(50, max(hist_points), 10), cmap="Greens", extend="both"
            # )
            scatter = ax.scatter(
                time_points,
                bin_points,
                s=0.2,
                marker="s",
                c=hist_points,
                cmap="Greens",
            )

            (m_t_plot,) = ax.plot(
                ts_ASMA[begin_idx:],
                m_t_ASMA[begin_idx:, a],
                "-r",
                alpha=0.5,
                label=r"$\delta_{m-\overline{m}_t}$",
            )
            # ax.plot(
            #     ts[begin_idx:],
            #     m_tp[begin_idx:, a],
            #     "-k",
            #     alpha=0.7,
            #     label=r"$\delta_{m-\overline{m}_{tp}}$",
            # )

            if a == 0:
                ax.set_title(fr"$\Delta = {delta}$ (mV)")

            # m_t distribution
            ax.set_xlabel(fr"Time $t$")
            ax.set_ylabel(fr"$m_{a}$")
            plt.tight_layout()
            if a == 0 and i == 0:
                handles, _ = scatter.legend_elements(alpha=0.6)
                ax.legend(
                    [m_t_plot, handles[-1]],
                    [
                        r"$\delta_{m-\overline{m}_t}$",
                        r"Density over $m$ (after spike)",
                    ],
                    loc="lower left",
                )

            xmin = 1e-9
            xmax = -1e9
            ymax = -1e9

            bins_min[a] = min(bin_min, bins_min[a])
            bins_max[a] = max(bin_max, bins_max[a])

            if i == len(Deltas) - 1:
                for j in range(len(Deltas)):
                    axs[a * len(Deltas) + j].set_ylim(bins_min[a], bins_max[a])

    if save:
        if dpi:
            fig.savefig(os.path.join(savepath, savename), dpi=dpi, transparent=True)
        else:
            fig.savefig(os.path.join(savepath, savename), transparent=True)
    if not noshow:
        plt.show()