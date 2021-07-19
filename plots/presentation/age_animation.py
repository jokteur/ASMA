import time
import copy
import os
from numba import jit, prange

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.text as mtext
from matplotlib import gridspec
from matplotlib.animation import FuncAnimation
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import matplotlib.animation as animation
import matplotlib

import flowrect
from flowrect.simulations.util import calculate_age, calculate_mt, eta_SRM
from flowrect.simulations import particle_population, ASA1, quasi_renewal_pde


def calculate_hist(array, **args):
    hist, bins = np.histogram(array, **args)
    bins = (bins[1:] + bins[:-1]) / 2
    return hist, bins  # moving_average(hist, w)


def right_after_spike(M, spikes):
    M_after_spike = np.copy(M)
    mask = spikes == 0
    M_after_spike[mask] = np.nan
    return M_after_spike


def fast_hist(array, num_bins=25, density=True):
    nums = array.shape[0]
    N = array.shape[1]
    hist_points = np.zeros((nums, num_bins))
    bins_points = np.zeros((nums, num_bins))
    mask = ~np.isnan(array)
    for i in range(nums):
        hist, bins = np.histogram(array[i][mask[i]], bins=num_bins, density=True)
        bins = (bins[1:] + bins[:-1]) / 2
        hist_points[i] = hist
        bins_points[i] = bins

    return hist_points, bins_points


@jit(nopython=True, cache=True)
def smooth(m, w):
    m = np.copy(m)
    ret = np.zeros(m.shape)
    for i in range(w, m.shape[0] - w):
        for j in range(m.shape[1]):
            ret[i, j] = np.nanmean(m[i - w : i + w, j])
    return ret


class AnimatedPlot:
    def __init__(
        self,
        params,
        M,
        M_after_spike,
        M_distr,
        hist_points,
        bin_points,
        age,
        color,
        num_points=1000,
        along_char=False,
        plot_hist=None,
        plot_m_distr=None,
        plot_mean=None,
        characteristics=None,
        num_bins=25,
        xlim=10,
        M_lim=None,
        dim=1,
    ):
        self.M = M
        self.M_lim = M_lim
        self.num_bins = num_bins
        self.params = params
        self.M_after_spike = M_after_spike
        self.age = age
        self.dt = params["dt"]
        self.color = color
        self.plot_hist = plot_hist
        self.plot_mean = plot_mean
        self.M_hist_points, self.M_bin_points = M_distr
        self.plot_m_distr = plot_m_distr
        self.characteristics = characteristics
        self.dim = dim
        self.num_points = num_points
        self.along_char = along_char

        self.hist_points = hist_points
        self.bin_points = bin_points

        self.on = False

        if plot_hist:
            self.fig = plt.figure(figsize=(7, 7), facecolor=color)
            gs = gridspec.GridSpec(1, 2, width_ratios=[1, 5])
            self.main_ax = plt.subplot(gs[1])
            self.hist_ax = plt.subplot(gs[0])
            self.hist_ax.set_facecolor(color)

            self.hist_ax.spines["top"].set_visible(False)
            self.hist_ax.spines["left"].set_visible(False)
            self.hist_ax.spines["bottom"].set_visible(False)

            self.hist_ax.set_xticks([])
            self.hist_ax.set_yticks([])
        else:
            self.fig = plt.figure(figsize=(7, 7))
            gs = gridspec.GridSpec(1, 1)
            self.main_ax = plt.subplot(gs[0])

        self.main_ax.set_facecolor(color)
        if self.M_lim:
            self.main_ax.set_ylim(*M_lim)
            self.Mmin, self.Mmax = M_lim
        else:
            self.main_ax.set_ylim(np.min(M), np.max(M))
            self.Mmin, self.Mmax = np.min(M), np.max(M)

        if plot_hist:
            self.hist_ax.set_ylim(self.Mmin, self.Mmax)
            self.hist_ax.set_xlim(-np.mean(self.hist_points) * 4, 0)

        self.xlim = xlim
        self.plots = {}

    def init_plot(self):
        self.plots["title"] = self.main_ax.text(
            0.75,
            0.85,
            "",
            bbox={"facecolor": self.color, "alpha": 0.5, "pad": 5},
            transform=self.main_ax.transAxes,
            ha="center",
        )
        self.plots["input_t"] = self.main_ax.text(
            0.75,
            0.65,
            "external input = OFF",
            bbox={"facecolor": self.color, "alpha": 0.5, "pad": 5},
            transform=self.main_ax.transAxes,
            ha="center",
        )
        self.plots["p"] = self.main_ax.scatter([], [], color="k", s=1.3, label="state of 1 neuron")

        self.main_ax.set_xlim(0, self.xlim)
        self.main_ax.set_xlabel(r"Age $a$ (s)")
        self.main_ax.set_ylabel(r"$m$ (mV)")

        if self.characteristics:
            m = np.linspace(self.Mmin, self.Mmax, self.characteristics)
            a = np.linspace(0, self.xlim, 101)

            line = np.exp(-self.params["Lambda"][self.dim] * a)
            if self.along_char:
                line = np.ones(a.shape)
            for i in range(self.characteristics):
                self.main_ax.plot(
                    a,
                    m[i] * line,
                    "-",
                    color="r",
                    linewidth=0.9,
                    alpha=0.5,
                )

        if self.plot_hist:
            (self.plots["hist"],) = self.hist_ax.plot([], [], "-", color="g")
            if self.plot_mean:
                (self.plots["mean"],) = self.hist_ax.plot([], [], "-", color="r")
            if self.plot_m_distr:
                (self.plots["m_distr"],) = self.hist_ax.plot([], [], "-", color="orange")
        return tuple(self.plots.values())

    def animate(self, i):
        t = self.dt * i
        self.plots["title"].set_text(fr"Time $t=${t:.2f}s")

        lim = 3000 if self.M.shape[1] > 3000 else self.M.shape[1]

        y = self.M[i, :lim, self.dim]
        x = self.age[i, :lim] * self.dt
        if self.along_char:
            y = np.exp(self.params["Lambda"][self.dim] * x) * y
        # Scatter
        self.plots["p"].set_offsets(list(zip(x, y)))

        if t >= self.params["I_ext_time"] and not self.on:
            self.plots["input_t"].set_text("external input = ON")
            self.plots["input_t"].set_color("r")
            self.on = True

        if self.plot_hist:
            self.plots["hist"].set_data(-self.hist_points[i], self.bin_points[i])
            if self.plot_mean:
                yline = [np.nanmean(self.M_after_spike[i])] * 2
                xline = [-1e6, 0]
                self.plots["mean"].set_data(xline, yline)
            if self.plot_m_distr:
                self.plots["m_distr"].set_data(-self.M_hist_points[i], self.M_bin_points[i])

        return tuple(self.plots.values())


def plot_animation(
    params,
    N,
    xlim=10,
    M_lim=None,
    num_bins=25,
    anim_int=4,
    t_from=4,
    t_to=6,
    select_dim=1,
    num_points=1000,
    plot_hist=False,
    plot_mean=False,
    plot_m_distr=False,
    characteristics=False,
    along_char=False,
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
        matplotlib.rcParams["text.latex.preamble"] = [r"\usepackage{amsmath}"]

    dim = len(params["Lambda"])
    dt = params["dt"]
    color = "#fafafa"

    # Particle simulation
    t = time.time()
    ts, M, spikes, A, H = particle_population(**params, N=N)
    m_tp = calculate_mt(M, spikes)
    print(f"Particle simulation done in {time.time()- t:.2f}s")

    age = calculate_age(spikes.T).T

    M_after_spike = right_after_spike(M, spikes)
    smooth_M_after_spike = smooth(M_after_spike[:, :, select_dim], 11)
    hist_points, bin_points = fast_hist(smooth_M_after_spike, num_bins)
    M_hist_points, M_bins_points = fast_hist(smooth(M, 11), num_bins)

    pl = AnimatedPlot(
        params,
        M,
        smooth_M_after_spike,
        (M_hist_points, M_bins_points),
        hist_points,
        bin_points,
        age,
        color,
        plot_m_distr=plot_m_distr,
        along_char=along_char,
        num_bins=num_bins,
        dim=select_dim,
        plot_hist=plot_hist,
        plot_mean=plot_mean,
        characteristics=characteristics,
        xlim=xlim,
        M_lim=M_lim,
    )

    idx_start = int(t_from / dt)
    idx_end = int(t_to / dt)

    ani = animation.FuncAnimation(
        pl.fig,
        func=pl.animate,
        frames=range(idx_start, idx_end, anim_int),
        init_func=pl.init_plot,
        interval=25,
        blit=True,
    )

    if save:
        ani.save(os.path.join(savepath, savename))
    if not noshow:
        plt.show()