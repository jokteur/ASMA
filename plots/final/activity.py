import time
import copy
import os

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.text as mtext
from matplotlib import gridspec
from matplotlib.animation import FuncAnimation
import matplotlib.animation as animation
import matplotlib

import flowrect
from flowrect.simulations.util import calculate_age, calculate_mt, eta_SRM, moving_average
from flowrect.simulations import (
    particle_population_nomemory,
    ASA1,
    quasi_renewal_pde,
)

# params["Gamma"] = params["Gamma"] / params["Lambda"]


def plot_activity(
    params,
    time_before_input,
    ylim,
    N,
    w=None,
    a_cutoff=7,
    plot_QR=True,
    I_ylim=None,
    params_p=None,
    plot_I=True,
    plot_H=False,
    savepath="",
    savename="",
    save=False,
    usetex=False,
    figsize=(8, 8),
    inset=None,
    dpi=None,
    title=None,
    loc="best",
    font_family="serif",
    font_size="12",
    noshow=False,
):
    if usetex:
        plt.rc("text", usetex=True)
        plt.rc("font", family=font_family, size=font_size)
        matplotlib.rcParams["text.latex.preamble"] = [r"\usepackage{amsmath}"]

    dt = params["dt"]
    I_time = params["I_ext_time"]
    base_I = params["base_I"]
    I_ext = params["I_ext"]
    I_ext_vec = np.concatenate(
        (
            base_I * np.ones(int(I_time / dt)),
            (base_I + I_ext) * np.ones(int((params["time_end"] - I_time) / dt)),
        )
    )

    if not params_p:
        params_p = params

    # Particle simulation
    t = time.time()
    ts_P, A, H, _ = particle_population_nomemory(**params_p, N=N)
    print(f"Particle simulation done in {time.time()- t:.2f}s")
    # ASMA simulation
    t = time.time()
    (ts_ASMA, a_grid_ASMA, rho_t_ASMA, m_t_ASMA, h_t_ASMA, en_cons_ASMA, A_t_ASMA,) = ASA1(
        a_cutoff=a_cutoff,
        **params,
    )
    print(f"ASMA simulation done in {time.time()- t:.2f}s")
    ts_QR, h_t_QR, A_t_QR = None, None, None
    if plot_QR:
        # QR pde
        t = time.time()
        ts_QR, _, _, h_t_QR, _, A_t_QR = quasi_renewal_pde(
            **params,
            a_cutoff=a_cutoff,
        )
    print(f"QR simulation done in {time.time()- t:.2f}s")

    begin_idx = int((params["I_ext_time"] - time_before_input) / params["dt"])
    begin_P_idx = int((params_p["I_ext_time"] - time_before_input) / params_p["dt"])

    num_plots = 1 + int(plot_I) + int(plot_H)
    if num_plots == 1:
        height_ratios = [1]
    elif num_plots == 2:
        height_ratios = [5, 1]
    else:
        height_ratios = [5, 1, 1]

    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(num_plots, 1, height_ratios=height_ratios)

    ax1 = plt.subplot(gs[0])
    if title:
        ax1.set_title(title)

    plots = dict()

    if w:
        new_A = moving_average(A, w)
        (plots["p"],) = ax1.plot(
            ts_P[begin_P_idx + w // 2 - 1 : -w // 2],
            new_A[begin_P_idx:],
            "--k",
            label=r"$25\cdot10^3$ neurons",
        )
    else:
        (plots["p"],) = ax1.plot(
            ts_P[begin_P_idx:], A[begin_P_idx:], "--k", label=r"$25\cdot10^3$ neurons"
        )
    (plots["ASMA"],) = ax1.plot(ts_ASMA[begin_idx:], A_t_ASMA[begin_idx:], "-r", label="ASMA")
    if plot_QR:
        (plots["QR"],) = ax1.plot(ts_QR[begin_idx:], A_t_QR[begin_idx:], "-b", label="QR")

    ax1.set_ylim(ylim[0], ylim[1])
    ax1.set_xlim(ts_ASMA[begin_idx], ts_ASMA[-1])
    ax1.set_ylabel(r"Activity $A_t$ (Hz)")
    if num_plots == 1:
        ax1.set_xlabel(r"Time $t$ (s)")
    else:
        ax1.tick_params(direction="in")

    if inset:
        x1, x2, y1, y2 = inset[1]
        axins = ax1.inset_axes(inset[0])
        axins.set_xlim(x1, x2)
        axins.set_ylim(y1, y2)

        if w:
            w = 3
            new_A = moving_average(A, w)
            axins.plot(
                ts_P[begin_P_idx + w // 2 - 1 : -w // 2],
                new_A[begin_P_idx:],
                "--k",
                label=r"$25\cdot10^3$ neurons",
            )
        else:
            axins.plot(ts_P[begin_P_idx:], A[begin_P_idx:], "--k", label=r"$25\cdot10^3$ neurons")
        axins.plot(ts_ASMA[begin_idx:], A_t_ASMA[begin_idx:], "-r", label="ASMA")
        if plot_QR:
            axins.plot(ts_QR[begin_idx:], A_t_QR[begin_idx:], "-b", label="QR")

        axins.set_xticklabels("")
        axins.set_yticklabels("")
        ax1.indicate_inset_zoom(axins, edgecolor="black")
    ax1.legend(handles=plots.values(), loc=loc)

    i = 1

    I_ylim = I_ylim if I_ylim else (base_I + I_ext) * 1.1

    if plot_H:
        ax = plt.subplot(gs[i], sharex=ax1)
        i += 1

        ax.set_ylim(0, I_ylim)
        ax.plot(ts_P[begin_idx:], H[begin_idx:], "--k")
        ax.plot(ts_ASMA[begin_idx:], h_t_ASMA[begin_idx:], "-r")
        if plot_QR:
            ax.plot(ts_QR[begin_idx:], h_t_QR[begin_idx:], "-b")
        ax.set_ylabel(r"$h_t$ (mV)")
        ax.tick_params(direction="in")
        if i == num_plots:
            ax.tick_params(direction="out")
            ax.set_xlabel(r"Time $t$ (s)")

    if plot_I:
        ax = plt.subplot(gs[i], sharex=ax1)
        i += 1

        ax.set_ylim(0, I_ylim)
        ax.plot(ts_ASMA[begin_idx:], I_ext_vec[begin_idx:], "-k")
        ax.set_xlabel(r"$t$ (s)")
        ax.set_ylabel(r"$I^{\text{ext}}$ (A)")
        ax.tick_params(direction="in")
        if i == num_plots:
            ax.tick_params(direction="out")
            ax.set_xlabel(r"Time $t$ (s)")

    if save:
        if dpi:
            fig.savefig(os.path.join(savepath, savename), dpi=dpi, transparent=True)
        else:
            fig.savefig(os.path.join(savepath, savename), transparent=True)
    if not noshow:
        plt.show()