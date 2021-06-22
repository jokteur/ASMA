import time
import copy
import os

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.text as mtext
from matplotlib import gridspec
from matplotlib.animation import FuncAnimation
import matplotlib.animation as animation

import flowrect
from flowrect.simulations.util import calculate_age, calculate_mt, eta_SRM
from flowrect.simulations import (
    particle_population,
    ASA1,
    quasi_renewal_pde,
    particle_population_nomemory,
    quasi_renewal,
)

from flowrect.simulations.pdes.QR import quasi_renewal_pde

# Plot saving parameters
save = False
usetex = False
save_path = "final\\plots"
save_name = "delta_to_0_A_t.pdf"


def moving_average(x, w):
    return np.convolve(x, np.ones(w), "valid") / w


def calculate_hist(array, **args):
    hist, bins = np.histogram(array, **args)
    bins = (bins[1:] + bins[:-1]) / 2
    return hist, bins  # moving_average(hist, w)


def plot_delta_0(
    params,
    N,
    Deltas,
    I_exts,
    ylim=6,
    time_before_input=2,
    a_cutoff=10,
    w=None,
    plot_QR=True,
    params_p=None,
    savepath="",
    savename="",
    save=False,
    usetex=False,
    figsize=(12, 5),
    dpi=None,
    title=None,
    font_family="serif",
    font_size="12",
    noshow=False,
):
    if usetex:
        plt.rc("text", usetex=True)
        plt.rc("font", family=font_family, size=font_size)

    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(1, len(Deltas))
    axs = [plt.subplot(gs[i]) for i in range(len(Deltas))]

    if title:
        fig.suptitle(title)
        fr"Activity when $\Delta \rightarrow 0$"

    for i, (delta, I_ext) in enumerate(zip(Deltas, I_exts)):
        params["Delta"] = delta
        params["I_ext"] = I_ext

        # Particle simulation
        t = time.time()
        ts_P, A, H, _ = particle_population_nomemory(**params, N=N)
        print(f"Particle simulation done in {time.time()- t:.2f}s")
        # ASMA simulation
        t = time.time()
        (ts_ASMA, _, _, _, _, _, A_t_ASMA,) = ASA1(
            a_cutoff=a_cutoff,
            **params,
        )
        print(f"ASMA simulation done in {time.time()- t:.2f}s")
        ts_QR, A_t_QR = None, None
        if plot_QR:
            # QR pde
            t = time.time()
            ts_QR, _, _, _, _, A_t_QR = quasi_renewal_pde(
                **params,
                a_cutoff=a_cutoff,
            )
            print(f"QR simulation done in {time.time()- t:.2f}s")

        begin_idx = int((params["I_ext_time"] - time_before_input) / params["dt"])
        begin_P_idx = int((params["I_ext_time"] - time_before_input) / params["dt"])

        ax = axs[i]
        ax.set_title(fr"$\Delta = {delta} (mV), " r"I^{\mathrm{ext}}" fr"={I_ext}$ (A)")
        if w:
            new_A = moving_average(A, w)
            ax.plot(
                ts_P[begin_P_idx + w // 2 - 1 : -w // 2],
                new_A[begin_P_idx:],
                "--k",
                label="Particle",
            )
        else:
            ax.plot(ts_P[begin_P_idx:], A[begin_P_idx:], "--k", label="Particle")

        ax.plot(ts_ASMA[begin_idx:], A_t_ASMA[begin_idx:], "-r", label="ASMA")
        if plot_QR:
            ax.plot(ts_QR[begin_idx:], A_t_QR[begin_idx:], "-b", label="QR")

        ax.set_ylim(0, ylim)
        ax.set_xlabel(r"Time $t$ (s)")
        ax.set_ylabel(r"Activity $A_t$ (Hz)")
        if not i:
            ax.legend()

    if save:
        if dpi:
            fig.savefig(os.path.join(savepath, savename), dpi=dpi, transparent=True)
        else:
            fig.savefig(os.path.join(savepath, savename), transparent=True)
    if not noshow:
        plt.show()
