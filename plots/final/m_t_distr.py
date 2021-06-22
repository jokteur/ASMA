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


# Simulation parameters
N = 5000
dt = 1e-2
params = dict(
    time_end=40,
    dt=dt,
    Lambda=np.array([10.3, 2.5]),
    Gamma=np.array([-2.0, -1.0]),
    c=20,
    theta=0,
    lambda_kappa=50,
    I_ext=0,
    I_ext_time=30,
    interaction=0.0,
    kappa_type="exp",
)
time_before_input = 0.1  # Number of seconds to take into account before the I_ext
k_span = 1.8  # Number of second to show the evolution
num_k = 2  # Number of m_t to show per plot
a_cutoff = 10
Deltas = [4, 0.4, 0.04]
I_exts = [1.5, 1.5, 1.5]

dim = len(params["Lambda"])


def right_after_spike(M, spikes):
    M_after_spike = np.zeros(M.shape)
    for i in range(1, M.shape[0]):
        mask = spikes[i] == 1
        M_after_spike[i][mask] = M[i][mask]
        M_after_spike[i][~mask] = M_after_spike[i - 1][~mask]
    return M_after_spike


fig = plt.figure(figsize=(12, 5))
gs = gridspec.GridSpec(dim, len(Deltas))
axs = [plt.subplot(gs[i]) for i in range(dim * len(Deltas))]

fig.suptitle(fr"$m$ distribution just after spike")

bins_min = [1e9] * dim
bins_max = [-1e9] * dim
for i, (delta, I_ext) in enumerate(zip(Deltas, I_exts)):
    params["Delta"] = delta
    params["I_ext"] = I_ext

    # Particle simulation
    t = time.time()
    ts, M, spikes, A, H = particle_population(**params, N=N, use_LambdaGamma=True)
    m_tp = calculate_mt(M, spikes)
    print(f"Particle simulation done in {time.time()- t:.2f}s")
    # ASMA simulation
    t = time.time()
    (ts_ASMA, a_grid_ASMA, rho_t_ASMA, m_t_ASMA, x_t_ASMA, en_cons_ASMA, A_t_ASMA,) = ASA1(
        a_cutoff=a_cutoff,
        use_LambdaGamma=True,
        **params,
    )
    print(f"ASMA simulation done in {time.time()- t:.2f}s")
    # QR pde
    # t = time.time()
    # ts_QR, _, _, _, _, A_t_QR = quasi_renewal_pde(
    #     **params,
    #     a_cutoff=a_cutoff,
    #     use_LambdaGamma=True,
    # )
    # print(f"QR simulation done in {time.time()- t:.2f}s")

    # Figure out the indices
    begin_idx = int((params["I_ext_time"] - time_before_input) / dt)
    end_idx = int(params["I_ext_time"] / dt)

    for a in range(dim):
        # Subplots
        ax = axs[a * len(Deltas) + i]

        if a == 0:
            ax.set_title(fr"$\Delta = {delta}$")

        # m_t distribution
        ax.set_xlabel(fr"$m_{a}$ (after spike)")
        ax.set_ylabel(r"Density")

        alphas = np.linspace(1, 0.3, num_k)
        k_jump = int(k_span / dt / num_k)
        xmin = 1e-9
        xmax = -1e9
        ymax = -1e9
        # Figure out the distribution of m after spike
        M_after_spike = right_after_spike(M, spikes)

        # Legend stuff
        legends = []
        times = []
        texts = []
        for k, alpha in enumerate(alphas):
            # Find out the distribution after spike
            time_av_after_spike = M_after_spike[end_idx + k * k_jump, :, a]
            hist, bins = calculate_hist(time_av_after_spike, bins=25, density=True)

            m_t_mean = m_t_ASMA[end_idx + k * k_jump, a]
            m_tp_mean = m_tp[end_idx + k * k_jump, a]

            xmin = min(np.min(bins), xmin)
            xmax = max(np.max(bins), xmax)
            ymax = max(np.max(hist), ymax)

            # ax.set_title(fr"$\Delta = {delta}$")
            ax.set_yticks([0, np.round(max(ymax))])
            (legend1,) = ax.plot(bins, hist, "-k", alpha=alpha)
            legend2 = ax.axvline(
                x=m_t_mean,
                linestyle="--",
                color="r",
                alpha=alpha,
            )
            texts.append("Particle")
            texts.append(r"$\delta_{m - \overline{m}_t}$")
            times.append((end_idx + k * k_jump) * dt)
            legends.append(legend1)
            legends.append(legend2)

        bins_min[a] = min(xmin, bins_min[a])
        bins_max[a] = max(xmax, bins_max[a])

        if not i and not a:
            artists = []
            locs = ["upper left", "center left", "lower left"]
            handles = []
            handles_text = []
            for k, t in enumerate(times):
                (d1,) = ax.plot([0], marker="None", linestyle="None", label="dummy1")
                (d2,) = ax.plot([0], marker="None", linestyle="None", label="dummy2")
                handles += [d1, d2]
                handles_text += [fr"Time $t={t:.1f}$s", ""]
            handles += legends
            handles_text += texts

            ax.legend(handles, handles_text, ncol=2)
        if i == len(Deltas) - 1:
            for j in range(len(Deltas)):
                axs[a * len(Deltas) + j].set_xlim(bins_min[a] - 0.1, bins_max[a] + 0.1)
