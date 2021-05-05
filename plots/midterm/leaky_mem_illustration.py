import time
import copy
import os

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.animation import FuncAnimation
import matplotlib.animation as animation

import flowrect

from flowrect.simulations.util import calculate_age, calculate_mt, eta_SRM
from flowrect.simulations import particle_population, flow_rectification, quasi_renewal

# Plot saving parameters
save = False
save_path = ""
save_name = "leaky_memory2.pdf"

# Simulation parameters
Lambda = np.array([33.0, 8.0])
Gamma = np.array([-8, 1.0])

N = 10
dt = 1e-3
np.random.seed(123)
ts, M, spikes, A, X = particle_population(
    0.5, dt, Gamma, Lambda, 0, 3, 0, 2, c=10, Gamma_ext=True, N=N
)
mask = spikes.T == 1

fig = plt.figure(figsize=(6, 4))
gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1])

# Leaky memory plot
ax1 = plt.subplot(gs[0])
ax1.plot(ts, M[:, 0, 1], "-k", linewidth=0.9)
ax1.set_ylabel(r"Leaky memory $M$ (mV)")

ax1.annotate(
    r"decreases exp. at rate $\lambda$",
    color="grey",
    xy=(0.05, M[int(0.05 / dt), 0, 1]),
    xycoords="data",
    xytext=(0.5, 0),
    textcoords="axes fraction",
    arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=-0.3"),
    horizontalalignment="right",
    verticalalignment="bottom",
)

text = r"spikes with probability $dt \cdot f(M + x_t)$" " \n"
# text = r"spikes with probability $dt \cdot f(M + x_t)$" " \n" r"and $M$ jumps with size of $\Gamma$"
ax1.annotate(
    text,
    color="grey",
    xy=(0.096, 1.08),
    xycoords="data",
    xytext=(0.05, 0.8),
    textcoords="axes fraction",
    arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=-0.4"),
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
if save:
    fig.savefig(os.path.join(save_path, save_name), transparent=True)
plt.show()