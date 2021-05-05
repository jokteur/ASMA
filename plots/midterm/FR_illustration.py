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
save_name = "m_t2.pdf"

# Simulation parameters

Lambda = np.array([33.0, 8.0])
Gamma = np.array([-8, 1.0])

N = 10
dt = 1e-4
np.random.seed(123)
ts, M, spikes, A, X = particle_population(
    0.18, dt, Gamma, Lambda, 0, 3, 0, 2, c=10, Gamma_ext=True, N=N
)
mask = spikes.T == 1

ticks = ts[spikes.T[0] == 1]
ticks_text = [r"$t^{(1)}$", r"$t^{(2)}$"]

fig = plt.figure(figsize=(6, 4))
gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1])

# Calculate m_t
spike_mask = spikes.T[0] == 1
m_t = np.zeros(len(ts))
for s in range(1, len(ts)):
    if spike_mask[s]:
        m_t[s] = M[s, 0, 1]
    else:
        m_t[s] = m_t[s - 1]

# Leaky memory plot
ax1 = plt.subplot(gs[0])
ax1.set_yticks([])
ax1.plot(ts, M[:, 0, 1], "-k", linewidth=0.9, label=r"$M$")
ax1.plot(ts, m_t, "-r", linewidth=0.9, label=r"$m_t$")
ax1.set_ylim(0, 2)
ax1.legend()

text = (
    r"$m_t(t^{(2)}) = m_t(t^{(1)})"
    "\cdot e^{-\lambda (t^{(2)} - t^{(1)})} + \Gamma$"
    "\n"
    r"           $= m_t(t^{(1)}) \cdot e^{-\lambda a} + \Gamma $"
)
ax1.annotate(
    text,
    color="grey",
    xy=(0.11, 1.08),
    xycoords="data",
    xytext=(0.2, 0.9),
    textcoords="axes fraction",
    arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=-0.3"),
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
ax2.set_xticks(ticks)
ax2.set_xticklabels(ticks_text)

ax2.set_yticks([])
ax2.set_ylabel("Spikes")
ax2.set_ylim(0, 1)
if save:
    fig.savefig(os.path.join(save_path, save_name), transparent=True)
plt.show()