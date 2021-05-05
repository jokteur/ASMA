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
from flowrect.simulations import particle_population, flow_rectification, quasi_renewal

save = False
save_path = ""
save_name = "bursting_kernel.pdf"

dt = 1e-2
N = 10
I_ext = 2.5
# Take similar as in article
time_end = 10
Lambda = np.array([28.0, 8.0, 1.0])
Gamma = np.array([-3.5, 3.0, -1.0])
params = dict(
    time_end=time_end,
    dt=dt,
    # Lambda=[1.0, 5.5],
    # Gamma=[-4.0, -1.0],
    Lambda=Lambda,
    Gamma=Gamma,
    c=20,
    lambda_kappa=2,
    I_ext=2,
    I_ext_time=0,
    interaction=0.0,
)
np.random.seed(123)
print(f"Particle simulation")
t = time.time()
ts, M, spikes, A, X = particle_population(**params, N=N, Gamma_ext=True)
print(f"{time.time() - t:.2f}")

fig = plt.figure(figsize=(8, 5))
gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1])

# Kernel
ax1 = plt.subplot(gs[0])
t = np.linspace(0, 10, 1000)
eta = eta_SRM(t, Gamma, Lambda)
ax1.plot(t, eta, "-r")
ax1.plot(t, np.zeros(len(t)), "-k", linewidth=0.5)
ax1.set_title(r"Kernel $\eta(t)$")
ax1.set_ylabel(r"$\eta(t)$")
# ax1.set_xlabel(r"$t$ (s)")

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

if save:
    fig.savefig(os.path.join(save_path, save_name), transparent=True)
plt.show()