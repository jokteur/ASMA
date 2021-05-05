import time
import copy
from multiprocessing import Pool

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.animation as animation

import flowrect

from flowrect.simulations.util import calculate_age, calculate_mt, eta_SRM
from flowrect.simulations import particle_population

Gamma = np.array([1.0, -1.0, -0.5])
Lambda = np.array([2.0, 1.5, 1.0])

Lambda = np.array([28.0, 8.0, 1.0])
Gamma = np.array([-3.5, 3.0, -1.0])

N = 10
ts, M, spikes, A, X = particle_population(
    10, 1e-2, Gamma, Lambda, 0, 3, 0, 2, c=10, Gamma_ext=True, N=N
)
mask = spikes.T == 1
for i in range(3):
    plt.eventplot(
        ts[mask[i]],
        lineoffsets=i,
        colors="black",
        linewidths=0.5,
        linelengths=0.9,
    )

plt.figure()
plt.title(r"$\eta$ kernel")
plt.xlabel(r"$t$ (s)")
plt.ylabel(r"intensity (a.u)")
t = np.linspace(0, 1, 100)
eta = eta_SRM(t, Gamma, Lambda)
plt.plot(t, eta)
plt.show()