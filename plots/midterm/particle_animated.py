import time
import copy
import os

import numpy as np
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
save_name = "leaky_mem_anim.mp4"

# Simulation parameters
N = 500
dt = 1e-3
params = dict(
    time_end=10,
    dt=dt,
    # Lambda=[5.0, 2.5],
    # Gamma=[-4.0, 1.0],
    Lambda=np.array([28.0, 8.0, 1.0]),
    Gamma=np.array([-3.5, 3.0, -1.0]),
    c=10,
    lambda_kappa=1,
    I_ext=1,
    I_ext_time=20,
    interaction=0.1,
)


print(f"Particle simulation")
t = time.time()
ts, M, spikes, A, X = particle_population(**params, N=N, Gamma_ext=True)
print(f"{time.time() - t:.2f}")

# Animated plots


class AnimatedPlot:
    def __init__(self, xlim=10, ylim=10):
        self.fig = plt.figure(figsize=(6, 8))
        gs = gridspec.GridSpec(2, 1, height_ratios=[5, 1])
        self.fig.suptitle(fr"Leaky memory population simulation ($N=${N})")

        self.ax1 = plt.subplot(gs[0])
        self.ax2 = plt.subplot(gs[1])
        self.xlim, self.ylim = xlim, ylim
        self.plots = {}

    def init_plot(self):
        # M scatter plot
        self.plots["scat"] = self.ax1.scatter([], [], c="r", s=0.4)
        self.ax1.set_aspect("equal")
        self.plots["title"] = self.ax1.text(
            0.5,
            0.85,
            "",
            bbox={"facecolor": "w", "alpha": 0.5, "pad": 5},
            transform=self.ax1.transAxes,
            ha="center",
        )
        self.ax1.set_xlabel(r"$M_1$ (mV)")
        self.ax1.set_ylabel(r"$M_2$ (mV)")
        self.ax1.set_xlim(-self.xlim, self.xlim)
        self.ax1.set_ylim(-self.ylim, self.ylim)

        # Activity plot
        mask = spikes.T == 1
        self.plots["vline"] = self.ax2.plot([], [], "-r", linewidth=1)[0]
        for i in range(N):
            self.ax2.eventplot(
                ts[mask[i]],
                lineoffsets=i + 0.5,
                colors="black",
                linewidths=0.5,
            )
        self.ax2.set_ylim(0, N)
        self.ax2.set_xlabel(r"$t$ (s)")
        self.ax2.set_ylabel(r"Spikes")
        self.ax2.set_yticks([])

        return tuple(self.plots.values())

    def animate(self, i):
        t = dt * i
        # Scatter
        self.plots["title"].set_text(fr"Time $t=${t:.2f}s")
        self.plots["scat"].set_offsets(M[i, :, 0:2])
        self.plots["vline"].set_data(np.array([t, t]), np.array([0, N]))

        return tuple(self.plots.values())


# Scatter plot
lim = 10
pl = AnimatedPlot(xlim=lim, ylim=lim)
anim_int = 40


ani = animation.FuncAnimation(
    pl.fig,
    func=pl.animate,
    frames=range(0, len(M), anim_int),
    init_func=pl.init_plot,
    interval=anim_int,
    blit=True,
)

if save:
    ani.save(os.path.join(save_path, save_name))
plt.show()