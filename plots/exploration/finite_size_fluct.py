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
from flowrect.simulations import particle_population, flow_rectification, FR_finite_fluctuations


# Plot saving parameters
save = False
save_path = ""
save_name = ""


def moving_average(x, w):
    return np.convolve(x, np.ones(w), "valid") / w


# Simulation parameters
N = 500
dt = 1e-2
I_ext = 5
params = dict(
    time_end=10,
    dt=dt,
    Lambda=[1.0, 2.5],
    Gamma=[-5.5, 1.0],
    # Lambda=[67.0, 7.0, 3.0],
    # Gamma=[-3.0, 4.0, -3.0],
    c=10,
    lambda_kappa=1,
    I_ext=5,
    I_ext_time=5,
    interaction=0.3,
)

a_cutoff = 7

print(f"FR with finite size fluctuations")
t = time.time()
ts, a_grid, rho_tN, m_t_exact, x_t, en_cons, A_orig, A1, Abar, S = FR_finite_fluctuations(
    a_cutoff=a_cutoff, **params
)
print(f"{time.time() - t:.2f}")

print(f"Particle simulation")
t = time.time()
ts, M, spikes, A, X = particle_population(**params, N=N, Gamma_ext=True)
m_t = calculate_mt(M, spikes)
# m_ts = np.zeros(m_t.T.shape)
# w = 50
# m_ts[: -w + 1, 0] = moving_average(m_t.T[:, 0], w)
# m_ts[: -w + 1, 1] = moving_average(m_t.T[:, 1], w)
# m_ts[-w + 1 :, :] = m_ts[-w, :]
print(f"{time.time() - t:.2f}")

print(f"Flow rectification approximation")
t = time.time()
ts, a_grid, rho_t, m_t_exact, x_t, en_cons, A_t = flow_rectification(a_cutoff=a_cutoff, **params)
print(f"{time.time() - t:.2f}s")


I_ext_vec = np.concatenate((np.zeros(int(len(ts) / 2)), I_ext * np.ones(int(len(ts) / 2))))
ages = calculate_age(spikes.T) * params["dt"]
A_av = moving_average(A, 100)

# Animated plots


class AnimatedPlot:
    def __init__(self, xlim=10, ylim=10):
        self.fig = plt.figure(figsize=(5.5, 9))
        gs = gridspec.GridSpec(2, 1, height_ratios=[5, 1])
        self.fig.suptitle(fr"PDE vs particle simulation $N=${N}")

        self.ax1 = plt.subplot(gs[0])
        self.ax2 = plt.subplot(gs[1])
        self.xlim, self.ylim = xlim, ylim
        self.plots = {}

    def init_plot(self):

        self.plots["title"] = self.ax1.text(
            0.5,
            0.85,
            "",
            bbox={"facecolor": "w", "alpha": 0.5, "pad": 5},
            transform=self.ax1.transAxes,
            ha="center",
        )

        # density plot (PDE)
        self.plots["p_rho"] = self.ax1.plot([], [], "-k", label="Particle")[0]
        self.plots["rho"] = self.ax1.plot(a_grid, rho_t[0], "--r", linewidth=1, label="PDE")[0]
        self.plots["rhoN"] = self.ax1.plot(a_grid, rho_tN[0], "-b", linewidth=1, label="Finite")[0]
        self.plots["S"] = self.ax1.plot(a_grid, S[0], "g", linewidth=1)[0]
        self.ax1.set_ylim(0, 4)
        self.ax1.set_title("Probability density distribution")
        self.ax1.legend(handles=[self.plots["rho"], self.plots["p_rho"], self.plots["rhoN"]])
        self.ax1.set_xlabel("Age a (s)")
        self.ax1.set_ylabel(r"$\rho_t$")

        self.ax2.plot()
        self.ax2.set_title("External input")
        self.plots["vline"] = self.ax2.plot([], [], "-r", linewidth=1)[0]
        self.ax2.set_ylim(0, 6)
        self.ax2.plot(ts, I_ext_vec, "-k")
        self.ax2.set_ylabel(r"$I^{ext}$ (a.u.)")
        self.ax2.set_xlabel(r"$t$ (s)")

        return tuple(self.plots.values())

    def calculate_hist(self, i):
        hist, bins = np.histogram(ages[:, i], bins=50, density=True)
        bins = (bins[1:] + bins[:-1]) / 2
        # w = 2
        return bins, hist  # moving_average(hist, w)

    def animate(self, i):
        t = dt * i
        # Scatter
        self.plots["title"].set_text(fr"Time $t=${t:.2f}s")
        # Particule rho
        bins, hist = self.calculate_hist(i)
        self.plots["p_rho"].set_data(bins, hist)
        self.plots["vline"].set_data(np.array([t, t]), np.array([0, 6]))

        # PDE rho
        self.plots["rho"].set_data(a_grid, rho_t[i])
        self.plots["rhoN"].set_data(a_grid, rho_tN[i])
        self.plots["S"].set_data(a_grid, S[i])
        return tuple(self.plots.values())


# Scatter plot
lim = 20
pl = AnimatedPlot(xlim=lim, ylim=lim)
anim_int = 4  # Want every 10ms
print(anim_int)

ani = animation.FuncAnimation(
    pl.fig,
    func=pl.animate,
    frames=range(0, len(M), anim_int),
    init_func=pl.init_plot,
    interval=5,
    blit=True,
)

if save:
    ani.save(os.path.join(save_path, save_name))

plt.figure()
A_av = moving_average(A, 50)
plt.plot(ts, A, "--k", label="Particle")
plt.plot(ts[: len(A_av)], A_av, "--r", label="P. rolling av.")
plt.plot(ts, A_t, "-b", linewidth=1.5, label="PDE")
plt.plot(ts, A1, "-c", label="A1")
plt.ylim(0, 10)
plt.plot(ts, Abar, "-.g", label="Abar")
plt.legend()
plt.show()