import time
import copy

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.animation as animation

from core.util import calculate_age, calculate_mt, eta_SRM
from core.population import simulation
from core.pde import flow_rectification
from core.quasi_renewal import QR


def moving_average(x, w):
    return np.convolve(x, np.ones(w), "valid") / w


# Simulation parameters
N = 5000
dt = 1e-2
params = dict(
    time_end=40,
    dt=dt,
    Lambda=[33.3, 2.5],
    Gamma=[-8.0, -1.0],
    # Lambda=[67.0, 7.0, 3.0],
    # Gamma=[-3.0, 4.0, -3.0],
    c=1,
    lambda_kappa=1,
    I_ext=1,
    I_ext_time=20,
    interaction=0,
)

print(f"QR approximation")
QR_params = copy.copy(params)
QR_params["dt"] = 1e-2
t = time.time()
ts_QR, A_QR, cutoff = QR(**QR_params)
print(f"{time.time() - t:.2f}s")


print(f"Particle simulation")
t = time.time()
ts, M, spikes, A, X = simulation(**params, N=N, Gamma_ext=True)
m_t = calculate_mt(M, spikes)
# m_ts = np.zeros(m_t.T.shape)
# w = 50
# m_ts[: -w + 1, 0] = moving_average(m_t.T[:, 0], w)
# m_ts[: -w + 1, 1] = moving_average(m_t.T[:, 1], w)
# m_ts[-w + 1 :, :] = m_ts[-w, :]
print(f"{time.time() - t:.2f}")

print(f"Flow rectification approximation")
t = time.time()
ts, a_grid, rho_t, m_t_exact, x_t, en_cons, A_t = flow_rectification(a_cutoff=7, **params)
print(f"{time.time() - t:.2f}s")


ages = calculate_age(spikes.T) * params["dt"]
A_av = moving_average(A, 100)

# Animated plots


class AnimatedPlot:
    def __init__(self, xlim=10, ylim=10):
        self.fig, (
            (self.ax11, self.ax12, self.ax31),
            (self.ax21, self.ax22, self.ax32),
        ) = plt.subplots(2, 3, figsize=(15, 8))
        self.fig.suptitle(f"Particle simulation ({N=}) vs PDE")
        self.xlim, self.ylim = xlim, ylim
        self.plots = {}

    def init_plot(self):
        # M scatter plot
        self.plots["scat"] = self.ax11.scatter([], [], c="r", s=0.4)
        self.ax11.set_aspect("equal")
        self.plots["title"] = self.ax11.text(
            0.5,
            0.85,
            "",
            bbox={"facecolor": "w", "alpha": 0.5, "pad": 5},
            transform=self.ax11.transAxes,
            ha="center",
        )
        self.ax11.set_xlabel("M1")
        self.ax11.set_ylabel("M2")
        self.ax11.set_xlim(-self.xlim, self.xlim)
        self.ax11.set_ylim(-self.ylim, self.ylim)

        # Activity plot
        self.ax12.set_title("Activity")
        (A_1,) = self.ax12.plot(ts, A, "--k", linewidth=0.5, label="Particle")
        (A_2,) = self.ax12.plot(ts[: len(A_av)], A_av, "--r", label="P. rolling av.")
        (A_3,) = self.ax12.plot(ts, A_t, "-.g", linewidth=1.5, label="PDE")
        (A_4,) = self.ax12.plot(ts_QR, A_QR, "-b", linewidth=1.5, label="QR")
        self.ax12.set_ylim(0, 100)
        self.ax12.legend(handles=[A_1, A_2, A_3])
        self.ax12.set_xlabel("Time (s)")

        # m_t plot
        (m_t1,) = self.ax21.plot(ts, m_t[0], "-r", linewidth=0.5, label="m_t (1)")
        (m_t2,) = self.ax21.plot(ts, m_t[1], "-b", linewidth=0.5, label="m_t (2)")
        self.ax21.plot(ts, m_t_exact[:, 0], "-k", linewidth=1)
        self.ax21.plot(ts, m_t_exact[:, 1], "-k", linewidth=1)
        self.ax21.set_title("Evolution of m_t")
        self.ax21.legend(handles=[m_t1, m_t2])

        # density plot (PDE)
        self.plots["p_rho"] = self.ax22.plot([], label="Particle")[0]
        self.plots["rho"] = self.ax22.plot(a_grid, rho_t[0], "--r", linewidth=1, label="PDE")[0]
        self.ax22.set_ylim(0, 4)
        self.ax22.set_title("Rho density")
        self.ax22.legend(handles=[self.plots["rho"], self.plots["p_rho"]])
        self.ax22.set_xlabel("Age a (s)")

        # Interaction plot
        self.ax31.plot()
        self.ax31.set_title("∫ρdt = 1 verification")
        self.ax31.plot(ts, en_cons, "--k", linewidth=0.5)

        self.ax32.plot()
        self.ax32.set_title("Interaction parameter")
        (x_t1,) = self.ax32.plot(ts, X, "-r", linewidth=0.5, label="Particule")
        (x_t2,) = self.ax32.plot(ts, x_t, "-k", linewidth=0.5, label="PDE")
        self.ax32.legend(handles=[x_t1, x_t2])

        return tuple(self.plots.values())

    def calculate_hist(self, i):
        hist, bins = np.histogram(ages[:, i], bins=100, density=True)
        bins = (bins[1:] + bins[:-1]) / 2
        # w = 2
        return bins, hist  # moving_average(hist, w)

    def animate(self, i):
        t = dt * i
        # Scatter
        self.plots["title"].set_text(f"M points at time {t:.2f}s")
        self.plots["scat"].set_offsets(M[i, :, 0:2])

        # Particule rho
        bins, hist = self.calculate_hist(i)
        self.plots["p_rho"].set_data(bins, hist)
        # self.plots["p_rho_smooth"].set_data(bins[:-4], moving_average(hist, 5))

        # PDE rho
        self.plots["rho"].set_data(a_grid, rho_t[i])
        return tuple(self.plots.values())


# Scatter plot
lim = 20
pl = AnimatedPlot(xlim=lim, ylim=lim)
anim_int = int(1 / dt / 10)  # Want every 10ms
interval = 10

ani = animation.FuncAnimation(
    pl.fig,
    func=pl.animate,
    frames=range(0, len(M), anim_int),
    init_func=pl.init_plot,
    interval=20,
    blit=True,
)

plt.show()