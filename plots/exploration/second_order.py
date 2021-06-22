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
from flowrect.simulations import (
    particle_population,
    flow_rectification,
    flow_rectification_2nd_order,
    ASA1,
    quasi_renewal,
)

from flowrect.simulations.pdes.QR import quasi_renewal_pde

# Plot saving parameters
save = False
save_path = ""
save_name = ""


def moving_average(x, w):
    return np.convolve(x, np.ones(w), "valid") / w


def calculate_hist(ages, i):
    hist, bins = np.histogram(ages[:, i], bins=50, density=True)
    bins = (bins[1:] + bins[:-1]) / 2
    # w = 2
    return bins, hist  # moving_average(hist, w)


# Simulation parameters
N = 5000
dt = 1e-2
params = dict(
    time_end=25,
    dt=dt,
    Lambda=np.array([10.3, 2.5]),
    Gamma=np.array([-2.0, -1.0]),
    c=10,
    Delta=4,
    theta=4,
    lambda_kappa=50,
    I_ext=2.5,
    I_ext_time=15,
    interaction=0.0,
    kappa_type="exp",
)

# params["Gamma"] /= params["Lambda"]

a_cutoff = 10

print(f"Particle simulation")
t = time.time()
# params["dt"] = 1e-3
ts, M, spikes, A, H = particle_population(**params, N=N, Gamma_ext=True, use_LambdaGamma=True)
m_tp = calculate_mt(M, spikes)
before_spike = int(params["I_ext_time"] / dt) - 10
m_t0 = m_tp[before_spike]
h_t0 = H[before_spike]
A_t0 = A[before_spike]

I_ext_vec = np.concatenate(
    (np.zeros(int(len(ts) / 2)), params["I_ext"] * np.ones(int(len(ts) / 2)))
)
ages = calculate_age(spikes.T) * params["dt"]
A_av = moving_average(A, 100)
# params["dt"] = 1e-2
print(f"{time.time() - t:.2f}")

# print(f"Integral")
# t = time.time()
# params["dt"] = 5e-2
# integral_equation(1, 0.1, [1.0, 2.0], [-2.0, -1])
# print("Compiled")
# ts_int, m_t_int, A_t_int, h_t_int = integral_equation(
#     a_cutoff=a_cutoff, use_LambdaGamma=True, **params
# )
# print(A_t_int)
# params["dt"] = 1e-2
# print(f"{time.time() - t:.2f}s")


# print(f"Flow rectification approximation")
# t = time.time()
# # params["dt"] = 1e-3
# ts_1st, a_grid_1st, rho_t_1st, m_t_1st, x_t_1st, en_cons_1st, A_t_1st = flow_rectification(
#     a_cutoff=a_cutoff, use_LambdaGamma=True, **params
# )
# print(f"{time.time() - t:.2f}s")

# print(f"Quasi renewal")
# t = time.time()
# # ts, A_QR, _ = quasi_renewal(use_LambdaGamma=True, **params)
# ts_QR, a_grid_QR, rho_t_QR, h_t_QR, en_cons_QR, A_t_QR = quasi_renewal_pde(
#     **params,
#     a_cutoff=a_cutoff,
#     # rho0=rho_t_1st[before_spike],
#     # h_t0=h_t0,
#     # A_t0=A_t_1st[before_spike],
#     use_LambdaGamma=True,
# )
# print(f"{time.time() - t:.2f}s")


print(f"ASA1")
t = time.time()
(ts_ASA1, a_grid_ASA1, rho_t_ASA1, m_t_ASA1, x_t_ASA1, en_cons_ASA1, A_t_ASA1,) = ASA1(
    a_cutoff=a_cutoff,
    # rho0=rho_t_1st[before_spike],
    # h_t0=h_t0,
    # m_t0=m_t0,
    use_LambdaGamma=True,
    **params,
)
print(f"{time.time() - t:.2f}s")
print(f"ASA1 2")
params["kappa_type"] = "erlang"
t = time.time()
(
    ts_ASA1_2,
    a_grid_ASA1_2,
    rho_t_ASA1_2,
    m_t_ASA1_2,
    x_t_ASA1_2,
    en_cons_ASA1_2,
    A_t_ASA1_2,
) = ASA1(
    a_cutoff=a_cutoff,
    # rho0=rho_t_1st[before_spike],
    # h_t0=h_t0,
    # m_t0=m_t0,
    use_LambdaGamma=True,
    **params,
)
print(f"{time.time() - t:.2f}s")


def energy_conservation_plot(ax):
    ax.plot(ts_ASA1, en_cons_ASA1, "--k")
    # ax.plot(ts_QR, en_cons_QR, "-.b")
    ax.set_title("Energy conservation")
    ax.set_xlabel("time t")


def m_t_plot(ax):
    ax.set_title("m_t")
    ax.plot(ts, m_tp[:, 0], "--k", label="particulaire")
    ax.plot(ts_ASA1, m_t_ASA1[:, 0], "--r", label="ASA1")
    ax.plot(ts_ASA1_2, m_t_ASA1_2[:, 0], "-.m", label="ASA1 2")
    # ax.plot(ts_1st, m_t_1st[:, 0], "-b", label="FR")
    ax.plot(ts, m_tp[:, 1], "--k")
    ax.plot(ts_ASA1, m_t_ASA1[:, 1], "--r")
    ax.plot(ts_ASA1_2, m_t_ASA1_2[:, 1], "-.m")
    # ax.plot(ts_1st, m_t_1st[:, 1], "-b")
    ax.set_ylim(-30, 1)
    ax.legend()


def activity_plot(ax):
    ax.set_title("Activity")
    A_av = moving_average(A, 50)
    ax.plot(ts, A, "--k", label="Particle", alpha=0.9)
    # ax.plot(ts[24:-25], A_av, "--r", label="P. rolling av.")
    # ax.plot(ts_1st, A_t_1st, "-b", linewidth=1.5, label="1st")
    # ax.plot(ts_QR, A_t_QR, "-.m", linewidth=1.5, label="QR")
    ax.plot(ts_ASA1, A_t_ASA1, "-.", linewidth=1.5, label="ASA1")
    ax.plot(ts_ASA1_2, A_t_ASA1_2, "-.", linewidth=1.5, label="ASA1 erlang")
    # ax.plot(ts, A_t_2nd, "-g", linewidth=1.5, label="2nd")
    ax.set_ylim(0, 10)
    ax.legend()


def interaction_plot(ax):
    ax.set_title("h_t")
    ax.plot(ts, H, "--k", label="Particle")
    # ax.plot(ts_1st, x_t_1st, "--b", linewidth=1.5, label="1st")
    ax.plot(ts_ASA1, x_t_ASA1, "-.", linewidth=1.5, label="ASA1 v2")
    ax.legend()


# Animated plots
class AnimatedPlot:
    def __init__(self, xlim=10, ylim=20):
        self.fig = plt.figure(figsize=(15, 8))
        gs = gridspec.GridSpec(2, 3, height_ratios=[2, 1])
        self.fig.suptitle(fr"PDE vs particle simulation $N=${N}")

        self.ax1 = plt.subplot(gs[0])
        self.ax2 = plt.subplot(gs[1])
        self.ax3 = plt.subplot(gs[2])
        self.ax4 = plt.subplot(gs[3])
        self.ax5 = plt.subplot(gs[4])
        self.ax6 = plt.subplot(gs[5])
        self.xlim, self.ylim = xlim, ylim

        energy_conservation_plot(self.ax5)
        m_t_plot(self.ax2)
        activity_plot(self.ax3)
        interaction_plot(self.ax6)
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
        # self.plots["p_rho"] = self.ax1.plot([], [], "-k", label="Particle")[0]
        self.plots["rho_QR"] = self.ax1.plot(
            a_grid_QR, rho_t_QR[0], "--r", linewidth=1, label="QR"
        )[0]
        self.plots["rho_ASA"] = self.ax1.plot(
            a_grid_ASA1, rho_t_ASA1[0], "-b", linewidth=1, label="ASA1"
        )[0]
        # self.plots["S"] = self.ax1.plot(a_grid, S[0], "g", linewidth=1)[0]
        self.ax1.set_ylim(0, 10)
        self.ax1.set_title("Probability density distribution")
        self.ax1.legend(handles=self.plots.values())
        self.ax1.set_xlabel("Age a (s)")
        self.ax1.set_ylabel(r"$\rho_t$")

        self.ax4.plot()
        self.ax4.set_title("External input")
        self.plots["vline"] = self.ax4.plot([], [], "-r", linewidth=1)[0]
        self.ax4.set_ylim(0, 6)
        self.ax4.plot(ts, I_ext_vec, "-k")
        self.ax4.set_ylabel(r"$I^{ext}$ (a.u.)")
        self.ax4.set_xlabel(r"$t$ (s)")

        return tuple(self.plots.values())

    def animate(self, i):
        t = dt * i
        # Scatter
        self.plots["title"].set_text(fr"Time $t=${t:.2f}s")
        # Particule rho
        bins, hist = calculate_hist(ages, i)
        # self.plots["p_rho"].set_data(bins, hist)
        self.plots["vline"].set_data(np.array([t, t]), np.array([0, 6]))

        # PDE rho
        self.plots["rho_QR"].set_data(a_grid_QR, rho_t_QR[i])
        self.plots["rho_ASA"].set_data(a_grid_ASA1, rho_t_ASA1[i])
        # self.plots["S"].set_data(a_grid, S[i])
        return tuple(self.plots.values())


# Scatter plot
lim = 20
pl = AnimatedPlot(xlim=lim, ylim=lim)
anim_int = 4  # Want every 10ms
# print(anim_int)

# ani = animation.FuncAnimation(
#     pl.fig,
#     func=pl.animate,
#     frames=range(0, len(M), anim_int),
#     init_func=pl.init_plot,
#     interval=1,
#     blit=True,
# )

if save:
    ani.save(os.path.join(save_path, save_name))

plt.show()