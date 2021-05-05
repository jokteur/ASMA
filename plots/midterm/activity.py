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
from flowrect.simulations import particle_population
from flowrect.simulations import flow_rectification
from flowrect.simulations import quasi_renewal

save = False
save_path = ""
save_name = "activity.pdf"


def moving_average(x, w):
    return np.convolve(x, np.ones(w), "valid") / w


dt = 1e-2
N = 25000
I_ext = 2.5
# Take similar as in article
time_end = 40
params = dict(
    time_end=time_end,
    dt=dt,
    Lambda=[1.0, 5.5],
    Gamma=[-4.0, -1.0],
    # Lambda=np.array([28.0, 8.0, 1.0]),
    # Gamma=np.array([-3.5, 3.0, -1.0]),
    c=1,
    lambda_kappa=2,
    I_ext=I_ext,
    I_ext_time=20,
    interaction=0,
)

print(f"QR approximation")
QR_params = copy.copy(params)
QR_params["dt"] = 1e-2
t = time.time()
ts_QR, A_QR, cutoff = quasi_renewal(**QR_params)
print(f"{time.time() - t:.2f}s")


print(f"Particle simulation")
t = time.time()
ts, M, spikes, A, X = particle_population(**params, N=N, Gamma_ext=True)
m_t = calculate_mt(M, spikes)
A_av = moving_average(A, 50)
# m_ts = np.zeros(m_t.T.shape)
# w = 50
# m_ts[: -w + 1, 0] = moving_average(m_t.T[:, 0], w)
# m_ts[: -w + 1, 1] = moving_average(m_t.T[:, 1], w)
# m_ts[-w + 1 :, :] = m_ts[-w, :]
print(f"{time.time() - t:.2f}")

print(f"Flow rectification approximation")
t = time.time()
ts, a_grid, rho_t, m_t_exact, x_t, en_cons, A_t = flow_rectification(a_cutoff=10, **params)
print(f"{time.time() - t:.2f}s")

I_ext_vec = np.concatenate((np.zeros(int(len(ts) / 2)), I_ext * np.ones(int(len(ts) / 2))))

from_t = int(5 / dt)

fig = plt.figure(figsize=(8, 8))

gs = gridspec.GridSpec(2, 1, height_ratios=[5, 1])

ax1 = plt.subplot(gs[0])

# fig.suptitle(r"Activity response to a step input ($\Delta t=10^{-2}$)")
(A_1,) = ax1.plot(ts[from_t:], A[from_t:], "--k", linewidth=0.5, label=f"Particle ({N=})")
(A_2,) = ax1.plot(ts[from_t : len(A_av)], A_av[from_t:], "--r", label="P. rolling av.")
(A_3,) = ax1.plot(ts[from_t:], A_t[from_t:], "-.g", linewidth=1.5, label="PDE")
(A_4,) = ax1.plot(ts_QR[from_t:], A_QR[from_t:], "-b", linewidth=1.5, label="QR")
ax1.set_ylim(0, 1.5)
ax1.set_ylabel(r"$A(t)$ (Hz)")
ax1.legend(handles=[A_1, A_2, A_3, A_4])


ax2 = plt.subplot(gs[1], sharex=ax1)
ax2.plot(ts[from_t:], I_ext_vec[from_t:], "-k")
ax2.set_xlabel(r"$t$ (s)")
ax2.set_xlim(5, time_end)
ax2.set_ylabel(r"$I_0$ (A)")

if save:
    fig.savefig(os.path.join(save_path, save_name), transparent=True)
plt.show()