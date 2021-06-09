import time
import copy
import os
from multiprocessing import Pool

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.animation import FuncAnimation
import matplotlib.animation as animation
from multiprocessing import Pool

import flowrect

from flowrect.simulations.util import (
    calculate_age,
    calculate_mt,
    eta_SRM,
    moving_average,
    calculate_nt,
)
from flowrect.simulations import (
    particle_population,
    particle_population_fast,
    flow_rectification,
    ASA1,
    quasi_renewal,
    flow_rectification_2nd_order,
)

N = 500
a_cutoff = 7
# Take similar as in article
params = dict(
    time_end=40,
    dt=1e-2,
    # Lambda=np.array([33.3, 2.5, 1.0]),
    # Gamma=np.array([-8.0, -1.0, -0.5]),
    Lambda=np.array([10.3, 2.5]),
    Gamma=np.array([-5.0, -1.0]),
    c=5,
    Delta=1,
    lambda_kappa=2,
    I_ext=2.5,
    I_ext_time=20,
    interaction=0.0,
)

t = time.time()
ts_P, M_P, spikes_P, A_P, X_P = particle_population(N=N, use_LambdaGamma=True, **params)
m_tp = calculate_mt(M_P, spikes_P)
n_tp = calculate_nt(m_tp)
print(f"Particle population done in {time.time() - t:.2f}s")

# Simulations
t = time.time()
pre_input_idx = int(int(params["time_end"] / params["dt"]) / 2) - 1

# (
#     ts,
#     a_grid,
#     rho_t_2nd,
#     m_t_2nd,
#     n_t_2nd,
#     x_t_2nd,
#     en_cons_2nd,
#     A_t_2nd,
# ) = flow_rectification_2nd_order(
#     m_t0=m_tp[pre_input_idx], n_t0=n_tp[pre_input_idx], a_cutoff=a_cutoff, **params
# )
# print(f"2nd order flow rectification done in {time.time() - t:.2f}s")

params["dt"] = 1e-2
t = time.time()
ts_ASA1, a_grid_ASA1, rho_t_ASA1, m_t_ASA1, x_t_ASA1, en_cons_ASA1, A_ASA1 = ASA1(
    m_t0=m_tp[pre_input_idx], a_cutoff=a_cutoff, use_LambdaGamma=True, **params
)
print(f"ASA1 done in {time.time() - t:.2f}s")

params["dt"] = 1e-2
t = time.time()
ts_FR, a_grid, rho_t, m_t, x_t, en_cons, A_t = flow_rectification(
    m_t0=m_tp[pre_input_idx], a_cutoff=a_cutoff, use_LambdaGamma=True, **params
)
print(f"Flow rectification done in {time.time() - t:.2f}s")


# t = time.time()
# ts_QR, A_QR, tau_c = quasi_renewal(use_LambdaGamma=True, **params)
# print(f"QR done in {time.time() - t:.2f}s")

# plt.figure()
# plt.plot(ts, n_tp[:, 0, 0], "--k")
# plt.plot(ts, n_tp[:, 0, 1], "--r")
# plt.plot(ts, n_tp[:, 1, 0], "--g")
# plt.plot(ts, n_tp[:, 1, 1], "--b")
# plt.plot(ts, n_t_2nd[:, 0, 0], "-k", label="00")
# plt.plot(ts, n_t_2nd[:, 0, 1], "-r", label="01")
# plt.plot(ts, n_t_2nd[:, 1, 0], "-g", label="10")
# plt.plot(ts, n_t_2nd[:, 1, 1], "-b", label="11")
# plt.legend()


plt.figure()
plt.plot(ts_P, A_P, "--k")

# Particle activity
A_av = moving_average(A_P, 50)
missing = len(A_P) - len(A_av)
left = int(missing / 2)
right = len(A_P) - int(np.ceil(missing / 2))
plt.plot(ts_P[left:right], A_av, "-g", label="Particle")
# 2nd order
# plt.plot(ts, A_t_2nd, "-.b", label="2nd")
# 1st order
plt.plot(ts_FR, A_t, "-.r", label="FR")
plt.plot(ts_ASA1, A_ASA1, "-.b", label="ASA1")
# QR
# plt.plot(ts, A_QR, "-.m", label="QR")

plt.ylim(0, 7)
plt.legend()

ts = ts_P
plt.figure()
plt.plot(ts_P, m_tp[:, 0], "--k", label="Particle")
plt.plot(ts_P, m_tp[:, 1], "--k")
plt.plot(ts, m_t_ASA1[:, 0], "-b", label="2nd")
plt.plot(ts, m_t_ASA1[:, 1], "-b")
plt.plot(ts, m_t[:, 0], "-r", label="1st")
plt.plot(ts, m_t[:, 1], "-r")
plt.legend()
plt.show()