import time
import copy
import os
from multiprocessing import Pool

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.animation as animation

import flowrect
from flowrect.simulations.util import calculate_age, calculate_mt, eta_SRM
from flowrect.simulations import (
    particle_population,
    flow_rectification,
    quasi_renewal,
)

save = False
save_path = ""
save_name = "Ainf_burst.pdf"

# For the burst simulation, QR results have been cached
burst = True
burst_cache_path = ""

dt = 1e-2
N = 25000
# Take similar as in article
params = dict(
    dt=dt,
    # Lambda=[33.3, 2.5],
    # Gamma=[-8.0, -1.0],
    Lambda=np.array([28.0, 8.0, 1.0]),
    Gamma=np.array([-3.5, 3.0, -1.0]),
    c=10,
    lambda_kappa=2,
    I_ext=0.5,
    I_ext_time=20,
    interaction=0,
)

# Trigger compilations
t = time.time()
params["dt"] = 1e-2
params["time_end"] = 5
ts_QR, A_QR, cutoff = quasi_renewal(**params)
ts, a_grid, rho_t, m_t, x_t, en_cons, A = flow_rectification(**params)
print(f"Compilation time {time.time() - t:.2f}s")

# Correct parameters
t = time.time()
params["dt"] = dt
params["time_end"] = 30

# External input range
total_points = 30
I_end = 2.5
I_vec = np.linspace(0, I_end, total_points)
num_sim = 10
I_vec_sim = np.linspace(0, I_end, num_sim)


def simulate_pdes(i):
    t = time.time()
    params["I_ext"] = I_vec[i]
    params["dt"] = dt

    Ainf_QR = None
    if not burst:
        ts_QR, A_QR, cutoff = quasi_renewal(**params)
        Ainf_QR = A_QR[-1]

    cparams = copy.copy(params)
    cparams["dt"] = 1e-3
    ts_PDE, a_grid, rho_t, m_t, X_PDE, en_cons, A_PDE = flow_rectification(a_cutoff=7, **cparams)

    Ainf_PDE = A_PDE[-1]

    print(f"I_ext = {I_vec[i]:.2f}, done in {time.time() -t:.2f}s")
    return Ainf_PDE, Ainf_QR


def simulate_particle(i):
    t = time.time()
    Ainf_P = None
    Ainf_std_P = None
    if i % int(total_points / num_sim) == 0:
        cparams = copy.copy(params)
        cparams["dt"] = 1e-2
        cparams["I_ext"] = I_vec[i]
        ts_P, M, spikes, A_P, X_P = simulation(N=N, Gamma_ext=True, **cparams)

        # Take the last seconds in activity
        last_seconds_idx = len(A_P) - 1 - int(1 / dt * 5)
        last_A_P = A_P[last_seconds_idx:]
        Ainf_P = np.mean(last_A_P)
        Ainf_std_P = np.std(last_A_P)
    print(f"Particle I_ext = {I_vec[i]:.2f}, done in {time.time() -t:.2f}s")
    return Ainf_P, Ainf_std_P


if __name__ == "__main__":
    p = Pool(8)
    t = time.time()
    res = p.map(simulate_pdes, range(len(I_vec)))
    res2 = p.map(simulate_particle, range(len(I_vec)))

    Ainf_PDE = []
    Ainf_QR = []
    Ainf_P = []
    Ainf_std_P = []

    for A_PDE, A_QR in res:
        Ainf_PDE.append(A_PDE)
        Ainf_QR.append(A_QR)

    for A_P, A_std_P in res2:
        if A_P:
            Ainf_P.append(A_P)
            Ainf_std_P.append(A_std_P)

    if burst:
        Ainf_QR = np.load(os.path.join(burst_cache_path, "Ainf_QR.npy"), allow_pickle=True)

    print(Ainf_P, Ainf_std_P)
    # plt.rcParams.update(
    #     {"text.usetex": True, "font.family": "sans-serif", "font.sans-serif": ["Helvetica"]}
    # )
    # # for Palatino and other serif fonts use:
    # plt.rcParams.update(
    #     {
    #         "text.usetex": True,
    #         "font.family": "serif",
    #         "font.serif": ["Palatino"],
    #     }
    # )

    plt.figure()
    plt.title("Steps responses for bursting neurons")
    plt.plot(I_vec, Ainf_PDE, "-g", label="Flow rectification")
    plt.plot(I_vec, Ainf_QR, "-b", label="QR approximation")
    plt.errorbar(I_vec_sim, Ainf_P, Ainf_std_P, fmt=".k", capsize=2.0, label=f"Particles ({N=})")
    plt.xlabel(r"$I_0$ (A)")
    plt.ylabel(r"$A_{\infty}$ (Hz)")
    plt.legend()

    if save:
        plt.savefig(os.path.join(save_path, save_name), transparent=True)

    plt.figure()
    plt.title(r"$\eta$ kernel")
    plt.xlabel(r"$t$ (s)")
    plt.ylabel(r"intensity (a.u)")
    t = np.linspace(0, 10, 1000)
    eta = eta_SRM(t, np.array(params["Gamma"]), np.array(params["Lambda"]))
    plt.plot(t, eta)
    plt.show()