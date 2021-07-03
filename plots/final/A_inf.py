import time
import copy
import os
from multiprocessing import Pool

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.animation as animation

import flowrect
from flowrect.simulations.util import calculate_age, calculate_mt, eta_SRM, moving_average
from flowrect.simulations import (
    particle_population_nomemory,
    ASA1,
    quasi_renewal_pde,
)


def _simulate_QR(args):
    I_ext, params, a_cutoff = args
    t = time.time()
    params["I_ext"] = I_ext
    ts_QR, _, _, _, _, A_QR = quasi_renewal_pde(a_cutoff=a_cutoff, **params)
    print(f"I_ext = {I_ext:.2f}, QR done in {time.time() -t:.2f}s")
    Ainf_QR = A_QR[-1]

    return Ainf_QR


def _simulate_pdes(args):
    I_ext, params, a_cutoff = args
    t = time.time()
    params["I_ext"] = I_ext
    ts_PDE, _, _, _, _, _, A_PDE = ASA1(a_cutoff=a_cutoff, **params)
    print(f"I_ext = {I_ext:.2f}, PDE done in {time.time() -t:.2f}s")
    Ainf_PDE = A_PDE[-1]

    return Ainf_PDE


def _simulate_particle(args):
    I_ext, params, N, w = args
    t = time.time()
    cparams = copy.copy(params)
    cparams["I_ext"] = I_ext
    ts_P, A_P, _, _ = particle_population_nomemory(N=N, **cparams)
    dt = ts_P[1] - ts_P[0]

    new_A = A_P
    if w:
        new_A = moving_average(A_P, w)

    # Take the last seconds in activity
    last_seconds_idx = len(new_A) - 1 - int(1 / dt * 5)
    last_A_P = new_A[last_seconds_idx:]
    Ainf_P = np.mean(last_A_P)
    Ainf_std_P = np.std(last_A_P)
    print(f"I_ext = {I_ext:.2f}, Particle done in {time.time() -t:.2f}s")
    return Ainf_P, Ainf_std_P


def plot_A_inf(
    params,
    N,
    pool,
    num_sim=30,
    num_p_sim=10,
    I_end=5,
    w=None,
    cache_path="cache",
    cache_suppl_name="",
    QR_params=None,
    p_params=None,
    a_cutoff=7,
    savepath="",
    savename="",
    save=False,
    usetex=False,
    figsize=(8, 8),
    dpi=None,
    title=None,
    font_family="serif",
    font_size="12",
    noshow=False,
):
    if usetex:
        plt.rc("text", usetex=True)
        plt.rc("font", family=font_family, size=font_size)

    # External input range
    I_vec = np.linspace(0, I_end, num_sim)
    I_vec_p = np.linspace(0, I_end, num_p_sim)

    # Parameters
    if not p_params:
        p_params = copy.deepcopy(params)
    if not QR_params:
        QR_params = copy.deepcopy(params)

    # Check for cache

    t = time.time()
    params_multi_pde = [(I_vec[i], params, a_cutoff) for i in range(len(I_vec))]
    pde_res = pool.map(_simulate_pdes, params_multi_pde)

    params_multi_QR = [(I_vec[i], QR_params, a_cutoff) for i in range(len(I_vec))]
    QR_res = pool.map(_simulate_QR, params_multi_QR)

    params_multi_p = [(I_vec_p[i], p_params, N, w) for i in range(len(I_vec_p))]
    p_res = pool.map(_simulate_particle, params_multi_p)

    Ainf_PDE = []
    Ainf_QR = []
    Ainf_P = []
    Ainf_std_P = []

    # Fetch results
    for A_PDE in pde_res:
        Ainf_PDE.append(A_PDE)
    for A_QR in QR_res:
        Ainf_QR.append(A_QR)
    for A_P, A_std_P in p_res:
        if A_P:
            Ainf_P.append(A_P)
            Ainf_std_P.append(A_std_P)

    plt.figure()
    if title:
        plt.title(title)

    plt.plot(I_vec, Ainf_PDE, "-r", label="ASMA")
    plt.plot(I_vec, Ainf_QR, "-b", label="QR")
    plt.errorbar(I_vec_p, Ainf_P, Ainf_std_P, fmt=".k", capsize=2.0, label=r"$25\cdot10^3$ neurons")
    plt.xlabel(r"$I_1$ (A)")
    plt.ylabel(r"$A_{\infty}$ (Hz)")
    plt.legend()
    plt.tight_layout()

    if save:
        if dpi:
            plt.savefig(os.path.join(savepath, savename), dpi=dpi, transparent=True)
        else:
            plt.savefig(os.path.join(savepath, savename), transparent=True)
    if not noshow:
        plt.show()