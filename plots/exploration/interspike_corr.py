import time
import copy
import os
from multiprocessing import Pool

import numpy as np
import matplotlib.pyplot as plt

import flowrect
from flowrect.simulations import ISIC_particle, ISIC_2nd_order, particle_population
from flowrect.simulations.util import calculate_mt, calculate_nt

params = dict(
    time_end=40,
    dt=1e-2,
    Lambda=np.array([15.3, 2.5]),
    Gamma=np.array([-5.0, 0.5]),
    c=1,
    lambda_kappa=2,
    I_ext=2.0,
    I_ext_time=20,
    interaction=0.0,
)

I_vec = np.arange(0, 10, 0.2)

N = 500
_, M_P, spikes_P, _, _ = particle_population(N=N, use_LambdaGamma=True, **params)
m_tp = calculate_mt(M_P, spikes_P)
n_tp = calculate_nt(m_tp)
pre_input_idx = int(int(params["time_end"] / params["dt"]) / 2) - 1

_, _, ISIC, left, right, T_n, T_nplusone, Tbar = ISIC_particle(K=25000, **params)
ISIC_2nd_order(n_t0=n_tp[pre_input_idx], m_t0=m_tp[pre_input_idx], **params)
print(left, ISIC, right, Tbar)


# def multiproc(I):
#     cparams = copy.copy(params)
#     cparams["I_ext"] = I
#     _, _, ISIC, left, right, _, _, _ = ISIC_particle(K=25000, **cparams)

#     return ISIC, left, right


# if __name__ == "__main__":
#     p = Pool(12)
#     t = time.time()
#     res = p.map(multiproc, I_vec)

#     ISICS = []
#     ISIC_lefts = []
#     ISIC_rights = []
#     i = 0
#     for el in res:
#         ISIC, left, right = el
#         ISICS.append(ISIC)
#         ISIC_lefts.append(ISIC - left)
#         ISIC_rights.append(right - ISIC)

#     plt.errorbar(I_vec, ISICS, fmt="kx", yerr=[ISIC_lefts, ISIC_rights])
#     plt.xlabel(r"I_{ext}")
#     plt.ylim(-1, 1)
#     plt.ylabel(r"Correlation")
#     plt.show()