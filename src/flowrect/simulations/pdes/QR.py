import numpy as np
import time

from numba import jit, prange
import matplotlib.pyplot as plt
from matplotlib import gridspec

from ..util import (
    f_SRM,
    eta_SRM,
    eta_SRM_no_vector,
    kappa_interaction,
    h_exp_update,
    h_erlang_update,
)
from ..QR import find_cutoff


def quasi_renewal_pde(
    time_end,
    dt,
    Lambda,
    Gamma,
    c=1,
    Delta=1,
    theta=0,
    interaction=0,
    lambda_kappa=20,
    base_I=0,
    I_ext_time=0,
    I_ext=0,
    epsilon_c=1e-2,
    use_LambdaGamma=False,
    a_cutoff=5,
    A_t0=0,
    rho0=0,
    h_t0=0,
    kappa_type="exp",
):
    """"""
    # Check if model will not explode
    if np.sum(Gamma) > 0:
        print(f"Model will explode with {Gamma=}. Cannot performe QR.")
        return [], [], 0

    Gamma = np.array(Gamma)
    Lambda = np.array(Lambda)

    if use_LambdaGamma:
        Gamma = Gamma * Lambda
    # Find cutoff tau_c
    tau_c = find_cutoff(0, 100, dt, Lambda, Gamma, epsilon_c)
    tau_c = np.round(tau_c, decimals=int(-np.log10(dt)))

    # print(f"Calculated tau_c: {tau_c}")
    tau_c = a_cutoff

    # Need dt = da
    a_grid_size = int(tau_c / dt)
    a_grid = np.linspace(0, tau_c, a_grid_size)

    steps = int(time_end / dt)
    dim = Gamma.shape[0]

    # Init vectors
    ts = np.linspace(0, time_end, steps)
    rho_t = np.zeros((steps, a_grid_size))
    A_t = np.zeros(steps)
    h_t = np.zeros(steps)
    k_t = np.zeros(steps)

    rho_t[0, 0] = 1 / dt
    h_t[0] = h_t0
    # A_t[0] = rho_t[0, 0]
    # if A_t0:
    #     A_t[0] = A_t0
    if isinstance(rho0, np.ndarray):
        rho_t[0] = rho0

    K = a_grid_size
    da = dt
    J = interaction
    c = (c * 1.0) * np.exp(-theta / Delta)

    @jit(nopython=True, cache=True)
    def optimized(rho_t, h_t, k_t, A_t):
        Ks = np.arange(K)
        exp_eta = np.exp(1 / Delta * eta_SRM(dt * Ks, Gamma, Lambda))
        y = np.exp(1 / Delta * eta_SRM(np.linspace(0, 2 * tau_c, 2 * (K + 1)), Gamma, Lambda)) - 1
        # x = eta_SRM(np.linspace(0, 2 * tau_c, 2 * K), Gamma, Lambda)

        for n in range(0, steps - 1):
            x_fixed = I_ext + base_I if I_ext_time < dt * (n + 1) else base_I

            # Calculate f~(t|t - a)
            # f[0] = f~(t_n|t_n), f[1] = f~(t_n | t_{n-1}), ..
            f = np.zeros(K)
            t_n = n * dt

            sup = n - Ks
            inf = n - Ks - K
            integral = np.zeros(K)
            # int2 = np.zeros(K)
            for k in range(K):
                t_s_idx = np.arange(max(0, inf[k]), sup[k])

                idx = k + np.arange(len(t_s_idx))
                integral[k] = np.sum(y[idx][::-1] * A_t[t_s_idx] * dt)

            f = c * exp_eta * np.exp(1 / Delta * h_t[n] + integral)

            # firing_prob = np.clip(f * da, 0, 1)
            firing_prob = 1 - np.exp(-f * da)

            A_t[n] = np.sum(firing_prob * rho_t[n])

            h_t[n + 1] = h_t[n] + lambda_kappa * dt * (J * A_t[n] + x_fixed - h_t[n])
            # Next step
            # h_t[n + 1] = h_t[n] + dt * lambda_kappa * (-h_t[n] + (A_t[n] * J + x_fixed))

            # Mass loss
            mass_transfer = rho_t[n] * firing_prob
            # rho_t[n + 1] -= mass_transfer
            lass_cell_mass = rho_t[n, -1]  # Last cell necessarely spikes

            # Linear transport
            rho_t[n + 1, 1:] = rho_t[n, :-1] - mass_transfer[:-1]

            # Mass insertion
            rho_t[n + 1, 0] = np.sum(mass_transfer) + lass_cell_mass

        return rho_t, h_t, A_t

    rho_t, h_t, A_t = optimized(rho_t, h_t, k_t, A_t)

    mass_conservation = np.sum(rho_t * dt, axis=-1)
    activity = rho_t[:, 0]

    return ts, a_grid, rho_t, h_t, mass_conservation, activity