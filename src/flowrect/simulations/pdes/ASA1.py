import numpy as np
import time
from numba import jit, prange
import matplotlib.pyplot as plt

from ..util import f_SRM, h_exp_update, h_erlang_update


def ASA1(
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
    a_cutoff=5,
    use_LambdaGamma=True,
    m_t0=0,
    rho0=0,
    h_t0=0,
    kappa_type="exp",
):
    """"""
    if isinstance(Gamma, (float, int)):
        Gamma = [Gamma]
    if isinstance(Lambda, (float, int)):
        Lambda = [Lambda]

    Gamma = np.array(Gamma)
    Lambda = np.array(Lambda)

    if use_LambdaGamma:
        Gamma = Gamma * Lambda

    dim = Gamma.shape[0]

    # Need dt = da
    a_grid_size = int(a_cutoff / dt)
    a_grid = np.linspace(0, a_cutoff, a_grid_size)
    a_d_grid = np.vstack((a_grid,) * dim).T

    # Shape must be in order: len, d, d
    exp_La = np.exp(-Lambda * a_d_grid)

    steps = int(time_end / dt)
    dim = Gamma.shape[0]

    # Init vectors
    ts = np.linspace(0, time_end, steps)
    A_t = np.zeros(steps)
    rho_t = np.zeros((steps, a_grid_size))
    m_t = np.zeros((steps, dim))
    h_t = np.zeros(steps)
    k_t = np.zeros(steps)

    m_t[0] = m_t0
    rho_t[0, 0] = 1 / dt
    h_t[0] = h_t0
    if isinstance(rho0, np.ndarray):
        rho_t[0] = rho0

    # interaction = J from our equations
    J = interaction
    da = dt

    c = c * np.exp(-theta / Delta)
    f_SRM_args = dict(c=c, Delta=Delta, theta=theta)

    a_iplusone = np.exp(-Lambda * dt)

    # a_iplusone = 1

    h_args = dict(J=J, lambda_kappa=lambda_kappa, dt=dt)

    # @jit(nopython=True, cache=True)

    def optimized(rho_t, m_t, h_t):
        for s in range(0, steps - 1):
            x_fixed = I_ext + base_I if I_ext_time < dt * (s + 1) else base_I

            num_age_steps = min(s, a_grid_size)

            # A_t = rho_t[s, 0]
            # if A_t < 1e-5:
            #     A_t = 1e-5
            #     print("Low activity at step", s, ":", A_t)

            indices = s - np.arange(num_age_steps)
            m0 = m_t0 * np.ones((a_grid_size - num_age_steps, dim))
            m = np.concatenate((m_t[indices], m0), axis=0)
            exp_m_t = exp_La * m

            f = f_SRM(np.sum(exp_m_t, axis=1) + h_t[s], c=c, Delta=Delta, theta=theta)

            # firing_prob = np.zeros(a_grid_size)
            # for i in range(a_grid_size):
            #     firing_prob[i] = f[i] if i < 1 else 1
            # firing_prob = np.clip(f * da, 0, 1)
            firing_prob = 1 - np.exp(-f * da)

            A_t[s] = np.sum(firing_prob * rho_t[s])

            if A_t[s] < 1e-6:
                A_t[s] = 1e-6

            m_t[s + 1] = (
                np.sum((a_iplusone * exp_m_t + Gamma).T * firing_prob * rho_t[s], axis=1) / A_t[s]
            )

            if kappa_type == "erlang":
                h_t[s + 1], k_t[s + 1] = h_erlang_update(h_t[s], k_t[s], A_t[s], x_fixed, **h_args)
            else:
                h_t[s + 1] = h_exp_update(h_t[s], A_t[s], x_fixed, **h_args)
            # h_t[s + 1] = h_t[s] + dt * lambda_kappa * (-h_t[s] + (A_t[s] * J + x_fixed))

            # Mass loss
            mass_transfer = rho_t[s] * firing_prob
            # rho_t[s + 1] -= mass_transfer
            lass_cell_mass = rho_t[s, -1]  # Last cell necessarely spikes

            # Linear transport
            rho_t[s + 1, 1:] = rho_t[s, :-1] - mass_transfer[:-1]

            # Mass insertion
            rho_t[s + 1, 0] = np.sum(mass_transfer) + lass_cell_mass

        return rho_t, m_t, h_t

    rho_t, m_t, h_t = optimized(rho_t, m_t, h_t)
    A_t[-1] = rho_t[-1, 0]

    mass_conservation = np.sum(rho_t * dt, axis=-1)
    activity = rho_t[:, 0]
    return ts, a_grid, rho_t, m_t, h_t, mass_conservation, A_t
