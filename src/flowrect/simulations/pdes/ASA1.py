import numpy as np
import time
from numba import jit, prange

from ..util import f_SRM


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
    I_ext_time=0,
    I_ext=0,
    a_cutoff=5,
    use_LambdaGamma=True,
    m_t0=0,
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
    rho_t = np.zeros((steps, a_grid_size))
    m_t = np.zeros((steps, dim))
    m_t[0] = m_t0
    h_t = np.zeros(steps)

    rho_t[0, 0] = 1 / dt
    # interaction = J from our equations
    J = interaction
    da = dt

    f_SRM_args = dict(c=c, Delta=Delta, theta=theta)

    # Initial step
    x_fixed = I_ext if I_ext_time == 0 else 0
    m_t_sum = np.sum(exp_La * m_t[0], axis=1)
    f = f_SRM(m_t_sum + h_t[0], c=c)

    g = np.zeros(steps)

    # a_iplusone = np.exp(-Lambda * dt)
    a_iplusone = 1

    for s in range(0, steps - 1):
        x_fixed = I_ext if I_ext_time < dt * s else 0

        num_age_steps = min(s, a_grid_size)
        exp_m_t = np.zeros((a_grid_size, dim))
        decay_m_t = np.zeros((a_grid_size, dim))

        A_t = rho_t[s, 0]
        if A_t < 1e-5:
            A_t = 1e-5
            print(f"Low activity at step {s}, {A_t=}")

        for i in range(num_age_steps):
            exp_m_t[i] = exp_La[i] * m_t[s - i]

        f = f_SRM(np.sum(exp_m_t, axis=1) + h_t[s], **f_SRM_args)

        firing_prob = np.clip(f * da, 0, 1)

        m_t[s + 1] = np.sum((a_iplusone * exp_m_t + Gamma).T * firing_prob * rho_t[s], axis=1) / A_t

        h_t[s + 1] = h_t[s] + dt * lambda_kappa * (
            -h_t[s] + (np.sum(f * rho_t[s]) * da * J + x_fixed)
        )

        # Mass loss
        mass_transfer = rho_t[s] * firing_prob
        # rho_t[s + 1] -= mass_transfer
        lass_cell_mass = rho_t[s, -1]  # Last cell necessarely spikes

        # Linear transport
        rho_t[s + 1, 1:] = rho_t[s, :-1] - mass_transfer[:-1]

        # Mass insertion
        rho_t[s + 1, 0] = np.sum(mass_transfer) + lass_cell_mass

    mass_conservation = np.sum(rho_t * dt, axis=-1)
    activity = rho_t[:, 0]
    return ts, a_grid, rho_t, m_t, h_t, mass_conservation, activity
