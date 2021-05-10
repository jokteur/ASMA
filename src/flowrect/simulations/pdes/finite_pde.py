import numpy as np
import time
from numba import jit, prange

from ..util import f_SRM

# Not used
@jit(nopython=True, cache=True)
def _fast_pde(
    time_end, dt, N, a_grid_size, exp_a, Gamma, c, tau, lambda_kappa, I_ext, I_ext_time, interaction
):
    """"""
    steps = int(time_end / dt)
    dim = Gamma.shape[0]

    # Init vectors
    ts = np.linspace(0, time_end, steps)
    rho_t = np.zeros((steps, a_grid_size))
    rho2_t = np.zeros((steps, a_grid_size))
    m_t = np.zeros((steps, dim))
    x_t = np.zeros(steps)
    noise = np.random.rand(steps)

    A = np.zeros(steps)
    S = np.zeros((steps, a_grid_size))
    Abar = np.zeros(steps)

    # Vector of indices that goes from a=0 to a=a_cutoff
    a_indices = np.arange(a_grid_size)
    # This vector is used to build a matrix of indices that are used in the S(t,a) fct
    # Only the lower part of the matrix will be used
    a_idx_matrix = -a_indices.reshape((a_grid_size, 1)) + a_indices.reshape((1, a_grid_size))

    rho_t[0, 0] = 1 / dt
    # interaction = J from our equations
    J = interaction
    da = dt

    # Initial step
    x_fixed = I_ext if I_ext_time == 0 else 0
    m_t_sum = np.sum(exp_a * m_t[0], axis=1)
    f = f_SRM(m_t_sum + x_t[0], tau=tau, c=c)

    for s in range(1, steps):
        x_fixed = I_ext if I_ext_time < dt * s else 0

        m_t[s] = m_t[s - 1] + dt * np.sum(
            (Gamma - (1 - exp_a) * m_t[s - 1]).T * f * rho_t[s - 1] * da, axis=1
        )
        x_t[s] = x_t[s - 1] + dt * (
            -lambda_kappa * x_t[s - 1]
            + lambda_kappa * (np.sum(f * rho_t[s - 1] * da) * J + x_fixed)
        )
        m_t_sum = np.sum(exp_a * m_t[s], axis=1)
        f = f_SRM(m_t_sum + x_t[s], tau=tau, c=c)

        # Copy previous density
        rho_t[s] = rho_t[s - 1]

        S_f_idx = s + a_idx_matrix  # t_n - a_i + a_k'
        # For s < a_cutoff, we do not want to look past ages greater than the current time
        # of the simulation
        for a in range(min(s, a_grid_size)):
            idx = S_f_idx[a, :a]
            m_t_sum = np.sum(exp_a[:a] * m_t[idx], axis=1)
            S[s, a] = np.exp(-np.sum(f_SRM(m_t_sum + x_t[idx])) * da)

        # Once S has been determined, we can calculate Abar
        intensity = np.clip(f * dt, 0, 1)  # Limit intensity to 1

        S_int_vec = (1 - S[s]) * rho_t[s]
        S_sum = np.sum(S_int_vec)
        Abar[s] = np.sum(rho_t[s] * intensity)
        if S_sum > 0:
            correction_factor = np.sum(S_int_vec * f) / S_sum  # da can be simplified
            Abar[s] += correction_factor * (1 - np.sum(rho_t[s] * da))

        # Calculate the activity A
        p = N * Abar[s] * dt
        p = 1 if p > 1 else p
        K = np.random.binomial(N, p)
        A[s] = 1 / N * K

        # Mass loss on each cell
        intensity = np.clip(f * dt, 0, 1)  # Limit transfer
        mass_transfer = rho_t[s] * intensity
        rho_t[s] -= mass_transfer
        lass_cell_mass = rho_t[s, -1]  # Last cell necessarely spikes

        # Linear transport
        rho_t[s, 1:] = rho_t[s, :-1]

        # Mass insertion
        rho_t[s, 0] = A[s] * dt

    return ts, rho_t, m_t, x_t, A, Abar, S


def FR_finite_fluctuations(
    time_end,
    dt,
    Lambda,
    Gamma,
    c,
    lambda_kappa,
    I_ext=0,
    I_ext_time=0,
    interaction=0,
    N=500,
    tau=1,
    a_cutoff=5,
    epsilon=1e-8,
):
    if isinstance(Gamma, (float, int)):
        Gamma = [Gamma]
    if isinstance(Lambda, (float, int)):
        Lambda = [Lambda]

    Gamma = np.array(Gamma)
    Lambda = np.array(Lambda)

    dim = Gamma.shape[0]

    # Need dt = da
    a_grid_size = int(a_cutoff / dt)
    a_grid = np.linspace(0, a_cutoff, a_grid_size)
    a_d_grid = np.vstack((a_grid,) * dim).T

    # Shape must be in order: len, d, d
    exp_a = np.exp(-Lambda * a_d_grid)

    # Simulation
    ts, rho_t, m_t, x_t, A, Abar, S = _fast_pde(
        time_end,
        dt,
        N,
        a_grid_size,
        exp_a,
        Gamma,
        c,
        tau,
        lambda_kappa,
        I_ext,
        I_ext_time,
        interaction,
    )
    energy_conservation = np.sum(rho_t * dt, axis=-1)
    activity = rho_t[:, 0]
    return ts, a_grid, rho_t, m_t, x_t, energy_conservation, activity, A, Abar, S


# Trigger compilation
print("Compilation of flowrect with finite size fluctuations")
ret = FR_finite_fluctuations(
    time_end=0.5, dt=0.5, Lambda=[1, 1], Gamma=[-1, -1], c=1, lambda_kappa=1, a_cutoff=1
)