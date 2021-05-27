import numpy as np
import time
from numba import jit, prange

from ..util import f_SRM

# Not used
@jit(nopython=True, cache=True)
def _fast_pde(
    time_end, dt, a_grid_size, exp_a, Gamma, c, lambda_kappa, I_ext, I_ext_time, interaction, m_t0=0
):
    """"""
    steps = int(time_end / dt)
    dim = Gamma.shape[0]

    # Init vectors
    ts = np.linspace(0, time_end, steps)
    rho_t = np.zeros((steps, a_grid_size))
    m_t = np.zeros((steps, dim))
    m_t[0] = m_t0
    x_t = np.zeros(steps)

    rho_t[0, 0] = 1 / dt
    # interaction = J from our equations
    J = interaction
    da = dt

    # Initial step
    x_fixed = I_ext if I_ext_time == 0 else 0
    m_t_sum = np.sum(exp_a * m_t[0], axis=1)
    f = f_SRM(m_t_sum + x_t[0], c=c)

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
        # m_t_sum = np.sum(exp_a * m_ts[s], axis=1)
        f = f_SRM(m_t_sum + x_t[s], c=c)

        rho_t[s] = rho_t[s - 1]
        # Mass loss
        intensity = np.clip(f * dt, 0, 1)  # Limit transfer
        mass_transfer = rho_t[s] * intensity
        rho_t[s] -= mass_transfer
        lass_cell_mass = rho_t[s, -1]  # Last cell necessarely spikes

        # Linear transport
        rho_t[s, 1:] = rho_t[s, :-1]

        # Mass insertion
        rho_t[s, 0] = np.sum(mass_transfer) + lass_cell_mass

    return ts, rho_t, m_t, x_t


def flow_rectification(
    time_end,
    dt,
    Lambda,
    Gamma,
    c,
    interaction,
    lambda_kappa,
    I_ext,
    I_ext_time,
    a_cutoff=5,
    use_LambdaGamma=False,
    m_t0=0,
):
    """
    Simulates a population density equation approximation of an infinite SRM
    neuron population.


    Parameters
    ----------
    time_end : float
        simulation until time_end

    dt : float
        time step size

    Lambda : (d,) numpy array
        decay matrix

    Gamma : (d,) numpy array
        jump size

    c : float
        base firing rate

    interaction : float
        strength of the self interaction (variable J in equations)

    lambda_kappa : float
        decay parameter in the self interaction kernel kappa

    I_ext_time : float
        time at which a constant current is injected in the population

    I_ext : float
        intensity of the constant current

    a_cutoff : float
        in the integration, max age a that is considered. Anything greater than
        a_cutoff will not be considered.

    use_LambdaGamma : bool
        if True, replace Gamma by Gamma .* Lambda
        (to be compatible with previous versions, let it by default to False)

    Returns
    -------
    ts : numpy array
        time grid of the simulation

    a_grid : numpy array
        age grid (from 0 to a_cutoff)

    rho_t : numpy array
        population probability density of time

    m_t : numpy array
        mean time at spike of leaky memory variable

    x_t : numpy array
        self interaction of the population

    mass_conservation : numpy array
        mass conservation over time of the PDE

    activity : numpy array
        activity of the PDE
    """
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

    if use_LambdaGamma:
        Gamma = Gamma * Lambda

    # Simulation
    ts, rho_t, m_t, x_t = _fast_pde(
        time_end,
        dt,
        a_grid_size,
        exp_a,
        Gamma,
        c,
        lambda_kappa,
        I_ext,
        I_ext_time,
        interaction,
        m_t0,
    )

    mass_conservation = np.sum(rho_t * dt, axis=-1)
    activity = rho_t[:, 0]
    return ts, a_grid, rho_t, m_t, x_t, mass_conservation, activity