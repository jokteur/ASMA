import numpy as np
import time
from numba import jit, prange

from ..util import f_SRM

# Not used
# @jit(nopython=True, cache=True)
def _fast_pde(
    time_end,
    dt,
    a_grid,
    a_grid_size,
    exp_La,
    Lambda,
    Gamma,
    c,
    lambda_kappa,
    I_ext,
    I_ext_time,
    interaction,
):
    """"""
    steps = int(time_end / dt)
    dim = Gamma.shape[0]

    # Init vectors
    ts = np.linspace(0, time_end, steps)
    rho_t = np.zeros((steps, a_grid_size))  # Probability density distribution
    m_t = np.zeros((steps, dim))  # First moment vector
    m_t2 = np.zeros((steps, dim))
    n_t = np.zeros((steps, dim, dim))  # Second moments vector
    V_t = np.zeros((steps, dim, dim))  # Semi-definite covariance matrix
    x_t = np.zeros(steps)  # Interaction parameter

    rho_t[0, 0] = 1 / dt  # All the mass is concentrated at age 0 in the beginning of sim.
    J = interaction  # interaction = J from our equations
    da = dt

    # Precompute values
    Lambda_i_plus_j = np.stack((Lambda,) * dim) + np.stack((Lambda,) * dim).T
    exp_La_ij = np.exp(-np.einsum("ij,k->kij", Lambda_i_plus_j, a_grid))

    # # Initial step
    x_fixed = I_ext if I_ext_time == 0 else 0
    m_t_sum = np.sum(exp_La * m_t[0], axis=1)
    G0 = f_SRM(m_t_sum + x_t[0], c=c)
    f = G0
    G1 = 0
    G2 = 0

    for s in range(1, steps):
        x_fixed = I_ext if I_ext_time < dt * s else 0

        # Basis calculations for the Gaussian moments
        exp_variance = (V_t[s - 1] @ exp_La.T).T  # Makes a (a_grid_size, dim) matrix
        gauss_param = m_t[s - 1] + 0.5 * exp_variance  # (a_grid_size, dim) matrix
        g = c * np.exp(x_t[s - 1] + np.sum(exp_La * gauss_param, axis=1))  # (a_grid_size,) vector
        moment1 = m_t[s - 1] + exp_variance  # (a_grid_size, dim) matrix

        # From first moment (to erase)
        m_t_sum = np.sum(exp_La * m_t2[s - 1], axis=1)
        # m_t_sum = np.sum(exp_a * m_ts[s], axis=1)
        f = f_SRM(m_t_sum + x_t[s - 1], c=c)

        # Gaussian moments
        G0 = g
        G1 = (moment1.T * g).T
        G2 = ((V_t[s - 1] + np.einsum("ij,ik->ijk", moment1, moment1)).T * g).T

        # This values are reused in multiple calculations
        LambdaGamma = Lambda * Gamma

        # Update first moments
        m_t2[s] = m_t2[s - 1] + dt * np.sum(
            ((LambdaGamma - (1 - exp_La) * m_t2[s - 1]).T * f) * rho_t[s - 1] * da, axis=1
        )
        m_t[s] = m_t[s - 1] + dt * np.sum(
            ((exp_La - 1) * G1 + np.einsum("i,j->ji", LambdaGamma, G0)).T * rho_t[s - 1] * da,
            axis=1,
        )

        # Update second moments
        part_0moment = np.einsum("i,j,k->kij", LambdaGamma, LambdaGamma, G0)
        part_1moment_ij = np.einsum("ki,j,ki->kij", exp_La, LambdaGamma, G1)
        part_1moment_ji = np.einsum("kj,i,kj->kij", exp_La, LambdaGamma, G1)
        part_2moment = (exp_La_ij - 1) * G2

        n_t[s] = n_t[s - 1] + dt * np.sum(
            ((part_0moment + part_1moment_ij + part_1moment_ji + part_2moment).T * rho_t[s - 1]).T
            * da,
            axis=0,
        )

        # Update covariance matrix
        V_t[s] = n_t[s] - np.outer(m_t[s], m_t[s])

        # Update self interaction
        x_t[s] = x_t[s - 1] + dt * (
            -lambda_kappa * x_t[s - 1]
            + lambda_kappa * (np.sum(f * rho_t[s - 1] * da) * J + x_fixed)
        )

        rho_t[s] = rho_t[s - 1]
        # Mass loss
        intensity = np.clip(G0 * dt, 0, 1)  # Limit transfer
        mass_transfer = rho_t[s] * intensity
        rho_t[s] -= mass_transfer
        lass_cell_mass = rho_t[s, -1]  # Last cell necessarely spikes

        # Linear transport
        rho_t[s, 1:] = rho_t[s, :-1]

        # Mass insertion
        rho_t[s, 0] = np.sum(mass_transfer) + lass_cell_mass

    return ts, rho_t, m_t, n_t, x_t


def flow_rectification_2nd_order(
    time_end,
    dt,
    Lambda,
    Gamma,
    c,
    lambda_kappa,
    I_ext,
    I_ext_time,
    interaction,
    a_cutoff=5,
):
    """
    Calculates the flow rectification of second order for an exponential firing function.

    Parameters
    ----------
    time_end : float
        Number of seconds of the simulation
    dt : float
        Size of the time step in the simulation
        (recommended values are between 1e-2 and 1e-3)
    Lambda : 1D numpy array
        Parameters of the exponential decay in the eta function
        Lambda[0] = lambda_1 ; Lambda[1] = lambda_2 ; ...
    c : float
        base firing rate (in the function f(u) = c*exp(t))
    lambda_kappa : float
        exponential decay that is in the kappa function
    I_ext : float
        strength of an external constant current (must be specified along with I_ext_time)
    I_ext_time : float
        time (in seconds) at which the external constant current is injected
    interaction : float
        self-interaction strength
    a_cutoff : float
        maximum considered age in the simulation (in seconds)
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
    exp_La = np.exp(-Lambda * a_d_grid)

    # Simulation
    ts, rho_t, m_t, n_t, x_t = _fast_pde(
        time_end,
        dt,
        a_grid,
        a_grid_size,
        exp_La,
        Lambda,
        Gamma,
        c,
        lambda_kappa,
        I_ext,
        I_ext_time,
        interaction,
    )

    energy_conservation = np.sum(rho_t * dt, axis=-1)
    activity = rho_t[:, 0]
    return ts, a_grid, rho_t, m_t, n_t, x_t, energy_conservation, activity