import numpy as np
import time
from numba import jit, prange

from ..util import f_SRM, eta_SRM

# @jit(nopython=True, nogil=True)
# def f_SRM(x, tau=1, c=1):
#     return np.exp(x / tau) * c


# # @jit(nopython=True)
# def eta_SRM(x, Gamma, Lambda, tau=1):
#     ret = np.zeros(len(x))
#     for d in range(len(Gamma)):
#         ret += Gamma[d] * np.exp(-Lambda[d] * x)
#     return ret


def _simulation_slow(
    time_end,
    dt,
    Gamma,
    Lambda,
    I_ext_time,
    I_ext,
    interaction,
    lambda_kappa,
    M0,
    tau,
    c,
    N,
    Gamma_ext=False,
):
    """
    Parameters
    ----------
    time_end : float
        simulation until time_end
    dt : float
        time step size
    Gamma : (d,) numpy array
        jump size
    Lambda : (d,) numpy array
        decay matrix
    tau : float
        time scale
    M0 : (d,) numpy array
        initial condition
    N : int
        number of simulations
    """

    steps = int(time_end / dt)

    dim = Gamma.shape[0]

    J = interaction

    Gamma = np.tile(Gamma, (N, 1))
    Lambda = np.tile(Lambda, (N, 1))

    noise = np.random.rand(steps, N)  # N indep. Poisson variables
    spikes = np.zeros((steps, N))
    ts = np.linspace(0, time_end, steps)

    M = np.zeros((steps, N, dim))
    X = np.zeros(steps)
    A = np.zeros(steps)
    # M[0] = np.tile(M0, (N, 1))

    if Gamma_ext:
        for t in range(1, steps):
            x_fixed = I_ext if I_ext_time < dt * t else 0

            prob = np.sum(M[t - 1], axis=1) + X[t - 1]

            activation = dt * f_SRM(prob, tau=tau, c=c) > noise[t, :]
            decay = ~activation

            M[t, activation] = M[t - 1, activation] + Gamma[activation]
            M[t, decay] = M[t - 1, decay] + dt * (-1 * Lambda[decay] * M[t - 1, decay])

            spikes[t, activation] = 1
            A[t] = 1 / N * np.count_nonzero(activation) / dt
            X[t] = X[t - 1] + dt * (-lambda_kappa * X[t - 1] + lambda_kappa * (J * A[t] + x_fixed))
    else:
        for t in range(1, steps):
            x_fixed = I_ext if I_ext_time < dt * t else 0

            prob = np.sum(M[t - 1] * Gamma, axis=1) + X[t - 1]

            activation = dt * f_SRM(prob, tau=tau, c=c) > noise[t, :]
            decay = ~activation

            M[t, activation] = M[t - 1, activation] + 1
            M[t, decay] = M[t - 1, decay] + dt * (-1 * Lambda[decay] * M[t - 1, decay])

            spikes[t, activation] = 1
            A[t] = 1 / N * np.count_nonzero(activation) / dt
            X[t] = X[t - 1] + dt * (-lambda_kappa * X[t - 1] + lambda_kappa * (J * A[t] + x_fixed))

    return ts, M, spikes, A, X


def population(
    time_end,
    dt,
    Gamma,
    Lambda,
    I_ext_time=2,
    I_ext=2,
    interaction=1,
    lambda_kappa=20,
    M0=0,
    tau=1,
    c=1,
    N=500,
    Gamma_ext=False,
):
    """
    Parameters
    ----------
    time_end : float
        simulation until time_end
    dt : float
        time step size
    Gamma : (d,) numpy array
        jump size
    Lambda : (d,) numpy array
        decay matrix
    tau : float
        time scale
    M0 : (d,) numpy array
        initial condition
    """

    if isinstance(Gamma, (float, int)):
        Gamma = [Gamma]
    if isinstance(Lambda, (float, int)):
        Lambda = [Lambda]

    # if isinstance(M0, (float, int)):
    #     M0 = np.repeat()

    Gamma = np.array(Gamma)
    Lambda = np.array(Lambda)

    return _simulation_slow(
        time_end,
        dt,
        Gamma,
        Lambda,
        I_ext_time,
        I_ext,
        interaction,
        lambda_kappa,
        M0,
        tau,
        c,
        N,
        Gamma_ext,
    )
