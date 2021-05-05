import numpy as np
import time
from numba import jit, prange

from ..util import f_SRM, eta_SRM


@jit(nopython=True, nogil=True)
def _simulation_fast(time_end, dt, Gamma, Lambda, M0=0, tau=1, c=1, tolerance=1e-3):
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

    steps = int(time_end / dt)

    dim = Gamma.shape[0]

    noise = np.random.rand(steps)
    spikes = np.zeros((steps,))
    ts = np.linspace(0, time_end, steps)

    M = np.zeros((steps, dim))

    for t in range(1, steps):
        if dt * f_SRM(np.dot(M[t - 1], Gamma), tau=tau, c=c) > noise[t]:
            M[t] = M[t - 1] + 1
            spikes[t] = 1
        else:
            M[t] = M[t - 1] - dt * np.multiply(Lambda, M[t - 1])

    return ts, M, spikes


@jit(nopython=True, parallel=True)
def simulation_ND(time_end, dt, Gamma, Lambda, M0=0, tau=1, c=1, tolerance=1e-3, N=1):
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

    dim = Gamma.shape[0]
    steps = int(time_end / dt)
    spikes = np.zeros((N, steps))

    Ms = np.zeros((N, steps, dim))

    for i in prange(N):
        ts, M, spike = _simulation_fast(time_end, dt, Gamma, Lambda)
        Ms[i] = M
        spikes[i] = spike
    return Ms, spikes


def simulation_ND_slow(time_end, dt, Gamma, Lambda, M0=0, tau=1, c=1, tolerance=1e-3, N=1):
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

    if isinstance(Gamma, (float, int)):
        Gamma = [Gamma]
    if isinstance(Lambda, (float, int)):
        Lambda = [Lambda]

    steps = int(time_end / dt)
    Gamma = np.array(Gamma)
    Lambda = np.array(Lambda)

    dim = Gamma.shape[0]

    Gamma = np.tile(Gamma, (N, 1))
    Lambda = np.tile(Lambda, (N, 1))

    noise = np.random.rand(steps, N)
    spikes = np.zeros((steps, N))
    ts = np.linspace(0, time_end, steps)

    M = np.zeros((steps, N, dim))
    # M[0] = np.tile(M0, (N, 1))

    for t in range(1, steps):
        activation = dt * f_SRM(np.sum(M[t - 1] * Gamma, axis=1), tau=tau, c=c) > noise[t, :]
        decay = ~activation

        M[t, activation] = M[t - 1, activation] + 1
        spikes[t, activation] = 1

        M[t, decay] = M[t - 1, decay] + dt * (-1 * Lambda[decay] * M[t - 1, decay])

    return ts, M, spikes


def individual(time_end, dt, Gamma, Lambda, M0=0, tau=1, c=1, tolerance=1e-3):
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

    Gamma = np.array(Gamma)
    Lambda = np.array(Lambda)

    return _simulation_fast(time_end, dt, Gamma, Lambda, M0, tau, c)