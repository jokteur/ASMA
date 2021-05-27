import numpy as np
import time
from numba import jit, prange

from ..util import f_SRM, eta_SRM


@jit(nopython=True, nogil=True)
def _simulation_fast(
    time_end,
    dt,
    Gamma,
    Lambda,
    c,
    interaction,
    lambda_kappa,
    I_ext_time,
    I_ext,
    M0,
):

    steps = int(time_end / dt)

    dim = Gamma.shape[0]

    J = interaction

    noise = np.random.rand(steps)  # N indep. Poisson variables
    spikes = np.zeros(steps)
    ts = np.linspace(0, time_end, steps)

    M = np.zeros((steps, dim))
    X = np.zeros(steps)
    M[0] = M0

    # If True, jumps by Lambda*Gamma instead of Lambda
    for s in range(1, steps):
        x_fixed = I_ext if I_ext_time < dt * s else 0
        activation = 1 - np.exp(-dt * f_SRM(np.sum(M[s - 1]) + X[s - 1], c=c)) > noise[s]

        if activation:
            M[s] = M[s - 1] + Gamma
            spikes[s] = 1
        else:
            M[s] = M[s - 1] + dt * (-1 * Lambda * M[s - 1])

        X[s] = X[s - 1] + dt * (-lambda_kappa * X[s - 1] + lambda_kappa * (x_fixed))

    return ts, M, spikes, X


def individual(
    time_end,
    dt,
    Lambda,
    Gamma,
    c=1,
    interaction=0,
    lambda_kappa=20,
    I_ext_time=0,
    I_ext=0,
    M0=0,
    use_LambdaGamma=False,
):
    """
    Simulates an SRM neuron.


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
    M0 : float or numpy array
        initial conditions of the leaky memory
    use_LambdaGamma : bool
        if True, replace Gamma by Gamma .* Lambda
        (to be compatible with previous versions, let it by default to False)

    Returns
    -------
    ts : numpy array
        time grid of the simulation
    M : numpy array
        the leaky memory variable
    X : numpy array
        interaction of the neuron
    """

    if isinstance(Gamma, (float, int)):
        Gamma = [Gamma]
    if isinstance(Lambda, (float, int)):
        Lambda = [Lambda]

    Gamma = np.array(Gamma)
    Lambda = np.array(Lambda)

    if use_LambdaGamma:
        Gamma = Lambda * Gamma

    return _simulation_fast(
        time_end,
        dt,
        Gamma,
        Lambda,
        c,
        interaction,
        lambda_kappa,
        I_ext_time,
        I_ext,
        M0,
    )
