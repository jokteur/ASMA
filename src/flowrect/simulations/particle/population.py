import numpy as np
import time
from numba import jit, prange

from ..util import f_SRM, eta_SRM
from ...accelerated import population as _rust_population


def _simulation_slow(
    time_end,
    dt,
    Gamma,
    Lambda,
    c,
    interaction,
    lambda_kappa,
    I_ext_time,
    I_ext,
    N,
    M0,
):

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

    # If True, jumps by Lambda*Gamma instead of Lambda
    for s in range(1, steps):
        x_fixed = I_ext if I_ext_time < dt * s else 0

        activation = 1 - np.exp(-dt * f_SRM(np.sum(M[s - 1], axis=1) + X[s - 1], c=c)) > noise[s, :]
        decay = ~activation

        M[s, activation] = M[s - 1, activation] + Gamma[activation]
        M[s, decay] = M[s - 1, decay] + dt * (-1 * Lambda[decay] * M[s - 1, decay])

        spikes[s, activation] = 1
        A[s] = 1 / N * np.count_nonzero(activation) / dt
        X[s] = X[s - 1] + dt * (-lambda_kappa * X[s - 1] + lambda_kappa * (J * A[s] + x_fixed))

    return ts, M, spikes, A, X


def population_nomemory(
    time_end,
    dt,
    Lambda,
    Gamma,
    c=1,
    interaction=0,
    lambda_kappa=20,
    I_ext_time=0,
    I_ext=0,
    N=500,
    M0=0,
    use_LambdaGamma=False,
    Gamma_ext=True,
):
    if isinstance(Gamma, (float, int)):
        Gamma = [Gamma]
    if isinstance(Lambda, (float, int)):
        Lambda = [Lambda]
    if isinstance(M0, (float, int)):
        M0 = np.tile((N,), M0)

    M0 = M0.astype(np.float64)

    Gamma = np.array(Gamma)
    Lambda = np.array(Lambda)

    if use_LambdaGamma:
        Gamma = Gamma * Lambda

    steps = int(time_end / dt)

    dim = Gamma.shape[0]

    J = interaction

    Gamma = np.tile(Gamma, (N, 1))
    Lambda = np.tile(Lambda, (N, 1))

    ts = np.linspace(0, time_end, steps)

    m = np.zeros((N, dim))
    m_t = np.zeros((steps, dim))
    n_t = np.zeros((steps, dim, dim))
    X = np.zeros(steps)
    A = np.zeros(steps)
    # M[0] = np.tile(M0, (N, 1))

    # If True, jumps by Lambda*Gamma instead of Lambda
    for s in range(1, steps):
        x_fixed = I_ext if I_ext_time < dt * s else 0

        noise = np.zeros(N)

        activation = 1 - np.exp(-dt * f_SRM(np.sum(m, axis=1) + X[s - 1], c=c)) > noise
        decay = ~activation

        m[activation] += Gamma[activation]
        m[decay] += dt * (-1 * Lambda[decay] * m[decay])

        num_activations = np.count_nonzero(activation)
        if num_activations:
            m_t[s] = np.mean(m[activation])
        else:
            m_t[s] = m_t[s - 1]
        n_t[s] = np.outer(m_t[s], m_t[s])
        A[s] = 1 / N * np.count_nonzero(activation) / dt
        X[s] = X[s - 1] + dt * (-lambda_kappa * X[s - 1] + lambda_kappa * (J * A[s] + x_fixed))

    # return ts, M, spikes, A, X


def population_fast(
    time_end,
    dt,
    Lambda,
    Gamma,
    c=1,
    interaction=0,
    lambda_kappa=20,
    I_ext_time=0,
    I_ext=0,
    N=500,
    M0=0,
    use_LambdaGamma=False,
    Gamma_ext=True,
):
    """
    Simulates an SRM neuron population.


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
    N : int
        size of the population
    M0 : float or numpy array
        initial conditions of the leaky memory
    use_LambdaGamma : bool
        if True, replace Gamma by Gamma .* Lambda
        (to be compatible with previous versions, let it by default to False)
    Gamma_ext : boolean
        obsolete (don't put to false)

    Returns
    -------
    ts : numpy array
        time grid of the simulation
    M_t : numpy array
        the mean value of leaky memory at spike
    N_t : numpy array
        the second moments of leaky memory at spike
    A : numpy array
        activity of the population
    X : numpy array
        interaction of the population
    """
    if isinstance(Gamma, (float, int)):
        Gamma = [Gamma]
    if isinstance(Lambda, (float, int)):
        Lambda = [Lambda]
    if isinstance(M0, (float, int)):
        M0 = np.tile((N,), M0)

    M0 = M0.astype(np.float64)

    Gamma = np.array(Gamma)
    Lambda = np.array(Lambda)

    if use_LambdaGamma:
        Gamma = Gamma * Lambda

    _rust_population(
        time_end, dt, Gamma, Lambda, c, interaction, lambda_kappa, I_ext_time, I_ext, N, M0
    )


def population(
    time_end,
    dt,
    Lambda,
    Gamma,
    c=1,
    interaction=0,
    lambda_kappa=20,
    I_ext_time=0,
    I_ext=0,
    N=500,
    M0=0,
    use_LambdaGamma=False,
    Gamma_ext=True,
):
    """
    Simulates an SRM neuron population.


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
    N : int
        size of the population
    M0 : float or numpy array
        initial conditions of the leaky memory
    use_LambdaGamma : bool
        if True, replace Gamma by Gamma .* Lambda
        (to be compatible with previous versions, let it by default to False)
    Gamma_ext : boolean
        obsolete (don't put to false)

    Returns
    -------
    ts : numpy array
        time grid of the simulation
    M : numpy array
        the leaky memory variables
    A : numpy array
        activity of the population
    X : numpy array
        interaction of the population
    """

    if isinstance(Gamma, (float, int)):
        Gamma = [Gamma]
    if isinstance(Lambda, (float, int)):
        Lambda = [Lambda]

    # if isinstance(M0, (float, int)):
    #     M0 = np.repeat()

    Gamma = np.array(Gamma)
    Lambda = np.array(Lambda)

    if use_LambdaGamma:
        Gamma = Gamma * Lambda

    return _simulation_slow(
        time_end,
        dt,
        Gamma,
        Lambda,
        c,
        interaction,
        lambda_kappa,
        I_ext_time,
        I_ext,
        N,
        M0,
    )
