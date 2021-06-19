import numpy as np
import time

from numba import jit, prange

from .util import f_SRM, eta_SRM, kappa_interaction


def find_cutoff(begin, end, dt, Lambda, Gamma, epsilon_c):
    """
    Find recursively tau_c within the precision of dt
    """
    t_space = np.linspace(begin, end, 20)
    tau_c = 0

    exp = np.exp(eta_SRM(t_space, Gamma, Lambda))
    condition = exp > -epsilon_c + 1

    idx = np.argmax(condition > 0)

    if idx:  # Found a True following a False
        if t_space[idx] - t_space[idx - 1] > dt / 2:  # Not precise enough
            new_begin = t_space[idx - 1]
            new_end = t_space[idx]
            return find_cutoff(new_begin, new_end, dt, Lambda, Gamma, epsilon_c)
        return t_space[idx]
    else:
        if condition[0]:
            return find_cutoff(begin / 2, (begin + end) / 2, dt, Lambda, Gamma, epsilon_c)
        return find_cutoff(begin / 2, end * 2, dt, Lambda, Gamma, epsilon_c)


@jit(nopython=True, cache=True)
def _fast_QR(
    time_end,
    dt,
    Lambda,
    Gamma,
    c,
    Delta,
    theta,
    lambda_kappa,
    I_ext,
    I_ext_time,
    interaction,
    tau_c,
    use_LambdaGamma,
):
    """"""
    steps = int(time_end / dt)
    A = np.zeros(steps)

    ts = np.linspace(0, time_end, steps)

    # Number of steps between t - tau_c and t
    K = int(tau_c / dt)

    ks = np.arange(K)
    prev_m = np.zeros(K)
    h_int = 0

    J = interaction

    # Init conditions
    # At t = 0, all neurons spike. Before, A=0
    A[0] = 1 / dt

    # Fixed vectors
    eta = 1 / Delta * eta_SRM(np.linspace(tau_c, 0, K), Gamma, Lambda)
    y = np.exp(1 / Delta * eta_SRM(np.linspace(2 * tau_c, 0, 2 * K), Gamma, Lambda)) - 1

    # Use conventions as in article
    for s in range(1, steps):
        x_fixed = I_ext if I_ext_time < dt * s else 0
        t = dt * s
        grid_t = np.linspace(t - tau_c, t, K)

        # Build x vector as described by eq. 39 in article
        x = np.zeros(K)
        m = np.zeros(K)

        # Eta is η(t - t'), so the grid on k must be defined on [τ_c, 0]

        # Update x
        for k in range(K):
            yA = 0
            for j in range(k, k + K):
                idx = s - 1 - 2 * K + j  # Corresponding real index
                if idx >= 0:  # Take into account only positive indices
                    yA += y[j] * A[idx]
            x[k] = np.exp(eta[k] + yA * dt - theta)

        # Update kappa
        h_int = h_int + dt * lambda_kappa * (-h_int + (J * A[s - 1] + x_fixed))

        m[K - 1] = A[s - 1] * dt
        for k in list(range(0, K - 1))[::-1]:
            m[k] = prev_m[k + 1] * np.exp(-dt * c * np.exp(h_int) * x[k + 1])

        # Update A
        A[s] = c * np.exp(h_int / Delta) * (1 + np.dot(x - 1, m))

        # Prepare m for next loop
        prev_m = m

    return ts, A


def quasi_renewal(
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
    epsilon_c=1e-2,
    use_LambdaGamma=False,
):
    """
    Simulates the quasi-renewal approximation on an SRM neuron population model.


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

    epsilon_c : float
        degree of precision to which tau_cutoff must be found

    use_LambdaGamma : bool
        if True, replace Gamma by Gamma .* Lambda
        (to be compatible with previous versions, let it by default to False)

    Returns
    -------
    ts : numpy array
        time grid of the simulation

    A : numpy array
        activity of the PDE

    tau_c : float
        value of the cutoff calculated as explained in [Naud, Gerstner, 2012]
    """
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

    tau_c = 7

    return (
        *_fast_QR(
            time_end,
            dt,
            Lambda,
            Gamma,
            c,
            Delta,
            theta,
            lambda_kappa,
            I_ext,
            I_ext_time,
            interaction,
            tau_c,
            use_LambdaGamma,
        ),
        tau_c,
    )


# Trigger compilation
res = quasi_renewal(1, 1e-2, [1.0, 1.0], [-1.0, -1.0], 1, 1, 1, 0.5, 0)
