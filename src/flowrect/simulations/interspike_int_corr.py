import time

import numpy as np
import scipy.stats
from . import particle_individual, flow_rectification_2nd_order
from .util import f_SRM

import matplotlib.pyplot as plt


def ISIC_particle(transient_time=5, K=2000, alpha=0.99, **params):
    """
    Calculates the interspike interval correlation (ISIC) of an SRM neuron model.

    Parameters
    ----------
    transient_time : float
        time the simulation takes to settle to a stable point
    K : int
        target number of spikes that will be accounted in the ISIC
    alpha : float
        confidence level (in normal distr)
    **params : dict
        All the parameters that are plugged in particle_individual().
        In this simulation, make sure that at time_end, the simulation
        has converged to a stable point. The function here will add the
        correct amount of time to the simulation to collect enough spikes.

    Returns
    -------
    interspike interval correlation at t-> infty
    """

    params["interaction"] = 0.0
    params["use_LambdaGamma"] = True

    # Just for convenience
    time_end = params["time_end"]
    I_ext_time = params["I_ext_time"]
    dt = params["dt"]

    # Find the number of spikes since I_ext_time
    ts, M, spikes, X = particle_individual(**params)

    # Let the transient settle
    I_ext_time_idx = int(dt * (I_ext_time + transient_time))
    I_ext_time += transient_time

    num_spikes = np.count_nonzero(spikes[I_ext_time_idx:])
    # Approximate Interspike Interval
    ISI = (time_end + transient_time - I_ext_time) / num_spikes

    print(f"ISI found: {ISI}s")

    # Add the time to have at least K spikes after I_ext_time (crude estimation)
    params["time_end"] += int(3 * K * ISI)
    steps = int(params["time_end"] / dt)
    ts, M, spikes, X = particle_individual(**params)

    num_spikes = np.count_nonzero(spikes[I_ext_time_idx:])
    assert num_spikes > K + 1, "The ISI estimation was not successful. Please increase time_end."

    # Find the idx, such that there are K + 2 spikes after that
    idx = steps - np.argmax(np.cumsum(spikes[::-1]) > K + 1) - 1

    (indices,) = np.where(spikes[idx:] == 1)
    indices += idx

    spike_times = ts[indices]

    ISI = spike_times[1:] - spike_times[:-1]
    Tbar = np.mean(ISI)
    T_n = ISI[:K]
    T_nplusone = ISI[1:]

    corr = np.mean((T_n - Tbar) * (T_nplusone - Tbar))
    ISIC = corr / np.var(ISI)

    # Fisher's r-to-z transformation
    z = 1 / 2 * np.log((1 + ISIC) / (1 - ISIC))
    zalpha = scipy.stats.norm(0, 1).ppf(1 - (1 - alpha) / 2)
    zl = z - zalpha * np.sqrt(1 / (K - 3))
    zr = z + zalpha * np.sqrt(1 / (K - 3))

    ISIC_left = np.tanh(zl)
    ISIC_right = np.tanh(zr)

    return ts, spikes, ISIC, ISIC_left, ISIC_right, T_n, T_nplusone, Tbar


def ISIC_2nd_order(tau_c=10, dtau=1e-2, N=500, **params):
    """
    Calculates the Interspike Interval correlation from a 2nd order flow
    rectification.

    The end of the simulation must be in a stable rate such that the ISIC
    can be calculated.

    Parameters
    ----------
    K : int
        target number of spikes that will be accounted in the ISIC
    tau_c : float
        tau cutoff
    dtau : float
        spacing of tau
    N : int
        number of sampling points
    **params : dict
        All the parameters that are plugged in particle_individual().
        In this simulation, make sure that at time_end, the simulation
        has converged to a stable point. The function here will add the
        correct amount of time to the simulation to collect enough spikes.
    """

    Lambda = np.array(params["Lambda"])
    Gamma = np.array(params["Gamma"])
    dim = Lambda.shape[0]

    ts, _, _, m_t, n_t, x_t, _, A_t = flow_rectification_2nd_order(**params)

    m_t = m_t[-1]
    n_t = n_t[-1]
    x_t = x_t[-1]

    V_t = n_t - np.outer(m_t, m_t)

    samples = np.random.multivariate_normal(m_t, V_t, N)
    h = params["I_ext"]

    tau = np.linspace(0, tau_c, int(tau_c / dtau))
    tau_size = len(tau)

    exp_Ltau = np.exp(-np.einsum("i,j->ji", Lambda, tau))  # (tau_size, dim)

    theta = 0
    int_up = 0
    int_bottom = 0

    for i, sample in enumerate(samples):
        # Calculate phi(tau, x)
        exp_tau_x = exp_Ltau @ sample  # (tau_size, )
        f = f_SRM(exp_tau_x + x_t, c=params["c"])
        phi = (1 - np.exp(-f * dtau)) * np.exp(-(np.cumsum(f) - f) * dtau)

        conservation = np.sum(phi)
        moment1 = tau * phi

        # ti = time.time()
        # # Calculate phi(tau', [...]x + LambdaGamma) = phi_next
        # # We want that the shape of phi_next be (tau_size, tau_size').
        # # The prime indicates the innermost variable in the integral (tau'),
        # # and the tau indicates the outermost variable in the integral.
        # # In the next einsum, by convention i means on tau, and j means on tau'
        # # k designates the dimension along Lambda or Gamma
        # x = np.einsum("ik,k->ik", exp_Ltau, sample)  # OK (tau_size, dim)
        # x += Lambda * Gamma

        # exp_tau_tauprime = np.einsum("jk,ik->ij", exp_Ltau, x)  # OK (tau_size, tau_size')
        # f = f_SRM(exp_tau_tauprime + h)
        # phi_next = f * np.exp(-np.cumsum(f * dtau, axis=1))  # axis=1 designates tau' dimension

        # # Make sure to multiply phi by tau' on the innermost variable
        # moment_next = np.einsum("ij,j->ij", phi_next, tau)
        # print(np.sum(moment * np.sum(moment_next, axis=1)) * dtau ** 2)
        # print(time.time() - ti)

        # ti = time.time()
        # Manual calculation of integral
        inner_int = np.zeros(tau_size)

        for j in range(tau_size):
            x = np.exp(-Lambda * tau[j]) * sample + Lambda * Gamma
            exp_tau_x = exp_Ltau @ x  # OK (tau_size,)
            f = f_SRM(exp_tau_x + x_t, c=params["c"])
            phi_next = (1 - np.exp(-f * dtau)) * np.exp(-(np.cumsum(f) - f) * dtau)
            inner_int[j] = np.sum(tau * phi_next)

        int_up += np.sum(moment1 * inner_int)
        int_bottom += np.sum(tau * moment1)  # Ok
        theta += np.sum(tau * phi)  # Ok

        theta_correct = theta / (i + 1)
        int1 = int_up / (i + 1)
        int2 = int_bottom / (i + 1)
        var = int2 - theta_correct ** 2
        corr = (int1 - theta_correct ** 2) / (int2 - theta_correct ** 2)
        print(
            f"Sample {i}, Theta: {theta_correct:.3f}, Var: {var:.5f}, corr: {corr:.3f}, conservation: {conservation:.3f}"
        )
        # print(f"Sample {i}, {conservation}")