import numpy as np
from numba import jit
from numba.extending import overload


def moving_average(x, w):
    return np.convolve(x, np.ones(w), "valid") / w


@jit(nopython=True)
def calculate_age(array):
    ret = np.zeros(array.shape)
    for i in range(array.shape[0]):
        count = 0
        for j in range(array.shape[1]):
            count += 1
            if array[i, j]:
                count = 0
                ret[i, j] = 0
                continue
            ret[i, j] = count
    return ret


calculate_age(np.array([[0]]))


@jit(nopython=True)
def replace_NaN(array):
    prev_value = np.zeros(array.shape[1])
    for i in range(len(array)):
        if np.isnan(array[i, 0]):
            array[i] = prev_value
        else:
            prev_value = array[i]

    return array


def calculate_mt(M, spikes):
    m_t = np.copy(M)
    mask = spikes == 0

    m_t[mask] = np.NaN
    m_t = np.nanmean(m_t, axis=1)

    m_t = replace_NaN(m_t)
    return m_t


def calculate_nt(m_t):
    return np.einsum("ai,aj->aij", m_t, m_t)


calculate_mt(np.zeros((10, 5, 2)), np.zeros((10, 5)))


@jit(nopython=True, nogil=True)
def f_SRM(x, c=1, Delta=1, theta=0):
    return np.exp(x / Delta) * c


@jit(nopython=True)
def eta_SRM(x, Gamma, Lambda, tau=1):
    ret = np.zeros(len(x))
    for d in range(len(Gamma)):
        ret += Gamma[d] * np.exp(-Lambda[d] * x)
    return ret


@jit(nopython=True)
def eta_SRM_no_vector(x, Gamma, Lambda, tau=1):
    ret = 0
    for d in range(len(Gamma)):
        ret += Gamma[d] * np.exp(-Lambda[d] * x)
    return ret


@jit(nopython=True)
def kappa_interaction(t, lambda_kappa, strength):
    return strength * np.exp(-lambda_kappa * t)


def h_exp_update(h_t, A_t, I_ext, lambda_kappa, dt, J):
    return h_t + lambda_kappa * dt * (J * A_t + I_ext - h_t)


def h_erlang_update(h_t, k_t, A_t, I_ext, lambda_kappa, dt, J):
    h = h_t + dt * lambda_kappa * (-h_t + k_t)
    k = k_t + dt * lambda_kappa * (-k_t + J * A_t + I_ext)
    return h, k


@overload(np.clip)
def np_clip(a, a_min, a_max, out=None):
    def np_clip_impl(a, a_min, a_max, out=None):
        if out is None:
            out = np.empty_like(a)
        for i in range(len(a)):
            if a[i] < a_min:
                out[i] = a_min
            elif a[i] > a_max:
                out[i] = a_max
            else:
                out[i] = a[i]
        return out

    return np_clip_impl
