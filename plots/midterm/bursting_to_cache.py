import time
import copy
import os
from multiprocessing import Pool

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.animation as animation

from core.util import calculate_age, calculate_mt, eta_SRM
from core.population import simulation
from core.pde import flow_rectification
from core.quasi_renewal import QR

path = "C:\\Users\\jokte\\Dropbox\\EPFL\\PDM\\Midterm"

dt = 0.5e-2
# Take similar as in article
params = dict(
    dt=dt,
    Lambda=np.array([28.0, 8.0, 1.0]),
    Gamma=np.array([-3.5, 3.0, -1.0]),
    c=10,
    lambda_kappa=2,
    I_ext=0.5,
    I_ext_time=20,
    interaction=0,
)


# Trigger compilations
t = time.time()
params["dt"] = 1e-2
params["time_end"] = 5
ts_QR, A_QR, cutoff = QR(**params)
print(f"Compilation time {time.time() - t:.2f}s")

# Correct parameters
t = time.time()
params["dt"] = dt
params["time_end"] = 30

# External input range
total_points = 30
I_end = 2.5
I_vec = np.linspace(0, I_end, total_points)


def simulate_all(i, dt):
    t = time.time()
    params["I_ext"] = I_vec[i]
    params["dt"] = dt

    t = time.time()
    ts_QR, A_QR, cutoff = QR(**params)
    print(f"I_ext = {I_vec[i]:.2f}, done in {time.time() -t:.2f}s")
    Ainf_QR = A_QR[-1]

    return Ainf_QR


def simulate1(i):
    return simulate_all(i, 1e-2)


def simulate2(i):
    return simulate_all(i, 1e-3)


if __name__ == "__main__":
    p = Pool(8)
    t = time.time()

    Ainf_QR = p.map(simulate1, range(len(I_vec)))
    # p.join()
    Ainf_QR = p.map(simulate2, range(len(I_vec)))

    print(Ainf_QR)
    np.save(os.path.join(path, "Ainf_QR"), np.array(Ainf_QR))