import numpy as np
import os
import time

import flowrect
from flowrect.simulations import particle_population_fast, particle_population
from flowrect.simulations.particle.population import population_nomemory

params = dict(
    time_end=40,
    dt=1e-3,
    Lambda=np.array([15.3, 2.5]),
    Gamma=np.array([-5.0, 0.5]),
    c=1,
    N=500,
    lambda_kappa=2,
    I_ext=2.0,
    I_ext_time=20,
    interaction=0.0,
)
t = time.time()
particle_population_fast(**params)
print(f"Time: {time.time() - t:.2f}")
# t = time.time()
# particle_population(**params)
# print(f"Time: {time.time() - t:.2f}")
t = time.time()
population_nomemory(**params)
print(f"Time: {time.time() - t:.2f}")