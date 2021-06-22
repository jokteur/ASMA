import time

import matplotlib.pyplot as plt
import numpy as np
from kernel import plot_kernel

params = dict(
    dt=0.5 * 1e-2,
    time_end=10,
    Lambda=np.array([28.0, 8.0, 1.0]),
    Gamma=np.array([-3.5, 3.0, -1.0]),
    # Lambda=[1.0, 5.5],
    # Gamma=[-4.0, -1.0],
    c=30,
    lambda_kappa=50,
    Delta=1,
    theta=0,
    base_I=0.0,
    I_ext=0,
    I_ext_time=10,
    interaction=0.0,
    use_LambdaGamma=True,
)
params["Gamma"] = params["Gamma"] / params["Lambda"]

for i in range(70, 80):
    plot_kernel(
        params,
        time_kernel=3,
        N=10,
        savepath="git\\plots\\final\\results\\bursting",
        savename=f"bursting_kernel.png",
        seed=i,
        dpi=300,
        save=False,
        usetex=False,
        font_size=14,
        noshow=True,
    )

plt.show()