import numpy as np
import copy
import matplotlib.pyplot as plt

from delta_0 import plot_delta_0

# Simulation parameters
N = 5000
params = dict(
    dt=0.5 * 1e-2,
    time_end=20,
    Lambda=np.array([12.3, 2.5]),
    Gamma=np.array([-8.0, -2.5]),
    c=30,
    lambda_kappa=50,
    Delta=2,
    theta=0,
    base_I=0.0,
    I_ext=2.5,
    I_ext_time=15,
    interaction=0.0,
    use_LambdaGamma=True,
)
params["Gamma"] = params["Gamma"] / params["Lambda"]

plot_params = dict(
    dpi=300,
    savepath="git\\plots\\final\\results\\hard_threshold",
    save=True,
    usetex=True,
    noshow=False,
    font_size=14,
)

plot_delta_0(
    params,
    N=25000,
    w=7,
    Deltas=[2, 0.9, 0.1],
    I_exts=[1.5, 1, 0.3],
    time_before_input=2,
    plot_QR=True,
    ylim=6.5,
    a_cutoff=10,
    savename=f"delta_to_zero.png",
    **plot_params,
)