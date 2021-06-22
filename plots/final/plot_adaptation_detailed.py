import numpy as np
import copy
import matplotlib.pyplot as plt

from activity import plot_activity
from kernel import plot_kernel

# Simulation parameters
params = dict(
    dt=0.5 * 1e-2,
    time_end=20,
    Lambda=np.array([12.3, 2.5]),
    Gamma=np.array([-8.0, -2.5]),
    c=30,
    lambda_kappa=50,
    Delta=2,
    theta=0,
    base_I=0.5,
    I_ext=3,
    I_ext_time=15,
    interaction=0.0,
    use_LambdaGamma=True,
)
params["Gamma"] = params["Gamma"] / params["Lambda"]
plot_params = dict(
    dpi=300,
    savepath="git\\plots\\final\\results\\adaptation",
    save=True,
    usetex=True,
    noshow=False,
    font_size=16,
)

params_p = copy.deepcopy(params)
# params_p["dt"] = 1e-3

# Detailed activity plot
plot_activity(
    params,
    params_p=params_p,
    N=25000,
    time_before_input=3,
    plot_QR=True,
    plot_H=False,
    w=7,
    inset=[
        [0.49, 0.49, 0.5, 0.5],
        [params["I_ext_time"] - 0.1, params["I_ext_time"] + 1, 3.46, 8.2],
    ],
    ylim=15,
    I_ylim=4,
    a_cutoff=7,
    savename=f"A_t_detailed.png",
    **plot_params,
)
