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
    I_ext=2.5,
    I_ext_time=15,
    interaction=0.0,
    use_LambdaGamma=True,
)
params["Gamma"] = params["Gamma"] / params["Lambda"]

params_p = copy.deepcopy(params)
# params_p["dt"] = 1e-3

plot_params = dict(
    dpi=300,
    savepath="git\\plots\\final\\results\\adaptation",
    save=True,
    usetex=True,
    noshow=True,
    font_size=21,
)
p_params = copy.deepcopy(params)
p_params["dt"] = 1e-2

# Plot the kernel with spike example
kernel_params = copy.deepcopy(params)
kernel_params["theta"] = 0
kernel_params["time_end"] = 10
kernel_params["dt"] = 1e-2
plot_kernel(
    kernel_params, time_kernel=2, N=10, savename=f"adaptation_kernel.png", seed=111, **plot_params
)
# Three plots with A_inf plot
for I_ext in [0.5, 1.5, 2.5]:
    params["I_ext"] = I_ext
    params_p["I_ext"] = I_ext
    plot_activity(
        params,
        params_p=params_p,
        N=25000,
        w=7,
        time_before_input=3,
        plot_QR=True,
        plot_H=False,
        ylim=15,
        I_ylim=4,
        a_cutoff=10,
        savename=f"A_t_{I_ext}.png",
        **plot_params,
    )

plt.show()