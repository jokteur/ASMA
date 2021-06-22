import numpy as np
import copy
import matplotlib.pyplot as plt

from activity import plot_activity
from kernel import plot_kernel


# Simulation parameters
params = dict(
    dt=0.5 * 1e-2,
    time_end=40,
    Lambda=np.array([28.0, 8.0, 1.0]),
    Gamma=np.array([-3.5, 3.0, -1.0]),
    c=20,
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

plot_params = dict(
    dpi=300,
    savepath="git\\plots\\final\\results\\bursting",
    save=True,
    usetex=True,
    noshow=True,
    font_size=21,
)

# Plot the kernel with spike example
kernel_params = copy.deepcopy(params)
kernel_params["theta"] = 0
kernel_params["time_end"] = 10
kernel_params["dt"] = 1e-2
plot_kernel(
    kernel_params, time_kernel=3, N=10, savename=f"bursting_kernel.png", seed=97, **plot_params
)
# Three plots with A_inf plot
for I_ext in [0.5, 1, 1.5]:
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