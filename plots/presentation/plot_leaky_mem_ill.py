import numpy as np

from leaky_mem_ill import plot_leaky_memory


# Simulation parameters
params = dict(
    dt=1e-3,
    time_end=1,
    Lambda=np.array([12.3, 3.0, 2.5]),
    Gamma=np.array([-8.0, 1.0, -2.5]),
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

plot_params = dict(
    dpi=300,
    savepath="git\\plots\\presentation\\results",
    savename="leaky_memory.png",
    save=True,
    usetex=True,
    noshow=True,
    font_size=16,
)


plot_leaky_memory(params, 10, **plot_params)