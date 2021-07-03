import numpy as np
import copy
import matplotlib.pyplot as plt

from m_t_distr2 import plot_m_t_distr

# Simulation parameters
params = dict(
    dt=1e-2,
    time_end=18,
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


plot_params = dict(
    dpi=300,
    savepath="git\\plots\\final\\results\\hard_threshold",
    save=True,
    usetex=True,
    noshow=True,
    font_size=12,
)

plot_m_t_distr(
    params,
    N=10000,
    Deltas=[4, 0.9, 0.1],
    I_exts=[1.5, 1.5, 1.5],
    im_res=100,
    num_bins=30,
    margin=1,
    savename="m_t_distr.png",
    time_before_input=1,
    **plot_params
)
plot_m_t_distr(
    params,
    N=10000,
    Deltas=[4, 0.9, 0.1],
    I_exts=[1.5, 1.5, 1.5],
    im_res=100,
    num_bins=30,
    simple_m=True,
    cmap="Oranges",
    margin=1,
    savename="m_simple_distr.png",
    time_before_input=1,
    **plot_params
)
plt.show()