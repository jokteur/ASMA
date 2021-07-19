import numpy as np

from m_distribution import plot_m_distribution


# Simulation parameters
N = 15000
params = dict(
    dt=0.5 * 1e-2,
    time_end=13,
    Lambda=np.array([12.3, 2.5]),
    Gamma=np.array([-8.0, -2.5]),
    c=30,
    lambda_kappa=50,
    Delta=2,
    theta=0,
    base_I=0.5,
    I_ext=2.5,
    I_ext_time=10,
    interaction=0.0,
    use_LambdaGamma=True,
)
params["Gamma"] = params["Gamma"] / params["Lambda"]

plot_params = dict(
    dpi=300,
    savepath="git\\plots\\presentation\\results",
    savename="m_distribution.png",
    save=True,
    usetex=True,
    cmap="Oranges",
    noshow=True,
    font_size=16,
)


plot_m_distribution(params, N, **plot_params)