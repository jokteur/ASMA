import numpy as np
import matplotlib.pyplot as plt

from activity import plot_activity


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
    interaction=0.3,
    use_LambdaGamma=True,
)
params["Gamma"] = params["Gamma"] / params["Lambda"]

plot_activity(
    params,
    N=25000,
    time_before_input=3,
    plot_QR=False,
    ylim=20,
    I_ylim=5,
    a_cutoff=7,
    savepath="git\\plots\\final\\results",
    savename=f"A_d.png",
    dpi=300,
    save=False,
    plot_H=False,
    usetex=False,
    font_size=14,
    noshow=True,
)
plt.show()