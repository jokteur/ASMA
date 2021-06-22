import numpy as np
import matplotlib.pyplot as plt

from activity import plot_activity


# Simulation parameters
params = dict(
    dt=0.5 * 1e-2,
    time_end=20,
    Lambda=np.array([10.3, 2.5]),
    Gamma=np.array([-5.0, -1.0]),
    # Lambda=np.array([28.0, 8.0, 1.0]),
    # Gamma=np.array([-3.5, 3.0, -1.0]),
    c=20,
    lambda_kappa=50,
    Delta=1,
    theta=1,
    base_I=0.0,
    I_ext=2.5,
    I_ext_time=15,
    interaction=0.0,
    use_LambdaGamma=True,
)
params["Gamma"] = params["Gamma"] / params["Lambda"]

plot_activity(
    params,
    N=25000,
    time_before_input=3,
    plot_QR=True,
    ylim=20,
    I_ylim=5,
    a_cutoff=10,
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