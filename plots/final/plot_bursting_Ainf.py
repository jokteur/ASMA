import numpy as np
import copy
import matplotlib.pyplot as plt
from multiprocessing import Pool

from activity import plot_activity
from A_inf import plot_A_inf

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
plot_params = dict(
    dpi=300,
    savepath="git\\plots\\final\\results\\bursting",
    save=True,
    usetex=True,
    noshow=True,
    font_size=16,
)

if __name__ == "__main__":
    p = Pool(10)
    # params["dt"] = 1e-2

    p_params = copy.deepcopy(params)
    # p_params["dt"] = 0.5 * 1e-2

    plot_A_inf(
        params,
        # QR_params=QR_params,
        # p_params=p_params,
        w=3,
        I_end=8,
        num_sim=20,
        num_p_sim=10,
        N=25000,
        savename="A_inf.png",
        pool=p,
        **plot_params
    )
