import numpy as np

from age_animation import plot_animation

# Simulation parameters
N = 15000
params = dict(
    dt=0.5 * 1e-2,
    time_end=9,
    Lambda=np.array([12.3, 2.5]),
    Gamma=np.array([-8.0, -2.5]),
    c=30,
    lambda_kappa=50,
    Delta=2,
    theta=0,
    base_I=0.5,
    I_ext=2.0,
    I_ext_time=5,
    interaction=0.0,
    use_LambdaGamma=True,
)
params["Gamma"] = params["Gamma"] / params["Lambda"]

plot_params = dict(
    dpi=300,
    savepath="git\\plots\\presentation\\results",
    save=True,
    usetex=True,
    noshow=True,
    num_points=5000,
    anim_int=1,
    xlim=1.3,
    M_lim=(-8, 0),
    font_size=16,
    t_from=4,
    t_to=6,
)

plot_animation(params, N, num_bins=20, savename="age_animation.mp4", **plot_params)

plot_animation(
    params,
    N,
    num_bins=20,
    characteristics=9,
    along_char=False,
    savename="age_animation_with_charac.mp4",
    **plot_params
)

plot_animation(
    params,
    N,
    num_bins=20,
    characteristics=9,
    plot_hist=True,
    savename="age_animation_hist.mp4",
    **plot_params
)

plot_animation(
    params,
    N,
    num_bins=20,
    characteristics=9,
    plot_hist=True,
    plot_m_distr=True,
    # plot_mean=True,
    savename="age_animation_hist_and_other.mp4",
    **plot_params
)

plot_animation(
    params,
    N,
    num_bins=20,
    characteristics=9,
    plot_hist=True,
    plot_m_distr=False,
    plot_mean=True,
    savename="age_animation_mean.mp4",
    **plot_params
)