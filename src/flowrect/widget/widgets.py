import time

from skimage import measure

import numpy as np

from bokeh.plotting import Figure
from bokeh.layouts import column, row
from bokeh.io import curdoc
from bokeh.palettes import Spectral6
from bokeh.transform import linear_cmap
from bokeh.models import ColumnDataSource, Span
from bokeh.models import ColorBar, FixedTicker, LinearColorMapper, PrintfTickFormatter
from bokeh.models.widgets import Button, Div, Toggle, TextAreaInput, Slider

from .util import BaseElement, empty_placeholder, ThrottledEvent

from ..simulations import particle_individual, eta_SRM, f_SRM

palette_name = "Viridis11"


class MarkovEmbedding2DWidget(BaseElement):
    """Widget for interacting with 2D Markov Embedding"""

    def __init__(self, refresh_rate=100):
        super().__init__(refresh_rate)

        # Plotting
        self._time = 0
        self._animate_time = 15
        self._refresh_rate = 50
        self._dt = 1e-3
        self._tau = 1 / 20
        self._time_end = 10

        # Throttling
        self._is_busy = False
        self._last_time = 0
        self._throttle = ThrottledEvent()
        # curdoc().add_periodic_callback(self.check_last_call, 50)

        # Initialisation
        self.init_interface()
        self.init_plot()
        self.init_layout()
        self._call_id = None

    def init_interface(self):
        # Interactions buttons

        self._play_b = Button(label="► Play", width=50)
        self._play_b.on_click(self.animate)

        self._animate_500_b = Button(label="Simulate 500", width=120)
        self._animate_500_b.on_click(self.simulate_500)

        self._remove_all_b = Button(label="Remove 500", width=120)
        self._remove_all_b.on_click(self.remove_500)

        self._sliders = {}
        self._params = {}

        self.add_slider("gamma1", start=-10, end=20, value=-0.2, step=0.05, title="γ1")
        self.add_slider("gamma2", start=-20, end=0, value=-0.2, step=0.05, title="γ2")
        self.add_slider("lambda1", start=0, end=20, value=0, step=0.05, title="λ1")
        self.add_slider("lambda2", start=0, end=20, value=0, step=0.05, title="λ2")
        self.add_slider("tau", start=1, end=40, value=1, step=1, title="1/τ")
        self.add_slider("c", start=0.1, end=10, value=1, step=0.1, title="c")
        self.add_slider("seed", start=0, end=20, value=0, step=1, title="seed")

    def init_plot(self):
        self._figure = Figure(
            # x_range=(0, 5),
            # y_range=(0, 5),
            plot_width=500,
            plot_height=400,
            title="Phase plane",
            x_axis_label="M1",
            y_axis_label="M2",
        )
        self._spikes_fig = Figure(
            plot_width=400,
            plot_height=150,
            title="Spikes",
            x_axis_label="time",
            x_range=(0, self._time_end),
        )
        self._eta_fig = Figure(
            plot_width=400, plot_height=150, title="η kernel", x_axis_label="time"
        )

        self._ds = ColumnDataSource({"M1": [], "M2": [], "color": []})
        self._ds_pt = ColumnDataSource({"x": [], "y": []})
        self._ds_spikes = ColumnDataSource({"xs": [], "ys": []})
        self._ds_timeline = ColumnDataSource({"x": [], "y": []})
        self._ds_kernel = ColumnDataSource({"x": [], "y": []})

        self._ds_th_line = ColumnDataSource({"x": [], "y": []})
        # self._ds_image_countour = ColumnDataSource({"image": [], "x": [], "y": [], "dw": [], "dh": []})

        # self._figure.image(
        #     image="image",
        #     x="x",
        #     y="y",
        #     dh="dh",
        #     dw="dw",
        #     source=self._ds_image_countour,
        #     palette=palette_name,
        #     global_alpha=0.5,
        # )
        self._figure.line(x="x", y="y", source=self._ds_th_line, line_color="green", line_width=1.5)
        self._figure.line(
            x="M1", y="M2", source=self._ds, line_color="red", line_width=1.5, line_dash="dotted"
        )
        self._figure.circle(x="x", y="y", source=self._ds_pt)

        # For countour plot
        # self._c_mapper = LinearColorMapper(palette=palette_name, low=0, high=1)
        # levels = [0, 1]
        # self._color_bar = ColorBar(
        #     color_mapper=self._c_mapper,
        #     major_label_text_font_size="8pt",
        #     ticker=FixedTicker(ticks=levels),
        #     formatter=PrintfTickFormatter(format="%.2f"),
        #     label_standoff=6,
        #     border_line_color=None,
        #     location=(0, 0),
        # )
        # self._figure.add_layout(self._color_bar, "right")

        self._spikes_fig.multi_line(xs="xs", ys="ys", line_color="black", source=self._ds_spikes)
        self._spikes_fig.line(
            x="x", y="y", source=self._ds_timeline, line_color="green", line_dash="dashed"
        )
        self._spikes_fig.yaxis.visible = False

        self._eta_fig.line(x="x", y="y", source=self._ds_kernel, line_width=1.5)

        self.timeserie_plot()

    def init_layout(self):
        # Layout

        self._root = row(
            column(self._figure, self._spikes_fig, self._eta_fig),
            column(
                row(self._play_b, self._animate_500_b, self._remove_all_b), *self._sliders.values()
            ),
        )

    def timeserie_plot(self):
        """"""
        Gamma = [float(self._params["gamma1"]), float(self._params["gamma2"])]
        Lambda = [float(self._params["lambda1"]), float(self._params["lambda2"])]
        c = float(self._params["c"])

        np.random.seed(self._params["seed"])
        tau = 1 / self._params["tau"]
        self._ts, self._M, self._spikes = particle_individual(
            self._time_end, self._dt, Gamma, Lambda, c=c, tau=tau
        )

        spikes = self._ts[self._spikes == 1]
        spike_lines_xs = [[t, t] for t in spikes]
        spike_lines_ys = [[0, 1]] * len(spikes)

        # Actual simulation
        self._ds.data = dict(M1=self._M[:, 0], M2=self._M[:, 1], color=self._ts)
        self._ds_pt.data = dict(x=[self._M[0, 0]], y=[self._M[0, 1]])
        self._ds_spikes.data = dict(xs=spike_lines_xs, ys=spike_lines_ys)
        self._ds_timeline.data = dict(x=[0, 0], y=[0, 1])

        # Kernel fct
        x = np.linspace(0, 20, 500)
        self._ds_kernel.data = dict(x=x, y=eta_SRM(x, Gamma, Lambda))

        # Countour plot (only valid for 2D)
        x_max = np.max(self._M[:, 0])
        y_max = np.max(self._M[:, 1])

        # if x_max > y_max:
        #     self._figure.y_range.end = x_max
        # else:
        #     self._figure.x_range.end = y_max

        num = 100
        x = np.linspace(0, x_max, num)
        y = -Gamma[0] / Gamma[1] * x
        # y = np.linspace(0, y_max, num)

        self._ds_th_line.data = dict(x=x, y=y)
        # xx, yy = np.meshgrid(x, y)

        # d = f_SRM(xx * Gamma[0] + yy * Gamma[1], tau=tau, c=c)
        # levels = np.linspace(np.min(d), np.max(d), 10)
        # self._ds_image_countour.data = dict(image=[d], x=[0], y=[0], dh=[y_max], dw=[x_max])
        # if np.max(d) < 1e3:
        #     self._c_mapper.low = np.min(d)
        #     self._c_mapper.high = np.max(d)
        #     self._color_bar.ticker = FixedTicker(ticks=levels)

        self._Ms = None

    def simulate_500(self):
        Gamma = [self._params["gamma1"], self._params["gamma2"]]
        Lambda = [self._params["lambda1"], self._params["lambda2"]]
        c = self._params["c"]

        # _, self._Ms, _ = simulation_ND(self._time_end, self._dt, Gamma, Lambda, c=c, N=500)

        print(self._Ms.shape)

    def remove_500(self):
        self._Ms = None
        self._ds_pt.data = dict(x=[self._M[0, 0]], y=[self._M[0, 1]])

    def update_plot(self):
        """"""
        # self._throttle.add_event(self.timeserie_plot)
        self.timeserie_plot()

    def check_last_call(self):
        if self._execute_last:
            self.update_plot()
            self._execute_last = False

    def add_slider(self, name, **args):
        self._sliders[name] = Slider(**args)
        self._sliders[name].on_change(
            "value", lambda attr, old, new: self.slider_callback(name, attr, old, new)
        )
        self._params[name] = args["value"]

    def slider_callback(self, name, attr, old, new):
        self._params[name] = new
        self.stop()
        self.update_plot()

    def animate_update(self):
        # Only works for dt=1e-3 right now
        self._time += self._animate_time
        if self._time >= self._time_end * 1000:
            self._time = 0

        if type(self._Ms) is np.ndarray:
            self._ds_pt.data = {
                "x": [self._Ms[self._time, :, 0]],
                "y": [self._Ms[self._time, :, 1]],
            }
        else:
            self._ds_pt.data = {"x": [self._M[self._time, 0]], "y": [self._M[self._time, 1]]}
        self._ds_timeline.data["x"] = [self._time / 1000, self._time / 1000]

    def start(self):
        self._play_b.label = "❚❚ Pause"
        self._call_id = curdoc().add_periodic_callback(self.animate_update, self._refresh_rate)

    def stop(self):
        self._play_b.label = "► Play"
        if self._call_id:
            curdoc().remove_periodic_callback(self._call_id)
            self._call_id = None

        self._time = 0

    def animate(self):
        if self._play_b.label == "► Play":
            self.start()
        else:
            self.stop()