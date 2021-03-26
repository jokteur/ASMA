import numpy as np
import panel as pn
import holoviews as hv
import holoviews.plotting.bokeh

from .widgets import MarkovEmbedding2DWidget


def main():
    hv.extension("bokeh")

    widget = MarkovEmbedding2DWidget()

    pn.serve(widget.layout(), start=True, show=True, port=5006)