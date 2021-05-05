import time
from abc import ABCMeta

from bokeh.models.widgets import Div
from bokeh.plotting import Figure
from bokeh.io import curdoc


class BaseElement(metaclass=ABCMeta):
    """"""

    instance_num = 0

    def __init__(self, refresh_rate=100):
        """"""
        BaseElement.instance_num += 1
        self._root = None
        self._refresh_rate = refresh_rate
        self._reset = False

    def layout(self):
        return self._root


def empty_placeholder():
    return Div(text=" ")


class ThrottledEvent:
    _callback = None
    _lastcall = 0
    _numcalls = 0
    _total_time = 0

    def __init__(self, fire_rate=None, refresh_rate=50):
        """fire_rate in ms"""
        curdoc().add_periodic_callback(self._fire_event, refresh_rate)

        if fire_rate:
            self._dynamic_fire_rate = False
            self._fire_rate = fire_rate / 1000
        else:
            self._dynamic_fire_rate = True
            self._fire_rate = 0.05

    def add_event(self, callback):
        self._callback = callback
        print(time.time() - self._lastcall, self._fire_rate)
        if time.time() - self._lastcall > self._fire_rate:
            t = time.time() - self._lastcall
            print(f"Event added: {t}")
            curdoc().add_next_tick_callback(self._call_and_measure)

    def _call_and_measure(self):
        self._numcalls += 1
        self._lastcall = time.time()

        prev = time.time()
        self._callback()
        self._callback = None
        self._total_time += time.time() - prev

        if self._dynamic_fire_rate:
            # Use buffer (10)
            self._fire_rate = self._total_time / self._numcalls

    def _fire_event(self):
        if self._callback and time.time() - self._lastcall > self._fire_rate:
            curdoc().add_next_tick_callback(self._call_and_measure)
            self._lastcall = time.time()