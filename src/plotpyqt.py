import sys
import logging
import collections
import numpy as np

import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui

from psmon import config
from psmon.util import is_py_iter, arg_inflate_tuple
from psmon.plots import Hist, Image, XYPlot, MultiPlot
from psmon.format import parse_fmt_xyplot


LOG = logging.getLogger(__name__)


TypeMap = {
    Hist: 'HistClient',
    Image: 'ImageClient',
    XYPlot: 'XYPlotClient',
#    MultiPlot: 'MultiPlotClient',
}


def type_getter(data_type, mod_name=__name__):
    plot_type_name = TypeMap.get(data_type)
    return getattr(sys.modules[__name__], plot_type_name)


class Plot(object):
    def __init__(self, init, framegen, info, rate):
        # Boilerplate code from pyqtgraph examples - might not be the ideal way to do this for this case
        self.fig_win = pg.GraphicsLayoutWidget()
        self.fig_win.setWindowTitle(init.title)
        self.fig_win.show()
        self.plot_view = self.fig_win.addPlot()
        self.set_title(init.ts)
        self.set_title_axis('bottom', init.xlabel)
        self.set_title_axis('left', init.ylabel)
        # specific to this class
        self.framegen = framegen
        self.rate_ms = rate * 1000
        self.info = info
        self.multi_plot = False
        # set any user specified default axis ranges
        self.set_xy_ranges()

    def update(self, data):
        pass

    def animate(self):
        self.ani_func()

    def ani_func(self):
        # call the data update function
        self.update(self.framegen.next())
        # setup timer for calling next update call
        QtCore.QTimer.singleShot(self.rate_ms, self.ani_func)

    def set_title(self, title):
        if title is not None:
            self.plot_view.setTitle(title)

    def set_title_axis(self, axis_name, axis_label_data):
        """
        Function for setting a label on the axis specified by 'axis_name'. The label data can be either a simple
        string or a dictionary of keywords that is passed on to the pyqtgraph setLabel function.

        Supported keywords:
        - axis_title
        - axis_units
        - axis_unit_prefix
        """
        if isinstance(axis_label_data, collections.Mapping):
            self._set_title_axis(axis_name, **axis_label_data)
        else:
            self._set_title_axis(axis_name, axis_label_data)

    def _set_title_axis(self, axis_name, axis_title, axis_units=None, axis_unit_prefix=None):
        """
        Implementation function for creating a label for a specific axis - takes an axis_name, axis_title and optional
        axis_units, and axis_unit_prefix keyword args, which match to those for pyqtgraph's set label
        """
        if axis_title is not None:
            self.plot_view.setLabel(axis_name, text=axis_title, units=axis_units, unitPrefix=axis_unit_prefix)

    def set_aspect(self):
        """
        Set the ascept ratio of the viewbox of the plot/image to the specified ratio.

        Note: this is disabled if explicit x/y ranges are set for view box since the
        two options fight each other.
        """
        if self.info.xrange is None and self.info.yrange is None:
            self.plot_view.getViewBox().setAspectLocked(lock=True, ratio=self.info.aspect)

    def set_xy_ranges(self):
        if self.info.xrange is not None:
            self.plot_view.setXRange(*self.info.xrange)
        if self.info.yrange is not None:
            self.plot_view.setYRange(*self.info.yrange)


class ImageClient(Plot):
    def __init__(self, init_im, framegen, info, rate=1):
        super(ImageClient, self).__init__(init_im, framegen, info, rate)
        self.set_aspect()
        self.im = pg.ImageItem(image=init_im.image, border=config.PYQT_BORDERS)
        self.cb = pg.HistogramLUTItem(self.im, fillHistogram=True)

        # Setting up the color map to use
        cm = config.PYQT_COLOR_PALETTE
        if self.info.palette is not None:
            if self.info.palette in pg.graphicsItems.GradientEditorItem.Gradients:
                cm = self.info.palette
            else:
                LOG.warning('Inavlid color palette for pyqtgraph: %s - Falling back to default: %s', self.info.palette, cm)
        self.cb.gradient.loadPreset(cm)

        # Set up colorbar ranges if specified
        if self.info.zrange is not None:
            self.cb.setLevels(*self.info.zrange)
            self.cb.setHistogramRange(*self.info.zrange)

        self.plot_view.addItem(self.im)
        self.fig_win.addItem(self.cb)
        #print self.plot_view.getViewBox().getState()

    def update(self, data):
        """
        Updates the data in the image - none means their was no update for this interval
        """
        if data is not None:
            self.set_title(data.ts)
            self.im.setImage(data.image, autoLevels=False)
        return self.im


class XYPlotClient(Plot):
    def __init__(self, init_plot, framegen, info, rate=1):
        super(XYPlotClient, self).__init__(init_plot, framegen, info, rate)
        self.plots = []
        self.formats = []
        for xdata, ydata, format in arg_inflate_tuple(1, init_plot.xdata, init_plot.ydata, init_plot.formats):
            cval = len(self.plots)
            self.formats.append((format, cval))
            self.plots.append(
                self.plot_view.plot(
                    x=xdata,
                    y=ydata,
                    **parse_fmt_xyplot(format, cval)
                )
            )

    def update(self, data):
        """
        Updates the data in the plot - none means their was no update for this interval
        """
        if data is not None:
            self.set_title(data.ts)
            for index, (plot, data_tup, format_tup) in enumerate(zip(self.plots, arg_inflate_tuple(1, data.xdata, data.ydata, data.formats), self.formats)):
                xdata, ydata, format = data_tup
                old_format, cval = format_tup
                if format != old_format:
                    self.formats[index] = (format, cval)
                    plot.setData(x=xdata, y=ydata, **parse_fmt_xyplot(format, cval))
                else:
                    plot.setData(x=xdata, y=ydata)
        return self.plots


class HistClient(Plot):
    def __init__(self, init_hist, framegen, info, rate=1):
        super(HistClient, self).__init__(init_hist, framegen, info, rate)
        self.hists = []
        for bins, values, format in arg_inflate_tuple(1, init_hist.bins, init_hist.values, init_hist.formats):
            cval = len(self.hists)
            hist = pg.PlotCurveItem(
                x=bins,
                y=values,
                stepMode=True,
                fillLevel=0,
                brush=pg.intColor(cval, config.PYQT_AUTO_COLOR_MAX, alpha=config.PYQT_HIST_ALPHA)
            )
            self.plot_view.addItem(hist)
            self.hists.append(hist)

    def update(self, data):
        """
        Updates the data in the histogram - none means their was no update for this interval
        """
        if data is not None:
            self.set_title(data.ts)
            for hist, data_tup in zip(self.hists, arg_inflate_tuple(1, data.bins, data.values, data.formats)):
                bins, values, formats = data_tup
                hist.setData(x=bins, y=values, stepMode=True)
        return self.hists
