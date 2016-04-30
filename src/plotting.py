import time
import numpy as np

from psmon import publish
from psmon import util
from psmon import plots


class Manager(object):
    def __init__(self, topic, title=None, pubrate=None, publisher=None):
        self.topic = topic
        self._data = None
        self._title = title or self.topic
        self.pubrate = pubrate
        self._publisher = publisher or publish.send
        self.__last_pub = time.time()

    @property
    def title(self):
        return self._title

    @title.setter
    def title(self, title):
        if self._data is not None:
            self._data.title = title
        self._title = title

    def publish(self, timestamp=None):
        current_time = time.time()
        if self.pubrate is None or self.pubrate * (current_time - self.__last_pub) >= 1:
            self.__last_pub = current_time
            self._data.ts = timestamp or time.ctime()
            self._publisher(self.topic, self._data)


class OverlayManager(Manager):
    def __init__(self, topic, title=None, xlabel=None, ylabel=None, pubrate=None, publisher=None):
        super(OverlayManager, self).__init__(topic, title, pubrate, publisher)
        self._noverlay = 0
        self._names = []
        self._formats = []
        self._fills = []

    @property
    def names(self):
        return self._names

    @property
    def size(self):
        return self._noverlay

    def format(self, name, newformat=None):
        if newformat is None:
            return self._formats[self._get_index(name)]
        else:
            self._formats[self._get_index(name)] = newformat

    def fill(self, name, newfill=None):
        if newfill is None:
            return self._fills[self._get_index(name)]
        else:
            self._fills[self._get_index(name)] = newfill

    def rename(self, old_name, new_name):
        self._names[self._get_index(old_name)] = new_name

    def _make(self, name):
        if name in self._names:
            return self._names.index(name), False
        index = self._noverlay
        self._names.append(name)
        self._noverlay += 1
        return index, True

    def _get_index(self, name):
        if name in self._names:
            return self._names.index(name)
        else:
            raise KeyError('Unknown plot name: %s'%name)


class Image(Manager):
    def __init__(self, topic, title=None, xlabel=None, ylabel=None, pubrate=None, publisher=None, pedestal=None):
        super(Image, self).__init__(topic, title, pubrate, publisher)
        self._pedestal = pedestal
        self.use_pedestal = pedestal is not None
        self._data = plots.Image(None, self.title, None)

    def image(self, image=None):
        if image is None:
            return self._data.image
        else:
            if self.use_pedestal:
                self._data.image = image - self._pedestal
            else:
                self._data.image = image

    def pedestal(self, pedestal=None):
        if pedestal is None:
            return self._pedestal
        else:
            self._pedestal = pedestal
            self.use_pedestal = True


class LinePlot(OverlayManager):
    def __init__(self, topic, title=None, xlabel=None, ylabel=None, leg_offset=None, pubrate=None, publisher=None):
        super(LinePlot, self).__init__(topic, title, pubrate, publisher)
        self._xdata = []
        self._ydata = []
        self._data = plots.XYPlot(
            None,
            self._title,
            self._xdata,
            self._ydata,
            xlabel=xlabel,
            ylabel=ylabel,
            leg_label=self._names,
            leg_offset=leg_offset,
            formats=self._formats,
        )

    def xdata(self, name):
        return self._xdata[self._get_index(name)]

    def ydata(self, name):
        return self._ydata[self._get_index(name)]

    def make_plot(self, name, formatter='-'):
        index, new_plot = self._make(name)
        if new_plot:
            self._xdata.append(np.zeros(0))
            self._ydata.append(np.zeros(0))
            self._formats.append(formatter)
        else:
            self._xdata[index] = np.zeros(0)
            self._ydata[index] = np.zeros(0)
            self._formats[index] = formatter

    def add(self, name, xval, yval):
        index = self._get_index(name)
        xval = util.convert_to_array(xval)
        yval = util.convert_to_array(yval)
        input_sort = np.argsort(xval)
        xval_sorted = xval[input_sort]
        yval_sorted = yval[input_sort]
        inserts = np.searchsorted(self._xdata[index], xval_sorted)
        self._xdata[index] = np.insert(self._xdata[index], inserts, xval_sorted)
        self._ydata[index] = np.insert(self._ydata[index], inserts, yval_sorted)

    def clear(self, name=None):
        if name is None:
            for index in xrange(self._noverlay):
                self._xdata[index] = np.zeros(0)
                self._ydata[index] = np.zeros(0)
        else:
            index = self._get_index(name)
            self._xdata[index] = np.zeros(0)
            self._ydata[index] = np.zeros(0)


class ScatterPlot(OverlayManager):
    def __init__(self, topic, title=None, xlabel=None, ylabel=None, leg_offset=None, pubrate=None, publisher=None):
        super(ScatterPlot, self).__init__(topic, title, pubrate, publisher)
        self._indices = []
        self._xdata = []
        self._ydata = []
        self._data = plots.XYPlot(
            None,
            self._title,
            [],
            [],
            xlabel=xlabel,
            ylabel=ylabel,
            leg_label=self._names,
            leg_offset=leg_offset,
            formats=self._formats,
        )

    def xdata(self, name):
        return self._xdata[self._get_index(name)]

    def ydata(self, name):
        return self._ydata[self._get_index(name)]

    def make_plot(self, name, size=1000, formatter='.'):
        index, new_plot = self._make(name)
        if new_plot:
            self._indices.append(0)
            self._xdata.append(np.zeros(size))
            self._ydata.append(np.zeros(size))
            self._data.xdata.append(self._xdata[index][:self._indices[index]])
            self._data.ydata.append(self._ydata[index][:self._indices[index]])
            self._formats.append(formatter)
        else:
            self._indices[index] = 0
            self._xdata[index] = np.zeros(size)
            self._ydata[index] = np.zeros(size)
            self._data.xdata[index] = self._xdata[index][:self._indices[index]]
            self._data.ydata[index] = self._ydata[index][:self._indices[index]]
            self._formats[index] = formatter

    def add(self, name, xval, yval):
        index = self._get_index(name)
        insert_size = util.py_length(xval)
        if insert_size != util.py_length(yval):
            raise ValueError('length of xval and yval does not match: %d vs %d' % (insert_size, util.py_length(yval)))
        if self._indices[index] + insert_size > self._xdata[index].size:
            # double the size if we need to reallocate
            self._xdata[index] = np.resize(self._xdata[index], 2*(self._indices[index]+insert_size))
            self._ydata[index] = np.resize(self._ydata[index], 2*(self._indices[index]+insert_size))
        self._xdata[index][self._indices[index]:self._indices[index]+insert_size] = xval
        self._ydata[index][self._indices[index]:self._indices[index]+insert_size] = yval
        self._indices[index] += insert_size
        self._data.xdata[index] = self._xdata[index][:self._indices[index]]
        self._data.ydata[index] = self._ydata[index][:self._indices[index]]

    def clear(self, name=None):
        if name is None:
            for index in xrange(self._noverlay):
                self._indices[index] = 0
                self._data.xdata[index] = self._xdata[index][:self._indices[index]]
                self._data.ydata[index] = self._ydata[index][:self._indices[index]]
        else:
            index = self._get_index(name)
            self._indices[index] = 0
            self._data.xdata[index] = self._xdata[index][:self._indices[index]]
            self._data.ydata[index] = self._ydata[index][:self._indices[index]]


class Histogram(OverlayManager):
    def __init__(self, topic, title=None, xlabel=None, ylabel=None, leg_offset=None, pubrate=None, publisher=None):
        super(Histogram, self).__init__(topic, title, pubrate, publisher)
        self._nbins = []
        self._ranges = []
        self._bins = []
        self._values = []
        self._data = plots.Hist(
            None,
            self._title,
            self._bins,
            self._values,
            xlabel=xlabel,
            ylabel=ylabel,
            leg_label=self._names,
            leg_offset=leg_offset,
            formats=self._formats,
            fills=self._fills
        )

    def bins(self, name):
        return self._bins[self._get_index(name)]

    def values(self, name):
        return self._values[self._get_index(name)]

    def make_hist(self, name, nbins, bmin, bmax, formatter='-', fills=True):
        index, new_hist = self._make(name)
        if new_hist:
            self._nbins.append(nbins)
            self._ranges.append((bmin, bmax))
            self._bins.append(util.make_bins(nbins, bmin, bmax))
            self._values.append(np.zeros(nbins))
            self._formats.append(formatter)
            self._fills.append(fills)
        else:
            self._nbins[index] = nbins
            self._ranges[index] = (bmin, bmax)
            self._bins[index] = util.make_bins(nbins, bmin, bmax)
            self._values[index] = np.zeros(nbins)
            self._formats[index] = formatter
            self._fills[index] = fills

    def add(self, name, value):
        index = self._get_index(name)
        self._values[index] += np.histogram(value, self._nbins[index], range=self._ranges[index])[0]

    def clear(self, name=None):
        if name is None:
            for value in self._values:
                value[:] = 0
        else:
            index = self._get_index(name)
            self._values[index][:] = 0
