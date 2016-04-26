import os
import sys
import numpy as np
from itertools import chain, izip
from contextlib import contextmanager


def is_py_iter(obj):
    """
    Check if the object is an iterable python object excluding ndarrays
    """
    return hasattr(obj, '__iter__') and not isinstance(obj, np.ndarray)


def arg_inflate(index, *args):
    args = list(args)
    for i in range(len(args)):
        if i == index:
            continue
        if not is_py_iter(args[i]):
            args[i] = [args[i]] * len(args[index])
    return args


def arg_inflate_flat(index, *args):
    if is_py_iter(args[index]):
        return list(chain.from_iterable(izip(*arg_inflate(index, *args))))
    else:
        return args


def arg_inflate_tuple(index, *args):
    if is_py_iter(args[index]):
        return zip(*arg_inflate(index, *args))
    else:
        return [args]


def inflate_input(input, input_ref):
    if is_py_iter(input_ref):
        return arg_inflate(1, input, input_ref)[0]
    else:
        return [input]


def make_bins(nbins, bmin, bmax):
    step = (bmax - bmin)/nbins
    return np.arange(bmin, bmax + step, step)[:nbins+1]


def window_ratio(min_res, max_res):
    def window_ratio_calc(ncols, nrows):
        pref_x = min_res.x * ncols
        pref_y = min_res.y * nrows

        if pref_x > max_res.x or pref_y > max_res.y:
            num = min(max_res.x/float(ncols), max_res.y/float(nrows))
            pref_x = max(ncols * num, min_res.x)
            pref_y = max(nrows * num, min_res.y)

        return int(pref_x), int(pref_y)
    return window_ratio_calc

@contextmanager
def redirect_stdout():
  sys.stdout.flush()
  sys.stderr.flush()
  newstdout = os.dup(1)
  newstderr = os.dup(2)
  with open(os.devnull,'wb') as devnull:
    os.dup2(devnull.fileno(), 1)
    os.dup2(devnull.fileno(), 2)
    try:
      yield
    finally:
      sys.stdout = os.fdopen(newstdout, 'w')
      sys.stderr = os.fdopen(newstderr, 'w')
