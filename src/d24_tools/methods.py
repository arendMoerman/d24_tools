import decode as dc
import numpy as np
import xarray as xr
import scipy.interpolate as interp
from datetime import datetime, timedelta

import copy
from d24_tools import colors
from d24_tools import atmosphere
from functools import partial

CHOPSEP = -234

def _subtract_per_scan(dems, conv_factor=1):
    """Apply source-sky subtraction to a single-scan DEMS."""
    if len(states := np.unique(dems.state)) != 1:
        raise ValueError("State must be unique.")
    
    t_amb = np.nanmean(dems.temperature.data)

    if (state := states[0]) == "ON":
        src = dc.select.by(dems, "beam", include="A")
        sky = dc.select.by(dems, "beam", include="B")

    if state == "OFF":
        src = dc.select.by(dems, "beam", include="B")
        sky = dc.select.by(dems, "beam", include="A")

    signal = conv_factor * t_amb * (src - sky.mean("time").data) / ((t_amb - sky.mean("time")))

    average = signal.mean("time")
    variance = signal.var("time")

    ds_out = xr.merge([average.rename("avg"), variance.rename("var")])
    
    return ds_out

def _subtract_per_halfscan(dems, conv_factor=1):
    """Apply source-sky subtraction to a single-scan DEMS."""
    if len(states := np.unique(dems.state)) != 1:
        raise ValueError("State must be unique.")

    n_first = dems.state.size // 2
    n_last = dems.state.size - n_first 
    print(n_first, n_last, dems.data.shape)
    
    t_amb_first = np.nanmean(dems.temperature.data[:n_first-1])
    t_amb_last = np.nanmean(dems.temperature.data[n_first:])

    if (state := states[0]) == "ON":
        src = dc.select.by(dems, "beam", include="A")
        sky = dc.select.by(dems, "beam", include="B")

    if state == "OFF":
        src = dc.select.by(dems, "beam", include="B")
        sky = dc.select.by(dems, "beam", include="A")

    n_src_first = src.data.shape[0] // 2
    n_sky_first = sky.data.shape[0] // 2
    src_first = src[:n_src_first-1]
    src_last = src[n_src_first:]
    sky_first = sky[:n_sky_first-1]
    sky_last = sky[n_sky_first:]

    signal_first = conv_factor * t_amb_first * (src_first - sky_first.mean("time").data) / ((t_amb_first - sky_first.mean("time")))
    signal_last = conv_factor * t_amb_last * (src_last - sky_last.mean("time").data) / ((t_amb_last - sky_last.mean("time")))

    average_first = signal_first.mean("time")
    variance_first = signal_first.var("time")

    average_last = signal_last.mean("time")
    variance_last = signal_last.var("time")

    ds_out = xr.merge([average_first.rename("avg_first"), average_last.rename("avg_last"), 
                            variance_first.rename("var_first"), variance_last.rename("var_last")])
    
    return ds_out

def _overshoot_per_scan(dems, buff):
    """Apply overshoot removal to a single-scan DEMS."""
    if len(states := np.unique(dems.state)) != 1:
        raise ValueError("State must be unique.")

    if (state := states[0]) == "ON":
        good = (np.absolute(dems.lon.data) < buff) & (np.absolute(dems.lat.data) < buff)

    if state == "OFF":
        good = (np.absolute(dems.lon.data - CHOPSEP) < buff) & (np.absolute(dems.lat.data) < buff)

    startbuff = np.zeros(10)
    if dems.scan[0] != "1":
        good[:10] = startbuff

    #print(dems.scan) 
    #import matplotlib.pyplot as plt
    #plt.scatter(dems.time.values[good], dems.data[good,0])
    #plt.show()

    return dems[good]

    raise ValueError("State must be either ON or OFF.")

def _consecutive(data, stepsize=1):
    """
    Take numpy array and return list with arrays containing consecutive blocks
    
    @param data Array in which consecutive chunks are to be located.

    @returns List with arrays of consecutive chunks of data as elements.
    """

    return np.split(data, np.where(np.diff(data) != stepsize)[0]+1)
