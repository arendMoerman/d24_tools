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

def _subtract_per_scan_var_direct(dems, conv_factor, correct_atm):
    """
    Apply source-sky subtraction to a single-scan DEMS.
    This version subtracts the off-source beam average from the on-source timestream, 
    after which the variance in the on-source beam is directly calculated from the on-source stream.
    
    @param dems Data array which has been grouped.
    @param conv_factor Optional factor to convert brightness temperature to another quantity.
    
    @returns Data array containing on-source nod average and variance
    """

    if len(states := np.unique(dems.state)) != 1:
        raise ValueError("State must be unique.")
    
    t_amb = np.nanmean(dems.temperature.data)

    if (state := states[0]) == "ON":
        src = dc.select.by(dems, "beam", include="A")
        sky = dc.select.by(dems, "beam", include="B")

    if state == "OFF":
        src = dc.select.by(dems, "beam", include="B")
        sky = dc.select.by(dems, "beam", include="A")

    if correct_atm:
        signal = conv_factor * t_amb * (src - sky.mean("time").data) / ((t_amb - sky.mean("time")))

    else:
        signal = conv_factor * (src - sky.mean("time").data)

    average = signal.mean("time")
    variance = signal.var("time")

    ds_out = xr.merge([average.rename("avg"), variance.rename("var")])
    
    return ds_out

def _subtract_per_scan_var_A(dems, conv_factor, correct_atm):
    """
    Apply source-sky subtraction to a single-scan DEMS.
    This version subtracts the off-source beam average from the on-source beam, but solely uses beam A for variance calculations. 

    @param dems Data array which has been grouped.
    @param conv_factor Optional factor to convert brightness temperature to another quantity.
    
    @returns Data array containing on-source nod average and variance
    """
    
    if len(states := np.unique(dems.state)) != 1:
        raise ValueError("State must be unique.")
    
    t_amb = np.nanmean(dems.temperature.data)

    if (state := states[0]) == "ON":
        src = dc.select.by(dems, "beam", include="A")
        sky = dc.select.by(dems, "beam", include="B")

    if state == "OFF":
        src = dc.select.by(dems, "beam", include="B")
        sky = dc.select.by(dems, "beam", include="A")

    if correct_atm:
        prop_fac = (conv_factor * t_amb / (t_amb - sky.mean("time")))**2
        signal = conv_factor * t_amb * (src - sky.mean("time").data) / ((t_amb - sky.mean("time")))

    else:
        prop_fac = conv_factor**2
        signal = conv_factor * (src - sky.mean("time").data)

    average = signal.mean("time")

    if state == "OFF":
        variance = sky.var("time") * prop_fac 

    else:
        variance = signal.var("time")

    ds_out = xr.merge([average.rename("avg"), variance.rename("var")])
    
    return ds_out

def _subtract_per_scan_var_split(dems, conv_factor, correct_atm):
    """
    Apply source-sky subtraction to a single-scan DEMS.
    This version subtracts the off-source beam average from the on-source timestream, 
    but calculates the variances for B' and B'' separately in OFF nods, combining them at the end. 
    
    @param dems Data array which has been grouped.
    @param conv_factor Optional factor to convert brightness temperature to another quantity.
    
    @returns Data array containing on-source nod average and variance
    """
    if len(states := np.unique(dems.state)) != 1:
        raise ValueError("State must be unique.")
    
    t_amb = np.nanmean(dems.temperature.data)

    if (state := states[0]) == "ON":
        src = dc.select.by(dems, "beam", include="A")
        sky = dc.select.by(dems, "beam", include="B")
        
        if correct_atm:
            signal = conv_factor * t_amb * (src - sky.mean("time").data) / ((t_amb - sky.mean("time")))

        else:
            signal = conv_factor * (src - sky.mean("time").data)

        average = signal.mean("time")
        variance = signal.var("time")

    # In the OFF state, need to subdivide between B' and B''
    if state == "OFF":
        arg_sort = np.argsort(dems.time.values)
        
        idx_B = np.squeeze(np.argwhere(dems.beam.values[arg_sort] == "B"))
        chunk_list_B = _consecutive(idx_B)

        list_Bpr = []
        list_Bdpr = []

        for i, chunk in enumerate(chunk_list_B):
            subl = []
            for item in chunk:
                subl.append(item)
            if i % 2 == 0: # even -> Bpr
                list_Bpr += subl
            else:
                list_Bdpr += subl

        src_Bpr = dems[list_Bpr]
        src_Bdpr = dems[list_Bdpr]

        src = dc.select.by(dems, "beam", include="B")
        sky = dc.select.by(dems, "beam", include="A")
        
        if correct_atm:
            signal_pr = conv_factor * t_amb * (src_Bpr - sky.mean("time").data) / ((t_amb - sky.mean("time")))
            signal_dpr = conv_factor * t_amb * (src_Bdpr - sky.mean("time").data) / ((t_amb - sky.mean("time")))
        
            signal = conv_factor * t_amb * (src - sky.mean("time").data) / ((t_amb - sky.mean("time")))

        else:
            signal_pr = conv_factor * (src_Bpr - sky.mean("time").data)
            signal_dpr = conv_factor * (src_Bdpr - sky.mean("time").data)
        
            signal = conv_factor * (src - sky.mean("time").data)
        
        average = signal.mean("time")
        variance = (signal_pr.var("time") + signal_dpr.var("time")) / 4

    ds_out = xr.merge([average.rename("avg"), variance.rename("var")])
    
    return ds_out

def _subtract_per_scan_var_split_avgchop(dems, conv_factor, correct_atm):
    """
    Apply source-sky subtraction to a single-scan DEMS.
    This version subtracts the off-source beam average from the on-source timestream, 
    but calculates the variances for B' and B'' separately in OFF nods, combining them at the end. 
    Also, the variance is calculated from the chop averages instead of the splitted timestreams, in order to eliminate the internal chop modulation.
    
    @param dems Data array which has been grouped.
    @param conv_factor Optional factor to convert brightness temperature to another quantity.
    
    @returns Data array containing on-source nod average and variance
    """
    if len(states := np.unique(dems.state)) != 1:
        raise ValueError("State must be unique.")
    
    t_amb = np.nanmean(dems.temperature.data)

    if (state := states[0]) == "ON":
        src = dc.select.by(dems, "beam", include="A")
        sky = dc.select.by(dems, "beam", include="B")
        
        if correct_atm:
            signal = conv_factor * t_amb * (src - sky.mean("time").data) / ((t_amb - sky.mean("time")))

        else:
            signal = conv_factor * (src - sky.mean("time").data)


        average = signal.mean("time")
        variance = signal.var("time")

    # In the OFF state, need to subdivide between B' and B''
    if state == "OFF":
        arg_sort = np.argsort(dems.time.values)
        
        idx_B = np.squeeze(np.argwhere(dems.beam.values[arg_sort] == "B"))
        chunk_list_B = _consecutive(idx_B)

        src_Bpr = []
        src_Bdpr = []

        for i, chunk in enumerate(chunk_list_B):
            subl = []
            for item in chunk:
                subl.append(item)
            if i % 2 == 0: # even -> Bpr
                src_Bpr.append(dems[subl].mean("time").data)
            else:
                src_Bdpr.append(dems[subl].mean("time").data)

        src_Bpr = np.array(src_Bpr)
        src_Bdpr = np.array(src_Bdpr)

        src = dc.select.by(dems, "beam", include="B")
        sky = dc.select.by(dems, "beam", include="A")
        
        if correct_atm:
            signal_pr = conv_factor * t_amb * (src_Bpr - sky.mean("time").data) / ((t_amb - sky.mean("time")))
            signal_dpr = conv_factor * t_amb * (src_Bdpr - sky.mean("time").data) / ((t_amb - sky.mean("time")))
        
            signal = conv_factor * t_amb * (src - sky.mean("time").data) / ((t_amb - sky.mean("time")))

        else:
            signal_pr = conv_factor * (src_Bpr - sky.mean("time").data)
            signal_dpr = conv_factor * (src_Bdpr - sky.mean("time").data)
        
            signal = conv_factor * (src - sky.mean("time").data)

        average = signal.mean("time")
        variance = signal.var("time")
        variance.data = (np.nanvar(signal_pr, axis=0) + np.nanvar(signal_dpr, axis=0)) / 4

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

def _mean_in_time(dems):
    """Similar to DataArray.mean but keeps middle time."""
    middle = dems[len(dems) // 2 : len(dems) // 2 + 1]
    return xr.zeros_like(middle) + dems.mean("time")
