import decode as dc
import numpy as np
import xarray as xr
import scipy.interpolate as interp
from datetime import datetime, timedelta

import copy
from d24_tools import colors
from d24_tools import atmosphere
from d24_tools import tools
from d24_tools import methods
from functools import partial
KB = 1.380649e-23

def obs_to_nod_avg(da_sub, conv_factor, num_nods=2, var_B=0):
    """
    Reduce a full observation to nod (full ABBA) averages

    @param da_sub Despiked and overshoot-removed.
    @param var_B Which method to use to calculate the variance in beam B for OFF nods:
        0 : directly over beam B' and B''
        1 : over B' and B'' separately and then average
        2 : use variance of beam A and convert to beam B

    @returns 2D Numpy array containing ABBA cycles along first axis and frequency along second axis.
    """

    master_id = da_sub.chan.values
    freq = da_sub.d2_mkid_frequency.values

    if var_B == 0:
        da_sub = da_sub.groupby("scan").map(methods._subtract_per_scan_var_direct, args=(conv_factor,))
    elif var_B == 1:
        da_sub = da_sub.groupby("scan").map(methods._subtract_per_scan_var_split, args=(conv_factor,))
    elif var_B == 2:
        da_sub = da_sub.groupby("scan").map(methods._subtract_per_scan_var_A, args=(conv_factor,))
    elif var_B == 3:
        da_sub = da_sub.groupby("scan").map(methods._subtract_per_scan_var_split_avgchop, args=(conv_factor,))

    scan_labels = da_sub["avg"].scan.data.astype(int)
    args_sort = np.argsort(scan_labels)

    spec_avg = da_sub["avg"].data[args_sort]
    spec_var = da_sub["var"].data[args_sort]

    cycle_avg = np.zeros((spec_avg.shape[0] // num_nods, spec_avg.shape[1]))
    cycle_var = np.zeros((spec_var.shape[0] // num_nods, spec_var.shape[1]))
    for i in range(len(args_sort) // num_nods):
        for j in range(num_nods):
            cycle_avg[i] += spec_avg[i*num_nods + j]
            cycle_var[i] += spec_var[i*num_nods + j]

    cycle_avg /= num_nods
    cycle_var /= num_nods**2

    return cycle_avg, cycle_var, master_id, freq
