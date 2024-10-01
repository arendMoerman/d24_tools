import decode as dc
import numpy as np
import xarray as xr
import scipy.interpolate as interp
from datetime import datetime, timedelta

import copy
from d24_tools import colors
from d24_tools import atmosphere
from d24_tools import tools
from functools import partial
KB = 1.380649e-23
def obs_to_nod_avg(da_sub, conv_factor):
    """
    Reduce a full observation to nod (full ABBA) averages

    @param da_sub Despiked and overshoot-removed.

    @returns 2D Numpy array containing ABBA cycles along first axis and frequency along second axis.
    """

    master_id = da_sub.chan.values
    freq = da_sub.d2_mkid_frequency.values

    da_sub = da_sub.groupby("scan").map(tools._subtract_per_scan, args=(conv_factor,))
    

    scan_labels = da_sub["avg_last"].scan.data.astype(int)
    args_sort = np.argsort(scan_labels)

    spec_avg = da_sub["avg_last"].data[args_sort]
    spec_var = da_sub["var_last"].data[args_sort]
    assert(len(args_sort) % 2 == 0)

    cycle_avg = np.zeros((spec_avg.shape[0] // 4, spec_avg.shape[1]))
    cycle_var = np.zeros((spec_var.shape[0] // 4, spec_var.shape[1]))
    for i in range(len(args_sort) // 4):
        for j in range(4):
            cycle_avg[i] += spec_avg[i*4 + j]
            cycle_var[i] += spec_var[i*4 + j]

    cycle_avg /= 4
    cycle_var /= 4

    return cycle_avg, cycle_var, master_id, freq
