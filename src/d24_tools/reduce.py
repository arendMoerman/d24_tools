import decode as dc
import numpy as np
import xarray as xr
import scipy.interpolate as interp
from datetime import datetime, timedelta
import scipy.interpolate as interp
from astropy.units import Quantity
import copy
from scipy.stats import binned_statistic_2d
from multiprocessing import Pool
import os
import matplotlib.pyplot as plt
from sys import getsizeof
from gc import collect

from d24_tools import colors
from d24_tools import methods
from d24_tools import utils
from d24_tools import parallel
from functools import partial
import warnings
NCPU = os.cpu_count()
NCPU_USE = NCPU - 2 if NCPU <= 10 else 10

def despike(da, include=["ON", "OFF"], by="state", num_remove=1):
    """
    Despike a TOD and return despiked data array, given an inclusion string/list of strings.

    @param da Data array containing TOD
    @param include String or list of strings containing inclusion parameters.
        For pswsc observations, ["ON", "OFF"] is recommended.
        For daisy scans or still AB chopping, use "SCAN".
    @param by Label to include by. Default is 'state'.
    @param num_remove Number of points to remove from front and back of chop.

    @returns Despiked data array with only the included labels
    """

    # Select the part of the observation in the OFF position, so that beam A is looking at the atmosphere
    da = da.sortby("time")
    
    #collect()
    
    #da = dc.select.by(da, by, include=include)
    da = da.sel(time=da.state.isin(include))
    
    #collect()
    
    idx_A = np.squeeze(np.argwhere(da.beam.data == "A"))
    idx_B = np.squeeze(np.argwhere(da.beam.data == "B"))

    all_idx = np.arange(da.shape[0])
    
    assert(all_idx.size == (idx_A.size + idx_B.size))
    
    chunk_list_A = utils.consecutive(idx_A)
    chunk_list_B = utils.consecutive(idx_B)

    chunk_list_despiked_A = []
    chunk_list_despiked_B = []

    #print(colors._green("despiking TOD..."))
    for idx_chunk_A in chunk_list_A:
        if idx_chunk_A.size >= 3:
            chunk_list_despiked_A.append(idx_chunk_A[num_remove:-num_remove])
    for idx_chunk_B in chunk_list_B:
        if idx_chunk_B.size >= 3:
            chunk_list_despiked_B.append(idx_chunk_B[num_remove:-num_remove])

    chunk_list_despiked_A = np.array([x for xs in chunk_list_despiked_A for x in xs])
    chunk_list_despiked_B = np.array([x for xs in chunk_list_despiked_B for x in xs])
    

    despiked_indices = np.concatenate((chunk_list_despiked_A, chunk_list_despiked_B)).astype(int)
    da_out = da[despiked_indices]
    
    #collect()

    return da_out 

def dewing(da):
    """
    Dewing data array.
    Pass full data array, do not preselect on beam A or B.
    Currently, seems only beam B needs dewinging.
    """

    f = 5.000018697350202 # Fitted sine frequency
    buffer = 0.0024 # Temporal buffer around edges of datachunk

    # Make indices of beam B in total data array
    idx_A = np.arange(da.time.size)[da.beam == "A"]
    idx_B = np.arange(da.time.size)[da.beam == "B"]

    # Calculate time in seconds
    # The time_seconds is w.r.t. the FIRST timestamp in beam B after despiking
    time_B_differences = da[idx_B].time - da[idx_B].time[0]
    time_B_seconds = np.array(time_B_differences.astype('timedelta64[ns]').astype(float)*(10**(-9)))

    # Find x-values. We first fold the data to double the chopper frequency (so the two chopper wheel blades overlap)
    x = (time_B_seconds % (0.5/f))

    # If the chunk is cut off by the boundaries, add half a period to unwrap it
    if (np.max(x) - np.min(x)) > (0.5/f) /2:
        x = ((time_B_seconds + (0.5/f) /2) % (0.5/f))

    # Get a robust estimate for the lowest and highest folded time in the chunk (i.e. the 0.01% quantiles).
    # This makes sure the buffer works in case of weird outliers
    lower, upper= np.quantile(x, [0.0001, 0.9999])

    # Compute cutoffs
    low_cutoff = lower + buffer
    high_cutoff = upper - buffer

    despiking_mask = (x >= low_cutoff) * (x <= high_cutoff)
    idx_B_keep = idx_B[despiking_mask]

    return da[np.union1d(idx_A, idx_B_keep)]

