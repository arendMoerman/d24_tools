import decode as dc
import numpy as np
import xarray as xr
import scipy.interpolate as interp
import scipy.stats as scstats
from datetime import datetime, timedelta
import warnings
from multiprocessing import Pool

import copy
from d24_tools import colors
from d24_tools import utils
from d24_tools import parallel
from functools import partial
from tqdm import tqdm
import os

CHOPSEP = -234
ABBA_PHASES = {0, 1, 2, 3}

NCPU = os.cpu_count()
NCPU_USE = NCPU - 2 if NCPU <= 10 else 10

def subtract_per_scan(da_A, 
                      da_B,
                      overshoot_thres
                      ):
    """
    """
    if len(states := np.unique(da_A.state)) != 1:
        raise ValueError("State must be unique.")
    
    chan = da_A.chan.data

    da_A = da_A.sortby("time")
    da_B = da_B.sortby("time")
    
    if (state := states[0]) == "ON":
        da_on = da_A
        da_off = da_B
        time_on = utils.dt_to_seconds(da_A)
        time_off = utils.dt_to_seconds(da_B)

    if state == "OFF":
        da_on = da_B
        da_off = da_A
        time_on = utils.dt_to_seconds(da_B)
        time_off = utils.dt_to_seconds(da_A)
    
    t_amb_on = da_on.temperature.data[:,None]

    idx_off_chunks = utils.consecutive_gt(time_off)

    # Now convert to numpy for calculations
    data_on = da_on.to_numpy()
    data_off = da_off.to_numpy()
    
    func_par = partial(parallel.pswsc_baseline, 
                       xargs=(data_off,
                              time_off
                              )
                       )

    data_spl = utils.split(idx_off_chunks, NCPU_USE)
    ids = np.arange(NCPU_USE)

    par_args = [(_data_spl, _ids) for _data_spl, _ids in zip(data_spl, ids)]
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        with Pool(NCPU_USE) as p:
            out = p.map(func_par, par_args)

    off_avgs = np.concatenate([x[0] for x in out], axis=0)
    off_vars = np.concatenate([x[1] for x in out], axis=0)
    off_times = np.concatenate([x[2] for x in out], axis=0)
    
    bl_avg = interp.interp1d(off_times,
                               off_avgs,
                               fill_value='extrapolate', 
                               axis=0
                               )(time_on)
    
    bl_var = interp.interp1d(off_times,
                               off_vars,
                               fill_value='extrapolate', 
                               axis=0
                               )(time_on)
    
    nomin = data_on - bl_avg 
    denom = t_amb_on - bl_avg

    tod_on = t_amb_on * nomin / denom
    tod_var_on = t_amb_on * (nomin / denom**2 - bl_avg / denom) * bl_var

    tod_std_on = np.sqrt(np.absolute(tod_var_on))

    da_rel = dc.convert.frame(da_on, "relative")
    az, el = da_rel.lon.to_numpy(), da_rel.lat.to_numpy()

    sep = np.sqrt(az**2 + el**2)
    median_sep = np.nanmedian(sep)

    mask_sep = (sep - median_sep) < overshoot_thres

    avg_on = np.nanmean(
            tod_on[mask_sep], 
            axis=0
            )
    std_on = np.sqrt(
            np.nanstd(
                tod_on[mask_sep], 
                axis=0
                )**2 + np.nanmean(
                    tod_std_on[mask_sep], 
                    axis=0)**2
                ) / np.sqrt(mask_sep.size)

    import matplotlib.pyplot as plt
    #plt.errorbar(da_A.frequency, avg_on, std_on, ls=" ", marker=".", capsize=3)
    #plt.scatter(time_on, std_on[:,0])
    #plt.errorbar(time_on[mask_sep], 
    #             avg_on[mask_sep,0], 
    #             std_on[mask_sep,0],
    #             ls=" ",
    #             marker=".",
    #             capsize=3)
    #plt.show()


    #plt.scatter(time_on, sep - median_sep)
    #plt.axhline(0)
    #plt.axhline(overshoot_thres, color="black")
    #plt.axhline(-overshoot_thres, color="black")
    #plt.xlabel("time [s]")
    #plt.ylabel(r"$\theta$ [arcsec]")
    #plt.show()

    return avg_on, std_on

def _overshoot_per_scan(dems, buff):
    """Apply overshoot removal to a single-scan DEMS."""
    if len(states := np.unique(dems.state)) != 1:
        raise ValueError("State must be unique.")

    dems = dems.sortby("time")
    idx_tot = np.arange(dems.shape[0])

    time = dems.time.data

    time_on = dems[dems.beam.data == "A"].time.data
    time_off = dems[dems.beam.data == "B"].time.data

    on = dems[dems.beam.data == "A"]
    off = dems[dems.beam.data == "B"]


    if (state := states[0]) == "ON":
        idx_bad = idx_tot[(np.absolute(dems.lon.data) > buff) & (np.absolute(dems.lat.data) > buff)]

    if state == "OFF":
        idx_bad = idx_tot[(np.absolute(dems.lon.data - CHOPSEP) > buff) & (np.absolute(dems.lat.data) > buff)]

    if len(idx_bad) > 0:
        idx_good_start = idx_bad[-1] + 1 # Start on first index where everything is good

        return dems[idx_good_start:]

    else:
        return dems

