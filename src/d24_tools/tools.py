import decode as dc
import numpy as np
import xarray as xr
import scipy.interpolate as interp
from datetime import datetime, timedelta
import scipy.interpolate as interp
from astropy.units import Quantity

from scipy.interpolate import InterpolatedUnivariateSpline

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
from d24_tools import reduce
from d24_tools import parallel
from d24_tools import dewobble
from d24_tools import atm
from functools import partial
import warnings
NCPU = os.cpu_count()
NCPU_USE = NCPU# - 2 if NCPU <= 10 else 10

def get_dir_list(path, 
                 include_pattern = "cosmos",
                 exclude_pattern = " "):

    if isinstance(exclude_pattern, int):
        exclude_pattern = [exclude_pattern]

    out_list = []

    for item in os.listdir(path):
        if include_pattern in item:
            accept = True
            for exc in exclude_pattern:
                if exc in item:
                    accept = False
                    break
            if accept:
                out_list.append(item)

    return out_list

#### Reduction and averaging
def reduce_observation_still(da):
    """
    Average a still observation, without AB chopping.
    Note that any including/excluding to the data array needs to be done BEFORE passing the array to this method.
    
    @param da_sub Data array on which potentially include/exclude operations have been performed.

    @returns Array with the average signal, for each KID.
    @returns Array with the standard deviation, for each KID.
    @returns Array with master indices for each KID.
    @returns Array with filter frequencies, in GHz, for each KID.
    """
    
    chan = da.chan.values
    freq = da.frequency.values

    spec_avg = np.nanmean(da.data, axis=0)
    spec_var = np.nanvar(da.data, axis=0)

    return spec_avg, spec_var, chan, freq    

def reduce_pswsc(da,
                 overshoot_thres = 1.5):
    """
    Average a pswsc observation over the full observation.

    @param da Data-array. 

    @returns Array with the average signal, for each KID.
    @returns Array with the standard deviation, for each KID.
    @returns Array with channel indices for each KID.
    @returns Array with frequencies, in GHz, for each KID.
    """

    # First check if there are NaN tods (just in case)
    mean_check = np.mean(da.to_numpy(), axis=0)
    da = da[:,~np.isnan(mean_check)]

    chan = da.chan.values       # Channel index [-]
    freq = da.frequency.values  # Frequency [GHz]
    
    # Step 1: despike and select only ON/OFF states (alpha/beta)
    # Returns data-array ordered by time
    # For convenience, also extract scan label data
    da_sub = reduce.despike(da)
    scan_labels = da_sub.scan.data

    # Step 2: split into A and B data-arrays and apply dewinging
    # to beam B data-array
    da_A = da_sub[da_sub.beam == "A"].sortby(
            "time"
            )
    
    da_B = reduce.dewing(
            da_sub[da_sub.beam == "B"].sortby(
                "time"
                )
            ).sortby(
                    "time"
                    )

    # Step 3: dewobbling. This should be applied AFTER 
    # despiking/dewinging, but BEFORE proper atmospheric removal
    da_B = dewobble.reject_wobble(da_A, da_B)
    
    average = np.zeros(chan.size)
    variance = np.zeros(chan.size)

    avgs_list = []
    stds_list = []

    for sclabel in set(scan_labels):
        da_A_sub = dc.select.by(
                da_A, 
                "scan", 
                include=sclabel
                ).sortby("time")

        da_B_sub = dc.select.by(
                da_B, 
                "scan", 
                include=sclabel
                ).sortby("time")

        avg_on, std_on = methods.subtract_per_scan(
                da_A_sub,
                da_B_sub,
                overshoot_thres
                )
        
        avgs_list.append(avg_on)
        stds_list.append(std_on)

    avgs_list = np.array(avgs_list)
    stds_list = np.array(stds_list)
    avg_total = np.nanmean(avgs_list, axis=0)
    std_total = np.nanmean(stds_list, axis=0)

    return avg_total, std_total, chan, freq

# Daisy reduction tools
def reduce_daisy(da, 
                 source_radius=60,
                 split_petal=True,
                 squared_eta_f=True,
                 get_R2=False,
                 R2_thres = 0.85,
                 fit_pwv = False
                 ):
    """
    Reduce a daisy scan to cube.
    This function takes a data array of a daisy scan in either beam A or B.
    For a daisy AB, please use the `reduce_daisy_AB` method.

    @param da Data array that has been despiked.
    @param conv_factor A custom conversion factor, if something other than brightness temperature is desired.

    @returns Cube containing the average per pixel
    @returns Cube containing the variance per pixel
    @returns Array with master indices for each KID.
    @returns Array with filter frequencies, in GHz, for each KID.
    """
    # First check if there are NaN tods (just in case)
    mean_check = np.mean(da.to_numpy(), axis=0)
    da = da[:,~np.isnan(mean_check)]
    
    print(colors._green(f"Reducing obsid {da.aste_obs_id} - obstable : {da.aste_obs_file}"))
    
    da = da.sel(time=da.state.isin(
        ["SCAN", "GRAD", "TRAN"]
        )).sortby("time")
    
    eta_atm_avg = None
    
    if da.long_name == "df/f":
        a, b, idx_keep, R2, eta_atm_avg = utils.Tb_mini_skydip(
                da, 
                squared_eta_f
                )
        
        if R2_thres is not None:
            a = a[R2 > R2_thres]
            b = b[R2 > R2_thres]
            idx_keep = idx_keep[R2 > R2_thres]
            R2 = R2[R2 > R2_thres]
        
        da = a*da[:,idx_keep] + b
    
    da = da.sel(time=da.state.isin("SCAN")).sortby("time")
    
    da_rel = dc.convert.frame(da_sub, "relative")
    
    az_rel, el_rel = da_rel.lon.to_numpy(), da_rel.lat.to_numpy()
    az, el = da.lon.to_numpy()/3600, da.lat.to_numpy()/3600
    
    if fit_pwv:
        print(colors._green("Fitting PWV to off-source chunks..."))
        time = utils.dt_to_seconds(da)

        mask_fit = az_rel**2 + el_rel**2 > source_radius**2
        chunk_fit = np.arange(da.time.size)[mask_fit]
        chunk_list_fit = utils.consecutive(chunk_fit)

        chunk_spl = utils.split(chunk_fit, NCPU_USE)
        ids = np.arange(NCPU_USE)

        par_args = [(_chunks, _ids) for _chunks, _ids in zip(chunk_spl, ids)] 

        func_par = partial(parallel.fit_pwv_pool,
                           xargs=(da.to_numpy(),
                                  time,
                                  da.temperature.values,
                                  da.frequency.values,
                                  1/np.sin(el /180 * np.pi),
                                  da.chan.values)
                           )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            with Pool(NCPU_USE) as p:
                out = p.map(func_par, par_args)

        pwv_fit = np.concatenate([x[0] for x in out], axis=0)
        pwv_fit_std = np.concatenate([x[1] for x in out], axis=0)
        pwv_times = np.concatenate([x[2] for x in out], axis=0)

    
    pwv_interpol = InterpolatedUnivariateSpline(pwv_times, pwv_fit, k=1)
    
    time_conv = utils.dt_to_seconds(da)
    
    pwv_interp = pwv_interpol(time_conv)

    func_par = partial(parallel.daisy_outer_petal, 
                       xargs=(time_conv,
                              az_rel,
                              el_rel, 
                              source_radius, 
                              freq))

    data_spl = np.array_split(da.to_numpy(), NCPU_USE, axis=1)
    ids = np.arange(NCPU_USE)

    par_args = [(_data_spl, _ids) for _data_spl, _ids in zip(data_spl, ids)]
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        with Pool(NCPU_USE) as p:
            out = p.map(func_par, par_args)

    src = np.concatenate([x[0] for x in out], axis=-1)
    std = np.sqrt(np.concatenate([x[1] for x in out], axis=-1))
    baseline_outer_petal = np.concatenate([x[0] for x in out], axis=-1)

    

    src_times_dt = da.time.values.astype('datetime64[ns]')
    
    az_spl = np.array_split(
            az, 
            NCPU_USE, 
            axis=0)
    el_spl = np.array_split(
            el, 
            NCPU_USE, 
            axis=0)
    time_spl = np.array_split(
            src_times_dt, 
            NCPU_USE, 
            axis=0)

    par_args = [(_az_spl, _el_spl, _time_spl) for _az_spl, _el_spl, _time_spl in zip(az_spl, el_spl, time_spl)]
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        with Pool(NCPU_USE) as p:
            out = p.map(utils.azel_to_radec, par_args)
    
    ra_deg = np.concatenate([x[0] for x in out], axis=0)
    dec_deg = np.concatenate([x[1] for x in out], axis=0)
    q_deg = np.concatenate([x[2] for x in out], axis=0)
    
    chan = da.chan.values       # Channel index [-]
    freq = da.frequency.values  # Frequency [GHz]

    if fit_pwv:
        return {
                "src"           : src,
                "std"           : std,
                "ra_deg"        : ra_deg,
                "dec_deg"       : dec_deg,
                "chan"          : chan,
                "freq"          : freq,
                "time"          : time_src,
                "pwv"           : pwv_fit,
                "pwv_std"       : pwv_fit_std,
                "pwv_time"      : pwv_conv,
                "eta_atm_skd"   : eta_atm_avg,
                "baseline_outer": baseline_outer_petal
                }
    else:
        return {
                "src"           : src,
                "std"           : std,
                "ra_deg"        : ra_deg,
                "dec_deg"       : dec_deg,
                "chan"          : chan,
                "freq"          : freq,
                "time"          : time_conv,
                "eta_atm_skd"   : eta_atm_avg,
                "baseline_outer": baseline_outer_petal
                }

def reduce_daisy_AB(da, 
                    source_radius=60, 
                    subtr="out", 
                    split_petal=True,
                    squared_eta_f=True,
                    get_R2=False,
                    R2_thres = None,
                    fit_pwv = False,
                    subtract_outer_petal = True
                    ):
    """
    Reduce a daisy AB scan to cube.

    @param da Data array in terms of dfof.
    @param conv_factor A custom conversion factor, if something other than brightness temperature is desired.

    @returns Cube containing the average per pixel
    @returns Cube containing the variance per pixel
    @returns Array with master indices for each KID.
    @returns Array with filter frequencies, in GHz, for each KID.
    """
    # First check if there are NaN tods (just in case)
    mean_check = np.mean(da.to_numpy(), axis=0)
    da = da[:,~np.isnan(mean_check)]

    print(colors._green(f"Reducing obsid {da.aste_obs_id} - obstable : {da.aste_obs_file}"))
    
    # Step 1: despike and select only SCAN state
    # Returns data-array ordered by time
    da_sub = reduce.despike(da, "SCAN").sortby("time")

    eta_atm_avg = None
    
    if da_sub.long_name == "df/f":
        # Step 1.5: get Tb-delta x relation
        da_sub_skd = reduce.despike(da, include=["SCAN", "GRAD", "TRAN"]).sortby("time")
        a, b, idx_keep, R2, eta_atm_avg = utils.Tb_mini_skydip(da_sub_skd[da_sub_skd.beam == "A"], squared_eta_f)
        
        if R2_thres is not None:
            a = a[R2 > R2_thres]
            b = b[R2 > R2_thres]
            idx_keep = idx_keep[R2 > R2_thres]
            R2 = R2[R2 > R2_thres]
        
        da_sub = a*da_sub[:,idx_keep] + b

    # Step 2: fit PWV time-dependence to beam A chops
    # We calculate such that each chop unit has average PWV
    # Then, we make linear interpolator and interpolate on PWV -> eta_atm and calcuate T*_a
    idx_A = np.arange(da_sub.time.size)[da_sub.beam == "A"]
    chunk_list_A = utils.consecutive(idx_A)

    if fit_pwv:
        print(colors._green("Fitting PWV to beam A chops..."))
        # Make parallel across time - cannot paralellise across frequency, since we need all frequencies to estimate PWV
        time = utils.dt_to_seconds(da_sub)

        chunk_A_spl = utils.split(chunk_list_A, NCPU_USE)
        ids = np.arange(NCPU_USE)

        par_args = [(_chunks_A, _ids) for _chunks_A, _ids in zip(chunk_A_spl, ids)] 

        func_par = partial(parallel.fit_pwv_pool,
                           xargs=(da_sub.to_numpy(),
                                  time,
                                  da_sub.temperature.values,
                                  da_sub.frequency.values,
                                  1/np.sin(da_sub.lat.values / 3600 /180 * np.pi),
                                  da_sub.chan.values)
                           )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            with Pool(NCPU_USE) as p:
                out = p.map(func_par, par_args)

        pwv_fit = np.concatenate([x[0] for x in out], axis=0)
        pwv_fit_std = np.concatenate([x[1] for x in out], axis=0)
        pwv_times = np.concatenate([x[2] for x in out], axis=0)

    #plt.scatter(da_sub.time[da_sub.beam=="B"], da_sub[da_sub.beam=="B",0], label="before dewing") 

    # Step 2: dewing (only beam B currently) 
    da_sub = reduce.dewing(da_sub.sortby("time"))
    
    #plt.scatter(da_sub.time[da_sub.beam=="B"], da_sub[da_sub.beam=="B",0], label="after dewing") 

    idx_A = np.arange(da_sub.time.size)[da_sub.beam == "A"]
    idx_B = np.arange(da_sub.time.size)[da_sub.beam == "B"]

    # Step 3: dewobble
    da_sub = dewobble.reject_wobble(da_sub, idx_A, idx_B)
    #plt.scatter(da_sub.time[da_sub.beam=="B"], da_sub[da_sub.beam=="B",0], label="after dewobble") 
    #plt.legend()
    #plt.show()

    chan = da_sub.chan.values       # Channel index [-]
    freq = da_sub.frequency.values  # Frequency [GHz]

    time_conv = utils.dt_to_seconds(da_sub)
    time_conv_dt = da_sub.time.values.astype('datetime64[ns]')

    t_amb = da.temperature.to_numpy()
    
    az_spl = np.array_split(da_sub.lon.to_numpy()/3600, NCPU_USE, axis=0)
    el_spl = np.array_split(da_sub.lat.to_numpy()/3600, NCPU_USE, axis=0)
    time_spl = np.array_split(time_conv_dt, NCPU_USE, axis=0)

    par_args = [(_az_spl, _el_spl, _time_spl) for _az_spl, _el_spl, _time_spl in zip(az_spl, el_spl, time_spl)]
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        with Pool(NCPU_USE) as p:
            out = p.map(utils.azel_to_radec, par_args)

    
    ra_deg = np.concatenate([x[0] for x in out], axis=0)
    dec_deg = np.concatenate([x[1] for x in out], axis=0)

    da_rel = dc.convert.frame(da_sub, "relative")
    az, el = da_rel.lon.to_numpy(), da_rel.lat.to_numpy()

    idx_A = np.arange(da_sub.time.size)[da_sub.beam == "A"]
    idx_B = np.arange(da_sub.time.size)[da_sub.beam == "B"]

    func_par = partial(parallel.daisy_AB, 
                       xargs=(time_conv,            # Time of full data array.
                              idx_A,                # Indices of beam A in full data array, after despiking.
                              idx_B,                # Indices of beam B in full data array, after despiking and dewinging
                              t_amb,                # Ambient temperature for full data array.
                              az,                   
                              el, 
                              source_radius,
                              split_petal,
                              subtract_outer_petal))

    data_spl = np.array_split(da_sub.to_numpy(), NCPU_USE, axis=1)
    ids = np.arange(NCPU_USE)

    par_args = [(_data_spl, _ids) for _data_spl, _ids in zip(data_spl, ids)]
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        with Pool(NCPU_USE) as p:
            out = p.map(func_par, par_args)

    src = np.concatenate([x[0] for x in out], axis=-1)
    std = np.sqrt(np.concatenate([x[1] for x in out], axis=-1))
    time_src = out[0][2]
    baseline = np.concatenate([x[3] for x in out], axis=-1)
    
    pwv_interpol = InterpolatedUnivariateSpline(pwv_times, pwv_fit, k=1)
    
    pwv_interp = pwv_interpol(time_src)

    baseline_outer_petal = None

    if subtract_outer_petal:
        baseline_outer_petal = np.concatenate([x[4] for x in out], axis=-1)

    if fit_pwv:
        return {
                "src"           : src,
                "std"           : std,
                "ra_deg"        : ra_deg,
                "dec_deg"       : dec_deg,
                "chan"          : chan,
                "freq"          : freq,
                "time"          : time_src,
                "pwv"           : pwv_fit,
                "pwv_std"       : pwv_fit_std,
                "pwv_time"      : pwv_times,
                "eta_atm_skd"   : eta_atm_avg,
                "baseline"      : baseline,
                "baseline_outer": baseline_outer_petal
                }
    else:
        return {
                "src"           : src,
                "std"           : std,
                "ra_deg"        : ra_deg,
                "dec_deg"       : dec_deg,
                "chan"          : chan,
                "freq"          : freq,
                "time"          : time_src,
                "eta_atm_skd"   : eta_atm_avg,
                "baseline"      : baseline,
                "baseline_outer": baseline_outer_petal
                }

def reduce_raster(da, source_radius, return_extra_pointing_data=False):
    """
    Reduce a raster scan to subtracted TODs.
    """
    
    return reduce_daisy(da, source_radius, return_extra_pointing_data)
    
def rebin(avg, std, freq, freq_rebin, dfreq_rebin, rebin_method="uniform", filter_negative=False):
    out_sh_l = [x for x in avg.shape[:-1]]
    out_sh_l.append(freq_rebin.size)
    out_shape = tuple(out_sh_l)
    
    avg_cube_rebinned = np.zeros(out_shape).T
    std_cube_rebinned = np.zeros(out_shape).T
    for ii, cfreq in enumerate(freq_rebin):
        mask = np.absolute(freq - cfreq) < dfreq_rebin
        avg_bin = avg.T[mask]
        std_bin = std.T[mask]

        if filter_negative:
            if len(avg_bin.shape) == 3:
                ax = (1,2)
            else:
                ax = (1,)
            check_arr = np.nanmean(avg_bin, axis=ax)
            idx_pos = check_arr > 0

            avg_bin = avg_bin[idx_pos]
            std_bin = std_bin[idx_pos]

        if rebin_method == "uniform":
            avg_out = np.nanmean(avg_bin, axis=0)
            std_avg = np.nanstd(avg_bin, axis=0)
            std_out = np.sqrt((np.nanmean(std_bin, axis=0)**2 + std_avg**2) / std_bin.shape[0])
        else:
            avg_out, std_out = utils.avg_inv_var(avg_bin.T, std_bin.T) 
        avg_cube_rebinned[ii] = avg_out.T
        std_cube_rebinned[ii] = std_out.T

    return avg_cube_rebinned.T, std_cube_rebinned.T

def coadd(avg_list, std_list, freq_list, chan_list, weighting_scheme="inv_var"):
    assert((tot_size := len(avg_list)) == len(std_list))
    assert(tot_size == len(freq_list))
    assert(tot_size == len(chan_list))

    for ii in range(tot_size):         
        chan = chan_list[ii]                     
        freq = freq_list[ii]                     
                                             
        if not ii:                            
            tot_chan = chan                  
            continue                         
                                             
        tot_chan = np.union1d(tot_chan, chan)

    coadd_shape = [x for x in avg_list[0].shape[:-1]]
    coadd_shape.insert(0, tot_size)
    coadd_shape.insert(1, tot_chan.size)
    coadd_shape = tuple(coadd_shape)

    tot_std = np.empty(coadd_shape)
    tot_avg = np.empty(coadd_shape)
    tot_freq = np.empty((tot_size, tot_chan.size))

    tot_std[:] = np.nan
    tot_avg[:] = np.nan
    tot_freq[:] = np.nan

    for ii in range(tot_size):
        # Transpose so that spectral axis is slowest
        avg = avg_list[ii].T
        std = std_list[ii].T
        freq = freq_list[ii]
        chan = chan_list[ii]

        idxs, idx_in_total, idx_in_current = np.intersect1d(tot_chan, chan, assume_unique=True, return_indices=True)

        tot_std[ii,idx_in_total] = std
        tot_avg[ii,idx_in_total] = avg
        tot_freq[ii,idx_in_total] = freq

    if weighting_scheme == "inv_var":
        weights = np.nansum(1 / tot_std**2, axis=0)

        tot_avg = np.nansum(tot_avg / tot_std**2, axis=0) / weights
        tot_std = np.sqrt(1 / weights)
        tot_freq = np.nanmean(tot_freq, axis=0)

    else:
        tot_avg = np.nanmean(tot_avg, axis=0)
        tot_std = np.nanmean(tot_std, axis=0)
        tot_freq = np.nanmean(tot_freq, axis=0)

    return tot_avg.T, tot_std.T, tot_chan, tot_freq

