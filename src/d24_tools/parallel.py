import decode as dc
import numpy as np
import xarray as xr
import scipy.interpolate as interp
from datetime import datetime, timedelta
import scipy.interpolate as interp
from scipy.optimize import curve_fit
from astropy.units import Quantity
import copy
from scipy.stats import binned_statistic_2d, linregress
from multiprocessing import Pool
import os
#import matplotlib.pyplot as plt
from tqdm import tqdm
import sys
import warnings
from astropy.modeling import models, fitting           
from astropy.stats import sigma_clip
from astropy.utils.exceptions import AstropyUserWarning

from d24_tools import colors
from d24_tools import methods
from d24_tools import utils
from d24_tools import atm

from functools import partial

import random

NPARFIT = 6                                                                                                  

def daisy_AB(par_args, xargs):
    data, thread_id = par_args

    time, idx_A, idx_B, T_amb, az, el, outer_radius, split_petal, subtract_outer_petal = xargs
    chunk_list_B = utils.consecutive(idx_B)
    chunk_list_A = utils.consecutive(idx_A)

    if len(chunk_list_B) % 2:
        chunk_list_B.pop(-1)
    
    sky_means = np.zeros((len(chunk_list_B), data.shape[1]))
    sky_times = np.zeros(len(chunk_list_B))

    if thread_id == 0:
        print(colors._green("Calculating chop averages in off-source beam..."))
    for i, idx_chunk_sky in parallel_iterator(
            enumerate(chunk_list_B), thread_id, len(chunk_list_B)
            ):
        if (i + 1) == len(chunk_list_B):
            continue

        # Get counterpoint
        avg_points, avg_time = utils.get_avg_counterpoints([data[chunk_list_B[i]], data[chunk_list_B[i+1]]], 
                                                           [time[chunk_list_B[i]], time[chunk_list_B[i+1]]])
        sky_means[i] = avg_points 
        sky_times[i] = avg_time
        
        #sky_means[i] = np.nanmean(data[chunk_list_B[i]], axis=0)
        #sky_times[i] = np.nanmean(time[chunk_list_B[i]])

        #if thread_id == 0:
        #    import matplotlib.pyplot as plt

        #    plt.scatter(time[chunk_list_B[i]], data[chunk_list_B[i],0]-sky_means[i,0])
        #    plt.scatter(time[chunk_list_B[i]], data[chunk_list_B[i],10]-sky_means[i,10])
        #    plt.scatter(time[chunk_list_B[i]], data[chunk_list_B[i],20]-sky_means[i,20])
        #    plt.show()


    sky_means = sky_means[~np.isnan(sky_times)]
    sky_times = sky_times[~np.isnan(sky_times)]

    sky_means = sky_means[sky_times > 0]
    sky_times = sky_times[sky_times > 0]


    bl_avg = interp.interp1d(sky_times, 
                                     sky_means, 
                                     fill_value='extrapolate', 
                                     axis=0)(time[idx_A])
    
    #bl_var = interp.interp1d(sky_times, 
    #                                 sky_vars, 
    #                                 fill_value='extrapolate', 
    #                                 axis=0)(time[idx_A])

    t_amb_on = T_amb[idx_A,None] 
    nomin = data[idx_A] - bl_avg 
    denom = t_amb_on - bl_avg

    #tod_on = t_amb_on * nomin / denom
    tod_on = data[idx_A] - bl_avg 
    #tod_var_on = np.absolute(t_amb_on * (nomin / denom**2 - bl_avg / denom))**2 * bl_var
    
    if not subtract_outer_petal:
        if thread_id == 0:
            print("", end="\n")
            print(tod_on.shape, idx_A.shape)
        
        return tod_on, tod_on, time[idx_A], bl_avg
    
    #for ii in range(tod_on.shape[-1]):
    #    mask_clip_tod = ~sigma_clip(tod_on[:,ii], sigma=3, maxiters=1, masked=True).recordmask
    #    tod_on = tod_on
    
    mask = (az[idx_A]**2 + el[idx_A]**2) > outer_radius**2
    
    src_outer_idx = np.arange(tod_on.shape[0])[mask]
    
    chunk_list_outer = utils.consecutive(src_outer_idx)

    outer_reject = 0
    chunk_reject = []

    N_accept = 2

    outer_means = []
    outer_vars = []
    outer_times = []

    if thread_id == 0:
        print(colors._green("Calculating outer-petal averages in on-source beam..."))
    for i, idx_chunk_outer in parallel_iterator(
            enumerate(chunk_list_outer), thread_id, len(chunk_list_outer)
            ):

        if ((N_chunk := idx_chunk_outer.size) < N_accept):
            outer_reject += 1
            chunk_reject.append(idx_chunk_outer)
            continue


        size_subchunk = idx_chunk_outer.size 
        #size_subchunk = N_accept
        
        if not split_petal:
            outer_means.append(np.nanmean(tod_on[idx_chunk_outer,:], axis=0))
            outer_vars.append(np.nanvar(tod_on[idx_chunk_outer,:], axis=0) * 2)
            outer_times.append(np.nanmean(time[idx_A][idx_chunk_outer]))

        else:
            outer_means.append(np.nanmean(tod_on[idx_chunk_outer[:size_subchunk//2],:], axis=0))
            outer_means.append(np.nanmean(tod_on[idx_chunk_outer[-size_subchunk//2:],:], axis=0))
         
            # Factor two to take into account A-B subtraction in variance
            outer_vars.append(np.nanvar(tod_on[idx_chunk_outer[:size_subchunk//2],:], axis=0) * 2)
            outer_vars.append(np.nanvar(tod_on[idx_chunk_outer[-size_subchunk//2:],:], axis=0) * 2)
            
            outer_times.append(np.nanmean(time[idx_A][idx_chunk_outer[:size_subchunk//2]]))
            outer_times.append(np.nanmean(time[idx_A][idx_chunk_outer[-size_subchunk//2:]]))
    if thread_id == 0:
        print(f"Number of rejected chunks: {outer_reject}")

    outer_inner_interp = interp.interp1d(outer_times, 
                                     outer_means, 
                                     fill_value='extrapolate', 
                                     axis=0)(time[idx_A])
    outer_inner_var_interp = interp.interp1d(outer_times, 
                                     outer_vars, 
                                     fill_value='extrapolate', 
                                     axis=0)(time[idx_A])
    
    tod_on_sub = tod_on - outer_inner_interp
    tod_var_sub = outer_inner_var_interp

    if thread_id == 0:
        print("", end="\n")

    return tod_on_sub, tod_var_sub, time[idx_A], bl_avg, outer_inner_interp

def daisy_outer_petal(par_args, xargs):
    data, thread_id = par_args

    time_conv, az, el, outer_radius, freq = xargs
    
    mask = (az**2 + el**2) > outer_radius**2
    src = data
    src_outer = src[mask,:]

    frac_inner = 1 - src_outer.shape[0] / src.shape[0]

    src_outer_idx = np.arange(src.shape[0])[mask]
    chunk_list_outer = utils.consecutive(src_outer_idx)
    num_points_chunk = np.array([x.size for x in chunk_list_outer])
    frac_long = num_points_chunk[num_points_chunk > 80].size / len(chunk_list_outer)

    if not thread_id:
        print(colors._yellow(f"Inner petal fraction = {frac_inner:.3f}"))
    
    outer_means = []
    outer_vars = []
    outer_times = []

    outer_reject = 0
    chunk_reject = []

    N_accept = 2

    mean_check_first = []
    mean_check_latter = []

    if thread_id == 0:
        print(colors._green("Calculating outer-petal averages..."))
    for i, idx_chunk_outer in parallel_iterator(
            enumerate(chunk_list_outer), thread_id, len(chunk_list_outer)
            ):

        if ((N_chunk := idx_chunk_outer.size) < N_accept) & (thread_id == 0):
            outer_reject += 1
            chunk_reject.append(idx_chunk_outer)
            continue

        outer_means.append(np.nanmean(src[idx_chunk_outer[:N_accept//2],:].data, axis=0))
        outer_means.append(np.nanmean(src[idx_chunk_outer[-N_accept//2:],:].data, axis=0))

        outer_vars.append(np.nanvar(src[idx_chunk_outer[:N_accept//2],:].data, axis=0) / (N_accept//2))
        outer_vars.append(np.nanvar(src[idx_chunk_outer[-N_accept//2:],:].data, axis=0) / (N_accept//2))
        
        outer_times.append(np.nanmean(time_conv[idx_chunk_outer[:N_accept//2]], axis=0))
        outer_times.append(np.nanmean(time_conv[idx_chunk_outer[-N_accept//2:]], axis=0))

    outer_means = np.array(outer_means)
    outer_vars = np.array(outer_vars)
    outer_times = np.array(outer_times)

    bl_avg = interp.interp1d(outer_times, 
                             outer_means, 
                             fill_value='extrapolate', 
                             axis=0)(time_conv)
    bl_var = interp.interp1d(outer_times, 
                             outer_vars, 
                             fill_value='extrapolate', 
                             axis=0)(time_conv)
    
    tod_on = data - bl_avg 
    tod_var_on = bl_var 

    if thread_id == 0:
        print("", end="\n")

    return tod_on, tod_var_on, bl_avg

def pswsc_baseline(par_args, xargs):
    idx_off_chunks, thread_id = par_args
    data_off, time_off = xargs
    off_avgs = np.zeros(
            (len(
                idx_off_chunks
                ), 
             data_off.shape[1]
             )
            )
    
    off_vars = np.zeros(
            (len(
                idx_off_chunks
                ), 
             data_off.shape[1]
             )
            )
    
    off_times = np.zeros(
            len(
                idx_off_chunks
                )
            )

    if thread_id == 0:
        print(colors._green("Calculating chop averages in off-source beam..."))
    for i, idx_off in parallel_iterator(
            enumerate(idx_off_chunks), thread_id, len(idx_off_chunks)
            ):
        off_avgs[i] = np.nanmean(
                data_off[idx_off,:], 
                axis=0
                )
        
        off_vars[i] = np.nanvar(
                data_off[idx_off,:], 
                axis=0
                ) / idx_off.size

        off_times[i] = np.nanmean(time_off[idx_off])

    off_avgs = np.array(off_avgs)
    off_vars = np.array(off_vars)
    off_times = np.array(off_times)

    return off_avgs, off_vars, off_times

def make_cube_pool_inv_var(par_args, xargs):
    tod, std, chan, freq, thread_id = par_args
    ra, dec, nbins = xargs
    
    avg_cube = np.zeros((nbins, nbins, freq.size))                      
    std_cube = np.zeros((nbins, nbins, freq.size))                      
                                                                        
    if thread_id == 0:
        print(colors._green("Converting tods to cube..."))
    for idx, (ch, fr) in parallel_iterator(
            enumerate(zip(chan, freq)), thread_id, chan.size
            ):
        hist_weights = binned_statistic_2d(                                 
                ra,                                                 
                dec,                                                
                1 / std[:,idx]**2,                                             
                bins=nbins,
                statistic="sum"
                )                                                       
                                                                        
        ra_binned = hist_weights.x_edge                                     
        dec_binned = hist_weights.y_edge                                    
                                                                        
        ra_bincenters = (ra_binned[:-1] + ra_binned[1:]) / 2            
        dec_bincenters = (dec_binned[:-1] + dec_binned[1:]) / 2         
                                                                        
        hist_w_sum = binned_statistic_2d(                             
                ra,                                                 
                dec,                                                
                tod[:,idx] / std[:,idx]**2,                                             
                bins=nbins,
                statistic="sum"
                )                                                       
        
        std_tot = np.sqrt(1 / hist_weights.statistic)
        map_tot = hist_w_sum.statistic / hist_weights.statistic

        avg_cube[:,:,idx] = map_tot
        std_cube[:,:,idx] = std_tot

    return avg_cube, std_cube, ra_bincenters, dec_bincenters

def make_cube_pool(par_args, xargs):
    tod, std, chan, freq, thread_id = par_args
    ra, dec, nbins = xargs
    
    avg_cube = np.zeros((nbins, nbins, freq.size))                      
    std_cube = np.zeros((nbins, nbins, freq.size))                      
                                                                        
    if thread_id == 0:
        print(colors._green("Converting tods to cube..."))
    for idx, (ch, fr) in parallel_iterator(
            enumerate(zip(chan, freq)), thread_id, chan.size
            ):
        hist_avg = binned_statistic_2d(                                 
                ra,                                                 
                dec,                                                
                tod[:,idx],                                             
                bins=nbins                                              
                )                                                       
                                                                        
        ra_binned = hist_avg.x_edge                                     
        dec_binned = hist_avg.y_edge                                    
                                                                        
        ra_bincenters = (ra_binned[:-1] + ra_binned[1:]) / 2            
        dec_bincenters = (dec_binned[:-1] + dec_binned[1:]) / 2         
                                                                        
        hist_tod_std = binned_statistic_2d(                             
                ra,                                                 
                dec,                                                
                std[:,idx],                                             
                bins=nbins                                              
                )                                                       
        
        hist_avg_std = binned_statistic_2d(
                ra,
                dec,
                tod[:,idx],
                bins=nbins,
                statistic="std"
                )

        hist_count = binned_statistic_2d(
                ra,
                dec,
                tod[:,idx],
                bins=nbins,
                statistic="count"
                )

        std_tot = np.sqrt((hist_tod_std.statistic**2 + hist_avg_std.statistic**2) / hist_count.statistic)
        map_tot = hist_avg.statistic


        avg_cube[:,:,idx] = map_tot
        std_cube[:,:,idx] = std_tot

    return avg_cube, std_cube, ra_bincenters, dec_bincenters

def fit_gauss_pool(par_args, xargs):                                                                   
    avg_cube, std_cube, thread_id = par_args 
    ra, dec, double_gauss = xargs

    thres = -10

    ptot = np.zeros((NPARFIT, avg_cube.shape[-1]))                                                               
    ptot_2gauss = np.zeros((NPARFIT, avg_cube.shape[-1]))                                                               
    
    ptot_std = np.zeros((NPARFIT, avg_cube.shape[-1]))                                                               
    ptot_2gauss_std = np.zeros((NPARFIT-2, avg_cube.shape[-1]))                                                               
    
    model_array = np.zeros(avg_cube.shape)                                                                    
    bounds = {                                                                                                
            "amplitude" : [0, np.inf],                                                                        
            "x_mean"    : [np.nanmin(ra), np.nanmax(ra)],                                                     
            "y_mean"    : [np.nanmin(dec), np.nanmax(dec)],                                                   
            "x_stddev"  : [0, np.inf],                                                                        
            "y_stddev"  : [0, np.inf],                                                                        
            "theta"     : [-np.pi/2, np.pi/2]                                                                 
            }                                                                                                 
    if thread_id == 0:
        print(colors._green("Fitting Gauss to cube..."))
                                                                                                              
    for ii in parallel_iterator(range(avg_cube.shape[-1]), thread_id, avg_cube.shape[-1]):                                                                
        mask_f = avg_cube[:,:,ii] >= (10**(thres/20) * np.nanmax(avg_cube[:,:,ii]))
        
        p0 = utils.calc_estimates(avg_cube[:,:,ii]*mask_f, ra, dec)                                                        

        p_init = models.Gaussian2D(*p0, bounds=bounds)                                                        
        fit_p = fitting.TRFLSQFitter()                                                                         
        #fit_p = fitting.LMLSQFitter()                                                                         
        
        if double_gauss:
            def tie_x(model):
                return model.x_mean_0
            def tie_y(model):
                return model.y_mean_0

            p0_2gauss = [p0[0]*0.1, p0[1], p0[2], p0[3]*4, p0[4]*4, 1e-8]
            p_2gauss_init = models.Gaussian2D(*p0_2gauss, bounds=bounds)                                                        
            
            p_2gauss_init.x_mean.tied = tie_x
            p_2gauss_init.y_mean.tied = tie_y

            comp_model = p_init + p_2gauss_init

        else:
            comp_model = p_init
                                                                                                              
        with warnings.catch_warnings():                                                                       
            # Ignore model linearity warning from the fitter                                                  
            warnings.filterwarnings('ignore', message='Model is linear in parameters',                        
                                    category=AstropyUserWarning)                                              
            #try:
            p = fit_p(comp_model, ra, dec, z=avg_cube[:,:,ii], weights=1/std_cube[:,:,ii], filter_non_finite=True)
            #    print(p)
            #except:
            #    print(p0_2gauss)

            perr = np.sqrt(np.diag(fit_p.fit_info["param_cov"]))
            pfit = np.squeeze([getattr(p, par).value for par in p.param_names])                                      
            if pfit[3] < pfit[4]:
                temp = pfit[3]
                pfit[3] = pfit[4]
                pfit[4] = temp
                pfit[5] += np.pi/2
            pfit[3] *= 2 * np.sqrt(2 * np.log(2))                                                             
            pfit[4] *= 2 * np.sqrt(2 * np.log(2))                                                             
            pfit[5] *= 180/np.pi
        ptot[:,ii] = pfit[:NPARFIT]                                                                                     
        ptot_std[:,ii] = perr[:NPARFIT]                                                                                
        
        if double_gauss:
            ptot_2gauss[:,ii] = pfit[NPARFIT:]                                                                                     
            ptot_2gauss_std[:,ii] = perr[NPARFIT:]                                                                                
        
        model_array[:,:,ii] = p(ra, dec)                                                                      
                                                 
    return ptot, ptot_std, ptot_2gauss, ptot_2gauss_std, model_array

def Tb_mini_skydip_pool(par_args, xargs):
    data, gamma, eta_fdf, freq, thread_id = par_args
    eta_atm, Tatm = xargs

    a = np.zeros(freq.size)                                                                
    b = np.zeros(freq.size)                                                                
    R2 = np.zeros(freq.size)
    if thread_id == 0:
        print(colors._green("Calibrating to mini skydip..."))
                                                                                                              
    for ii in parallel_iterator(range(freq.size), thread_id, freq.size):                                                            
        eta_atm_los = np.nansum(eta_fdf[:,ii]*eta_atm, axis=1) / gamma[ii]                 
                                                                                           
        Tb = (1 - eta_atm_los) * Tatm                                                      
        result = linregress(data[:,ii], Tb)                                 
        
        a[ii] = result.slope                                                               
        b[ii] = result.intercept                                                           

        model = a[ii] * data[:,ii] + b[ii]

        SSR = np.nansum((model - Tb)**2)
        SST = np.nansum((model - np.nanmean(model))**2)

        R2[ii] = 1 - SSR / SST
    return a, b, R2

def fit_pwv_pool(par_args, xargs):
    """
    Parallel in time
    """
    chunks_A, thread_id = par_args 
    data, time, T_amb, freq, csc_el, chan = xargs
    
    eta_f, chan_toptica, f0_scan, freq_toptica = utils.load_filters()
    eta_f = eta_f**2
    
    idxs, idx_in_toptica, idx_in_obs = np.intersect1d(chan_toptica, chan, assume_unique=True, return_indices=True)
                                                                                                                  
    chan_toptica = chan_toptica[idx_in_toptica]                                                                   
    eta_f = eta_f[:,idx_in_toptica] # PASS TO fit_func                                                            
    f0_scan = f0_scan[idx_in_toptica]                                                                             
                                                                                                                  
    # Just to be sure, also select intersect in obs                                                               
    chan = chan[idx_in_obs]                                                                                       
    freq = freq[idx_in_obs]                                                                                       
    data = data[:,idx_in_obs]                                                                             
    
    gamma = np.nansum((eta_f[:-1] + eta_f[1:]) / 2 * np.diff(freq_toptica)[:,None], axis=0)                       
    
    eta_fdf = (eta_f[:-1] + eta_f[1:]) / 2 * np.diff(freq_toptica)[:,None]
    
    f_toptica_mid = (freq_toptica[1:] + freq_toptica[:-1]) / 2                             
    
    mask_freq = freq > 250

    gamma = gamma[mask_freq]
    eta_fdf = eta_fdf[:,mask_freq]
    freq = freq[mask_freq]
    data = data[:, mask_freq]
    
    
    atm_interp = atm.get_eta_atm()


    pwv_arr = np.zeros(len(chunks_A))
    pwv_std_arr = np.zeros(len(chunks_A))
    time_arr = np.zeros(len(chunks_A))

    for ii in parallel_iterator(range(len(chunks_A)), thread_id, len(chunks_A)):
        data_chunk = data[chunks_A[ii]]
        time_chunk = np.nanmean(time[chunks_A[ii]])
        T_amb_chunk = np.nanmean(T_amb[chunks_A[ii]])
        csc_chunk = np.nanmean(csc_el[chunks_A[ii]])

        spec_deshima = np.nanmean(data_chunk, axis=0)
        
        pwv0 = 1.5 
        
        fit_func = partial(atm_avg_filter, args=(f_toptica_mid, eta_fdf, gamma, T_amb_chunk, csc_chunk, atm_interp)) 
        popt, pcov = curve_fit(fit_func, freq, spec_deshima, p0=pwv0, bounds=((0.1,), (5.5,)))

        pwv_arr[ii] = np.squeeze(popt )   
        pwv_std_arr[ii] = np.sqrt(np.squeeze(pcov))
        time_arr[ii] = time_chunk

        #eta_atm = atm_interp(f_toptica_mid, popt).squeeze()**csc_chunk
        #eta_atm_avg = np.nansum(eta_fdf * eta_atm[:,None], axis=0) / gamma
        
    return pwv_arr, pwv_std_arr, time_arr

def atm_avg_filter(freq, pwv, args):
    f_toptica_mid, eta_fdf, gamma, T_amb, csc_el, atm_interp = args
    
    eta_atm = atm_interp(f_toptica_mid, pwv).squeeze()**csc_el
    eta_atm_avg = np.nansum(eta_fdf * eta_atm[:,None], axis=0) / gamma

    return (1 - eta_atm_avg) * T_amb


def parallel_iterator(iterator, thread_idx, size):
    if not thread_idx:
        return tqdm(iterator, total=size, colour="green", leave=False, position=0)
    else:
        return iterator
