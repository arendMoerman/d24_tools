import decode as dc
from astropy.stats import sigma_clip
import numpy as np
import xarray as xr
import scipy.interpolate as interp
from datetime import datetime, timedelta
from astropy.units import Quantity
import copy
from scipy.stats import binned_statistic_2d
from multiprocessing import Pool
import os
import matplotlib.pyplot as plt
from sys import getsizeof
from gc import collect
import pickle
from scipy.optimize import least_squares
from d24_tools import colors
from d24_tools import methods
from d24_tools import utils
from d24_tools import parallel
from functools import partial
import warnings
NCPU = os.cpu_count()
NCPU_USE = NCPU - 2 if NCPU <= 10 else 10
FNUM = 50
CUTOFF = 0.1
BUFF = 100
from scipy.signal import butter, filtfilt

FSAMPLE = 2e9 / 2**19 / 24

FDEF = 5.000018697350202

C = 293

def time_in_seconds(this_data, data):
    time_differences = (this_data.time) - (data.time[0])
    time_seconds = np.array(time_differences.astype('timedelta64[ns]').astype(float)*(10**(-9)))
    return time_seconds

def butter_lowpass_filter(data, cutoff, order=2):
    """
    Apply a Butterworth low-pass filter to 1D time series data.
    
    Parameters:
        data (np.ndarray): The input time series data.
        cutoff (float): The cutoff frequency of the filter (Hz).
        order (int): The order of the filter.

    Returns: 
        np.ndarray: The filtered time series. 
    """
    
    nyquist = 0.5 * FSAMPLE
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    filtered_data = filtfilt(b, a, data, axis=0)
    return filtered_data

def get_amp_offset(phase, t, y):
    predicted_sine = np.sin(2*np.pi*FDEF*t + phase)

    M = np.vstack([predicted_sine, np.ones_like(predicted_sine)]).T
    beta, _, _, _ = np.linalg.lstsq(M, y, rcond=None)
    A = beta[0]
    o = beta[1]
    return A, o, predicted_sine

def fit_phi(phase, t, y):
    """
    params = [frequency, phase_1, phase_2, ..., phase_K]
    """
    A, o, predicted_sine = get_amp_offset(phase, t, y)

    # Build fitted model y_fit: (C × T_k)
    y_fit = A[None, :] * predicted_sine[:, None] + o[None, :]

    return (y_fit - y).ravel()

def get_phase(da_B, idx_keep):
    trend = butter_lowpass_filter(da_B, cutoff=1)[idx_keep]
    da_B_fit = da_B[idx_keep]
    detrended_beamB = da_B_fit.data - trend
    time_seconds = time_in_seconds(da_B_fit, da_B_fit) 
    t = np.asarray(time_seconds)

    ### phase guess
    # m goes as 1- y/(C-a)
    m_est = np.asarray(1 - detrended_beamB / (C - trend))
    # is first half is lower than second half, phase should be pi
    t_folded = time_seconds % (1/FDEF)
    first_half = t_folded < np.mean(t_folded)
    if np.mean(m_est[first_half,:]) < np.mean(m_est[~first_half,:]):
        p0 = np.pi
    else:
        p0 = 0

    res = least_squares(fit_phi, p0, args=(t, m_est))
    A, o, _ = get_amp_offset(res.x, t, m_est)
    return res.x, trend, A, o
#def reject_wobble(da_A, da_B):
def reject_wobble(da, idx_A, idx_B):
    """
    I only use alpha nods (beam B off source) if pswsc.
    If daisy AB, use entire beam B.
    """
    print(colors._green("Dewobbling beam B..."))

    da_A = da[idx_A]
    da_B = da[idx_B]    

    #if "pswsc" in da_A.aste_obs_file:
    #    da_B_use = da_B[da_B.state == "ON"]
    #    # Also need to remove first/last 10 points in each state
    #    time_ON = time_in_seconds(da_B[da_B.state == "ON"], 
    #                              da_B[da_B.state == "ON"])
    #    chunks_B_ON = utils.consecutive_gt(time_ON, stepsize=10)
    #    idx_keep = []
    #    for chu_B in chunks_B_ON:
    #        idx_keep.append(chu_B[BUFF:-BUFF])
    #    idx_keep = [x for y in idx_keep for x in y]
    #else:
    #    da_B_use = da_B
    #    idx_keep = np.arange(da_B.data.shape[0])
    da_B_use = da_B
    idx_keep = np.arange(da_B.data.shape[0])
        

    phi, baseline_B, A, o = get_phase(da_B_use, idx_keep)

    da_B_use = da_B_use[idx_keep]

    freqs = da_B_use.chan.frequency

    time_seconds = time_in_seconds(da_B_use, da_B_use) 
    t = np.asarray(time_seconds)
    t_full = np.asarray(time_in_seconds(da_B, da_B_use))
    
    #plt.scatter(t, da_B_use[:,0] - baseline_B[:,0])
    #plt.scatter(t, da_B_use[:,0])# - baseline_B[:,0])
    #plt.scatter(t, baseline_B[:,0])
    #plt.xlabel("time [s]")
    #plt.ylabel(r"$T^\star_\mathrm{a}$")
    #plt.show()

    modulation = A[None,:]*np.sin(2*np.pi*FDEF*t_full[:,None] + phi) + o[None,:]

    da[idx_B] = (da_B - (1-modulation) * C)/modulation
    
    #modulation_check = A[None,:]*np.sin(2*np.pi*FDEF*t[:,None] + phi) + o[None,:]
    #dewobbled_check = (da_B_use - (1-modulation_check) * C)/modulation_check
    #fig, ax = plt.subplots(1,1)
    #ax.hist((da_B_use-baseline_B)[:,10], bins=100, alpha=0.7)
    #ax.hist((dewobbled_check-baseline_B)[:,10], bins=100, alpha=0.7)
    #plt.show()

    #phi_c, baseline_B_c, A_c, o_c = get_phase(dewobbled, idx_keep)
    #modulation_0 = A_c[None,:]*np.sin(2*np.pi*FDEF*t[:,None] + phi_c) + o_c[None,:]
    
    x = (t%(1/FDEF))
    #x = (t_full%(1/FDEF))

    #plt.hist((da_B-baseline_B)[:,0], bins=100)
    #plt.show()
    #yfit = (da_B_use - C) * (modulation_check - 1)
    #yfit_0 = (dewobbled_check - C) * (modulation_0 - 1)

    #fig, ax = plt.subplots(2,1)
    #ax[0].scatter(x, (da_B_use-baseline_B)[:,10], c = time_seconds // (1/FDEF), cmap = "viridis", marker = ".")
    #ax[0].scatter(x, yfit[:,10], c = "grey", cmap = "viridis", marker = ".")
    #ax[1].scatter(x, (dewobbled_check-baseline_B)[:,10], c = time_seconds // (1/FDEF), cmap = "viridis", marker = ".")
    #ax[1].scatter(x, yfit_0[:,10], c = "grey", cmap = "viridis", marker = ".")
    #ax[1].set_xlabel("folded time [s]")
    #ax[0].set_ylabel("$T^\star_\mathrm{a}$ - baseline [K]")
    #ax[1].set_ylabel("$T^\star_\mathrm{a}$ - baseline [K]")
    #plt.show()
    
        
    return da
