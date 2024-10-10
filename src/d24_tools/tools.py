import decode as dc
import numpy as np
import xarray as xr
import scipy.interpolate as interp
from datetime import datetime, timedelta

import copy
from d24_tools import colors
from d24_tools import atmosphere
from d24_tools import methods
from functools import partial

def _green(string, bold=True):
    """
    Formatter for green terminal output.

    @param string String to be formatted.

    @returns Formatted string
    """
    if bold:
        return colors.GREEN + colors.BOLD + string + colors.END
    return colors.GREEN + string + colors.END

def _yellow(string, bold=True):
    """
    Formatter for yellow terminal output.

    @param string String to be formatted.

    @returns Formatted string
    """
    if bold:
        return colors.YELLOW + colors.BOLD + string + colors.END
    return colors.YELLOW + string + colors.END


def remove_bad_indices(da, indices):
    """
    Remove a list/array of master indices from a data array by setting them to NaN.
    """

    master_id = da.chan.values

    removed,_,to_remove = np.intersect1d(indices, master_id, assume_unique=True, return_indices=True)

    da[to_remove] = np.nan

    return da

def despike(da, include=["ON", "OFF"], by="state"):
    """
    Despike a TOD and return despiked data array, given an inclusion string/list of strings.

    @param da Data array containing TOD
    @param include String or list of strings containing inclusion parameters.
        For pswsc observations, ["ON", "OFF"] is recommended.
        For daisy scans or still AB chopping, use "SCAN".
    @param by Label to include by. Default is 'state'
    
    @returns Despiked data array with only the included labels
    """

    # Select the part of the observation in the OFF position, so that beam A is looking at the atmosphere
    da_sub = dc.select.by(da, by, include=include)
    
    idx_A = np.squeeze(np.argwhere(da_sub.beam.data == "A"))
    idx_B = np.squeeze(np.argwhere(da_sub.beam.data == "B"))

    all_idx = np.arange(da_sub.shape[0])
    
    assert(all_idx.size == (idx_A.size + idx_B.size))
    
    chunk_list_A = methods._consecutive(idx_A)
    chunk_list_B = methods._consecutive(idx_B)

    chunk_list_despiked_A = []
    chunk_list_despiked_B = []

    print(_green("despiking TOD..."))
    for idx_chunk_A in chunk_list_A:
        if idx_chunk_A.size >= 3:
            chunk_list_despiked_A.append(idx_chunk_A[1:-1])
        
    for idx_chunk_B in chunk_list_B:
        if idx_chunk_B.size >= 3:
            chunk_list_despiked_B.append(idx_chunk_B[1:-1])

    chunk_list_despiked_A = np.array([x for xs in chunk_list_despiked_A for x in xs])
    chunk_list_despiked_B = np.array([x for xs in chunk_list_despiked_B for x in xs])
    
    despiked_indices = np.concatenate((chunk_list_despiked_A, chunk_list_despiked_B))

    da_despiked = da_sub[despiked_indices]

    return da_despiked

def remove_overshoot(da_sub, buff=0.5):
    """
    Remove overshoot in chunks of data array of pswsc observation type.
    Essentially, it only selects the indices inside the data array for which the dAz and dEl agree with the given offset.
    
    @param da_sub Data array that has already been despiked.
    @param buff Range of angle in arcsec around dAz and dEl in which to select good points. Default is 0.5
    
    @returns List (or numpy array if idx_A was flat) containing despiked and on-point beam A indices.
    @returns List (or numpy array if idx_B was flat) containing despiked and on-point beam B indices.
    """

    return da_sub.groupby("scan").map(methods._overshoot_per_scan, args=(buff,))

#### Reduction and averaging
def reduce_observation_still(da_sub, thres=None):
    """
    Average a still observation, without AB chopping.
    Note that any including/excluding to the data array needs to be done BEFORE passing the array to this method.
    
    @param da_sub Data array on which potentially include/exclude operations have been performed.

    @returns Array with the average signal, for each KID.
    @returns Array with the standard deviation, for each KID.
    @returns Array with master indices for each KID.
    @returns Array with filter frequencies, in GHz, for each KID.
    """
    
    master_id = da_sub.chan.values
    freq = da_sub.d2_mkid_frequency.values

    spec_avg = np.nanmean(da_sub.data, axis=0)
    spec_var = np.nanvar(da_sub.data, axis=0)

    if thres is not None:
        mask = (spec_avg < thres)
        spec_avg = spec_avg[mask]
        spec_var = spec_var[mask]
        master_id = master_id[mask]
        freq = freq[mask]

    return spec_avg, spec_var, master_id, freq    

def reduce_observation_full(da_sub, conv_factor=1):
    """
    Average a pswsc observation over the full observation.

    @param da_sub Data array that has been despiked and overshoot-removed.
    @param conv_factor A custom conversion factor, if something other than brightness temperature is desired.

    @returns Array with the average signal, for each KID.
    @returns Array with the standard deviation, for each KID.
    @returns Array with master indices for each KID.
    @returns Array with filter frequencies, in GHz, for each KID.
    """

    master_id = da_sub.chan.values
    freq = da_sub.d2_mkid_frequency.values

    da_sub = da_sub.groupby("scan").map(methods._subtract_per_scan, args=(conv_factor,))

    spec_avg = np.nanmean(da_sub["avg"].data, axis=0)

    N = da_sub["avg"].data.shape[0]
        
    spec_var = np.nansum(da_sub["var"].data, axis=0) / N**2

    return spec_avg, spec_var, master_id, freq

def reduce_observation_nods(da_sub, num_nods=2, conv_factor=1):
    """
    Average ABBA off-source and on-source beams and subtract.
    The averaging is 

    @param da_sub Data array that has been despiked and overshoot-removed.
    @parm num_nods Number of nods to include in each average. Must be an even number.
    @param conv_factor A custom conversion factor, if something other than brightness temperature is desired.

    @returns Array with the average signal, for each KID.
    @returns Array with the standard deviation, for each KID.
    @returns Array with master indices for each KID.
    @returns Array with filter frequencies, in GHz, for each KID.
    """

    if not (num_nods % 2) == 0:
        print(f"num_nods needs to be even, {num_nods} is not even!")
        exit(1)

    master_id = da_sub.chan.values
    freq = da_sub.d2_mkid_frequency.values

    da_sub = da_sub.groupby("scan").map(methods._subtract_per_scan, args=(conv_factor,))

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

    weight = np.nansum(1 / cycle_var, axis=0)
    spec_avg = np.nansum(cycle_avg / cycle_var, axis=0) / weight
    spec_var = 1 / weight
    
    return spec_avg, spec_var, master_id, freq

def reduce_observation(da_sub, resolution="nod", conv_factor=1):
    """
    Average ABBA off-source and on-source beams and subtract.

    @param da_sub Data array that has been despiked and overshoot-removed.
    @param resolution Unit to average over. Default is 'nod'.
        If 'nod' : Apply variance weighted averaging over nod cycles.
        If 'obs' : Apply averaging over full observation

    @returns Array with the average signal, for each KID.
    @returns Array with the standard deviation, for each KID.
    @returns Array with master indices for each KID.
    @returns Array with filter frequencies, in GHz, for each KID.
    """

    master_id = da_sub.chan.values
    freq = da_sub.d2_mkid_frequency.values

    da_sub = da_sub.groupby("scan").map(_subtract_per_scan, args=(conv_factor,))

    if resolution == 'obs':
        spec_avg = np.nanmean(da_sub["avg_first"].data, axis=0)
        spec_var = np.nanvar(da_sub["avg_first"].data, axis=0)
    
    elif resolution == 'nod':
        scan_labels = da_sub["avg_first"].scan.data.astype(int)
        args_sort = np.argsort(scan_labels)

        spec_avg_first = da_sub["avg_first"].data[args_sort]
        spec_var_first = da_sub["var_first"].data[args_sort]
        
        spec_avg_last = da_sub["avg_last"].data[args_sort]
        spec_var_last = da_sub["var_last"].data[args_sort]

        avg_l = []
        var_l = []

        for i in range(len(args_sort)):
            avg_l.append(spec_avg_first[i])
            avg_l.append(spec_avg_last[i])
            
            var_l.append(spec_var_first[i])
            var_l.append(spec_var_last[i])

        running_avg = avg_l[0]
        running_var = var_l[0]

        cycle_avg = []
        cycle_var = []

        n = 0
        nods = 0
        N = 5
        for i in range(1,len(avg_l) - 1):
            running_avg += avg_l[i]
            running_var += var_l[i]

            n += 1
            if n == 4:
                cycle_avg.append(running_avg / N)

                N = 4
                n = 0

        print(len(avg_l))

        assert(len(args_sort) % 2 == 0)

        cycle_store_avg = []
        cycle_store_var = []

        cycle_store_avg.append((spec_avg_first[0]+spec_avg_last[0]) / 2)

        cycle_avg = np.zeros((spec_avg.shape[0] // 4, spec_avg.shape[1]))
        cycle_var = np.zeros((spec_var.shape[0] // 4, spec_var.shape[1]))
        for i in range(len(args_sort) // 4):
            for j in range(4):
                cycle_avg[i] += spec_avg[i*4 + j]
                cycle_var[i] += spec_var[i*4 + j]

        cycle_avg /= 4
        cycle_var /= 4

        weight = np.nansum(1 / cycle_var, axis=0)
        spec_avg = np.nansum(cycle_avg / cycle_var, axis=0) / weight
        spec_var = 1 / weight

        spec_avg = np.nanmean(cycle_avg, axis=0)
        spec_var = np.nanvar(cycle_avg, axis=0)

        print(spec_avg.shape)

    else:
        print(f"Unknown value {resolution} for resolution!")
        exit(1)

    return spec_avg, spec_var, master_id, freq

def stack_spectra(npy_loc, obsids):
    """
    Stack a collection of measured pswsc spectra on top of each other.

    @param npy_loc Directory containing fully averaged analysis npy files.
    @param obsids List of obsids to be included in the stacking.

    @returns Array containing average stacked signal, inverse variance weighted.
    @returns Array containing variances of stacked averages.
    @returns master_ids Master indexes for each entry in average and variance.
    @returns freq Array with channel frequencies corresponding to master_ids
    """

    import matplotlib.pyplot as plt

    obsid_good = []

    # In this loop we just set length of output frequency array
    first = True
    for i, obs in enumerate(obsids):
        try:
            chan = np.load(f"npys/{obs}_chan.npy")
            freq = np.load(f"npys/{obs}_freq.npy")
            obsid_good.append(obs)
        except:
            print(_yellow(f"Could not find obsid {obs}, skipping..."))
            continue
        
        if first:
            tot_chan = chan
            first = False
            continue
        
        tot_chan = np.union1d(tot_chan, chan)

    observation_avg = np.empty((len(obsid_good), tot_chan.size))
    observation_var = np.empty((len(obsid_good), tot_chan.size))
    tot_freq = np.empty((len(obsid_good), tot_chan.size))

    observation_avg[:] = np.nan
    observation_var[:] = np.nan
    tot_freq[:] = np.nan

    for i, obs in enumerate(obsid_good):
        avg = np.load(f"npys/{obs}_avg.npy")
        var = np.load(f"npys/{obs}_var.npy")
        chan = np.load(f"npys/{obs}_chan.npy")
        freq = np.load(f"npys/{obs}_freq.npy")
        
        idxs, idx_in_total, idx_in_current = np.intersect1d(tot_chan, chan, assume_unique=True, return_indices=True)

        assert(idx_in_total.size == avg.size)

        observation_avg[i,idx_in_total] = avg
        observation_var[i,idx_in_total] = var
        tot_freq[i,idx_in_total] = freq
    
    #avg_arr = np.nanmean(observation_avg, axis=0)
    #var_arr = np.nanvar(observation_var, axis=0)
    #freq_arr = np.nanmean(tot_freq, axis=0)
    #return avg_arr, var_arr, tot_chan, freq_arr

    freq_arr = np.nanmean(tot_freq, axis=0)

    weights = np.nansum(1/observation_var, axis=0)

    avg_arr = np.nansum(observation_avg/observation_var, axis=0) / weights
    var_arr = 1 / weights

    return avg_arr, var_arr, tot_chan, freq_arr

def rebin(center_freqs, bw, in_avg, in_var, in_freq):

    avg_binned = []
    var_binned = []
    freq_binned = []

    for cfreq in center_freqs:
        mask = np.absolute(in_freq - cfreq) < bw
        avg_in_bin = in_avg[mask]
        var_in_bin = in_var[mask]

        #weights = np.nansum(1/var_in_bin)
        avg_bin = np.nanmean(avg_in_bin)#np.nansum(avg_in_bin/var_in_bin) / weights
        var_bin = np.nanmean(var_in_bin)#1 / weights
        avg_freq = np.nanmean(in_freq[mask])

        avg_binned.append(avg_bin)
        var_binned.append(var_bin)
        freq_binned.append(avg_freq)

    return np.array(avg_binned), np.array(var_binned), np.array(freq_binned)




