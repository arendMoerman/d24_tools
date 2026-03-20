import numpy as np
from tqdm import tqdm
import warnings
from functools import partial
from multiprocessing import Pool
import os

from d24_tools import parallel

NCPU = os.cpu_count()
NCPU_USE = NCPU - 2 if NCPU <= 10 else 10
NFIT = 15

def fit_gauss(avg_cube, std_cube, ra, dec, double_gauss=False):                                                                   
    func_par = partial(parallel.fit_gauss_pool, 
                       xargs=(ra, dec, double_gauss)
                       )

    avg_spl = np.array_split(avg_cube, NCPU_USE, axis=-1)
    std_spl = np.array_split(std_cube, NCPU_USE, axis=-1)
    thread_idx = np.arange(NCPU_USE)

    par_args = [(avgs, stds, tid) for avgs, stds, tid in zip(avg_spl, std_spl, thread_idx)]
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        with Pool(NCPU_USE) as p:
            out = p.map(func_par, par_args)
    ptot = np.concatenate([x[0] for x in out], axis=-1)
    perr = np.concatenate([x[1] for x in out], axis=-1)
    p2tot = np.concatenate([x[2] for x in out], axis=-1)
    p2err = np.concatenate([x[3] for x in out], axis=-1)
    model_array = np.concatenate([x[4] for x in out], axis=-1)

    if double_gauss:
        return ptot, perr, p2tot, p2err, model_array                                                                                  
    else:
        return ptot, perr, model_array

def pwv_fit_func(param, eta_fdf_mid, f_toptica_mid, csc_el, atm_interpol, gamma, T_atm, data):
    stride = csc_el.size // NFIT

    eta_atm = np.squeeze(atm_interpol(f_toptica_mid, param))[None,:]**csc_el[::stride,None]
    eta_atm_smooth = np.nansum(eta_fdf_mid[None,:,:] * eta_atm[:,:,None], axis=1) / gamma[None,:]

    T_los = (1 - eta_atm_smooth) * T_atm 

    return np.nansum((T_los - data[::stride])**2)
