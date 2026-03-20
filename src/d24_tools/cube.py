import numpy as np
import os
import warnings
from functools import partial
from multiprocessing import Pool

from d24_tools import parallel
import matplotlib.pyplot as plt

NCPU = os.cpu_count()
NCPU_USE = NCPU - 2 if NCPU <= 10 else 10

def make_cube(tod, std, ra, dec, chan, freq, nbins, map_method="uniform"):
    if map_method == "uniform":
        func_par = partial(parallel.make_cube_pool, 
                           xargs=(ra, dec, nbins)
                           )
    else:
        func_par = partial(parallel.make_cube_pool_inv_var, 
                           xargs=(ra, dec, nbins)
                           )

    tod_spl = np.array_split(tod, NCPU_USE, axis=1)
    std_spl = np.array_split(std, NCPU_USE, axis=1)
    freq_spl = np.array_split(freq, NCPU_USE, axis=0)
    chan_spl = np.array_split(chan, NCPU_USE, axis=0)
    thread_idx = np.arange(NCPU_USE)

    par_args = [(ts, ss, cs, fs, tid) for ts, ss, cs, fs, tid in zip(tod_spl, std_spl, chan_spl, freq_spl, thread_idx)]
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        with Pool(NCPU_USE) as p:
            out = p.map(func_par, par_args)
    avg_cube = np.concatenate([x[0] for x in out], axis=-1)
    std_cube = np.concatenate([x[1] for x in out], axis=-1)
    ra_centers = out[0][2]
    dec_centers = out[0][3]

    return avg_cube, std_cube, ra_centers, dec_centers
