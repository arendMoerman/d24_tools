from multiprocessing import Pool
from astropy.coordinates import AltAz, ICRS, SkyCoord, EarthLocation
from astropy.time import Time
import astropy.units as u
from datetime import datetime
from functools import partial
from multiprocessing import Pool
import os
import numpy as np
import copy
import pathlib
from astropy.io import fits
from d24_tools.atm import get_pwv, get_eta_atm
from d24_tools.parallel import Tb_mini_skydip_pool
from d24_tools.fitting import pwv_fit_func
from scipy.stats import linregress
import decode as dc
from scipy.optimize import minimize

from d24_tools import colors

import warnings
from tqdm import tqdm

FSAMP = 2e9 / 2**19 / 24
NCPU = os.cpu_count()
NCPU_USE = NCPU - 2 if NCPU <= 10 else 10

ddb_path = pathlib.Path(__file__).parent.parent.parent.resolve() / "ddb" / "ddb_20250814.fits.gz"

def split(lst, n):
    """Split a list into exactly n sub-lists, as evenly as possible."""
    ix = np.linspace(0, len(lst), n + 1, dtype=int)
    return [lst[i:j] for i, j in zip(ix[:-1], ix[1:])]
def parallel_job(job, da, args, num_threads=NCPU_USE):
    """
    Execute a job in parallel. 
    Parallelisation is always carried out along the 'chan' coordinate of the DEMS file.

    @param job Function handle of job to be performed.
        First argument of job should be a DEMS file.
        second argument should be named args, and expect a list of arguments.
    @param da DEMS file on which to perform job.
    @param args List of extra arguments passed to job.
    
    @returns Concatenated outputs of all jobs
    """

    chan_full = da.chan.values
    
    chan_ids = np.arange(chan_full.size)

    da_spl = np.array_split(da.data, num_threads, axis=1)
    chan_id_spl = np.array_split(chan_ids, num_threads)
    ids = np.arange(num_threads)

    par_args = [(_chan_id_spl, _ids) for _chan_id_spl, _ids in zip(chan_id_spl, ids)]

    args_copy = copy.deepcopy(args)
    args_copy.insert(0, da)

    job_part = partial(job, args=args_copy)

    with Pool(num_threads) as p:
        out = p.map(job_part, par_args)

    return out

def azel_to_radec(args, observer_lat=-22.9714, observer_lon=-67.7028, observer_height_m=4860):
    """
    Converts Azimuth-Elevation to Right Ascension-Declination.
    Observer latitude, longitude, and altidue are defaulted to ASTE.
    
    Parameters:
        az_deg (float): Azimuth in degrees (0° = North, 90° = East)
        el_deg (float): Elevation in degrees (0° = horizon, 90° = zenith)
        obs_time (str or None): Observation time (UTC), e.g., "2025-07-28 12:00:00". Uses current UTC if None.
        observer_lat (float): Latitude of observer in degrees
        observer_lon (float): Longitude of observer in degrees
        observer_height_m (float): Height above sea level in meters
    
    Returns:
        SkyCoord object containing altaz coordinate as ICRS coordinate
    """

    az_deg, el_deg, obs_time = args

    if obs_time is None:
        obs_time = Time(datetime.utcnow())
    else:
        obs_time = Time(obs_time)

    # Observer's location
    location = EarthLocation(lat=observer_lat*u.deg, lon=observer_lon*u.deg, height=observer_height_m*u.m)

    # Azimuth-Elevation to AltAz object
    altaz = SkyCoord(
            AltAz(
                az=az_deg*u.deg, alt=el_deg*u.deg, location=location, obstime=obs_time
                )
            )

    icrs = altaz.transform_to('icrs')

    lst = obs_time.sidereal_time("mean", longitude=location.lon)
    ha = (lst - icrs.ra).radian
    dec = icrs.dec.radian
    lat = location.lat.radian
    
    q = np.arctan2(np.sin(ha), (np.tan(lat) * np.cos(dec) - np.sin(dec) * np.cos(ha)))

    return icrs.ra.deg, icrs.dec.deg, np.degrees(q)

def dt_to_seconds(da):
    """
    Return time of data-array da in seconds, to nanosecond precision.
    """
    return da.time.values.astype(
            "datetime64[ns]"
            ).astype(
                    float
                    )*1e-9

def dt_to_unix(da):
    """
    Return time of data-array da in seconds, to nanosecond precision.
    """
    return np.array([x.astype("datetime64[s]").astype(int) for x in da.time.values])

def calc_estimates(avg_cube, ra, dec):
    """Calculate estimates for Gaussian fitting to sky map.
    
    Args:
        az_red: 1D grid of Azimuth co-ordinates in arcseconds, where signal is not NaN.
        dec_red: 1D grid of Elevation co-ordinates in arcseconds, where signal is not NaN.
        signal_red: 1D grid of sky temperatures in Kdecvin, with NaN values removed.
        freq: Frequency at which map is taken in Hertz.

    Returns:
        Dictionary containing initial estimates: 
            x0 = Major axis (std).
            y0 = Minor axis (std).
            xm = x-center.
            ym = y-center.
            psi = position angle w.r.t. positive x-axis.
            amp = Peak amplitude of map.
            floor = Floor levdec in map.

    """
    # Calculate estimate of floor levdec
    
    M00 = get_raw_moment(0, 0, ra, dec, avg_cube)
    M10 = get_raw_moment(1, 0, ra, dec, avg_cube)
    M01 = get_raw_moment(0, 1, ra, dec, avg_cube)
    xm = M10 / M00
    ym = M01 / M00

    ra_cen = ra - xm
    dec_cen = dec - ym

    mu00 = get_raw_moment(0, 0, ra_cen, dec_cen, avg_cube)
    mu11 = get_raw_moment(1, 1, ra_cen, dec_cen, avg_cube)
    mu20 = get_raw_moment(2, 0, ra_cen, dec_cen, avg_cube)
    mu02 = get_raw_moment(0, 2, ra_cen, dec_cen, avg_cube)

    phi = 0.5 * np.arctan2(2*mu11, mu20-mu02)

    if phi > np.pi/2:
        phi -= np.pi/2

    elif phi < -np.pi/2:
        phi += np.pi/2
   
    l1 = ((mu20 + mu02) + np.sqrt(4*mu11**2 + (mu20-mu02)**2)) / 2
    l2 = ((mu20 + mu02) - np.sqrt(4*mu11**2 + (mu20-mu02)**2)) / 2
    
    if l2 < 0:
        l2 = 0.8*l1

    x0 = np.sqrt(l1)
    y0 = np.sqrt(l2)

    if y0 > x0:
        _y0 = x0
        x0 = y0
        y0 = x0

    amp = np.nanmax(avg_cube, axis=(0,1))

    return [amp, xm, ym, x0, y0, phi]

def get_raw_moment(n, m, x, y, g):
    """Get raw image moments from map.

    Args:
        n Order of x-moment.
        m Order of y-moment.
        g Image defined over x and y.

    Returns:
        Image moment of order nm.
    """
    return np.nansum(x**n * y**m * g, axis=(0,1))    

def avg_inv_var(avg, std):
    weight = np.nansum(1 / std.T**2, axis=0)
    avg = np.nansum(avg.T / std.T**2, axis=0) / weight
    std = np.sqrt(1/weight)
    return avg.T, std.T

def consecutive(data, stepsize=1):
    """
    Take numpy array and return list with arrays containing consecutive blocks
    
    @param data Array in which consecutive chunks are to be located.

    @returns List with arrays of consecutive chunks of data as elements.
    """

    return np.split(data, np.where(np.diff(data) != stepsize)[0]+1)

def consecutive_gt(data, stepsize=4/FSAMP):
    """
    Take numpy array and return list with arrays containing consecutive blocks
    
    @param data Array in which consecutive chunks are to be located.

    @returns List with arrays of consecutive chunks of indices as elements.
    """
    dat_idx = np.arange(data.size)

    return np.split(dat_idx, np.where(np.diff(data) > stepsize)[0]+1)

def get_mini_skydip(da):
    state = da.state
    indices = np.arange(da.shape[0])
    state_GRAD = indices[state == "GRAD"]

    chunks = consecutive(state_GRAD)
    return da[chunks[1]]

def get_avg_counterpoints(points_l, times_l):
    distance = 0.1007
    margin = 0.001

    # In case one array bigger than other, select smallest one and start from there
    # In case they are equally sized, we start with first array in list
    if times_l[0].size <= times_l[1].size:
        ii_first = 0
        ii_second = 1
    else:
        ii_first = 1
        ii_second = 0

    points_avg = np.zeros(points_l[0].shape[-1])
    times_avg = 0
    N_in_avg = 0
    for ii in range(times_l[ii_first].size):
        time_loc_ii = times_l[ii_first][ii]
        for jj in range(times_l[ii_second].size):
            time_loc_jj = times_l[ii_second][jj]
            if np.absolute(np.absolute(time_loc_ii - time_loc_jj) - distance) <= margin:
                # Found a counterpoint!
                points_avg += (points_l[ii_first][ii] + points_l[ii_second][jj]) / 2
                times_avg += (time_loc_ii + time_loc_jj) / 2
                N_in_avg += 1
                break
    if not N_in_avg:
        return np.full(shape=points_l[0].shape[-1], fill_value=np.nan), np.nan
    else:
        return points_avg / N_in_avg, times_avg / N_in_avg

def load_filters():
    hdul = fits.open(str(ddb_path))
    freq_toptica = hdul["KIDFILT"].data[0][6]

    # eta_f: Transmission curve of filtershape
    # chan_toptica: channels in toptica sweep
    # f0_scan: center frequencies of filter

    eta_f = []
    chan_toptica = []
    f0_scan = []
    
    eta_f_test = []

    master_id_test = []

    import matplotlib.pyplot as plt
    for hrow in hdul["KIDRESP"].data:
        ch_scan = hrow[1]
        if ch_scan < 0:
            continue
        master_id_test.append(hrow[1])
        f = hrow[8]
        gnuf_other = hrow[9] # Transmission         
        gnuf = hrow[10] # Transmission         

        eta_f_test.append(hrow[9])

    master_id = []

    for hrow in hdul["KIDFILT"].data:
        ch_scan = hrow[1]
        if ch_scan < 0:
            continue
        master_id.append(hrow[1])
        gnuf = hrow[7] # Transmission         
        f0 = hrow[3][0]   # Central frequency
        
        eta_f.append(gnuf)
        chan_toptica.append(ch_scan)
        f0_scan.append(f0)

    eta_f_test = np.array(eta_f_test)
    eta_f = np.array(eta_f)
    avg_wb = (eta_f[-3,:] + eta_f[-4,:]) / 2

    eta_f /= avg_wb
    
    eta_f = np.array(eta_f).T
    eta_f /= np.nanmax(eta_f, axis=0)
    chan_toptica = np.array(chan_toptica)
    f0_scan = np.array(f0_scan)

    return eta_f, chan_toptica, f0_scan, freq_toptica

def eta_atm_avg_filter(freq, pwv, f_toptica_mid, eta_fdf, gamma, csc_el, atm_interp):
    f_toptica_mid, eta_fdf, gamma, T_amb, csc_el, atm_interp = args
    
    return 


def Tb_mini_skydip(da_dfof, squared_eta_f):
    eta_f, chan_toptica, f0_scan, freq_toptica = load_filters()
    eta_f = eta_f**2

    # Select on skydip                                                                                            
    skd_dfof = get_mini_skydip(da_dfof)                                                                  
                                                                                                                  
    Tatm = np.nanmean(skd_dfof.temperature.values)                                                                
    freq = skd_dfof.frequency.to_numpy()                                                                          
                                                                                                                  
    chan = skd_dfof.chan.to_numpy()                                                                                    
                                                                                                                  
    idxs, idx_in_toptica, idx_in_obs = np.intersect1d(chan_toptica, chan, assume_unique=True, return_indices=True)
                                                                                                                  
    chan_toptica = chan_toptica[idx_in_toptica]                                                                   
    eta_f = eta_f[:,idx_in_toptica] # PASS TO fit_func                                                            
    f0_scan = f0_scan[idx_in_toptica]                                                                             
                                                                                                                  
    # Just to be sure, also select intersect in obs                                                               
    chan = chan[idx_in_obs]                                                                                       
    freq = freq[idx_in_obs]                                                                                       
    skd_dfof = skd_dfof[:,idx_in_obs]                                                                             
                                                                                                                  
    # gamma is integral over filterbank                                                                           
    gamma = np.nansum((eta_f[:-1] + eta_f[1:]) / 2 * np.diff(freq_toptica)[:,None], axis=0)                       
    
    eta_fdf = (eta_f[:-1] + eta_f[1:]) / 2 * np.diff(freq_toptica)[:,None]                 
                                                                                           
    el = skd_dfof.lat.to_numpy() / 3600                                                    
                                                                                           
    csc_el = 1 / np.sin(el*np.pi/180) # PASS TO fit_func                                   
                                                                                           
    f_toptica_mid = (freq_toptica[1:] + freq_toptica[:-1]) / 2                             
                                                                                           
    pwv0 = np.nanmean(get_pwv(dt_to_seconds(skd_dfof)))                         
    atm_interpol = get_eta_atm()
    
    # Fit PWV to mini skydip using full skydip parameters
    da_tb_skd = dc.convert.to_brightness(skd_dfof)
    args = (eta_fdf, 
            f_toptica_mid, 
            csc_el, 
            atm_interpol, 
            gamma, 
            Tatm, 
            da_tb_skd.to_numpy())    

    res = minimize(pwv_fit_func, pwv0, args=args, bounds=((0.1, 5.4),))
    pwv_aste = np.squeeze(res.x)

    print(colors._green(f"mini-skydip : ALMA PWV = {pwv0:.2f} mm, ASTE PWV = {pwv_aste:.2f} mm"))

    eta_atm = atm_interpol(f_toptica_mid, pwv_aste).squeeze()[None,:]**(csc_el[:,None])

    # Get average eta during the skydip
    eta_atm_avg = np.nanmean(np.nansum(eta_fdf[None,:,:] * eta_atm[:,:,None], axis=0), axis=0) / gamma

    # From here can and should make parallel    
    data_spl = np.array_split(skd_dfof.to_numpy(), NCPU_USE, axis=1)
    gamma_spl = np.array_split(gamma, NCPU_USE, axis=0)
    eta_fdf_spl = np.array_split(eta_fdf, NCPU_USE, axis=1)
    f_spl = np.array_split(freq, NCPU_USE, axis=0)
    ids = np.arange(NCPU_USE)
    
    par_args = [(_data_spl, _gamma, _eta_fdf, _f_spl, _ids) for _data_spl, _gamma, _eta_fdf, _f_spl, _ids in zip(data_spl, gamma_spl, eta_fdf_spl, f_spl, ids)]

    func_par = partial(
            Tb_mini_skydip_pool,
                       xargs=(eta_atm, Tatm))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        with Pool(NCPU_USE) as p:
            out = p.map(func_par, par_args)

    a = np.concatenate([x[0] for x in out], axis=-1)
    b = np.concatenate([x[1] for x in out], axis=-1)
    R2 = np.concatenate([x[2] for x in out], axis=-1)

    return a, b, idx_in_obs, R2, eta_atm_avg


# Example usage:
if __name__ == "__main__":
    az = 120.0     # degrees
    el = 45.0      # degrees
    lat = 37.7749  # San Francisco latitude
    lon = -122.4194 # San Francisco longitude
    height = 10     # meters
    time_str = "2025-07-28 12:00:00"

    ra, dec = azel_to_radec(az, el, lat, lon, height, time_str)
    print(f"RA: {ra:.6f}°, Dec: {dec:.6f}°")
