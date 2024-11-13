import numpy as np
import csv
import os
import scipy.interpolate as interp
from datetime import datetime, timedelta
import time
epoch = datetime.utcfromtimestamp(0)
ABSPATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def unix_time_millis(dt):
    return int((dt - epoch).total_seconds() * 1000.0)

def getInterpolatorATM():
    """!
    Read atmospheric transmission curves from csv file.

    @returns interpolated function taking as arguments frequencies and PWVs to interpolate over.
    """
    csv_loc = os.path.join(ABSPATH, "atm.csv")

    with open(csv_loc, 'r') as file:
        csvreader = list(csv.reader(file, delimiter=" "))

        data = []

        for row in csvreader:
            if row[0] == "#":
                continue
            elif row[0] == "F":
                pwv_curve = np.array(row[1:], dtype=np.float64)
                continue
            while("" in row):
                row.remove("")
            data.append(np.array(row, dtype=np.float64))

        _arr = np.array(data)
        freqs = _arr[:,0]
        eta_atm = _arr[:,1:]
        
        return interp.RectBivariateSpline(freqs, pwv_curve, eta_atm)

def getInterpolatorPWV(obsid):
    """
    Look in the PWV folder and get interpolator for PWV given an obsid.

    @param obsid String containing an obsid in the format yyyymmddHHMMSS.

    @returns Interpolator that needs to be given an array of Unix times with millisecond resolution, cast to integers.
    """

    # Convert unix timestamps to UTC datetime objects
    # Also check if a day has passed over 
    
    fmt_obsid = '%Y%m%d%H%M%S'
    
    date_obsid = datetime.strptime(obsid, fmt_obsid)

    day = date_obsid.day
    month = date_obsid.month

    day_next = (date_obsid + timedelta(days=1)).day
    month_next = (date_obsid + timedelta(days=1)).month
    
    day_prev = (date_obsid + timedelta(days=-1)).day
    month_prev = (date_obsid + timedelta(days=-1)).month

    file = f"PWV_{day}-{month}.csv" 
    csv_loc = os.path.join(ABSPATH, "APEX_PWV_files", file)
    
    file_next = f"PWV_{day_next}-{month_next}.csv" 
    csv_loc_next = os.path.join(ABSPATH, "APEX_PWV_files", file_next)
    
    file_prev = f"PWV_{day_prev}-{month_prev}.csv" 
    csv_loc_prev = os.path.join(ABSPATH, "APEX_PWV_files", file_prev)
    
    time_l = []
    PWV_l = []
    fmt_csv = "%Y-%m-%dT%H:%M:%S"
    with open(csv_loc_prev, 'r') as fi:
        csvreader = list(csv.reader(fi, delimiter=","))

        for i, row in enumerate(csvreader):
            if i == 0: 
                continue
            elif row[1] == "":
                continue
            _time = datetime.strptime(row[0], fmt_csv)
            time_l.append(_time)
            PWV_l.append(float(row[1]))
    
    with open(csv_loc, 'r') as fi:
        csvreader = list(csv.reader(fi, delimiter=","))

        for i, row in enumerate(csvreader):
            if i == 0 or i == 1: # Exclude first line, is already included in previous file
                continue
            elif row[1] == "":
                continue
            _time = datetime.strptime(row[0], fmt_csv)
            time_l.append(_time)
            PWV_l.append(float(row[1]))
    
    with open(csv_loc_next, 'r') as fi:
        csvreader = list(csv.reader(fi, delimiter=","))

        for i, row in enumerate(csvreader):
            if i == 0 or i == 1: # Exclude first line, is already included in previous file
                continue
            elif row[1] == "":
                continue
            _time = datetime.strptime(row[0], fmt_csv)
            time_l.append(_time)
            PWV_l.append(float(row[1]))

    obsid_milli = unix_time_millis(date_obsid)

    return date_obsid, interp.interp1d(np.array(time_l).astype('datetime64[ms]').astype('int'), np.array(PWV_l))(obsid_milli)

