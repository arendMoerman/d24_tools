import numpy as np
import csv
import pathlib
from scipy.interpolate import RectBivariateSpline
from scipy.optimize import curve_fit

eta_atm_path = pathlib.Path(__file__).parent.parent.parent.resolve() / "2024" / "eta_atm"

pwv_interpolator_path = pathlib.Path(__file__).parent.parent.parent.resolve() / "2024" / "pwv_interpolator.pkl"
try:
    import cPickle as pickle
except ImportError:
    import pickle

def get_pwv(dt):
    with open(str(pwv_interpolator_path), 'rb') as f:
        interpolator = pickle.load(f)
        return interpolator(dt)

def get_eta_atm():
    with open(str(eta_atm_path), 'r') as file:
        reader = csv.reader(file, delimiter=' ')
        pwv = []
        f_atm = []
        eta = []
        for i, row in enumerate(reader):
            if i == 0:
                pwv = np.array([float(x) for x in row])
                continue

            f_atm.append(float(row[0]))
            eta.append([float(x) for x in row[1:]])

        f_atm = np.array(f_atm)
        eta = np.array(eta)

        eta_atm_interp = RectBivariateSpline(f_atm, 
                                             pwv,
                                             eta, 
                                             kx=1, ky=1)

        return eta_atm_interp

