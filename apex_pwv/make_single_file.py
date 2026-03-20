import csv
import datetime
import pathlib
import os

import numpy as np
from tqdm import tqdm

from scipy.interpolate import InterpolatedUnivariateSpline

try:
    import cPickle as pickle
except ImportError:
    import pickle

files = pathlib.Path(".").rglob("*")

pwv_l = []
dt_l = []

for file in tqdm(list(files), total=len(list(files))):
    if str(file).endswith(".csv"):
        with open(file, 'r') as csvfile:
            reader = csv.reader(csvfile, delimiter=',', )
            next(reader)
            next(reader)
            for line in reader:
                if line:
                    if line[0][0] != "#": 
                        print(line[0])

                        dt = datetime.datetime.fromisoformat(line[0]).timestamp()
                        dt_l.append(dt)
                        pwv_l.append(float(line[1]))

pwv = np.array(pwv_l)
dt = np.array(dt_l)

dt = dt[pwv > 0.1]
pwv = pwv[pwv > 0.1]

idx_sort = np.argsort(dt)

dt = dt[idx_sort]
pwv = pwv[idx_sort]

dt, idx_uniq = np.unique(dt, return_index=True)
pwv = pwv[idx_uniq]
import matplotlib.pyplot as plt


interpolator = InterpolatedUnivariateSpline(dt, pwv, k=1)

#Pickle, unpickle and then plot again
with open('pwv_interpolator.pkl', 'wb') as f:
    pickle.dump(interpolator, f)
with open('pwv_interpolator.pkl', 'rb') as f:
    interpolator = pickle.load(f)
plt.scatter(dt, pwv)
plt.plot(dt, interpolator(dt))
plt.show()





