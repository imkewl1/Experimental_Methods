import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import linregress
from scipy.optimize import curve_fit

# Load data
data2 = pd.read_csv("C:/Users/jerem/python/WIP/Data2.csv", skiprows=1)

# Setup
num_trials = int(data2.shape[1] / 2)  # assuming each trial uses 2 columns: time + pixel
mean_amplitudes = []
slopes = []

# Loop through all trials
for z in range(num_trials):
    if z==10:
        continue
    x = z * 2  # time column
    time = data2.iloc[:, x]
    amplitude = data2.iloc[:, x + 1]

    # Drop NaNs
    df_clean = pd.DataFrame({'time': time, 'amplitude': amplitude}).dropna()
    time = df_clean['time'].to_numpy()
    amplitude = df_clean['amplitude'].to_numpy()

    if len(time) == 0 or np.all(time == time[0]):
        print(f"Skipping trial {z + 1}: Insufficient or constant time values.")
        continue

    # Linear regression
    slope, intercept, *_ = linregress(time, amplitude)

    # Save results
    mean_amplitudes.append(np.mean(amplitude))
    slopes.append(slope)

# Convert to numpy arrays
mean_amplitudes = np.array(mean_amplitudes)
slopes = np.array(slopes)

t1 = [1.5730809943865276, 1.5736160975609754, 1.5733525116599127, 1.568322616955007, 1.5734627273656243, 1.5750000000000002, 1.5699340913715913, 1.5546981754658384, 1.5710144927536231, 1.5693587895278316, 1.5682853173609053, 1.5714110011799944, 1.5717056113423185]
t2 = [1.5707625257138873, 1.5736160975609752, 1.5733525116599127, 1.5561548246831347, 1.5675071530758227, 1.56025641025641, 1.571801501035197, 1.534550571710897, 1.5722222222222224, 1.5591510592031426, 1.5648645607961926, 1.5711996926467178, 1.5719580941714497]
t = [(x + y) / 2 for x, y in zip(t1, t2)]
theta = [4.69, 5.18, 3.23, 3.03, 3.62, 2.84, 3.23, 3.03, 3.42, 3.13, 3.33, 2.06, 2.74]
A = np.deg2rad(theta)
rfric = 0.0000078
rquad = 0.000175

