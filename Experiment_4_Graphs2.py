import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataA = pd.read_csv("C:/Users/jerem/python/WIP/DataA.csv", skiprows=1)
dataB = pd.read_csv("C:/Users/jerem/python/WIP/DataB.csv", skiprows=1)

num_trials = dataA.shape[1] // 2  

fig, axes = plt.subplots(nrows=num_trials, ncols=2, figsize=(12, 4 * num_trials))
fig.suptitle("Displacement and Peak Amplitudes per Trial", fontsize=16, y=1.02)

for i in range(num_trials):
    col_time_A = 2 * i
    col_disp_A = 2 * i + 1

    col_time_B = 2 * i
    col_peak_B = 2 * i + 1

    time_A = dataA.iloc[:, col_time_A].dropna()
    disp_A = dataA.iloc[:, col_disp_A].dropna()
    
    time_B = dataB.iloc[:, col_time_B].dropna()
    peak_B = dataB.iloc[:, col_peak_B].dropna()

    axes[i, 0].plot(time_A, disp_A, color='tab:blue')
    axes[i, 0].set_title(f"Trial {i+1} - Raw Motion")
    axes[i, 0].set_xlabel("Time [s]")
    axes[i, 0].set_ylabel("Displacement [px or m]")
    axes[i, 0].grid(True)

    axes[i, 1].plot(time_B, peak_B, 'o-', color='tab:red')
    axes[i, 1].set_title(f"Trial {i+1} - Peak Amplitudes")
    axes[i, 1].set_xlabel("Time [s]")
    axes[i, 1].set_ylabel("Amplitude [m]")
    axes[i, 1].grid(True)

plt.tight_layout()
plt.subplots_adjust(top=0.95)
plt.show()
