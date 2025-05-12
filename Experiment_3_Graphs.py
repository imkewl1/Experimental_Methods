import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import linregress

# Load data
data1 = pd.read_csv("C:/Users/jerem/python/WIP/Data1.csv", skiprows=1)
data2 = pd.read_csv("C:/Users/jerem/python/WIP/Data2.csv", skiprows=1)

# Select trial number
z = 0  # 15th rial
x = z * 2

# Extract and clean data for data2 (second graph)
time = data2.iloc[:, x]
amplitude = data2.iloc[:, x + 1]

# Combine and drop NaNs
clean_df = pd.DataFrame({'time': time, 'amplitude': amplitude}).dropna()

# Convert to numpy arrays
time = clean_df['time'].to_numpy()
amplitude = clean_df['amplitude'].to_numpy()

# Ensure time has variation
if np.all(time == time[0]):
    print("Error: Time values are constant. Cannot perform linear regression.")
else:
    # Perform linear regression
    slope, intercept, r_value, p_value, std_err = linregress(time, amplitude)
    fit_line = slope * time + intercept

    # Compute mean amplitude
    mean_amplitude = np.mean(amplitude)

    # Create plots
    fig, axs = plt.subplots(2, figsize=(10, 6))
    fig.suptitle('Experiment 3 - Trial 15')

    # Plot Data1
    axs[0].plot(data1.iloc[:, x], data1.iloc[:, x + 1], color='tab:blue')
    axs[0].set_title('Camera 1')
    axs[0].set_xlabel('Time [s]')
    axs[0].set_ylabel('Pixel')
    axs[0].set_xlim([0, 45])

    # Plot Data2 with linear fit
    axs[1].plot(time, amplitude, label='Original Data', color='tab:orange')
    axs[1].plot(time, fit_line, label=f'Fit: y = {slope:.3f}x + {intercept:.2f}', color='black', linestyle='--')
    axs[1].set_title('Camera 2 (Linear Fit)')
    axs[1].set_xlabel('Time [s]')
    axs[1].set_ylabel('Pixel')
    axs[1].set_xlim([0, 45])
    axs[1].legend()
    axs[1].grid(True)

    # Output results
    print(f"Mean amplitude: {mean_amplitude:.3f} pixels")
    print(f"Slope of linear fit: {slope:.6f} pixels/s")

    # Layout
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

    time = data1.iloc[:, x]
    amplitude = data1.iloc[:, x + 1]
    d_amplitude = np.gradient(amplitude, time[1])

    times = time[:-1]

    T = np.diff(times[np.diff(np.sign(d_amplitude)) == 2])
    print(T)
    print(np.mean(T[(T>1) & (T<2)]))
    T2 = np.diff(times[np.diff(np.sign(d_amplitude)) == -2])
    print(T2)
    print(np.mean(T2[(T2>1) & (T2<2)]))

    plt.plot(time, d_amplitude, label="Derivative")
    plt.legend()
    #plt.show()

    

