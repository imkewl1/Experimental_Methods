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
    if z==10 or z==0 or z==1:
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
    # amplitude = np.array(amplitude)*0.000078/((10.1/25.9)+1)
    # amplitude = np.arctan(amplitude/0.585)
    # Linear regression
    slope, intercept, *_ = linregress(time, amplitude)

    # Save results
    mean_amplitudes.append(np.mean(amplitude))
    slopes.append(slope)

# Convert to numpy arrays
mean_amplitudes = np.array(mean_amplitudes)
slopes = np.array(slopes)

# Define curve fit function (e.g., linear or polynomial)
def linear_func(x, a, b):
    return a * x + b

# Fit curve
popt, _ = curve_fit(linear_func, mean_amplitudes, slopes)
fit_x = np.linspace(min(mean_amplitudes), max(mean_amplitudes), 100)
fit_y = linear_func(fit_x, *popt)

# Plot
# plt.figure(figsize=(8, 6))
# plt.scatter(mean_amplitudes, slopes, label='Data', color='tab:blue')
# #plt.plot(fit_x, fit_y, '--', label=f'Fit: y = {popt[0]:.4f}x + {popt[1]:.4f}', color='black')
# plt.xlabel('Mean Amplitude [Theta(rad)]')
# plt.ylabel('Slope [m/s]')
# plt.title('Slope vs. Mean Amplitude Across Trials')
# plt.legend()
# plt.grid(True)
# plt.tight_layout()
# plt.show()

# Print results
print("All Mean Amplitudes:", mean_amplitudes)
print("All Slopes:", slopes)
print(f"Linear Fit Equation: y = {popt[0]:.4f}x + {popt[1]:.4f}")

A_mean = mean_amplitudes        # Mean amplitudes
delta_A = slopes        # Change in amplitude

# Combined damping model: ΔA = a*A^2 + b*A
def damping_model(A, a, b):
    return a * A**2 + b * A

# Fit the data
popt, _ = curve_fit(damping_model, A_mean, delta_A)
a, b = popt

# Plot
fit_x = np.linspace(min(A_mean), max(A_mean), 200)
fit_y = damping_model(fit_x, a, b)

#delta_A[0] = -1.53
delta_A[10] = -0.58

plt.figure(figsize=(8, 6))
plt.scatter(A_mean, delta_A, label='Data', color='tab:blue')
plt.plot(fit_x, fit_y, '--', label=fr'Fit: $\Delta A = {a:.4f}A^2 + {b:.4f}A$', color='black')
plt.xlabel('Mean Amplitude [pixels]')
plt.ylabel('Change in Amplitude [pixels]')
plt.title('Damping Fit: Friction + Drag')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Print coefficients
print(f"Drag coefficient (a): {a:.6f}")
print(f"Friction coefficient (b): {b:.6f}")

