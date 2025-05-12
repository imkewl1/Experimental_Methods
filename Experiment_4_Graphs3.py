import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.fft import rfft, rfftfreq
from scipy.stats import linregress

# ==== Load all CSVs ====
dataA = pd.read_csv("C:/Users/jerem/python/WIP/DataA.csv", skiprows=1)
dataB = pd.read_csv("C:/Users/jerem/python/WIP/DataB.csv", skiprows=1)
dataC = pd.read_csv("C:/Users/jerem/python/WIP/DataC.csv", skiprows=1)
dataD = pd.read_csv("C:/Users/jerem/python/WIP/DataD.csv", skiprows=1)

# ==== Physical constants ====
mass = 0.5406  # kg
g = 9.81       # m/s²
length = 0.90  # m (string length)
fps = 60       # assumed FPS
meter_pixel = 0

# ==== Experimental parallax setup ====
d_bc = 0.65  # distance from bob to camera [meters] (e.g., camera at 65 cm)
d_rb = 0.50  # distance from ruler to bob (at equilibrium) [meters]
parallax_factor = meter_pixel/((d_rb / d_bc)+1)  # your derived correction ratio

# ==== Loop over each trial ====
num_trials = dataA.shape[1] // 2

for i in range(num_trials):
    print(f"\n📊 Trial {i + 1} Analysis")

    # === Load trial data ===
    time = dataA.iloc[:, 2*i].dropna().to_numpy()
    disp_raw = dataA.iloc[:, 2*i+1].dropna().to_numpy()
    vel_raw = dataD.iloc[:, 2*i+1].dropna().to_numpy()
    amp_time = dataB.iloc[:, 2*i].dropna().to_numpy()
    amp_raw = dataB.iloc[:, 2*i+1].dropna().to_numpy()
    period1 = dataC.iloc[:, 2*i].dropna().to_numpy()
    period2 = dataC.iloc[:, 2*i+1].dropna().to_numpy()

    if len(time) == 0 or len(disp_raw) == 0:
        continue

    # === Apply parallax correction based on triangle geometry ===
    disp = disp_raw * parallax_factor
    amp_val = amp_raw * parallax_factor
    velocity = vel_raw * parallax_factor

    # === Energy calculations ===
    KE = 0.5 * mass * velocity**2
    PE = mass * g * (length - np.sqrt(length**2 - disp**2))
    E_total = KE + PE

    # === Log amplitude regression ===
    ln_amp = np.log(amp_val)
    slope, intercept, *_ = linregress(amp_time, ln_amp)

    # === FFT ===
    N = len(disp)
    yf = rfft(disp)
    xf = rfftfreq(N, 1/fps)

    # === Plot all ===
    fig, axs = plt.subplots(4, 2, figsize=(14, 16))
    fig.suptitle(f'📐 Full Pendulum Analysis (Geometric Parallax Corrected) - Trial {i+1}', fontsize=18)

    # 1. Amplitude vs. Time
    axs[0, 0].plot(time, disp)
    axs[0, 0].set_title("Amplitude vs. Time")
    axs[0, 0].set_xlabel("Time [s]")
    axs[0, 0].set_ylabel("Displacement [m]")

    # 2. ln(Amplitude) vs. Time
    axs[0, 1].plot(amp_time, ln_amp, 'o-')
    axs[0, 1].set_title(f"ln(Amplitude) vs. Time (slope = {slope:.3f})")
    axs[0, 1].set_xlabel("Time [s]")
    axs[0, 1].set_ylabel("ln(Amplitude)")

    # 3. Period vs. Amplitude
    axs[1, 0].plot(amp_val[:len(period1)], period1[:len(amp_val)], 'o')
    axs[1, 0].plot(amp_val[:len(period2)], period2[:len(amp_val)], 'o')
    axs[1, 0].set_title("Period vs. Amplitude")
    axs[1, 0].set_xlabel("Amplitude [m]")
    axs[1, 0].set_ylabel("Period [s]")

    # 4. Velocity vs. Time
    axs[1, 1].plot(time[:len(velocity)], velocity)
    axs[1, 1].set_title("Velocity vs. Time")
    axs[1, 1].set_xlabel("Time [s]")
    axs[1, 1].set_ylabel("Velocity [m/s]")

    # 5. Phase Space (x vs v)
    axs[2, 0].plot(disp[:len(velocity)], velocity)
    axs[2, 0].set_title("Phase Space: Displacement vs. Velocity")
    axs[2, 0].set_xlabel("Displacement [m]")
    axs[2, 0].set_ylabel("Velocity [m/s]")

    # 6. Energy vs. Time
    axs[2, 1].plot(time[:len(E_total)], E_total)
    axs[2, 1].set_title("Total Energy vs. Time")
    axs[2, 1].set_xlabel("Time [s]")
    axs[2, 1].set_ylabel("Energy [J]")

    # 7. Drag Force vs. Velocity
    drag_force = np.gradient(amp_val, amp_time)
    axs[3, 0].scatter(amp_val[:len(drag_force)], drag_force, alpha=0.7)
    axs[3, 0].set_title("Drag vs. Velocity")
    axs[3, 0].set_xlabel("Amplitude [m]")
    axs[3, 0].set_ylabel("dA/dt ~ Drag [m/s]")

    # 8. FFT of Displacement
    axs[3, 1].plot(xf, np.abs(yf))
    axs[3, 1].set_xlim(0, 5)
    axs[3, 1].set_title("FFT: Frequency Spectrum")
    axs[3, 1].set_xlabel("Frequency [Hz]")
    axs[3, 1].set_ylabel("Amplitude")

    plt.tight_layout()
    plt.subplots_adjust(top=0.93)
    plt.savefig(f"C:/Users/jerem/python/WIP/Full_Plot_Trial_{i+1}_geom_corrected.pdf")
    plt.show()
