import numpy as np
import pandas as pd

def calculate_g_full(T, l, ls, m, M, R, theta_0):
    inertia_string = (1/12) * M * (ls**2) / (m * l)
    inertia_bob = (1/5) * (R**2) / (l**2)
    angle_corr = (1/8) * theta_0**2 + (1/256) * theta_0**4
    correction = 1 - inertia_string + inertia_bob + angle_corr
    g = (4 * np.pi**2 * l / T**2) * correction
    return g
def calculate_ug(l, ul, T, uT, mt, umt, R, uR, theta, utheta):
    pi2 = np.pi**2

    A = (4 * pi2/ T**2)**2 * ul**2 * (1-(mt/12)-(R**2/(5*l**2))+(theta**2/8)+(theta**4/256))

    B = (8 * pi2 * R / (5 * T**2*l))**2 * umt**2

    C = (8 * pi2 * l / T**3)**2 * uT**2 * (1-(mt/12)+(R**2/(5*l**2))+(theta**2/8)+(theta**4/256))

    angle_sens = theta*(1/4) + (3/64) * theta**3
    D = (4 * pi2 * l / T**2 * angle_sens)**2 * utheta**2

    # Total uncertainty
    ug = np.sqrt(A + B + C + D)
    return ug


dataB = pd.read_csv("C:/Users/jerem/python/WIP/DataB.csv", skiprows=1)
dataC = pd.read_csv("C:/Users/jerem/python/WIP/DataC.csv", skiprows=1)

m = 0.067
um = 0.005    # kg
p = 0.00283
up = 0.00001      # kg
ls = 0.666      # m
uls = 0.0005
ms = ls*p
ums = np.sqrt(up**2+uls**2)
mt = ms/m
umt = np.sqrt((ums/m)**2 + (um*ms/m**2)**2)
R = 0.0254     # m (radius of bob)
uR = 0.0001
l = ls+R        # m (length from pivot to center of mass)
ul = 0.005
meter_pixel = 0.00024
d_bc = 0.307     # m
d_rb = 0.045     # m
parallax_factor = meter_pixel/((d_rb / d_bc)+1)
amp = []

theta = []
utheta = 0.001

num_trials = dataB.shape[1] // 2
g_results = []
ug_results = []

for i in range(num_trials):
        # Get first period (from T1 column)
        t = np.mean(dataC.iloc[:, 2*i])

        uT = 0.01

        # Get first amplitude (from DataB)
        A_raw = dataB.iloc[0, 2*i+1]  # amplitude column
        A_real = A_raw * parallax_factor  # apply geometric parallax correction

        # Compute θ₀ from arc length
        theta_0 = np.arctan(A_real / (l-0.24))

        # Calculate g
        g_val = calculate_g_full(t, l, ls, mt, p, R, theta_0)
        ug_val = calculate_ug(l, ul, t, uT, mt, umt, R, uR, theta_0, utheta)
        g_results.append(g_val)
        ug_results.append(ug_val)

        print(f"Trial {i+1}: T = {t:.4f} s, θ₀ = {np.rad2deg(theta_0):.2f}°, g = {g_val:.6f} m/s²")
        print(ug_val)
        x=l*2*np.pi*theta_0
        lin = (1.6*10**-4)*0.025*R*x
        quad = 0.0025*R**2*x**2
        print(lin/quad)

    # except Exception as e:
    #     print(f"Trial {i+1}: Error occurred - {e}")
    #     g_results.append(np.nan)
