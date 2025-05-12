import numpy as np
import matplotlib.pyplot as plt

period = [1.6467, 1.6565, 1.6567, 1.6613, 1.6647, 1.6641, 1.6676, 1.6719, 1.6852]
angle = [3.17, 7.99, 10.23, 13.88, 14.06, 18.06, 22.83, 24.75, 31.04]
period_theory = []
L = 0.69
g = 9.81 
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

for i in range(len(period)):
    theta_rad = np.radians(angle[i])
    correction = 1 + (mt/12)+(R**2/(5*ls**2))+(theta_rad**2) / 16
    T_theory = 2 * np.pi * np.sqrt(L / g) * correction
    period_theory.append(T_theory)

plt.plot(angle, period, label="Experimental")
plt.plot(angle, period_theory, label="Theoretical", linestyle='--')
plt.xlabel("Initial Angle (degrees)")
plt.ylabel("Period (s)")
plt.legend()
plt.grid(True)
plt.show()
