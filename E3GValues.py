import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
t1 = [1.5730809943865276, 1.5736160975609754, 1.5733525116599127, 1.568322616955007, 1.5734627273656243, 1.5750000000000002, 1.5699340913715913, 1.5546981754658384, 1.5710144927536231, 1.5693587895278316, 1.5682853173609053, 1.5714110011799944, 1.5717056113423185]
t2 = [1.5707625257138873, 1.5736160975609752, 1.5733525116599127, 1.5561548246831347, 1.5675071530758227, 1.56025641025641, 1.571801501035197, 1.534550571710897, 1.5722222222222224, 1.5591510592031426, 1.5648645607961926, 1.5711996926467178, 1.5719580941714497]
t = [(x + y) / 2 for x, y in zip(t1, t2)]
ut = np.std(t)
utheta = 0.00174
ls = 0.585
r = 0.02635
ur = 0.0001
l = ls + r
ul = 0.0005
m = 0.0003
theta = [4.69, 5.18, 3.23, 3.03, 3.62, 2.84, 3.23, 3.03, 3.42, 3.13, 3.33, 2.06, 2.74]
theta = np.sort(theta)
g_value = []
ug_value = []
for i in range(len(theta)):
    theta[i] = np.deg2rad(theta[i])
    g = (4 * np.pi ** 2 * (ls + r) / t[i] ** 2) * (1 - (m / 12) + (r ** 2 / (5 * ls ** 2)) + (theta[i] ** 2 / 8) + (theta[i] ** 4 / 256))
    pi2 = np.pi**2

    A = (4 * pi2/ t[i]**2)**2 * ul**2 * (1-(t[i]/12)-(r**2/(5*ls**2))+(theta[i]**2/8)+(theta[i]**4/256))

    B = (8 * pi2 * r / (5 * t[i]**2*l))**2 * ur**2

    C = (8 * pi2 * l / t[i]**3)**2 * ut**2 * (1-(t[i]/12)+(r**2/(5*ls**2))+(theta[i]**2/8)+(theta[i]**4/256))

    angle_sens = theta[i]*(1/4) + (3/64) * theta[i]**3
    D = (4 * pi2 * l / t[i]**2 * angle_sens)**2 * utheta**2

    # Total uncertainty
    ug = np.sqrt(A + B + C + D)
    g_value.append(g)
    ug_value.append(ug)
for j in range(len(theta)):
    print(np.rad2deg(theta[j]))
    # print(g_value[j])
    # print(ug_value[i])
    # discrep = np.abs(9.80667 - g_value[j])
    # print(discrep)
    # print(discrep/ug_value[i])
    maxv = l*2*np.pi*theta[j]/(t[j])
    #print(maxv)
    lindrag = (1.6 * 10**-4)*2*r*maxv
    quaddrag = 0.25 * (2*r)**2 * maxv**2
    print(lindrag/quaddrag)
print(np.mean(ug_value))
print(np.mean(g_value))

theta = np.rad2deg(theta)
plt.figure(figsize=(8, 6))
plt.scatter(theta, t, label='Data', color='tab:blue')
plt.xlabel('Theta[degrees]')
plt.ylabel('Period[s]')
plt.title('Angle vs Period')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
