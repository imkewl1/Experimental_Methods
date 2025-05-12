import numpy as np

# Function to compute period from a row of times
def trial(time_row):
    valid_times = np.array([t for t in time_row if t > 0])  # Ignore missing values (zeros)
    if len(valid_times) == 0:
        return None  # Handle empty cases safely
    return np.mean(valid_times) / 10  # Each time represents 10 oscillations

# Function to compute gravity for each trial
def gravity(l, mt, ul, umt):
    g = ((4 * (np.pi ** 2) * l) / (mt ** 2))
    ug = g * np.sqrt((ul / l) ** 2 + (2 * umt / mt) ** 2)
    print((ul / l) ** 2 / (2 * umt / mt) ** 2)
    return g, ug

def string(l, mt, ms, mb, ul, umt, ums, umb):
    gis = ((4 * (np.pi) ** 2 * l) / (mt ** 2)) * (1 - (ms / (12 * mb))) ** 2  # Gravity with inertia of string
    ugis = gis * np.sqrt((ul / l) ** 2 + (2 * umt / mt) ** 2 + (ums / (6 * mb - 0.5 * ms)) ** 2 + ((umb * ms * np.log(mb) / 6) / (1 - (ms / (12 * mb)))) ** 2) 
    return gis, ugis

def bob(ls, mt, lb, uls, umt, ulb):
    gib = ((4 * (np.pi) ** 2 * (ls+lb)) / (mt ** 2)) * (1 + (lb ** 2 / (5 * (ls ** 2)))) ** 2  # Gravity considering bob's effect
    ugib = gib * np.sqrt((2 * umt / mt) ** 2 + ((4 * lb * ulb) / (5 * (ls ** 2) + lb ** 2)) ** 2 + (((4 * (lb ** 2) * uls) / ((5 * (ls ** 3)) + (lb ** 2 * ls))) + (uls / ls)) ** 2)
    return gib, ugib

def gtheta(l, mt, theta, ul, umt, utheta):
    g =  ((4*((np.pi)**2)*l)/(mt**2)) * (1+(theta**2/16))**2
    ug = g*np.sqrt((ul/l)**2+(2*umt/mt)**2+((theta*utheta)/(4+(theta**2/4)))**2)
    return(g, ug)

def calculate_g_full(mt, l, mb, ms, lb, theta, umt, ul, umb, ums, ulb, utheta):
    m = ms/mb
    um = np.sqrt((ums/mb)**2+(ms*umb/mb**2)**2)
    inertia_string = (1/12) * ms  / (mb)
    inertia_bob = (1/5) * (lb**2) / (l**2)
    angle_corr = (1/8) * theta**2 + (1/256) * theta**4
    correction = 1 - inertia_string + inertia_bob + angle_corr
    g = (4 * np.pi**2 * l / mt**2) * correction
    #ug = np.sqrt((4*np.pi**2*l*ul*(1-m/12+lb**2/(5*l**2)+theta**2/8+theta**4/256)/mt**2)**2+(8*np.pi**2*l*umt*(1-m/12-lb**2/(5*l**2)+theta**2/8+theta**4/256)/umt**3)**2+(8*np.pi**2*ulb*um/(5*mt**2*l))**2+(4*np.pi**2*l*utheta*(theta/4+theta**3/64)/mt**2)**2)
    pi2 = np.pi**2

    A = (4 * pi2/ mt**2)**2 * ul**2 * (1-(mt/12)-(lb**2/(5*l**2))+(theta**2/8)+(theta**4/256))

    B = (8 * pi2 * lb / (5 * mt**2*l))**2 * um**2

    C = (8 * pi2 * l / mt**3)**2 * umt**2 * (1-(mt/12)+(lb**2/(5*l**2))+(theta**2/8)+(theta**4/256))

    angle_sens = theta*(1/4) + (3/64) * theta**3
    D = (4 * pi2 * l / mt**2 * angle_sens)**2 * utheta**2

    # Total uncertainty
    ug = np.sqrt(A + B + C + D)
    return(g, ug, np.sqrt(A), np.sqrt(B), np.sqrt(C), np.sqrt(D))

# Given values
ls = 7.51  # Length of string
uls = 0.01  # Uncertainty of length of string
lb = 0.02635  # Radius of bob
ulb = 0.0001  # Uncertainty of radius of bob
l = ls + lb  # Length to center of mass 
ul = np.sqrt(uls ** 2 + ulb ** 2)  # Uncertainty of length to center of mass
ms = (0.000281/0.993)*l  #Mass of string
ums = 0.000001  #Uncertainty of mass of string
mb = 0.541 #Mass of bob
umb = 0.001 #Uncertainty of mass of bob

# Time data
time = np.array([
    [27.59, 27.57, 27.51, 27.62, 0, 0], [27.54, 27.55, 27.67, 27.44, 27.49, 27.62],
    [27.53, 27.54, 27.56, 27.47, 0, 0], [27.52, 27.56, 27.56, 27.50, 27.61, 0], 
    [27.49, 27.59, 27.52, 0, 0, 0], [27.54, 27.56, 27.52, 27.57, 0, 0], 
    [27.59, 27.44, 27.60, 0, 0, 0], [27.59, 27.47, 27.89, 27.30, 27.42, 27.61], 
    [27.40, 27.59, 27.58, 27.57, 0, 0], [27.27, 27.62, 27.44, 27.79, 27.39, 0],
    [27.54, 27.58, 27.61, 27.43, 27.56, 0], [27.53, 27.58, 27.58, 27.60, 0, 0], 
    [27.65, 27.46, 27.61, 27.50, 0, 0], [27.63, 27.40, 27.60, 27.61, 27.43, 0], 
    [27.47, 27.64, 27.54, 27.51, 27.54, 27.54], [27.57, 27.56, 27.48, 27.51, 0, 0], 
    [27.48, 27.52, 27.66, 27.60, 27.55, 0], [27.53, 27.66, 27.56, 27.49, 27.70, 0], 
    [27.49, 27.55, 27.63, 27.53, 0, 0], [27.48, 27.59, 27.56, 27.51, 0, 0], 
    [27.28, 27.68, 27.54, 27.15, 27.56, 27.55], [27.50, 27.72, 27.30, 27.73, 27.5, 27.57], 
    [27.55, 27.48, 27.76, 0, 0, 0], [27.38, 27.64, 27.40, 27.60, 27.51, 27.63], [27.38, 27.64, 27.40, 27.60, 27.51, 27.63]
])

# Calculate total time for each trial (sum of nonzero values)
total_times = np.array([np.sum(row[row > 0]) for row in time])

# Calculate the number of oscillations for each trial
num_oscillations = np.array([5 * np.count_nonzero(row) for row in time])

# Calculate periods (total time divided by number of oscillations)
periods = total_times / num_oscillations

# Compute trial_periods by averaging every 5 periods
total_trials = len(periods) // 5
trial_periods = np.array([np.mean(periods[i*5:(i+1)*5]) for i in range(total_trials)])
uncertainty_trial_periods = np.array([np.std(periods[i*5:(i+1)*5]) / np.sqrt(5) for i in range(total_trials)])

# Angle calculations
x = ls - 0.054
ux = 0.01
y = np.array([0.377, 0.15, 0.485, 0.61, 0.07])
uy = 0.01

theta = np.zeros(len(y)) 
utheta = np.zeros(len(y)) 


# Compute angles
for i in range(len(y)):
    theta[i] = np.arctan(y[i] / x)
    utheta[i] = np.sqrt(((y[i] * ux) / (x**2 + y[i]**2))**2 + ((x * uy) / (x**2 + y[i]**2))**2)

# Compute mean and uncertainty for each g value array
def compute_mean_and_uncertainty(g_values, ug_values):
    mean_g = np.mean(g_values)
    total_uncertainty = np.mean(ug_values)  # Quadrature sum of uncertainties
    return mean_g, total_uncertainty

# Compute gravity for each trial period
g = np.zeros(len(y))
ug = np.zeros(len(y))
for i in range(len(y)):
    g[i], ug[i] = gravity(l, trial_periods[i], ul, uncertainty_trial_periods[i])
mean_g, total_ug = compute_mean_and_uncertainty(g, ug)
print("Computed g values:", g)
print("Uncertainty in g values:", ug)
print("Mean g:", mean_g, "Total Uncertainty:", total_ug)

# Compute gravity considering inertia of the string
gis = np.zeros(len(y))
ugis = np.zeros(len(y))
for i in range(len(y)):
    gis[i], ugis[i] = string(l, trial_periods[i], ms, mb, ul, uncertainty_trial_periods[i], ums, umb)
mean_gis, total_ugis = compute_mean_and_uncertainty(gis, ugis)
print("Computed gis values:", gis)
print("Uncertainty in gis values:", ugis)
print("Mean gis:", mean_gis, "Total Uncertainty:", total_ugis)

# Compute gravity considering the bob's effect
gib = np.zeros(len(y))
ugib = np.zeros(len(y))
for i in range(len(y)):
    gib[i], ugib[i] = bob(ls, trial_periods[i], lb, uls, uncertainty_trial_periods[i], ulb)
mean_gib, total_ugib = compute_mean_and_uncertainty(gib, ugib)
print("Computed gib values:", gib)
print("Uncertainty in gib values:", ugib)
print("Mean gib:", mean_gib, "Total Uncertainty:", total_ugib)

gt = np.zeros(len(y))
ugt = np.zeros(len(y))

for i in range(len(y)):
    gt[i], ugt[i] = gtheta(l, trial_periods[i], theta[i], ul, uncertainty_trial_periods[i], utheta[i])
mean_gt, total_ugt = compute_mean_and_uncertainty(gt, ugt)
print("Computed gt values:", gt)
print("Uncertainty in gt values:", ugt)
print("Mean gt:", mean_gt, "Total Uncertainty:", total_ugt)

for i in range(len(y)):
    gt[i], ugt[i], A, B, C, D = calculate_g_full(trial_periods[i], l, mb, ms, lb, theta[i], uncertainty_trial_periods[i], ul, umb, ms, ulb, utheta[i])
mean_gt, total_ugt = compute_mean_and_uncertainty(gt, ugt)
print(A, B, C, D)
print("Computed g cor values:", gt)
print("Uncertainty in g cor values:", ugt)
print("Discrepancy:", 9.807-gt)
print("Mean gt:", mean_gt, "Total Uncertainty:", total_ugt)