import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

def angle(x, y, ux, uy, t):
    theta = np.arctan(x/y)
    utheta = np.sqrt(((y*ux)/(x**2+y**2))**2+((x*uy)/(x**2+y**2))**2)
    mt = np.mean(t) #Mean period
    ut = np.std(t)  #Uncertainty of a measurement of the period
    umt = ut/np.sqrt(len(t))    #Uncertainty of the mean
    return(theta, utheta, mt, ut, umt)

def gravity(l, mt, theta, ul, umt, utheta):
    g = (4*((np.pi)**2)*l)/(mt**2)*(1-(ms/(12*mb))+(lb**2/(5*(l**2)))* (1+(theta**2/16))**2)**2
    ug = g*np.sqrt((ul/l)**2+(2*umt/mt)**2+((theta*utheta)/(4+(theta**2/4)))**2)
    return(g, ug)


ls = 0.305  #Length of string
uls = 0.0005    #Uncertainty of length of string

lb = 0.02635  #Radius of bob
ulb = 0.0001    #Uncertainty of radius of bob

l = ls + lb #Length of center of mass 
ul = np.sqrt(uls**2 + ulb**2)   #Uncertainty of length of center of mass

n = 10  #Number of oscillations
y = 0.095
uy = 0.001
ux = 0.001



time1 = np.array([11.68, 11.57, 11.70, 11.56, 11.66, 11.68, 11.71, 11.63, 11.46, 11.64, 11.63])   #Trial value in array
t1 = time1 / n    #Period value in array
x1 = 0.046
theta1, utheta1, mt1, ut1, umt1 = angle(x1, y, ux, uy, t1)
g1, ug1 = gravity(l, mt1, theta1, ul, umt1, utheta1)

time2 = np.array([11.77, 11.71, 11.81, 11.87, 11.76, 11.72, 11.75, 11.68, 11.73]) 
t2 = time2 / n    
x2 = 0.106
theta2, utheta2, mt2, ut2, umt2 = angle(x2, y, ux, uy, t2)
g2, ug2 = gravity(l, mt2, theta2, ul, umt2, utheta2)

time3 = np.array([11.64, 11.64, 11.53, 11.60, 11.60, 11.53, 11.50, 11.57, 11.52, 11.47]) 
t3 = time3 / n  
x3 = 0.032
theta3, utheta3, mt3, ut3, umt3 = angle(x3, y, ux, uy, t3)
g3, ug3 = gravity(l, mt3, theta3, ul, umt3, utheta3)

time4 = np.array([12.13, 12.06, 11.99, 11.98, 12.01, 11.98, 11.91, 11.94, 11.99, 11.97]) 
t4 = time4 / n  
x4 = 0.204
theta4, utheta4, mt4, ut4, umt4 = angle(x4, y, ux, uy, t4)
g4, ug4 = gravity(l, mt4, theta4, ul, umt4, utheta4)

ms = 0.000281   #Mass of string
ums = 0.000001  #Uncertainty of mass of string

mb = 0.541 #Mass of bob
umb = 0.001 #Uncertainty of mass of bob


print(f"Pendulum at {np.rad2deg(theta3):.4f}: g = {g3:.4f} ± {ug3:.4f} m/s²")
print(f"Pendulum at {np.rad2deg(theta1):.4f}: g = {g1:.4f} ± {ug1:.4f} m/s²")
print(f"Pendulum at {np.rad2deg(theta2):.4f}: g = {g2:.4f} ± {ug2:.4f} m/s²")
print(f"Pendulum at {np.rad2deg(theta4):.4f}: g = {g4:.4f} ± {ug4:.4f} m/s²")

theta_values = np.array([theta1, theta2, theta3, theta4])  # In radians
t_values = np.array([mt1, mt2, mt3, mt4])  # Measured periods
ut_values = np.array([umt1, umt2, umt3, umt4])  # Uncertainties

theta_degrees = np.rad2deg(theta_values)  # Convert theta to degrees

# Gravity constant
g = 9.81

# Generate a range of angles for the theoretical curve
theta_range = np.linspace(0, max(theta_values) * 1.1, 400)

# Compute theoretical period
z = 2 * np.pi * np.sqrt(l / g) * (1 + (theta_range**2 / 16))
theta_range_degrees = np.rad2deg(theta_range)  # Convert to degrees

# Define a new fit function with 2 degrees of freedom (A and B)
def new_fit_function(theta, A, B):
    return A * (1 + B * (theta**2))

# Define the objective function: Sum of squared differences
def objective(params, theta_values, t_values):
    A, B = params
    t_fit = new_fit_function(theta_values, A, B)
    return np.sum((t_fit - t_values) ** 2)

# Initial guess for A and B
initial_guess = [2 * np.pi * np.sqrt(l / g), 1/16]

# Perform optimization
result = minimize(objective, initial_guess, args=(theta_values, t_values))
A_fit, B_fit = result.x

# Generate fitted curve using the new model
t_fitted_new = new_fit_function(theta_range, A_fit, B_fit)

# Plot measured periods with uncertainties
plt.figure(figsize=(8, 5))
plt.errorbar(theta_degrees, t_values, yerr=ut_values, fmt='o', capsize=5, label="Measured Period values")

# Plot theoretical period
plt.plot(theta_range_degrees, z, 'r--', label="Theoretical Period")

# Plot new fitted curve
plt.plot(theta_range_degrees, t_fitted_new, 'b-', label="Curve Fit")

# Labels and title
plt.xlabel("Angle θ (degrees)")
plt.ylabel("Period (seconds)")
plt.title("Measured vs. Theoretical Period vs. New Curve Fit of Pendulum")
plt.legend()
plt.grid(True)

# Show plot
plt.show()