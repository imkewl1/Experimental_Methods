import numpy as np
import matplotlib.pyplot as plt

def discrepancy(actual, u):
    dif = 9.81 - actual
    udif = u
    return(f"{dif:.4f} ± {u:.4f}")
ls = 0.429  #Length of string
uls = 0.0005    #Uncertainty of length of string

lb = 0.02635  #Radius of bob
ulb = 0.0001    #Uncertainty of radius of bob

l = ls + lb #Length of center of mass 
ul = np.sqrt(uls**2 + ulb**2)   #Uncertainty of length of center of mass

n = 10  #Number of oscillations
time = np.array([13.52, 13.51, 13.65, 13.56, 13.58, 13.48, 13.61, 13.64, 13.72, 13.54])   #Trial value in array
t = time / n    #Period value in array

mt = np.mean(t) #Mean period
ut = np.std(t)  #Uncertainty of a measurement of the period
umt = ut/np.sqrt(len(t))    #Uncertainty of the mean
print(mt)

ms = (0.000281/0.993)*l  #Mass of string
ums = 0.000001  #Uncertainty of mass of string

mb = 0.541 #Mass of bob
umb = 0.001 #Uncertainty of mass of bob

g = (4*((np.pi)**2)*l)/(mt**2)*(1-(ms/(12*mb))+(lb**2/(5*(l**2))))**2  #Gravity of simple pendulum
ug = g*np.sqrt((ul/l)**2+(2*umt/mt)**2)   #Uncertainty in gravity of simple pendulum

gis = ((4*((np.pi)**2)*l)/(mt**2))*(1-(ms/(12*mb)))**2  #Gravity of pendulum with inertia of string
ugis = g*np.sqrt((ul/l)**2+(2*umt/mt)**2+(ums/(6*mb-0.5*ms))**2+((umb*ms*np.log(mb)/6)/(1-(ms/(12*mb))))**2)    #Uncertainty of pendulum with inertia of string

gib = ((4*((np.pi)**2)*l)/(mt**2))*(1+(lb**2/(5*(l**2))))**2    #Gravity of pendulum with inertia of ball
ugib = g*np.sqrt((2*umt/mt)**2+((4*lb*ulb)/(5*(l**2)+lb**2))**2+(((4*(lb**2)*ul)/((5*(l**3))+(lb**2*l)))+(ul/l))**2)

gibs = ((4*((np.pi)**2)*l)/(mt**2))*(((2*mb*lb/5)+(ms*(ls**2)/3)+(mb*((ls+lb)**2)))/((ms*ls/2)+(mb*ls)+(mb*lb)))  #Gravity of pendulum with inertia of ball and string



print(f"Pendulum with all factors: g = {g:.4f} ± {ug:.4f} m/s²")
print(discrepancy(g, ug))

