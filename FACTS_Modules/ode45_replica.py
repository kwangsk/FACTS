# The differential "equations" are just simply expressed
# as variables from the state estimate.
# These are called to integrate adotdot via the Runge-Kutta method, 
# using the solve_ivp method (which is equivalent 
# to ode45 Runge-Kutta in Matlab)

import numpy as np

#debug function. Do not remove for now.
def fdebug(t,y,adotdot):
    #=== RUNGE-KUTTA===%
    adot = y[10:20];
    dy = np.append(adot,adotdot)
    return dy

#For 6 articulatory parameters, used by the plant.
def ode45_dim6(t,y,adotdot):
    #=== RUNGE-KUTTA===%
    adot = y[6:12];
    dy = np.append(adot,adotdot)
    return dy
