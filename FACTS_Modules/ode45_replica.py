#The differential "equations" are just simply expressed
#as variables from the state estimate.

import numpy as np


def fdebug(t,y,adotdot):
    #=== RUNGE-KUTTA===%
    adot = y[10:20];
    dy = np.append(adot,adotdot)
    return dy

def ode45_dim6(t,y,adotdot):
    #=== RUNGE-KUTTA===%
    adot = y[6:12];
    dy = np.append(adot,adotdot)
    return dy
