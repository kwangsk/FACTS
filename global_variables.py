# This file contains all *constant* global variables for the FACTS model. Please use and define carefully.

import numpy as np
import math

M = []  # Mass of the Task Dynamics dynamical system 
B = []  # Damping of the Task Dynamics dynamical system 
K = []  # Stiffness of the Task Dynamics dynamical system 

#Defining Constants used for Neutral attractors
#RESTAN = np.array([-0.6,-0.18,0,0,1,0]) #neutral artic position (rest position)
RESTAN = np.array([-0.7,0,0,0,1.1,0]) #neutral artic position (rest position)
#RESTAN = np.array([0.0,0.0,0.0,0.0,0.0,0.0])
FREQ  = np.array([2.0,2.0,2.0,2.0,2.0,2.0]) * 2.0 * math.pi # FREQ. = RAD/SEC             
k_NEUT = FREQ**2                       # STIFFNESS.
ARTIC_DAMPRAT = np.ones(6,'double')
d_NEUT = 2 * ARTIC_DAMPRAT * FREQ  # DAMPING COEFFICIENT.

x_dim = 7
a_dim = len(ARTIC_DAMPRAT)
