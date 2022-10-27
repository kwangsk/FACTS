#Task State Feedback Control Law
#The overall code is from TADA (e.g., a_forward.m in TADA) 
#and Saltzman & Munhall (1989). Here, we compute xdotdot 
#which is equivalent to tvdotdot in TADA.

#Nam, H., Goldstein, L., Saltzman, E., & Byrd, D. (2004). 
#TADA: An enhanced, portable Task Dynamics model in MATLAB. 
#The Journal of the Acoustical Society of America, 115(5), 2430-2430.

# Saltzman, E. L., & Munhall, K. G. (1989). 
# A dynamical approach to gestural patterning 
# in speech production. Ecological psychology, 
#1(4), 333-382.


import numpy as np
import math
import copy
import global_variables as gv

class TaskSFCLaw():
    def run(self,x_tilde,TV_SCORE,i_frm):
        d_BLEND = np.zeros(len(TV_SCORE))
        x_0 = np.zeros(len(TV_SCORE))
        k_BLEND = np.zeros(len(TV_SCORE))
        PROMACT= np.zeros(len(TV_SCORE))
    
        MAXPROM = 1
    
        for i_TV in range(len(TV_SCORE)):
            PROMACT[i_TV] = min(MAXPROM, math.ceil(copy.deepcopy(TV_SCORE[i_TV][0]["PROMSUM"][i_frm])))
            x_0[i_TV] = copy.deepcopy(TV_SCORE[i_TV][0]["xBLEND"][i_frm])
            k_BLEND[i_TV] = copy.deepcopy(TV_SCORE[i_TV][0]["kBLEND"][i_frm])
            d_BLEND[i_TV] = copy.deepcopy(TV_SCORE[i_TV][0]["dBLEND"][i_frm])
    
        
        B_times_xdot_tilde =  np.dot(np.diag(d_BLEND),x_tilde[gv.x_dim:2*gv.x_dim])
        K_times_x_tildeminusx0 = np.dot(np.diag(k_BLEND),x_tilde[0:gv.x_dim] - x_0) # using the old one without the 'spi' pi gesture functionality and etc
        xdotdot= -B_times_xdot_tilde -K_times_x_tildeminusx0 # task state acceleration (xdotdot)
        
        return xdotdot, PROMACT