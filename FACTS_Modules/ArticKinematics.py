#ArticKinematics module is the articulatory kinematic plant.
#It receives articulatory motor command (adotdot) and 
#integrates it to get the articulatory state varialbes.
#It can also add noise to adotdot before the integration.

#Integrator that uses a single-step solver (Runge-Kutta)
#for ordinary differential equations.  It is modelled to 
#mimic ode45 (a nonstiff differential equations solver)
#from Matlab. In FACTS, this this simply integrates adotdot
#and adot to compute the current articulatroy state
#(position and velocity).

from .ode45_replica import ode45_dim6
from scipy.integrate import solve_ivp
from .util import string2dtype_array
import global_variables as gv
import numpy as np

class ArticKinematics():
    def run(self, prev_a_actual, adotdot, ms_frm):
        a = solve_ivp(fun=lambda t, y: ode45_dim6(t,y,adotdot), 
                      t_span=[0, ms_frm/1000], y0=prev_a_actual, method='RK45', 
                      dense_output=True, rtol=1e-13, atol=1e-22).y[:,-1]
        return a

class ArticKinematics_Noise(ArticKinematics):
    def __init__(self,kin_configs):
        self.norms_adotdot = string2dtype_array(kin_configs['norms_ADOTDOT'],'float32')
        self.plant_scale = float(kin_configs['plant_scale']) #for controller noise, move it later 1/26/2021
    def run(self, prev_a_actual, adotdot, ms_frm):
        adotdot_noise = add_plant_noise(self.plant_scale,self.norms_adotdot,adotdot)
        a = super().run(prev_a_actual, adotdot_noise, ms_frm)
        return a

def add_plant_noise(plant_scale,norms_adotdot,adotdot):
    np.random.seed()
    adotdot_noise = adotdot + plant_scale*norms_adotdot*np.random.normal(0,1,gv.a_dim)
    #print("adotdot", adotdot)
    #print("adotdot_noise", adotdot_noise)
    return adotdot_noise

    