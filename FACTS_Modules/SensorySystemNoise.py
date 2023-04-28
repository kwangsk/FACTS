# Module to add Auditory and Somatosensory noise

from .util import string2dtype_array
import numpy as np
import global_variables as gv

# Gaussian case -- noise is simulated as Gaussian noise (np.random.normal).
class SensorySystemNoise():
    def __init__(self,sensory_configs):
        # read in config data
        if sensory_configs:
            Auditory_sensor_scale = float(sensory_configs['Auditory_sensor_scale'])
            Somato_sensor_scale = float(sensory_configs['Somato_sensor_scale'])
            nAuditory = int(sensory_configs['nAuditory'])
            norms_Auditory = string2dtype_array(sensory_configs['norms_Auditory'],'float32')#will have to tune later KSK 1/20/2021
            norms_AADOT = string2dtype_array(sensory_configs['norms_AADOT'],'float32') #will have to tune later KSK 1/20/2021
            # set class data
            self.R_Auditory = 1e0*Auditory_sensor_scale*np.ones(nAuditory)*norms_Auditory
            self.R_Somato = 1e0*Somato_sensor_scale*np.ones(gv.a_dim*2)*norms_AADOT
            print(self.R_Auditory)
    def run(self,Auditory_sense,Somato_sense):
        #generate perceived feedcback (plant output + noise + AAF)
        np.random.seed()
        Wsensor_Auditory = self.R_Auditory*np.random.normal(0,1,3) #sensor noise standard deviation in Hz
        Wsensor_Somato = self.R_Somato*np.random.normal(0,1,2*gv.a_dim) #sensor noise standard deviation in maeda unit
        Auditory_sense = Auditory_sense + Wsensor_Auditory        
        Somato_sense = Somato_sense + Wsensor_Somato
        return  Auditory_sense, Somato_sense
    def get_R_Auditory(self):
        return self.R_Auditory
    def get_R_Somato(self):
        return self.R_Somato

# Pass-through case, no noise added    
class SensorySystemNoise_None(SensorySystemNoise):
    def __init__(self):
        self.R_Auditory = np.array([])
        self.R_Somato = np.array([])
    def run(self,Auditory_sense,Somato_sense):
        return  Auditory_sense, Somato_sense
        