from .util import string2dtype_array
import numpy as np
import global_variables as gv

class AuditoryPerturbation():
    def __init__(self,auditory_perturb_configs):
        # read in config data
        if auditory_perturb_configs:
            self.PerturbMode = int(auditory_perturb_configs['PerturbMode'])
            self.PerturbExtentF1 = float(auditory_perturb_configs['PerturbExtentF1'])
            self.PerturbExtentF2 = float(auditory_perturb_configs['PerturbExtentF2'])
            self.PerturbExtentF3 = float(auditory_perturb_configs['PerturbExtentF3'])
            self.PerturbOnsetFrame = int(auditory_perturb_configs['PerturbOnsetFrame'])
            self.PerturbOffsetFrame = int(auditory_perturb_configs['PerturbOffsetFrame'])
            self.PerturbOnsetTrial = int(auditory_perturb_configs['PerturbOnsetTrial'])
            self.PerturbOffsetTrial = int(auditory_perturb_configs['PerturbOffsetTrial'])
            if 'PerturbRamp' in auditory_perturb_configs.keys():
                self.PerturbRamp = 1
                self.PerturbRampOnOff = string2dtype_array(auditory_perturb_configs['PerturbRamp'], dtype='int')
                self.PerturbExtentRampF1 = np.linspace(0,self.PerturbExtentF1,self.PerturbRampOnOff[1]-self.PerturbRampOnOff[0]+1)
                self.PerturbExtentRampF2 = np.linspace(0,self.PerturbExtentF2,self.PerturbRampOnOff[1]-self.PerturbRampOnOff[0]+1)
                self.PerturbExtentRampF3 = np.linspace(0,self.PerturbExtentF3,self.PerturbRampOnOff[1]-self.PerturbRampOnOff[0]+1)
            else:
                self.PerturbRamp = 0
        else:
            self.PerturbRamp = 0
            self.PerturbMode = 0
            self.PerturbOnsetFrame = -1
            self.PerturbOffsetFrame = -1
            self.PerturbOnsetTrial = -1
            self.PerturbOffsetTrial = -1

    def run(self,Auditory_sense,i_frm,trial,catch):
        #generate perceived feedcback (plant output + noise + AAF)
        formants_perturbed = [0,0,0]
        formants_perturbed[0] = Auditory_sense[0] 
        formants_perturbed[1] = Auditory_sense[1]      
        formants_perturbed[2] = Auditory_sense[2]
        
        if trial >= self.PerturbOnsetTrial and trial < self.PerturbOffsetTrial and self.PerturbMode !=0 and not catch:
            if i_frm >= self.PerturbOnsetFrame and i_frm < self.PerturbOffsetFrame and self.PerturbMode != 0:
                if self.PerturbRamp and trial < self.PerturbRampOnOff[1]:
                    if self.PerturbMode == 1: #absolute shift
                        print("pert F1 ram: ", self.PerturbExtentRampF1[trial-self.PerturbOnsetTrial])
                        formants_perturbed[0] = Auditory_sense[0] + self.PerturbExtentRampF1[trial-self.PerturbOnsetTrial]
                        formants_perturbed[1] = Auditory_sense[1] + self.PerturbExtentRampF2[trial-self.PerturbOnsetTrial]       
                        formants_perturbed[2] = Auditory_sense[2] + self.PerturbExtentRampF3[trial-self.PerturbOnsetTrial]
                    elif self.PerturbMode == 2: #cents shift
                        formants_perturbed[0] = Auditory_sense[0] * (2**(self.PerturbExtentRampF1[trial-self.PerturbOnsetTrial]/1200))
                        formants_perturbed[1] = Auditory_sense[1] * (2**(self.PerturbExtentRampF2[trial-self.PerturbOnsetTrial]/1200))
                        formants_perturbed[2] = Auditory_sense[2] * (2**(self.PerturbExtentRampF3[trial-self.PerturbOnsetTrial]/1200))
                else:
                    if self.PerturbMode == 1: #absolute shift
                        print("pert F1 sud: ", self.PerturbExtentF1)
                        formants_perturbed[0] = Auditory_sense[0] + self.PerturbExtentF1
                        formants_perturbed[1] = Auditory_sense[1] + self.PerturbExtentF2       
                        formants_perturbed[2] = Auditory_sense[2] + self.PerturbExtentF3
                    elif self.PerturbMode == 2: #cents shift
                        formants_perturbed[0] = Auditory_sense[0] * (2**(self.PerturbExtentF1/1200))
                        formants_perturbed[1] = Auditory_sense[1] * (2**(self.PerturbExtentF2/1200))
                        formants_perturbed[2] = Auditory_sense[2] * (2**(self.PerturbExtentF3/1200))  

        return  formants_perturbed

class AuditoryPerturbation_None(AuditoryPerturbation):
    def __init__(self):
        self.PerturbOnsetTrial = np.inf
        self.PerturbOffsetTrial = np.inf
    def run(self,Auditory_sense, i_frm, trial, catch):
        return  Auditory_sense