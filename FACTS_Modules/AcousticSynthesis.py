#AcousticSynthesis module is the acoustic plant (Maeda).
#This module returns the first three formants (F1, F2, F3)
#produced by the High Resolution Maeda model.

#The source code for Maeda model can be retrieved from
#https://github.com/sensein/VocalTractModels, which is
#licensed under the Apache License, Version 2.0.

from .util import string2dtype_array
import global_variables as gv
import numpy as np
from .maeda import maedaplant

class AcousticSynthesis():
    def __init__(self, synth_configs):
        self.TC = string2dtype_array(synth_configs['TC'],'float32')
        self.PC = string2dtype_array(synth_configs['PC'],'float32')
        self.anc = float(synth_configs['anc'])
    def run(self, a):
        AM = np.zeros(7,dtype='float32') #Maeda model requires 6 articulators + larynx
        AM[0:gv.a_dim] = a[0:gv.a_dim]
        formants,internal_x,internal_y,external_x,external_y= maedaplant(5,29,29,29,29,self.TC,self.PC,AM,self.anc)
        return formants[0:3]