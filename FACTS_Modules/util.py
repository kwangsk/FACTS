#General utility methods
#A place holder because future verions
#of FACTS might need more shared methods
#in the model.

import numpy as np
import re
    
def string2dtype_array(string,dtype):
    string = re.sub(r'[\s\[\]]', '', string)
    string_array = np.array(re.split(',', string))
    if len(string_array) == 1 and string_array[0] == '':
        return np.array([])
    type_array = string_array.astype(dtype)
    return type_array

def scale(point,min,max):
    return (point - min)/(max-min)
def unscale(point,min,max):
    return (point * (max-min) + min)
    