#This module's content is a Python-implemented version of
#the TADA code (make_gest.m) which can be requested via
#https://haskinslabs.org/about-us/features-and-demos/
#tada-task-dynamic-model-inter-articulator-speech-coordination.
#The original code (in Matlab) is owned and managed by Hosung Nam 
#(hnam@korea.ac.kr).

#Nam, H., Goldstein, L., Saltzman, E., & Byrd, D. (2004). 
#TADA: An enhanced, portable Task Dynamics model in MATLAB. 
#The Journal of the Acoustical Society of America, 115(5), 2430-2430.


import global_variables as gv
import numpy as np
import copy
import math

def MakeGestScore(fileName,target_noise):
    wag_frm = 5;

    #TV numbers (note that it is 1 less than matlab index because python index starts at 0 rather than 1)
    #parameters initialization
    # TV index
    TV_index = {"TT_Den":0,"TT_Alv":1,"TB_Pal":2,"TB_Vel":3,"TB_Pha":4,"LA":5,"LPRO":6}    
    nTV = len(TV_index)
    
    #These are counters that will be used for allocating tract variables (TV_SCORE)
    TV_counter = np.zeros(nTV)
   
    LastFrm = [None] * 0
    with open(fileName, "r") as f:
        for line in f:
            #print(line)
            gfileElement=line.split()
            if len(gfileElement) > 3:
                LastFrm.append(int(gfileElement[3], base = 10))
            if 'str' in line:
                break
    
    f = open(fileName, "r") # open the gestural score file 
    gfileLine = f.readline() # scan the first line
    gfileElement = gfileLine.split() # scan the first element
    ms_frm = int(gfileElement[0],base=10) #msec frame
    last_frm = int(gfileElement[1],base=10) #last frame No.
    if max(LastFrm) > last_frm:
        last_frm = max(LastFrm)

    n_frm = int((last_frm)*ms_frm/wag_frm);

    phon_onset = gfileElement[2]
    if len(gfileElement) > 3:
        phon_offset = gfileElement[3]
    else:
        phon_offset = float("NaN")

    GEST = {"BEG":0,"END":0,"xVALUE":0,"kVALUE":0,"dVALUE":0,"wVALUE":0,"alpha":0,"beta":0,
            "phon_onset":phon_onset,"phon_offset":phon_offset,"PROM":np.zeros(n_frm),
            "PROMSUM":np.zeros(n_frm)}
    
    GestList = [None] * 0
    GestList.append(copy.deepcopy(GEST))
    TV_SCORE = [None] * 0
    for i in range(nTV):       
        TV_SCORE.append(copy.deepcopy(GestList))
        
    #Read each line from g file and allocate TV according to the info in the g file
    for line in f:
        #print(line)
        gfileElement=line.split()
        if gfileElement[0].strip("''") in TV_index:
            currentInd = TV_index[gfileElement[0].strip("''")]
            TV_counter[currentInd] = TV_counter[currentInd] + 1 
            TV_SCORE = allocateGestScore(TV_SCORE, gfileElement, currentInd, TV_counter[currentInd], n_frm, ms_frm/wag_frm, TV_index,target_noise)        
        else: 
            raise NameError('Gestural Score Error: check the gestural score file')
        if 'str' in line:
            break        
        
###################UPDATE TV SCORE for PARAMETERS#############################
    #Blending different lines of TV.g files accoridng to each task state index 
    TV_SCORE = BlendTV(TV_SCORE,n_frm)    
    # Note for later... work on ART term and weight matrix (TOTWGT), necessary for ArticSFCLaw
    ART = makeTOTWGT(TV_SCORE,ms_frm, last_frm,5) 
    return TV_SCORE, ART, ms_frm, last_frm

def makeTOTWGT(TV_SCORE,ms_frm,last_frm,wag_frm):
    # parameters initialization
    n_frm = int((last_frm)*ms_frm/wag_frm)
    
    # Creating an ART (will be used for neutral attractor)
    #"PROMSUM_JNT":np.zeros(n_frm) does not seem to be needed. Commented out 11/28/2020 KSK
    ART = [None] * 0
    ART_dict = {"TOTWGT":np.zeros(n_frm),"PROM_ACT_JNT":np.zeros(n_frm),"PROM_NEUT":np.zeros(n_frm)}
    
    #Compute PROM_ACT_JNT and TOTWGT
    PROMSUM_sumforPROMACTJNT = np.zeros(n_frm)
    MAXPROM_JNT = 1
    PROMSUM_WGT_sum = np.zeros(n_frm)
    PROMSUM_sum = np.zeros(n_frm)
    WGT_NEUT = 1.0
    
    for i_ART in range(gv.a_dim): # number of maeda input
        #re-define i_ARTIC_TV here
        
        if i_ART < 4: #JA TG TS TA
            i_ARTIC_TV = [0,1,2,3,4] # TT_Den TT_Alv TB_Pal TB_Vel TB_Pha 
        elif i_ART == 4: #LH
            i_ARTIC_TV = [5] #LA
        elif i_ART == 5: #LP
            i_ARTIC_TV = [6] #LPRO
        else:
            i_ARTIC_TV = []
        
        ART.append(copy.deepcopy(ART_dict))
        #Compute PROM_ACT_JNT and TOTWGT
        PROMSUM_sumforPROMACTJNT = np.zeros(n_frm)
        MAXPROM_JNT = 1
        PROMSUM_WGT_sum = np.zeros(n_frm)
        PROMSUM_sum = np.zeros(n_frm)
        WGT_NEUT = 1.0

        for nframe in range(n_frm):
            for i in i_ARTIC_TV:       
                PROMSUM_WGT_sum[nframe] += min(1, TV_SCORE[i][0]["PROMSUM"][nframe]) * TV_SCORE[i][0]["WGT_TV"][nframe,i]
                PROMSUM_sum[nframe] += min(1, TV_SCORE[i][0]["PROMSUM"][nframe])
                PROMSUM_sumforPROMACTJNT[nframe] += TV_SCORE[i][0]["PROMSUM"][nframe]
            ART[i_ART]["PROM_ACT_JNT"][nframe] = min (MAXPROM_JNT, PROMSUM_sumforPROMACTJNT[nframe])
            ART[i_ART]["PROM_NEUT"][nframe] = 1.0 - ART[i_ART]["PROM_ACT_JNT"][nframe]
            ART[i_ART]["TOTWGT"][nframe] = (PROMSUM_WGT_sum[nframe] + ART[i_ART]["PROM_NEUT"][nframe] * WGT_NEUT)/(PROMSUM_sum[nframe] + ART[i_ART]["PROM_NEUT"][nframe])
        #numerically check this loop by changing matlab input/output...
    return ART

def BlendTV(TV_SCORE,n_frm): 
    wgtshape = [n_frm, 7] #surprised that i have to do this in python.. because i can't do np.zeros(n_frm, 7) for WGT_TV below
    for i_TV in range(len(TV_SCORE)):
        #print(i_TV)
        
        #Initializing the final d_blend, k_blend, x_blend which will be used in the SFCLaw
        TV_SCORE[i_TV][0].update({"xBLEND":np.zeros(n_frm),"kBLEND":np.zeros(n_frm),"dBLEND":np.zeros(n_frm),"WGT_TV":np.zeros(wgtshape)})
            
        # This code seems to be a bit of switcheroo
        for ind1 in range(len(TV_SCORE[i_TV])):
            #Initializing values (being defined here for the first time)
            TV_SCORE[i_TV][ind1].update({"xPROMSUM_BLEND":np.zeros(n_frm),"kPROMSUM_BLEND":np.zeros(n_frm),"dPROMSUM_BLEND":np.zeros(n_frm),"wPROMSUM_BLEND":np.zeros(n_frm)})
              
            for ind2 in range(len(TV_SCORE[i_TV])): #First nested forloop adds (+) elements over differnet nGest
                TV_SCORE[i_TV][ind1]["xPROMSUM_BLEND"] = (copy.deepcopy(TV_SCORE[i_TV][ind1]["xPROMSUM_BLEND"])
                                                         + copy.deepcopy(TV_SCORE[i_TV][ind2]["alpha"]) 
                                                         * copy.deepcopy(TV_SCORE[i_TV][ind2]["PROM"]))
                TV_SCORE[i_TV][ind1]["kPROMSUM_BLEND"] = (copy.deepcopy(TV_SCORE[i_TV][ind1]["kPROMSUM_BLEND"])
                                                         + copy.deepcopy(TV_SCORE[i_TV][ind2]["alpha"]) 
                                                         * copy.deepcopy(TV_SCORE[i_TV][ind2]["PROM"]))
                TV_SCORE[i_TV][ind1]["dPROMSUM_BLEND"] = (copy.deepcopy(TV_SCORE[i_TV][ind1]["dPROMSUM_BLEND"])
                                                         + copy.deepcopy(TV_SCORE[i_TV][ind2]["alpha"]) 
                                                         * copy.deepcopy(TV_SCORE[i_TV][ind2]["PROM"]))
                TV_SCORE[i_TV][ind1]["wPROMSUM_BLEND"] = (copy.deepcopy(TV_SCORE[i_TV][ind1]["wPROMSUM_BLEND"])
                                                         + copy.deepcopy(TV_SCORE[i_TV][ind2]["alpha"]) 
                                                         * copy.deepcopy(TV_SCORE[i_TV][ind2]["PROM"]))
            
            #Then you substract (-) the ind1 (original alpha*prom) from the sum    
            TV_SCORE[i_TV][ind1]["xPROMSUM_BLEND"] = (copy.deepcopy(TV_SCORE[i_TV][ind1]["xPROMSUM_BLEND"])
                                                     - copy.deepcopy(TV_SCORE[i_TV][ind1]["alpha"]) 
                                                     * copy.deepcopy(TV_SCORE[i_TV][ind1]["PROM"]))
            TV_SCORE[i_TV][ind1]["kPROMSUM_BLEND"] = (copy.deepcopy(TV_SCORE[i_TV][ind1]["kPROMSUM_BLEND"])
                                                     - copy.deepcopy(TV_SCORE[i_TV][ind1]["alpha"]) 
                                                     * copy.deepcopy(TV_SCORE[i_TV][ind1]["PROM"]))
            TV_SCORE[i_TV][ind1]["dPROMSUM_BLEND"] = (copy.deepcopy(TV_SCORE[i_TV][ind1]["dPROMSUM_BLEND"])
                                                     - copy.deepcopy(TV_SCORE[i_TV][ind1]["alpha"]) 
                                                     * copy.deepcopy(TV_SCORE[i_TV][ind1]["PROM"]))
            TV_SCORE[i_TV][ind1]["wPROMSUM_BLEND"] = (copy.deepcopy(TV_SCORE[i_TV][ind1]["wPROMSUM_BLEND"])
                                                     - copy.deepcopy(TV_SCORE[i_TV][ind1]["alpha"]) 
                                                     * copy.deepcopy(TV_SCORE[i_TV][ind1]["PROM"]))
            #so for ind1, it will end up with ind2 alpha*prom
            #and for ind2, it will end up with ind1 alpha*prom, I guess this is what 'blending' is? 
            
            TV_SCORE[i_TV][ind1]["xPROMBLEND"] = (copy.deepcopy(TV_SCORE[i_TV][ind1]["PROM"]) /
                                                  (1+copy.deepcopy(TV_SCORE[i_TV][ind1]["beta"])*copy.deepcopy(TV_SCORE[i_TV][ind1]["xPROMSUM_BLEND"])))
            TV_SCORE[i_TV][ind1]["kPROMBLEND"] = (copy.deepcopy(TV_SCORE[i_TV][ind1]["PROM"]) /
                                                  (1+copy.deepcopy(TV_SCORE[i_TV][ind1]["beta"])*copy.deepcopy(TV_SCORE[i_TV][ind1]["kPROMSUM_BLEND"])))
            TV_SCORE[i_TV][ind1]["dPROMBLEND"] = (copy.deepcopy(TV_SCORE[i_TV][ind1]["PROM"]) /
                                                  (1+copy.deepcopy(TV_SCORE[i_TV][ind1]["beta"])*copy.deepcopy(TV_SCORE[i_TV][ind1]["dPROMSUM_BLEND"])))
            TV_SCORE[i_TV][ind1]["wPROMBLEND"] = (copy.deepcopy(TV_SCORE[i_TV][ind1]["PROM"]) /
                                                  (1+1*copy.deepcopy(TV_SCORE[i_TV][ind1]["wPROMSUM_BLEND"])))
            #wPROMBLEND in matlab version is never used, instead they create PROMSUM_BLEND_SYN which 
            #is very similar with wPROMBLEND except that the beta value is just 1 
            #To eliminate potential confusion, I will just use wPROMBLEND ans PROMSUM_BLEND_SYN

            #Now make the final d_blend, k_blend, x_blend which will be used in the SFCLaw
            TV_SCORE[i_TV][0]["xBLEND"] = (copy.deepcopy(TV_SCORE[i_TV][0]["xBLEND"]) +
                                                  (copy.deepcopy(TV_SCORE[i_TV][ind1]["xPROMBLEND"])*copy.deepcopy(TV_SCORE[i_TV][0]["xVALUE"])))
            TV_SCORE[i_TV][0]["kBLEND"] = (copy.deepcopy(TV_SCORE[i_TV][0]["kBLEND"]) +
                                                  (copy.deepcopy(TV_SCORE[i_TV][ind1]["kPROMBLEND"])*copy.deepcopy(TV_SCORE[i_TV][0]["kVALUE"])))
            TV_SCORE[i_TV][0]["dBLEND"] = (copy.deepcopy(TV_SCORE[i_TV][0]["dBLEND"]) +
                                                  (copy.deepcopy(TV_SCORE[i_TV][ind1]["dPROMBLEND"])*copy.deepcopy(TV_SCORE[i_TV][0]["dVALUE"])))            
            TV_SCORE[i_TV][0]["WGT_TV"] = (copy.deepcopy(TV_SCORE[i_TV][0]["WGT_TV"]) +
                                                (np.outer(copy.deepcopy(TV_SCORE[i_TV][ind1]["wPROMBLEND"]),copy.deepcopy(TV_SCORE[i_TV][0]["wVALUE"]))))  
            
            #Similarly, I will just use WGT_TV here because it is technically the same as matlab code (make_WGT_TV)
    
    return TV_SCORE

def allocateGestScore(TV_SCORE, gfile, i_TV, nGEST, n_frm, frmratio, index_TV,target_noise): 
    
    frq_tmp = float(gfile[6])*2*math.pi    #frq_tmp = w0
    
    #The next 5 lines equivalent to TV_SCORE(i_TV).GEST(nGEST).w.VALUE = get_w(fp, i_TV);   
    wvalue = [0] * len(index_TV) 
    wlist = gfile[8].split(',')
    for j in range(len(wlist)):
        art= getArticName(wlist[j][0:2])
        wvalue[art-1]=int(wlist[j][3],base=10)
        
    # Creating an Gest_dict
    GEST_dict = {"BEG":gfile[2],"END":gfile[3],"kVALUE":frq_tmp**2,"dVALUE":float(gfile[7])*2*frq_tmp,
                 "wVALUE":wvalue,"alpha":float(gfile[9]),"beta":float(gfile[10]),"PROM":np.zeros(n_frm)}
    if i_TV in index_TV.values():
        #GEST_dict["xVALUE"] = float(gfile[5]) 
        GEST_dict["xVALUE"] = np.random.normal(0,target_noise,1)+float(gfile[5]) #e.g., 0.2 mm std normal distribution
        #print(GEST_dict["xVALUE"])

    BEG_frm = int(GEST_dict["BEG"],base=10)*int(frmratio)
    END_frm = int(GEST_dict["END"],base=10)*int(frmratio)
    
    GEST_dict["PROM"][range(BEG_frm,END_frm)] = 1
  
    if nGEST < 2: 
        # Adding dictionary as a list element to TV_SCORE (first time)
        GEST_dict["PROMSUM"] = copy.deepcopy(GEST_dict["PROM"])
        TV_SCORE[i_TV][0] = copy.deepcopy(GEST_dict)
    else:
        GEST_dict["PROMSUM"] = copy.deepcopy(TV_SCORE[i_TV][0]["PROMSUM"]) + copy.deepcopy(GEST_dict["PROM"])
        TV_SCORE[i_TV].append(copy.deepcopy(GEST_dict))
    return TV_SCORE

def getArticName(Name):
    ArtDict = {"JA":1,"TG":2,"TS":3,"TA":4,"LH":5,"LP":6}
    return ArtDict[Name]
    
def read_file(filename):
    with open(filename, encoding='utf-8') as file:
        return file.readlines()