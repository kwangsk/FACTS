#Articulatory State Feedback Control Law
#The code is largely based on Saltzman & Munhall (1989).
#Implemented in Python by Kwang S. Kim and Jessica L. Gaines.

# Saltzman, E. L., & Munhall, K. G. (1989). 
# A dynamical approach to gestural patterning 
# in speech production. Ecological psychology, 
#1(4), 333-382.


import numpy as np
import global_variables as gv
from .util import string2dtype_array
from .LWPR_Model.lwpr import LWPR

class ArticSFCLaw():
    def __init__(self):
        self.prejb = []
        self.reset_prejb()
    def run(self, xdotdot,a_tilde,ART,i_frm,PROMACT,ms_frm):
        # Overridden by child class
        adotdot = []
        return adotdot;
    def reset_prejb(self):
        self.prejb = np.zeros((gv.x_dim, gv.a_dim))

#A "debug" mode. In this mode, the Jacobian used is 
#the artic-to-task transformation is changing throughout adaptation.
#Used for Fig 5. See the "regular" mode below for more detailed 
#information on each step.
class ArticSFCLaw_LWPR_JacUpdateDebug(ArticSFCLaw):
    def __init__(self,artic_configs):
        super().__init__()
        self.nullmodel = LWPR(artic_configs['model_path'])

    def run(self, xdotdot,a_tilde,ART,i_frm,PROMACT,ms_frm, model,catch):
        PROM_NEUT = get_prom_neut(ART,i_frm)
        NEUTACC,NULLACC = get_neutacc_nullacc(a_tilde)

        if catch == 1 or catch == 2:  #2 null in TSE.
            jb = model.predict_J(a_tilde[0:gv.a_dim]) #learned LWPR jacobian  
            print("STILL USING LEARNED")          
        else:#catch 3, 4 , 5 #5 null jacobian in SFC.
            jb = self.nullmodel.predict_J(a_tilde[0:gv.a_dim]) #null (unlearned) jacobian

        JNTACC_NULL, JAC, IPJAC = get_null_prj(ART,i_frm,PROMACT,jb,NULLACC)

        #Jdot*adot
        Jdot = (JAC-self.prejb)/(ms_frm/1000) #Jdot has to be estimated
        #because it cannot be computed arithmetically as in TADA

        Jdotadot = np.matmul(Jdot,a_tilde[gv.a_dim:2*gv.a_dim])
        
        self.prejb = JAC #save the current jacobian matrix for the next frame's
        #Jdot estimation.

        #adotdot = fromtask + NullProj for stability + NeutralAtt for nonactive gestures
        adotdot = np.matmul(IPJAC,xdotdot) - np.matmul(IPJAC,Jdotadot)  + JNTACC_NULL + np.multiply(PROM_NEUT,NEUTACC) 
        #print("adotdot: ",adotdot)
        return adotdot
    
#"Regular" mode used in most situations
#The Jacobian used is always native.
class ArticSFCLaw_LWPR_noupdate(ArticSFCLaw):
    def __init__(self,artic_configs):
        #print("Update model")
        super().__init__()
        self.model = LWPR(artic_configs['model_path'])

    def run(self, xdotdot,a_tilde,ART,i_frm,PROMACT,ms_frm):
        PROM_NEUT = get_prom_neut(ART,i_frm) #Gating for neutral task
        NEUTACC,NULLACC = get_neutacc_nullacc(a_tilde) #Compute neutral attractor and null projection
        jb = self.model.predict_J(a_tilde[0:gv.a_dim]) #LWPR jacobian
        JNTACC_NULL, JAC, IPJAC = get_null_prj(ART,i_frm,PROMACT,jb,NULLACC)
        Jdot = (JAC-self.prejb)/(ms_frm/1000) #an estimate of Jdot
        Jdotadot = np.matmul(Jdot,a_tilde[gv.a_dim:2*gv.a_dim]) #Jdot*adot
        self.prejb = JAC

        #adotdot = fromtask + NullProj for stability + NeutralAtt for nonactive gestures
        adotdot = np.matmul(IPJAC,xdotdot) - np.matmul(IPJAC,Jdotadot)  + JNTACC_NULL + np.multiply(PROM_NEUT,NEUTACC) 
        
        #print("online: ",np.matmul(IPJAC,xdotdot))
        return adotdot

    
def get_prom_neut(ART,i_frm):
    PROM_NEUT= [None] * gv.a_dim
    for n in range(gv.a_dim):
        PROM_NEUT[n] = ART[n]["PROM_NEUT"][i_frm]
    return PROM_NEUT

#Computes neutral attractor and null projection (damping only)
def get_neutacc_nullacc(a_tilde):
    NEUTARTIC_DEL = a_tilde[0:gv.a_dim] - gv.RESTAN
    NEUTACC_SPR = gv.k_NEUT * NEUTARTIC_DEL
    NEUTACC_DMP = gv.d_NEUT * a_tilde[gv.a_dim:2*gv.a_dim]
    NEUTACC = -NEUTACC_SPR - NEUTACC_DMP
    # null projection (with damping only)
    NULLACC_SPR = -NEUTACC_SPR
    NULLACC_DMP = -NEUTACC_DMP
    NULL_KSCL = 0
    NULL_DSCL = 1 #damping only
    NULLACC = NULL_KSCL *NULLACC_SPR + NULL_DSCL *NULLACC_DMP
    return NEUTACC,NULLACC

#Computes the final null projection, inverse jacobian, and etc.
def get_null_prj(ART,i_frm,PROMACT,jb,NULLACC):
    TOTWGT = [None] * gv.a_dim
    PROM_ACT_JNT = [None] * gv.a_dim
    for n in range(gv.a_dim):
        TOTWGT[n] = ART[n]["TOTWGT"][i_frm]
        PROM_ACT_JNT[n] = ART[n]["PROM_ACT_JNT"][i_frm]
    PROMACT = np.diag(PROMACT)
    TOTWGT = np.diag(TOTWGT)
    Ident_TV = np.eye(gv.x_dim, gv.x_dim)
    TOTWGTINV = np.linalg.pinv(TOTWGT)
    JAC = np.matmul(PROMACT,jb[1]) # multiply active task by jacobian
    TJAC = np.transpose(JAC) 
    TOTWGTINV_TJAC = np.matmul(TOTWGTINV,TJAC) #inverse weight multiplied by trasponsed active task jacobian
    #print("JAC",JAC)
    #print("TOTWGTINV_TJAC",TOTWGTINV_TJAC)
    IPJAC= np.matmul(TOTWGTINV_TJAC, np.linalg.pinv(np.matmul(JAC,TOTWGTINV_TJAC) + Ident_TV - PROMACT)) 
    Ident_ARTIC = np.eye(gv.a_dim,gv.a_dim)
        
    #Id - J* J 
    NULLPROJ = np.diag(np.matmul(Ident_ARTIC,PROM_ACT_JNT)) - np.matmul(IPJAC,JAC)
    
    #Null Proj * NULLACC : only damping, no stiffness
    JNTACC_NULL = np.matmul(NULLPROJ,NULLACC)
    return JNTACC_NULL, JAC, IPJAC

