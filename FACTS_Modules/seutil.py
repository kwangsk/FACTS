# Various methods used for state estimators

#The UKF methods (e.g., Tigmas) contain contents 
#from the file exchange code published by Yi Cao:
#Yi Cao (2022). Learning the Unscented Kalman Filter 
#(https://www.mathworks.com/matlabcentral/fileexchange/18217-learning-the-unscented-kalman-filter)
#MATLAB Central File Exchange. Retrieved October 26, 2021.
#Copyright (c) 2009, Yi Cao All rights reserved.

import global_variables as gv
import numpy as np
from .ode45_replica import ode45_dim6
from scipy.integrate import solve_ivp

# Updating Articulatory State Prediction
# Design A in Kim et al. (in review)
def UpdateArticStatePrediction(ASPmodel,atildemem):
    for i in range(len(atildemem)):
        for z in range(gv.a_dim):
            atildeplusu = np.append(atildemem[i][z],np.append(atildemem[i][z+gv.a_dim],atildemem[i][z+gv.a_dim*2]))
            #print(atildeplusu)
            newatilde = np.append(atildemem[i][z+gv.a_dim*3],atildemem[i][z+gv.a_dim*4])
            #print(newatilde)
            ASPmodel[z].update(atildeplusu,newatilde)

    return ASPmodel

# Checks if the auditory prediction error exceeds the Auditory Prediction 
# Error Threshold (APET), and save them in the memory module if it does.
# Used by Design A (to update articulatory state prediction later)
def sensoryerrorandatildesave(y,z,senmem,x,i_frm,u,finalatilde,oldatilde,atildemem,APET):
    if abs(y[0]-z[0]) > APET: # will have to incorporate F2/F3 later
        #print("*****************perturb detect(*****************")
        senmem.append(np.append(x[0:gv.a_dim*2],z))
        #print(oldatilde) #pos + vel
        #print(u) #
        #print(finalatilde) # pos + vel
        #print(np.append(oldatilde,np.append(u,finalatilde)))
        atildemem.append(np.append(oldatilde,np.append(u,finalatilde)))
        
    return senmem, atildemem

# Checks if the auditory prediction error exceeds the Auditory Prediction 
# Error Threshold (APET), and save them in the memory module if it does.
# Used by Design C (to update articulatory-to-task state transformation later)
def auderror(y,z,senmem,x,taskmem,finalx,atilde,APET):
    if abs(y[0]-z[0]) > APET: # will have to incorporate F2/F3 later
    #if abs((y[0]-z[0])/z[0]) > APET:
        #print("*****************perturb detect(*****************")
        senmem.append(np.append(x[0:gv.x_dim],z))
        taskmem.append(np.append(atilde[0:gv.a_dim],finalx[0:gv.x_dim]))
    return senmem, taskmem

# Update sensory prediction module (used by Design B)
def UpdateSensoryPrediction(feedbackType, LWPRformant,LWPRsom,senmem):

    if feedbackType == 'audOnly':
        LWPRformant.init_lambda = 0.99
        LWPRformant.final_lambda =0.99
        #print("aud ref: ", LWPRformant.predict(np.array([ 0.01011647, -0.02873244, -0.08194522,  0.01002986, -0.00972066, -0.00251087])))
        #print(LWPRformant.predict(np.array([ 0.01011647, -0.02873244, -0.08194522,  0.01002986, -0.00972066, -0.00251087]))[0])
        for i in range(len(senmem)):
            LWPRformant.update(senmem[i][0:gv.x_dim],senmem[i][gv.x_dim:gv.x_dim+3])

    elif feedbackType == 'somatOnly':
        LWPRsom.init_lambda = 0.99
        LWPRsom.final_lambda =0.99
        #print(LWPRsom.num_rfs)
        #print("som ref: ", LWPRsom.predict(np.array([ 0.01011647,-0.02873244,-0.08194522,0.01002986,-0.00972066,-0.00251087,0.61945505,-1.22553245,0.68911496,0.40743455,-0.01990424,0.0036914])))
        #print(LWPRformant.predict(np.array([ 0.01011647, -0.02873244, -0.08194522,  0.01002986, -0.00972066, -0.00251087]))[0])
        for i in range(len(senmem)):
            LWPRformant.update(senmem[i][0:gv.a_dim],senmem[i][gv.a_dim*2:gv.a_dim*2+3])
            LWPRsom.update(senmem[i][0:gv.a_dim*2],senmem[i][gv.a_dim*2+3:gv.a_dim*4+3])

    elif feedbackType == 'full':
        #print("aud ref: ", LWPRformant.predict(np.array([ 0.01011647, -0.02873244, -0.08194522,  0.01002986, -0.00972066, -0.00251087])))
        #print("som ref: ", LWPRsom.predict(np.array([ 0.01011647,-0.02873244,-0.08194522,0.01002986,-0.00972066,-0.00251087,0.61945505,-1.22553245,0.68911496,0.40743455,-0.01990424,0.0036914])))
        #print(LWPRformant.predict(np.array([ 0.01011647, -0.02873244, -0.08194522,  0.01002986, -0.00972066, -0.00251087]))[0])
        for i in range(len(senmem)):
            #print("aud pre:", LWPRformant.predict(senmem[i][0:gv.a_dim]))
            LWPRformant.update(senmem[i][0:gv.a_dim],senmem[i][gv.a_dim*2:gv.a_dim*2+3])
    
    senmem = []
    return senmem, LWPRformant

# Update task prediction module (not used in the manuscript. Debug)
def UpdateTaskPrediction(Taskmodel,Taskmem,senmem):
    for i in range(len(senmem)):
        Taskmodel.update(Taskmem[i][0:gv.a_dim],Taskmem[i][gv.a_dim:gv.a_dim+gv.x_dim])
    Taskmem = []
    return Taskmem, Taskmodel

# Update auditory prediction module in Design C (not used in the
# manuscript. Debug)
def UpdateAuditoryPrediction(Audmodel,Taskmem,senmem):
    if len(senmem) > 15:
        for i in range(len(senmem)):
        #print("TASK UPDATE")
        #print(Taskmem[i])

            newFormant = Audmodel.predict(Taskmem[i][gv.a_dim:gv.a_dim+gv.x_dim])
            Audmodel.update(senmem[i][0:gv.x_dim],newFormant)
    return Audmodel

#checks and save sensory errors for learning in future trials
def sensoryerrorsave(y,z,senmem,x,i_frm):
    if abs(y[0]-z[0]) > 10 or max(abs(y[3:12]-z[3:12])) > 0.1: # will have to incorporate F2/F3 later
        #print("*****************perturb detect(*****************")
        senmem.append(np.append(x[0:gv.a_dim*2],z))
        #print(senmem)
        #print("detected frame", i_frm)
    return senmem

# Return the Cholesky decomposition
# Used to compute sigma points
def chol(Phat):
    try:
        R = np.linalg.cholesky(Phat)
        return R,0
    except:
        return 0,1

# Computes sigma points for UKF (or AUKF)
def sigmas(x,P,c):
    #Sigma points around reference point
    #Inputs:
    #       x: reference point
    #       P: covariance
    #       c: coefficient
    #Output:
    #       X: Sigma points
    #P = np.identity(20) * 0.01
    #c = 0.0017
    #x = np.array([0.09107,1.259796,-0.00982,0.0098,0.7643,-0.174,0,0,0.3125,0,0,0,0,0,0,0,0,0,0,0])
    if np.linalg.det(P) != 1:
        #from nearestSPD.m
        B = (P + np.transpose(P))/2
        u,s,V =np.linalg.svd(B)#check with matlab 3:16 PM 1/18
        Sigma = np.diag(s)
        H = np.matmul(np.matmul(V,Sigma),np.transpose(V))
        Phat = (B+H)/2
        Phat = (Phat + np.transpose(Phat))/2
        p = 1
        k = 0
        while p != 0:
            R,p = chol(Phat)
            k = k + 1
            #print("k: ", k)
            if p != 0:
                print("Chol exception occurred")
                # Ahat failed the chol test. It must have been just a hair off,
                # due to floating point trash, so it is simplest now just to
                # tweak by adding a tiny multiple of an identity matrix.
                # If this error keeps, it is very likely that the state estimate variables
                # are not realistic. 

                mineig = min(np.linalg.eigvals(Phat))
                #print("MINEIG" ,(-mineig*(k**2) + np.spacing(mineig)))
                Phat = Phat + (-mineig*(k**2) + np.spacing(mineig))*np.eye(np.size(P,0),np.size(P,1))
                #print("Phat: ", Phat)
                #print((-mineig*(k**2) + np.spacing(mineig)))
        P = Phat

    A = c*np.transpose(np.linalg.cholesky(P))
    Y = np.array([x,]*np.size(x)).transpose()
    X = np.concatenate((np.array([x]).T,Y+A,Y-A), axis =1) # to be used for Unscented Kalman Filter
    return X

# A helper function for computing covariance matrices
def transformedDevandCov(Y,y,Wc,R):
    Y1 = (Y.T - y).T
    P=np.matmul(np.matmul(Y1,np.diag(Wc)),Y1.T)+R
      
    return Y1,P

# Articulatory State Prediction that uses actual ODE45
# (see Parrell et al., 2019). We no longer use this since we
# have implemented this with LWPRs (see below).
def ArticStatePredict(X,Wm,Wc,n,R,u,ms_frm):
    #Unscented Transformation for process model
    #Input:
    #        X: sigma points
    #       Wm: weights for mean
    #       Wc: weights for covraiance
    #        n: numer of outputs of f
    #        R: additive covariance
    #        u: motor command
    #Output:
    #        y_tmean: transformed mean.
    #        Y: transformed sampling points
    #        P: transformed covariance
    #       Y1: transformed deviations
    
    L=X.shape[1]
    y_tmean=np.zeros(n)
    Y=np.zeros([n,L])
    Y1 = np.zeros([n,L])
    for k in range(L): 
        sol2 = solve_ivp(fun=lambda t, y: ode45_dim6(t, y, u), t_span=[0, ms_frm/1000], y0=X[:,k], method='RK45', dense_output=True, rtol=1e-13, atol=1e-22)     
        Y[:,k] = sol2.y[:,-1]
        y_tmean=y_tmean+Wm[k]*Y[:,k]
        
    Y1,P = transformedDevandCov(Y,y_tmean,Wc,R)
    return y_tmean,Y,P,Y1
    

# Articulatory State Prediction that uses LWPR
# Used in all designs in Kim et al. (in review)
def ArticStatePredict_LWPR(X,Wm,Wc,n,R,u,ms_frm,ASPmodel):
    #Unscented Transformation for process model
    #Input:
    #        X: sigma points
    #       Wm: weights for mean
    #       Wc: weights for covraiance
    #        n: numer of outputs of f
    #        R: additive covariance
    #        u: motor command
    #Output:
    #        y_tmean: transformed mean.
    #        Y: transformed sampling points
    #        P: transformed covariance
    #       Y1: transformed deviations
    
    L=X.shape[1]
    y_tmean=np.zeros(n)
    Y=np.zeros([n,L])
    Y1 = np.zeros([n,L])
    temp = np.zeros([gv.a_dim,2])
    for k in range(L): 
        #sol2 = solve_ivp(fun=lambda t, y: ode45_dim6(t, y, u), t_span=[0, ms_frm/1000], y0=X[:,k], method='RK45', dense_output=True, rtol=1e-13, atol=1e-22)     
        for z in range(gv.a_dim):
            temp[z,0:2] = ASPmodel[z].predict(np.array([X[z,k],X[z+gv.a_dim,k],u[z]]))
            Y[z,k] = temp[z,0]
            Y[z+gv.a_dim,k] = temp[z,1]


        y_tmean=y_tmean+Wm[k]*Y[:,k]
        #print(Wm[k])
        #print(Y[:,k])
        #print(Wm[k]*Y[:,k])
        
    Y1,P = transformedDevandCov(Y,y_tmean,Wc,R)
    return y_tmean,Y,P,Y1

# Auditory prediction moodule for the classic architecture (Parrell et al., 2019)
# Also used in Design A and B in Kim et al. (in review)
def AuditoryPrediction(LWPRformant,X1,Wm):    
    L=X1.shape[1]
    y=np.zeros(3)
    Y=np.zeros([3,L])
    for k in range(L):
        tempFormant = LWPRformant.predict_J(X1[0:gv.a_dim,k])
        Y[:,k] = tempFormant[0][0:3]    
    #    Y(:,k)=lwpr_predict_J(f,X(:,k));       
    #    y=y+Wm(k)*Y(:,k);
        y=y+Wm[k]*Y[:,k]
    
    return Y,y


# Auditory prediction moodule for the new architecture 
# (e.g., Design C in in Kim et al. in review)
def TaskAuditoryPrediction(LWPRformant,X1,Wm):    
    L=X1.shape[1]
    y=np.zeros(3)
    Y=np.zeros([3,L])
    for k in range(L):
        tempFormant = LWPRformant.predict_J(X1[0:gv.x_dim,k])
        Y[:,k] = tempFormant[0][0:3]    
    #    Y(:,k)=lwpr_predict_J(f,X(:,k));       
    #    y=y+Wm(k)*Y(:,k);
        y=y+Wm[k]*Y[:,k]
    
    return Y,y

#Y1 = (Y.T - y).T
#Y1 = Y-y
#Y1=Y-y(:,ones(1,L));
#P=np.matmul(np.matmul(Y1,np.diag(Wc)),Y1.T)+R

# Somatosensory prediction moodule
def SomatosensoryPrediction(feedbackType,LWPRsom,org_Y,org_y,X1,Wm):
    L=X1.shape[1]
    y=np.zeros(gv.a_dim*2)
    Y=np.zeros([gv.a_dim*2,L])
    for k in range(L):
        for j in range(gv.a_dim*2):
            Y[j,k] = LWPRsom[j].predict(np.array([X1[j,k]]))
        #Y[:,k] = tempsom[0][0:gv.a_dim*2] 
    #    Y(:,k)=lwpr_predict_J(f,X(:,k));       
    #    y=y+Wm(k)*Y(:,k);
        y=y+Wm[k]*Y[:,k]
    

    if feedbackType == 'full' and org_y.shape[0]>1: #if feedback is full and if the y variable has more than 1 value (in this case it's likley the auditory (3 formants))
        Y_full = np.vstack([org_Y,Y])
        y_full = np.append(org_y,y)
        return Y_full,y_full
    else:#if it's not full or the y variable is 1 then this is likely the only sensory variable and we don't need to append anything.
        return Y,y
    #if feedbackType == 'full':
    #    y = np.append(y,x1)
    #    Y = np.vstack([Y,X1])    
    #else:    
    #    y = x1
    #    Y = X1
        
    #return Y,y

# State correction in which Kalman gain is multiplied to the prediction error
def StateCorrection(X2,Wc,Y1,P,z,y):
    #Kalman gain calculation
    P12=np.matmul(np.matmul(X2,np.diag(Wc)),Y1.T)  #transformed cross-covariance
    K=np.matmul(P12,np.linalg.inv(P))
    #print("predict",z[0:3])
    #print("sensory",y[0:3])
    #print("K: ", K)
    return np.matmul(K,z-y),np.matmul(K,P12.T)  #Kalman*prediction error (Eq 5 and/or 6) and covariance update