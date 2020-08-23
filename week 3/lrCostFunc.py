from __future__ import division
import numpy as np
import scipy.special as ex

def lrCostFunction(theta, X, y, lambdaa):
    
    m = y.shape[0] # number of training examples
    J = 0
    
    t=np.zeros((len(theta),1))
    for i in range(len(theta)):
        t[i,0]=theta[i]

    A=X@t
    h=ex.expit(A)
    
    J=-(((y.T@(np.log(h)))+((1-y).T@(np.log(1-h))))/m)
    J=float(J)
    for i in range(1,t.shape[0]):
        J+=(lambdaa*(t[i,0]**2))/(2*m)
    
    return J;
