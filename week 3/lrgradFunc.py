import numpy as np
import sigmoid
import scipy.special as ex

def lrgradFunc(theta, X, y, lambdaa):
    
    m = y.shape[0] # number of training examples

    t=np.zeros((len(theta),1))
    for k in range(len(theta)):
        t[k,0]=theta[k]
        
    g = [0]*len(theta)
    grad=np.zeros((len(theta),1))

    A=X@t
    h=ex.expit(A)
    a=h-y
    
    grad=(X.T@a)/m
    
    for j in range(1,t.shape[0]):
        grad[j,0]+=(np.multiply(lambdaa,t[j,0]))/m
    
    for i in range(len(g)):
        g[i]=grad[i,0]
    
    g=np.array(g)
        
    return g;
