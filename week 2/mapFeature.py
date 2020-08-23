from __future__ import division
import numpy as np

def mapFeature(X1,X2):
    degree=6
    X1=np.matrix(X1)
    X2=np.matrix(X2)
    out=np.ones((X1.shape[0],1))
    for i in range(1,degree+1):
        for j in range(i+1):
            a=np.power(X1,(i-j))
            b=np.power(X2,j)
            c=np.multiply(a,b)
            out=np.concatenate((out,c),axis=1)
    return out;
