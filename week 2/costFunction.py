from __future__ import division
import numpy as np
import math
import sigmoid

def costFunction(theta,X,y):
    m = len(y)
    J = 0
    theta=np.matrix(theta)
    if theta.shape[0]<2:
        theta=theta.T
    ##print(theta.shape,X.shape)
    A=X@theta
    h=sigmoid.sigmoid(A)
    x=np.ones(h.shape)
    h1=np.subtract(x,h)
    lh=np.log(h)
    h2=np.log(h1)
    y1=np.subtract(x,y)
    a=(y.T)@lh
    b=(y1.T)@h2
    J=(-a-b)/m
    return J;
