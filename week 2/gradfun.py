from __future__ import division
import numpy as np
import math
import sigmoid

def gradfun(theta,X,y):
    m=len(y)
    grad=np.zeros(theta.shape)
    A=X@theta
    h=sigmoid.sigmoid(A)
    c=h-y
    grad=(X.T)@c
    grad=grad/m
    return grad;
