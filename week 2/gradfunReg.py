from __future__ import division
import numpy as np
import gradfun

def gradfunReg(theta, X, y, lambdaa):
    m=X.shape[0]
    grad=np.zeros(theta.shape)
    grad=gradfun.gradfun(theta, X, y)
    for i in range(1,grad.shape[0]):
        grad[i,0]=grad[i,0]+((lambdaa*theta[i])/m)
    return grad;
