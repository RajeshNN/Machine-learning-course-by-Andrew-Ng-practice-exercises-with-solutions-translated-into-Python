from __future__ import division
import numpy as np
import sigmoid
import costFunction

def costFunctionReg(theta, X, y, lambdaa):
    m=X.shape[0]
    J=0
    t=0
    J=costFunction.costFunction(theta, X, y)
    for i in range(1,theta.shape[0]):
        t+=theta[i]**2
    l=lambdaa/(2*m)
    J+=l*t
    return J;
