import numpy as np 
def computeCostMulti(X, y, theta):
    m = len(y) # number of training examples
    J = 0
    h=X@theta
    divergence=h-y
    J=(np.sum(divergence**2))/(2*m)
    return J;
