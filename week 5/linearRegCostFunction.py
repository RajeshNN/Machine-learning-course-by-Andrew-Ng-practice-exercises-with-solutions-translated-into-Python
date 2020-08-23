import numpy as np

def func(theta, X, y, lambdaa):

# Initialize some useful values
    m = y.shape[0] # number of training examples
    J = 0
    t=np.zeros((len(theta),1))
    for i in range(len(theta)):
        t[i,0]=theta[i]
    z=X@t

    J=((np.sum((z-y)**2))/(2*m))+((lambdaa*(np.sum(t[1:]**2)))/(2*m))
    
    J=float(J)
    return J;
