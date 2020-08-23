import numpy as np

def func(theta, X, y, lambdaa):

# Initialize some useful values
    m = y.shape[0] # number of training examples
    
    t=np.zeros((len(theta),1))
    for i in range(len(theta)):
        t[i,0]=theta[i]
    
    z=X@t
    
    grad = np.zeros(t.shape)

    grad=(1/m)*(X.T@(z-y))
    grad[1:]+=(lambdaa*t[1:])/m
    
    return grad;
