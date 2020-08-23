import numpy as np
import trainLinearReg
import linearRegCostFunction

def func(X, y, Xval, yval, lambdaa):

# Number of training examples
    m = X.shape[0]
             
# You need to return these values correctly
    error_train = np.zeros((m, 1))
    error_val   = np.zeros((m, 1))


    for i in range(m):
        theta = trainLinearReg.func(X[range(i+1),:],y[range(i+1)], lambdaa)
        error_train[i]=linearRegCostFunction.func(theta,X[range(i+1),:],y[range(i+1)],0)
        error_val[i]=linearRegCostFunction.func(theta,Xval,yval,0)

    return error_train, error_val;
