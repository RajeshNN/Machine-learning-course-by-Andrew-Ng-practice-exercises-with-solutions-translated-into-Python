import numpy as np
import trainLinearReg
import linearRegCostFunction

def func (X, y, Xval, yval):

# Selected values of lambda (you should not change this)
    lambda_vec = np.matrix([0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10]).T
    lambda_vec = np.array(lambda_vec)

# You need to return these variables correctly.
    error_train = np.zeros((lambda_vec.shape[0], 1))
    error_val = np.zeros((lambda_vec.shape[0], 1))


    for i in range(lambda_vec.shape[0]):
        theta=trainLinearReg.func(X,y,lambda_vec[i])
        error_train[i]=linearRegCostFunction.func(theta,X,y,0)
        error_val[i]=linearRegCostFunction.func(theta,Xval,yval,0)


    return lambda_vec, error_train, error_val;
