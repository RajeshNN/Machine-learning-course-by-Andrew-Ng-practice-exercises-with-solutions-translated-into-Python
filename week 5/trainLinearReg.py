import numpy as np
import linearRegCostFunction
import linearRegGradFunction
import scipy.optimize as sc

def func(X, y, lambdaa):
#TRAINLINEARREG Trains linear regression given a dataset (X, y) and a
#regularization parameter lambdaa
#   [theta] = TRAINLINEARREG (X, y, lambdaa) trains linear regression using
#   the dataset (X, y) and regularization parameter lambdaa. Returns the
#   trained parameters theta.
#

# Initialize Theta
    initial_theta = np.zeros((X.shape[1], 1))

    res = sc.minimize(linearRegCostFunction.func, initial_theta,method='TNC',\
                        args=(X,y,lambdaa),jac=linearRegGradFunction.func,\
                        options={'maxiter':100})
    theta=res.x
    
    return theta;
